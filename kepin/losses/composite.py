# -*- coding: utf-8 -*-
"""
KePIN Composite Loss — Physics-informed multi-task loss with Kendall
uncertainty-based auto-balancing.

Components (degradation mode — all 7):
  1. RUL prediction (Huber, δ=20)
  2. Koopman one-step reconstruction
  3. Spectral stability (eigenvalue constraint)
  4. Monotonicity prior (RUL must not increase over time)
  5. Multi-step dynamics fidelity
  6. NASA-inspired asymmetric scoring
  7. Slope matching (degradation rate tracking)

Forecasting mode keeps only {1, 2, 3, 5}, disabling degradation-specific
priors for weather / finance / physics domains.
"""

import tensorflow as tf
import keras
from keras.saving import register_keras_serializable


# -------------------------------------------------------------------------
# Individual loss components
# -------------------------------------------------------------------------

def rul_mse_loss(y_true, y_pred):
    """Huber loss for RUL prediction (δ=20)."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    delta = 20.0
    error = tf.abs(y_true - y_pred)
    quadratic = tf.minimum(error, delta)
    linear = error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)


def koopman_one_step_loss(one_step_pred, one_step_target):
    """Latent dynamics consistency: ||K·z(t) - z(t+1)||²."""
    return tf.reduce_mean(tf.square(
        tf.cast(one_step_pred, tf.float32) - tf.cast(one_step_target, tf.float32)))


def spectral_stability_loss(eigenvalues):
    """Penalise eigenvalues with |λ| > 1 (growing modes)."""
    violation = tf.maximum(tf.abs(eigenvalues) - 1.0, 0.0)
    return tf.reduce_mean(tf.square(violation))


def monotonicity_loss(y_true, y_pred):
    """Physics prior: penalise predicted RUL increases within same engine."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    same_engine_mask = tf.cast(true_diffs < 0.0, tf.float32)
    denom = tf.reduce_sum(same_engine_mask) + 1e-7
    mono_violation = tf.nn.relu(pred_diffs) * same_engine_mask
    return tf.reduce_sum(mono_violation) / denom


def multi_step_loss(multi_step_pred, multi_step_target):
    """Long-horizon dynamics fidelity: ||K^k·z(t) - z(t+k)||²."""
    return tf.reduce_mean(tf.square(
        tf.cast(multi_step_pred, tf.float32) - tf.cast(multi_step_target, tf.float32)))


def asymmetric_loss(y_true, y_pred):
    """NASA-inspired asymmetric scoring (late predictions penalised 2.5×)."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    diff = y_pred - y_true
    weight = tf.where(diff > 0, 2.5, 1.0)
    return tf.reduce_mean(weight * tf.square(diff))


def slope_matching_loss(y_true, y_pred):
    """Degradation rate tracking: predicted slope ≈ true slope."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    true_diffs = y_true[1:] - y_true[:-1]
    pred_diffs = y_pred[1:] - y_pred[:-1]
    same_engine_mask = tf.cast(true_diffs < 0.0, tf.float32)
    denom = tf.reduce_sum(same_engine_mask) + 1e-7
    slope_err = tf.square((pred_diffs - true_diffs) * same_engine_mask)
    return tf.reduce_sum(slope_err) / denom


# -------------------------------------------------------------------------
# Kendall uncertainty-based auto-balancing layer
# -------------------------------------------------------------------------

class KePINLossWeights(keras.layers.Layer):
    """Learnable log-variance parameters for uncertainty-based loss weighting.

    weighted_L_i = 0.5 * exp(-s_i) * L_i + 0.5 * s_i

    where s_i = log(σ_i²) is the learnable parameter (clamped to [-6, 6]).

    Reference: Kendall, Gal & Cipolla, CVPR 2018.
    """

    def __init__(self, n_losses: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_losses = n_losses

    def build(self, input_shape):
        self.log_vars = self.add_weight(
            name="log_vars", shape=(self.n_losses,),
            initializer=keras.initializers.Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, loss_values):
        total = tf.constant(0.0)
        clamped = tf.clip_by_value(self.log_vars, -6.0, 6.0)
        for i, loss_val in enumerate(loss_values):
            precision = tf.exp(-clamped[i])
            total = total + 0.5 * precision * loss_val + 0.5 * clamped[i]
        weights = tf.exp(-clamped)
        return total, weights

    def get_config(self):
        config = super().get_config()
        config.update({"n_losses": self.n_losses})
        return config


# -------------------------------------------------------------------------
# Factory: composite loss function
# -------------------------------------------------------------------------

def make_kepin_loss(loss_weights_layer=None, use_auto_weights=True,
                    fixed_weights=None, domain_mode="degradation"):
    """Create the composite KePIN loss function.

    Args:
        loss_weights_layer: ``KePINLossWeights`` instance (for auto mode)
        use_auto_weights:   use Kendall uncertainty weighting
        fixed_weights:      dict of fixed weights (for ablation)
        domain_mode:        ``"degradation"`` (7 losses) or ``"forecasting"`` (4 losses)

    Returns:
        Callable (y_true, y_pred, koopman_outputs) → (total_loss, loss_dict)
    """
    if fixed_weights is None:
        fixed_weights = {"rul": 1.0, "koopman": 0.1, "spectral": 0.01,
                         "mono": 0.001, "multi_step": 0.05, "asym": 0.05,
                         "slope": 0.0003}

    if domain_mode == "forecasting":
        active_losses = {"rul", "koopman", "spectral", "multi_step"}
    else:
        active_losses = {"rul", "koopman", "spectral", "mono",
                         "multi_step", "asym", "slope"}

    loss_names = ["rul", "koopman", "spectral", "mono",
                  "multi_step", "asym", "slope"]
    active_names = [n for n in loss_names if n in active_losses]

    def loss_fn(y_true, y_pred, koopman_outputs):
        l_rul = rul_mse_loss(y_true, y_pred)
        l_koop = koopman_one_step_loss(koopman_outputs["one_step_pred"],
                                       koopman_outputs["one_step_target"])
        l_spec = spectral_stability_loss(koopman_outputs["eigenvalues"])
        l_mono = monotonicity_loss(y_true, y_pred) if "mono" in active_losses else tf.constant(0.0)
        l_multi = multi_step_loss(koopman_outputs["multi_step_pred"],
                                  koopman_outputs["multi_step_target"])
        l_asym = asymmetric_loss(y_true, y_pred) if "asym" in active_losses else tf.constant(0.0)
        l_slope = slope_matching_loss(y_true, y_pred) if "slope" in active_losses else tf.constant(0.0)

        all_losses = [l_rul, l_koop, l_spec, l_mono, l_multi, l_asym, l_slope]
        active_loss_values = [all_losses[loss_names.index(n)] for n in active_names]

        if use_auto_weights and loss_weights_layer is not None:
            total, eff_weights = loss_weights_layer(active_loss_values)
            weight_dict = {n: float(eff_weights[i]) for i, n in enumerate(active_names)}
        else:
            total = tf.constant(0.0)
            weight_dict = {}
            for name in active_names:
                idx = loss_names.index(name)
                w = fixed_weights.get(name, 0.0)
                total = total + w * all_losses[idx]
                weight_dict[name] = w

        loss_dict = {
            "total": total, "rul_mse": l_rul, "koopman_1step": l_koop,
            "spectral": l_spec, "monotonicity": l_mono,
            "multi_step": l_multi, "asymmetric": l_asym,
            "slope": l_slope, "weights": weight_dict,
        }
        return total, loss_dict

    return loss_fn


@register_keras_serializable(package="KePIN")
def kepin_rul_loss(y_true, y_pred):
    """Basic RUL loss for ``model.compile()`` fallback."""
    return rul_mse_loss(y_true, y_pred)
