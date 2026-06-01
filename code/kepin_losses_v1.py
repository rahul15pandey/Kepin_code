# -*- coding: utf-8 -*-
"""
KePIN Loss Functions — Physics-informed Koopman loss with auto-balanced weights.

The composite loss simultaneously optimises:
  1. RUL prediction accuracy (MSE)
  2. Latent dynamics consistency (Koopman one-step reconstruction)
  3. Spectral stability (eigenvalue constraint)
  4. Monotonicity of predicted RUL (physics prior)
  5. Multi-step dynamics fidelity (long-horizon Koopman rollout)

Loss weights are auto-balanced using Kendall et al. (2018) uncertainty
weighting: each weight w_i = exp(-s_i) where s_i is a learnable parameter.
This eliminates per-dataset manual weight tuning — the key domain-specificity
problem in the original PI-DP-FCN.

Reference:
  Kendall, Gal, Cipolla — "Multi-Task Learning Using Uncertainty to Weigh
  Losses for Scene Geometry and Semantics", CVPR 2018.
"""

import tensorflow as tf
import keras
from keras.saving import register_keras_serializable

# =========================================================================
# Mixed-precision safety: all loss computations use float32
# When mixed_float16 policy is active, model outputs may be float16.
# We cast to float32 at the start of each loss to prevent underflow.
# =========================================================================


# =========================================================================
# Individual loss components
# =========================================================================

def rul_mse_loss(y_true, y_pred):
    """Standard MSE for RUL prediction accuracy."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.square(tf.reshape(y_true, [-1]) - tf.reshape(y_pred, [-1])))


def koopman_one_step_loss(one_step_pred, one_step_target):
    """Latent dynamics consistency: ||K·z(t) - z(t+1)||².

    Ensures the learned Koopman operator accurately models the
    state transition dynamics in the latent space.

    Args:
        one_step_pred:   (batch, T-1, d) — K·z(t)
        one_step_target: (batch, T-1, d) — z(t+1)
    """
    one_step_pred = tf.cast(one_step_pred, tf.float32)
    one_step_target = tf.cast(one_step_target, tf.float32)
    return tf.reduce_mean(tf.square(one_step_pred - one_step_target))


def spectral_stability_loss(eigenvalues):
    """Spectral constraint: penalise eigenvalues with |λ| > 1.

    For degradation systems, all dynamics should be stable or decaying.
    Eigenvalues outside the unit circle indicate exponentially growing
    modes, which are physically unreasonable for degradation.

    Args:
        eigenvalues: (d,) complex tensor
    """
    eig_magnitudes = tf.abs(eigenvalues)  # |λ_i|
    # Penalty only when |λ_i| > 1 (growing modes)
    violation = tf.maximum(eig_magnitudes - 1.0, 0.0)
    return tf.reduce_mean(tf.square(violation))


def monotonicity_loss(y_true, y_pred):
    """Physics prior: RUL must not increase over time within the same unit.

    Uses the masked differencing trick from the original PI-DP-FCN:
    only penalise positive predicted RUL differences where the true
    RUL is decreasing (same-engine transitions, not cross-engine
    boundaries in the batch).

    Args:
        y_true: (batch, 1) — true RUL values
        y_pred: (batch, 1) — predicted RUL values
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    true_diffs = y_true_flat[1:] - y_true_flat[:-1]
    pred_diffs = y_pred_flat[1:] - y_pred_flat[:-1]

    # Mask: only within-engine transitions (where true RUL decreases)
    same_engine_mask = tf.cast(true_diffs < 0.0, tf.float32)
    denom = tf.reduce_sum(same_engine_mask) + 1e-7

    # Penalise positive predicted diffs (RUL increasing)
    mono_violation = tf.nn.relu(pred_diffs) * same_engine_mask
    return tf.reduce_sum(mono_violation) / denom


def multi_step_loss(multi_step_pred, multi_step_target):
    """Long-horizon dynamics fidelity: ||K^k·z(t) - z(t+k)||².

    Tests whether the learned linear dynamics remain accurate over
    multiple time steps, not just one-step ahead.

    Args:
        multi_step_pred:   (batch, T-H, H, d) — K^k·z(t)
        multi_step_target: (batch, T-H, H, d) — z(t+k)
    """
    multi_step_pred = tf.cast(multi_step_pred, tf.float32)
    multi_step_target = tf.cast(multi_step_target, tf.float32)
    return tf.reduce_mean(tf.square(multi_step_pred - multi_step_target))


def asymmetric_loss(y_true, y_pred):
    """Penalise over-estimation (predicting RUL too high = late warning).

    Over-estimation is dangerous because it gives a false sense of
    remaining life. Under-estimation (early warning) is conservative.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    over = tf.nn.relu(y_pred_flat - y_true_flat)
    return tf.reduce_mean(tf.square(over))


def slope_matching_loss(y_true, y_pred):
    """Degradation rate tracking: predicted slope should match true slope.

    Ensures smooth, physically consistent degradation trajectories.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    true_diffs = y_true_flat[1:] - y_true_flat[:-1]
    pred_diffs = y_pred_flat[1:] - y_pred_flat[:-1]

    same_engine_mask = tf.cast(true_diffs < 0.0, tf.float32)
    denom = tf.reduce_sum(same_engine_mask) + 1e-7

    slope_err = tf.square((pred_diffs - true_diffs) * same_engine_mask)
    return tf.reduce_sum(slope_err) / denom


# =========================================================================
# Auto-Balanced Composite Loss (Kendall uncertainty weighting)
# =========================================================================

class KePINLossWeights(keras.layers.Layer):
    """Learnable log-variance parameters for uncertainty-based loss weighting.

    Each loss term L_i is weighted by:
        weighted_L_i = (1 / (2 * σ_i²)) * L_i + log(σ_i)
                     = 0.5 * exp(-s_i) * L_i + 0.5 * s_i

    where s_i = log(σ_i²) is the learnable parameter.

    This avoids manual per-dataset weight tuning — the network learns
    the appropriate balance automatically.
    """

    def __init__(self, n_losses: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_losses = n_losses

    def build(self, input_shape):
        # Initialise log-variances to 0 (σ² = 1, equal initial weighting)
        self.log_vars = self.add_weight(
            name="log_vars",
            shape=(self.n_losses,),
            initializer=keras.initializers.Zeros(),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, loss_values):
        """Apply uncertainty weighting to a list of loss scalars.

        Args:
            loss_values: list of scalar tensors, one per loss component

        Returns:
            total_loss: scalar — sum of weighted losses
            weights:    (n_losses,) tensor — effective weights exp(-s_i)
        """
        total = tf.constant(0.0)
        # Clamp log_vars to [-6, 6] to prevent extreme weighting
        clamped_log_vars = tf.clip_by_value(self.log_vars, -6.0, 6.0)
        for i, loss_val in enumerate(loss_values):
            precision = tf.exp(-clamped_log_vars[i])      # 1/(2σ²)
            total = total + 0.5 * precision * loss_val + 0.5 * clamped_log_vars[i]
        weights = tf.exp(-clamped_log_vars)  # effective weights for monitoring
        return total, weights

    def get_config(self):
        config = super().get_config()
        config.update({"n_losses": self.n_losses})
        return config


# =========================================================================
# Factory: complete KePIN loss function
# =========================================================================

def make_kepin_loss(loss_weights_layer: KePINLossWeights = None,
                    use_auto_weights: bool = True,
                    fixed_weights: dict = None):
    """Create the composite KePIN loss function.

    This returns a callable that takes (y_true, y_pred, koopman_outputs)
    and computes the total loss with either auto-balanced or fixed weights.

    Args:
        loss_weights_layer: KePINLossWeights instance (for auto mode)
        use_auto_weights:   if True, use uncertainty weighting
        fixed_weights:      dict of fixed weights for ablation, e.g.:
            {"rul": 1.0, "koopman": 0.1, "spectral": 0.01,
             "mono": 0.001, "multi_step": 0.05, "asym": 0.05, "slope": 0.0003}

    Returns:
        loss_fn(y_true, y_pred, koopman_outputs) → (total_loss, loss_dict)
    """
    if fixed_weights is None:
        fixed_weights = {
            "rul": 1.0, "koopman": 0.1, "spectral": 0.01,
            "mono": 0.001, "multi_step": 0.05, "asym": 0.05, "slope": 0.0003,
        }

    loss_names = ["rul", "koopman", "spectral", "mono", "multi_step", "asym", "slope"]

    def loss_fn(y_true, y_pred, koopman_outputs):
        """Compute all loss components and return weighted total.

        Args:
            y_true: (batch, 1) ground truth RUL
            y_pred: (batch, 1) predicted RUL
            koopman_outputs: dict from KoopmanOperator.call()

        Returns:
            total_loss:  scalar
            loss_dict:   dict of individual loss values and weights
        """
        # Individual components
        l_rul = rul_mse_loss(y_true, y_pred)
        l_koop = koopman_one_step_loss(
            koopman_outputs["one_step_pred"],
            koopman_outputs["one_step_target"],
        )
        l_spec = spectral_stability_loss(koopman_outputs["eigenvalues"])
        l_mono = monotonicity_loss(y_true, y_pred)
        l_multi = multi_step_loss(
            koopman_outputs["multi_step_pred"],
            koopman_outputs["multi_step_target"],
        )
        l_asym = asymmetric_loss(y_true, y_pred)
        l_slope = slope_matching_loss(y_true, y_pred)

        losses = [l_rul, l_koop, l_spec, l_mono, l_multi, l_asym, l_slope]

        if use_auto_weights and loss_weights_layer is not None:
            total, eff_weights = loss_weights_layer(losses)
            weight_dict = {n: float(eff_weights[i]) for i, n in enumerate(loss_names)}
        else:
            total = tf.constant(0.0)
            weight_dict = {}
            for i, (name, loss_val) in enumerate(zip(loss_names, losses)):
                w = fixed_weights.get(name, 0.0)
                total = total + w * loss_val
                weight_dict[name] = w

        loss_dict = {
            "total": total,
            "rul_mse": l_rul,
            "koopman_1step": l_koop,
            "spectral": l_spec,
            "monotonicity": l_mono,
            "multi_step": l_multi,
            "asymmetric": l_asym,
            "slope": l_slope,
            "weights": weight_dict,
        }

        return total, loss_dict

    return loss_fn


# =========================================================================
# Keras-compatible wrapper for model.compile()
# =========================================================================

@register_keras_serializable(package="KePIN")
def kepin_rul_loss(y_true, y_pred):
    """Basic RUL loss for model.compile() (Koopman terms added via train_step).

    The full composite loss requires koopman_outputs which aren't available
    through the standard Keras loss API. This is the fallback for the RUL
    prediction output only. The full loss is computed in a custom training
    loop (see kepin_training.py).
    """
    return rul_mse_loss(y_true, y_pred)


# =========================================================================
# Unit test
# =========================================================================

if __name__ == "__main__":
    import numpy as np

    print("=== KePIN Losses — Unit Test ===\n")

    batch = 32
    T = 20
    d = 16

    # Fake data
    y_true = tf.constant(np.linspace(100, 0, batch).reshape(-1, 1), dtype=tf.float32)
    y_pred = y_true + tf.random.normal((batch, 1), stddev=5.0)

    koopman_out = {
        "one_step_pred": tf.random.normal((batch, T - 1, d)),
        "one_step_target": tf.random.normal((batch, T - 1, d)),
        "multi_step_pred": tf.random.normal((batch, T - 4, 3, d)),
        "multi_step_target": tf.random.normal((batch, T - 4, 3, d)),
        "eigenvalues": tf.complex(
            tf.random.uniform((d,), 0.5, 1.1),
            tf.random.uniform((d,), -0.3, 0.3),
        ),
    }

    # Test individual losses
    print(f"RUL MSE:       {rul_mse_loss(y_true, y_pred).numpy():.4f}")
    print(f"Koopman 1-step:{koopman_one_step_loss(koopman_out['one_step_pred'], koopman_out['one_step_target']).numpy():.4f}")
    print(f"Spectral:      {spectral_stability_loss(koopman_out['eigenvalues']).numpy():.6f}")
    print(f"Monotonicity:  {monotonicity_loss(y_true, y_pred).numpy():.4f}")
    print(f"Multi-step:    {multi_step_loss(koopman_out['multi_step_pred'], koopman_out['multi_step_target']).numpy():.4f}")
    print(f"Asymmetric:    {asymmetric_loss(y_true, y_pred).numpy():.4f}")
    print(f"Slope:         {slope_matching_loss(y_true, y_pred).numpy():.4f}")

    # Test auto-weighted composite loss
    lw_layer = KePINLossWeights(n_losses=7)
    lw_layer.build(None)

    loss_fn = make_kepin_loss(loss_weights_layer=lw_layer, use_auto_weights=True)
    total, loss_dict = loss_fn(y_true, y_pred, koopman_out)
    print(f"\nAuto-weighted total:  {total.numpy():.4f}")
    print(f"Effective weights:    {loss_dict['weights']}")

    # Test fixed-weight composite loss
    loss_fn_fixed = make_kepin_loss(use_auto_weights=False)
    total_fixed, loss_dict_fixed = loss_fn_fixed(y_true, y_pred, koopman_out)
    print(f"\nFixed-weight total:   {total_fixed.numpy():.4f}")

    print("\n✓ All losses compute without errors.")
