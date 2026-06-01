# -*- coding: utf-8 -*-
"""
Koopman Operator Module — core novelty layer for KePIN.

Implements a learnable linear Koopman operator K in a latent space:

    z(t+1) ≈ K · z(t)

K is parameterised via SVD factorisation:

    K = U · diag(σ(s)) · V^T

where σ is the sigmoid function bounding singular values to [0, 1],
ensuring spectral stability (all eigenvalues |λ_i| ≤ 1).
"""

import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable


@register_keras_serializable(package="KePIN")
class KoopmanOperator(keras.layers.Layer):
    """Learnable Koopman operator with SVD-parameterised stability.

    Given latent states Z = [z(1), ..., z(T)] of shape (batch, T, d),
    computes one-step predictions, multi-step rollouts, and eigenvalue
    decomposition for spectral analysis and physics constraints.

    Parameters
    ----------
    latent_dim : int
        Dimensionality d of the latent state vectors.
    rollout_steps : int
        Number of multi-step rollout predictions (default: 3).
    stability_mode : str
        'svd' — parameterise K = U · diag(σ(s)) · V^T (default)
        'full' — unconstrained K (for ablation baseline)
    """

    def __init__(self, latent_dim: int, rollout_steps: int = 3,
                 stability_mode: str = "svd", **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.rollout_steps = rollout_steps
        self.stability_mode = stability_mode

    def build(self, input_shape):
        d = self.latent_dim
        if self.stability_mode == "svd":
            self.U = self.add_weight("U", (d, d),
                                     initializer=keras.initializers.Orthogonal(),
                                     trainable=True)
            self.s = self.add_weight("s", (d,),
                                     initializer=keras.initializers.Zeros(),
                                     trainable=True)
            self.V = self.add_weight("V", (d, d),
                                     initializer=keras.initializers.Orthogonal(),
                                     trainable=True)
        else:
            self.K_raw = self.add_weight("K_raw", (d, d),
                                         initializer=keras.initializers.GlorotUniform(),
                                         trainable=True)
        super().build(input_shape)

    def _get_K(self):
        """Construct the Koopman operator matrix K."""
        if self.stability_mode == "svd":
            sigma = tf.nn.sigmoid(self.s)
            return tf.matmul(self.U, tf.matmul(tf.linalg.diag(sigma),
                                                tf.transpose(self.V)))
        return self.K_raw

    def call(self, z_sequence, training=None):
        """Forward pass.

        Args:
            z_sequence: (batch, T, d) — sequence of latent states

        Returns:
            dict with keys: one_step_pred, one_step_target, multi_step_pred,
            multi_step_target, K, eigenvalues, final_state
        """
        K = tf.cast(self._get_K(), tf.float32)
        z_sequence = tf.cast(z_sequence, tf.float32)
        T = tf.shape(z_sequence)[1]

        # One-step prediction: z_hat(t+1) = K · z(t)
        z_input = z_sequence[:, :-1, :]
        z_target = z_sequence[:, 1:, :]
        z_pred_one = tf.matmul(z_input, tf.transpose(K))

        # Multi-step rollout
        H = min(self.rollout_steps, 5)
        K_powers = [K]
        for k in range(1, H):
            K_powers.append(tf.matmul(K_powers[-1], K))

        max_start = T - H - 1
        multi_preds, multi_targets = [], []
        for k_idx in range(H):
            k = k_idx + 1
            z_start = z_sequence[:, :max_start + 1, :]
            z_pred_k = tf.matmul(z_start, tf.transpose(K_powers[k_idx]))
            z_true_k = z_sequence[:, k:max_start + 1 + k, :]
            multi_preds.append(z_pred_k)
            multi_targets.append(z_true_k)

        multi_step_pred = tf.stack(multi_preds, axis=2)
        multi_step_target = tf.stack(multi_targets, axis=2)

        # Eigenvalues for spectral analysis
        eigenvalues = tf.linalg.eigvals(tf.cast(K, tf.float32))

        return {
            "one_step_pred": z_pred_one,
            "one_step_target": z_target,
            "multi_step_pred": multi_step_pred,
            "multi_step_target": multi_step_target,
            "K": K,
            "eigenvalues": eigenvalues,
            "final_state": z_sequence[:, -1, :],
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "rollout_steps": self.rollout_steps,
            "stability_mode": self.stability_mode,
        })
        return config


def extract_spectral_features(eigenvalues, top_k: int = 4):
    """Extract interpretable spectral features from Koopman eigenvalues.

    Returns a float tensor of shape (top_k * 2 + 2,) with per-mode
    decay rates, frequencies, spectral radius, and spectral gap.
    """
    eig_mags = tf.abs(eigenvalues)
    sorted_indices = tf.argsort(eig_mags, direction="DESCENDING")
    top_eigs = tf.gather(eigenvalues, sorted_indices[:top_k])

    log_eigs = tf.math.log(tf.cast(top_eigs, tf.complex64) + 1e-10)
    decay_rates = -tf.math.real(log_eigs)
    frequencies = tf.abs(tf.math.imag(log_eigs))

    spectral_radius = tf.reduce_max(eig_mags)
    all_mags_sorted = tf.sort(eig_mags, direction="DESCENDING")
    spectral_gap = all_mags_sorted[0] - all_mags_sorted[1]

    return tf.concat([
        tf.cast(decay_rates, tf.float32),
        tf.cast(frequencies, tf.float32),
        tf.expand_dims(tf.cast(spectral_radius, tf.float32), 0),
        tf.expand_dims(tf.cast(spectral_gap, tf.float32), 0),
    ], axis=0)


def spectral_features_dim(top_k: int = 4) -> int:
    """Return the output dimension of ``extract_spectral_features()``."""
    return top_k * 2 + 2
