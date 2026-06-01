# -*- coding: utf-8 -*-
"""
KePIN Model — Koopman-Enhanced Physics-Informed Network.

Architecture:

  [Temporal Encoder]  →  [Koopman Module]  →  [Spectral Features]  →  [RUL Head]
   Conv1D+SE blocks      K·z(t) = z(t+1)      eigenvalue features       Dense→RUL
   (auto-configured)     (dynamics discovery)   (decay rates, freq)

Key design decisions:
  - Conv1D (not Conv2D): natural for temporal data, removes C-MAPSS-specific
    spatial dimension artifact.
  - SE attention on Conv1D: channel-wise attention via squeeze-excite.
  - Koopman operates on the full temporal feature map, not just pooled vectors.
  - Spectral features from eigenvalues concatenated with final latent state
    for physics-informed RUL prediction.
  - Auto-configuration: architecture depth/width adapt to input dimensions.

Input: (batch, seq_len, n_features)  — standard 3D time-series
Output: (batch, 1) — predicted RUL

Also exports Koopman outputs for loss computation and analysis.
"""

import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable

from koopman_module import KoopmanOperator, extract_spectral_features, spectral_features_dim
from kepin_losses import KePINLossWeights


# =========================================================================
# SE Block for Conv1D
# =========================================================================

def se_block_1d(x, ratio=8):
    """Squeeze-and-Excitation for Conv1D feature maps.

    Input:  (batch, timesteps, channels)
    Output: (batch, timesteps, channels) — channel-reweighted
    """
    filters = int(x.shape[-1])
    se = keras.layers.GlobalAveragePooling1D()(x)                     # (batch, C)
    se = keras.layers.Dense(max(filters // ratio, 4), activation='relu',
                            kernel_initializer='he_normal')(se)       # (batch, C//r)
    se = keras.layers.Dense(filters, activation='sigmoid',
                            kernel_initializer='he_normal')(se)       # (batch, C)
    se = keras.layers.Reshape((1, filters))(se)                       # (batch, 1, C)
    return keras.layers.Multiply()([x, se])                           # broadcast


# =========================================================================
# Encoder block
# =========================================================================

def conv_block_1d(x, filters, kernel_size, use_se=True, se_ratio=8):
    """Conv1D → BatchNorm → ReLU → optional SE attention."""
    x = keras.layers.Conv1D(filters, kernel_size, padding="same",
                            kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if use_se:
        x = se_block_1d(x, ratio=se_ratio)
    return x


# =========================================================================
# Auto-configuration
# =========================================================================

def auto_configure(n_features: int, seq_len: int, n_train: int = None):
    """Determine architecture hyperparameters from data shape.

    Returns a config dict with:
        n_blocks:     number of conv blocks (2-4)
        filters:      list of filter counts per block
        kernels:      list of kernel sizes per block
        latent_dim:   Koopman latent space dimension
        dropout:      dropout rate for RUL head
        rollout:      multi-step rollout horizon
        spectral_k:   number of top eigenvalues to extract features from
    """
    # Determine complexity tier based on features
    if n_features <= 6:
        tier = "small"
    elif n_features <= 16:
        tier = "medium"
    else:
        tier = "large"

    configs = {
        "small": {
            "n_blocks": 2,
            "filters": [32, 64],           # multiples of 8 for Tensor Core alignment
            "kernels": [7, 5],
            "latent_dim": 32,
            "dropout": 0.3,
            "rollout": 2,
            "spectral_k": 3,
        },
        "medium": {
            "n_blocks": 3,
            "filters": [32, 64, 128],      # multiples of 8 for Tensor Core alignment
            "kernels": [11, 9, 5],
            "latent_dim": 64,
            "dropout": 0.4,
            "rollout": 3,
            "spectral_k": 4,
        },
        "large": {
            "n_blocks": 4,
            "filters": [32, 64, 128, 256], # multiples of 8 for Tensor Core alignment
            "kernels": [11, 9, 5, 3],
            "latent_dim": 128,
            "dropout": 0.5,
            "rollout": 3,
            "spectral_k": 4,
        },
    }

    cfg = configs[tier]
    cfg["tier"] = tier

    # Adaptive kernel clipping: kernel can't exceed seq_len
    cfg["kernels"] = [min(k, seq_len) for k in cfg["kernels"]]
    # Ensure odd kernels for symmetric padding
    cfg["kernels"] = [k if k % 2 == 1 else k - 1 for k in cfg["kernels"]]

    return cfg


# =========================================================================
# KePIN Model Builder
# =========================================================================

class KePINModel(keras.Model):
    """Koopman-Enhanced Physics-Informed Network for domain-independent RUL prediction.

    This is a custom Keras Model that exposes both RUL predictions and
    Koopman outputs (for physics loss computation) in a single forward pass.
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        """
        Args:
            input_shape_tuple: (seq_len, n_features) — shape of one sample
            arch_config: dict from auto_configure(), or None for auto
            n_train: number of training samples (for auto-config)
        """
        super().__init__(**kwargs)

        seq_len, n_features = input_shape_tuple
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len, n_train)

        self.arch_config = arch_config
        self.seq_len = seq_len
        self.n_features = n_features

        # --- Encoder layers with SE blocks ---
        self.encoder_blocks = []
        for i in range(arch_config["n_blocks"]):
            filters = arch_config["filters"][i]
            se_bottleneck = max(filters // 8, 4)
            block = {
                "conv": keras.layers.Conv1D(
                    filters,
                    arch_config["kernels"][i],
                    padding="same",
                    kernel_initializer="he_normal",
                    name=f"enc_conv_{i}",
                ),
                "bn": keras.layers.BatchNormalization(name=f"enc_bn_{i}"),
                "relu": keras.layers.Activation("relu", name=f"enc_relu_{i}"),
                "se_gap": keras.layers.GlobalAveragePooling1D(name=f"se_gap_{i}"),
                "se_dense1": keras.layers.Dense(
                    se_bottleneck, activation="relu",
                    kernel_initializer="he_normal", name=f"se_dense1_{i}",
                ),
                "se_dense2": keras.layers.Dense(
                    filters, activation="sigmoid",
                    kernel_initializer="he_normal", name=f"se_dense2_{i}",
                ),
                "se_reshape": keras.layers.Reshape((1, filters), name=f"se_reshape_{i}"),
                "se_mul": keras.layers.Multiply(name=f"se_mul_{i}"),
            }
            self.encoder_blocks.append(block)

        # --- Latent projection ---
        self.latent_proj = keras.layers.Conv1D(
            arch_config["latent_dim"], 1, padding="same",
            kernel_initializer="he_normal", name="latent_projection",
        )
        self.latent_bn = keras.layers.BatchNormalization(name="latent_bn")

        # --- Koopman operator ---
        self.koopman = KoopmanOperator(
            latent_dim=arch_config["latent_dim"],
            rollout_steps=arch_config["rollout"],
            stability_mode="svd",
            name="koopman_operator",
        )

        # --- Dual pooling on latent sequence ---
        self.gap = keras.layers.GlobalAveragePooling1D(name="latent_gap")
        self.gmp = keras.layers.GlobalMaxPooling1D(name="latent_gmp")
        self.concat = keras.layers.Concatenate(name="dual_pool_concat")

        # --- Spectral feature dimension ---
        self.spectral_k = arch_config["spectral_k"]
        self.spec_dim = spectral_features_dim(self.spectral_k)

        # --- RUL prediction head (deeper for better representation) ---
        pool_dim = 2 * arch_config["latent_dim"]     # dual pooling
        head_input_dim = pool_dim + self.spec_dim     # + spectral features
        self.head_dense1 = keras.layers.Dense(
            128, activation="relu", kernel_initializer="he_normal", name="head_dense1",
        )
        self.head_bn1 = keras.layers.BatchNormalization(name="head_bn1")
        self.head_dense2 = keras.layers.Dense(
            64, activation="relu", kernel_initializer="he_normal", name="head_dense2",
        )
        self.head_dropout = keras.layers.Dropout(
            arch_config["dropout"], name="head_dropout",
        )
        # Output layer — use linear activation for RUL prediction (values can be 0-125)
        self.head_output = keras.layers.Dense(
            1, activation="relu", dtype="float32", name="rul_output",
        )

        # --- Loss weight layer (for auto-balancing) ---
        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        # Build by calling on dummy input
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs: (batch, seq_len, n_features) — raw sensor windows

        Returns:
            rul_pred:        (batch, 1) — predicted RUL
            koopman_outputs: dict with Koopman analysis results
        """
        x = inputs

        # --- Temporal Encoder ---
        for block in self.encoder_blocks:
            x = block["conv"](x)
            x = block["bn"](x, training=training)
            x = block["relu"](x)
            # SE attention (pre-built layers)
            se = block["se_gap"](x)
            se = block["se_dense1"](se)
            se = block["se_dense2"](se)
            se = block["se_reshape"](se)
            x = block["se_mul"]([x, se])

        # --- Project to latent space ---
        z_seq = self.latent_proj(x)           # (batch, T, d)
        z_seq = self.latent_bn(z_seq, training=training)

        # --- Koopman operator ---
        koopman_out = self.koopman(z_seq, training=training)

        # --- Dual pooling on full latent sequence ---
        # Cast to float32 for pooling and head to avoid mixed-precision dtype issues
        z_seq_f32 = tf.cast(z_seq, tf.float32)
        pool_avg = self.gap(z_seq_f32)             # (batch, d)
        pool_max = self.gmp(z_seq_f32)             # (batch, d)
        pooled = tf.concat([pool_avg, pool_max], axis=-1)  # (batch, 2d)

        # --- Spectral features ---
        # Extract from eigenvalues (same for all samples in batch — shared K)
        spec_feats = extract_spectral_features(
            koopman_out["eigenvalues"], top_k=self.spectral_k,
        )  # (spec_dim,)
        # Broadcast to batch
        batch_size = tf.shape(inputs)[0]
        spec_feats_batch = tf.tile(
            tf.expand_dims(tf.cast(spec_feats, tf.float32), 0), [batch_size, 1],
        )  # (batch, spec_dim)

        # --- Concatenate pooled + spectral ---
        head_input = tf.concat([pooled, spec_feats_batch], axis=-1)

        # --- RUL prediction head ---
        h = self.head_dense1(head_input)
        h = self.head_bn1(h, training=training)
        h = self.head_dense2(h)
        h = self.head_dropout(h, training=training)
        rul_pred = self.head_output(h)

        # Store koopman outputs for loss computation
        koopman_out["spectral_features"] = spec_feats

        return rul_pred, koopman_out

    def predict_rul(self, inputs):
        """Convenience method: predict RUL only (no Koopman outputs)."""
        rul_pred, _ = self(inputs, training=False)
        return rul_pred

    def get_koopman_matrix(self):
        """Extract the learned Koopman operator matrix."""
        return self.koopman._get_K().numpy()

    def get_eigenvalues(self):
        """Extract eigenvalues of the learned Koopman operator."""
        K = self.koopman._get_K()
        return tf.linalg.eigvals(K).numpy()

    def get_latent_states(self, inputs):
        """Extract the latent state sequence z(t) for visualization."""
        x = inputs
        for block in self.encoder_blocks:
            x = block["conv"](x)
            x = block["bn"](x, training=False)
            x = block["relu"](x)
            se = block["se_gap"](x)
            se = block["se_dense1"](se)
            se = block["se_dense2"](se)
            se = block["se_reshape"](se)
            x = block["se_mul"]([x, se])
        z_seq = self.latent_proj(x)
        z_seq = self.latent_bn(z_seq, training=False)
        return z_seq.numpy()

    def summary_config(self):
        """Print architecture configuration."""
        cfg = self.arch_config
        lines = [
            f"KePIN Architecture Configuration",
            f"  Tier: {cfg['tier']}",
            f"  Input: ({self.seq_len}, {self.n_features})",
            f"  Encoder: {cfg['n_blocks']} blocks",
        ]
        for i in range(cfg['n_blocks']):
            lines.append(f"    Block {i+1}: Conv1D({cfg['filters'][i]}, k={cfg['kernels'][i]}) + SE")
        lines.extend([
            f"  Latent dim: {cfg['latent_dim']}",
            f"  Koopman rollout: {cfg['rollout']} steps",
            f"  Spectral features: {self.spec_dim} (top-{self.spectral_k} modes)",
            f"  Head: Dense(64) → Dropout({cfg['dropout']}) → Dense(1)",
            f"  Loss weights: auto-balanced (7 terms)",
        ])
        return "\n".join(lines)


# =========================================================================
# Functional API builder (alternative to subclassed model)
# =========================================================================

def build_kepin_model(seq_len: int, n_features: int,
                      n_train: int = None, arch_config: dict = None):
    """Build KePIN model and return it.

    Args:
        seq_len: sequence length
        n_features: number of input features
        n_train: number of training samples (for auto-config)
        arch_config: override auto-configuration

    Returns:
        model: KePINModel instance
    """
    model = KePINModel(
        input_shape_tuple=(seq_len, n_features),
        arch_config=arch_config,
        n_train=n_train,
    )
    return model


# =========================================================================
# Adapter: convert 4D PI-DP-FCN data to 3D KePIN format
# =========================================================================

def convert_4d_to_3d(X):
    """Convert (samples, seq_len, 1, n_feat) → (samples, seq_len, n_feat).

    The GenericTimeSeriesDataset outputs 4D arrays for the original Conv2D
    architecture. KePIN uses Conv1D and expects 3D inputs.
    """
    if X.ndim == 4 and X.shape[2] == 1:
        return X[:, :, 0, :]
    elif X.ndim == 3:
        return X
    else:
        raise ValueError(f"Unexpected input shape: {X.shape}. "
                         f"Expected (N, T, 1, F) or (N, T, F).")


# =========================================================================
# Unit test
# =========================================================================

if __name__ == "__main__":
    print("=== KePIN Model — Unit Test ===\n")

    # Simulate different dataset sizes
    test_cases = [
        {"name": "Small (battery-like)", "seq_len": 20, "n_feat": 5, "n_train": 500},
        {"name": "Medium (bearing-like)", "seq_len": 30, "n_feat": 12, "n_train": 5000},
        {"name": "Large (turbofan-like)", "seq_len": 31, "n_feat": 15, "n_train": 20000},
        {"name": "XLarge (FD002-like)",   "seq_len": 21, "n_feat": 21, "n_train": 50000},
    ]

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        model = build_kepin_model(tc["seq_len"], tc["n_feat"], n_train=tc["n_train"])
        print(model.summary_config())

        # Test forward pass
        X = np.random.randn(8, tc["seq_len"], tc["n_feat"]).astype(np.float32)
        rul_pred, koopman_out = model(tf.constant(X), training=False)

        print(f"  RUL pred shape:     {rul_pred.shape}")
        print(f"  Koopman 1-step:     {koopman_out['one_step_pred'].shape}")
        print(f"  Eigenvalues:        {koopman_out['eigenvalues'].numpy()[:3]}...")
        print(f"  Spectral features:  {koopman_out['spectral_features'].shape}")

    # Test 4D → 3D converter
    X_4d = np.random.randn(16, 30, 1, 12).astype(np.float32)
    X_3d = convert_4d_to_3d(X_4d)
    print(f"\n4D→3D conversion: {X_4d.shape} → {X_3d.shape}")

    print("\n✓ All model configurations build and run successfully.")
