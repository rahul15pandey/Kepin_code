# -*- coding: utf-8 -*-
"""
KePIN Model v2 — Koopman-Enhanced Physics-Informed Network.

Major improvements over v1:
  - Residual Conv1D blocks with skip connections
  - Bidirectional LSTM for temporal modeling
  - Multi-head self-attention for focusing on critical time steps
  - Deeper RUL head with skip connections
  - Layer normalization for training stability
  - Channel attention (SE) retained

Architecture:
  [Input] → [ResConv1D+SE blocks] → [BiLSTM] → [Multi-Head Attention]
          → [Koopman Module] → [Spectral Features]
          → [Deep RUL Head with Skip] → RUL
"""

import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable

from koopman_module import KoopmanOperator, extract_spectral_features, spectral_features_dim
from kepin_losses import KePINLossWeights


# =========================================================================
# Auto-configuration
# =========================================================================

def auto_configure(n_features: int, seq_len: int, n_train: int = None):
    """Determine architecture hyperparameters from data shape."""
    if n_features <= 10:
        tier = "small"
    elif n_features <= 16:
        tier = "medium"
    else:
        tier = "large"

    configs = {
        "small": {
            "n_blocks": 3,
            "filters": [64, 128, 128],
            "kernels": [7, 5, 3],
            "latent_dim": 64,
            "lstm_units": 64,
            "n_heads": 4,
            "head_key_dim": 16,
            "dropout": 0.3,
            "rollout": 3,
            "spectral_k": 4,
        },
        "medium": {
            "n_blocks": 4,
            "filters": [64, 128, 128, 256],
            "kernels": [7, 5, 5, 3],
            "latent_dim": 128,
            "lstm_units": 128,
            "n_heads": 4,
            "head_key_dim": 32,
            "dropout": 0.35,
            "rollout": 3,
            "spectral_k": 5,
        },
        "large": {
            "n_blocks": 4,
            "filters": [64, 128, 256, 256],
            "kernels": [7, 5, 5, 3],
            "latent_dim": 128,
            "lstm_units": 128,
            "n_heads": 8,
            "head_key_dim": 32,
            "dropout": 0.4,
            "rollout": 3,
            "spectral_k": 5,
        },
    }

    cfg = configs[tier]
    cfg["tier"] = tier
    cfg["kernels"] = [min(k, seq_len) for k in cfg["kernels"]]
    cfg["kernels"] = [k if k % 2 == 1 else k - 1 for k in cfg["kernels"]]
    return cfg


# =========================================================================
# KePIN Model v2
# =========================================================================

class KePINModel(keras.Model):
    """Koopman-Enhanced Physics-Informed Network v2."""

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None,
                 n_active_losses=7, **kwargs):
        super().__init__(**kwargs)

        seq_len, n_features = input_shape_tuple
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len, n_train)

        self.arch_config = arch_config
        self.seq_len = seq_len
        self.n_features = n_features

        # ---- Input projection ----
        self.input_proj = keras.layers.Conv1D(
            arch_config["filters"][0], 1, padding="same",
            kernel_initializer="he_normal", name="input_proj",
        )
        self.input_bn = keras.layers.BatchNormalization(name="input_bn")

        # ---- Residual Conv1D Encoder Blocks with SE ----
        self.encoder_blocks = []
        prev_filters = arch_config["filters"][0]  # after input_proj
        for i in range(arch_config["n_blocks"]):
            filters = arch_config["filters"][i]
            se_bottleneck = max(filters // 8, 4)
            needs_proj = (filters != prev_filters)
            block = {
                "conv1": keras.layers.Conv1D(
                    filters, arch_config["kernels"][i], padding="same",
                    kernel_initializer="he_normal", name=f"enc_conv1_{i}",
                ),
                "bn1": keras.layers.BatchNormalization(name=f"enc_bn1_{i}"),
                "conv2": keras.layers.Conv1D(
                    filters, 3, padding="same",
                    kernel_initializer="he_normal", name=f"enc_conv2_{i}",
                ),
                "bn2": keras.layers.BatchNormalization(name=f"enc_bn2_{i}"),
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
                "skip_proj": keras.layers.Conv1D(
                    filters, 1, padding="same",
                    kernel_initializer="he_normal", name=f"skip_proj_{i}",
                ) if needs_proj else None,
                "skip_bn": keras.layers.BatchNormalization(name=f"skip_bn_{i}")
                if needs_proj else None,
            }
            self.encoder_blocks.append(block)
            prev_filters = filters

        # ---- BiLSTM ----
        lstm_units = arch_config["lstm_units"]
        self.bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_units, return_sequences=True,
                              dropout=0.1, recurrent_dropout=0.0,
                              name="lstm_fwd"),
            name="bilstm",
        )
        self.lstm_ln = keras.layers.LayerNormalization(name="lstm_ln")

        self.post_lstm_proj = keras.layers.Conv1D(
            arch_config["latent_dim"], 1, padding="same",
            kernel_initializer="he_normal", name="post_lstm_proj",
        )
        self.post_lstm_bn = keras.layers.BatchNormalization(name="post_lstm_bn")

        # ---- Multi-head self-attention ----
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=arch_config["n_heads"],
            key_dim=arch_config["head_key_dim"],
            dropout=0.1,
            name="temporal_mha",
        )
        self.mha_ln = keras.layers.LayerNormalization(name="mha_ln")
        self.mha_dropout = keras.layers.Dropout(0.1, name="mha_dropout")

        self.ff_dense1 = keras.layers.Dense(
            arch_config["latent_dim"] * 2, activation="gelu",
            kernel_initializer="he_normal", name="ff_dense1",
        )
        self.ff_dense2 = keras.layers.Dense(
            arch_config["latent_dim"],
            kernel_initializer="he_normal", name="ff_dense2",
        )
        self.ff_ln = keras.layers.LayerNormalization(name="ff_ln")
        self.ff_dropout = keras.layers.Dropout(0.1, name="ff_dropout")

        # ---- Latent projection ----
        self.latent_proj = keras.layers.Conv1D(
            arch_config["latent_dim"], 1, padding="same",
            kernel_initializer="he_normal", name="latent_projection",
        )
        self.latent_bn = keras.layers.BatchNormalization(name="latent_bn")

        # ---- Koopman operator ----
        self.koopman = KoopmanOperator(
            latent_dim=arch_config["latent_dim"],
            rollout_steps=arch_config["rollout"],
            stability_mode="svd",
            name="koopman_operator",
        )

        # ---- Dual pooling ----
        self.gap = keras.layers.GlobalAveragePooling1D(name="latent_gap")
        self.gmp = keras.layers.GlobalMaxPooling1D(name="latent_gmp")

        # ---- Spectral features ----
        self.spectral_k = arch_config["spectral_k"]
        self.spec_dim = spectral_features_dim(self.spectral_k)

        # ---- Deep RUL head ----
        self.head_dense1 = keras.layers.Dense(
            256, activation="relu", kernel_initializer="he_normal", name="head_dense1",
        )
        self.head_bn1 = keras.layers.BatchNormalization(name="head_bn1")
        self.head_drop1 = keras.layers.Dropout(arch_config["dropout"] * 0.5, name="head_drop1")

        self.head_dense2 = keras.layers.Dense(
            128, activation="relu", kernel_initializer="he_normal", name="head_dense2",
        )
        self.head_bn2 = keras.layers.BatchNormalization(name="head_bn2")
        self.head_drop2 = keras.layers.Dropout(arch_config["dropout"], name="head_drop2")

        self.head_dense3 = keras.layers.Dense(
            64, activation="relu", kernel_initializer="he_normal", name="head_dense3",
        )
        self.head_drop3 = keras.layers.Dropout(arch_config["dropout"], name="head_drop3")

        # Skip from pool to final
        self.head_skip_dense = keras.layers.Dense(
            64, activation="relu", kernel_initializer="he_normal", name="head_skip_dense",
        )

        self.head_output = keras.layers.Dense(
            1, activation="linear", dtype="float32", name="rul_output",
        )

        # ---- Loss weight layer ----
        self.loss_weight_layer = KePINLossWeights(n_losses=n_active_losses, name="loss_weights")

        # Build
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = inputs

        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x, training=training)
        x = tf.nn.relu(x)

        # Residual Conv1D Encoder
        for block in self.encoder_blocks:
            residual = x
            h = block["conv1"](x)
            h = block["bn1"](h, training=training)
            h = tf.nn.relu(h)
            h = block["conv2"](h)
            h = block["bn2"](h, training=training)
            se = block["se_gap"](h)
            se = block["se_dense1"](se)
            se = block["se_dense2"](se)
            se = block["se_reshape"](se)
            h = block["se_mul"]([h, se])
            if block["skip_proj"] is not None:
                residual = block["skip_proj"](residual)
                residual = block["skip_bn"](residual, training=training)
            x = tf.nn.relu(h + residual)

        # BiLSTM
        x = self.bilstm(x, training=training)
        x = self.lstm_ln(x, training=training)
        x = self.post_lstm_proj(x)
        x = self.post_lstm_bn(x, training=training)

        # Multi-head self-attention (with residual)
        attn_out = self.mha(x, x, training=training)
        attn_out = self.mha_dropout(attn_out, training=training)
        x = self.mha_ln(x + attn_out, training=training)

        # Feed-forward (with residual)
        ff_out = self.ff_dense1(x)
        ff_out = self.ff_dense2(ff_out)
        ff_out = self.ff_dropout(ff_out, training=training)
        x = self.ff_ln(x + ff_out, training=training)

        # Latent projection
        z_seq = self.latent_proj(x)
        z_seq = self.latent_bn(z_seq, training=training)

        # Koopman operator
        koopman_out = self.koopman(z_seq, training=training)

        # Dual pooling
        z_seq_f32 = tf.cast(z_seq, tf.float32)
        pool_avg = self.gap(z_seq_f32)
        pool_max = self.gmp(z_seq_f32)
        pooled = tf.concat([pool_avg, pool_max], axis=-1)

        # Spectral features
        spec_feats = extract_spectral_features(
            koopman_out["eigenvalues"], top_k=self.spectral_k,
        )
        batch_size = tf.shape(inputs)[0]
        spec_feats_batch = tf.tile(
            tf.expand_dims(tf.cast(spec_feats, tf.float32), 0), [batch_size, 1],
        )

        head_input = tf.concat([pooled, spec_feats_batch], axis=-1)

        # Deep RUL head with skip
        h = self.head_dense1(head_input)
        h = self.head_bn1(h, training=training)
        h = self.head_drop1(h, training=training)
        h = self.head_dense2(h)
        h = self.head_bn2(h, training=training)
        h = self.head_drop2(h, training=training)
        h = self.head_dense3(h)
        h = self.head_drop3(h, training=training)

        h_skip = self.head_skip_dense(pooled)
        h = h + h_skip

        rul_pred = self.head_output(h)
        koopman_out["spectral_features"] = spec_feats
        return rul_pred, koopman_out

    def predict_rul(self, inputs):
        rul_pred, _ = self(inputs, training=False)
        return rul_pred

    def get_koopman_matrix(self):
        return self.koopman._get_K().numpy()

    def get_eigenvalues(self):
        K = self.koopman._get_K()
        return tf.linalg.eigvals(tf.cast(K, tf.float32)).numpy()

    def get_latent_states(self, inputs):
        x = inputs
        x = self.input_proj(x)
        x = self.input_bn(x, training=False)
        x = tf.nn.relu(x)
        for block in self.encoder_blocks:
            residual = x
            h = block["conv1"](x)
            h = block["bn1"](h, training=False)
            h = tf.nn.relu(h)
            h = block["conv2"](h)
            h = block["bn2"](h, training=False)
            se = block["se_gap"](h)
            se = block["se_dense1"](se)
            se = block["se_dense2"](se)
            se = block["se_reshape"](se)
            h = block["se_mul"]([h, se])
            if block["skip_proj"] is not None:
                residual = block["skip_proj"](residual)
                residual = block["skip_bn"](residual, training=False)
            x = tf.nn.relu(h + residual)
        x = self.bilstm(x, training=False)
        x = self.lstm_ln(x, training=False)
        x = self.post_lstm_proj(x)
        x = self.post_lstm_bn(x, training=False)
        attn_out = self.mha(x, x, training=False)
        x = self.mha_ln(x + attn_out, training=False)
        ff_out = self.ff_dense1(x)
        ff_out = self.ff_dense2(ff_out)
        x = self.ff_ln(x + ff_out, training=False)
        z_seq = self.latent_proj(x)
        z_seq = self.latent_bn(z_seq, training=False)
        return z_seq.numpy()

    def summary_config(self):
        cfg = self.arch_config
        lines = [
            f"KePIN v2 Architecture",
            f"  Tier: {cfg['tier']}",
            f"  Input: ({self.seq_len}, {self.n_features})",
            f"  Encoder: {cfg['n_blocks']} residual Conv1D+SE blocks",
        ]
        for i in range(cfg['n_blocks']):
            lines.append(f"    Block {i+1}: Conv1D({cfg['filters'][i]}, k={cfg['kernels'][i]}) + Conv1D(3) + SE [residual]")
        lines.extend([
            f"  BiLSTM: {cfg['lstm_units']} units (bidirectional)",
            f"  Multi-Head Attention: {cfg['n_heads']} heads, key_dim={cfg['head_key_dim']}",
            f"  Latent dim: {cfg['latent_dim']}",
            f"  Koopman rollout: {cfg['rollout']} steps",
            f"  Spectral features: {self.spec_dim} (top-{self.spectral_k} modes)",
            f"  Head: Dense(256)→BN→Dense(128)→BN→Dense(64)→Skip→Dense(1)",
            f"  Dropout: {cfg['dropout']}",
        ])
        return "\n".join(lines)


# =========================================================================
# Builders & converters
# =========================================================================

def build_kepin_model(seq_len, n_features, n_train=None, arch_config=None,
                      n_active_losses=7):
    return KePINModel(
        input_shape_tuple=(seq_len, n_features),
        arch_config=arch_config, n_train=n_train,
        n_active_losses=n_active_losses,
    )


def convert_4d_to_3d(X):
    if X.ndim == 4 and X.shape[2] == 1:
        return X[:, :, 0, :]
    elif X.ndim == 3:
        return X
    else:
        raise ValueError(f"Unexpected input shape: {X.shape}.")


if __name__ == "__main__":
    print("=== KePIN Model v2 — Unit Test ===\n")
    test_cases = [
        {"name": "Medium (FD001-like)", "seq_len": 30, "n_feat": 18, "n_train": 20000},
        {"name": "Large (FD002-like)", "seq_len": 30, "n_feat": 24, "n_train": 50000},
    ]
    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        model = build_kepin_model(tc["seq_len"], tc["n_feat"], n_train=tc["n_train"])
        print(model.summary_config())
        X = np.random.randn(4, tc["seq_len"], tc["n_feat"]).astype(np.float32)
        rul_pred, koopman_out = model(tf.constant(X), training=False)
        print(f"  RUL pred shape: {rul_pred.shape}")
        n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
        print(f"  Total params:   {n_params:,}")
    print("\n✓ All v2 configurations OK.")
