# -*- coding: utf-8 -*-
"""
Baseline Models for KePIN Comparison Study.

Implements 6 established baseline architectures for fair comparison
against KePIN ablation variants. All models share the same:
  - Input format: (batch, seq_len, n_features) — 3D time-series
  - Output format: (batch, 1) — predicted RUL
  - Interface: predict_rul(), get_eigenvalues() [returns dummy]

Baselines:
  1. MLP          — Flatten → Dense layers (no temporal modelling)
  2. LSTM         — Stacked LSTM → Dense head
  3. BiLSTM       — Bidirectional LSTM → Dense head
  4. CNN-LSTM     — Conv1D encoder → LSTM → Dense head
  5. Vanilla FCN  — Conv1D blocks (no SE, no dual-pool) + GAP → Dense
  6. PI-DP-FCN    — Original Conv1D+SE+DualPool with physics loss (adapted)

Each model class follows the same interface as KePINModel and BaselineFCN
to allow drop-in replacement in the ablation/comparison pipeline.
"""

import numpy as np
import tensorflow as tf
import keras

from kepin_losses import KePINLossWeights
from kepin_model import auto_configure


# =========================================================================
# Common interface mixin
# =========================================================================

class BaselineModelMixin:
    """Shared interface methods for all baseline models."""

    def predict_rul(self, inputs):
        rul_pred, _ = self(inputs, training=False)
        return rul_pred

    def get_eigenvalues(self):
        d = getattr(self, '_latent_dim', 32)
        return np.zeros(d, dtype=np.complex128)

    def get_koopman_matrix(self):
        d = getattr(self, '_latent_dim', 32)
        return np.zeros((d, d))

    def _dummy_koopman(self, inputs, d=None):
        """Return dummy Koopman outputs for training loop compatibility."""
        if d is None:
            d = getattr(self, '_latent_dim', 32)
        batch_size = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        return {
            "one_step_pred": tf.zeros((batch_size, T - 1, d)),
            "one_step_target": tf.zeros((batch_size, T - 1, d)),
            "multi_step_pred": tf.zeros((batch_size, 1, 1, d)),
            "multi_step_target": tf.zeros((batch_size, 1, 1, d)),
            "eigenvalues": tf.zeros((d,), dtype=tf.complex64),
            "final_state": tf.zeros((batch_size, d)),
        }


# =========================================================================
# 1. MLP Baseline
# =========================================================================

class MLPBaseline(keras.Model, BaselineModelMixin):
    """Multi-Layer Perceptron — no temporal modelling.

    Flattens the input window and passes through dense layers.
    Tests whether temporal structure matters at all.
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        self._latent_dim = 32

        flat_dim = seq_len * n_features
        hidden = min(256, max(64, flat_dim // 2))

        self.flatten_layer = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(hidden, activation="relu",
                                         kernel_initializer="he_normal")
        self.bn1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(hidden // 2, activation="relu",
                                         kernel_initializer="he_normal")
        self.bn2 = keras.layers.BatchNormalization()
        self.dense3 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout = keras.layers.Dropout(0.4)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")

        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = self.flatten_layer(inputs)
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dense3(x)
        x = self.dropout(x, training=training)
        rul_pred = self.output_layer(x)
        return rul_pred, self._dummy_koopman(inputs)


# =========================================================================
# 2. LSTM Baseline
# =========================================================================

class LSTMBaseline(keras.Model, BaselineModelMixin):
    """Stacked LSTM — standard recurrent approach for sequence modelling.

    Two LSTM layers with dropout, followed by dense head.
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        self._latent_dim = 64

        units = 64 if n_features <= 10 else 128

        self.lstm1 = keras.layers.LSTM(units, return_sequences=True,
                                       kernel_initializer="glorot_uniform")
        self.dropout1 = keras.layers.Dropout(0.3)
        self.lstm2 = keras.layers.LSTM(units // 2,
                                       kernel_initializer="glorot_uniform")
        self.dropout2 = keras.layers.Dropout(0.3)
        self.dense1 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout3 = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")

        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = self.lstm1(inputs, training=training)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        rul_pred = self.output_layer(x)
        return rul_pred, self._dummy_koopman(inputs, d=self._latent_dim)


# =========================================================================
# 3. BiLSTM Baseline
# =========================================================================

class BiLSTMBaseline(keras.Model, BaselineModelMixin):
    """Bidirectional LSTM — captures both forward and backward temporal patterns.

    Commonly used baseline in RUL prediction literature.
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        self._latent_dim = 64

        units = 64 if n_features <= 10 else 128

        self.bilstm1 = keras.layers.Bidirectional(
            keras.layers.LSTM(units, return_sequences=True,
                              kernel_initializer="glorot_uniform"),
            name="bilstm1",
        )
        self.dropout1 = keras.layers.Dropout(0.3)
        self.bilstm2 = keras.layers.Bidirectional(
            keras.layers.LSTM(units // 2,
                              kernel_initializer="glorot_uniform"),
            name="bilstm2",
        )
        self.dropout2 = keras.layers.Dropout(0.3)
        self.dense1 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout3 = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")

        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = self.bilstm1(inputs, training=training)
        x = self.dropout1(x, training=training)
        x = self.bilstm2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense1(x)
        x = self.dropout3(x, training=training)
        rul_pred = self.output_layer(x)
        return rul_pred, self._dummy_koopman(inputs, d=self._latent_dim)


# =========================================================================
# 4. CNN-LSTM Hybrid
# =========================================================================

class CNNLSTMBaseline(keras.Model, BaselineModelMixin):
    """CNN-LSTM hybrid — Conv1D feature extraction followed by LSTM.

    Popular architecture that combines local pattern extraction (CNN)
    with sequential modelling (LSTM).
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len, n_train)
        self._latent_dim = 64

        # CNN encoder (2 Conv1D blocks, no SE)
        self.conv1 = keras.layers.Conv1D(
            64, min(7, seq_len), padding="same",
            kernel_initializer="he_normal",
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.relu1 = keras.layers.Activation("relu")

        self.conv2 = keras.layers.Conv1D(
            128, min(5, seq_len), padding="same",
            kernel_initializer="he_normal",
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.relu2 = keras.layers.Activation("relu")

        # LSTM on encoded features
        self.lstm = keras.layers.LSTM(64, kernel_initializer="glorot_uniform")
        self.dropout1 = keras.layers.Dropout(0.3)

        # Dense head
        self.dense1 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout2 = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")

        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.lstm(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        rul_pred = self.output_layer(x)
        return rul_pred, self._dummy_koopman(inputs, d=self._latent_dim)


# =========================================================================
# 5. Vanilla FCN (no SE, no dual pooling)
# =========================================================================

class VanillaFCN(keras.Model, BaselineModelMixin):
    """Vanilla Fully Convolutional Network — Conv1D + BN + ReLU + GAP.

    No SE attention, no dual pooling. Tests the contribution of
    SE and dual pooling to the overall performance.
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len, n_train)
        self._latent_dim = arch_config.get("latent_dim", 64)

        self.conv_blocks = []
        for i in range(arch_config["n_blocks"]):
            self.conv_blocks.append({
                "conv": keras.layers.Conv1D(
                    arch_config["filters"][i],
                    arch_config["kernels"][i],
                    padding="same",
                    kernel_initializer="he_normal",
                    name=f"conv_{i}",
                ),
                "bn": keras.layers.BatchNormalization(name=f"bn_{i}"),
                "relu": keras.layers.Activation("relu", name=f"relu_{i}"),
            })

        # Simple GAP only (no dual pooling, no SE)
        self.gap = keras.layers.GlobalAveragePooling1D()

        self.dense1 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout = keras.layers.Dropout(arch_config.get("dropout", 0.4))
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")

        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = inputs
        for block in self.conv_blocks:
            x = block["conv"](x)
            x = block["bn"](x, training=training)
            x = block["relu"](x)

        x = self.gap(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        rul_pred = self.output_layer(x)
        return rul_pred, self._dummy_koopman(inputs, d=self._latent_dim)


# =========================================================================
# 6. PI-DP-FCN (Physics-Informed Dual-Pooling FCN — original method, Conv1D)
# =========================================================================

class PIDPFCNBaseline(keras.Model, BaselineModelMixin):
    """PI-DP-SE-FCN adapted to Conv1D — the original method from this codebase.

    Conv1D + SE attention + Dual Pooling, using the original physics-informed
    loss (MSE + asymmetric + monotonicity + slope). No Koopman operator.

    This is the direct predecessor of KePIN — the main comparison target.
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len, n_train)
        self._latent_dim = arch_config.get("latent_dim", 64)
        self.arch_config = arch_config

        self.encoder_blocks = []
        for i in range(arch_config["n_blocks"]):
            self.encoder_blocks.append({
                "conv": keras.layers.Conv1D(
                    arch_config["filters"][i],
                    arch_config["kernels"][i],
                    padding="same",
                    kernel_initializer="he_normal",
                    name=f"enc_conv_{i}",
                ),
                "bn": keras.layers.BatchNormalization(name=f"enc_bn_{i}"),
                "relu": keras.layers.Activation("relu", name=f"enc_relu_{i}"),
            })

        # Dual pooling
        self.gap = keras.layers.GlobalAveragePooling1D(name="gap")
        self.gmp = keras.layers.GlobalMaxPooling1D(name="gmp")
        self.concat = keras.layers.Concatenate(name="dual_concat")

        # Head
        self.head_dense = keras.layers.Dense(
            64, activation="relu", kernel_initializer="he_normal", name="head_dense",
        )
        self.head_dropout = keras.layers.Dropout(
            arch_config.get("dropout", 0.4), name="head_dropout",
        )
        self.head_output = keras.layers.Dense(
            1, activation="relu", dtype="float32", name="rul_output",
        )

        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def _se_block(self, x, ratio=8):
        filters = int(x.shape[-1])
        se = keras.layers.GlobalAveragePooling1D()(x)
        se = keras.layers.Dense(max(filters // ratio, 4), activation='relu')(se)
        se = keras.layers.Dense(filters, activation='sigmoid')(se)
        se = keras.layers.Reshape((1, filters))(se)
        return keras.layers.Multiply()([x, se])

    def call(self, inputs, training=None):
        x = inputs
        for block in self.encoder_blocks:
            x = block["conv"](x)
            x = block["bn"](x, training=training)
            x = block["relu"](x)
            x = self._se_block(x)

        pool_avg = self.gap(x)
        pool_max = self.gmp(x)
        pooled = self.concat([pool_avg, pool_max])

        h = self.head_dense(pooled)
        h = self.head_dropout(h, training=training)
        rul_pred = self.head_output(h)

        return rul_pred, self._dummy_koopman(inputs, d=self._latent_dim)


# =========================================================================
# 7. Transformer Encoder Baseline
# =========================================================================

class TransformerBaseline(keras.Model, BaselineModelMixin):
    """Transformer Encoder — self-attention for temporal modelling.

    Lightweight transformer with positional encoding, multi-head attention,
    and feedforward layers. Represents the attention-based SOTA family.
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        self._latent_dim = 64

        d_model = 64
        n_heads = 4
        ff_dim = 128
        n_layers = 2
        dropout_rate = 0.3

        # Input projection to d_model
        self.input_proj = keras.layers.Dense(d_model, name="input_proj")

        # Positional encoding (learnable)
        self.pos_embedding = keras.layers.Embedding(
            input_dim=max(seq_len, 512), output_dim=d_model, name="pos_emb",
        )

        # Transformer encoder layers
        self.enc_layers = []
        for i in range(n_layers):
            self.enc_layers.append({
                "mha": keras.layers.MultiHeadAttention(
                    num_heads=n_heads, key_dim=d_model // n_heads,
                    name=f"mha_{i}",
                ),
                "ln1": keras.layers.LayerNormalization(name=f"ln1_{i}"),
                "ff1": keras.layers.Dense(ff_dim, activation="relu", name=f"ff1_{i}"),
                "ff2": keras.layers.Dense(d_model, name=f"ff2_{i}"),
                "ln2": keras.layers.LayerNormalization(name=f"ln2_{i}"),
                "drop1": keras.layers.Dropout(dropout_rate, name=f"drop1_{i}"),
                "drop2": keras.layers.Dropout(dropout_rate, name=f"drop2_{i}"),
            })

        # Pooling and head
        self.gap = keras.layers.GlobalAveragePooling1D()
        self.head_dense = keras.layers.Dense(64, activation="relu",
                                             kernel_initializer="he_normal")
        self.head_dropout = keras.layers.Dropout(dropout_rate)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")

        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Project to d_model
        x = self.input_proj(inputs)  # (batch, T, d_model)

        # Add positional encoding
        positions = tf.range(seq_len)
        pos_enc = self.pos_embedding(positions)  # (T, d_model)
        x = x + tf.expand_dims(pos_enc, 0)  # broadcast

        # Transformer encoder
        for layer in self.enc_layers:
            # Multi-head self-attention
            attn_out = layer["mha"](x, x, training=training)
            attn_out = layer["drop1"](attn_out, training=training)
            x = layer["ln1"](x + attn_out)

            # Feedforward
            ff_out = layer["ff1"](x)
            ff_out = layer["ff2"](ff_out)
            ff_out = layer["drop2"](ff_out, training=training)
            x = layer["ln2"](x + ff_out)

        # Pool and predict
        x = self.gap(x)
        x = self.head_dense(x)
        x = self.head_dropout(x, training=training)
        rul_pred = self.output_layer(x)

        return rul_pred, self._dummy_koopman(inputs, d=self._latent_dim)


# =========================================================================
# Model registry
# =========================================================================

BASELINE_REGISTRY = {
    "mlp": {
        "class": MLPBaseline,
        "name": "MLP",
        "description": "Flatten → Dense (no temporal modelling)",
    },
    "lstm": {
        "class": LSTMBaseline,
        "name": "LSTM",
        "description": "Stacked LSTM → Dense",
    },
    "bilstm": {
        "class": BiLSTMBaseline,
        "name": "BiLSTM",
        "description": "Bidirectional LSTM → Dense",
    },
    "cnn_lstm": {
        "class": CNNLSTMBaseline,
        "name": "CNN-LSTM",
        "description": "Conv1D → LSTM → Dense",
    },
    "vanilla_fcn": {
        "class": VanillaFCN,
        "name": "Vanilla FCN",
        "description": "Conv1D + BN + ReLU + GAP (no SE, no dual-pool)",
    },
    "pi_dp_fcn": {
        "class": PIDPFCNBaseline,
        "name": "PI-DP-FCN",
        "description": "Conv1D + SE + Dual-Pool with physics loss (original method)",
    },
    "transformer": {
        "class": TransformerBaseline,
        "name": "Transformer",
        "description": "Transformer encoder with self-attention",
    },
}


def build_baseline_model(model_key: str, seq_len: int, n_features: int,
                         n_train: int = None):
    """Build a baseline model by registry key.

    Args:
        model_key: key in BASELINE_REGISTRY
        seq_len: sequence length
        n_features: number of input features
        n_train: number of training samples

    Returns:
        model: baseline model instance
    """
    if model_key not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {model_key}. "
                         f"Available: {list(BASELINE_REGISTRY.keys())}")

    entry = BASELINE_REGISTRY[model_key]
    arch_config = auto_configure(n_features, seq_len, n_train)
    model = entry["class"](
        input_shape_tuple=(seq_len, n_features),
        arch_config=arch_config,
        n_train=n_train,
    )
    return model


# =========================================================================
# Unit test
# =========================================================================

if __name__ == "__main__":
    print("=== Baseline Models — Unit Test ===\n")

    seq_len, n_feat = 30, 14
    batch = 8

    X = np.random.randn(batch, seq_len, n_feat).astype(np.float32)

    for key, entry in BASELINE_REGISTRY.items():
        print(f"\n--- {entry['name']} ({key}) ---")
        print(f"  {entry['description']}")

        model = build_baseline_model(key, seq_len, n_feat, n_train=5000)

        rul_pred, koopman_out = model(tf.constant(X), training=False)
        print(f"  RUL pred shape:     {rul_pred.shape}")
        print(f"  Params:             {model.count_params():,}")
        print(f"  Koopman output:     {list(koopman_out.keys())}")

        # Test predict_rul
        rul_only = model.predict_rul(tf.constant(X))
        assert rul_only.shape == (batch, 1), f"Expected ({batch}, 1), got {rul_only.shape}"

    print("\n✓ All 7 baseline models build and run successfully.")
