# -*- coding: utf-8 -*-
"""
Baseline architectures for comparison against KePIN.

Seven models sharing a common interface (``predict_rul``, ``get_eigenvalues``):
  1. MLP — Flatten → Dense (no temporal modelling)
  2. LSTM — Stacked LSTM → Dense
  3. BiLSTM — Bidirectional LSTM → Dense
  4. CNN-LSTM — Conv1D encoder → LSTM → Dense
  5. VanillaFCN — Conv1D + BN + ReLU + GAP
  6. PI-DP-FCN — Conv1D + SE + Dual-Pool (original method)
  7. Transformer — Multi-head self-attention encoder
"""

import numpy as np
import tensorflow as tf
import keras

from kepin.losses.composite import KePINLossWeights
from kepin.models.kepin_model import auto_configure


class BaselineModelMixin:
    """Shared interface for all baseline models."""

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


class MLPBaseline(keras.Model, BaselineModelMixin):
    """Multi-Layer Perceptron baseline — no temporal modelling."""

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
        rul = self.output_layer(x)
        return rul, self._dummy_koopman(inputs)


class LSTMBaseline(keras.Model, BaselineModelMixin):
    """Stacked LSTM baseline."""

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        self._latent_dim = 64
        self.lstm1 = keras.layers.LSTM(64, return_sequences=True, name="lstm1")
        self.lstm2 = keras.layers.LSTM(64, name="lstm2")
        self.bn = keras.layers.BatchNormalization()
        self.dense1 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")
        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        x = self.bn(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        rul = self.output_layer(x)
        return rul, self._dummy_koopman(inputs)


class BiLSTMBaseline(keras.Model, BaselineModelMixin):
    """Bidirectional LSTM baseline."""

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        self._latent_dim = 64
        self.bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True), name="bilstm")
        self.lstm2 = keras.layers.LSTM(64, name="lstm2")
        self.bn = keras.layers.BatchNormalization()
        self.dense1 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")
        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = self.bilstm(inputs, training=training)
        x = self.lstm2(x, training=training)
        x = self.bn(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        rul = self.output_layer(x)
        return rul, self._dummy_koopman(inputs)


class CNNLSTMBaseline(keras.Model, BaselineModelMixin):
    """CNN-LSTM hybrid baseline."""

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        self._latent_dim = 64
        self.conv1 = keras.layers.Conv1D(64, 5, padding="same",
                                         kernel_initializer="he_normal")
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv1D(128, 3, padding="same",
                                         kernel_initializer="he_normal")
        self.bn2 = keras.layers.BatchNormalization()
        self.lstm = keras.layers.LSTM(64, name="lstm")
        self.dense1 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")
        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.lstm(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        rul = self.output_layer(x)
        return rul, self._dummy_koopman(inputs)


class VanillaFCN(keras.Model, BaselineModelMixin):
    """Vanilla FCN — Conv1D + BN + ReLU + GAP (no SE, no dual-pool)."""

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len, n_train)
        self._latent_dim = arch_config.get("latent_dim", 64)

        self.convs, self.bns = [], []
        for i in range(arch_config["n_blocks"]):
            f = arch_config["filters"][i]
            k = arch_config["kernels"][i]
            self.convs.append(keras.layers.Conv1D(f, k, padding="same",
                                                   kernel_initializer="he_normal",
                                                   name=f"conv_{i}"))
            self.bns.append(keras.layers.BatchNormalization(name=f"bn_{i}"))

        self.gap = keras.layers.GlobalAveragePooling1D()
        self.dense1 = keras.layers.Dense(128, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout = keras.layers.Dropout(arch_config.get("dropout", 0.3))
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")
        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = inputs
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        rul = self.output_layer(x)
        return rul, self._dummy_koopman(inputs)


class PIDPFCNBaseline(keras.Model, BaselineModelMixin):
    """PI-DP-FCN — Conv1D + SE + Dual-Pool with physics loss (original method)."""

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len, n_train)
        self._latent_dim = arch_config.get("latent_dim", 64)

        self.convs, self.bns = [], []
        self.se_layers = []
        for i in range(arch_config["n_blocks"]):
            f = arch_config["filters"][i]
            k = arch_config["kernels"][i]
            self.convs.append(keras.layers.Conv1D(f, k, padding="same",
                                                   kernel_initializer="he_normal",
                                                   name=f"conv_{i}"))
            self.bns.append(keras.layers.BatchNormalization(name=f"bn_{i}"))
            se_b = max(f // 8, 4)
            self.se_layers.append({
                "gap": keras.layers.GlobalAveragePooling1D(name=f"se_gap_{i}"),
                "d1": keras.layers.Dense(se_b, activation="relu", name=f"se_d1_{i}"),
                "d2": keras.layers.Dense(f, activation="sigmoid", name=f"se_d2_{i}"),
                "reshape": keras.layers.Reshape((1, f), name=f"se_rs_{i}"),
                "mul": keras.layers.Multiply(name=f"se_mul_{i}"),
            })

        self.gap = keras.layers.GlobalAveragePooling1D(name="gap")
        self.gmp = keras.layers.GlobalMaxPooling1D(name="gmp")
        self.dense1 = keras.layers.Dense(128, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout = keras.layers.Dropout(arch_config.get("dropout", 0.3))
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")
        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = inputs
        for conv, bn, se in zip(self.convs, self.bns, self.se_layers):
            x = conv(x)
            x = bn(x, training=training)
            x = tf.nn.relu(x)
            s = se["gap"](x)
            s = se["d1"](s)
            s = se["d2"](s)
            s = se["reshape"](s)
            x = se["mul"]([x, s])
        pool_avg = self.gap(x)
        pool_max = self.gmp(x)
        x = tf.concat([pool_avg, pool_max], axis=-1)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        rul = self.output_layer(x)
        return rul, self._dummy_koopman(inputs)


class TransformerBaseline(keras.Model, BaselineModelMixin):
    """Transformer encoder baseline with multi-head self-attention."""

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)
        seq_len, n_features = input_shape_tuple
        self._latent_dim = 64

        self.input_proj = keras.layers.Dense(64, activation="relu",
                                             kernel_initializer="he_normal",
                                             name="input_proj")
        self.mha1 = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16,
                                                     dropout=0.1, name="mha1")
        self.ln1 = keras.layers.LayerNormalization(name="ln1")
        self.mha2 = keras.layers.MultiHeadAttention(num_heads=4, key_dim=16,
                                                     dropout=0.1, name="mha2")
        self.ln2 = keras.layers.LayerNormalization(name="ln2")
        self.ff1 = keras.layers.Dense(128, activation="gelu", name="ff1")
        self.ff2 = keras.layers.Dense(64, name="ff2")
        self.ln3 = keras.layers.LayerNormalization(name="ln3")
        self.gap = keras.layers.GlobalAveragePooling1D()
        self.dense1 = keras.layers.Dense(64, activation="relu",
                                         kernel_initializer="he_normal")
        self.dropout = keras.layers.Dropout(0.3)
        self.output_layer = keras.layers.Dense(1, activation="relu", dtype="float32")
        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = self.input_proj(inputs)
        attn = self.mha1(x, x, training=training)
        x = self.ln1(x + attn, training=training)
        attn = self.mha2(x, x, training=training)
        x = self.ln2(x + attn, training=training)
        ff = self.ff1(x)
        ff = self.ff2(ff)
        x = self.ln3(x + ff, training=training)
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        rul = self.output_layer(x)
        return rul, self._dummy_koopman(inputs)


# -------------------------------------------------------------------------
# Registry & factory
# -------------------------------------------------------------------------

BASELINE_REGISTRY = {
    "mlp": {"class": MLPBaseline, "desc": "Multi-Layer Perceptron"},
    "lstm": {"class": LSTMBaseline, "desc": "Stacked LSTM"},
    "bilstm": {"class": BiLSTMBaseline, "desc": "Bidirectional LSTM"},
    "cnn_lstm": {"class": CNNLSTMBaseline, "desc": "CNN-LSTM Hybrid"},
    "vanilla_fcn": {"class": VanillaFCN, "desc": "Vanilla FCN"},
    "pi_dp_fcn": {"class": PIDPFCNBaseline, "desc": "PI-DP-FCN (Original)"},
    "transformer": {"class": TransformerBaseline, "desc": "Transformer Encoder"},
}


def build_baseline_model(model_key, seq_len, n_features, n_train=None):
    """Build a baseline model by key name."""
    if model_key not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {model_key}. "
                         f"Choose from: {list(BASELINE_REGISTRY.keys())}")
    cls = BASELINE_REGISTRY[model_key]["class"]
    return cls(input_shape_tuple=(seq_len, n_features), n_train=n_train)
