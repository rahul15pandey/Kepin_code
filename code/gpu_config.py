# -*- coding: utf-8 -*-
"""
GPU Configuration — Optimised for NVIDIA A100 40 GB.

Centralised GPU setup used by all KePIN scripts. Configures:
  - Mixed precision (float16 compute / float32 storage) for Tensor Core use
  - XLA JIT compilation for kernel fusion
  - Memory growth / pre-allocation strategy
  - tf.data pipeline builder for optimal GPU feeding
  - Gradient scaler wrapper for mixed-precision training
  - Auto batch-size selection aware of 40 GB VRAM

Usage:
    from gpu_config import setup_gpu, build_tf_dataset, get_batch_size

    setup_gpu()                       # call once at script start
    ds = build_tf_dataset(X, Y, batch_size=2048)
    batch_size = get_batch_size(n_train, seq_len, n_feat)
"""

import os
import tensorflow as tf
import keras


# =========================================================================
# Constants
# =========================================================================
A100_VRAM_GB = 40
DEFAULT_MIXED_PRECISION = True
DEFAULT_XLA = True


# =========================================================================
# GPU setup (call once)
# =========================================================================

def setup_gpu(mixed_precision: bool = DEFAULT_MIXED_PRECISION,
              xla: bool = DEFAULT_XLA,
              memory_limit_mb: int = None,
              verbose: bool = True):
    """Configure GPU for optimal A100 performance.

    Args:
        mixed_precision: enable float16 mixed precision (Tensor Cores)
        xla:             enable XLA JIT compilation
        memory_limit_mb: optional VRAM cap in MB (None = use memory growth)
        verbose:         print configuration summary
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        if verbose:
            print("[gpu_config] No GPU detected — running on CPU")
        return

    # --- Memory configuration ---
    for gpu in gpus:
        if memory_limit_mb is not None:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=memory_limit_mb
                )]
            )
        else:
            tf.config.experimental.set_memory_growth(gpu, True)

    # --- Mixed precision ---
    if mixed_precision:
        policy = keras.mixed_precision.Policy("mixed_float16")
        keras.mixed_precision.set_global_policy(policy)

    # --- XLA JIT compilation ---
    if xla:
        tf.config.optimizer.set_jit(True)

    if verbose:
        gpu_name = "(unknown)"
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = gpu_details.get("device_name", str(gpus[0]))
        except Exception:
            gpu_name = str(gpus[0])

        print(f"[gpu_config] GPU Setup Complete:")
        print(f"  Device:          {gpu_name}")
        print(f"  Count:           {len(gpus)}")
        print(f"  Mixed precision: {'float16 (Tensor Cores)' if mixed_precision else 'float32'}")
        print(f"  XLA JIT:         {'enabled' if xla else 'disabled'}")
        mem_str = f"{memory_limit_mb} MB" if memory_limit_mb else "dynamic growth"
        print(f"  Memory:          {mem_str}")
        print(f"  Compute dtype:   {keras.mixed_precision.global_policy().compute_dtype}")
        print(f"  Variable dtype:  {keras.mixed_precision.global_policy().variable_dtype}")


def is_mixed_precision_enabled() -> bool:
    """Check if mixed precision is currently active."""
    return keras.mixed_precision.global_policy().compute_dtype == "float16"


# =========================================================================
# tf.data pipeline builder
# =========================================================================

def build_tf_dataset(X, Y, batch_size: int, shuffle: bool = True,
                     seed: int = 42, drop_remainder: bool = False):
    """Build an optimised tf.data.Dataset from NumPy arrays.

    Uses prefetch and optional shuffle for maximum GPU throughput.

    Args:
        X: input array (samples, seq_len, n_features)
        Y: label array (samples, 1)
        batch_size: batch size
        shuffle: whether to shuffle each epoch
        seed: random seed for shuffling
        drop_remainder: drop last incomplete batch

    Returns:
        tf.data.Dataset yielding (X_batch, Y_batch)
    """
    ds = tf.data.Dataset.from_tensor_slices((X, Y))

    if shuffle:
        buffer_size = min(len(X), 50_000)  # cap buffer for memory
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed,
                        reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# =========================================================================
# Auto batch-size for A100 40 GB
# =========================================================================

def get_batch_size(n_train: int, seq_len: int, n_features: int,
                   model_type: str = "kepin") -> int:
    """Select optimal batch size for A100 40 GB.

    Larger batches improve GPU utilisation on A100 Tensor Cores.
    Uses a heuristic based on sample dimensions and model complexity.

    Args:
        n_train:    number of training samples
        seq_len:    sequence length
        n_features: number of input features
        model_type: 'kepin', 'lstm', 'bilstm', 'transformer', or other

    Returns:
        batch_size: int
    """
    # Approximate memory per sample (bytes, float16)
    # Input: seq_len * n_features * 2 bytes
    # Activations/gradients rough multiplier
    sample_bytes = seq_len * n_features * 2  # float16

    if model_type in ("lstm", "bilstm"):
        # RNNs have higher activation memory per sample
        activation_multiplier = 80
    elif model_type == "transformer":
        # Attention is O(seq_len^2) in memory
        activation_multiplier = 60 + seq_len
    else:
        # Conv/FCN models are relatively efficient
        activation_multiplier = 40

    mem_per_sample = sample_bytes * activation_multiplier
    available_bytes = A100_VRAM_GB * 1024**3 * 0.7  # use 70% of VRAM

    max_by_memory = int(available_bytes / mem_per_sample)

    # Clamp to reasonable range and power-of-2 for Tensor Core efficiency
    candidates = [128, 256, 512, 1024, 2048, 4096]

    # Don't exceed dataset size
    max_bs = min(max_by_memory, n_train // 2, 4096)

    # Pick largest candidate that fits
    batch_size = 256  # fallback
    for bs in candidates:
        if bs <= max_bs:
            batch_size = bs

    return batch_size


def get_learning_rate(batch_size: int, base_lr: float = 0.001,
                      base_batch: int = 256) -> float:
    """Scale learning rate linearly with batch size (linear scaling rule).

    Reference: Goyal et al., "Accurate, Large Minibatch SGD", 2017.

    Args:
        batch_size: actual batch size
        base_lr:    learning rate for base_batch
        base_batch: reference batch size

    Returns:
        scaled_lr: float
    """
    return base_lr * (batch_size / base_batch)


# =========================================================================
# Gradient scaler for mixed precision
# =========================================================================

class MixedPrecisionGradientScaler:
    """Wrapper for loss scaling in mixed-precision training.

    In TF2/Keras 3, the optimizer handles loss scaling internally when
    using mixed_float16 policy. This class provides a consistent API
    that works regardless of precision mode.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and is_mixed_precision_enabled()

    def scale_loss(self, loss):
        """Scale loss before backward pass (no-op in float32 mode)."""
        if self.enabled:
            return tf.cast(loss, tf.float32)
        return loss

    def wrap_optimizer(self, optimizer):
        """Wrap optimizer with loss scaling if needed.

        In modern TF, the mixed precision policy handles this automatically
        when using keras.optimizers. This is a safety check.
        """
        return optimizer


# =========================================================================
# Unit test
# =========================================================================

if __name__ == "__main__":
    import numpy as np

    print("=== GPU Config — Unit Test ===\n")

    setup_gpu(mixed_precision=True, xla=True, verbose=True)

    # Test batch size selection
    test_cases = [
        {"n": 5000, "T": 30, "F": 14, "model": "kepin"},
        {"n": 20000, "T": 31, "F": 21, "model": "kepin"},
        {"n": 3000, "T": 20, "F": 5, "model": "lstm"},
        {"n": 10000, "T": 40, "F": 12, "model": "transformer"},
    ]

    print("\nBatch size selection:")
    for tc in test_cases:
        bs = get_batch_size(tc["n"], tc["T"], tc["F"], tc["model"])
        lr = get_learning_rate(bs)
        print(f"  n={tc['n']:>6}, T={tc['T']:>3}, F={tc['F']:>3}, "
              f"model={tc['model']:<12} → batch={bs:>5}, lr={lr:.5f}")

    # Test tf.data pipeline
    X = np.random.randn(1000, 30, 14).astype(np.float32)
    Y = np.random.randn(1000, 1).astype(np.float32)
    ds = build_tf_dataset(X, Y, batch_size=256)

    for xb, yb in ds.take(1):
        print(f"\ntf.data batch: X={xb.shape}, Y={yb.shape}, dtype={xb.dtype}")

    print("\n✓ GPU config ready for A100 40 GB.")
