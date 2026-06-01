# -*- coding: utf-8 -*-
"""
GPU configuration — optimised for NVIDIA A100 40 GB.

Provides ``setup_gpu()``, ``build_tf_dataset()``, and auto batch-size
selection for efficient Tensor Core utilisation.
"""

import os
import tensorflow as tf
import keras

A100_VRAM_GB = 40


def setup_gpu(mixed_precision: bool = True, xla: bool = True,
              memory_limit_mb: int = None, verbose: bool = True):
    """Configure GPU for optimal A100 performance."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if verbose:
            print("[gpu_config] No GPU detected — running on CPU")
        return

    for gpu in gpus:
        if memory_limit_mb is not None:
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(
                    memory_limit=memory_limit_mb)])
        else:
            tf.config.experimental.set_memory_growth(gpu, True)

    if mixed_precision:
        keras.mixed_precision.set_global_policy(
            keras.mixed_precision.Policy("mixed_float16"))

    if xla:
        tf.config.optimizer.set_jit(True)

    if verbose:
        gpu_name = "(unknown)"
        try:
            gpu_name = tf.config.experimental.get_device_details(
                gpus[0]).get("device_name", str(gpus[0]))
        except Exception:
            gpu_name = str(gpus[0])
        print(f"[gpu_config] GPU: {gpu_name} | Count: {len(gpus)} | "
              f"Mixed: {'fp16' if mixed_precision else 'fp32'} | XLA: {xla}")


def is_mixed_precision_enabled() -> bool:
    return keras.mixed_precision.global_policy().compute_dtype == "float16"


def build_tf_dataset(X, Y, batch_size: int, shuffle: bool = True,
                     seed: int = 42, drop_remainder: bool = False):
    """Build an optimised ``tf.data.Dataset`` from NumPy arrays."""
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(min(len(X), 50_000), seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_batch_size(n_train: int, seq_len: int, n_features: int,
                   model_type: str = "kepin") -> int:
    """Select optimal batch size for A100 40 GB."""
    sample_bytes = seq_len * n_features * 2
    mult = {"lstm": 80, "bilstm": 80, "transformer": 60 + seq_len}.get(
        model_type, 40)
    mem_per_sample = sample_bytes * mult
    available = A100_VRAM_GB * 1024**3 * 0.7
    max_by_memory = int(available / mem_per_sample)
    max_bs = min(max_by_memory, n_train // 2, 4096)

    batch_size = 256
    for bs in [128, 256, 512, 1024, 2048, 4096]:
        if bs <= max_bs:
            batch_size = bs
    return batch_size


def get_learning_rate(batch_size: int, base_lr: float = 0.001,
                      base_batch: int = 256) -> float:
    """Linear learning rate scaling rule."""
    return base_lr * (batch_size / base_batch)
