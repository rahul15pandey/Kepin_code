# -*- coding: utf-8 -*-
"""
Reproducibility utilities for KePIN.
"""

from __future__ import annotations

import os
import random
import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set RNG seeds across Python, NumPy, TensorFlow, and PyTorch (if present).

    Args:
        seed: Random seed value.
        deterministic: Enable deterministic kernels where supported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # Required for deterministic cuBLAS in PyTorch when CUDA is used.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        pass
