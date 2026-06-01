# -*- coding: utf-8 -*-
"""
KePIN — Koopman-Enhanced Physics-Informed Network.

A domain-independent deep learning framework for time-series prediction
that embeds a learnable SVD-parameterised Koopman operator within a
multi-scale encoder, enabling physically interpretable latent dynamics.
"""

__version__ = "1.0.0"
__author__ = "Rahul"

from kepin.models.kepin_model import KePINModel, build_kepin_model, auto_configure
from kepin.models.koopman import KoopmanOperator, extract_spectral_features
from kepin.losses.composite import make_kepin_loss, KePINLossWeights
from kepin.training.trainer import KePINTrainer
from kepin.data.loader import load_dataset_from_config

__all__ = [
    "KePINModel",
    "build_kepin_model",
    "auto_configure",
    "KoopmanOperator",
    "extract_spectral_features",
    "make_kepin_loss",
    "KePINLossWeights",
    "KePINTrainer",
    "load_dataset_from_config",
]
