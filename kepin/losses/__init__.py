# -*- coding: utf-8 -*-
"""KePIN loss functions."""

from kepin.losses.composite import (
    make_kepin_loss,
    KePINLossWeights,
    rul_mse_loss,
    koopman_one_step_loss,
    spectral_stability_loss,
    monotonicity_loss,
    multi_step_loss,
    asymmetric_loss,
    slope_matching_loss,
)
