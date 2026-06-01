# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for KePIN.

Provides unified access to:
  - C-MAPSS turbofan degradation (CSV with unit_id / cycle structure)
  - Generic time-series datasets (weather, fluid dynamics, energy, finance)
  - Synthetic data generators (degradation, ODE, weather, finance)
  - NASA Bearing, PHM 2012, Battery adapters

All loaders produce 4-D arrays: (samples, seq_len, 1, n_features) for
backward compatibility, converted to 3-D by ``convert_4d_to_3d`` before
entering the KePIN encoder.
"""

import os
import sys
from typing import Dict

# Add legacy code directory to path for backward compatibility
_code_dir = os.path.join(os.path.dirname(__file__), "..", "..", "code")
if os.path.isdir(_code_dir) and _code_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_code_dir))

# Re-export the full GenericTimeSeriesDataset class and factory function.
# The original module is ~1500 lines supporting 13 dataset types; we
# re-export rather than duplicate to keep a single source of truth while
# the refactored package is stabilised.
from GenericTimeSeriesDataset import (       # noqa: F401
    GenericTimeSeriesDataset,
    load_dataset_from_config as _legacy_load_dataset_from_config,
    _sliding_window as sliding_window,
)

# Re-export the legacy C-MAPSS loader for direct FD-number access.
from CMAPSSDataset import CMAPSSDataset      # noqa: F401


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(path: str | None, data_root: str | None) -> str | None:
    if not path:
        return path
    if os.path.isabs(path):
        return path

    project_root = _project_root()
    code_root = os.path.join(project_root, "code")

    candidates = []
    if data_root:
        candidates.append(os.path.join(data_root, path))
    candidates.append(os.path.join(project_root, path))
    candidates.append(os.path.join(code_root, path))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return os.path.join(data_root, path) if data_root else path


def resolve_config_paths(config: Dict, data_root: str | None = None) -> Dict:
    """Resolve relative dataset paths using data_root or repo defaults."""
    resolved = dict(config)
    env_root = os.environ.get("KEPIN_DATA_ROOT") or os.environ.get("KPDD_DATA_ROOT")
    root = data_root or env_root
    if root:
        root = os.path.abspath(os.path.expanduser(root))

    path_keys = (
        "train_path",
        "test_path",
        "test_rul_path",
        "csv_path",
        "data_dir",
        "train_X_path",
        "train_Y_path",
        "test_X_path",
        "test_Y_path",
    )

    for key in path_keys:
        if key in resolved:
            resolved[key] = _resolve_path(resolved.get(key), root)

    return resolved


def load_dataset_from_config(config: Dict, data_root: str | None = None):
    """Load dataset using a config dict with optional data_root overrides."""
    resolved = resolve_config_paths(config, data_root=data_root)
    return _legacy_load_dataset_from_config(resolved)

__all__ = [
    "GenericTimeSeriesDataset",
    "CMAPSSDataset",
    "load_dataset_from_config",
    "sliding_window",
    "resolve_config_paths",
]
