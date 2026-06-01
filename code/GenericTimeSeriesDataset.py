"""
Generic Time-Series Dataset Loader for Physics-Informed RUL / Degradation Prediction.

Supports:
  1. CSV files with configurable column names (unit_id, cycle, target, features)
  2. Direct NumPy arrays (X_train, Y_train, X_test, Y_test)
  3. Built-in adapters for popular prognostics benchmarks:
     - NASA Bearing Dataset (IMS / FEMTO)
     - PHM 2012 Prognostics Challenge (PRONOSTIA bearings)
     - Battery (capacity fade as RUL proxy)
     - Turbofan Degradation Simulation (any CMAPSS-like format)
  4. Synthetic degradation data for quick sanity checks

The loader mirrors the CMAPSSDataset API (windowed sequences + labels) so it
plugs directly into the PI-DP-FCN training pipeline.
"""

import os
import glob
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================================================================
# Helper: sliding-window segmentation
# =========================================================================

def _sliding_window(data: np.ndarray, labels: np.ndarray,
                    seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping windows of length `seq_len`.

    Args:
        data:   (total_timesteps, n_features)
        labels: (total_timesteps,)  – target at each timestep
        seq_len: window length

    Returns:
        X: (n_windows, seq_len, n_features)
        Y: (n_windows, 1)  – label at the last timestep of each window
    """
    n = data.shape[0]
    if n < seq_len:
        raise ValueError(f"Series length {n} < seq_len {seq_len}")
    X, Y = [], []
    for i in range(n - seq_len + 1):
        X.append(data[i:i + seq_len])
        Y.append(labels[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, 1)


# =========================================================================
# Main class
# =========================================================================

class GenericTimeSeriesDataset:
    """Flexible loader that produces windowed (samples, timesteps, 1, features)
    arrays matching the PI-DP-FCN input convention."""

    def __init__(
        self,
        name: str = "custom",
        sequence_length: int = 30,
        rul_cap: float = 125.0,
        test_last_only: bool = True,
    ):
        self.name = name
        self.sequence_length = sequence_length
        self.rul_cap = rul_cap
        self.test_last_only = test_last_only

        self.scaler = StandardScaler()

        # Populated by load_* methods
        self.X_train: Optional[np.ndarray] = None
        self.Y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.Y_test: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    # -----------------------------------------------------------------
    # Public API: get ready-to-use 4-D arrays
    # -----------------------------------------------------------------

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, Y_train, X_test, Y_test) in PI-DP-FCN shape:
        X: (samples, seq_len, 1, n_features)
        Y: (samples, 1)
        """
        assert self.X_train is not None, "Data not loaded. Call a load_* method first."
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    # -----------------------------------------------------------------
    # 1. From CSV with unit_id column
    # -----------------------------------------------------------------

    def load_from_csv(
        self,
        train_path: str,
        test_path: Optional[str] = None,
        test_rul_path: Optional[str] = None,
        unit_col: str = "unit_id",
        cycle_col: str = "cycle",
        target_col: str = "RUL",
        feature_cols: Optional[List[str]] = None,
        delimiter: str = ",",
        train_ratio: float = 0.8,
        column_names: Optional[List[str]] = None,
    ):
        """Load from CSV files.

        If `test_path` is None, the training CSV is split into train/test
        by unit_id using `train_ratio`.

        If the CSV has no pre-computed RUL column, provide `target_col=None`
        and the loader will compute piece-wise linear RUL from max cycle.

        If `test_rul_path` is given (like C-MAPSS RUL_FDxxx.txt), it is used
        to compute the true RUL for test units.

        If `column_names` is provided, the file is read with header=None and
        the given names are assigned (for headerless formats like C-MAPSS).
        """
        # Use regex whitespace delimiter for space-delimited files
        sep = r"\s+" if delimiter.strip() == "" or delimiter == " " else delimiter
        if column_names is not None:
            train_df = pd.read_csv(train_path, sep=sep, header=None,
                                    engine="python")
            # Handle extra trailing columns (C-MAPSS has trailing spaces)
            if train_df.shape[1] > len(column_names):
                train_df = train_df.iloc[:, :len(column_names)]
            train_df.columns = column_names
        else:
            train_df = pd.read_csv(train_path, sep=sep, engine="python")

        # Auto-detect feature columns
        exclude = {unit_col, cycle_col, target_col} if target_col else {unit_col, cycle_col}
        if feature_cols is None:
            feature_cols = [c for c in train_df.columns if c not in exclude]
        self.feature_names = list(feature_cols)

        # Compute RUL if not present
        if target_col is None or target_col not in train_df.columns:
            target_col = "RUL"
            max_cycles = train_df.groupby(unit_col)[cycle_col].transform("max")
            train_df[target_col] = max_cycles - train_df[cycle_col]

        # Split into train / test if no separate test file
        if test_path is None:
            unit_ids = train_df[unit_col].unique()
            np.random.seed(42)
            np.random.shuffle(unit_ids)
            split = int(len(unit_ids) * train_ratio)
            train_ids, test_ids = unit_ids[:split], unit_ids[split:]
            test_df = train_df[train_df[unit_col].isin(test_ids)].copy()
            train_df = train_df[train_df[unit_col].isin(train_ids)].copy()
        else:
            if column_names is not None:
                test_df = pd.read_csv(test_path, sep=sep, header=None,
                                       engine="python")
                if test_df.shape[1] > len(column_names):
                    test_df = test_df.iloc[:, :len(column_names)]
                test_df.columns = column_names
            else:
                test_df = pd.read_csv(test_path, sep=sep, engine="python")
            if target_col not in test_df.columns:
                if test_rul_path is not None:
                    truth = pd.read_csv(test_rul_path, delimiter=r"\s+", header=None)
                    truth.columns = ["truth"]
                    truth[unit_col] = truth.index + 1
                    test_rul_info = test_df.groupby(unit_col)[cycle_col].max().reset_index()
                    test_rul_info.columns = [unit_col, "elapsed"]
                    test_rul_info = test_rul_info.merge(truth, on=unit_col, how="left")
                    test_rul_info["max"] = test_rul_info["elapsed"] + test_rul_info["truth"]
                    test_df = test_df.merge(test_rul_info[[unit_col, "max"]], on=unit_col, how="left")
                    test_df[target_col] = test_df["max"] - test_df[cycle_col]
                    test_df.drop("max", axis=1, inplace=True)
                else:
                    max_cycles = test_df.groupby(unit_col)[cycle_col].transform("max")
                    test_df[target_col] = max_cycles - test_df[cycle_col]

        # Normalize features (fit on train)
        self.scaler.fit(train_df[feature_cols])
        train_df[feature_cols] = self.scaler.transform(train_df[feature_cols])
        test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])

        # Cap RUL
        if self.rul_cap is not None:
            train_df.loc[train_df[target_col] > self.rul_cap, target_col] = self.rul_cap
            test_df.loc[test_df[target_col] > self.rul_cap, target_col] = self.rul_cap

        # Window per unit
        self._window_per_unit(train_df, test_df, unit_col, feature_cols, target_col)

    # -----------------------------------------------------------------
    # 2. From NumPy arrays (already windowed or not)
    # -----------------------------------------------------------------

    def load_from_numpy(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        already_windowed: bool = False,
        feature_names: Optional[List[str]] = None,
    ):
        """Load directly from NumPy arrays.

        If `already_windowed=True`, expects X shape (samples, seq_len, features).
        Otherwise expects X shape (total_timesteps, features) and will apply
        sliding-window segmentation.
        """
        if feature_names:
            self.feature_names = feature_names

        if already_windowed:
            # Reshape to 4-D: (samples, seq_len, 1, features)
            if X_train.ndim == 3:
                X_train = X_train[:, :, np.newaxis, :]
            if X_test.ndim == 3:
                X_test = X_test[:, :, np.newaxis, :]
            self.X_train = X_train.astype(np.float32)
            self.Y_train = Y_train.astype(np.float32).reshape(-1, 1)
            self.X_test = X_test.astype(np.float32)
            self.Y_test = Y_test.astype(np.float32).reshape(-1, 1)
        else:
            # Apply sliding window
            Xtr, Ytr = _sliding_window(X_train, Y_train.flatten(), self.sequence_length)
            Xte, Yte = _sliding_window(X_test, Y_test.flatten(), self.sequence_length)
            self.X_train = Xtr[:, :, np.newaxis, :]
            self.Y_train = Ytr
            self.X_test = Xte[:, :, np.newaxis, :]
            self.Y_test = Yte

        # Normalise
        n_feat = self.X_train.shape[-1]
        flat_train = self.X_train.reshape(-1, n_feat)
        self.scaler.fit(flat_train)
        self.X_train = self.scaler.transform(
            self.X_train.reshape(-1, n_feat)
        ).reshape(self.X_train.shape).astype(np.float32)
        self.X_test = self.scaler.transform(
            self.X_test.reshape(-1, n_feat)
        ).reshape(self.X_test.shape).astype(np.float32)

        # Cap RUL
        if self.rul_cap is not None:
            self.Y_train[self.Y_train > self.rul_cap] = self.rul_cap
            self.Y_test[self.Y_test > self.rul_cap] = self.rul_cap

    # -----------------------------------------------------------------
    # 3. Synthetic degradation data (for quick experiments)
    # -----------------------------------------------------------------

    def load_synthetic(
        self,
        n_units_train: int = 80,
        n_units_test: int = 20,
        max_life: int = 200,
        n_features: int = 14,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        """Generate synthetic multi-sensor degradation data.

        Each unit has a random lifetime ∈ [max_life//2, max_life].
        Features follow exponential degradation curves with Gaussian noise.
        """
        rng = np.random.RandomState(seed)
        self.feature_names = [f"sensor_{i+1}" for i in range(n_features)]

        def _gen_units(n_units):
            all_X, all_Y = [], []
            for _ in range(n_units):
                life = rng.randint(max_life // 2, max_life + 1)
                t = np.linspace(0, 1, life)
                features = np.zeros((life, n_features))
                for f in range(n_features):
                    rate = rng.uniform(1.0, 5.0)
                    amp = rng.uniform(0.5, 2.0)
                    features[:, f] = amp * (np.exp(rate * t) - 1) + rng.randn(life) * noise_std
                rul = np.arange(life - 1, -1, -1, dtype=np.float32)
                all_X.append(features)
                all_Y.append(rul)
            return all_X, all_Y

        train_X_list, train_Y_list = _gen_units(n_units_train)
        test_X_list, test_Y_list = _gen_units(n_units_test)

        # Create windowed samples
        X_tr, Y_tr = [], []
        for feats, rul in zip(train_X_list, train_Y_list):
            if len(rul) >= self.sequence_length:
                xw, yw = _sliding_window(feats, rul, self.sequence_length)
                X_tr.append(xw)
                Y_tr.append(yw)

        X_te, Y_te = [], []
        for feats, rul in zip(test_X_list, test_Y_list):
            if len(rul) >= self.sequence_length:
                if self.test_last_only:
                    # Only the last window per unit
                    xw, yw = _sliding_window(feats, rul, self.sequence_length)
                    X_te.append(xw[-1:])
                    Y_te.append(yw[-1:])
                else:
                    xw, yw = _sliding_window(feats, rul, self.sequence_length)
                    X_te.append(xw)
                    Y_te.append(yw)

        X_train = np.concatenate(X_tr, axis=0)
        Y_train = np.concatenate(Y_tr, axis=0)
        X_test = np.concatenate(X_te, axis=0)
        Y_test = np.concatenate(Y_te, axis=0)

        # Normalise
        n_feat = X_train.shape[-1]
        flat = X_train.reshape(-1, n_feat)
        self.scaler.fit(flat)
        X_train = self.scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

        # Cap RUL
        if self.rul_cap:
            Y_train[Y_train > self.rul_cap] = self.rul_cap
            Y_test[Y_test > self.rul_cap] = self.rul_cap

        # 4-D
        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    # -----------------------------------------------------------------
    # 4. NASA Bearing (IMS) adapter
    # -----------------------------------------------------------------

    def load_nasa_bearing(self, data_dir: str, test_ratio: float = 0.2):
        """Load NASA IMS Bearing dataset.

        Expected structure:  data_dir/<set_name>/<channel_files>
        Each file has one vibration reading per row (sampled at 20 kHz, 1-sec records).
        We extract statistical features per record and compute RUL from remaining records.
        """
        self.feature_names = ["rms", "kurtosis", "crest_factor", "skewness",
                              "peak", "std", "peak_to_peak", "shape_factor"]

        def _extract_features(filepath: str) -> np.ndarray:
            """Extract 8 time-domain features from a raw vibration file."""
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            feats = []
            for col in range(data.shape[1]):
                sig = data[:, col]
                rms = np.sqrt(np.mean(sig ** 2))
                kurt = float(pd.Series(sig).kurtosis())
                peak = np.max(np.abs(sig))
                crest = peak / (rms + 1e-10)
                skew = float(pd.Series(sig).skew())
                std = np.std(sig)
                p2p = np.max(sig) - np.min(sig)
                shape = rms / (np.mean(np.abs(sig)) + 1e-10)
                feats.extend([rms, kurt, crest, skew, peak, std, p2p, shape])
            return np.array(feats, dtype=np.float32)

        # Discover sets
        sets = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])
        if not sets:
            raise FileNotFoundError(f"No sub-directories found in {data_dir}")

        all_features, all_rul = [], []
        for set_name in sets:
            set_dir = os.path.join(data_dir, set_name)
            files = sorted(glob.glob(os.path.join(set_dir, "*")))
            if not files:
                continue
            feat_seq = np.array([_extract_features(f) for f in files])
            n = len(feat_seq)
            rul = np.arange(n - 1, -1, -1, dtype=np.float32)
            all_features.append(feat_seq)
            all_rul.append(rul)

        # Split by bearing sets
        n_sets = len(all_features)
        n_test = max(1, int(n_sets * test_ratio))
        train_feats = all_features[:n_sets - n_test]
        train_ruls = all_rul[:n_sets - n_test]
        test_feats = all_features[n_sets - n_test:]
        test_ruls = all_rul[n_sets - n_test:]

        # Window & concatenate
        def _window_concat(feat_list, rul_list, last_only=False):
            Xs, Ys = [], []
            for f, r in zip(feat_list, rul_list):
                if len(r) < self.sequence_length:
                    continue
                xw, yw = _sliding_window(f, r, self.sequence_length)
                if last_only:
                    Xs.append(xw[-1:])
                    Ys.append(yw[-1:])
                else:
                    Xs.append(xw)
                    Ys.append(yw)
            return np.concatenate(Xs), np.concatenate(Ys)

        X_train, Y_train = _window_concat(train_feats, train_ruls)
        X_test, Y_test = _window_concat(test_feats, test_ruls, last_only=self.test_last_only)

        self.load_from_numpy(X_train, Y_train, X_test, Y_test,
                             already_windowed=True, feature_names=self.feature_names)

    # -----------------------------------------------------------------
    # 5. Battery capacity fade
    # -----------------------------------------------------------------

    def load_battery(
        self,
        csv_path: str,
        cycle_col: str = "cycle",
        capacity_col: str = "capacity",
        cell_col: str = "cell_id",
        eol_threshold: float = 0.7,
        feature_cols: Optional[List[str]] = None,
        train_ratio: float = 0.8,
    ):
        """Load battery degradation data.

        RUL is defined as remaining cycles until capacity drops below
        `eol_threshold` fraction of initial capacity.

        Args:
            csv_path: CSV with cycle-level battery data
            eol_threshold: fraction of initial capacity defining end-of-life
            feature_cols: columns to use as features (default: all except id/cycle/capacity)
        """
        df = pd.read_csv(csv_path)

        exclude = {cell_col, cycle_col, capacity_col}
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in exclude]
        # Always include capacity as a feature
        if capacity_col not in feature_cols:
            feature_cols = [capacity_col] + feature_cols
        self.feature_names = list(feature_cols)

        # Compute RUL per cell
        cells = df[cell_col].unique()
        for cell in cells:
            mask = df[cell_col] == cell
            cell_data = df.loc[mask].sort_values(cycle_col)
            init_cap = cell_data[capacity_col].iloc[0]
            threshold = init_cap * eol_threshold

            # Find EOL cycle
            eol_mask = cell_data[capacity_col] < threshold
            if eol_mask.any():
                eol_cycle = cell_data.loc[eol_mask, cycle_col].iloc[0]
            else:
                eol_cycle = cell_data[cycle_col].max()

            df.loc[mask, "RUL"] = eol_cycle - df.loc[mask, cycle_col]
            df.loc[mask & (df["RUL"] < 0), "RUL"] = 0

        self.load_from_csv(
            train_path=csv_path,
            unit_col=cell_col,
            cycle_col=cycle_col,
            target_col="RUL",
            feature_cols=feature_cols,
            train_ratio=train_ratio,
        )

    # -----------------------------------------------------------------
    # 6. PHM 2012 (PRONOSTIA) Bearing adapter
    # -----------------------------------------------------------------

    def load_phm2012(
        self,
        data_dir: str,
        condition: str = "1",
        train_bearings: Optional[List[str]] = None,
        test_bearings: Optional[List[str]] = None,
    ):
        """Load PHM 2012 PRONOSTIA bearing dataset.

        Expected structure:
          data_dir/Learning_set/Bearing<cond>_<id>/acc_<timestamp>.csv
          data_dir/Test_set/Bearing<cond>_<id>/acc_<timestamp>.csv

        Each acc CSV has columns: hour, minute, second, microsecond, h_acc, v_acc.
        We extract statistical features from h_acc and v_acc per file (one snapshot).
        """
        self.feature_names = [
            "h_rms", "h_kurtosis", "h_crest", "h_skew", "h_peak", "h_std",
            "v_rms", "v_kurtosis", "v_crest", "v_skew", "v_peak", "v_std",
        ]

        def _extract_phm_features(filepath: str) -> np.ndarray:
            df = pd.read_csv(filepath, header=None)
            feats = []
            for col_idx in [4, 5]:  # h_acc, v_acc
                sig = df.iloc[:, col_idx].values.astype(np.float64)
                rms = np.sqrt(np.mean(sig ** 2))
                kurt = float(pd.Series(sig).kurtosis())
                peak = np.max(np.abs(sig))
                crest = peak / (rms + 1e-10)
                skew = float(pd.Series(sig).skew())
                std = np.std(sig)
                feats.extend([rms, kurt, crest, skew, peak, std])
            return np.array(feats, dtype=np.float32)

        def _load_bearing_set(base_dir: str, bearing_ids: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
            feat_list, rul_list = [], []
            for bid in bearing_ids:
                bearing_dir = os.path.join(base_dir, f"Bearing{condition}_{bid}")
                if not os.path.isdir(bearing_dir):
                    print(f"  Warning: {bearing_dir} not found, skipping.")
                    continue
                files = sorted(glob.glob(os.path.join(bearing_dir, "acc_*.csv")))
                if not files:
                    continue
                feats = np.array([_extract_phm_features(f) for f in files])
                rul = np.arange(len(feats) - 1, -1, -1, dtype=np.float32)
                feat_list.append(feats)
                rul_list.append(rul)
            return feat_list, rul_list

        # Default bearing splits
        if train_bearings is None:
            train_bearings = ["1", "2"]
        if test_bearings is None:
            test_bearings = ["3"]

        learn_dir = os.path.join(data_dir, "Learning_set")
        test_dir = os.path.join(data_dir, "Test_set") if os.path.isdir(
            os.path.join(data_dir, "Test_set")) else learn_dir

        train_feats, train_ruls = _load_bearing_set(learn_dir, train_bearings)
        test_feats, test_ruls = _load_bearing_set(test_dir, test_bearings)

        # Window & concatenate
        def _wc(flist, rlist, last_only=False):
            Xs, Ys = [], []
            for f, r in zip(flist, rlist):
                if len(r) < self.sequence_length:
                    continue
                xw, yw = _sliding_window(f, r, self.sequence_length)
                if last_only:
                    Xs.append(xw[-1:])
                    Ys.append(yw[-1:])
                else:
                    Xs.append(xw)
                    Ys.append(yw)
            if not Xs:
                raise ValueError("No windows could be created. Check data or reduce seq_len.")
            return np.concatenate(Xs), np.concatenate(Ys)

        X_train, Y_train = _wc(train_feats, train_ruls)
        X_test, Y_test = _wc(test_feats, test_ruls, last_only=self.test_last_only)

        self.load_from_numpy(X_train, Y_train, X_test, Y_test,
                             already_windowed=True, feature_names=self.feature_names)

    # -----------------------------------------------------------------
    # Internal: window per unit from DataFrames
    # -----------------------------------------------------------------

    def _window_per_unit(self, train_df, test_df, unit_col, feature_cols, target_col):
        """Create windowed arrays from train/test DataFrames with a unit column."""
        def _make_windows(df, last_only=False):
            Xs, Ys = [], []
            for uid in sorted(df[unit_col].unique()):
                unit_data = df[df[unit_col] == uid].sort_values(
                    by=[c for c in df.columns if "cycle" in c.lower()] or df.columns[:2]
                )
                feats = unit_data[feature_cols].values
                labels = unit_data[target_col].values
                if len(labels) < self.sequence_length:
                    continue
                xw, yw = _sliding_window(feats, labels, self.sequence_length)
                if last_only:
                    Xs.append(xw[-1:])
                    Ys.append(yw[-1:])
                else:
                    Xs.append(xw)
                    Ys.append(yw)
            if not Xs:
                raise ValueError("No windows created. Lower sequence_length or check data.")
            return np.concatenate(Xs), np.concatenate(Ys)

        X_train, Y_train = _make_windows(train_df, last_only=False)
        X_test, Y_test = _make_windows(test_df, last_only=self.test_last_only)

        # Reshape to 4-D: (samples, seq_len, 1, features)
        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    # -----------------------------------------------------------------
    # 7. Synthetic ODE system (for Koopman eigenvalue recovery validation)
    # -----------------------------------------------------------------

    def load_synthetic_ode(
        self,
        n_units_train: int = 80,
        n_units_test: int = 20,
        max_life: int = 300,
        dt: float = 0.1,
        noise_std: float = 0.05,
        failure_threshold: float = 2.0,
        seed: int = 42,
    ):
        """Generate data from a known linear ODE system for Koopman validation.

        The system evolves as:

            dx/dt = A · x,   A = [[-0.05,  0.1 ],
                                   [-0.1,  -0.03]]

        This produces exponential decay (degradation) coupled with oscillation
        (cyclic loading effects). The analytical eigenvalues of A are:

            λ_{1,2} = -0.04 ± 0.0917i

        So the discrete-time Koopman operator K = exp(A·dt) has eigenvalues:

            μ_{1,2} = exp(λ_{1,2} · dt)

        The dataset includes 2 state variables + 3 nonlinear observables
        (x₁², sin(x₂), x₁·x₂) to make Koopman lifting non-trivial.
        RUL = remaining time until ||x|| exceeds failure_threshold.

        This is the "smoking gun" for reviewers: recovering known eigenvalues
        proves the model discovers dynamics, not just fits curves.

        Args:
            n_units_train:      number of training trajectories
            n_units_test:       number of test trajectories
            max_life:           max trajectory length (timesteps)
            dt:                 time step for ODE integration
            noise_std:          Gaussian noise added to observations
            failure_threshold:  ||x|| value defining end-of-life
            seed:               random seed for reproducibility

        Attributes set after loading:
            self.ode_A:              the true system matrix A
            self.ode_true_eigenvalues: analytical eigenvalues of A
            self.ode_true_K_eigenvalues: analytical eigenvalues of exp(A·dt)
            self.ode_dt:             time step
        """
        from scipy.linalg import expm

        rng = np.random.RandomState(seed)

        # True system matrix
        A = np.array([[-0.05, 0.1],
                      [-0.1, -0.03]], dtype=np.float64)

        # Discrete-time transition matrix (ground truth Koopman for linear system)
        K_true = expm(A * dt)

        # Store ground truth for validation
        self.ode_A = A.copy()
        self.ode_true_eigenvalues = np.linalg.eigvals(A)
        self.ode_true_K_eigenvalues = np.linalg.eigvals(K_true)
        self.ode_dt = dt

        self.feature_names = ["x1", "x2", "x1_sq", "sin_x2", "x1_x2"]

        def _generate_trajectory(rng_local):
            """Generate one unit's trajectory from random initial condition."""
            # Random initial state (small perturbation from origin)
            x0 = rng_local.randn(2) * 0.5 + np.array([0.3, 0.2])

            states = [x0.copy()]
            for t in range(max_life - 1):
                x_next = K_true @ states[-1]
                states.append(x_next.copy())

                # Check failure condition
                if np.linalg.norm(x_next) < 0.01:
                    break  # system has decayed to near-zero

            states = np.array(states, dtype=np.float64)  # (T, 2)
            T_actual = len(states)

            # Add nonlinear observables
            x1 = states[:, 0]
            x2 = states[:, 1]
            features = np.column_stack([
                x1,                              # state 1
                x2,                              # state 2
                x1 ** 2,                         # nonlinear: x1²
                np.sin(x2),                      # nonlinear: sin(x2)
                x1 * x2,                         # nonlinear: x1·x2
            ])

            # Add observation noise
            features += rng_local.randn(*features.shape) * noise_std

            # Compute RUL: remaining steps until norm < 0.01 or end
            # For a decaying system, RUL = (total_life - current_step)
            norms = np.linalg.norm(states, axis=1)
            rul = np.zeros(T_actual, dtype=np.float32)
            for t in range(T_actual):
                rul[t] = max(0, T_actual - 1 - t)

            return features.astype(np.float32), rul

        # Generate trajectories
        train_feats, train_ruls = [], []
        for _ in range(n_units_train):
            f, r = _generate_trajectory(rng)
            if len(r) >= self.sequence_length:
                train_feats.append(f)
                train_ruls.append(r)

        test_feats, test_ruls = [], []
        for _ in range(n_units_test):
            f, r = _generate_trajectory(rng)
            if len(r) >= self.sequence_length:
                test_feats.append(f)
                test_ruls.append(r)

        # Create sliding windows
        X_tr, Y_tr = [], []
        for feats, rul in zip(train_feats, train_ruls):
            xw, yw = _sliding_window(feats, rul, self.sequence_length)
            X_tr.append(xw)
            Y_tr.append(yw)

        X_te, Y_te = [], []
        for feats, rul in zip(test_feats, test_ruls):
            xw, yw = _sliding_window(feats, rul, self.sequence_length)
            if self.test_last_only:
                X_te.append(xw[-1:])
                Y_te.append(yw[-1:])
            else:
                X_te.append(xw)
                Y_te.append(yw)

        X_train = np.concatenate(X_tr, axis=0)
        Y_train = np.concatenate(Y_tr, axis=0)
        X_test = np.concatenate(X_te, axis=0)
        Y_test = np.concatenate(Y_te, axis=0)

        # Normalize
        n_feat = X_train.shape[-1]
        flat = X_train.reshape(-1, n_feat)
        self.scaler.fit(flat)
        X_train = self.scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
        X_test = self.scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

        # Cap RUL
        if self.rul_cap is not None:
            Y_train[Y_train > self.rul_cap] = self.rul_cap
            Y_test[Y_test > self.rul_cap] = self.rul_cap

        # 4-D output (matching existing API)
        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    # -----------------------------------------------------------------
    # 7b. Physics CSV (fluid dynamics, energy systems, etc.)
    # -----------------------------------------------------------------

    def load_physics_csv(
        self,
        csv_path: str,
        unit_col: str = "unit_id",
        cycle_col: str = "cycle",
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        train_ratio: float = 0.8,
    ):
        """Load a physics-based CSV dataset with unit_id + cycle structure.
        
        This is a generic loader for any time-series dataset where each row
        has a unit identifier, a time step (cycle), features, and a target.
        Suitable for fluid dynamics, energy systems, and similar datasets.
        """
        import os
        csv_path = os.path.join(os.path.dirname(__file__), csv_path) if not os.path.isabs(csv_path) else csv_path
        
        df = pd.read_csv(csv_path)
        
        # Determine feature columns
        if feature_cols is None:
            exclude = {unit_col, cycle_col, target_col}
            feature_cols = [c for c in df.columns if c not in exclude]
        
        self.feature_names = feature_cols
        
        # Get unique units
        units = sorted(df[unit_col].unique())
        n_units = len(units)
        n_train_units = int(n_units * train_ratio)
        
        train_units = units[:n_train_units]
        test_units = units[n_train_units:]
        
        # Split into train/test by unit
        train_df = df[df[unit_col].isin(train_units)]
        test_df = df[df[unit_col].isin(test_units)]
        
        # Fit scaler on train data
        self.scaler.fit(train_df[feature_cols].values)
        
        # Process each split
        def process_split(split_df, unit_ids):
            all_X, all_Y = [], []
            for uid in unit_ids:
                udf = split_df[split_df[unit_col] == uid].sort_values(cycle_col)
                feats = self.scaler.transform(udf[feature_cols].values)
                targets = udf[target_col].values
                
                if len(feats) < self.sequence_length:
                    continue
                
                X, Y = _sliding_window(feats, targets, self.sequence_length)
                all_X.append(X)
                all_Y.append(Y)
            
            if not all_X:
                raise ValueError("No valid sequences found")
            return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0)
        
        X_train, Y_train = process_split(train_df, train_units)
        X_test, Y_test = process_split(test_df, test_units)
        
        # Add channel dimension for compatibility: (N, T, F) → (N, T, 1, F)
        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    # -----------------------------------------------------------------
    # 8. Weather / Climate time-series
    # -----------------------------------------------------------------

    def load_weather(
        self,
        csv_path: str,
        datetime_col: str = "Date Time",
        target_col: str = "T (degC)",
        feature_cols: Optional[List[str]] = None,
        resample_minutes: int = 60,
        prediction_horizon: int = 24,
        train_ratio: float = 0.8,
        delimiter: str = ",",
    ):
        """Load weather / climate time-series data for dynamics discovery.

        Supports any tabular weather CSV (e.g. Jena Climate, NOAA hourly,
        OpenMeteo exports).  The framework treats weather as a *general
        temporal dynamics discovery* problem: the target is predicting
        a chosen variable `prediction_horizon` steps into the future.

        This is NOT RUL — it demonstrates that the Koopman operator
        discovers governing atmospheric dynamics (diurnal cycles, pressure
        systems) across a completely different physical domain.

        Args:
            csv_path:           path to weather CSV
            datetime_col:       column with timestamps (used for ordering)
            target_col:         variable to predict (e.g. temperature)
            feature_cols:       sensor columns to use (None = auto-detect)
            resample_minutes:   resample to this resolution (0 = no resampling)
            prediction_horizon: predict target this many steps ahead
            train_ratio:        temporal train/test split ratio
            delimiter:          CSV delimiter
        """
        df = pd.read_csv(csv_path, delimiter=delimiter)

        # Parse datetime if present (for ordering & resampling)
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
            df = df.dropna(subset=[datetime_col])
            df = df.sort_values(datetime_col).reset_index(drop=True)

            if resample_minutes > 0:
                df = df.set_index(datetime_col)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df = df[numeric_cols].resample(f"{resample_minutes}min").mean()
                df = df.dropna().reset_index()

        # Auto-detect numeric feature columns
        exclude = {datetime_col}
        numeric_df = df.select_dtypes(include=[np.number])
        if feature_cols is None:
            feature_cols = [c for c in numeric_df.columns if c not in exclude]
        if target_col not in feature_cols:
            feature_cols = [target_col] + [c for c in feature_cols if c != target_col]
        self.feature_names = list(feature_cols)

        # Drop rows with NaN in feature columns
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        # Create target: predict target_col `prediction_horizon` steps ahead
        target_values = df[target_col].values
        n = len(df)
        horizon_target = np.zeros(n, dtype=np.float32)
        horizon_target[:n - prediction_horizon] = target_values[prediction_horizon:]
        horizon_target[n - prediction_horizon:] = target_values[-1]

        features = df[feature_cols].values.astype(np.float32)

        # Temporal train/test split (no data leakage)
        split_idx = int(len(features) * train_ratio)
        train_feat = features[:split_idx]
        train_target = horizon_target[:split_idx]
        test_feat = features[split_idx:]
        test_target = horizon_target[split_idx:]

        # Normalize (fit on train only)
        self.scaler.fit(train_feat)
        train_feat = self.scaler.transform(train_feat)
        test_feat = self.scaler.transform(test_feat)

        # Sliding windows
        X_train, Y_train = _sliding_window(train_feat, train_target, self.sequence_length)
        X_test, Y_test = _sliding_window(test_feat, test_target, self.sequence_length)

        # Cap target if rul_cap is set
        if self.rul_cap is not None:
            Y_train[Y_train > self.rul_cap] = self.rul_cap
            Y_test[Y_test > self.rul_cap] = self.rul_cap

        # 4-D output
        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    def load_weather_synthetic(
        self,
        n_days: int = 365,
        dt_hours: float = 1.0,
        n_stations: int = 1,
        noise_std: float = 0.5,
        prediction_horizon: int = 24,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """Generate realistic synthetic weather data for testing.

        Simulates temperature, pressure, humidity, and wind with:
          - Diurnal cycle (24h period)
          - Seasonal trend (365-day period)
          - Random weather fronts (pressure perturbations)
          - Correlated multi-variate dynamics

        The Koopman operator should discover the diurnal and seasonal
        eigenvalues: λ_diurnal = exp(i·2π/24) and λ_seasonal = exp(i·2π/8760).

        Args:
            n_days:              simulation duration in days
            dt_hours:            time resolution in hours
            n_stations:          number of weather stations (treated as units)
            noise_std:           observation noise
            prediction_horizon:  steps ahead to predict temperature
            train_ratio:         temporal train/test split
            seed:                random seed
        """
        rng = np.random.RandomState(seed)
        n_steps = int(n_days * 24 / dt_hours)
        t = np.arange(n_steps) * dt_hours  # time in hours

        self.feature_names = ["temperature", "pressure", "humidity",
                              "wind_speed", "wind_dir_sin", "wind_dir_cos",
                              "dew_point", "solar_radiation"]

        all_features = []
        all_targets = []

        for station in range(max(1, n_stations)):
            # Base signals with known dynamics
            # Temperature: diurnal + seasonal + noise
            temp_base = (15.0
                         + 10.0 * np.sin(2 * np.pi * t / (365 * 24) - np.pi / 2)  # seasonal
                         + 5.0 * np.sin(2 * np.pi * t / 24 - np.pi / 3)            # diurnal
                         + rng.randn(n_steps) * noise_std * 2)

            # Pressure: slower dynamics + weather fronts
            pressure = (1013.0
                        + 5.0 * np.sin(2 * np.pi * t / (7 * 24))  # weekly cycle
                        + rng.randn(n_steps) * noise_std)

            # Humidity: anti-correlated with temperature
            humidity = np.clip(60 - 0.8 * (temp_base - 15) + rng.randn(n_steps) * noise_std * 3,
                               10, 100)

            # Wind: semi-random with diurnal modulation
            wind_speed = np.abs(3.0 + 2.0 * np.sin(2 * np.pi * t / 24)
                                + rng.randn(n_steps) * noise_std)
            wind_dir = (rng.randn(n_steps).cumsum() * 0.1) % (2 * np.pi)
            wind_sin = np.sin(wind_dir)
            wind_cos = np.cos(wind_dir)

            # Dew point: function of temp & humidity
            dew_point = temp_base - ((100 - humidity) / 5.0)

            # Solar radiation: positive during day, zero at night
            solar = np.maximum(0, 800 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 6)
                               + rng.randn(n_steps) * noise_std * 50)

            features = np.column_stack([
                temp_base, pressure, humidity, wind_speed,
                wind_sin, wind_cos, dew_point, solar,
            ]).astype(np.float32)

            # Target: temperature `prediction_horizon` steps ahead
            target = np.zeros(n_steps, dtype=np.float32)
            target[:n_steps - prediction_horizon] = temp_base[prediction_horizon:]
            target[n_steps - prediction_horizon:] = temp_base[-1]

            all_features.append(features)
            all_targets.append(target)

        features = np.concatenate(all_features, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Temporal split
        split_idx = int(len(features) * train_ratio)
        train_f, test_f = features[:split_idx], features[split_idx:]
        train_t, test_t = targets[:split_idx], targets[split_idx:]

        # Normalize
        self.scaler.fit(train_f)
        train_f = self.scaler.transform(train_f)
        test_f = self.scaler.transform(test_f)

        # Sliding windows
        X_train, Y_train = _sliding_window(train_f, train_t, self.sequence_length)
        X_test, Y_test = _sliding_window(test_f, test_t, self.sequence_length)

        if self.rul_cap is not None:
            Y_train[Y_train > self.rul_cap] = self.rul_cap
            Y_test[Y_test > self.rul_cap] = self.rul_cap

        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    # -----------------------------------------------------------------
    # 9. Financial time-series
    # -----------------------------------------------------------------

    def load_finance(
        self,
        csv_path: str,
        datetime_col: str = "Date",
        close_col: str = "Close",
        feature_cols: Optional[List[str]] = None,
        prediction_horizon: int = 5,
        drawdown_threshold: float = 0.05,
        target_mode: str = "drawdown",
        train_ratio: float = 0.8,
        delimiter: str = ",",
    ):
        """Load financial time-series for dynamics discovery.

        Supports stock/ETF/index CSVs (e.g. Yahoo Finance, Alpha Vantage).
        Two target modes:
          - 'drawdown':  time until next drawdown ≥ threshold (RUL analog)
          - 'return':    predict return over prediction_horizon steps

        The Koopman operator should discover market regime dynamics
        (mean-reversion timescales, volatility clustering periodicity).

        Args:
            csv_path:            path to financial CSV (OHLCV format)
            datetime_col:        date/timestamp column
            close_col:           closing price column
            feature_cols:        columns to use (None = auto-generate technical indicators)
            prediction_horizon:  steps ahead for return prediction
            drawdown_threshold:  fraction drop defining a "failure event" for drawdown mode
            target_mode:         'drawdown' (time-to-event) or 'return' (regression)
            train_ratio:         temporal train/test split
            delimiter:           CSV delimiter
        """
        df = pd.read_csv(csv_path, delimiter=delimiter)

        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
            df = df.dropna(subset=[datetime_col])
            df = df.sort_values(datetime_col).reset_index(drop=True)

        # Ensure close column exists
        if close_col not in df.columns:
            close_candidates = [c for c in df.columns if "close" in c.lower()]
            if close_candidates:
                close_col = close_candidates[0]
            else:
                raise ValueError(f"Close price column '{close_col}' not found. "
                                 f"Available: {list(df.columns)}")

        close = df[close_col].values.astype(np.float64)

        # Auto-generate technical indicator features if not provided
        if feature_cols is None:
            # Build features from price data
            feat_dict = {}

            # Log returns
            log_ret = np.diff(np.log(close + 1e-10), prepend=np.log(close[0] + 1e-10))
            feat_dict["log_return"] = log_ret

            # Rolling statistics
            for window in [5, 10, 20]:
                s = pd.Series(close)
                feat_dict[f"sma_{window}"] = (s.rolling(window, min_periods=1).mean().values
                                               / (close + 1e-10))
                feat_dict[f"volatility_{window}"] = (pd.Series(log_ret)
                                                      .rolling(window, min_periods=1)
                                                      .std().fillna(0).values)

            # RSI (14-period)
            delta = pd.Series(close).diff().fillna(0)
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss_val = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rs = gain / (loss_val + 1e-10)
            feat_dict["rsi_14"] = (100 - 100 / (1 + rs)).values

            # MACD
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            feat_dict["macd"] = (ema12 - ema26).values / (close + 1e-10)

            # Volume if available
            vol_candidates = [c for c in df.columns if "volume" in c.lower()]
            if vol_candidates:
                vol = df[vol_candidates[0]].values.astype(np.float64)
                feat_dict["volume_norm"] = vol / (vol.mean() + 1e-10)

            # Bollinger Band width
            sma20 = pd.Series(close).rolling(20, min_periods=1).mean()
            std20 = pd.Series(close).rolling(20, min_periods=1).std().fillna(1e-10)
            feat_dict["bb_width"] = (2 * std20 / (sma20 + 1e-10)).values

            # Price relative to high/low
            if "High" in df.columns and "Low" in df.columns:
                feat_dict["hl_ratio"] = ((df["High"] - df["Low"]) / (close + 1e-10)).values

            features = np.column_stack(list(feat_dict.values())).astype(np.float32)
            self.feature_names = list(feat_dict.keys())
        else:
            features = df[feature_cols].values.astype(np.float32)
            self.feature_names = list(feature_cols)

        # Replace inf/nan
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute target
        n = len(features)
        if target_mode == "drawdown":
            # Time-to-drawdown: how many steps until price drops by threshold
            running_max = np.maximum.accumulate(close)
            drawdown = (running_max - close) / (running_max + 1e-10)

            target = np.zeros(n, dtype=np.float32)
            for i in range(n):
                # Look forward to find next drawdown >= threshold
                future_dd = drawdown[i:]
                exceed = np.where(future_dd >= drawdown_threshold)[0]
                if len(exceed) > 0:
                    target[i] = float(exceed[0])
                else:
                    target[i] = float(n - i)  # remaining until end
        elif target_mode == "return":
            # Future return over prediction_horizon
            target = np.zeros(n, dtype=np.float32)
            for i in range(n - prediction_horizon):
                target[i] = (close[i + prediction_horizon] - close[i]) / (close[i] + 1e-10)
            # Last few steps: use last known return
            target[n - prediction_horizon:] = target[n - prediction_horizon - 1]
            # Scale to reasonable range
            target = target * 100  # convert to percentage
        else:
            raise ValueError(f"Unknown target_mode: {target_mode}. Use 'drawdown' or 'return'.")

        # Temporal split
        split_idx = int(n * train_ratio)
        train_f, test_f = features[:split_idx], features[split_idx:]
        train_t, test_t = target[:split_idx], target[split_idx:]

        # Normalize features
        self.scaler.fit(train_f)
        train_f = self.scaler.transform(train_f)
        test_f = self.scaler.transform(test_f)

        # Sliding windows
        X_train, Y_train = _sliding_window(train_f, train_t, self.sequence_length)
        X_test, Y_test = _sliding_window(test_f, test_t, self.sequence_length)

        if self.rul_cap is not None:
            Y_train[Y_train > self.rul_cap] = self.rul_cap
            Y_test[Y_test > self.rul_cap] = self.rul_cap

        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    def load_finance_synthetic(
        self,
        n_days: int = 2520,
        n_assets: int = 1,
        initial_price: float = 100.0,
        mu: float = 0.0005,
        sigma: float = 0.02,
        mean_reversion_speed: float = 0.01,
        regime_switch_prob: float = 0.005,
        prediction_horizon: int = 5,
        target_mode: str = "drawdown",
        drawdown_threshold: float = 0.05,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """Generate synthetic financial data with known dynamics.

        Simulates a mean-reverting geometric Brownian motion with
        regime switching, producing:
          - Bull/bear regimes with different drift (mu) and volatility (sigma)
          - Mean-reversion to a slowly varying fundamental value
          - Auto-generated technical indicators as features

        The Koopman operator should discover:
          - The mean-reversion timescale (eigenvalue with |λ| < 1)
          - Regime-specific dynamics

        Args:
            n_days:               number of trading days
            n_assets:             number of assets (treated as units)
            initial_price:        starting price
            mu:                   average daily drift in bull regime
            sigma:                daily volatility in bull regime
            mean_reversion_speed: pull-back speed toward fundamental
            regime_switch_prob:   probability of switching regime per day
            prediction_horizon:   steps for return prediction
            target_mode:          'drawdown' or 'return'
            drawdown_threshold:   drawdown fraction for failure event
            train_ratio:          train/test split
            seed:                 random seed
        """
        rng = np.random.RandomState(seed)

        self.feature_names = ["log_return", "sma_5_ratio", "sma_20_ratio",
                              "volatility_5", "volatility_20", "rsi_14",
                              "macd_norm", "bb_width", "volume_norm",
                              "momentum_10"]

        all_features = []
        all_targets = []

        for asset in range(max(1, n_assets)):
            # Generate price path
            price = np.zeros(n_days, dtype=np.float64)
            price[0] = initial_price * (1 + rng.randn() * 0.1)

            regime = 0  # 0 = bull, 1 = bear
            fundamental = price[0]

            for t in range(1, n_days):
                # Regime switching
                if rng.random() < regime_switch_prob:
                    regime = 1 - regime

                # Regime-dependent parameters
                if regime == 0:
                    drift = mu
                    vol = sigma
                else:
                    drift = -mu * 0.5
                    vol = sigma * 1.5

                # Mean-reverting GBM
                fundamental *= np.exp(mu * 0.5)
                mr_pull = mean_reversion_speed * (np.log(fundamental) - np.log(price[t - 1]))
                log_ret = drift + mr_pull + vol * rng.randn()
                price[t] = price[t - 1] * np.exp(log_ret)

            # Generate technical indicator features
            close_s = pd.Series(price)
            log_returns = np.diff(np.log(price + 1e-10), prepend=np.log(price[0] + 1e-10))

            sma5 = close_s.rolling(5, min_periods=1).mean().values / (price + 1e-10)
            sma20 = close_s.rolling(20, min_periods=1).mean().values / (price + 1e-10)
            vol5 = pd.Series(log_returns).rolling(5, min_periods=1).std().fillna(0).values
            vol20 = pd.Series(log_returns).rolling(20, min_periods=1).std().fillna(0).values

            # RSI
            delta = close_s.diff().fillna(0)
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss_r = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rsi = (100 - 100 / (1 + gain / (loss_r + 1e-10))).values

            # MACD
            ema12 = close_s.ewm(span=12, adjust=False).mean()
            ema26 = close_s.ewm(span=26, adjust=False).mean()
            macd = (ema12 - ema26).values / (price + 1e-10)

            # Bollinger Band width
            std20 = close_s.rolling(20, min_periods=1).std().fillna(1e-10)
            bb = (2 * std20 / (close_s.rolling(20, min_periods=1).mean() + 1e-10)).values

            # Synthetic volume (correlated with volatility)
            volume = np.abs(1.0 + vol5 * 10 + rng.randn(n_days) * 0.3)

            # Momentum
            momentum = close_s.pct_change(10).fillna(0).values

            features = np.column_stack([
                log_returns, sma5, sma20, vol5, vol20,
                rsi, macd, bb, volume, momentum,
            ]).astype(np.float32)

            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Target
            if target_mode == "drawdown":
                running_max = np.maximum.accumulate(price)
                dd = (running_max - price) / (running_max + 1e-10)
                target = np.zeros(n_days, dtype=np.float32)
                for i in range(n_days):
                    exceed = np.where(dd[i:] >= drawdown_threshold)[0]
                    if len(exceed) > 0:
                        target[i] = float(exceed[0])
                    else:
                        target[i] = float(n_days - i)
            else:  # return
                target = np.zeros(n_days, dtype=np.float32)
                for i in range(n_days - prediction_horizon):
                    target[i] = ((price[i + prediction_horizon] - price[i])
                                 / (price[i] + 1e-10)) * 100
                target[n_days - prediction_horizon:] = target[n_days - prediction_horizon - 1]

            all_features.append(features)
            all_targets.append(target)

        features = np.concatenate(all_features, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Temporal split
        split_idx = int(len(features) * train_ratio)
        train_f, test_f = features[:split_idx], features[split_idx:]
        train_t, test_t = targets[:split_idx], targets[split_idx:]

        # Normalize
        self.scaler.fit(train_f)
        train_f = self.scaler.transform(train_f)
        test_f = self.scaler.transform(test_f)

        # Sliding windows
        X_train, Y_train = _sliding_window(train_f, train_t, self.sequence_length)
        X_test, Y_test = _sliding_window(test_f, test_t, self.sequence_length)

        if self.rul_cap is not None:
            Y_train[Y_train > self.rul_cap] = self.rul_cap
            Y_test[Y_test > self.rul_cap] = self.rul_cap

        self.X_train = X_train[:, :, np.newaxis, :].astype(np.float32)
        self.Y_train = Y_train.astype(np.float32)
        self.X_test = X_test[:, :, np.newaxis, :].astype(np.float32)
        self.Y_test = Y_test.astype(np.float32)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.name}",
            f"  Sequence length: {self.sequence_length}",
            f"  RUL cap: {self.rul_cap}",
            f"  Features ({len(self.feature_names)}): {self.feature_names[:10]}{'...' if len(self.feature_names) > 10 else ''}",
        ]
        if self.X_train is not None:
            lines.append(f"  X_train: {self.X_train.shape}  Y_train: {self.Y_train.shape}")
            lines.append(f"  X_test:  {self.X_test.shape}   Y_test:  {self.Y_test.shape}")
            lines.append(f"  Y_train range: [{self.Y_train.min():.1f}, {self.Y_train.max():.1f}]")
            lines.append(f"  Y_test  range: [{self.Y_test.min():.1f}, {self.Y_test.max():.1f}]")
        else:
            lines.append("  [Data not yet loaded]")
        return "\n".join(lines)


# =========================================================================
# Config-driven loader
# =========================================================================

def load_dataset_from_config(config: Dict) -> GenericTimeSeriesDataset:
    """Create and populate a GenericTimeSeriesDataset from a config dict.

    Example configs:

    Synthetic:
        {"type": "synthetic", "name": "synth", "sequence_length": 30,
         "n_units_train": 80, "n_units_test": 20, "max_life": 200}

    CSV:
        {"type": "csv", "name": "my_turbines", "sequence_length": 30,
         "train_path": "data/train.csv", "unit_col": "turbine_id",
         "cycle_col": "timestamp", "target_col": "RUL"}

    NASA Bearing:
        {"type": "nasa_bearing", "data_dir": "data/IMS", "sequence_length": 50}

    Battery:
        {"type": "battery", "csv_path": "data/battery.csv",
         "sequence_length": 20, "eol_threshold": 0.7}

    PHM 2012:
        {"type": "phm2012", "data_dir": "data/PHM2012", "condition": "1",
         "sequence_length": 40}

    NumPy:
        {"type": "numpy", "train_X_path": "data/X_train.npy",
         "train_Y_path": "data/Y_train.npy", ...}
    """
    ds_type = config.get("type", "synthetic")
    seq_len = config.get("sequence_length", 30)
    rul_cap = config.get("rul_cap", 125.0)
    test_last = config.get("test_last_only", True)
    name = config.get("name", ds_type)

    ds = GenericTimeSeriesDataset(
        name=name, sequence_length=seq_len,
        rul_cap=rul_cap, test_last_only=test_last,
    )

    if ds_type == "synthetic":
        ds.load_synthetic(
            n_units_train=config.get("n_units_train", 80),
            n_units_test=config.get("n_units_test", 20),
            max_life=config.get("max_life", 200),
            n_features=config.get("n_features", 14),
            noise_std=config.get("noise_std", 0.3),
            seed=config.get("seed", 42),
        )

    elif ds_type == "csv":
        ds.load_from_csv(
            train_path=config["train_path"],
            test_path=config.get("test_path"),
            test_rul_path=config.get("test_rul_path"),
            unit_col=config.get("unit_col", "unit_id"),
            cycle_col=config.get("cycle_col", "cycle"),
            target_col=config.get("target_col", "RUL"),
            feature_cols=config.get("feature_cols"),
            delimiter=config.get("delimiter", ","),
            train_ratio=config.get("train_ratio", 0.8),
            column_names=config.get("column_names"),
        )

    elif ds_type == "nasa_bearing":
        ds.load_nasa_bearing(
            data_dir=config["data_dir"],
            test_ratio=config.get("test_ratio", 0.2),
        )

    elif ds_type == "battery":
        ds.load_battery(
            csv_path=config["csv_path"],
            cycle_col=config.get("cycle_col", "cycle"),
            capacity_col=config.get("capacity_col", "capacity"),
            cell_col=config.get("cell_col", "cell_id"),
            eol_threshold=config.get("eol_threshold", 0.7),
            feature_cols=config.get("feature_cols"),
            train_ratio=config.get("train_ratio", 0.8),
        )

    elif ds_type == "phm2012":
        ds.load_phm2012(
            data_dir=config["data_dir"],
            condition=config.get("condition", "1"),
            train_bearings=config.get("train_bearings"),
            test_bearings=config.get("test_bearings"),
        )

    elif ds_type == "numpy":
        X_train = np.load(config["train_X_path"])
        Y_train = np.load(config["train_Y_path"])
        X_test = np.load(config["test_X_path"])
        Y_test = np.load(config["test_Y_path"])
        ds.load_from_numpy(
            X_train, Y_train, X_test, Y_test,
            already_windowed=config.get("already_windowed", False),
            feature_names=config.get("feature_names"),
        )

    elif ds_type == "synthetic_ode":
        ds.load_synthetic_ode(
            n_units_train=config.get("n_units_train", 80),
            n_units_test=config.get("n_units_test", 20),
            max_life=config.get("max_life", 300),
            dt=config.get("dt", 0.1),
            noise_std=config.get("noise_std", 0.05),
            failure_threshold=config.get("failure_threshold", 2.0),
            seed=config.get("seed", 42),
        )

    elif ds_type == "weather":
        ds.load_weather(
            csv_path=config["csv_path"],
            datetime_col=config.get("datetime_col", "Date Time"),
            target_col=config.get("target_col", "T (degC)"),
            feature_cols=config.get("feature_cols"),
            resample_minutes=config.get("resample_minutes", 60),
            prediction_horizon=config.get("prediction_horizon", 24),
            train_ratio=config.get("train_ratio", 0.8),
            delimiter=config.get("delimiter", ","),
        )

    elif ds_type == "weather_synthetic":
        ds.load_weather_synthetic(
            n_days=config.get("n_days", 365),
            dt_hours=config.get("dt_hours", 1.0),
            n_stations=config.get("n_stations", 1),
            noise_std=config.get("noise_std", 0.5),
            prediction_horizon=config.get("prediction_horizon", 24),
            train_ratio=config.get("train_ratio", 0.8),
            seed=config.get("seed", 42),
        )

    elif ds_type == "finance":
        ds.load_finance(
            csv_path=config["csv_path"],
            datetime_col=config.get("datetime_col", "Date"),
            close_col=config.get("close_col", "Close"),
            feature_cols=config.get("feature_cols"),
            prediction_horizon=config.get("prediction_horizon", 5),
            drawdown_threshold=config.get("drawdown_threshold", 0.05),
            target_mode=config.get("target_mode", "drawdown"),
            train_ratio=config.get("train_ratio", 0.8),
            delimiter=config.get("delimiter", ","),
        )

    elif ds_type == "finance_synthetic":
        ds.load_finance_synthetic(
            n_days=config.get("n_days", 2520),
            n_assets=config.get("n_assets", 1),
            initial_price=config.get("initial_price", 100.0),
            mu=config.get("mu", 0.0005),
            sigma=config.get("sigma", 0.02),
            mean_reversion_speed=config.get("mean_reversion_speed", 0.01),
            regime_switch_prob=config.get("regime_switch_prob", 0.005),
            prediction_horizon=config.get("prediction_horizon", 5),
            target_mode=config.get("target_mode", "drawdown"),
            drawdown_threshold=config.get("drawdown_threshold", 0.05),
            train_ratio=config.get("train_ratio", 0.8),
            seed=config.get("seed", 42),
        )

    elif ds_type in ("fluid_dynamics", "energy_systems"):
        ds.load_physics_csv(
            csv_path=config["csv_path"],
            unit_col=config.get("unit_col", "unit_id"),
            cycle_col=config.get("cycle_col", "cycle"),
            target_col=config.get("target_col", "target"),
            feature_cols=config.get("feature_cols"),
            train_ratio=config.get("train_ratio", 0.8),
        )

    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    return ds


if __name__ == "__main__":
    # Quick test with synthetic data
    ds = GenericTimeSeriesDataset(name="synth_test", sequence_length=30, rul_cap=125)
    ds.load_synthetic(n_units_train=50, n_units_test=10, max_life=200, n_features=14)
    print(ds.summary())
    X_train, Y_train, X_test, Y_test = ds.get_data()
    print(f"\nReady for PI-DP-FCN training:")
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")
