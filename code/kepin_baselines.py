#!/usr/bin/env python3
"""
Baseline Models for KePIN Comparison — PyTorch implementations.

Implements controlled baselines under identical settings:
  1. MLP (simple feedforward)
  2. LSTM
  3. BiLSTM
  4. CNN-LSTM
  5. Transformer
  6. Vanilla FCN (no Koopman)
  7. PINN-like (FCN + physics loss, no Koopman)

All baselines use the same:
  - Data splits and preprocessing
  - Training budget (same epochs, patience)
  - Optimizer (AdamW)
  - Batch size
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================================================
# Baseline Models
# =========================================================================

class MLPBaseline(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        inp = seq_len * n_feat
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inp, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class LSTMBaseline(nn.Module):
    def __init__(self, seq_len, n_feat, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True, num_layers=2, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class BiLSTMBaseline(nn.Module):
    def __init__(self, seq_len, n_feat, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class CNNLSTMBaseline(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :])


class TransformerBaseline(nn.Module):
    def __init__(self, seq_len, n_feat, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.shape[1], :]
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class VanillaFCN(nn.Module):
    """FCN without Koopman — same encoder as KePIN but no Koopman module."""
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.conv(x.permute(0, 2, 1))
        h = self.gap(h).squeeze(-1)
        return self.head(h)


BASELINES = {
    'MLP': MLPBaseline,
    'LSTM': LSTMBaseline,
    'BiLSTM': BiLSTMBaseline,
    'CNN-LSTM': CNNLSTMBaseline,
    'Transformer': TransformerBaseline,
    'Vanilla FCN': VanillaFCN,
}


# =========================================================================
# Training
# =========================================================================

def train_baseline(model_cls, name, X_train, Y_train_raw, X_test, Y_test_raw,
                   epochs=100, batch_size=128, lr=0.001, patience=30):
    """Train a baseline model with target normalization."""
    torch.manual_seed(SEED)

    # Target normalization to [0, 1]
    y_min = float(Y_train_raw.min())
    y_max = float(Y_train_raw.max())
    y_range = max(y_max - y_min, 1e-6)
    Y_train = (Y_train_raw - y_min) / y_range
    Y_test = (Y_test_raw - y_min) / y_range

    seq_len, n_feat = X_train.shape[1], X_train.shape[2]
    model = model_cls(seq_len, n_feat).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.15)

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test)),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    best_loss = float('inf')
    best_state = None
    wait = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for X_b, Y_b in train_dl:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            loss = criterion(pred, Y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, Y_b in val_dl:
                X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
                pred = model(X_b)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(1)
                val_losses.append(criterion(pred, Y_b).item())

        val_loss = np.mean(val_losses)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Final eval
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for X_b, Y_b in val_dl:
            X_b = X_b.to(DEVICE)
            pred = model(X_b)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(Y_b.numpy())

    Y_pred = np.concatenate(all_preds)
    Y_true_norm = np.concatenate(all_tgts)

    # Denormalize
    Y_pred_orig = Y_pred * y_range + y_min
    Y_true_orig = Y_true_norm * y_range + y_min
    y_true_flat = Y_true_orig.flatten()
    y_pred_flat = Y_pred_orig.flatten()
    rmse = float(np.sqrt(((y_true_flat - y_pred_flat) ** 2).mean()))
    mae = float(np.abs(y_true_flat - y_pred_flat).mean())
    ss_res = float(((y_true_flat - y_pred_flat) ** 2).sum())
    ss_tot = float(((y_true_flat - y_true_flat.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    elapsed = time.time() - start

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'model': name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'params': n_params,
        'epochs': epoch + 1,
        'time': elapsed,
    }


def run_all_baselines(config_path, output_dir=None, dataset_filter=None):
    """Run all baselines on all datasets."""
    from kepin_torch_training import load_dataset

    with open(config_path) as f:
        configs = json.load(f)

    if output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_DIR, "experiments_result",
                                  f"baselines_{ts}")

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for ds_config in configs:
        ds_name = ds_config.get("name", "unknown")
        if dataset_filter and ds_name not in dataset_filter:
            continue

        print(f"\n{'='*60}")
        print(f"  Baselines on: {ds_name}")
        print(f"{'='*60}")

        X_train, Y_train, X_test, Y_test = load_dataset(ds_config)
        print(f"  Data: train={X_train.shape}, test={X_test.shape}")

        for bname, bcls in BASELINES.items():
            try:
                res = train_baseline(
                    bcls, bname, X_train, Y_train, X_test, Y_test,
                    epochs=100, batch_size=128, lr=0.001, patience=30)
                res['dataset'] = ds_name
                all_results.append(res)
                print(f"  {bname:15s}: RMSE={res['rmse']:.4f}  MAE={res['mae']:.4f}  "
                      f"R²={res['r2']:.4f}  Params={res['params']:,}  Time={res['time']:.0f}s")
            except Exception as e:
                print(f"  {bname}: FAILED — {e}")
                all_results.append({'model': bname, 'dataset': ds_name, 'error': str(e)})

    # Summary
    df = pd.DataFrame([r for r in all_results if 'error' not in r])
    if len(df) > 0:
        summary_path = os.path.join(output_dir, "baselines_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nBaselines saved to: {output_dir}")

        # Print pivot table
        pivot = df.pivot_table(index='model', columns='dataset', values='rmse')
        print("\nRMSE Comparison:")
        print(pivot.to_string())

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "datasets_all_config.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    run_all_baselines(args.config, args.output, args.datasets)
