#!/usr/bin/env python3
"""
KePIN PyTorch Model — Koopman-Enhanced Physics-Informed Network.

Architecture:
  [Input] → [ResConv1D+SE blocks] → [BiLSTM] → [Multi-Head Attention]
          → [Koopman Module] → [Spectral Features]
          → [Deep Prediction Head] → target
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Koopman Operator (SVD-parameterized for stability)
# =========================================================================

class KoopmanOperator(nn.Module):
    """Learnable Koopman operator K = U Σ V^T with bounded singular values."""

    def __init__(self, latent_dim, rollout_steps=3, sigma_max=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.rollout_steps = rollout_steps
        self.sigma_max = sigma_max

        # Learnable SVD components
        self.U_raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.V_raw = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.sigma_raw = nn.Parameter(torch.zeros(latent_dim))

    def _orthogonalize(self, M):
        """Gram-Schmidt orthogonalization."""
        Q, R = torch.linalg.qr(M)
        return Q

    def _get_K(self):
        """Construct K = U diag(sigma) V^T with bounded sigmas."""
        U = self._orthogonalize(self.U_raw)
        V = self._orthogonalize(self.V_raw)
        sigma = self.sigma_max * torch.sigmoid(self.sigma_raw)
        K = U @ torch.diag(sigma) @ V.T
        return K

    def forward(self, z_seq):
        """
        Args:
            z_seq: (batch, T, d) latent states
        Returns:
            dict with one_step_pred, one_step_target, multi_step_pred,
            multi_step_target, eigenvalues
        """
        K = self._get_K()
        batch, T, d = z_seq.shape

        # One-step: K @ z(t) vs z(t+1)
        z_t = z_seq[:, :-1, :]      # (B, T-1, d)
        z_tp1 = z_seq[:, 1:, :]     # (B, T-1, d)
        one_step_pred = torch.einsum('ij,btj->bti', K, z_t)

        # Multi-step rollout: K^h @ z(t) vs z(t+h)
        horizons = list(range(2, min(self.rollout_steps + 2, T)))
        multi_preds = []
        multi_targets = []
        K_pow = K.clone()
        for h in horizons:
            K_pow = K_pow @ K if h > 2 else K @ K
            n_valid = T - h
            if n_valid <= 0:
                break
            pred_h = torch.einsum('ij,btj->bti', K_pow, z_seq[:, :n_valid, :])
            tgt_h = z_seq[:, h:h+n_valid, :]
            multi_preds.append(pred_h)
            multi_targets.append(tgt_h)

        if multi_preds:
            # Pad to same time dim and stack
            min_t = min(p.shape[1] for p in multi_preds)
            multi_pred = torch.stack([p[:, :min_t, :] for p in multi_preds], dim=2)
            multi_tgt = torch.stack([t[:, :min_t, :] for t in multi_targets], dim=2)
        else:
            multi_pred = one_step_pred[:, :1, :].unsqueeze(2)
            multi_tgt = z_tp1[:, :1, :].unsqueeze(2)

        # Eigenvalues
        eigenvalues = torch.linalg.eigvals(K)

        return {
            'one_step_pred': one_step_pred,
            'one_step_target': z_tp1,
            'multi_step_pred': multi_pred,
            'multi_step_target': multi_tgt,
            'eigenvalues': eigenvalues,
            'K': K,
        }


# =========================================================================
# SE (Squeeze-and-Excitation) Block
# =========================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: (B, C, T) for Conv1d
        s = x.mean(dim=-1)  # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


# =========================================================================
# Residual Conv1D Block
# =========================================================================

class ResConv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.skip_bn = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        residual = self.skip_bn(self.skip(x)) if isinstance(self.skip, nn.Conv1d) else x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(h + residual)


# =========================================================================
# Auto-configuration
# =========================================================================

def auto_configure(n_features, seq_len):
    """Determine architecture hyperparameters from data shape."""
    if n_features <= 10:
        tier = "small"
    elif n_features <= 16:
        tier = "medium"
    else:
        tier = "large"

    configs = {
        "small": {
            "filters": [64, 128, 128],
            "kernels": [7, 5, 3],
            "latent_dim": 64,
            "lstm_units": 64,
            "n_heads": 4,
            "dropout": 0.3,
            "rollout": 3,
        },
        "medium": {
            "filters": [64, 128, 128, 256],
            "kernels": [7, 5, 5, 3],
            "latent_dim": 128,
            "lstm_units": 128,
            "n_heads": 4,
            "dropout": 0.35,
            "rollout": 3,
        },
        "large": {
            "filters": [64, 128, 256, 256],
            "kernels": [7, 5, 5, 3],
            "latent_dim": 128,
            "lstm_units": 128,
            "n_heads": 8,
            "dropout": 0.4,
            "rollout": 3,
        },
    }

    cfg = configs[tier]
    cfg["tier"] = tier
    cfg["kernels"] = [min(k, seq_len) | 1 for k in cfg["kernels"]]  # ensure odd
    return cfg


# =========================================================================
# KePIN Model
# =========================================================================

class KePINModel(nn.Module):
    """Koopman-Enhanced Physics-Informed Network (PyTorch)."""

    def __init__(self, seq_len, n_features, arch_config=None):
        super().__init__()
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len)
        self.arch_config = arch_config
        self.seq_len = seq_len
        self.n_features = n_features

        filters = arch_config["filters"]
        kernels = arch_config["kernels"]
        latent_dim = arch_config["latent_dim"]
        lstm_units = arch_config["lstm_units"]
        n_heads = arch_config["n_heads"]
        dropout = arch_config["dropout"]

        # Input projection
        self.input_proj = nn.Conv1d(n_features, filters[0], 1)
        self.input_bn = nn.BatchNorm1d(filters[0])

        # Residual Conv1D Encoder
        self.encoder = nn.ModuleList()
        in_ch = filters[0]
        for i, (f, k) in enumerate(zip(filters, kernels)):
            self.encoder.append(ResConv1DBlock(in_ch, f, k))
            in_ch = f

        # BiLSTM
        self.bilstm = nn.LSTM(
            filters[-1], lstm_units, batch_first=True,
            bidirectional=True, dropout=dropout * 0.5,
        )
        self.lstm_ln = nn.LayerNorm(lstm_units * 2)
        self.post_lstm = nn.Conv1d(lstm_units * 2, latent_dim, 1)
        self.post_lstm_bn = nn.BatchNorm1d(latent_dim)

        # Multi-head attention
        self.mha = nn.MultiheadAttention(
            latent_dim, n_heads, dropout=dropout * 0.3, batch_first=True,
        )
        self.mha_ln = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Dropout(dropout * 0.5),
        )
        self.ff_ln = nn.LayerNorm(latent_dim)

        # Latent projection
        self.latent_proj = nn.Conv1d(latent_dim, latent_dim, 1)
        self.latent_bn = nn.BatchNorm1d(latent_dim)

        # Koopman operator
        self.koopman = KoopmanOperator(
            latent_dim, rollout_steps=arch_config["rollout"],
        )

        # Prediction head
        head_in = latent_dim * 2 + latent_dim  # dual pool + spectral
        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_skip = nn.Linear(latent_dim * 2, 64)
        self.head_out = nn.Linear(64, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            pred: (batch, 1)
            koopman_out: dict
        """
        B = x.shape[0]

        # Conv encoder: (B, T, F) -> (B, F, T) -> encoder -> (B, C, T)
        h = x.permute(0, 2, 1)
        h = F.relu(self.input_bn(self.input_proj(h)))
        for block in self.encoder:
            h = block(h)

        # BiLSTM: (B, C, T) -> (B, T, C) -> BiLSTM
        h = h.permute(0, 2, 1)
        h, _ = self.bilstm(h)
        h = self.lstm_ln(h)
        h = h.permute(0, 2, 1)   # (B, 2*lstm, T)
        h = F.relu(self.post_lstm_bn(self.post_lstm(h)))

        # Multi-head attention: (B, C, T) -> (B, T, C)
        h = h.permute(0, 2, 1)
        attn_out, _ = self.mha(h, h, h)
        h = self.mha_ln(h + attn_out)
        ff_out = self.ff(h)
        h = self.ff_ln(h + ff_out)

        # Latent projection: (B, T, C) -> (B, C, T) -> proj -> (B, T, d)
        z = h.permute(0, 2, 1)
        z = self.latent_bn(self.latent_proj(z))
        z = z.permute(0, 2, 1)  # (B, T, d)

        # Koopman
        koopman_out = self.koopman(z)

        # Dual pooling
        pool_avg = z.mean(dim=1)   # (B, d)
        pool_max = z.max(dim=1)[0] # (B, d)
        pooled = torch.cat([pool_avg, pool_max], dim=-1)  # (B, 2d)

        # Spectral features from eigenvalues
        eigs = koopman_out['eigenvalues']  # (d,) complex
        spec_feats = torch.cat([
            torch.abs(eigs).unsqueeze(0).expand(B, -1),
        ], dim=-1)  # (B, d)

        # Head
        head_in = torch.cat([pooled, spec_feats], dim=-1)
        h_deep = self.head(head_in)
        h_skip = F.relu(self.head_skip(pooled))
        pred = self.head_out(h_deep + h_skip)

        return pred, koopman_out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            pred, _ = self(x)
        return pred

    def get_eigenvalues(self):
        K = self.koopman._get_K()
        return torch.linalg.eigvals(K).detach().cpu().numpy()

    def get_koopman_matrix(self):
        return self.koopman._get_K().detach().cpu().numpy()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary_config(self):
        cfg = self.arch_config
        n = self.count_params()
        return (f"KePIN({cfg['tier']}) | params={n:,} | "
                f"latent={cfg['latent_dim']} | "
                f"blocks={len(cfg['filters'])} | "
                f"heads={cfg['n_heads']}")


# =========================================================================
# Loss Functions (4 core losses)
# =========================================================================

def prediction_loss(y_true, y_pred, delta=0.15):
    """Huber loss — robust to outliers. Delta tuned for [0,1]-normalized targets."""
    return F.huber_loss(y_pred.flatten(), y_true.flatten(), reduction='mean', delta=delta)


def koopman_consistency_loss(one_step_pred, one_step_target):
    """||K z(t) - z(t+1)||^2 — latent dynamics accuracy."""
    return F.mse_loss(one_step_pred, one_step_target)


def spectral_stability_loss(eigenvalues):
    """|λ| should stay ≤ 1 — penalize growing modes."""
    mags = torch.abs(eigenvalues)
    violation = F.relu(mags - 1.0)
    return (violation ** 2).mean()


def multistep_rollout_loss(multi_pred, multi_target):
    """||K^h z(t) - z(t+h)||^2 — long-horizon fidelity."""
    return F.mse_loss(multi_pred, multi_target)


class AutoBalancedLoss(nn.Module):
    """Simple prediction-anchored auxiliary weighting.

    Prediction loss always has weight 1.0.
    Auxiliary losses (Koopman, spectral, multi-step) use fixed small weights
    that keep them as gentle regularizers without interfering with prediction.

    Total = L_pred + aux_weight * (L_koop + L_spec + L_multi)
    """

    def __init__(self, n_aux=3, aux_cap=0.5):
        super().__init__()
        self.aux_weight = aux_cap  # fixed weight for each auxiliary loss
        self.n_aux = n_aux
        # Keep as parameter for compatibility (not actually learned)
        self.log_vars = nn.Parameter(torch.zeros(n_aux), requires_grad=False)

    def forward(self, losses):
        """
        Args:
            losses: [L_pred, L_koop, L_spec, L_multi]
        Returns:
            total: scalar
            weights: (4,) effective weights for monitoring
        """
        pred_loss = losses[0]
        aux_losses = losses[1:]

        total = pred_loss
        for aux_l in aux_losses:
            total = total + self.aux_weight * aux_l

        all_weights = torch.tensor(
            [1.0] + [self.aux_weight] * self.n_aux,
            device=pred_loss.device)
        return total, all_weights


def compute_kepin_loss(y_true, y_pred, koopman_out, loss_balancer, aux_scale=1.0):
    """Compute the 4-component KePIN loss.

    Components:
      1. L_pred:  Huber prediction loss — drives accuracy
      2. L_koop:  Koopman consistency  — enforces linear dynamics
      3. L_spec:  Spectral stability   — prevents explosive modes
      4. L_multi: Multi-step rollout   — ensures long-horizon fidelity

    Args:
        aux_scale: float in [0, 1], scales auxiliary losses for warmup.
                   0 = prediction-only, 1 = full auxiliary contribution.
    Returns:
        total_loss, loss_dict
    """
    l_pred = prediction_loss(y_true, y_pred)
    l_koop = koopman_consistency_loss(
        koopman_out['one_step_pred'], koopman_out['one_step_target'])
    l_spec = spectral_stability_loss(koopman_out['eigenvalues'])
    l_multi = multistep_rollout_loss(
        koopman_out['multi_step_pred'], koopman_out['multi_step_target'])

    total, weights = loss_balancer(
        [l_pred, l_koop * aux_scale, l_spec * aux_scale, l_multi * aux_scale])

    loss_dict = {
        'total': total.item(),
        'pred': l_pred.item(),
        'koopman': l_koop.item(),
        'spectral': l_spec.item(),
        'multistep': l_multi.item(),
        'weights': weights.cpu().numpy(),
    }

    return total, loss_dict


# =========================================================================
# Test
# =========================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    for name, sf, nf in [("Small", 30, 8), ("Medium", 40, 14), ("Large", 30, 24)]:
        model = KePINModel(sf, nf).to(device)
        x = torch.randn(4, sf, nf, device=device)
        pred, kout = model(x)
        print(f"{name}: {model.summary_config()} | pred={pred.shape}")

    # Test loss
    y_true = torch.randn(4, 1, device=device)
    balancer = AutoBalancedLoss(n_aux=3, aux_cap=0.5).to(device)
    total, ld = compute_kepin_loss(y_true, pred, kout, balancer)
    print(f"\nLoss: {total.item():.4f} | components: pred={ld['pred']:.4f} "
          f"koop={ld['koopman']:.4f} spec={ld['spectral']:.6f} "
          f"multi={ld['multistep']:.4f}")
    print(f"Weights: {ld['weights']}")
    print("\n✓ All tests passed.")
