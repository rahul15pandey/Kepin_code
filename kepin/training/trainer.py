# -*- coding: utf-8 -*-
"""
KePIN Training Pipeline.

Two trainer classes:
  - ``KePINTrainer`` — Base custom training loop with Koopman-aware loss
  - ``EnhancedKePINTrainer`` — Adds LR warmup, SWA, curriculum loss, mixup
"""

import math
import numpy as np
import tensorflow as tf
import keras

from kepin.utils.gpu import build_tf_dataset, is_mixed_precision_enabled
from kepin.utils.metrics import rmse_np

SEED = 42


class KePINTrainer:
    """Custom training loop handling Koopman-aware composite loss.

    Keras ``model.fit()`` only passes (y_true, y_pred) to the loss;
    KePIN needs the full Koopman outputs (eigenvalues, one-step predictions,
    multi-step rollouts). This trainer handles that.
    """

    def __init__(self, model, loss_fn, optimizer, clip_norm: float = 2.0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self._mixed = is_mixed_precision_enabled()

    @tf.function(jit_compile=False)
    def train_step(self, X_batch, Y_batch):
        with tf.GradientTape() as tape:
            rul_pred, koopman_out = self.model(X_batch, training=True)
            rul_pred_f32 = tf.cast(rul_pred, tf.float32)
            Y_batch_f32 = tf.cast(Y_batch, tf.float32)
            koopman_f32 = {k: tf.cast(v, tf.float32)
                           if isinstance(v, tf.Tensor) and v.dtype != tf.complex64 else v
                           for k, v in koopman_out.items()
                           if isinstance(v, tf.Tensor)}
            total_loss, loss_dict = self.loss_fn(Y_batch_f32, rul_pred_f32, koopman_f32)
            scaled_loss = total_loss
            if self._mixed and hasattr(self.optimizer, 'get_scaled_loss'):
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        if self._mixed and hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return total_loss, loss_dict, rul_pred_f32

    @tf.function(jit_compile=False)
    def eval_step(self, X_batch, Y_batch):
        rul_pred, koopman_out = self.model(X_batch, training=False)
        rul_pred_f32 = tf.cast(rul_pred, tf.float32)
        Y_batch_f32 = tf.cast(Y_batch, tf.float32)
        koopman_f32 = {k: tf.cast(v, tf.float32)
                       if isinstance(v, tf.Tensor) and v.dtype != tf.complex64 else v
                       for k, v in koopman_out.items()
                       if isinstance(v, tf.Tensor)}
        total_loss, loss_dict = self.loss_fn(Y_batch_f32, rul_pred_f32, koopman_f32)
        return total_loss, loss_dict, rul_pred_f32

        def fit(self, X_train, Y_train, X_val, Y_val,
            epochs=200, batch_size=512, patience=40,
            lr_schedule="cosine", initial_lr=0.003, min_lr=1e-5,
            seed: int = SEED, verbose=1):
        """Full training loop with early stopping and cosine annealing.

        Returns:
            history dict with per-epoch metrics and eigenvalue snapshots
        """
        n_train = len(X_train)
        train_ds = build_tf_dataset(X_train, Y_train, batch_size, shuffle=True, seed=seed)
        val_ds = build_tf_dataset(X_val, Y_val, batch_size, shuffle=False, seed=seed)

        history = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "train_rmse": [], "val_rmse": [], "eigenvalues": [],
            "loss_weights": [], "lr": [],
            "train_rul_mse": [], "train_koopman": [], "train_spectral": [],
            "train_mono": [], "train_multi_step": [],
        }

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(epochs):
            # Cosine annealing with warm restarts
            if lr_schedule == "cosine":
                T_0 = max(epochs // 3, 50)
                T_cur = epoch % T_0
                lr = min_lr + 0.5 * (initial_lr - min_lr) * (
                    1 + math.cos(math.pi * T_cur / T_0))
            else:
                lr = initial_lr
            self.optimizer.learning_rate.assign(lr)
            history["lr"].append(lr)

            epoch_losses, epoch_preds = [], []
            comp = {"rul_mse": [], "koopman_1step": [],
                    "spectral": [], "monotonicity": [], "multi_step": []}

            for X_b, Y_b in train_ds:
                total_loss, loss_dict, rul_pred = self.train_step(X_b, Y_b)
                epoch_losses.append(float(total_loss))
                epoch_preds.append(rul_pred.numpy())
                for key in comp:
                    if key in loss_dict:
                        comp[key].append(float(loss_dict[key]))

            train_loss = np.mean(epoch_losses)
            train_preds = np.concatenate(epoch_preds, axis=0)[:n_train]
            train_rmse = rmse_np(Y_train[:len(train_preds)], train_preds)

            val_preds_list, val_losses = [], []
            for X_vb, Y_vb in val_ds:
                v_loss, _, v_pred = self.eval_step(X_vb, Y_vb)
                val_losses.append(float(v_loss))
                val_preds_list.append(v_pred.numpy())

            val_loss = np.mean(val_losses)
            val_preds = np.concatenate(val_preds_list, axis=0)
            val_rmse = rmse_np(Y_val[:len(val_preds)], val_preds)

            eigs = self.model.get_eigenvalues()
            history["eigenvalues"].append(eigs.copy())

            try:
                eff = tf.exp(-self.model.loss_weight_layer.log_vars).numpy()
                history["loss_weights"].append(eff.copy())
            except Exception:
                history["loss_weights"].append(None)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)
            for key in comp:
                if f"train_{key}" in history:
                    history[f"train_{key}"].append(
                        np.mean(comp[key]) if comp[key] else 0.0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                eig_mags = np.sort(np.abs(eigs))[::-1][:3]
                print(f"  Epoch {epoch+1:4d}/{epochs} | LR={lr:.6f} | "
                      f"Train: {train_loss:.4f}/{train_rmse:.2f} | "
                      f"Val: {val_loss:.4f}/{val_rmse:.2f} | |λ|={eig_mags}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_weights is not None:
            self.model.set_weights(best_weights)
            if verbose:
                print(f"  Restored best weights (val_loss={best_val_loss:.4f})")

        return history


class EnhancedKePINTrainer(KePINTrainer):
    """Extended trainer with LR warmup, SWA, curriculum loss, and mixup."""

    def __init__(self, model, loss_fn, optimizer, clip_norm=2.0,
                 warmup_epochs=5, curriculum_warmup=30,
                 mixup_alpha=0.2, swa_start_frac=0.75):
        super().__init__(model, loss_fn, optimizer, clip_norm)
        self.warmup_epochs = warmup_epochs
        self.curriculum_warmup = curriculum_warmup
        self.mixup_alpha = mixup_alpha
        self.swa_start_frac = swa_start_frac
        self.swa_weights = None
        self.swa_count = 0

    @tf.function(jit_compile=False)
    def train_step_mixup(self, X1, Y1, X2, Y2, lam):
        X_mix = lam * X1 + (1.0 - lam) * X2
        lam_y = tf.reshape(lam[:, 0, 0], (-1, 1))
        Y_mix = lam_y * Y1 + (1.0 - lam_y) * Y2

        with tf.GradientTape() as tape:
            rul_pred, koopman_out = self.model(X_mix, training=True)
            rul_pred_f32 = tf.cast(rul_pred, tf.float32)
            Y_mix_f32 = tf.cast(Y_mix, tf.float32)
            koopman_f32 = {k: tf.cast(v, tf.float32)
                           if isinstance(v, tf.Tensor) and v.dtype != tf.complex64 else v
                           for k, v in koopman_out.items()
                           if isinstance(v, tf.Tensor)}
            total_loss, loss_dict = self.loss_fn(Y_mix_f32, rul_pred_f32, koopman_f32)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return total_loss, loss_dict, rul_pred_f32

    def _update_swa(self):
        current_weights = self.model.get_weights()
        if self.swa_weights is None:
            self.swa_weights = [w.copy() for w in current_weights]
            self.swa_count = 1
        else:
            self.swa_count += 1
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (
                    self.swa_weights[i] * (self.swa_count - 1) + current_weights[i]
                ) / self.swa_count

    def fit_enhanced(self, X_train, Y_train, X_val, Y_val,
                     epochs=200, batch_size=128, patience=40,
                     initial_lr=0.001, min_lr=1e-6,
                     noise_std=0.01, seed: int = SEED, verbose=1):
        """Enhanced training with warmup, SWA, curriculum, and mixup.

        Returns:
            (history, swa_weights)
        """
        n_train = len(X_train)
        swa_start_epoch = int(epochs * self.swa_start_frac)

        train_ds = build_tf_dataset(X_train, Y_train, batch_size, shuffle=True, seed=seed)
        val_ds = build_tf_dataset(X_val, Y_val, batch_size, shuffle=False, seed=seed)
        train_ds_2 = build_tf_dataset(X_train, Y_train, batch_size, shuffle=True, seed=seed + 7)

        history = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "train_rmse": [], "val_rmse": [], "eigenvalues": [],
            "loss_weights": [], "lr": [],
            "train_rul_mse": [], "train_koopman": [], "train_spectral": [],
            "train_mono": [], "train_multi_step": [],
        }

        best_val_rmse = float("inf")
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(epochs):
            # LR: warmup + cosine annealing
            if epoch < self.warmup_epochs:
                lr = initial_lr * (epoch + 1) / self.warmup_epochs
            else:
                T_0 = max((epochs - self.warmup_epochs) // 3, 40)
                T_cur = (epoch - self.warmup_epochs) % T_0
                lr = min_lr + 0.5 * (initial_lr - min_lr) * (
                    1 + math.cos(math.pi * T_cur / T_0))
            self.optimizer.learning_rate.assign(lr)
            history["lr"].append(lr)

            epoch_losses, epoch_preds = [], []
            comp = {"rul_mse": [], "koopman_1step": [],
                    "spectral": [], "monotonicity": [], "multi_step": []}

            use_mixup = self.mixup_alpha > 0 and epoch >= self.warmup_epochs

            if use_mixup:
                for (X_b1, Y_b1), (X_b2, Y_b2) in zip(train_ds, train_ds_2):
                    min_bs = tf.minimum(tf.shape(X_b1)[0], tf.shape(X_b2)[0])
                    X_b1, Y_b1 = X_b1[:min_bs], Y_b1[:min_bs]
                    X_b2, Y_b2 = X_b2[:min_bs], Y_b2[:min_bs]
                    lam = tf.maximum(
                        tf.random.uniform((min_bs, 1, 1), 0.0, 1.0),
                        1.0 - tf.random.uniform((min_bs, 1, 1), 0.0, 1.0))
                    total_loss, loss_dict, rul_pred = self.train_step_mixup(
                        X_b1, Y_b1, X_b2, Y_b2, lam)
                    epoch_losses.append(float(total_loss))
                    epoch_preds.append(rul_pred.numpy())
                    for key in comp:
                        if key in loss_dict:
                            comp[key].append(float(loss_dict[key]))
            else:
                for X_b, Y_b in train_ds:
                    if noise_std > 0:
                        X_b = X_b + tf.random.normal(tf.shape(X_b), stddev=noise_std)
                    total_loss, loss_dict, rul_pred = self.train_step(X_b, Y_b)
                    epoch_losses.append(float(total_loss))
                    epoch_preds.append(rul_pred.numpy())
                    for key in comp:
                        if key in loss_dict:
                            comp[key].append(float(loss_dict[key]))

            train_loss = np.mean(epoch_losses)
            train_preds = np.concatenate(epoch_preds, axis=0)[:n_train]
            train_rmse = rmse_np(Y_train[:len(train_preds)], train_preds)

            val_preds_list, val_losses = [], []
            for X_vb, Y_vb in val_ds:
                v_loss, _, v_pred = self.eval_step(X_vb, Y_vb)
                val_losses.append(float(v_loss))
                val_preds_list.append(v_pred.numpy())

            val_loss = np.mean(val_losses)
            val_preds = np.concatenate(val_preds_list, axis=0)
            val_rmse = rmse_np(Y_val[:len(val_preds)], val_preds)

            if epoch >= swa_start_epoch:
                self._update_swa()

            eigs = self.model.get_eigenvalues()
            history["eigenvalues"].append(eigs.copy())

            try:
                eff = tf.exp(-self.model.loss_weight_layer.log_vars).numpy()
                history["loss_weights"].append(eff.copy())
            except Exception:
                history["loss_weights"].append(None)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)
            for key in comp:
                if f"train_{key}" in history:
                    history[f"train_{key}"].append(
                        np.mean(comp[key]) if comp[key] else 0.0)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                eig_mags = np.sort(np.abs(eigs))[::-1][:3]
                swa_tag = " [SWA]" if epoch >= swa_start_epoch else ""
                print(f"  Epoch {epoch+1:4d}/{epochs} | LR={lr:.6f} | "
                      f"Train: {train_loss:.4f}/{train_rmse:.2f} | "
                      f"Val: {val_loss:.4f}/{val_rmse:.2f} | "
                      f"|λ|={eig_mags}{swa_tag}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_weights is not None:
            self.model.set_weights(best_weights)
            if verbose:
                print(f"  Restored best weights (val_rmse={best_val_rmse:.4f})")

        return history, self.swa_weights
