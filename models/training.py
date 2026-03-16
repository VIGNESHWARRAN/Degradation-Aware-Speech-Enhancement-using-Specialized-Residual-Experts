"""
training.py
-----------
Trainer class implementing two-stage training:

Stage 1  (stage1_epochs):
    - Backbone fully frozen
    - Train: ExpertSet + WaveformDecoder
    - Higher LR for fast convergence of new modules

Stage 2  (stage2_epochs):
    - Top-N transformer layers of backbone unfrozen
    - Train: full model end-to-end with lower LR
    - LR warmup + cosine decay

Features:
    - Mixed precision (AMP) via torch.cuda.amp
    - Gradient clipping
    - Best-model checkpoint saving (based on val loss)
    - TensorBoard logging
    - Per-degradation-type loss logging
    - Resume from checkpoint
"""

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Avoid torch.utils.tensorboard which triggers a TensorFlow import on some systems
import sys
import types
# Patch: prevent tensorflow lazy-loading from breaking tensorboard
_tb_writer_mod = None
try:
    # Try standalone tensorboard first (avoids TF conflict)
    import importlib
    import tensorboard
    from tensorboard.summary.writer.event_file_writer import EventFileWriter  # noqa: ensures tb is loaded
    # Now safely import SummaryWriter via torch wrapper
    import torch.utils.tensorboard as _tbu
    SummaryWriter = _tbu.SummaryWriter
except AttributeError:
    # TensorFlow is corrupting tensorboard — use a no-op fallback writer
    class SummaryWriter:  # type: ignore
        """No-op fallback when TensorBoard is broken by TF conflict."""
        def __init__(self, log_dir=None, **kwargs):
            os.makedirs(log_dir, exist_ok=True) if log_dir else None
            print(f"[TensorBoard] Using no-op writer (TF conflict detected). "
                  f"Fix: pip uninstall tensorflow  OR  pip install tensorboard==2.13.0")
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

from models.model   import DegradationAwareSpeechEnhancer
from models.losses  import CompositeLoss
from data.dataset   import build_dataloaders, DEGRADATION_TO_IDX


IDX_TO_NAME = {v: k for k, v in DEGRADATION_TO_IDX.items()}


# ------------------------------------------------------------------
# Learning-rate scheduler helper
# ------------------------------------------------------------------
def _build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: dict):
        """
        cfg keys (all have defaults):
            data_root         : str    — path to final_processed/
            output_dir        : str    — where to save checkpoints / logs
            # Model
            expert_dim        : int    (512)
            num_expert_blocks : int    (4)
            decoder_dim       : int    (512)
            num_upsample      : int    (8)
            dropout           : float  (0.1)
            unfreeze_top_n    : int    (4)   # Stage 2 backbone layers
            # Training
            batch_size        : int    (8)
            num_workers       : int    (4)
            stage1_epochs     : int    (20)
            stage2_epochs     : int    (20)
            stage1_lr         : float  (3e-4)
            stage2_lr         : float  (5e-5)
            warmup_ratio      : float  (0.05)
            grad_clip         : float  (1.0)
            # Loss weights
            lambda_l1         : float  (0.5)
            lambda_stft       : float  (1.0)
            lambda_sisnr      : float  (0.5)
            # Misc
            seed              : int    (42)
            resume_from       : str    (None)
            log_interval      : int    (50)   # steps between console logs
        """
        self.cfg = {
            # defaults
            "data_root"        : "data/final_processed",
            "output_dir"       : "outputs",
            "expert_dim"       : 512,
            "num_expert_blocks": 4,
            "decoder_dim"      : 512,
            "num_upsample"     : 8,
            "dropout"          : 0.1,
            "unfreeze_top_n"   : 4,
            "batch_size"       : 8,
            "num_workers"      : 4,
            "stage1_epochs"    : 20,
            "stage2_epochs"    : 20,
            "stage1_lr"        : 3e-4,
            "stage2_lr"        : 5e-5,
            "warmup_ratio"     : 0.05,
            "grad_clip"        : 1.0,
            "lambda_l1"        : 0.5,
            "lambda_stft"      : 1.0,
            "lambda_sisnr"     : 0.5,
            "seed"             : 42,
            "resume_from"      : None,
            "log_interval"     : 50,
            **cfg,              # user overrides
        }

        torch.manual_seed(self.cfg["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")

        # Directories
        self.output_dir = Path(self.cfg["output_dir"])
        self.ckpt_dir   = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer     = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        # Data
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            data_root   = self.cfg["data_root"],
            batch_size  = self.cfg["batch_size"],
            num_workers = self.cfg["num_workers"],
        )

        # Model
        self.model = DegradationAwareSpeechEnhancer(
            expert_dim        = self.cfg["expert_dim"],
            num_expert_blocks = self.cfg["num_expert_blocks"],
            decoder_dim       = self.cfg["decoder_dim"],
            num_upsample      = self.cfg["num_upsample"],
            dropout           = self.cfg["dropout"],
        ).to(self.device)

        # Loss
        self.criterion = CompositeLoss(
            lambda_l1    = self.cfg["lambda_l1"],
            lambda_stft  = self.cfg["lambda_stft"],
            lambda_sisnr = self.cfg["lambda_sisnr"],
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Optional resume
        if self.cfg["resume_from"]:
            self._load_checkpoint(self.cfg["resume_from"])

    # ==================================================================
    # Public API
    # ==================================================================
    def train(self):
        self._run_stage(
            stage       = 1,
            num_epochs  = self.cfg["stage1_epochs"],
            lr          = self.cfg["stage1_lr"],
        )
        self._run_stage(
            stage       = 2,
            num_epochs  = self.cfg["stage2_epochs"],
            lr          = self.cfg["stage2_lr"],
        )
        self.writer.close()
        print("[Trainer] Training complete.")

    # ==================================================================
    # Stage runner
    # ==================================================================
    def _run_stage(self, stage: int, num_epochs: int, lr: float):
        print(f"\n{'='*60}")
        print(f"  STAGE {stage}  |  {num_epochs} epochs  |  LR={lr}")
        print(f"{'='*60}")

        # Configure model parameters for this stage
        if stage == 1:
            self.model.configure_stage1()
        else:
            self.model.configure_stage2(self.cfg["unfreeze_top_n"])

        # Build optimizer over trainable params only
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-4,
        )

        # Scheduler
        total_steps  = num_epochs * len(self.train_loader)
        warmup_steps = int(total_steps * self.cfg["warmup_ratio"])
        scheduler    = _build_scheduler(optimizer, warmup_steps, total_steps)

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(optimizer, scheduler, epoch, stage)
            val_loss   = self._val_epoch(epoch, stage)

            print(f"[Stage {stage} | Epoch {epoch:03d}]  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(f"best_stage{stage}.pt", epoch, val_loss)
                print(f"  → New best val loss: {val_loss:.4f}  (saved)")

            # Always save latest
            self._save_checkpoint(f"latest_stage{stage}.pt", epoch, val_loss)

    # ==================================================================
    # Single training epoch
    # ==================================================================
    def _train_epoch(self, optimizer, scheduler, epoch: int, stage: int) -> float:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            degraded = batch["degraded"].to(self.device)   # (B, T)
            clean    = batch["clean"].to(self.device)       # (B, T)
            labels   = batch["label"].to(self.device)       # (B,)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(self.device.type == "cuda")):
                enhanced = self.model(degraded, labels)         # (B, T)
                loss, sub = self.criterion(enhanced, clean)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.cfg["grad_clip"]
            )
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss       += loss.item()
            self.global_step += 1

            # TensorBoard
            self.writer.add_scalar("train/loss_total", loss.item(), self.global_step)
            for k, v in sub.items():
                self.writer.add_scalar(f"train/loss_{k}", v, self.global_step)
            self.writer.add_scalar(
                "train/lr", optimizer.param_groups[0]["lr"], self.global_step
            )

            # Console log
            if (step + 1) % self.cfg["log_interval"] == 0:
                elapsed = time.time() - t0
                print(f"  [S{stage} E{epoch} | {step+1}/{len(self.train_loader)}]  "
                      f"loss={loss.item():.4f}  "
                      f"l1={sub['l1']:.4f}  stft={sub['stft']:.4f}  "
                      f"sisnr={sub['sisnr']:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                      f"t={elapsed:.1f}s")
                t0 = time.time()

        return total_loss / len(self.train_loader)

    # ==================================================================
    # Validation epoch
    # ==================================================================
    @torch.no_grad()
    def _val_epoch(self, epoch: int, stage: int) -> float:
        self.model.eval()
        total_loss = 0.0
        per_type   = {name: {"loss": 0.0, "count": 0} for name in IDX_TO_NAME.values()}

        for batch in self.val_loader:
            degraded = batch["degraded"].to(self.device)
            clean    = batch["clean"].to(self.device)
            labels   = batch["label"].to(self.device)

            with autocast(enabled=(self.device.type == "cuda")):
                enhanced = self.model(degraded, labels)
                loss, _  = self.criterion(enhanced, clean)

            total_loss += loss.item()

            # Per-type breakdown
            for idx, name in IDX_TO_NAME.items():
                mask = (labels == idx)
                if mask.any():
                    sub_loss, _ = self.criterion(enhanced[mask], clean[mask])
                    per_type[name]["loss"]  += sub_loss.item() * mask.sum().item()
                    per_type[name]["count"] += mask.sum().item()

        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("val/loss_total", avg_loss, epoch)

        for name, info in per_type.items():
            if info["count"] > 0:
                per_avg = info["loss"] / info["count"]
                self.writer.add_scalar(f"val/loss_{name}", per_avg, epoch)

        return avg_loss

    # ==================================================================
    # Checkpoint helpers
    # ==================================================================
    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        path = self.ckpt_dir / filename
        torch.save({
            "epoch"       : epoch,
            "global_step" : self.global_step,
            "model_state" : self.model.state_dict(),
            "val_loss"    : val_loss,
            "cfg"         : self.cfg,
        }, path)

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.global_step   = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[Trainer] Resumed from {path}  (epoch {ckpt['epoch']}, "
              f"val_loss={self.best_val_loss:.4f})")