"""
training.py — v3
----------------
Key changes:
  - Stage 1 uses MSE-dominant loss (bootstraps decoder amplitude)
  - Stage 2 uses SI-SNR + MSE + STFT (perceptual quality)
  - Loss switches automatically between stages via loss.set_stage()
  - Stage 2 LR raised to 3e-5 (was 1e-5 — too low to update backbone)
  - Early stop patience raised to 10
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
except AttributeError:
    class SummaryWriter:
        def __init__(self, log_dir=None, **kw):
            os.makedirs(log_dir, exist_ok=True) if log_dir else None
            print("[TensorBoard] No-op writer active.")
        def add_scalar(self, *a, **kw): pass
        def close(self): pass

from models.model  import DegradationAwareSpeechEnhancer
from models.losses import CompositeLoss
from data.dataset  import build_dataloaders, DEGRADATION_TO_IDX

IDX_TO_NAME = {v: k for k, v in DEGRADATION_TO_IDX.items()}


def _build_scheduler(optimizer, warmup_steps, total_steps):
    import math
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = {
            "data_root"           : "data/final_processed",
            "output_dir"          : "outputs/experiment_03",
            "expert_dim"          : 256,
            "num_expert_blocks"   : 3,
            "decoder_dim"         : 256,
            "num_upsample"        : 8,
            "dropout"             : 0.05,
            "unfreeze_top_n"      : 4,
            "batch_size"          : 8,
            "num_workers"         : 0,
            "stage1_epochs"       : 40,
            "stage2_epochs"       : 30,
            "stage1_lr"           : 3e-4,
            "stage2_lr"           : 3e-5,    # raised from 1e-5
            "warmup_ratio"        : 0.05,
            "grad_clip"           : 1.0,
            "early_stop_patience" : 10,      # raised from 7
            "seed"                : 42,
            "resume_from"         : None,
            "log_interval"        : 20,
            **cfg,
        }

        torch.manual_seed(self.cfg["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")

        self.output_dir = Path(self.cfg["output_dir"])
        self.ckpt_dir   = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer     = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            data_root   = self.cfg["data_root"],
            batch_size  = self.cfg["batch_size"],
            num_workers = self.cfg["num_workers"],
        )

        self.model = DegradationAwareSpeechEnhancer(
            expert_dim        = self.cfg["expert_dim"],
            num_expert_blocks = self.cfg["num_expert_blocks"],
            decoder_dim       = self.cfg["decoder_dim"],
            num_upsample      = self.cfg["num_upsample"],
            dropout           = self.cfg["dropout"],
        ).to(self.device)

        # Stage-aware loss — starts at stage 1 (MSE-dominant)
        self.criterion  = CompositeLoss(stage=1)
        self.scaler     = GradScaler(enabled=(self.device.type == "cuda"))
        self.global_step = 0

        if self.cfg["resume_from"]:
            self._load_checkpoint(self.cfg["resume_from"])

    def train(self):
        # Stage 1: MSE loss bootstraps decoder
        self.criterion.set_stage(1)
        self._run_stage(stage=1, num_epochs=self.cfg["stage1_epochs"],
                        lr=self.cfg["stage1_lr"])

        # Stage 2: switch to perceptual loss
        self.criterion.set_stage(2)
        self._run_stage(stage=2, num_epochs=self.cfg["stage2_epochs"],
                        lr=self.cfg["stage2_lr"])

        self.writer.close()
        print("[Trainer] Training complete.")

    def _run_stage(self, stage, num_epochs, lr):
        print(f"\n{'='*65}")
        print(f"  STAGE {stage}  |  {num_epochs} epochs  |  LR={lr:.1e}")
        loss_desc = "MSE+STFT" if stage == 1 else "SISNR+STFT+MSE"
        print(f"  Loss: {loss_desc}")
        print(f"{'='*65}")

        if stage == 1:
            self.model.configure_stage1()
        else:
            self.model.configure_stage2(self.cfg["unfreeze_top_n"])

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=1e-5,
        )
        total_steps  = num_epochs * len(self.train_loader)
        warmup_steps = int(total_steps * self.cfg["warmup_ratio"])
        scheduler    = _build_scheduler(optimizer, warmup_steps, total_steps)

        best_val_loss  = float("inf")
        patience_count = 0
        patience       = self.cfg["early_stop_patience"]

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(optimizer, scheduler, epoch, stage)
            val_loss   = self._val_epoch(epoch, stage)

            print(f"[S{stage} | E{epoch:03d}/{num_epochs}]  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"best={best_val_loss:.6f}  patience={patience_count}/{patience}")

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                self._save_checkpoint(f"best_stage{stage}.pt", epoch, val_loss)
                print(f"  ✓ New best Stage {stage}: {val_loss:.6f}")
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

            self._save_checkpoint(f"latest_stage{stage}.pt", epoch, val_loss)

    def _train_epoch(self, optimizer, scheduler, epoch, stage):
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            degraded = batch["degraded"].to(self.device)
            clean    = batch["clean"].to(self.device)
            labels   = batch["label"].to(self.device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(self.device.type == "cuda")):
                enhanced  = self.model(degraded, labels)
                loss, sub = self.criterion(enhanced, clean)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.cfg["grad_clip"])
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss       += loss.item()
            self.global_step += 1

            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            for k, v in sub.items():
                self.writer.add_scalar(f"train/{k}", v, self.global_step)

            if (step + 1) % self.cfg["log_interval"] == 0:
                elapsed = time.time() - t0
                print(f"  [S{stage} E{epoch} {step+1}/{len(self.train_loader)}]  "
                      f"loss={loss.item():.6f}  mse={sub['mse']:.6f}  "
                      f"stft={sub['stft']:.4f}  sisnr={sub['sisnr']:.3f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.1e}  {elapsed:.1f}s")
                t0 = time.time()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _val_epoch(self, epoch, stage):
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            degraded = batch["degraded"].to(self.device)
            clean    = batch["clean"].to(self.device)
            labels   = batch["label"].to(self.device)
            with autocast(enabled=(self.device.type == "cuda")):
                enhanced = self.model(degraded, labels)
                loss, _  = self.criterion(enhanced, clean)
            total_loss += loss.item()
        avg = total_loss / len(self.val_loader)
        self.writer.add_scalar("val/loss", avg, epoch)
        return avg

    def _save_checkpoint(self, filename, epoch, val_loss):
        torch.save({
            "epoch": epoch, "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "val_loss": val_loss, "cfg": self.cfg,
        }, self.ckpt_dir / filename)

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=False)
        self.global_step = ckpt.get("global_step", 0)
        print(f"[Trainer] Resumed from {path}")