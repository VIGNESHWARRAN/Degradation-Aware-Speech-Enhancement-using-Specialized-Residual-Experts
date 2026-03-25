"""
dataset.py
----------
PyTorch Dataset for Degradation-Aware Speech Enhancement.

Directory layout:
    root/
      train/  device/ noise/ reverb/  → each contains sample_XXXXXX/{clean.wav, degraded.wav, metadata.json}
      val/
      test/

Cap sizes (balanced across degradation types):
    train : 3000 samples  (1000 per type)
    val   :  600 samples  ( 200 per type)
    test  :  600 samples  ( 200 per type)

This prevents overfitting on the 18 GB train set and keeps training
fast enough on an RTX 3050 (4 GB VRAM).
"""

import os
import json
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DEGRADATION_TO_IDX = {"noise": 0, "reverb": 1, "device": 2}
IDX_TO_NAME        = {v: k for k, v in DEGRADATION_TO_IDX.items()}
TARGET_SR          = 16_000               # wav2vec2 requires 16 kHz
CLIP_SECONDS       = 4                    # seconds per training clip
MAX_AUDIO_LEN      = CLIP_SECONDS * TARGET_SR   # 64 000 samples

# Balanced caps per degradation type per split
SPLIT_CAPS = {
    "train": 1000,   # 3000 total
    "val"  :  200,   #  600 total
    "test" :  200,   #  600 total
}


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class SpeechDegradationDataset(Dataset):
    """
    Loads (degraded, clean, degradation_label) triplets.

    Args:
        root_dir    : path to split root, e.g. .../final_processed/train
        max_len     : clip length in samples (crops / pads to this)
        augment     : random crop + random gain augmentation during training
        cap_per_type: max samples per degradation type (None = no cap)
        seed        : random seed for reproducible subsampling
    """

    def __init__(self, root_dir: str, max_len: int = MAX_AUDIO_LEN,
                 augment: bool = False, cap_per_type: int = None, seed: int = 42):
        self.root_dir     = root_dir
        self.max_len      = max_len
        self.augment      = augment
        self.cap_per_type = cap_per_type
        self.seed         = seed
        self.samples      = []   # list of (degraded_path, clean_path, label_idx)
        self._scan()

    # ------------------------------------------------------------------
    def _scan(self):
        rng = random.Random(self.seed)

        for deg_type in ("noise", "reverb", "device"):
            deg_dir = os.path.join(self.root_dir, deg_type)
            if not os.path.isdir(deg_dir):
                continue

            candidates = []
            for sample_name in sorted(os.listdir(deg_dir)):
                sample_dir = os.path.join(deg_dir, sample_name)
                if not os.path.isdir(sample_dir):
                    continue

                degraded = os.path.join(sample_dir, "degraded.wav")
                clean    = os.path.join(sample_dir, "clean.wav")
                meta     = os.path.join(sample_dir, "metadata.json")

                if not (os.path.exists(degraded) and os.path.exists(clean)):
                    continue

                # Prefer metadata label, fall back to folder name
                label = deg_type
                if os.path.exists(meta):
                    with open(meta) as f:
                        info = json.load(f)
                    label = info.get("degradation_type", deg_type)

                label_idx = DEGRADATION_TO_IDX.get(label, -1)
                if label_idx == -1:
                    continue

                candidates.append((degraded, clean, label_idx))

            # Subsample to cap
            if self.cap_per_type and len(candidates) > self.cap_per_type:
                rng.shuffle(candidates)
                candidates = candidates[:self.cap_per_type]

            self.samples.extend(candidates)

        # Shuffle the final combined list
        rng.shuffle(self.samples)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load wav → mono → 16 kHz → 1-D tensor (T,)."""
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        return wav.squeeze(0)   # (T,)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        degraded_path, clean_path, label_idx = self.samples[idx]

        degraded = self._load_audio(degraded_path)
        clean    = self._load_audio(clean_path)

        # ── Aligned crop / pad ───────────────────────────────────────
        T = min(degraded.shape[0], clean.shape[0])
        degraded = degraded[:T]
        clean    = clean[:T]

        if T >= self.max_len:
            if self.augment:
                start = torch.randint(0, T - self.max_len + 1, (1,)).item()
            else:
                start = 0
            degraded = degraded[start: start + self.max_len]
            clean    = clean   [start: start + self.max_len]
        else:
            pad = self.max_len - T
            degraded = F.pad(degraded, (0, pad))
            clean    = F.pad(clean,    (0, pad))

        # ── Normalize jointly to preserve clean/degraded amplitude ratio ──
        # CRITICAL: do NOT normalize clean and degraded independently.
        # The dataset generator normalizes them separately (both to max=1),
        # which destroys amplitude relationships. We re-normalize jointly
        # using the degraded signal's scale so the model learns to map
        # degraded amplitude → clean amplitude correctly.
        max_val = degraded.abs().max().clamp(min=1e-8)
        degraded = degraded / max_val
        clean    = clean    / max_val   # same scale — preserves ratio

        # ── Random gain augmentation (applied JOINTLY after normalization) ──
        if self.augment:
            gain = torch.empty(1).uniform_(0.5, 1.0).item()
            degraded = degraded * gain
            clean    = clean    * gain

        return {
            "degraded": degraded,                              # (max_len,)
            "clean"   : clean,                                 # (max_len,)
            "label"   : torch.tensor(label_idx, dtype=torch.long),
        }


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------
def build_dataloaders(data_root: str, batch_size: int = 8, num_workers: int = 0,
                      max_len: int = MAX_AUDIO_LEN, seed: int = 42):
    """Returns train_loader, val_loader, test_loader with balanced caps."""

    def _make(split, augment):
        cap = SPLIT_CAPS.get(split)
        return SpeechDegradationDataset(
            root_dir     = os.path.join(data_root, split),
            max_len      = max_len,
            augment      = augment,
            cap_per_type = cap,
            seed         = seed,
        )

    train_ds = _make("train", augment=True)
    val_ds   = _make("val",   augment=False)
    test_ds  = _make("test",  augment=False)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=(num_workers > 0), drop_last=False)

    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"[Dataset] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader