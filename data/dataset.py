"""
dataset.py
----------
PyTorch Dataset for Degradation-Aware Speech Enhancement.

Directory layout expected:
    root/
      train/
        device/   sample_000001/ {clean.wav, degraded.wav, metadata.json}
        noise/    ...
        reverb/   ...
      val/
        ...
      test/
        ...

metadata.json schema:
    {"speaker_id": "39", "degradation_type": "device"}

degradation_type → expert index mapping:
    noise   → 0
    reverb  → 1
    device  → 2
"""

import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DEGRADATION_TO_IDX = {"noise": 0, "reverb": 1, "device": 2}
IDX_TO_NAME        = {v: k for k, v in DEGRADATION_TO_IDX.items()}
TARGET_SR = 16_000          # wav2vec2 requires 16 kHz
MAX_AUDIO_LEN = 4 * TARGET_SR   # 4-second clips (64 000 samples)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class SpeechDegradationDataset(Dataset):
    """
    Loads (degraded, clean, degradation_label) triplets.

    Args:
        root_dir  : path to split root, e.g. .../final_processed/train
        max_len   : maximum number of samples per clip (crops/pads to this)
        augment   : if True, apply random time-crop for training
    """

    def __init__(self, root_dir: str, max_len: int = MAX_AUDIO_LEN, augment: bool = False):
        self.root_dir = root_dir
        self.max_len  = max_len
        self.augment  = augment
        self.samples  = []          # list of (degraded_path, clean_path, label_idx)

        self._scan()

    # ------------------------------------------------------------------
    def _scan(self):
        for deg_type in ("noise", "reverb", "device"):
            deg_dir = os.path.join(self.root_dir, deg_type)
            if not os.path.isdir(deg_dir):
                continue

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
                    continue    # unknown type, skip

                self.samples.append((degraded, clean, label_idx))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load wav, resample if needed, convert to mono, return 1-D tensor."""
        wav, sr = torchaudio.load(path)

        # stereo → mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

        return wav.squeeze(0)   # (T,)

    # ------------------------------------------------------------------
    def _pad_or_crop(self, wav: torch.Tensor) -> torch.Tensor:
        """Deterministic crop/pad to self.max_len."""
        T = wav.shape[0]
        if T >= self.max_len:
            # random crop during training, fixed crop during eval
            if self.augment:
                start = torch.randint(0, T - self.max_len + 1, (1,)).item()
            else:
                start = 0
            return wav[start: start + self.max_len]
        else:
            # zero-pad on the right
            return torch.nn.functional.pad(wav, (0, self.max_len - T))

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        degraded_path, clean_path, label_idx = self.samples[idx]

        degraded = self._load_audio(degraded_path)
        clean    = self._load_audio(clean_path)

        # Use same crop window for both so they stay aligned
        T = degraded.shape[0]
        if T >= self.max_len:
            if self.augment:
                start = torch.randint(0, T - self.max_len + 1, (1,)).item()
            else:
                start = 0
            degraded = degraded[start: start + self.max_len]
            clean    = clean[start: min(start + self.max_len, clean.shape[0])]
        # pad
        if degraded.shape[0] < self.max_len:
            degraded = torch.nn.functional.pad(degraded, (0, self.max_len - degraded.shape[0]))
        if clean.shape[0] < self.max_len:
            clean    = torch.nn.functional.pad(clean,    (0, self.max_len - clean.shape[0]))

        return {
            "degraded"  : degraded,           # (max_len,)
            "clean"     : clean,              # (max_len,)
            "label"     : torch.tensor(label_idx, dtype=torch.long),
        }


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------
def build_dataloaders(data_root: str, batch_size: int = 8, num_workers: int = 4,
                      max_len: int = MAX_AUDIO_LEN):
    """
    Returns train_loader, val_loader, test_loader.
    """
    train_ds = SpeechDegradationDataset(
        os.path.join(data_root, "train"), max_len=max_len, augment=True)
    val_ds   = SpeechDegradationDataset(
        os.path.join(data_root, "val"),   max_len=max_len, augment=False)
    test_ds  = SpeechDegradationDataset(
        os.path.join(data_root, "test"),  max_len=max_len, augment=False)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, drop_last=False)

    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    print(f"[Dataset] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader