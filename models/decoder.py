"""
decoder.py — v4 (U-Net with skip connections)
----------------------------------------------
CRITICAL FIX: U-Net architecture with skip connections from CNN layers.

Previous versions failed because they tried to decode waveforms purely
from wav2vec2's semantic transformer features — information is too compressed.

This version uses skip connections from the CNN extractor's intermediate
layers, which contain fine-grained temporal structure. The decoder
progressively upsamples while fusing skip features at matching resolutions.

Architecture:
    Transformer features (B, T', 768)
        │  
        ├─ Input projection → (B, T', 256)
        │
        ├─ Upsample ×2 → fuse with skip_2 (B, 512, T'×2)  
        ├─ Upsample ×2 → fuse with skip_1 (B, 512, T'×4)
        ├─ Upsample ×2 → fuse with skip_0 (B, 512, T'×8)
        ├─ Upsample ×2 → (B, 128, T'×16)
        ├─ Upsample ×2 → (B, 64,  T'×32)
        ├─ Upsample ×2 → (B, 32,  T'×64)
        ├─ Upsample ×2 → (B, 32,  T'×128)
        ├─ Upsample ×2 → (B, 32,  T'×256)
        │
        └─ Output Conv1d → (B, 1, T) → squeeze → (B, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleFuseBlock(nn.Module):
    """
    ×2 transposed conv upsample + optional skip connection fusion.
    skip_channels=0 means no skip (pure upsample).
    """

    def __init__(self, in_ch: int, out_ch: int, skip_channels: int = 0):
        super().__init__()
        self.up   = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4,
                                        stride=2, padding=1)
        fuse_in   = out_ch + skip_channels
        self.fuse = nn.Sequential(
            nn.Conv1d(fuse_in, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)                          # (B, out_ch, T×2)
        if skip is not None:
            # Align length (skip may be slightly different due to strides)
            if skip.shape[-1] != x.shape[-1]:
                skip = F.interpolate(skip, size=x.shape[-1], mode='linear',
                                     align_corners=False)
            x = torch.cat([x, skip], dim=1)     # (B, out_ch+skip_ch, T×2)
        return self.fuse(x)                     # (B, out_ch, T×2)


class WaveformDecoder(nn.Module):
    """
    U-Net decoder: transformer features + CNN skip connections → waveform.

    Args:
        hidden_dim   : transformer feature dim (768)
        decoder_dim  : internal decoder width (256)
        cnn_skip_dim : CNN skip feature dim (512 for wav2vec2)
        num_upsample : total ×2 stages (8 → ×256 to recover T from T')
    """

    def __init__(self, hidden_dim: int = 768, decoder_dim: int = 256,
                 cnn_skip_dim: int = 512, num_upsample: int = 8):
        super().__init__()

        # Project transformer features to decoder_dim
        self.input_proj = nn.Linear(hidden_dim, decoder_dim)
        self.input_norm = nn.LayerNorm(decoder_dim)

        # 8 upsample stages — skip connections at stages 0,1,2 (matching CNN taps)
        # CNN taps at strides ×2, ×4, ×8 from input
        # Decoder upsamples from ×320 back to ×1:
        #   stage 0: ×320 → ×160  ← skip from CNN stride ×8  (tap layer 5)
        #   stage 1: ×160 → ×80   ← skip from CNN stride ×4  (tap layer 3)
        #   stage 2: ×80  → ×40   ← skip from CNN stride ×2  (tap layer 1)
        #   stage 3..7: no skip, just upsample
        skip_chs = [cnn_skip_dim, cnn_skip_dim, cnn_skip_dim, 0, 0, 0, 0, 0]
        in_chs   = [decoder_dim, 256, 128, 64, 64, 32, 32, 32]
        out_chs  = [256,         128, 64,  64, 32, 32, 32, 32]

        self.stages = nn.ModuleList([
            UpsampleFuseBlock(in_chs[i], out_chs[i], skip_chs[i])
            for i in range(num_upsample)
        ])
        self.final_ch = out_chs[-1]

        # Output waveform projection
        self.output_conv = nn.Sequential(
            nn.Conv1d(self.final_ch, self.final_ch, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(self.final_ch, 1, 7, padding=3),
        )

    def forward(self, main_features: torch.Tensor,
                skip_features: list,
                target_len: int) -> torch.Tensor:
        """
        Args:
            main_features : (B, T', 768) from transformer
            skip_features : list of 3 tensors (B, 512, T_i) from CNN taps
                            ordered [tap1, tap3, tap5] i.e. coarse→fine stride
            target_len    : target waveform length

        Returns:
            wav : (B, target_len)
        """
        # Project to decoder_dim, switch to channels-first
        h = self.input_norm(self.input_proj(main_features))  # (B, T', 256)
        h = h.transpose(1, 2)                                # (B, 256, T')

        # skip_features from backbone: [tap1, tap3, tap5]
        # tap5 is closest to transformer (stride×8), tap1 is finest (stride×2)
        # We use them in order: stage0←tap5, stage1←tap3, stage2←tap1
        skips = [
            skip_features[2],   # tap layer 5, stride ×8  — used at stage 0
            skip_features[1],   # tap layer 3, stride ×4  — used at stage 1
            skip_features[0],   # tap layer 1, stride ×2  — used at stage 2
        ]

        for i, stage in enumerate(self.stages):
            skip = skips[i] if i < len(skips) else None
            h = stage(h, skip)

        wav = self.output_conv(h).squeeze(1)    # (B, T_up)

        if wav.shape[-1] != target_len:
            wav = F.interpolate(wav.unsqueeze(1), size=target_len,
                                mode='linear', align_corners=False).squeeze(1)
        return wav

    @staticmethod
    def post_process(enhanced: torch.Tensor,
                     degraded: torch.Tensor) -> torch.Tensor:
        """RMS amplitude matching at inference."""
        rms_deg = degraded.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
        rms_enh = enhanced.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
        return enhanced * (rms_deg / rms_enh)