"""
decoder.py
----------
Reconstructs a time-domain waveform from (B, T', 768) latent features
produced by the backbone + expert stack.

wav2vec2-base downsamples input by ~320× (stride product of CNN extractor).
At 16 kHz that means 1 frame ≈ 20 ms, so T' = ceil(T / 320).

The decoder must upsample T' back to T (the original waveform length).

Architecture:
    (B, T', 768)
      │
      ├─ Linear(768 → decoder_dim)                           # feature projection
      │
      ├─ N × ResidualBlock1D(decoder_dim)                    # temporal refinement
      │
      ├─ ConvTranspose1d upsampler × K stages                # learned upsampling
      │   each stage: ConvTranspose1d(C, C//2, k=4, stride=2) → 2× upsample
      │   + residual Conv1d + GELU
      │
      ├─ final Conv1d(decoder_dim // 2^K → 1, k=7)          # waveform projection
      │
      └─ tanh activation                                     # bound to [-1, 1]

Upsampling factor: 2^K stages.  K=8 gives ×256, then a final interp to
exact length handles any rounding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """
    One transposed-conv upsampling stage (×2 resolution).
    Input/output in channels-first format: (B, C, T).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=2, padding=kernel_size // 2 - 1
        )
        self.refine = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.GroupNorm(max(1, out_channels // 8), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.refine(x)
        x = self.norm(x)
        return x


class WaveformDecoder(nn.Module):
    """
    Decodes (B, T', hidden_dim) → (B, target_len) waveform.

    Args:
        hidden_dim      : dimension from backbone/experts (768)
        decoder_dim     : internal width for upsampling pipeline (512)
        num_res_blocks  : temporal residual blocks before upsampling (2)
        num_upsample    : number of ×2 upsample stages (8 → ×256 total)
        dropout         : dropout in residual blocks
    """

    def __init__(self, hidden_dim: int = 768, decoder_dim: int = 512,
                 num_res_blocks: int = 2, num_upsample: int = 8,
                 dropout: float = 0.05):
        super().__init__()

        # 1. Input projection: hidden_dim → decoder_dim
        self.input_proj = nn.Linear(hidden_dim, decoder_dim)
        self.input_norm = nn.LayerNorm(decoder_dim)

        # 2. Temporal residual blocks (channels-last, reuse pattern from expert.py)
        self.res_blocks = nn.ModuleList([
            _ResBlock1D(decoder_dim, dropout=dropout)
            for _ in range(num_res_blocks)
        ])

        # 3. Upsampling stages: decoder_dim → decoder_dim // 2^K
        upsample_stages = []
        in_ch = decoder_dim
        for _ in range(num_upsample):
            out_ch = max(in_ch // 2, 16)
            upsample_stages.append(UpsampleBlock(in_ch, out_ch))
            in_ch = out_ch
        self.upsample_stages = nn.ModuleList(upsample_stages)
        self.final_channels  = in_ch

        # 4. Waveform projection: final_channels → 1
        self.output_conv = nn.Sequential(
            nn.Conv1d(self.final_channels, self.final_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(self.final_channels, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            x          : (B, T', hidden_dim)  latent features
            target_len : desired output length in samples (= original waveform length)

        Returns:
            wav        : (B, target_len)  reconstructed waveform
        """
        # --- Input projection (channels-last) ---
        h = self.input_norm(self.input_proj(x))    # (B, T', decoder_dim)

        # --- Residual blocks ---
        for block in self.res_blocks:
            h = block(h)                           # (B, T', decoder_dim)

        # --- Switch to channels-first for Conv1d ---
        h = h.transpose(1, 2)                      # (B, decoder_dim, T')

        # --- Progressive upsampling ---
        for stage in self.upsample_stages:
            h = stage(h)                           # (B, C_i, T' * 2^i)

        # --- Waveform projection ---
        wav = self.output_conv(h)                  # (B, 1, T_up)
        wav = wav.squeeze(1)                       # (B, T_up)

        # --- Exact-length alignment (handles rounding from strides) ---
        if wav.shape[-1] != target_len:
            wav = F.interpolate(
                wav.unsqueeze(1), size=target_len, mode="linear", align_corners=False
            ).squeeze(1)

        return wav   # (B, target_len)


# ------------------------------------------------------------------
# Internal helper (keep decoder self-contained)
# ------------------------------------------------------------------
class _ResBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x).transpose(1, 2)
        h = self.act(self.conv1(h)).transpose(1, 2)
        h = self.drop(h)
        h = self.norm2(h).transpose(1, 2)
        h = self.conv2(h).transpose(1, 2)
        return self.act(h + x)