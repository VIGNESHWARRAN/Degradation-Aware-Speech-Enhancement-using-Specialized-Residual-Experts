"""
expert.py
---------
Residual Expert Blocks for degradation-specific processing.

Each expert is a stack of 1-D ResNet blocks operating on the
(B, T', 768) hidden states produced by the wav2vec2 backbone.

Architecture per expert:
    Input  (B, T', 768)
      │
      ├─ projection: Linear(768 → expert_dim)
      │
      ├─ N × ResidualBlock1D(expert_dim)
      │      ├─ LayerNorm
      │      ├─ Conv1d(expert_dim, expert_dim, k=3, pad=1)
      │      ├─ GELU
      │      ├─ Conv1d(expert_dim, expert_dim, k=3, pad=1)
      │      └─ skip connection
      │
      └─ output projection: Linear(expert_dim → 768)   (back to backbone dim)

Three experts: noise(0), reverb(1), device(2)
"""

import torch
import torch.nn as nn
from typing import List


# ------------------------------------------------------------------
# Building block
# ------------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    """
    1-D convolutional residual block with pre-activation (LayerNorm → Conv → GELU).

    Input/output shape: (B, T', channels)   [channels-last convention]
    Internally operates as (B, channels, T') for Conv1d, then transposes back.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)

        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T', C)
        residual = x

        # Pre-norm → Conv path
        h = self.norm1(x)                        # (B, T', C)
        h = h.transpose(1, 2)                    # (B, C, T')
        h = self.act(self.conv1(h))
        h = self.dropout(h)
        h = h.transpose(1, 2)                    # (B, T', C)

        h = self.norm2(h)
        h = h.transpose(1, 2)                    # (B, C, T')
        h = self.conv2(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)                    # (B, T', C)

        return self.act(h + residual)


# ------------------------------------------------------------------
# Single Expert
# ------------------------------------------------------------------
class ResidualExpert(nn.Module):
    """
    One degradation-specific residual expert.

    Args:
        input_dim   : dimension from backbone (768 for wav2vec2-base)
        expert_dim  : internal channel width of the expert
        num_blocks  : number of ResidualBlock1D layers
        dropout     : dropout rate inside residual blocks
        name        : human-readable label ("noise" / "reverb" / "device")
    """

    def __init__(self, input_dim: int = 768, expert_dim: int = 512,
                 num_blocks: int = 4, dropout: float = 0.1, name: str = "expert"):
        super().__init__()
        self.name = name

        self.input_proj  = nn.Linear(input_dim, expert_dim)
        self.blocks      = nn.ModuleList([
            ResidualBlock1D(expert_dim, kernel_size=3, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(expert_dim, input_dim)
        self.out_norm    = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T', input_dim)   contextual features from backbone

        Returns:
            out : (B, T', input_dim)  expert-refined features
        """
        h = self.input_proj(x)           # (B, T', expert_dim)
        for block in self.blocks:
            h = block(h)
        h = self.output_proj(h)          # (B, T', input_dim)
        return self.out_norm(h + x)      # residual skip over entire expert


# ------------------------------------------------------------------
# Expert Set (container)
# ------------------------------------------------------------------
class ExpertSet(nn.Module):
    """
    Container for all degradation experts.
    Supports hard routing: given a label index, forward through the
    corresponding expert only.

    Experts are indexed:  0=noise, 1=reverb, 2=device
    """

    EXPERT_NAMES = ["noise", "reverb", "device"]

    def __init__(self, input_dim: int = 768, expert_dim: int = 512,
                 num_blocks: int = 4, dropout: float = 0.1):
        super().__init__()

        self.experts = nn.ModuleList([
            ResidualExpert(
                input_dim  = input_dim,
                expert_dim = expert_dim,
                num_blocks = num_blocks,
                dropout    = dropout,
                name       = name,
            )
            for name in self.EXPERT_NAMES
        ])

        print(f"[ExpertSet] Created {len(self.experts)} experts: {self.EXPERT_NAMES}")
        params = sum(p.numel() for e in self.experts for p in e.parameters())
        print(f"[ExpertSet] Total expert parameters: {params:,}")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Hard routing: each sample in the batch is processed by its
        assigned expert (determined by ground-truth label).

        Args:
            x      : (B, T', D)   backbone features
            labels : (B,)         integer expert indices  (0, 1, or 2)

        Returns:
            out    : (B, T', D)   expert-processed features
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)

        for expert_idx, expert in enumerate(self.experts):
            mask = (labels == expert_idx)   # (B,) boolean
            if mask.any():
                out[mask] = expert(x[mask])

        return out