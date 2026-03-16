"""
model.py
--------
Full end-to-end Degradation-Aware Speech Enhancement model.

Pipeline:
    waveform (B, T)
        │
        ▼
    Wav2Vec2Backbone        → (B, T', 768)   contextual features
        │
        ▼
    ExpertSet (hard routing) → (B, T', 768)  degradation-specific refinement
        │
        ▼
    WaveformDecoder          → (B, T)        reconstructed clean waveform
"""

import torch
import torch.nn as nn

from models.backbone import Wav2Vec2Backbone, HIDDEN_DIM
from models.expert   import ExpertSet
from models.decoder  import WaveformDecoder


class DegradationAwareSpeechEnhancer(nn.Module):
    """
    Full model: backbone → expert routing → decoder.

    Args:
        expert_dim      : internal width of each residual expert  (default 512)
        num_expert_blocks : ResidualBlock1D layers per expert     (default 4)
        decoder_dim     : internal width of the decoder           (default 512)
        num_upsample    : ×2 upsample stages in decoder           (default 8 → ×256)
        dropout         : shared dropout rate
    """

    def __init__(
        self,
        expert_dim        : int = 512,
        num_expert_blocks : int = 4,
        decoder_dim       : int = 512,
        num_upsample      : int = 8,
        dropout           : float = 0.1,
    ):
        super().__init__()

        self.backbone = Wav2Vec2Backbone()
        self.experts  = ExpertSet(
            input_dim  = HIDDEN_DIM,
            expert_dim = expert_dim,
            num_blocks = num_expert_blocks,
            dropout    = dropout,
        )
        self.decoder  = WaveformDecoder(
            hidden_dim   = HIDDEN_DIM,
            decoder_dim  = decoder_dim,
            num_upsample = num_upsample,
            dropout      = dropout / 2,
        )

    # ------------------------------------------------------------------
    def forward(self, waveform: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, T)   degraded input at 16 kHz
            labels   : (B,)     degradation expert index (0=noise,1=reverb,2=device)

        Returns:
            enhanced : (B, T)   estimated clean waveform
        """
        target_len = waveform.shape[-1]

        # 1. Encode with wav2vec2
        features = self.backbone(waveform)          # (B, T', 768)

        # 2. Expert refinement (hard routing by label)
        refined   = self.experts(features, labels)  # (B, T', 768)

        # 3. Decode to waveform
        enhanced  = self.decoder(refined, target_len)  # (B, T)

        return enhanced

    # ------------------------------------------------------------------
    # Staged training helpers (called by Trainer)
    # ------------------------------------------------------------------
    def configure_stage1(self):
        """Stage 1: freeze backbone entirely, train experts + decoder."""
        self.backbone.freeze_all()
        for p in self.experts.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True
        self._report_trainable("Stage 1")

    def configure_stage2(self, unfreeze_top_n: int = 4):
        """Stage 2: unfreeze top-N backbone transformer layers + full experts + decoder."""
        self.backbone.unfreeze_top_n_transformer_layers(unfreeze_top_n)
        for p in self.experts.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True
        self._report_trainable(f"Stage 2 (backbone top-{unfreeze_top_n})")

    def _report_trainable(self, tag: str):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model | {tag}]  trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")