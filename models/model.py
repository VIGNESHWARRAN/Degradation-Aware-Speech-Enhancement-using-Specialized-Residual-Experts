"""
model.py — v4
-------------
Updated to pass skip connections from backbone CNN to decoder.
"""

import torch
import torch.nn as nn

from models.backbone import Wav2Vec2Backbone, HIDDEN_DIM, CNN_DIM
from models.expert   import ExpertSet
from models.decoder  import WaveformDecoder


class DegradationAwareSpeechEnhancer(nn.Module):
    def __init__(self, expert_dim=256, num_expert_blocks=3,
                 decoder_dim=256, num_upsample=8, dropout=0.05):
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
            cnn_skip_dim = CNN_DIM,
            num_upsample = num_upsample,
        )

    def forward(self, waveform: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, T) degraded input at 16 kHz
            labels   : (B,) expert index

        Returns:
            enhanced : (B, T)
        """
        target_len = waveform.shape[-1]

        # Backbone: returns transformer features + CNN skip features
        main_features, skip_features = self.backbone(waveform)

        # Expert refinement (hard routing)
        refined = self.experts(main_features, labels)

        # U-Net decode with skip connections
        enhanced = self.decoder(refined, skip_features, target_len)

        return enhanced

    def configure_stage1(self):
        """Freeze backbone, train experts + decoder."""
        self.backbone.freeze_all()
        for p in self.experts.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True
        self._report_trainable("Stage 1")

    def configure_stage2(self, unfreeze_top_n: int = 4):
        """Unfreeze top-N transformer layers."""
        self.backbone.unfreeze_top_n_transformer_layers(unfreeze_top_n)
        for p in self.experts.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True
        self._report_trainable(f"Stage 2 (top-{unfreeze_top_n})")

    def _report_trainable(self, tag):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model | {tag}]  trainable: {trainable:,} / {total:,} "
              f"({100*trainable/total:.1f}%)")