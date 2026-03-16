"""
backbone.py
-----------
Wraps facebook/wav2vec2-base as a frozen / partially-unfrozen feature encoder.

wav2vec2-base architecture:
    - Feature Extractor  : 7-layer CNN  → produces frame-level features (512-d)
    - Transformer        : 12 encoder layers → contextual features (768-d)

We expose:
    - encode(waveform)  → (B, T', 768) hidden states from the last transformer layer
    - freeze() / unfreeze_top_n(n) for staged training
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


BACKBONE_NAME = "facebook/wav2vec2-base"
HIDDEN_DIM    = 768   # output dim of wav2vec2-base transformer


class Wav2Vec2Backbone(nn.Module):
    """
    Wav2Vec2 encoder wrapper.

    Args:
        pretrained_name : HuggingFace model identifier
        freeze_feature_extractor : always freeze the CNN feature extractor
                                   (standard practice — it's a fixed spectrogram front-end)
    """

    def __init__(self, pretrained_name: str = BACKBONE_NAME,
                 freeze_feature_extractor: bool = True):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(pretrained_name)
        self.hidden_dim = HIDDEN_DIM

        # Always freeze the CNN feature extractor
        if freeze_feature_extractor:
            self.model.feature_extractor._freeze_parameters()

    # ------------------------------------------------------------------
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, T)  raw waveform at 16 kHz, values in [-1, 1]

        Returns:
            hidden   : (B, T', 768)  contextual features
                       T' ≈ T / 320  (wav2vec2 stride is ~20 ms at 16 kHz)
        """
        outputs = self.model(input_values=waveform, output_hidden_states=False)
        return outputs.last_hidden_state   # (B, T', 768)

    # ------------------------------------------------------------------
    def freeze_all(self):
        """Freeze entire backbone (Stage 1)."""
        for p in self.model.parameters():
            p.requires_grad = False
        # CNN extractor is always frozen
        self.model.feature_extractor._freeze_parameters()
        print("[Backbone] All parameters frozen.")

    # ------------------------------------------------------------------
    def unfreeze_top_n_transformer_layers(self, n: int = 4):
        """
        Unfreeze the top-N transformer encoder layers (Stage 2).
        CNN feature extractor remains frozen.

        Args:
            n : number of top transformer layers to unfreeze (max 12 for wav2vec2-base)
        """
        # First freeze everything
        self.freeze_all()

        layers = self.model.encoder.layers
        total  = len(layers)
        unfreeze_from = max(0, total - n)

        for layer in layers[unfreeze_from:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Also unfreeze the final layer norm
        for p in self.model.encoder.layer_norm.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Backbone] Unfroze top {n}/{total} transformer layers. "
              f"Trainable backbone params: {trainable:,}")

    # ------------------------------------------------------------------
    def unfreeze_all(self):
        """Fully unfreeze transformer (CNN extractor stays frozen)."""
        for p in self.model.encoder.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Backbone] Fully unfrozen transformer. Trainable: {trainable:,}")