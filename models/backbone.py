"""
backbone.py — v4
----------------
CRITICAL FIX: Expose intermediate CNN feature extractor layers
for U-Net skip connections to the decoder.

wav2vec2-base CNN extractor has 7 conv layers producing features at
progressively smaller time resolutions. We expose 3 of these as skip
connections so the decoder can recover fine-grained temporal structure
that is lost in the transformer's semantic compression.

Skip connection outputs (channels, approx stride from input):
    skip_0: (512, ×2)   — layer 1 output
    skip_1: (512, ×4)   — layer 3 output  
    skip_2: (512, ×8)   — layer 5 output
    main:   (768, ×320) — transformer output (contextual features)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

BACKBONE_NAME = "facebook/wav2vec2-base"
HIDDEN_DIM    = 768
CNN_DIM       = 512   # wav2vec2 CNN feature dimension


class Wav2Vec2Backbone(nn.Module):
    def __init__(self, pretrained_name: str = BACKBONE_NAME,
                 freeze_feature_extractor: bool = True):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(pretrained_name)
        self.hidden_dim = HIDDEN_DIM
        self.cnn_dim    = CNN_DIM

        if freeze_feature_extractor:
            self.model.feature_extractor._freeze_parameters()

    def forward(self, waveform: torch.Tensor):
        """
        Args:
            waveform: (B, T) at 16kHz

        Returns:
            main_features : (B, T', 768)  transformer contextual features
            skip_features : list of 3 tensors (B, 512, T_i) CNN intermediate features
                           in channels-first format for direct use in decoder
        """
        # ── Extract CNN intermediate features via hooks ──────────────
        skip_features = []
        hooks = []

        # Tap layers 1, 3, 5 of the 7-layer CNN extractor
        tap_layers = [1, 3, 5]

        def make_hook(idx):
            def hook(module, input, output):
                # output shape: (B, C, T_i)
                skip_features.append(output.detach() if not self.training else output)
            return hook

        cnn_layers = self.model.feature_extractor.conv_layers
        for i in tap_layers:
            hooks.append(cnn_layers[i].register_forward_hook(make_hook(i)))

        # Forward pass
        outputs = self.model(input_values=waveform, output_hidden_states=False)
        main_features = outputs.last_hidden_state  # (B, T', 768)

        # Remove hooks
        for h in hooks:
            h.remove()

        return main_features, skip_features   # skip_features: list of 3 (B, 512, T_i)

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.feature_extractor._freeze_parameters()
        print("[Backbone] All parameters frozen.")

    def unfreeze_top_n_transformer_layers(self, n: int = 4):
        self.freeze_all()
        layers = self.model.encoder.layers
        total  = len(layers)
        for layer in layers[max(0, total-n):]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.model.encoder.layer_norm.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Backbone] Unfroze top {n}/{total} transformer layers. "
              f"Trainable: {trainable:,}")

    def unfreeze_all(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = True