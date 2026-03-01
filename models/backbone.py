"""
Pretrained encoder (wav2vec2) wrapper. Handles freezing and partial unfreezing.
"""

import torch


class Wav2Vec2Encoder(torch.nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base"):
        super().__init__()
        # placeholder for actual model loading
        self.model = None

    def forward(self, waveform):
        """Return latent representations for input waveform."""
        # expect waveform shape (batch, time)
        raise NotImplementedError

    def freeze(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_layers(self, num_layers=2):
        """Unfreeze last `num_layers` transformer layers."""
        pass
