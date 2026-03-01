"""
Lightweight reconstruction module converting enhanced latent features back to waveform.
Includes ConvTranspose1d and optional STFT consistency.
"""

import torch


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.transposed = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, features):
        # features: (batch, channels, time)
        waveform = self.transposed(features)
        return waveform

    def stft_consistency(self, waveform):
        """Optional STFT loss / consistency enforcement."""
        pass
