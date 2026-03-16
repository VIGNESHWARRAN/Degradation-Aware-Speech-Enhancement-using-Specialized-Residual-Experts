"""
losses.py
---------
Composite loss for speech enhancement:

    L_total = λ1 * L_L1 + λ2 * L_STFT + λ3 * L_SISDR

    L_L1    : time-domain L1 — global amplitude fidelity
    L_STFT  : multi-resolution STFT magnitude loss — spectral fidelity
    L_SISDR : Scale-Invariant SDR — perceptual signal quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Individual loss components
# ------------------------------------------------------------------

class L1WaveformLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Sum of magnitude-spectrum L1 losses at multiple FFT resolutions.
    Captures both fine-grained and coarse spectral structure.

    Args:
        fft_sizes   : list of FFT window sizes
        hop_sizes   : corresponding hop lengths
        win_lengths : corresponding window lengths
    """

    def __init__(
        self,
        fft_sizes   = (512,  1024, 2048),
        hop_sizes   = (120,  240,  480),
        win_lengths = (480,  960,  1920),
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes   = fft_sizes
        self.hop_sizes   = hop_sizes
        self.win_lengths = win_lengths

    def _stft_magnitude(self, x: torch.Tensor, n_fft: int,
                        hop: int, win_len: int) -> torch.Tensor:
        window = torch.hann_window(win_len, device=x.device)
        stft   = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=win_len,
            window=window, return_complex=True
        )
        return stft.abs()   # (B, F, T')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft, hop, win_len in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_mag   = self._stft_magnitude(pred,   n_fft, hop, win_len)
            target_mag = self._stft_magnitude(target, n_fft, hop, win_len)

            # Spectral convergence + log magnitude loss
            sc   = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-8)
            logm = F.l1_loss(torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7))
            loss = loss + sc + logm

        return loss / len(self.fft_sizes)


class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio loss.
    Higher SI-SNR = better, so we return the negative.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Zero-mean
        pred   = pred   - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        # SI-SNR
        dot        = (pred * target).sum(dim=-1, keepdim=True)
        target_pow = (target * target).sum(dim=-1, keepdim=True) + 1e-8
        proj       = dot / target_pow * target          # target projection of pred
        noise      = pred - proj

        si_snr = 10 * torch.log10(
            (proj * proj).sum(dim=-1) / ((noise * noise).sum(dim=-1) + 1e-8) + 1e-8
        )
        return -si_snr.mean()   # negative → we minimise


# ------------------------------------------------------------------
# Composite loss
# ------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    Weighted combination of L1, multi-res STFT, and SI-SNR losses.

    Args:
        lambda_l1    : weight for L1 waveform loss     (default 0.5)
        lambda_stft  : weight for multi-res STFT loss  (default 1.0)
        lambda_sisnr : weight for SI-SNR loss           (default 0.5)
    """

    def __init__(self, lambda_l1: float = 0.5,
                 lambda_stft: float = 1.0,
                 lambda_sisnr: float = 0.5):
        super().__init__()
        self.lambda_l1    = lambda_l1
        self.lambda_stft  = lambda_stft
        self.lambda_sisnr = lambda_sisnr

        self.l1_loss   = L1WaveformLoss()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.sisnr_loss = SISNRLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1    = self.l1_loss(pred, target)
        stft  = self.stft_loss(pred, target)
        sisnr = self.sisnr_loss(pred, target)

        total = (self.lambda_l1    * l1
               + self.lambda_stft  * stft
               + self.lambda_sisnr * sisnr)

        return total, {"l1": l1.item(), "stft": stft.item(), "sisnr": sisnr.item()}