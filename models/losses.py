"""
losses.py — v3
--------------
Redesigned loss for stable waveform reconstruction.

Stage 1 (backbone frozen):
    PRIMARY = MSE  — forces decoder to actually reconstruct waveform amplitude
    + STFT         — spectral shape guidance
    
Stage 2 (backbone unfrozen):
    PRIMARY = SI-SNR  — perceptual quality
    + STFT            — spectral shape
    + MSE             — amplitude anchor

The key insight: SI-SNR alone cannot train a decoder from scratch because
it is scale-invariant and gives no gradient signal when the output is near-zero.
MSE must dominate Stage 1 to bootstrap the decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEWaveformLoss(nn.Module):
    """Direct MSE between predicted and target waveform. Scale-sensitive."""
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512,1024,2048),
                 hop_sizes=(120,240,480), win_lengths=(480,960,1920)):
        super().__init__()
        self.fft_sizes   = fft_sizes
        self.hop_sizes   = hop_sizes
        self.win_lengths = win_lengths

    def _stft_mag(self, x, n_fft, hop, win_len):
        window = torch.hann_window(win_len, device=x.device)
        return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_len,
                          window=window, return_complex=True).abs()

    def forward(self, pred, target):
        loss = 0.0
        for n_fft, hop, win_len in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pm = self._stft_mag(pred,   n_fft, hop, win_len)
            tm = self._stft_mag(target, n_fft, hop, win_len)
            sc   = torch.norm(tm - pm, p="fro") / (torch.norm(tm, p="fro") + 1e-8)
            logm = F.l1_loss(torch.log(pm + 1e-7), torch.log(tm + 1e-7))
            loss = loss + sc + logm
        return loss / len(self.fft_sizes)


class SISNRLoss(nn.Module):
    def forward(self, pred, target):
        pred   = pred   - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        dot        = (pred * target).sum(dim=-1, keepdim=True)
        target_pow = (target * target).sum(dim=-1, keepdim=True) + 1e-8
        proj   = dot / target_pow * target
        noise  = pred - proj
        si_snr = 10 * torch.log10(
            (proj*proj).sum(dim=-1) / ((noise*noise).sum(dim=-1) + 1e-8) + 1e-8)
        return -si_snr.mean()


class CompositeLoss(nn.Module):
    """
    Stage-aware composite loss.
    
    stage=1:  MSE (10.0) + STFT (1.0)        — bootstrap decoder reconstruction
    stage=2:  SI-SNR (1.5) + STFT (1.0) + MSE (2.0)  — perceptual quality
    """

    def __init__(self, stage: int = 1):
        super().__init__()
        self.stage      = stage
        self.mse_loss   = MSEWaveformLoss()
        self.stft_loss  = MultiResolutionSTFTLoss()
        self.sisnr_loss = SISNRLoss()

    def set_stage(self, stage: int):
        self.stage = stage
        print(f"[Loss] Switched to Stage {stage} loss weights")

    def forward(self, pred, target):
        mse   = self.mse_loss(pred, target)
        stft  = self.stft_loss(pred, target)
        sisnr = self.sisnr_loss(pred, target)

        if self.stage == 1:
            # MSE dominates — forces amplitude-correct reconstruction
            total = 10.0 * mse + 1.0 * stft
        else:
            # SI-SNR dominates + MSE anchor prevents amplitude collapse
            total = 1.5 * sisnr + 1.0 * stft + 2.0 * mse

        return total, {
            "mse"  : mse.item(),
            "stft" : stft.item(),
            "sisnr": sisnr.item(),
        }