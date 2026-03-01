"""
Training loop, loss definitions, and scheduling logic following the staged procedure.
"""

import torch


def waveform_loss(pred, target):
    return torch.nn.functional.l1_loss(pred, target)


def stft_loss(pred, target):
    # placeholder for multi-resolution STFT loss
    raise NotImplementedError


def sisdr_loss(pred, target):
    # compute SI-SDR
    raise NotImplementedError


class Trainer:
    def __init__(self, model, optimizer, dataloader, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device

    def train_epoch(self):
        self.model.train()
        for batch in self.dataloader:
            degraded, clean = batch
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(degraded)
            loss = waveform_loss(output, clean)
            loss.backward()
            self.optimizer.step()

    def validate(self):
        pass
