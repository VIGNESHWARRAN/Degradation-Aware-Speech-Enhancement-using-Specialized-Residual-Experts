"""
Expert fusion strategies: hard routing and gating network for soft combination.
"""

import torch


class HardRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, index):
        # simply return index for selection elsewhere
        return index


class GatingNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_experts):
        super().__init__()
        self.fc = torch.nn.Linear(in_channels, num_experts)

    def forward(self, features):
        # global pooling then linear to weights
        x = features.mean(dim=-1)  # (batch, channels)
        logits = self.fc(x)
        weights = torch.softmax(logits, dim=-1)
        return weights
