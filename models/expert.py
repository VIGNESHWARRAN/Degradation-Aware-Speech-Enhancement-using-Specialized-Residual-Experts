"""
Residual expert modules for different degradation types.
Each expert implements a residual stack: output = x + F(x), where F is a small convolutional network.
"""

import torch


class ResidualExpert(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # define a few Conv1d layers with dilations
        self.conv1 = torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.conv2 = torch.nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (batch, channels, time)
        residual = self.conv1(x)
        residual = self.norm1(residual.transpose(1,2)).transpose(1,2)
        residual = torch.relu(residual)
        residual = self.conv2(residual)
        return x + residual


class ExpertSet(torch.nn.Module):
    """Container for K experts; can select by index or weighted sum."""
    def __init__(self, num_experts, in_channels, hidden_channels):
        super().__init__()
        self.experts = torch.nn.ModuleList([
            ResidualExpert(in_channels, hidden_channels) for _ in range(num_experts)
        ])

    def forward(self, x, weights=None, selected_index=None):
        if selected_index is not None:
            return self.experts[selected_index](x)
        if weights is not None:
            out = 0
            for w, e in zip(weights, self.experts):
                out = out + w * e(x)
            return out
        raise ValueError("Must provide either weights or selected_index")
