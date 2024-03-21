# Coded by: Yuzhuang Xu
# Reference: Parameter-Efficient Transfer Learning for NLP

import torch
import torch.nn as nn
import numpy as np


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    x: float Tensor to perform activation.
    Returns:
    `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + torch.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * cdf


class AdapterLayer(nn.Module):

    def __init__(self, embed_dim, hidden_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, hidden_size, bias=True)
        self.nonlinearity = gelu
        self.fc2 = nn.Linear(hidden_size, self.embed_dim, bias=True)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.nonlinearity(x)
        x = self.fc2(x)
        return x + residual
