"""A minimal LoRA layer for models built in this package.

Hugging Face backbones use PEFT; the tiny backbone of the test-suite and the quick start
uses this dependency-free equivalent: ``W x + (alpha / r) B A dropout(x)`` with ``A`` drawn
Kaiming-uniform and ``B`` zero, so that the adapted model equals the base model at
``theta_0`` (Hu et al., 2021).
"""

from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, rank: int, alpha: float, dropout: float, generator: torch.Generator
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        bound = 1.0 / (base.in_features**0.5)
        self.lora_A = nn.Parameter(
            (torch.rand(rank, base.in_features, generator=generator) * 2 - 1) * bound
        )
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale
        return out
