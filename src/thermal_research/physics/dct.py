"""Orthonormal DCT-II via explicit matrices (exact, differentiable, device-agnostic).

For the map sizes here (~185x154), the dense-matrix transform costs ~10 MFLOP per map, so there is no
need for FFT-based tricks.
"""
from __future__ import annotations

import numpy as np
import torch


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix C with C @ x = scipy.fft.dct(x, type=2, norm='ortho')."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    C = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    C[0] /= np.sqrt(2.0)
    return C


class DCT2:
    """2-D orthonormal DCT-II / inverse on the last two dims of a (..., H, W) tensor."""

    def __init__(self, H: int, W: int, dtype=torch.float64, device="cpu"):
        self.H, self.W = H, W
        self.CH = torch.tensor(dct_matrix(H), dtype=dtype, device=device)
        self.CW = torch.tensor(dct_matrix(W), dtype=dtype, device=device)

    def to(self, dtype=None, device=None) -> "DCT2":
        self.CH = self.CH.to(dtype=dtype, device=device)
        self.CW = self.CW.to(dtype=dtype, device=device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.CH @ x @ self.CW.T

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        return self.CH.T @ X @ self.CW
