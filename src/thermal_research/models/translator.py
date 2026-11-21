"""Translator: telemetry -> non-negative power map (basis maps x coefficients) + base temperature,
passed through a frozen heat model.

Because the heat model is linear in q, theta = sum_k a_k * H(B_k): each step only needs the heat
response of the K basis maps, not of every frame. That keeps calibration cheap on a CPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class HeatOperator(nn.Module):
    """Interface: forward(q: (K, H, W)) -> theta (K, H, W), linear in q. Global params may be learnable."""

    def forward(self, q: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class SpecOperator(HeatOperator):
    """H-spec wrapper: G fixed (D-006), log h learnable (fitted on calibration data only)."""

    def __init__(self, spectral, G: float, h_init: float, learn_h: bool = True):
        super().__init__()
        self.spec = spectral
        self.G = G
        self.log_h = nn.Parameter(torch.tensor(math.log(h_init), dtype=torch.float64), requires_grad=learn_h)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.spec.steady(q, torch.tensor(self.G, dtype=q.dtype), torch.exp(self.log_h).to(q.dtype))


class Translator(nn.Module):
    """a(t) = softplus(MLP(z)) (K coefficients), B_k = softplus(U_k) at low resolution, upsampled.
    T_pred = T_base(z) + sum_k a_k(t) * H(B_k)."""

    def __init__(self, n_in: int, K: int, out_hw: tuple[int, int], low_hw: tuple[int, int] = (37, 31),
                 hidden: int = 0, init_basis: torch.Tensor | None = None, t_base_init: float = 40.0):
        super().__init__()
        self.out_hw, self.K = out_hw, K
        if hidden > 0:
            self.coef = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, K))
        else:
            self.coef = nn.Linear(n_in, K)
        self.base = nn.Linear(n_in, 1)
        nn.init.zeros_(self.base.weight)
        nn.init.constant_(self.base.bias, t_base_init)
        if init_basis is not None:
            u = torch.log(torch.expm1(init_basis.clamp_min(1e-4)))  # inverse softplus
        else:
            u = torch.randn(K, *low_hw) * 0.1
        self.U = nn.Parameter(u.to(torch.float64))

    def basis(self) -> torch.Tensor:
        B = F.softplus(self.U)[None]
        B = F.interpolate(B, size=self.out_hw, mode="bilinear", align_corners=False)[0]
        return B / B.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)  # unit-mean maps (scale -> coefficients)

    def forward(self, z: torch.Tensor, heat: HeatOperator, scale: float = 1.0) -> tuple[torch.Tensor, dict]:
        a = F.softplus(self.coef(z))  # (n, K)
        B = self.basis()  # (K, H, W)
        HB = heat(B * scale)  # (K, H, W)
        theta = torch.einsum("nk,khw->nhw", a, HB)
        T = self.base(z)[:, :, None] + theta
        return T, {"a": a, "B": B}

    @staticmethod
    def tv(B: torch.Tensor) -> torch.Tensor:
        return (B[:, 1:, :] - B[:, :-1, :]).abs().mean() + (B[:, :, 1:] - B[:, :, :-1]).abs().mean()
