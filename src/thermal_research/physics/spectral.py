"""H-spec: exact spectral solver for the 1-layer, uniform-property thin-die model.

Per cell (cell-centred FV, pitch D, adiabatic edges):
    (G/D^2) * sum_nbrs (th_P - th_Q) + h * th_P = q            [W/m^2]
The Neumann 5-point Laplacian is diagonalised exactly by the 2-D DCT-II with eigenvalues
    lam_mn = (G/D^2) * (4 sin^2(pi m / 2H) + 4 sin^2(pi n / 2W)) + h.
So the steady solution is th = IDCT(DCT(q) / lam), exact to round-off vs the sparse FV solve.
Transient (implicit Euler):  th^{n+1} = IDCT( (C/dt * DCT(th^n) + DCT(q^{n+1})) / (C/dt + lam) ).
"""
from __future__ import annotations

import numpy as np
import torch

from .dct import DCT2


def laplacian_eigs(H: int, W: int) -> np.ndarray:
    """Eigenvalues (H, W) of the unit-spacing Neumann graph Laplacian, matching DCT-II order."""
    m = np.arange(H)[:, None]
    n = np.arange(W)[None, :]
    return 4 * np.sin(np.pi * m / (2 * H)) ** 2 + 4 * np.sin(np.pi * n / (2 * W)) ** 2


class SpectralHeat(torch.nn.Module):
    """Differentiable steady/transient solver. Parameters G [W/K], h [W/m^2/K], C [J/m^2/K] are passed in
    (so they can be learnable elsewhere)."""

    def __init__(self, H: int, W: int, pitch_m: float, dtype=torch.float64):
        super().__init__()
        self.H, self.W, self.pitch = H, W, pitch_m
        self.dct = DCT2(H, W, dtype=dtype)
        self.register_buffer("L", torch.tensor(laplacian_eigs(H, W), dtype=dtype))

    def _apply(self, fn, recurse=True):  # keep the DCT matrices on the same device/dtype
        super()._apply(fn, recurse)
        self.dct.CH = fn(self.dct.CH)
        self.dct.CW = fn(self.dct.CW)
        return self

    def eigs(self, G: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        G = torch.as_tensor(G, dtype=self.L.dtype, device=self.L.device)
        h = torch.as_tensor(h, dtype=self.L.dtype, device=self.L.device)
        return (G / self.pitch**2)[..., None, None] * self.L + h[..., None, None]

    def steady(self, q: torch.Tensor, G, h) -> torch.Tensor:
        """q: (..., H, W) W/m^2  ->  temperature rise (..., H, W) K."""
        return self.dct.inverse(self.dct.forward(q) / self.eigs(G, h))

    def transient(self, q_seq: torch.Tensor, G, h, C, dt: float, theta0: torch.Tensor | None = None) -> torch.Tensor:
        """q_seq: (T, ..., H, W). Starts from the steady state of q_seq[0] unless theta0 is given.
        Returns the full trajectory (T, ..., H, W)."""
        lam = self.eigs(G, h)
        C = torch.as_tensor(C, dtype=lam.dtype, device=lam.device)[..., None, None]
        a = C / dt
        Th = self.dct.forward(theta0) if theta0 is not None else self.dct.forward(q_seq[0]) / lam
        out = []
        for t in range(q_seq.shape[0]):
            Th = (a * Th + self.dct.forward(q_seq[t])) / (a + lam)
            out.append(self.dct.inverse(Th))
        return torch.stack(out)
