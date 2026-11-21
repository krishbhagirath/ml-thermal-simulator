"""Steady-state inversion of measured maps to (relative) power maps.

q = (G/D^2) * L th + h th, with L the Neumann 5-point Laplacian (FV-consistent) and th = T - T_base.
VSCLab's method uses only the Laplacian term with negatives clipped [paper-reported]; we include the sink
term for consistency with H-spec and expose both. A Gaussian pre-filter tames noise amplification.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def neumann_laplacian(T: np.ndarray) -> np.ndarray:
    """sum_nbrs (T_P - T_Q) with mirrored (adiabatic) edges; positive at local maxima."""
    P = np.pad(T, [(0, 0)] * (T.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    c = P[..., 1:-1, 1:-1]
    return 4 * c - P[..., :-2, 1:-1] - P[..., 2:, 1:-1] - P[..., 1:-1, :-2] - P[..., 1:-1, 2:]


def invert(T: np.ndarray, G: float, h: float, T_base: float, pitch: float, sigma: float = 2.0,
           clip: bool = True) -> np.ndarray:
    """Relative power map from a temperature map (H, W) or stack (n, H, W)."""
    Ts = gaussian_filter(T, sigma=[0] * (T.ndim - 2) + [sigma, sigma]) if sigma > 0 else T
    q = (G / pitch**2) * neumann_laplacian(Ts) + h * (Ts - T_base)
    return np.maximum(q, 0.0) if clip else q
