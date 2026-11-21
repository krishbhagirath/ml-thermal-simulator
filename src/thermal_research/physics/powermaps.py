"""Generic random power maps for simulation (not fitted to any real chip)."""
from __future__ import annotations

import numpy as np


def random_power_map(shape: tuple[int, int], rng: np.random.Generator, total_w_per_m2_mean: float = 1.0) -> np.ndarray:
    """Non-negative map = background + random rectangles + Gaussian blobs, with random intensities.

    Returned map has mean value `total_w_per_m2_mean` (relative units; physics is linear in q).
    """
    H, W = shape
    q = np.full(shape, rng.uniform(0.0, 0.3))
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(rng.integers(1, 9)):  # rectangles (functional blocks)
        h = int(rng.integers(3, max(4, H // 2)))
        w = int(rng.integers(3, max(4, W // 2)))
        y0 = int(rng.integers(0, H - h + 1))
        x0 = int(rng.integers(0, W - w + 1))
        q[y0 : y0 + h, x0 : x0 + w] += rng.lognormal(0.0, 0.8)
    for _ in range(rng.integers(0, 5)):  # blobs (hotspots)
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        s = rng.uniform(2, max(3, min(H, W) / 6))
        q += rng.lognormal(0.5, 0.8) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    return q * (total_w_per_m2_mean / q.mean())
