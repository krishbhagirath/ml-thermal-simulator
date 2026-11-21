"""Feature preprocessing fitted on calibration data only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sensor_columns(X: np.ndarray, Y: np.ndarray, idx: np.ndarray, corr_thresh: float = 0.8) -> np.ndarray:
    """Data rule (D-004) for on-die temperature-sensor-like columns, computed on rows `idx` only.

    A column qualifies if it is integer-valued, lies within [0, 110], and |corr| with the frame-mean
    map temperature exceeds `corr_thresh`.
    """
    x = X[idx]
    mean_t = Y[idx].reshape(len(idx), -1).mean(1)
    cols = []
    for j in range(x.shape[1]):
        c = x[:, j]
        if c.std() == 0 or c.min() < 0 or c.max() > 110 or not np.allclose(c, np.round(c)):
            continue
        r = np.corrcoef(c, mean_t)[0, 1]
        if abs(r) > corr_thresh:
            cols.append(j)
    return np.array(cols, dtype=int)


@dataclass
class Standardizer:
    """z-score with constant-column removal. Fit on training rows only."""

    keep: np.ndarray | None = None
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, X: np.ndarray, drop: np.ndarray | None = None) -> "Standardizer":
        keep = X.std(0) > 0
        if drop is not None and len(drop):
            keep[drop] = False
        self.keep = np.flatnonzero(keep)
        self.mu = X[:, self.keep].mean(0)
        self.sd = X[:, self.keep].std(0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.keep is not None and self.mu is not None and self.sd is not None
        return (X[:, self.keep] - self.mu) / self.sd

    @property
    def n_features(self) -> int:
        return 0 if self.keep is None else len(self.keep)


def windowed(X: np.ndarray, group: np.ndarray, frame: np.ndarray, w: int) -> np.ndarray:
    """Stack the previous w-1 frames of the same workload (edge-padded with the first frame).

    Returns (n, w*F). Only uses past frames, so it is causal.
    """
    if w == 1:
        return X
    out = np.empty((len(X), w * X.shape[1]), dtype=X.dtype)
    for g in np.unique(group):
        ii = np.flatnonzero(group == g)
        ii = ii[np.argsort(frame[ii])]
        xs = X[ii]
        for lag in range(w):
            shifted = np.concatenate([np.repeat(xs[:1], lag, axis=0), xs[: len(xs) - lag]], axis=0)
            out[ii, lag * X.shape[1] : (lag + 1) * X.shape[1]] = shifted
    return out
