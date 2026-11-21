"""Per-frame thermal-map metrics (deg C). All functions take (n, H, W) arrays."""
from __future__ import annotations

import numpy as np


def per_map_rmse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.sqrt(((pred - true) ** 2).reshape(len(true), -1).mean(1))


def per_map_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true).reshape(len(true), -1).mean(1)


def peak_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """|max(pred) - max(true)| per frame."""
    n = len(true)
    return np.abs(pred.reshape(n, -1).max(1) - true.reshape(n, -1).max(1))


def hotspot_distance_mm(pred: np.ndarray, true: np.ndarray, pitch_mm: float) -> np.ndarray:
    """Euclidean distance between argmax locations, in mm."""
    n, H, W = true.shape
    ip = np.unravel_index(pred.reshape(n, -1).argmax(1), (H, W))
    it = np.unravel_index(true.reshape(n, -1).argmax(1), (H, W))
    return pitch_mm * np.hypot(ip[0] - it[0], ip[1] - it[1])


def hotspot_iou(pred: np.ndarray, true: np.ndarray, top_frac: float = 0.01) -> np.ndarray:
    """IoU of the hottest `top_frac` pixel sets."""
    n = len(true)
    p, t = pred.reshape(n, -1), true.reshape(n, -1)
    k = max(1, int(round(p.shape[1] * top_frac)))
    out = np.empty(n)
    for i in range(n):
        a = set(np.argpartition(-p[i], k - 1)[:k].tolist())
        b = set(np.argpartition(-t[i], k - 1)[:k].tolist())
        out[i] = len(a & b) / len(a | b)
    return out


def map_corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    n = len(true)
    p = pred.reshape(n, -1) - pred.reshape(n, -1).mean(1, keepdims=True)
    t = true.reshape(n, -1) - true.reshape(n, -1).mean(1, keepdims=True)
    den = np.sqrt((p**2).sum(1) * (t**2).sum(1))
    return np.where(den > 0, (p * t).sum(1) / np.maximum(den, 1e-12), 0.0)


def frame_metrics(pred: np.ndarray, true: np.ndarray, pitch_mm: float) -> dict[str, np.ndarray]:
    return {
        "rmse": per_map_rmse(pred, true),
        "mae": per_map_mae(pred, true),
        "peak_err": peak_error(pred, true),
        "hotspot_mm": hotspot_distance_mm(pred, true, pitch_mm),
        "hotspot_iou": hotspot_iou(pred, true),
        "corr": map_corr(pred, true),
    }


def per_workload(metrics: dict[str, np.ndarray], group: np.ndarray) -> dict[int, dict[str, float]]:
    """Average frame metrics within each workload (the statistical unit)."""
    out: dict[int, dict[str, float]] = {}
    for g in np.unique(group):
        m = group == g
        out[int(g)] = {k: float(v[m].mean()) for k, v in metrics.items()}
    return out
