"""Shared evaluation runner: identical preprocessing, splits and metrics for every model.

Model interface: `fit(X, Y, group=, frame=, val=(Xv, Yv), Ytrain_full=...)` and
`predict(X, group=, frame=)`. X is already standardized on training rows only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..data.features import Standardizer, sensor_columns, windowed
from ..data.splits import Split
from ..data.ucr import ChipData
from . import metrics as M

METRIC_KEYS = ("rmse", "mae", "peak_err", "hotspot_mm", "hotspot_iou", "corr")


@dataclass
class Prepared:
    Xtr: np.ndarray
    Xva: np.ndarray | None
    Xte: np.ndarray
    sensor_cols: np.ndarray
    n_features: int


def prepare(cd: ChipData, split: Split, use_sensors: bool, w: int = 1) -> Prepared:
    """Standardize on training rows; optionally drop sensor-like columns (rule fit on training rows)."""
    fit_rows = split.train if split.val is None else np.concatenate([split.train, split.val])
    sens = sensor_columns(cd.X, cd.Y, fit_rows)
    st = Standardizer().fit(cd.X[split.train], drop=None if use_sensors else sens)
    Z = st.transform(cd.X)
    Z = windowed(Z, cd.group, cd.frame, w)
    return Prepared(Z[split.train], None if split.val is None else Z[split.val], Z[split.test], sens,
                    st.n_features)


def evaluate_split(factory: Callable[[], object], cd: ChipData, split: Split, use_sensors: bool = True,
                   w: int = 1) -> dict:
    p = prepare(cd, split, use_sensors, w)
    model = factory()
    t0 = time.perf_counter()
    model.fit(
        p.Xtr, cd.Y[split.train], group=cd.group[split.train], frame=cd.frame[split.train],
        val=None if p.Xva is None else (p.Xva, cd.Y[split.val]),
        val_group=None if split.val is None else cd.group[split.val],
    )
    fit_s = time.perf_counter() - t0
    pred = model.predict(p.Xte, group=cd.group[split.test], frame=cd.frame[split.test])
    fm = M.frame_metrics(pred, cd.Y[split.test], cd.pitch_mm)
    return {
        "per_workload": M.per_workload(fm, cd.group[split.test]),
        "frame_mean": {k: float(v.mean()) for k, v in fm.items()},
        "fit_seconds": fit_s,
        "n_sensor_cols": int(len(p.sensor_cols)),
        "n_features": p.n_features,
        "chosen": getattr(model, "chosen_", None),
    }


def collect(results: list[dict], key: str = "rmse") -> dict[int, dict[int, float]]:
    """results[i] -> {draw i: {workload: metric}} for hierarchical bootstrap."""
    return {i: {w: m[key] for w, m in r["per_workload"].items()} for i, r in enumerate(results)}
