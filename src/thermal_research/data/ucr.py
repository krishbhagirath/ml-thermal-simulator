"""Loading the UC Riverside commercial thermal-map excerpts into a cached, typed form."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .safe_pickle import load_pickle

PIXEL_PITCH_MM = {"i7": 0.05, "i5": 0.05}  # 50 um/px [paper-reported]
_PREFIX = {"i7": "CPU_i7_", "i5": "CPU_i5_"}


def data_root() -> Path:
    """Root for data files; override with THERMAL_DATA_ROOT."""
    env = os.environ.get("THERMAL_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data"


def raw_dir() -> Path:
    return data_root() / "raw" / "ucr_commercial_thermal" / "data_files"


@dataclass(frozen=True)
class ChipData:
    """One chip's excerpt. Frames are time-ordered within each workload."""

    chip: str
    X: np.ndarray  # (n, F) telemetry, float64
    Y: np.ndarray  # (n, H, W) temperature, deg C, float32
    group: np.ndarray  # (n,) workload index
    frame: np.ndarray  # (n,) frame index within workload (time order)
    workloads: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def shape(self) -> tuple[int, int]:
        return self.Y.shape[1], self.Y.shape[2]

    @property
    def pitch_mm(self) -> float:
        return PIXEL_PITCH_MM[self.chip]

    def subset(self, idx: np.ndarray) -> "ChipData":
        return ChipData(self.chip, self.X[idx], self.Y[idx], self.group[idx], self.frame[idx], self.workloads)


def _from_raw(chip: str, root: Path) -> ChipData:
    prefix = _PREFIX[chip]
    files = sorted(f for f in os.listdir(root) if f.startswith(prefix) and f.endswith(".pkl"))
    if not files:
        raise FileNotFoundError(f"no {prefix}*.pkl files in {root}")
    X, Y, G, F, names = [], [], [], [], []
    for gi, fn in enumerate(files):
        d = load_pickle(root / fn)
        x = np.asarray(d["input"], dtype=np.float64)
        y = np.asarray(d["output"], dtype=np.float32)
        if x.ndim != 2 or y.ndim != 3 or len(x) != len(y):
            raise ValueError(f"unexpected shapes in {fn}: {x.shape} {y.shape}")
        X.append(x)
        Y.append(y)
        G.append(np.full(len(x), gi))
        F.append(np.arange(len(x)))
        names.append(fn[len(prefix) : -4])
    return ChipData(chip, np.concatenate(X), np.concatenate(Y), np.concatenate(G), np.concatenate(F), tuple(names))


def load_chip(chip: str = "i7", use_cache: bool = True) -> ChipData:
    """Load a chip excerpt (cached as .npz under data/cache)."""
    cache = data_root() / "cache" / f"{chip}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        return ChipData(chip, z["X"], z["Y"], z["group"], z["frame"], tuple(str(s) for s in z["workloads"]))
    cd = _from_raw(chip, raw_dir())
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=cd.X, Y=cd.Y, group=cd.group, frame=cd.frame, workloads=np.array(cd.workloads))
    return cd


def raw_available(chip: str = "i7") -> bool:
    root = raw_dir()
    return root.exists() and any(f.startswith(_PREFIX[chip]) for f in os.listdir(root))
