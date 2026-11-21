"""Split protocols. Every split returns integer index arrays; overlaps are tested."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

DRAW_SEED = 20261121  # fixed: calibration draws are shared by all models


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray | None = None
    cal_workloads: tuple[int, ...] = ()
    name: str = ""


def random_frames(n: int, test_frac: float = 0.2, seed: int = 0) -> Split:
    """S-RAND: random frame split (leakage demonstration only)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return Split(np.sort(perm[n_test:]), np.sort(perm[:n_test]), name="S-RAND")


def calibration_draws(n_workloads: int, k: int, n_draws: int, seed: int = DRAW_SEED) -> list[tuple[int, ...]]:
    """Deterministic distinct k-subsets of workloads (all of them if fewer than n_draws exist)."""
    total = comb(n_workloads, k)
    if total <= n_draws:
        return [tuple(c) for c in combinations(range(n_workloads), k)]
    rng = np.random.default_rng([seed, n_workloads, k])
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    while len(out) < n_draws:
        c = tuple(sorted(rng.choice(n_workloads, size=k, replace=False).tolist()))
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def time_tail(group: np.ndarray, frame: np.ndarray, idx: np.ndarray, frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Split `idx` into (head, tail): the last `frac` of frames of each workload goes to tail."""
    head, tail = [], []
    for g in np.unique(group[idx]):
        ii = idx[group[idx] == g]
        ii = ii[np.argsort(frame[ii])]
        n_tail = max(1, int(round(len(ii) * frac)))
        head.append(ii[:-n_tail])
        tail.append(ii[-n_tail:])
    return np.sort(np.concatenate(head)), np.sort(np.concatenate(tail))


def calibration_split(group: np.ndarray, frame: np.ndarray, cal: tuple[int, ...], val_frac: float = 0.2) -> Split:
    """S-CAL-k: calibrate on workloads `cal` (with a time-tail validation part), test on all others."""
    is_cal = np.isin(group, cal)
    cal_idx = np.flatnonzero(is_cal)
    train, val = time_tail(group, frame, cal_idx, val_frac)
    return Split(train, np.flatnonzero(~is_cal), val, tuple(cal), name=f"S-CAL-{len(cal)}")


def lowo(group: np.ndarray, frame: np.ndarray, held: int, val_frac: float = 0.2) -> Split:
    """S-LOWO: hold out one workload; validate on the time tail of the training workloads."""
    others = tuple(int(g) for g in np.unique(group) if g != held)
    s = calibration_split(group, frame, others, val_frac)
    return Split(s.train, np.flatnonzero(group == held), s.val, others, name="S-LOWO")
