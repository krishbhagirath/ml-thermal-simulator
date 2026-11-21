"""Hierarchical bootstrap over (calibration draw -> test workload) with workloads as the unit."""
from __future__ import annotations

import numpy as np


def hier_bootstrap(values: dict[int, dict[int, float]], n_boot: int = 5000, seed: int = 0,
                   ci: float = 0.95) -> dict[str, float]:
    """values[draw][workload] = metric. Resample draws, then workloads within each draw.

    Returns mean (over draws of the per-draw workload mean) and a percentile CI.
    """
    draws = [np.array(list(v.values()), dtype=float) for v in values.values() if len(v)]
    if not draws:
        raise ValueError("no values")
    point = float(np.mean([d.mean() for d in draws]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    nd = len(draws)
    for b in range(n_boot):
        pick = rng.integers(0, nd, nd)
        boots[b] = np.mean([draws[i][rng.integers(0, len(draws[i]), len(draws[i]))].mean() for i in pick])
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": point, "lo": float(lo), "hi": float(hi), "n_draws": nd,
            "n_units": int(sum(len(d) for d in draws))}


def paired_diff(a: dict[int, dict[int, float]], b: dict[int, dict[int, float]], **kw) -> dict[str, float]:
    """Bootstrap CI of (a - b) on matching (draw, workload) pairs."""
    diff: dict[int, dict[int, float]] = {}
    for d in a:
        if d in b:
            common = set(a[d]) & set(b[d])
            if common:
                diff[d] = {w: a[d][w] - b[d][w] for w in common}
    out = hier_bootstrap(diff, **kw)
    flat = np.array([v for dd in diff.values() for v in dd.values()])
    out["frac_a_better"] = float((flat < 0).mean())  # for error metrics, negative diff = a better
    try:
        from scipy.stats import wilcoxon

        out["wilcoxon_p"] = float(wilcoxon(flat).pvalue) if np.any(flat != 0) else 1.0
    except Exception:  # pragma: no cover
        out["wilcoxon_p"] = float("nan")
    return out
