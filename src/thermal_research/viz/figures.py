"""Headless figure helpers (Agg backend), consistent style."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#3b6ea8", "#d9822b", "#5a9e5a", "#b0483f", "#7d5ba6", "#8c8c8c"]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    })


def grouped_bars(groups: list[str], series: dict[str, list[tuple[float, float, float]]], ylabel: str,
                 title: str, out: Path, log: bool = False) -> None:
    """series[name] = [(mean, lo, hi) per group]."""
    style()
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(groups), 3.2))
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        m = np.array([v[0] for v in vals])
        lo = np.array([v[1] for v in vals])
        hi = np.array([v[2] for v in vals])
        ax.bar(x + (i - (n - 1) / 2) * width, m, width, yerr=[m - lo, hi - m], capsize=2,
               label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def map_panel(maps: list[np.ndarray], titles: list[str], out: Path, cmap: str = "inferno",
              shared: bool = True, cbar_label: str = "°C") -> None:
    style()
    fig, axes = plt.subplots(1, len(maps), figsize=(2.6 * len(maps), 2.6))
    axes = np.atleast_1d(axes)
    vmin = min(float(m.min()) for m in maps) if shared else None
    vmax = max(float(m.max()) for m in maps) if shared else None
    for ax, m, t in zip(axes, maps, titles):
        im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if not shared:
            fig.colorbar(im, ax=ax, fraction=0.046, label=cbar_label)
    if shared:
        fig.colorbar(im, ax=list(axes), fraction=0.02, label=cbar_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
