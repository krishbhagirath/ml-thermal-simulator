"""M2 gates on the public i7 excerpt (diagnostics only; nothing fitted here is reused for testing).

G1 (physics expressiveness): per frame, fit a non-negative low-resolution power map + base temperature,
shared cooling h, through H-spec. Its per-frame RMSE is the representation floor of the physics family.
Control: a directly fitted smooth surface at the same resolution (bilinear-upsampled low-res map).

G2 (layout consistency): Laplacian-inverted power maps averaged per workload; cosine similarity across
workloads shows whether the hot regions are stable (a learnable layout) or workload-specific.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from thermal_research.data import ucr  # noqa: E402
from thermal_research.physics.inversion import invert  # noqa: E402
from thermal_research.physics.spectral import SpectralHeat  # noqa: E402
from thermal_research.training.checkpoint import JsonProgress  # noqa: E402
from thermal_research.viz import figures  # noqa: E402

G_FIXED = 150 * 0.5e-3  # W/K (D-006)


def subset(cd, per: int = 20) -> np.ndarray:
    idx = []
    for g in range(len(cd.workloads)):
        ii = np.flatnonzero(cd.group == g)
        idx.extend(ii[np.linspace(0, len(ii) - 1, per).round().astype(int)])
    return np.array(idx)


def fit_physics(Y: np.ndarray, low_hw, pitch, steps=1500, h_init=1000.0, seed=0):
    torch.manual_seed(seed)
    n, H, W = Y.shape
    spec = SpectralHeat(H, W, pitch, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    u = torch.nn.Parameter(torch.full((n, 1, *low_hw), -2.0))
    base = torch.nn.Parameter(torch.tensor(np.percentile(Y.reshape(n, -1), 2, axis=1), dtype=torch.float32))
    log_h = torch.nn.Parameter(torch.tensor(math.log(h_init)))
    opt = torch.optim.Adam([u, base, log_h], lr=5e-2)
    scale = 1e5
    for step in range(steps):
        opt.zero_grad()
        q = F.interpolate(F.softplus(u), size=(H, W), mode="bilinear", align_corners=False)[:, 0] * scale
        th = spec.steady(q, torch.tensor(G_FIXED), torch.exp(log_h))
        pred = base[:, None, None] + th
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rmse = ((pred - y) ** 2).reshape(n, -1).mean(1).sqrt().numpy()
    h = float(torch.exp(log_h))
    return rmse, h, pred.detach().numpy()


def fit_smooth_control(Y: np.ndarray, low_hw) -> np.ndarray:
    """Least-squares fit of a bilinear-upsampled low-res surface (closed form via the upsampling matrix)."""
    n, H, W = Y.shape

    def up_matrix(n_out, n_in):
        eye = torch.eye(n_in)[None]  # (1, n_in, n_in) -> interpolate rows
        M = F.interpolate(eye, size=n_out, mode="linear", align_corners=False)[0].T  # (n_out, n_in)
        return M.numpy()

    Uh, Uw = up_matrix(H, low_hw[0]), up_matrix(W, low_hw[1])
    Ph, Pw = np.linalg.pinv(Uh), np.linalg.pinv(Uw)
    low = Ph @ Y @ Pw.T
    rec = Uh @ low @ Uw.T
    return np.sqrt(((rec - Y) ** 2).reshape(n, -1).mean(1))


def main() -> None:
    cd = ucr.load_chip("i7")
    idx = subset(cd)
    Y = cd.Y[idx].astype(np.float64)
    out: dict = {"n_frames": int(len(idx)), "G_fixed_W_per_K": G_FIXED, "G1": {}, "G2": {}}
    progress = JsonProgress(ROOT / "runs" / "gates" / "progress.json")  # resume after disconnects
    for low_hw in ((37, 31), (19, 16)):
        key = f"{low_hw[0]}x{low_hw[1]}"
        if key in progress:  # cached: reuse numbers (and the fitted h for G2); figure already written
            out["G1"][key] = progress[key]
            if low_hw == (37, 31):
                h_fit = progress[key]["fitted_h"]
            continue
        rmse, h, pred = fit_physics(Y, low_hw, cd.pitch_mm * 1e-3)
        ctrl = fit_smooth_control(Y, low_hw)
        ell_mm = math.sqrt(G_FIXED / h) * 1e3
        out["G1"][key] = {
            "physics_floor_median": float(np.median(rmse)), "physics_floor_p90": float(np.quantile(rmse, 0.9)),
            "smooth_control_median": float(np.median(ctrl)), "fitted_h": h, "spreading_length_mm": ell_mm,
        }
        progress.put(key, out["G1"][key])
        print(f"G1 {key}: physics floor median {np.median(rmse):.3f} C (p90 {np.quantile(rmse, .9):.3f}), "
              f"smooth control {np.median(ctrl):.3f} C, h={h:.0f} W/m2K, l={ell_mm:.2f} mm")
        if low_hw == (37, 31):
            h_fit = h
            k = int(np.argmax(rmse))
            figures.map_panel([Y[k], pred[k], pred[k] - Y[k]],
                              ["measured (worst-fit frame)", "physics fit", "fit - measured"],
                              ROOT / "results" / "figures" / "g1_worst_fit.png", shared=False)

    # G2: inverted power maps per workload (calibration-free diagnostic)
    Q = invert(cd.Y.astype(np.float64), G_FIXED, h_fit, T_base=float(np.percentile(cd.Y, 1)),
               pitch=cd.pitch_mm * 1e-3, sigma=2.0)
    means = np.stack([Q[cd.group == g].mean(0) for g in range(len(cd.workloads))])
    v = means.reshape(len(means), -1)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    S = v @ v.T
    off = S[~np.eye(len(S), dtype=bool)]
    out["G2"] = {"cosine_mean": float(off.mean()), "cosine_min": float(off.min()),
                 "cosine_p10": float(np.quantile(off, 0.1))}
    print(f"G2: cosine similarity of per-workload power maps: mean {off.mean():.3f}, min {off.min():.3f}")
    order = np.argsort(-means.reshape(len(means), -1).max(1))[:6]
    figures.map_panel([means[i] for i in order], [cd.workloads[i] for i in order],
                      ROOT / "results" / "figures" / "g2_power_maps.png", cmap="magma", shared=False,
                      cbar_label="rel. W/m²")
    (ROOT / "results" / "gates.json").write_text(json.dumps(out, indent=2))
    print("wrote results/gates.json")


if __name__ == "__main__":
    main()
