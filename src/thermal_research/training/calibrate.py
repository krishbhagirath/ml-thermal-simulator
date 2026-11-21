"""Fit the translator (and the heat model's global params) on calibration workloads only."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from ..models.translator import HeatOperator, Translator
from ..physics.inversion import invert


@dataclass
class CalibConfig:
    K: int = 8
    hidden: int = 0
    low_hw: tuple[int, int] = (37, 31)
    steps: int = 1500
    lr: float = 3e-2
    tv: float = 1e-2
    l1: float = 1e-4
    patience: int = 200
    seed: int = 0
    power_scale: float = 1e5  # W/m^2 per unit coefficient (relative; D-006)


def init_basis_from_inversion(Y: np.ndarray, K: int, low_hw: tuple[int, int], G: float, h: float,
                              pitch: float, seed: int = 0) -> torch.Tensor:
    """NMF of Laplacian-inverted calibration maps -> K non-negative low-res basis maps."""
    from scipy.ndimage import zoom

    T_base = float(np.percentile(Y, 1))
    Q = invert(Y.astype(np.float64), G, h, T_base, pitch, sigma=2.0)
    n, H, W = Q.shape
    Ql = np.stack([zoom(q, (low_hw[0] / H, low_hw[1] / W), order=1) for q in Q])
    Ql = np.maximum(Ql.reshape(n, -1), 0) + 1e-9
    rng = np.random.default_rng(seed)
    Wm = rng.uniform(0.1, 1.0, (n, K))
    Hm = rng.uniform(0.1, 1.0, (K, Ql.shape[1]))
    for _ in range(200):  # multiplicative-update NMF (Lee & Seung)
        Hm *= (Wm.T @ Ql) / (Wm.T @ Wm @ Hm + 1e-12)
        Wm *= (Ql @ Hm.T) / (Wm @ Hm @ Hm.T + 1e-12)
    B = Hm.reshape(K, *low_hw)
    B = B / B.mean(axis=(1, 2), keepdims=True)
    return torch.tensor(B)


class TranslatorModel:
    """Adapter to the evaluation-protocol interface (fit / predict on standardized numpy features)."""

    def __init__(self, heat_factory, cfg: CalibConfig, out_hw: tuple[int, int], pitch: float, G: float,
                 h_init: float):
        self.heat_factory, self.cfg, self.out_hw = heat_factory, cfg, out_hw
        self.pitch, self.G, self.h_init = pitch, G, h_init
        self.chosen_: dict = {}

    def fit(self, X, Y, val=None, **_):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        heat: HeatOperator = self.heat_factory()
        init = init_basis_from_inversion(Y, cfg.K, cfg.low_hw, self.G, self.h_init, self.pitch, cfg.seed)
        model = Translator(X.shape[1], cfg.K, self.out_hw, cfg.low_hw, cfg.hidden, init,
                           t_base_init=float(np.percentile(Y, 5))).double()
        params = list(model.parameters()) + [p for p in heat.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=cfg.lr)
        z = torch.tensor(X)
        y = torch.tensor(Y, dtype=torch.float64)
        zv = yv = None
        if val is not None:
            zv, yv = torch.tensor(val[0]), torch.tensor(val[1], dtype=torch.float64)
        best, best_state, since = np.inf, None, 0
        for step in range(cfg.steps):
            opt.zero_grad()
            pred, aux = model(z, heat, cfg.power_scale)
            loss = ((pred - y) ** 2).mean() + cfg.tv * Translator.tv(aux["B"]) + cfg.l1 * aux["a"].mean()
            loss.backward()
            opt.step()
            if zv is not None and (step % 10 == 0 or step == cfg.steps - 1):
                with torch.no_grad():
                    v = float(((model(zv, heat, cfg.power_scale)[0] - yv) ** 2).mean().sqrt())
                if v < best - 1e-4:
                    best, since = v, 0
                    best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(heat.state_dict()))
                else:
                    since += 10
                    if since >= cfg.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state[0])
            heat.load_state_dict(best_state[1])
        self.model, self.heat = model, heat
        self.chosen_ = {"val_rmse": float(best) if np.isfinite(best) else None, "steps": step + 1,
                        "h": float(torch.exp(heat.log_h)) if hasattr(heat, "log_h") else None}
        return self

    @torch.no_grad()
    def predict(self, X, **_):
        return self.model(torch.tensor(X), self.heat, self.cfg.power_scale)[0].numpy().astype(np.float32)

    @torch.no_grad()
    def power_maps(self, X) -> np.ndarray:
        _, aux = self.model(torch.tensor(X), self.heat, self.cfg.power_scale)
        return torch.einsum("nk,khw->nhw", aux["a"], aux["B"]).numpy()
