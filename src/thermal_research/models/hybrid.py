"""Hybrid model with a physics latent space, fitted in two stages on calibration workloads only.

Stage 1, physics factorisation: calibration maps ~= T_base(t) + sum_k a_k(t) * H(B_k), with B_k >= 0
(smooth, low-res, learned), a_k(t) >= 0 and T_base(t) free per frame, plus the heat model's global
params (H-spec: cooling h). This learns the chip's "layout" (where power goes) and the physical state
of every calibration frame.

Stage 2, regression: tuned ridge from standardised telemetry to [a(t), T_base(t)], with the same grouped
inner CV as the RidgePCA baseline (scored in map space). Predictions clip a >= 0, so any extrapolation
stays a physically valid heat map: non-negative sources spread by conduction.

The fair comparison is with RidgePCA: same regressor, same CV, different latent space (physics vs PCA).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..training.calibrate import init_basis_from_inversion


@dataclass
class FactorConfig:
    K: int = 8
    low_hw: tuple[int, int] = (37, 31)
    steps: int = 800
    lr: float = 5e-2
    tv: float = 1e-2
    power_scale: float = 1e5
    background: bool = True  # static smooth map shared by all frames (emissivity / rig pattern)
    bg_low_hw: tuple[int, int] = (19, 16)
    seed: int = 0


def physics_factorise(Y: np.ndarray, heat, cfg: FactorConfig, init_B: torch.Tensor, ckpt_path=None,
                      ckpt_every: int = 100):
    """Returns (HB (K,P), B (K,H,W), a (n,K), T_base (n,), bg (P,), info), all numpy.

    With `ckpt_path`, optimisation state is saved atomically every `ckpt_every` steps and resumed if the
    file exists; the file is removed on completion.
    """
    from ..training import checkpoint as ck

    torch.manual_seed(cfg.seed)
    n, H, W = Y.shape
    y = torch.tensor(Y, dtype=torch.float64)
    U = torch.nn.Parameter(torch.log(torch.expm1(init_B.clamp_min(1e-4))).to(torch.float64))
    a_raw = torch.nn.Parameter(torch.full((n, cfg.K), -1.0, dtype=torch.float64))
    base = torch.nn.Parameter(torch.tensor(np.percentile(Y.reshape(n, -1), 2, axis=1), dtype=torch.float64))
    bg_low = torch.nn.Parameter(torch.zeros(1, 1, *cfg.bg_low_hw, dtype=torch.float64), requires_grad=cfg.background)
    params = [U, a_raw, base] + [p for p in heat.parameters() if p.requires_grad]
    if cfg.background:
        params.append(bg_low)
    opt = torch.optim.Adam(params, lr=cfg.lr)
    start = 0
    st = ck.load(ckpt_path)
    if st is not None and st.get("n") == n:
        with torch.no_grad():
            for p, v in zip(params, st["params"]):
                p.copy_(v)
        opt.load_state_dict(st["opt"])
        start = st["step"]
    for step in range(start, cfg.steps):
        if ckpt_path is not None and step > start and step % ckpt_every == 0:
            ck.save(ckpt_path, {"n": n, "step": step, "params": [p.detach().clone() for p in params],
                                "opt": opt.state_dict()})
        opt.zero_grad()
        B = F.interpolate(F.softplus(U)[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        B = B / B.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        HB = heat(B * cfg.power_scale)
        bg = F.interpolate(bg_low, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        bg = bg - bg.mean()  # zero-mean so it can't trade off with T_base
        pred = base[:, None, None] + bg + torch.einsum("nk,khw->nhw", F.softplus(a_raw), HB)
        tv = (B[:, 1:] - B[:, :-1]).abs().mean() + (B[:, :, 1:] - B[:, :, :-1]).abs().mean()
        loss = ((pred - y) ** 2).mean() + cfg.tv * tv
        loss.backward()
        opt.step()
    ck.clear(ckpt_path)
    with torch.no_grad():
        rmse = float(((pred - y) ** 2).mean().sqrt())
        info = {"factor_rmse": rmse}
        if hasattr(heat, "log_h"):
            info["h"] = float(torch.exp(heat.log_h))
        return (HB.reshape(cfg.K, -1).numpy(), B.numpy(), F.softplus(a_raw).numpy(), base.numpy(),
                bg.reshape(-1).numpy(), info)


@dataclass
class HybridRidge:
    """Protocol adapter: fit(X, Y, group=, val=) / predict(X)."""

    heat_factory: object
    pitch: float
    G: float
    h_init: float
    cfg: FactorConfig = field(default_factory=FactorConfig)
    lams: tuple[float, ...] = tuple(np.logspace(-4, 2, 13))
    ckpt_path: object = None
    chosen_: dict = field(default_factory=dict)

    def _ridge_fit(self, A, T, lam):
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        return Vt.T @ ((S / (S**2 + lam * len(A)))[:, None] * (U.T @ T))

    def _decode(self, Tp: np.ndarray) -> np.ndarray:
        a = np.maximum(Tp[:, : self.cfg.K], 0.0)
        return Tp[:, self.cfg.K][:, None] + self.bg_[None] + a @ self.HB_

    def fit(self, X, Y, group=None, val=None, **_):
        heat = self.heat_factory()
        init = init_basis_from_inversion(Y, self.cfg.K, self.cfg.low_hw, self.G, self.h_init, self.pitch, self.cfg.seed)
        self.HB_, self.B_, a, base, self.bg_, info = physics_factorise(Y, heat, self.cfg, init, self.ckpt_path)
        self.shape_ = Y.shape[1:]
        Yf = Y.reshape(len(Y), -1).astype(np.float64)
        T = np.column_stack([a, base])  # regression targets in physics space
        self.xmean_, self.tmean_ = X.mean(0), T.mean(0)
        A, Tc = X - self.xmean_, T - self.tmean_
        scores = np.zeros(len(self.lams))
        folds = []
        if group is not None and len(np.unique(group)) >= 2:
            folds = [(np.flatnonzero(group != g), np.flatnonzero(group == g)) for g in np.unique(group)]
        if folds:
            for tr, va in folds:
                for i, lam in enumerate(self.lams):
                    Wl = self._ridge_fit(A[tr], Tc[tr], lam)
                    pred = self._decode(A[va] @ Wl + self.tmean_)
                    scores[i] += ((pred - Yf[va]) ** 2).sum()
        elif val is not None:
            Av = val[0] - self.xmean_
            Yv = val[1].reshape(len(val[1]), -1)
            for i, lam in enumerate(self.lams):
                pred = self._decode(Av @ self._ridge_fit(A, Tc, lam) + self.tmean_)
                scores[i] = ((pred - Yv) ** 2).sum()
        lam = float(self.lams[int(np.argmin(scores))])
        self.W_ = self._ridge_fit(A, Tc, lam)
        self.chosen_ = {"lam": lam, **info}
        return self

    def predict(self, X, **_):
        Tp = (X - self.xmean_) @ self.W_ + self.tmean_
        return self._decode(Tp).reshape((len(X),) + tuple(self.shape_)).astype(np.float32)

    def power_maps(self, X) -> np.ndarray:
        Tp = (X - self.xmean_) @ self.W_ + self.tmean_
        a = np.maximum(Tp[:, : self.cfg.K], 0.0)
        return np.einsum("nk,khw->nhw", a, self.B_)


def h_from_ell(ell_mm: float, G: float) -> float:
    return G / (ell_mm * 1e-3) ** 2 if ell_mm > 0 else math.inf
