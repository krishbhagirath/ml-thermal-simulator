"""Pure-ML baselines trained on the same calibration workloads as the hybrid.

All predict a residual to the calibration mean map, in units of a scalar std, and share one trainer
with early stopping on the calibration validation tail.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .baselines import _pca


class MLPPCA(nn.Module):
    def __init__(self, n_in: int, basis: torch.Tensor, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("basis", basis)  # (r, P)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, basis.shape[0]))

    def forward(self, x):
        return self.net(x) @ self.basis


class CNNDecoder(nn.Module):
    """MLP -> (C, 6, 5) seed -> 5 upsampling conv blocks -> (192, 160) -> crop."""

    def __init__(self, n_in: int, out_hw: tuple[int, int], width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.out_hw = out_hw
        self.fc = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, width * 30))
        self.width = width
        blocks = []
        for _ in range(5):
            blocks += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        self.up = nn.Sequential(*blocks, nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, x):
        z = self.fc(x).view(-1, self.width, 6, 5)
        y = self.up(z)[:, 0]
        H, W = self.out_hw
        return y[:, :H, :W].reshape(len(x), -1)


class FeatureTokenTransformer(nn.Module):
    """One token per telemetry feature (iTransformer/TherMapNet-style), mean-pooled, PCA head."""

    def __init__(self, n_in: int, basis: torch.Tensor, d: int = 32, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.register_buffer("basis", basis)
        self.val_w = nn.Parameter(torch.randn(n_in, d) * 0.1)
        self.feat_emb = nn.Parameter(torch.randn(n_in, d) * 0.1)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True,
                                         activation="gelu", norm_first=True)  # pre-norm keeps value magnitude
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, basis.shape[0])

    def forward(self, x):
        tok = x[:, :, None] * self.val_w[None] + self.feat_emb[None]
        return self.head(self.enc(tok).mean(1)) @ self.basis


class TorchRegressor:
    """Protocol adapter. kind in {'mlp', 'cnn', 'tf'}."""

    def __init__(self, kind: str, out_hw: tuple[int, int], rank: int = 20, max_epochs: int = 400,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch: int = 64, patience: int = 40, seed: int = 0):
        self.kind, self.out_hw, self.rank = kind, out_hw, rank
        self.max_epochs, self.lr, self.wd, self.batch, self.patience, self.seed = (
            max_epochs, lr, weight_decay, batch, patience, seed)
        self.ckpt_path = None  # set by the protocol runner; enables mid-fit resume
        self.ckpt_every = 10
        self.chosen_: dict = {}

    def _build(self, n_in: int, Yc: np.ndarray) -> nn.Module:
        if self.kind == "cnn":
            return CNNDecoder(n_in, self.out_hw)
        r = min(self.rank, len(Yc) - 1)
        basis = torch.tensor(_pca(Yc, r), dtype=torch.float32)
        return MLPPCA(n_in, basis) if self.kind == "mlp" else FeatureTokenTransformer(n_in, basis)

    def fit(self, X, Y, val=None, **_):
        torch.manual_seed(self.seed)
        n = len(X)
        Yf = Y.reshape(n, -1).astype(np.float64)
        self.mean_ = Yf.mean(0)
        self.std_ = float((Yf - self.mean_).std()) or 1.0
        Yc = (Yf - self.mean_) / self.std_
        self.net = self._build(X.shape[1], Yc)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Yc, dtype=torch.float32)
        if val is not None:
            xv = torch.tensor(val[0], dtype=torch.float32)
            yv = torch.tensor((val[1].reshape(len(val[1]), -1) - self.mean_) / self.std_, dtype=torch.float32)
        from ..training import checkpoint as ck

        g = torch.Generator().manual_seed(self.seed)
        best, best_state, since, start = np.inf, None, 0, 0
        st = ck.load(self.ckpt_path)
        if st is not None and st.get("n") == n:
            self.net.load_state_dict(st["net"])
            opt.load_state_dict(st["opt"])
            g.set_state(st["gen"])
            ck.set_rng_state(st["rng"])
            best, best_state, since, start = st["best"], st["best_state"], st["since"], st["epoch"]
        epoch = start - 1
        for epoch in range(start, self.max_epochs):
            if self.ckpt_path is not None and epoch > start and epoch % self.ckpt_every == 0:
                ck.save(self.ckpt_path, {"n": n, "net": self.net.state_dict(), "opt": opt.state_dict(),
                                         "gen": g.get_state(), "rng": ck.rng_state(), "best": best,
                                         "best_state": best_state, "since": since, "epoch": epoch})
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                b = perm[i : i + self.batch]
                opt.zero_grad()
                loss = F.mse_loss(self.net(x[b]), y[b])
                loss.backward()
                opt.step()
            if val is None:
                continue
            self.net.eval()
            with torch.no_grad():
                v = float(F.mse_loss(self.net(xv), yv))
            if v < best - 1e-5:
                best, since, best_state = v, 0, copy.deepcopy(self.net.state_dict())
            else:
                since += 1
                if since >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        ck.clear(self.ckpt_path)
        self.chosen_ = {"epochs": epoch + 1, "val_rmse_c": float(np.sqrt(best) * self.std_) if np.isfinite(best) else None,
                        "params": int(sum(p.numel() for p in self.net.parameters()))}
        return self

    @torch.no_grad()
    def predict(self, X, **_):
        self.net.eval()
        out = self.net(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)
        return (self.mean_ + out * self.std_).reshape((len(X),) + tuple(self.out_hw)).astype(np.float32)
