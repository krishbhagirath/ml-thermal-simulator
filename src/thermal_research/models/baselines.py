"""Non-deep-learning baselines: mean map, nearest-in-time copy (leakage diagnostic), tuned ridge->PCA."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class MeanMap:
    def fit(self, X: np.ndarray | None, Y: np.ndarray, **_) -> "MeanMap":
        self.mean_ = Y.mean(0)
        return self

    def predict(self, X: np.ndarray, **_) -> np.ndarray:
        return np.broadcast_to(self.mean_, (len(X),) + self.mean_.shape).copy()


class CopyNearest:
    """Predict each test frame with the training frame of the same workload nearest in time.

    Only meaningful when test frames share workloads with training frames (S-RAND). It is a leakage
    diagnostic: it uses no telemetry at all.
    """

    def fit(self, X, Y, group, frame, **_) -> "CopyNearest":
        self.Y_, self.group_, self.frame_ = Y, group, frame
        return self

    def predict(self, X, group, frame, **_) -> np.ndarray:
        out = np.empty((len(X),) + self.Y_.shape[1:], dtype=self.Y_.dtype)
        for i, (g, f) in enumerate(zip(group, frame)):
            cand = np.flatnonzero(self.group_ == g)
            if len(cand) == 0:
                raise ValueError("CopyNearest needs the test workload in training data")
            out[i] = self.Y_[cand[np.argmin(np.abs(self.frame_[cand] - f))]]
        return out


def _pca(Yc: np.ndarray, rank: int) -> np.ndarray:
    """Top-`rank` principal directions (rank, P) via the n x n Gram matrix (n << P)."""
    gram = Yc @ Yc.T
    w, v = np.linalg.eigh(gram)
    order = np.argsort(w)[::-1][:rank]
    w, v = np.maximum(w[order], 1e-12), v[:, order]
    return (v.T @ Yc) / np.sqrt(w)[:, None]


@dataclass
class RidgePCA:
    """Ridge regression from standardized features to the top-k PCA coefficients of the maps.

    lambda and k are chosen by inner cross-validation over training *workloads* (leave-one-workload-out)
    when >= 2 workloads are present, else by the provided validation rows. lambda is relative:
    the penalty is lam * n_train.
    """

    lams: tuple[float, ...] = tuple(np.logspace(-4, 2, 13))
    ranks: tuple[int, ...] = (5, 10, 20, 40)
    chosen_: dict = field(default_factory=dict)

    @staticmethod
    def _solve(A: np.ndarray, Z: np.ndarray, lam_abs: float) -> np.ndarray:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        return Vt.T @ ((S / (S**2 + lam_abs))[:, None] * (U.T @ Z))

    def _score(self, A, Yc, basis, tr, va) -> np.ndarray:
        """Returns a (n_lams, n_ranks) array of validation RMSE."""
        Z = Yc[tr] @ basis.T
        U, S, Vt = np.linalg.svd(A[tr], full_matrices=False)
        UtZ = U.T @ Z
        out = np.empty((len(self.lams), len(self.ranks)))
        for i, lam in enumerate(self.lams):
            W = Vt.T @ ((S / (S**2 + lam * len(tr)))[:, None] * UtZ)
            Zp = A[va] @ W
            for j, r in enumerate(self.ranks):
                pred = Zp[:, :r] @ basis[:r]
                out[i, j] = np.sqrt(((pred - Yc[va]) ** 2).mean())
        return out

    def fit(self, X: np.ndarray, Y: np.ndarray, group: np.ndarray | None = None,
            val: tuple[np.ndarray, np.ndarray] | None = None, **_) -> "RidgePCA":
        n = len(X)
        self.shape_ = Y.shape[1:]
        Yf = Y.reshape(n, -1).astype(np.float64)
        self.ymean_ = Yf.mean(0)
        Yc = Yf - self.ymean_
        max_rank = min(max(self.ranks), n - 1)
        self.ranks = tuple(r for r in self.ranks if r <= max_rank) or (max_rank,)
        basis = _pca(Yc, max(self.ranks))
        self.xmean_ = X.mean(0)
        A = X - self.xmean_  # intercept handled by centring both sides
        scores = np.zeros((len(self.lams), len(self.ranks)))
        if group is not None and len(np.unique(group)) >= 2:
            for g in np.unique(group):
                scores += self._score(A, Yc, basis, np.flatnonzero(group != g), np.flatnonzero(group == g))
        elif val is not None:
            Xv, Yv = val
            A2 = np.vstack([A, Xv - self.xmean_])
            Yc2 = np.vstack([Yc, Yv.reshape(len(Yv), -1) - self.ymean_])
            scores = self._score(A2, Yc2, basis, np.arange(n), np.arange(n, n + len(Xv)))
        else:
            scores[:, -1] = 0.0  # no CV possible: smallest-penalty fallback below
        i, j = np.unravel_index(np.argmin(scores), scores.shape)
        self.chosen_ = {"lam": float(self.lams[i]), "rank": int(self.ranks[j])}
        r = self.chosen_["rank"]
        self.basis_ = basis[:r]
        self.W_ = self._solve(A, Yc @ self.basis_.T, self.chosen_["lam"] * n)
        return self

    def predict(self, X: np.ndarray, **_) -> np.ndarray:
        Yp = self.ymean_ + ((X - self.xmean_) @ self.W_) @ self.basis_
        return Yp.reshape((len(X),) + self.shape_).astype(np.float32)
