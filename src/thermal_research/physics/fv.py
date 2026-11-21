"""Cell-centred finite-volume heat models (sparse direct solve). Reference for H-spec and the
simulator for H-net training data.

Units: G = k*t [W/K] (sheet conductance), h, g12 [W/m^2/K], q [W/m^2], pitch D [m].
Interior faces use the harmonic mean of the two cells' G (exact two-point flux for a material
interface on the face; verified 2nd-order in the v0 prototype). Edges are adiabatic unless h_edge > 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def lateral_operator(G: np.ndarray, pitch: float, h_edge: float = 0.0) -> sp.csr_matrix:
    """Sparse (N x N) operator of sum_nbrs Gf (th_P - th_Q) / D^2 for an (H, W) sheet-conductance field.
    Rows are per unit area (divided by the cell area D^2) so that A @ th has units W/m^2."""
    H, W = G.shape
    n = H * W
    idx = np.arange(n).reshape(H, W)
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    for a_sl, b_sl, Ga, Gb in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), G[:, :-1], G[:, 1:]),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), G[:-1, :], G[1:, :]),
    ):
        gf = (2 * Ga * Gb / (Ga + Gb)).ravel() / pitch**2
        a, b = idx[a_sl].ravel(), idx[b_sl].ravel()
        rows += [a, b]
        cols += [b, a]
        vals += [-gf, -gf]
        np.add.at(diag, a, gf)
        np.add.at(diag, b, gf)
    if h_edge > 0:  # Robin edges: half-cell conduction in series with film, per unit cell area
        for sl in ((slice(None), 0), (slice(None), -1), (0, slice(None)), (-1, slice(None))):
            Gc = G[sl]
            cond = 1.0 / (pitch / (2 * Gc) + 1.0 / h_edge) / pitch  # W/K per unit length / D
            np.add.at(diag, idx[sl].ravel(), (cond / pitch).ravel())
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


@dataclass
class OneLayer:
    """-div(G grad th) + h th = q on an (H, W) grid."""

    G: np.ndarray  # (H, W) or scalar broadcast by caller
    h: np.ndarray  # (H, W)
    pitch: float

    def __post_init__(self):
        H, W = self.G.shape
        self.shape = (H, W)
        A = lateral_operator(self.G, self.pitch) + sp.diags(self.h.ravel())
        self.A = A.tocsc()
        self._lu = spla.splu(self.A)

    def solve(self, q: np.ndarray) -> np.ndarray:
        return self._lu.solve(q.ravel().astype(np.float64)).reshape(self.shape)


@dataclass
class TwoLayer:
    """Die (layer 1, sheet conductance G1) over a spreader/package layer (G2), coupled vertically by g12
    [W/m^2/K]; layer 2 loses heat to the cold side through h2. Power enters layer 1.

        L1 th1 + g12 (th1 - th2)          = q
        L2 th2 + g12 (th2 - th1) + h2 th2 = 0
    """

    G1: float
    G2: float
    g12: float
    h2: float
    shape: tuple[int, int]
    pitch: float

    def __post_init__(self):
        H, W = self.shape
        n = H * W
        L1 = lateral_operator(np.full(self.shape, self.G1), self.pitch)
        L2 = lateral_operator(np.full(self.shape, self.G2), self.pitch)
        I = sp.identity(n, format="csr")
        A = sp.bmat([[L1 + self.g12 * I, -self.g12 * I], [-self.g12 * I, L2 + (self.g12 + self.h2) * I]])
        self.A = A.tocsc()
        self._lu = spla.splu(self.A)
        self.n = n

    def solve(self, q: np.ndarray, both: bool = False):
        rhs = np.concatenate([q.ravel().astype(np.float64), np.zeros(self.n)])
        x = self._lu.solve(rhs)
        th1 = x[: self.n].reshape(self.shape)
        if both:
            return th1, x[self.n :].reshape(self.shape)
        return th1
