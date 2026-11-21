import numpy as np
import torch

from thermal_research.models.hybrid import FactorConfig, HybridRidge
from thermal_research.models.translator import SpecOperator
from thermal_research.physics.spectral import SpectralHeat

PITCH = 50e-6


def _synthetic(n=150, H=24, W=20, seed=0):
    rng = np.random.default_rng(seed)
    spec = SpectralHeat(H, W, PITCH)
    B = np.zeros((2, H, W))
    B[0, 3:9, 2:8] = 1.0
    B[1, 14:21, 10:17] = 1.0
    X = rng.normal(size=(n, 5))
    a = np.maximum(np.column_stack([1.0 + 0.8 * X[:, 0], 0.7 - 0.5 * X[:, 1]]), 0)
    q = np.einsum("nk,khw->nhw", a, B) * 3e5
    th = spec.steady(torch.tensor(q), torch.tensor(0.075), torch.tensor(1500.0)).numpy()
    yy, xx = np.mgrid[0:H, 0:W]
    bg = 0.5 * np.sin(yy / 6.0)  # static rig/emissivity pattern
    group = np.repeat(np.arange(5), n // 5)
    return X, (40.0 + 2 * X[:, 2:3, None] + bg[None] + th).astype(np.float32), group, (H, W)


import pytest  # noqa: E402


@pytest.mark.slow
def test_hybrid_ridge_fits_model_family_with_background():
    X, Y, group, hw = _synthetic()
    m = HybridRidge(lambda: SpecOperator(SpectralHeat(*hw, PITCH), 0.075, 1000.0), PITCH, 0.075, 1000.0,
                    FactorConfig(K=3, low_hw=(12, 10), bg_low_hw=(12, 10), steps=600, tv=0.0))
    m.fit(X[:120], Y[:120], group=group[:120])
    err = np.sqrt(((m.predict(X[120:]) - Y[120:]) ** 2).mean())
    assert m.chosen_["factor_rmse"] < 0.3, m.chosen_
    assert err < 0.25 * Y[120:].std(), (err, Y[120:].std())
    assert m.power_maps(X[120:]).min() >= 0
