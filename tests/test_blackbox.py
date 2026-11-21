import numpy as np
import pytest

from thermal_research.models.blackbox import TorchRegressor


@pytest.mark.parametrize("kind", ["mlp", pytest.param("cnn", marks=pytest.mark.slow), "tf"])
def test_blackbox_learns_simple_mapping(kind):
    rng = np.random.default_rng(0)
    n, F_, H, W = 160, 6, 185, 154
    X = rng.normal(size=(n, F_)).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    bump = np.exp(-((yy - 60) ** 2 + (xx - 50) ** 2) / 400.0)
    Y = (45 + 5 * X[:, :1, None] * bump[None]).astype(np.float32)
    m = TorchRegressor(kind, (H, W), rank=5, max_epochs=60, patience=60, lr=3e-3)
    m.fit(X[:120], Y[:120], val=(X[120:140], Y[120:140]))
    p = m.predict(X[140:])
    assert p.shape == (20, H, W)
    base = np.sqrt(((Y[140:] - Y[:120].mean(0)) ** 2).mean())
    err = np.sqrt(((p - Y[140:]) ** 2).mean())
    assert err < 0.6 * base, (kind, err, base)
