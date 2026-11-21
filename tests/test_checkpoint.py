import numpy as np
import pytest
import torch

from thermal_research.models.blackbox import TorchRegressor
from thermal_research.training import checkpoint as ck


def _data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    Y = (40 + 3 * X[:, :1, None] * np.ones((1, 6, 5))).astype(np.float32)
    return X, Y


def test_atomic_save_load_clear(tmp_path):
    p = tmp_path / "c.pt"
    ck.save(p, {"a": torch.ones(3)})
    assert torch.equal(ck.load(p)["a"], torch.ones(3))
    ck.clear(p)
    assert ck.load(p) is None


def test_json_progress_persists(tmp_path):
    jp = ck.JsonProgress(tmp_path / "p.json")
    jp.put("x", {"v": 1})
    assert "x" in ck.JsonProgress(tmp_path / "p.json")


def test_regressor_resume_after_crash_is_identical(tmp_path, monkeypatch):
    X, Y = _data()
    ref = TorchRegressor("mlp", (6, 5), rank=3, max_epochs=30, patience=100, seed=1)
    ref.fit(X[:60], Y[:60], val=(X[60:], Y[60:]))

    crashing = TorchRegressor("mlp", (6, 5), rank=3, max_epochs=30, patience=100, seed=1)
    crashing.ckpt_path = tmp_path / "fit.pt"
    crashing.ckpt_every = 5
    calls = {"n": 0}
    orig = torch.optim.AdamW.step

    def boom(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 17:  # die mid-training (after a checkpoint at epoch 10 or 15)
            raise KeyboardInterrupt
        return orig(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "step", boom)
    with pytest.raises(KeyboardInterrupt):
        crashing.fit(X[:60], Y[:60], val=(X[60:], Y[60:]))
    monkeypatch.setattr(torch.optim.AdamW, "step", orig)
    assert (tmp_path / "fit.pt").exists()

    resumed = TorchRegressor("mlp", (6, 5), rank=3, max_epochs=30, patience=100, seed=1)
    resumed.ckpt_path = tmp_path / "fit.pt"
    resumed.ckpt_every = 5
    resumed.fit(X[:60], Y[:60], val=(X[60:], Y[60:]))
    assert not (tmp_path / "fit.pt").exists()
    assert np.allclose(resumed.predict(X[60:]), ref.predict(X[60:]), atol=1e-6)
