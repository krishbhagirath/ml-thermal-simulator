import numpy as np
import pytest

from thermal_research.data import ucr
from thermal_research.evaluation import metrics, stats
from thermal_research.models.baselines import CopyNearest, MeanMap, RidgePCA


def test_metrics_hand_cases():
    true = np.zeros((1, 5, 5))
    true[0, 1, 1] = 10.0
    pred = np.zeros((1, 5, 5))
    pred[0, 4, 1] = 8.0
    assert np.isclose(metrics.peak_error(pred, true)[0], 2.0)
    assert np.isclose(metrics.hotspot_distance_mm(pred, true, 0.05)[0], 0.15)
    assert np.isclose(metrics.per_map_rmse(true, true)[0], 0.0)
    assert np.isclose(metrics.map_corr(true, true)[0], 1.0)
    assert np.isclose(metrics.hotspot_iou(true, true)[0], 1.0)


def test_hier_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    vals = {d: {w: float(rng.normal(2.0, 0.3)) for w in range(10)} for d in range(8)}
    r = stats.hier_bootstrap(vals, n_boot=500)
    assert r["lo"] <= r["mean"] <= r["hi"] and abs(r["mean"] - 2.0) < 0.2


def test_paired_diff_sign():
    a = {0: {0: 1.0, 1: 1.0, 2: 1.2}, 1: {0: 0.9, 1: 1.1}}
    b = {0: {0: 2.0, 1: 2.1, 2: 2.0}, 1: {0: 2.0, 1: 2.2}}
    r = stats.paired_diff(a, b, n_boot=200)
    assert r["hi"] < 0 and r["frac_a_better"] == 1.0


def test_ridge_recovers_low_rank_linear_map():
    rng = np.random.default_rng(1)
    n, F, H, W = 300, 8, 6, 5
    X = rng.normal(size=(n, F))
    basis = rng.normal(size=(3, H * W))
    coef = X[:, :3] @ np.diag([2.0, -1.0, 0.5])
    Y = (coef @ basis + 50).reshape(n, H, W) + rng.normal(0, 0.01, (n, H, W))
    group = np.repeat(np.arange(6), n // 6)
    m = RidgePCA(ranks=(3, 5)).fit(X[:240], Y[:240], group=group[:240])
    err = np.sqrt(((m.predict(X[240:]) - Y[240:]) ** 2).mean())
    assert err < 0.05


def test_copy_nearest_and_mean():
    Y = np.arange(4.0)[:, None, None] * np.ones((4, 2, 2))
    group = np.zeros(4, int)
    frame = np.arange(4)
    cn = CopyNearest().fit(None, Y[[0, 2]], group=group[[0, 2]], frame=frame[[0, 2]])
    p = cn.predict(np.zeros((2, 1)), group=group[[1, 3]], frame=frame[[1, 3]])
    assert p[1, 0, 0] == 2.0
    assert MeanMap().fit(None, Y).predict(np.zeros((3, 1))).shape == (3, 2, 2)


@pytest.mark.data
@pytest.mark.skipif(not ucr.raw_available("i7"), reason="UCR excerpt not downloaded")
def test_i7_excerpt_shapes():
    cd = ucr.load_chip("i7")
    assert cd.X.shape == (1900, 170) and cd.shape == (185, 154) and len(cd.workloads) == 19
    assert np.all(np.bincount(cd.group) == 100)
