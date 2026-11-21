import os
import pickle

import numpy as np
import pytest

from thermal_research.data import features, splits
from thermal_research.data.safe_pickle import UnsafePickleError, loads_pickle


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_safe_pickle_rejects_code_execution():
    with pytest.raises(UnsafePickleError):
        loads_pickle(pickle.dumps(_Evil()))


def test_safe_pickle_loads_numpy_dict():
    d = {"input": np.arange(6.0).reshape(2, 3), "output": np.ones((2, 4, 4), np.float32)}
    out = loads_pickle(pickle.dumps(d))
    assert out["input"].shape == (2, 3) and out["output"].dtype == np.float32


def _groups(n_w=6, per=10):
    group = np.repeat(np.arange(n_w), per)
    frame = np.tile(np.arange(per), n_w)
    return group, frame


def test_calibration_split_disjoint_and_complete():
    group, frame = _groups()
    s = splits.calibration_split(group, frame, (1, 4))
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert len(np.unique(all_idx)) == len(group) == len(all_idx)
    assert set(np.unique(group[s.test])) == {0, 2, 3, 5}
    assert set(np.unique(group[np.concatenate([s.train, s.val])])) == {1, 4}
    # validation is the time tail of each calibration workload
    for g in (1, 4):
        assert frame[s.val][group[s.val] == g].min() > frame[s.train][group[s.train] == g].max()


def test_lowo_holds_out_exactly_one_workload():
    group, frame = _groups()
    s = splits.lowo(group, frame, held=3)
    assert set(np.unique(group[s.test])) == {3}
    assert 3 not in set(np.unique(group[np.concatenate([s.train, s.val])]))


def test_calibration_draws_deterministic_distinct():
    a = splits.calibration_draws(19, 3, 20)
    b = splits.calibration_draws(19, 3, 20)
    assert a == b and len(set(a)) == 20 and all(len(c) == 3 for c in a)
    assert splits.calibration_draws(5, 2, 50) == [c for c in splits.calibration_draws(5, 2, 50)]
    assert len(splits.calibration_draws(5, 2, 50)) == 10  # all combinations when fewer exist


def test_random_frames_disjoint():
    s = splits.random_frames(100, 0.2, seed=1)
    assert len(np.intersect1d(s.train, s.test)) == 0 and len(s.test) == 20


def test_sensor_rule_finds_integer_correlated_columns():
    rng = np.random.default_rng(0)
    n = 200
    temp = rng.uniform(40, 90, n)
    Y = temp[:, None, None] + rng.normal(0, 0.5, (n, 4, 4))
    X = np.column_stack([np.round(100 - temp), rng.normal(size=n), np.round(rng.uniform(0, 50, n))])
    cols = features.sensor_columns(X, Y, np.arange(n))
    assert cols.tolist() == [0]


def test_standardizer_uses_train_rows_and_drops_constants():
    X = np.column_stack([np.arange(10.0), np.ones(10), np.arange(10.0) * 2])
    st = features.Standardizer().fit(X[:5], drop=np.array([2]))
    assert st.keep.tolist() == [0]
    z = st.transform(X)
    assert np.isclose(z[:5].mean(), 0) and z.shape == (10, 1)


def test_windowed_is_causal():
    group, frame = _groups(n_w=2, per=4)
    X = np.arange(8.0)[:, None]
    W = features.windowed(X, group, frame, w=2)
    # second column is the previous frame (first frame repeats itself)
    assert W[:4, 1].tolist() == [0.0, 0.0, 1.0, 2.0]
    assert W[4:, 1].tolist() == [4.0, 4.0, 5.0, 6.0]
