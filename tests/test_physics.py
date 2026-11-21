import numpy as np
import pytest
import scipy.fft
import torch

from thermal_research.physics import fv, inversion, powermaps
from thermal_research.physics.dct import DCT2, dct_matrix
from thermal_research.physics.spectral import SpectralHeat

PITCH = 50e-6


def test_dct_matrix_matches_scipy_and_is_orthonormal():
    x = np.random.default_rng(0).normal(size=17)
    C = dct_matrix(17)
    assert np.allclose(C @ x, scipy.fft.dct(x, type=2, norm="ortho"))
    assert np.allclose(C @ C.T, np.eye(17))
    d = DCT2(9, 7)
    X = torch.randn(3, 9, 7, dtype=torch.float64)
    assert torch.allclose(d.inverse(d.forward(X)), X)


@pytest.mark.parametrize("H,W", [(12, 9), (37, 31)])
def test_spectral_equals_fv_one_layer(H, W):
    rng = np.random.default_rng(1)
    q = powermaps.random_power_map((H, W), rng, 1e5)
    G, h = 150 * 0.5e-3, 800.0
    ref = fv.OneLayer(np.full((H, W), G), np.full((H, W), h), PITCH).solve(q)
    sp = SpectralHeat(H, W, PITCH).steady(torch.tensor(q), torch.tensor(G), torch.tensor(h)).numpy()
    assert np.abs(sp - ref).max() / np.abs(ref).max() < 1e-10


def test_spectral_energy_balance_and_positivity():
    H, W = 20, 16
    q = powermaps.random_power_map((H, W), np.random.default_rng(2), 1e5)
    th = SpectralHeat(H, W, PITCH).steady(torch.tensor(q), torch.tensor(0.05), torch.tensor(500.0)).numpy()
    # adiabatic edges: all power leaves through the sink term
    assert abs((500.0 * th).sum() - q.sum()) / q.sum() < 1e-10
    assert th.min() > 0


def test_transient_converges_to_steady():
    H, W = 10, 8
    q = torch.tensor(powermaps.random_power_map((H, W), np.random.default_rng(3), 1e5))
    s = SpectralHeat(H, W, PITCH)
    G, h = torch.tensor(0.05), torch.tensor(500.0)
    th0 = torch.zeros(H, W, dtype=torch.float64)
    traj = s.transient(q.expand(400, H, W), G, h, C=torch.tensor(500.0), dt=0.5, theta0=th0)
    assert torch.allclose(traj[-1], s.steady(q, G, h), rtol=1e-6)
    assert traj[0].max() < traj[-1].max()  # heats up from zero


def test_spectral_is_differentiable_in_parameters():
    s = SpectralHeat(6, 5, PITCH)
    q = torch.rand(6, 5, dtype=torch.float64)
    logh = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
    th = s.steady(q, torch.tensor(0.05, dtype=torch.float64), torch.exp(logh))
    th.sum().backward()
    assert logh.grad is not None and logh.grad < 0  # stronger cooling -> cooler


def test_two_layer_energy_and_strong_coupling_limit():
    H, W = 15, 12
    q = powermaps.random_power_map((H, W), np.random.default_rng(4), 1e5)
    m = fv.TwoLayer(G1=0.05, G2=0.5, g12=2e4, h2=600.0, shape=(H, W), pitch=PITCH)
    th1, th2 = m.solve(q, both=True)
    assert abs((600.0 * th2).sum() - q.sum()) / q.sum() < 1e-10  # all power exits layer 2's bottom
    assert th1.mean() > th2.mean() > 0
    # very strong coupling + G2 tiny: behaves like one layer with conductance G1 and sink h2
    strong = fv.TwoLayer(G1=0.05, G2=1e-12, g12=1e12, h2=600.0, shape=(H, W), pitch=PITCH).solve(q)
    one = fv.OneLayer(np.full((H, W), 0.05), np.full((H, W), 600.0), PITCH).solve(q)
    assert np.abs(strong - one).max() / one.max() < 1e-5


def test_inversion_recovers_power_from_steady_solution():
    H, W = 24, 20
    G, h = 0.05, 500.0
    q = powermaps.random_power_map((H, W), np.random.default_rng(5), 1e5)
    th = SpectralHeat(H, W, PITCH).steady(torch.tensor(q), torch.tensor(G), torch.tensor(h)).numpy()
    qr = inversion.invert(th + 40.0, G, h, T_base=40.0, pitch=PITCH, sigma=0, clip=False)
    assert np.abs(qr - q).max() / q.max() < 1e-7  # G/D^2 ~ 2e7 amplifies round-off


def test_random_power_map_properties():
    q = powermaps.random_power_map((30, 20), np.random.default_rng(6), 2.0)
    assert q.min() >= 0 and np.isclose(q.mean(), 2.0)
