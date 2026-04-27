import numpy as np

from core.diffusion_ops import (
    apply_family_diffusion,
    apply_orientation_diffusion,
    diffuse_operator_scores,
    laplacian_family_1d,
    laplacian_orientation_1d,
)


def test_laplacian_wrap_conserves_sum():
    x = np.array([0.0, 1.0, 0.0, 0.0])
    lap = laplacian_family_1d(x, mode="wrap")
    assert abs(float(np.sum(lap))) < 1e-12


def test_apply_diffusion_zero_coeff_no_change():
    x = np.array([0.0, 1.0, 0.0, 0.0])
    y, diff, lap = apply_family_diffusion(x, coeff=0.0, mode="wrap")
    assert np.allclose(y, x)
    assert np.allclose(diff, 0.0)
    assert np.allclose(lap, 0.0)


def test_apply_diffusion_spreads_mass_locally():
    x = np.array([0.0, 1.0, 0.0, 0.0])
    y, diff, lap = apply_family_diffusion(x, coeff=0.1, mode="wrap")
    assert y[1] < x[1]
    assert y[0] > x[0]
    assert y[2] > x[2]


def test_orientation_laplacian_conserves_sum():
    x = np.array([0.0, 1.0, 0.0, 0.0])
    lap = laplacian_orientation_1d(x)
    assert abs(float(np.sum(lap))) < 1e-12


def test_apply_orientation_diffusion_zero_coeff_no_change():
    x = np.array([0.0, 1.0, 0.0, 0.0])
    y, diff, lap = apply_orientation_diffusion(x, coeff=0.0)
    assert np.allclose(y, x)
    assert np.allclose(diff, 0.0)
    assert np.allclose(lap, 0.0)


def test_apply_orientation_diffusion_spreads_mass():
    x = np.array([0.0, 1.0, 0.0, 0.0])
    y, diff, lap = apply_orientation_diffusion(x, coeff=0.1)
    assert y[1] < x[1]
    assert y[0] > x[0]
    assert y[2] > x[2]


def test_orientation_score_diffusion_coeff_zero_preserves_scores():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    out = diffuse_operator_scores(s, coeff=0.0)
    assert np.allclose(out, s)


def test_orientation_score_diffusion_smooths_scores_deterministically():
    # Neighbor mean excludes self (complete-graph diffusion).
    s = np.array([10.0, 0.0, 0.0, 0.0])
    out = diffuse_operator_scores(s, coeff=0.5)
    # For i=0, neighbors mean is 0 -> 10 + 0.5*(0-10) = 5
    assert abs(float(out[0]) - 5.0) < 1e-12
    # For i>0, neighbors mean is (10+0+0)/3 = 10/3 -> 0 + 0.5*(10/3) = 5/3
    assert abs(float(out[1]) - (5.0 / 3.0)) < 1e-12
    assert abs(float(out[2]) - (5.0 / 3.0)) < 1e-12
    assert abs(float(out[3]) - (5.0 / 3.0)) < 1e-12
