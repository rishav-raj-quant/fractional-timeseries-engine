"""Unit tests for Rough Volatility Engine."""

import numpy as np
import pytest
from fractional_engine.rough_volatility import (
    compute_fbm_covariance_matrix,
    simulate_fractional_brownian_motion,
    simulate_rough_bergomi,
    calibrate_hurst_from_realized_variance,
)


def test_fbm_covariance_properties():
    # Test symmetric positive-definiteness of covariance kernel
    cov = compute_fbm_covariance_matrix(n=50, h=0.3)
    assert np.allclose(cov, cov.T)
    eigenvals = np.linalg.eigvalsh(cov)
    assert np.all(eigenvals > 0.0)


def test_simulate_fractional_brownian_motion():
    t_grid, fbm = simulate_fractional_brownian_motion(n_steps=200, hurst=0.2, seed=42)
    assert len(fbm) == 201
    assert fbm[0] == 0.0
    assert len(t_grid) == 201


def test_rough_bergomi_simulation():
    res = simulate_rough_bergomi(n_steps=250, hurst=0.15, s0=100.0, seed=777)
    assert len(res.spot_price) == 251
    assert np.all(res.spot_price > 0.0)
    assert np.all(res.variance_path > 0.0)
    assert res.estimated_roughness < 0.40
