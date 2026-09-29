"""Unit tests for Fractional Differencing Engine."""

import numpy as np
import pytest
from fractional_engine.fracdiff import (
    FractionalDifferentiator,
    find_optimal_d,
)


def test_fracdiff_weights_sum_and_decay():
    diff = FractionalDifferentiator(threshold=1e-5)
    # For d = 1, weights should be [1, -1]
    w1 = diff.compute_weights(d=1.0, length=20)
    assert len(w1) == 2
    assert np.isclose(w1[-1], 1.0)
    assert np.isclose(w1[-2], -1.0)

    # For d = 0.5, weights decay monotonically in absolute value
    w_half = diff.compute_weights(d=0.5, length=50)
    # The last element corresponds to lag 0 (1.0)
    assert np.isclose(w_half[-1], 1.0)
    assert w_half[-2] < 0.0


def test_fracdiff_identity_and_stationarity():
    rng = np.random.default_rng(42)
    # Generate random walk (non-stationary)
    rw = np.cumsum(rng.standard_normal(400)) + 100.0

    diff = FractionalDifferentiator()
    # d = 0 preserves original series
    res_0 = diff.transform(rw, d=0.0)
    assert np.isclose(res_0.memory_correlation, 1.0)
    assert not res_0.is_stationary

    # d = 1 is standard first difference (stationary)
    res_1 = diff.transform(rw, d=1.0)
    assert res_1.is_stationary
    assert res_1.adf_pvalue < 0.05


def test_find_optimal_d():
    rng = np.random.default_rng(123)
    rw = np.cumsum(rng.standard_normal(500)) + 50.0
    optimal_d, records = find_optimal_d(rw, d_range=(0.0, 1.0), step=0.1)

    assert 0.0 < optimal_d <= 1.0
    assert len(records) > 0
    # Optimal d should achieve stationarity
    opt_record = [r for r in records if r["d"] == round(optimal_d, 4)][0]
    assert opt_record["stationary"] == 1.0
