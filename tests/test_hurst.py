"""Unit tests for Hurst Exponent Estimators."""

import numpy as np
import pytest
from fractional_engine.hurst_estimators import (
    geweke_porter_hudak_hurst,
    detrended_fluctuation_analysis,
    rescaled_range_hurst,
)
from fractional_engine.rough_volatility import simulate_fractional_brownian_motion


def test_hurst_estimators_on_brownian_motion():
    rng = np.random.default_rng(42)
    # Standard random walk increments have H ~ 0.5
    rw = np.cumsum(rng.standard_normal(1000))

    dfa_res = detrended_fluctuation_analysis(rw, order=1)
    # DFA on integrated random walk yields alpha ~ 1.5, or on increments ~ 0.5
    # Let's test on increments (white noise)
    noise = rng.standard_normal(1000)
    dfa_noise = detrended_fluctuation_analysis(noise, order=1)
    assert 0.40 <= dfa_noise.hurst <= 0.60
    assert dfa_noise.r_squared > 0.90

    rs_noise = rescaled_range_hurst(noise)
    assert 0.40 <= rs_noise.hurst <= 0.65

    gph_noise = geweke_porter_hudak_hurst(noise)
    assert 0.35 <= gph_noise.hurst <= 0.65


def test_hurst_subdiffusive_fbm():
    # Simulate sub-diffusive rough fBM with H = 0.25
    _, fbm_path = simulate_fractional_brownian_motion(n_steps=400, hurst=0.25, seed=101)
    increments = np.diff(fbm_path)
    dfa = detrended_fluctuation_analysis(increments, order=1)
    # Estimated H should be significantly < 0.5
    assert dfa.hurst < 0.45
