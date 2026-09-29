"""Unit tests for MODWT Multiresolution Wavelet Engine."""

import numpy as np
import pytest
from fractional_engine.wavelet_multiscale import (
    modwt,
    modwt_mra,
    wavelet_variance_decomposition,
)


def test_modwt_length_and_mra_reconstruction():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(256)

    details, smooth = modwt_mra(x, level=3, wavelet="haar")
    assert len(details) == 3
    # Check additive reconstruction: X = sum(D_j) + S_J
    reconstructed = sum(details) + smooth
    assert np.allclose(x, reconstructed, atol=1e-10)


def test_wavelet_variance_decomposition():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(512)
    decomp = wavelet_variance_decomposition(x, level=4, wavelet="haar")
    assert "total_variance" in decomp
    assert "explained_ratio" in decomp
    # Total variance should match sum of component variances
    assert 0.85 <= decomp["explained_ratio"] <= 1.15
