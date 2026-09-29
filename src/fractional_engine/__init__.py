"""
Fractional Time Series Engine: Memory-Preserving Differentiation, Multi-Scale Hurst Dynamics & Rough Volterra Paths.
"""

from .fracdiff import (
    FractionalDifferentiator,
    FracDiffResult,
    find_optimal_d,
)
from .hurst_estimators import (
    geweke_porter_hudak_hurst,
    detrended_fluctuation_analysis,
    rescaled_range_hurst,
    HurstResult,
)
from .rough_volatility import (
    simulate_fractional_brownian_motion,
    simulate_rough_bergomi,
    calibrate_hurst_from_realized_variance,
    RoughVolResult,
)
from .wavelet_multiscale import (
    modwt,
    modwt_mra,
    wavelet_variance_decomposition,
)

__version__ = "0.1.0"
__all__ = [
    "FractionalDifferentiator",
    "FracDiffResult",
    "find_optimal_d",
    "geweke_porter_hudak_hurst",
    "detrended_fluctuation_analysis",
    "rescaled_range_hurst",
    "HurstResult",
    "simulate_fractional_brownian_motion",
    "simulate_rough_bergomi",
    "calibrate_hurst_from_realized_variance",
    "RoughVolResult",
    "modwt",
    "modwt_mra",
    "wavelet_variance_decomposition",
]
