"""
Fractionally Differentiated Time Series Engine.

Implements memory-preserving fractional differencing based on the expanding window
binomial series expansion of (1 - B)^d:
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k,  k >= 1

Preserves maximum predictive memory while removing non-stationary stochastic trends.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class FracDiffResult:
    """Statistical summary of fractional differencing transformation."""
    d: float
    series: np.ndarray
    weights: np.ndarray
    memory_correlation: float
    adf_statistic: float
    adf_pvalue: float
    is_stationary: bool
    kpss_pvalue: Optional[float] = None


class FractionalDifferentiator:
    """
    Computes fractionally differentiated time series with memory thresholding.
    """

    def __init__(self, threshold: float = 1e-4, max_lags: Optional[int] = None) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be strictly positive.")
        self.threshold = float(threshold)
        self.max_lags = max_lags

    def compute_weights(self, d: float, length: int) -> np.ndarray:
        """
        Generates expanding window binomial weights w_k for fractional order d.
        Weights are truncated when |w_k| < threshold, bounded by max_lags or length // 2.
        """
        weights = [1.0]
        k = 1
        max_allowed = length // 2 if self.max_lags is None else min(length - 20, self.max_lags)
        limit = max(10, min(length, max_allowed))
        while k < limit:
            w_k = -weights[-1] * (d - k + 1.0) / k
            if abs(w_k) < self.threshold:
                break
            weights.append(w_k)
            k += 1
        return np.array(weights[::-1], dtype=np.float64)


    def transform(self, series: np.ndarray, d: float) -> FracDiffResult:
        """
        Applies fractional differencing of order d to a 1D sequence.
        Returns the differenced series aligned to valid observations,
        correlation with raw levels, and stationarity diagnostics.
        """
        x = np.asarray(series, dtype=np.float64).ravel()
        n = len(x)
        if n < 30:
            raise ValueError(f"Series length ({n}) is too short for reliable fractional differencing.")

        if abs(d) < 1e-7:
            # Order 0: identity transformation
            weights = np.array([1.0], dtype=np.float64)
            diff_series = x.copy()
            corr = 1.0
        else:
            w = self.compute_weights(d, n)
            w_len = len(w)
            # Convolution: valid overlap
            diff_vals = np.convolve(x, w, mode="valid")
            # Pad beginning with NaNs for full length alignment
            pad = np.full(w_len - 1, np.nan)
            diff_series = np.concatenate([pad, diff_vals])
            weights = w

        valid_mask = ~np.isnan(diff_series)
        valid_diff = diff_series[valid_mask]
        valid_raw = x[valid_mask]

        # Memory correlation between raw levels and differentiated values
        if np.std(valid_diff) > 1e-12 and np.std(valid_raw) > 1e-12:
            corr = float(np.corrcoef(valid_raw, valid_diff)[0, 1])
        else:
            corr = 0.0

        # Augmented Dickey-Fuller unit-root test
        adf_res = adfuller(valid_diff, maxlag=None, regression="c", autolag="AIC")
        adf_stat = float(adf_res[0])
        adf_pval = float(adf_res[1])
        is_stat = adf_pval < 0.05

        # KPSS test (stationary null hypothesis)
        try:
            kpss_res = kpss(valid_diff, regression="c", nlags="auto")
            kpss_pval = float(kpss_res[1])
        except Exception:
            kpss_pval = None

        return FracDiffResult(
            d=float(d),
            series=diff_series,
            weights=weights,
            memory_correlation=corr,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pval,
            is_stationary=is_stat,
            kpss_pvalue=kpss_pval,
        )


def find_optimal_d(
    series: np.ndarray,
    d_range: Tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,
    p_value_threshold: float = 0.05,
    threshold: float = 1e-4,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Scans candidate fractional integration orders d to find minimal d* that achieves
    stationarity (ADF p-value < threshold) while maximizing retained memory.
    """
    diff_engine = FractionalDifferentiator(threshold=threshold)
    d_grid = np.arange(d_range[0], d_range[1] + 1e-9, step)
    records: List[Dict[str, float]] = []

    optimal_d = d_range[1]
    found = False

    for d in d_grid:
        res = diff_engine.transform(series, d=float(d))
        records.append({
            "d": round(float(d), 4),
            "adf_pvalue": res.adf_pvalue,
            "adf_stat": res.adf_statistic,
            "correlation": res.memory_correlation,
            "stationary": 1.0 if res.is_stationary else 0.0,
        })
        if res.adf_pvalue < p_value_threshold and not found:
            optimal_d = float(d)
            found = True

    return optimal_d, records
