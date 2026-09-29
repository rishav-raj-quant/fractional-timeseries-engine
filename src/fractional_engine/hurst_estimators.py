"""
Hurst Exponent & Long-Range Dependence Estimators.

Implements three classical and non-parametric estimators:
1. Geweke-Porter-Hudak (GPH) Semiparametric Spectral Log-Periodogram Estimator
2. Detrended Fluctuation Analysis (DFA) Order-1 & Order-2 Detrending
3. Rescaled Range (R/S) Classical Mandelbrot-Wallis Estimator
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass(frozen=True)
class HurstResult:
    """Estimated Hurst parameter and regression diagnostics."""
    hurst: float
    method: str
    r_squared: float
    standard_error: float
    scales: Optional[np.ndarray] = None
    fluctuations: Optional[np.ndarray] = None


def geweke_porter_hudak_hurst(
    series: np.ndarray,
    bandwidth_exponent: float = 0.55,
) -> HurstResult:
    """
    Geweke-Porter-Hudak (1983) semiparametric log-periodogram estimator.
    Regresses log-periodogram on harmonic frequencies:
        ln(I(lambda_j)) = c - 2d * ln|2 * sin(lambda_j / 2)| + e_j
    Returns H = d + 0.5.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 64:
        raise ValueError(f"Series length ({n}) too small for GPH estimator.")

    # Remove mean
    x_centered = x - np.mean(x)

    # Compute periodogram at fundamental Fourier frequencies
    dft = np.fft.rfft(x_centered)[1:]
    periodogram = (np.abs(dft) ** 2) / (2.0 * np.pi * n)

    # Number of harmonic frequencies m = n^bandwidth_exponent
    m = int(np.floor(n ** bandwidth_exponent))
    m = max(5, min(m, len(periodogram)))

    j_indices = np.arange(1, m + 1, dtype=np.float64)
    lambda_j = 2.0 * np.pi * j_indices / n

    regressor = -2.0 * np.log(2.0 * np.sin(lambda_j / 2.0))
    dependent = np.log(periodogram[:m] + 1e-12)

    # OLS regression
    x_mean = np.mean(regressor)
    y_mean = np.mean(dependent)
    ss_xx = np.sum((regressor - x_mean) ** 2)
    ss_xy = np.sum((regressor - x_mean) * (dependent - y_mean))

    if ss_xx < 1e-12:
        return HurstResult(hurst=0.5, method="GPH", r_squared=0.0, standard_error=1.0)

    d_hat = ss_xy / ss_xx
    intercept = y_mean - d_hat * x_mean

    residuals = dependent - (intercept + d_hat * regressor)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((dependent - y_mean) ** 2)
    r2 = max(0.0, 1.0 - (ss_res / (ss_tot + 1e-12)))

    # Theoretical asymptotic variance of GPH estimator is pi^2 / (6 * ss_xx)
    se_d = np.sqrt((np.pi ** 2) / (6.0 * ss_xx))

    h_hat = float(d_hat + 0.5)

    return HurstResult(
        hurst=h_hat,
        method="GPH",
        r_squared=float(r2),
        standard_error=float(se_d),
        scales=lambda_j,
        fluctuations=periodogram[:m],
    )


def detrended_fluctuation_analysis(
    series: np.ndarray,
    order: int = 1,
    min_scale: int = 8,
    max_scale: Optional[int] = None,
    num_scales: int = 20,
) -> HurstResult:
    """
    Detrended Fluctuation Analysis (DFA) (Peng et al., 1994).
    Calculates root-mean-square fluctuations F(s) around local polynomial trends
    across scales s in [min_scale, max_scale].
    Scaling law: F(s) ~ s^alpha, where alpha = H.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 100:
        raise ValueError(f"Series length ({n}) too small for reliable DFA.")

    if max_scale is None:
        max_scale = n // 4

    if min_scale >= max_scale:
        raise ValueError(f"min_scale ({min_scale}) must be < max_scale ({max_scale}).")

    # Cumulative integrated profile
    y = np.cumsum(x - np.mean(x))

    # Logarithmically spaced scales
    scales = np.unique(
        np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_scales).astype(int)
    )

    fluctuations: List[float] = []

    for s in scales:
        n_segments = n // s
        f_sq = 0.0

        for i in range(n_segments):
            seg_y = y[i * s : (i + 1) * s]
            seg_x = np.arange(s)
            poly = np.polyfit(seg_x, seg_y, deg=order)
            trend = np.polyval(poly, seg_x)
            f_sq += np.sum((seg_y - trend) ** 2)

        f_s = np.sqrt(f_sq / (n_segments * s))
        fluctuations.append(f_s)

    log_scales = np.log(scales)
    log_f = np.log(np.array(fluctuations) + 1e-12)

    # Linear regression in log-log space
    slope, intercept = np.polyfit(log_scales, log_f, deg=1)

    predicted = slope * log_scales + intercept
    ss_res = np.sum((log_f - predicted) ** 2)
    ss_tot = np.sum((log_f - np.mean(log_f)) ** 2)
    r2 = max(0.0, 1.0 - (ss_res / (ss_tot + 1e-12)))
    se = np.sqrt(ss_res / max(1, len(scales) - 2)) / (np.std(log_scales) * np.sqrt(len(scales)))

    return HurstResult(
        hurst=float(slope),
        method=f"DFA-{order}",
        r_squared=float(r2),
        standard_error=float(se),
        scales=scales.astype(float),
        fluctuations=np.array(fluctuations, dtype=float),
    )


def rescaled_range_hurst(
    series: np.ndarray,
    min_chunk: int = 16,
    num_chunks: int = 15,
) -> HurstResult:
    """
    Classical Mandelbrot Rescaled Range (R/S) Hurst exponent estimator.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < 100:
        raise ValueError("Series length must be at least 100 for R/S estimation.")

    max_chunk = n // 2
    chunk_sizes = np.unique(
        np.logspace(np.log10(min_chunk), np.log10(max_chunk), num=num_chunks).astype(int)
    )

    rs_means: List[float] = []

    for s in chunk_sizes:
        num_blocks = n // s
        rs_block = []
        for b in range(num_blocks):
            sub = x[b * s : (b + 1) * s]
            std_dev = np.std(sub, ddof=1)
            if std_dev < 1e-9:
                continue
            centered = sub - np.mean(sub)
            y = np.cumsum(centered)
            range_val = np.max(y) - np.min(y)
            rs_block.append(range_val / std_dev)

        if len(rs_block) > 0:
            rs_means.append(float(np.mean(rs_block)))
        else:
            rs_means.append(1e-6)

    log_sizes = np.log(chunk_sizes)
    log_rs = np.log(np.array(rs_means) + 1e-12)

    slope, intercept = np.polyfit(log_sizes, log_rs, deg=1)
    pred = slope * log_sizes + intercept
    ss_res = np.sum((log_rs - pred) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r2 = max(0.0, 1.0 - (ss_res / (ss_tot + 1e-12)))
    se = np.sqrt(ss_res / max(1, len(chunk_sizes) - 2)) / (np.std(log_sizes) * np.sqrt(len(chunk_sizes)))

    return HurstResult(
        hurst=float(slope),
        method="R/S",
        r_squared=float(r2),
        standard_error=float(se),
        scales=chunk_sizes.astype(float),
        fluctuations=np.array(rs_means, dtype=float),
    )
