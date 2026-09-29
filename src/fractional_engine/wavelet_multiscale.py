"""
Multiresolution Wavelet Analysis for Non-Stationary Time Series.

Implements the Maximal Overlap Discrete Wavelet Transform (MODWT):
- Shift-invariant, undecimated dyadic filter banks (works for any sample length N).
- Multiresolution Analysis (MRA) detail signals D_1, ..., D_J and smooth signal S_J.
- Zero-phase shift energy conservation: Var(X) = sum_j Var(D_j) + Var(S_J).
"""

from typing import Tuple, List, Dict
import numpy as np


# Daubechies 4 (D4) normalized MODWT filter coefficients
_D4_H = np.array([
    (1.0 - np.sqrt(3.0)) / (4.0 * np.sqrt(2.0)),
    (3.0 - np.sqrt(3.0)) / (4.0 * np.sqrt(2.0)),
    (3.0 + np.sqrt(3.0)) / (4.0 * np.sqrt(2.0)),
    (1.0 + np.sqrt(3.0)) / (4.0 * np.sqrt(2.0)),
], dtype=np.float64) / np.sqrt(2.0)

# Quadrature mirror scaling filter: g_l = (-1)^l * h_{L - 1 - l}
_D4_G = np.array([
    (-1.0) ** l * _D4_H[len(_D4_H) - 1 - l] for l in range(len(_D4_H))
], dtype=np.float64)

# Haar wavelet filter coefficients
_HAAR_H = np.array([0.5, 0.5], dtype=np.float64)
_HAAR_G = np.array([-0.5, 0.5], dtype=np.float64)


def modwt(
    series: np.ndarray,
    level: int = 4,
    wavelet: str = "haar",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Computes Maximal Overlap Discrete Wavelet Transform (MODWT) up to scale J.
    Returns (wavelet_details [W_1, ..., W_J], scaling_coefficients V_J).
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    if n < (2 ** level):
        raise ValueError(f"Series length {n} too small for MODWT level {level}")

    if wavelet.lower() == "haar":
        h_filt, g_filt = _HAAR_H, _HAAR_G
    else:
        h_filt, g_filt = _D4_H, _D4_G

    l_filt = len(h_filt)
    w_details: List[np.ndarray] = []
    v_curr = x.copy()

    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        w_j = np.zeros(n, dtype=np.float64)
        v_next = np.zeros(n, dtype=np.float64)

        for t in range(n):
            val_w = 0.0
            val_v = 0.0
            for l in range(l_filt):
                idx = (t - l * step) % n
                val_w += g_filt[l] * v_curr[idx]
                val_v += h_filt[l] * v_curr[idx]
            w_j[t] = val_w
            v_next[t] = val_v

        w_details.append(w_j)
        v_curr = v_next

    return w_details, v_curr


def modwt_mra(
    series: np.ndarray,
    level: int = 4,
    wavelet: str = "haar",
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Additive Multiresolution Analysis (MRA) using MODWT:
        X(t) = sum_{j=1}^J D_j(t) + S_J(t)
    where D_j are detail signals and S_J is the smooth trend.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    n = len(x)
    w_details, v_j = modwt(x, level=level, wavelet=wavelet)

    if wavelet.lower() == "haar":
        h_filt, g_filt = _HAAR_H, _HAAR_G
    else:
        h_filt, g_filt = _D4_H, _D4_G

    l_filt = len(h_filt)
    d_details: List[np.ndarray] = []

    # Synthesize detail signals D_j
    for j in range(1, level + 1):
        w_j = w_details[j - 1]
        step = 2 ** (j - 1)
        d_j = np.zeros(n, dtype=np.float64)
        for t in range(n):
            val = 0.0
            for l in range(l_filt):
                idx = (t + l * step) % n
                val += g_filt[l] * w_j[idx]
            d_j[t] = val
        d_details.append(d_j)

    # Synthesize smooth trend S_J
    s_j = x - sum(d_details)
    return d_details, s_j


def wavelet_variance_decomposition(
    series: np.ndarray,
    level: int = 4,
    wavelet: str = "haar",
) -> Dict[str, float]:
    """
    Calculates multi-scale variance decomposition across dyadic scales tau_j = 2^{j-1}.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    total_var = float(np.var(x, ddof=1))
    w_details, v_j = modwt(x, level=level, wavelet=wavelet)

    decomp: Dict[str, float] = {"total_variance": total_var}
    sum_detail_var = 0.0

    for j, w_j in enumerate(w_details, start=1):
        var_j = float(np.var(w_j, ddof=1))
        decomp[f"scale_2^{j}"] = var_j
        sum_detail_var += var_j

    smooth_var = float(np.var(v_j, ddof=1))
    decomp["smooth_variance"] = smooth_var
    decomp["explained_ratio"] = float((sum_detail_var + smooth_var) / (total_var + 1e-12))
    return decomp
