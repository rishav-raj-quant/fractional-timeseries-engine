"""
Rough Volatility & Fractional Brownian Motion Engine.

Implements:
1. Exact Fractional Brownian Motion (fBM) Simulation via Cholesky Covariance Kernel
2. Rough Bergomi / Volterra Fractional Variance Dynamics (H < 0.5)
3. Realized Volatility Roughness Calibration via Moment Scaling:
       E[|ln sigma_{t+Delta} - ln sigma_t|^q] = c_q * Delta^{q * H}
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


@dataclass(frozen=True)
class RoughVolResult:
    """Simulation path and roughness parameters."""
    hurst: float
    time_grid: np.ndarray
    fbm_path: np.ndarray
    spot_price: np.ndarray
    variance_path: np.ndarray
    estimated_roughness: float


def compute_fbm_covariance_matrix(n: int, h: float, dt: float = 1.0) -> np.ndarray:
    """
    Computes exact fractional Brownian motion covariance kernel:
        Gamma(i, j) = 0.5 * dt^{2H} * (|i|^{2H} + |j|^{2H} - |i - j|^{2H})
    """
    t = np.arange(1, n + 1, dtype=np.float64)
    # Outer mesh
    t_i, t_j = np.meshgrid(t, t, indexing="ij")
    cov = 0.5 * (dt ** (2.0 * h)) * (
        (t_i ** (2.0 * h)) + (t_j ** (2.0 * h)) - (np.abs(t_i - t_j) ** (2.0 * h))
    )
    # Add numerical jitter to ensure positive definiteness
    cov += np.eye(n) * 1e-11
    return cov


def simulate_fractional_brownian_motion(
    n_steps: int,
    hurst: float,
    t_final: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a 1D Fractional Brownian Motion path B_H(t) on [0, t_final]
    using exact Cholesky factorization of the fBM covariance matrix.
    """
    if not (0.01 <= hurst <= 0.99):
        raise ValueError(f"Hurst parameter must be in (0, 1), got {hurst}")

    rng = np.random.default_rng(seed)
    dt = t_final / float(n_steps)
    cov = compute_fbm_covariance_matrix(n_steps, hurst, dt=dt)

    # Cholesky decomposition L * L^T = Cov
    l_factor = np.linalg.cholesky(cov)
    z = rng.standard_normal(n_steps)
    fbm_increments = l_factor @ z

    # Path begins at B_H(0) = 0
    t_grid = np.linspace(0, t_final, n_steps + 1)
    fbm_path = np.concatenate([[0.0], fbm_increments])
    return t_grid, fbm_path


def simulate_rough_bergomi(
    n_steps: int = 500,
    hurst: float = 0.12,
    eta_vol_of_vol: float = 1.5,
    xi_0: float = 0.04,
    rho: float = -0.7,
    s0: float = 100.0,
    t_final: float = 1.0,
    seed: Optional[int] = None,
) -> RoughVolResult:
    """
    Simulates bivariate Rough Bergomi log-volatility model:
        S_t = S_0 * exp( int_0^t sqrt(V_s) dW_s - 0.5 int_0^t V_s ds )
        V_t = xi_0 * exp( eta * sqrt(2H) * W_t^H - 0.5 * eta^2 * t^{2H} )
    where Corr(W_s, W_s^H) = rho (leverage effect).
    """
    rng = np.random.default_rng(seed)
    dt = t_final / float(n_steps)
    t_grid, fbm_path = simulate_fractional_brownian_motion(n_steps, hurst, t_final=t_final, seed=seed)

    # Compute fractional driver increments
    fbm_increments = np.diff(fbm_path)

    # Generate correlated Brownian motion for spot price
    z_independent = rng.standard_normal(n_steps)
    dw_spot = rho * (fbm_increments / np.std(fbm_increments)) * np.sqrt(dt) + \
              np.sqrt(max(0.0, 1.0 - rho ** 2)) * z_independent * np.sqrt(dt)

    # Volatility process V_t
    t_eval = t_grid[1:]
    variance_increments = xi_0 * np.exp(
        eta_vol_of_vol * np.sqrt(2.0 * hurst) * fbm_increments - 0.5 * (eta_vol_of_vol ** 2) * (t_eval ** (2.0 * hurst))
    )
    variance_path = np.concatenate([[xi_0], np.maximum(1e-6, variance_increments)])

    # Spot price Euler-Maruyama simulation
    log_s = np.zeros(n_steps + 1)
    log_s[0] = np.log(s0)

    for i in range(n_steps):
        v_current = max(1e-6, variance_path[i])
        vol_curr = np.sqrt(v_current)
        log_s[i + 1] = log_s[i] - 0.5 * v_current * dt + vol_curr * dw_spot[i]

    spot_path = np.exp(log_s)

    # Calibrate estimated roughness
    h_est = calibrate_hurst_from_realized_variance(variance_path)

    return RoughVolResult(
        hurst=hurst,
        time_grid=t_grid,
        fbm_path=fbm_path,
        spot_price=spot_path,
        variance_path=variance_path,
        estimated_roughness=h_est,
    )


def calibrate_hurst_from_realized_variance(
    volatility_series: np.ndarray,
    lags: Optional[List[int]] = None,
    q_order: float = 1.0,
) -> float:
    """
    Calibrates Hurst parameter H from the power-law scaling of absolute log-volatility increments:
        m(q, Delta) = E[|ln(sigma_{t+Delta}) - ln(sigma_t)|^q] ~ Delta^{q * H}
    """
    vol = np.asarray(volatility_series, dtype=np.float64)
    vol = np.maximum(vol, 1e-8)
    log_vol = np.log(vol)

    if lags is None:
        lags = [1, 2, 3, 4, 5, 7, 10, 15, 20]

    valid_lags = [l for l in lags if l < len(log_vol) // 4]
    if len(valid_lags) < 3:
        return 0.5

    moments = []
    actual_lags = []

    for lag in valid_lags:
        diffs = np.abs(log_vol[lag:] - log_vol[:-lag])
        if len(diffs) > 0:
            m_q = float(np.mean(diffs ** q_order))
            if m_q > 1e-12:
                moments.append(m_q)
                actual_lags.append(lag)

    if len(moments) < 3:
        return 0.5

    log_lags = np.log(actual_lags)
    log_moments = np.log(moments)

    slope, _ = np.polyfit(log_lags, log_moments, deg=1)
    estimated_h = float(slope / q_order)
    return float(np.clip(estimated_h, 0.02, 0.98))
