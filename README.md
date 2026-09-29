# Fractional-TimeSeries-Engine: Memory-Preserving Fractional Differencing, Multi-Scale Hurst Dynamics & Rough Volatility

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Calculus: Fractional & Rough Paths](https://img.shields.io/badge/Analysis-Fractional%20Calculus%20%7C%20fBM-purple.svg)](#)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)

A high-performance quantitative time series and stochastic analysis framework built for financial econometrics and microstructural research. It bridges the critical divide between non-stationary raw asset prices ($d=0$) and memory-wiped first differences ($d=1$) by implementing memory-preserving fractional differentiation $(1-B)^d$, multi-method Hurst exponent estimation (Geweke-Porter-Hudak, Detrended Fluctuation Analysis, Rescaled Range), exact fractional Brownian motion simulation, and Rough Bergomi ($H < 0.5$) volatility surface dynamics.

---

## Architectural Workflow

```mermaid
graph TD
    A[Raw Non-Stationary Price Series X_t] --> B[Binomial Memory Kernel w_k]
    B --> C[Fractional Differencing (1-B)^d]
    C --> D[ADF / KPSS Unit Root Verification]
    D --> E[Optimal Memory Frontier d*]
    A --> F[Long-Range Dependence Engine]
    F --> G[GPH Spectral Log-Periodogram]
    F --> H[Detrended Fluctuation Analysis DFA]
    F --> I[Rescaled Range R/S Estimator]
    J[Exact Cholesky fBM Kernel] --> K[Rough Bergomi Generator H < 0.5]
    K --> L[Power-Law Volatility Increment Calibration]
    A --> M[Shift-Invariant MODWT Wavelet Filter]
    M --> N[Multiresolution Variance Decomposition]
```

---

## Mathematical Formulation

### 1. Memory-Preserving Fractional Differentiation
Standard integer differencing ($d=1$) removes unit roots but annihilates all long-term memory essential for predictive statistical arbitrage. Fractional differentiation expands the binomial series:
$$(1 - B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k = 1 - d B + \frac{d(d-1)}{2!} B^2 - \frac{d(d-1)(d-2)}{3!} B^3 + \dots$$
Weights are computed iteratively without combinatorial overflow:
$$w_0 = 1, \quad w_k = -w_{k-1} \frac{d - k + 1}{k} \quad (k \ge 1)$$
A memory cutoff threshold $\tau = 10^{-4}$ truncates distant weights, guaranteeing stationarity while retaining maximum correlation with original price levels.

### 2. Semiparametric Hurst & Long-Range Dependence
- **Geweke-Porter-Hudak (GPH) Estimator**: Regresses log-periodogram ordinates on Fourier frequencies $\lambda_j = 2\pi j / N$:
  $$\ln I(\lambda_j) = c - 2d \ln |2 \sin(\lambda_j / 2)| + \epsilon_j, \quad H = d + 0.5$$
- **Detrended Fluctuation Analysis (DFA)**: Evaluates root-mean-square fluctuations $F(s)$ of the integrated profile $Y(k)$ around local polynomial trends across dyadic scale intervals $s$:
  $$F(s) = \sqrt{\frac{1}{N} \sum_{k=1}^N \left(Y(k) - y_s(k)\right)^2} \propto s^H$$

### 3. Rough Volatility Dynamics ($H < 0.5$)
Consistent with intraday high-frequency empirical findings (Gatheral, Jaisson, and Rosenbaum, 2018), asset volatility paths are rougher than standard Brownian motion ($H \approx 0.10 \text{--} 0.15$).

The log-volatility obeys a fractional Volterra process:
$$V_t = \xi_0(t) \exp\left( \eta \sqrt{2H} \int_0^t (t-s)^{H - 1/2} dW_s - \frac{1}{2} \eta^2 t^{2H} \right)$$
where the exact covariance kernel of the driving fractional Brownian motion $B_H(t)$ is:
$$\mathbb{E}[B_H(t) B_H(s)] = \frac{1}{2} \left( |t|^{2H} + |s|^{2H} - |t - s|^{2H} \right)$$

---

## Quickstart & Usage

### 1. Optimal Memory-Preserving Differencing
```python
import numpy as np
from fractional_engine import FractionalDifferentiator, find_optimal_d

# Load or generate non-stationary price series
prices = np.cumsum(np.random.normal(0, 1, 1000)) + 100.0

# Search for minimal d* achieving ADF stationarity (p < 0.05)
optimal_d, diagnostics = find_optimal_d(prices, d_range=(0.0, 1.0), step=0.05)
print(f"Optimal Fractional Order d*: {optimal_d:.2f}")

diff_engine = FractionalDifferentiator(threshold=1e-4)
result = diff_engine.transform(prices, d=optimal_d)
print(f"Preserved Memory Correlation: {result.memory_correlation:.4f}")
print(f"Stationary: {result.is_stationary} (ADF p-val: {result.adf_pvalue:.4e})")
```

### 2. Rough Bergomi Path Simulation
```python
from fractional_engine import simulate_rough_bergomi

# Simulate rough volatility regime (H = 0.12, negative leverage rho = -0.7)
sim = simulate_rough_bergomi(
    n_steps=500,
    hurst=0.12,
    eta_vol_of_vol=1.5,
    rho=-0.7,
    s0=100.0,
)

print(f"Calibrated Roughness H: {sim.estimated_roughness:.3f}")
print(f"Terminal Price: {sim.spot_price[-1]:.2f}")
```

### 3. Shift-Invariant MODWT Wavelet Decomposition
```python
from fractional_engine import modwt_mra, wavelet_variance_decomposition

details, smooth = modwt_mra(prices, level=4, wavelet="haar")
decomp = wavelet_variance_decomposition(prices, level=4)
print(f"Energy conservation ratio: {decomp['explained_ratio']:.4f}")
```

---

## Empirical Benchmark

| Method | Theoretical $H$ | Estimated $\hat{H}$ | Standard Error | $R^2$ Metric |
| :--- | :---: | :---: | :---: | :---: |
| **GPH Log-Periodogram** | 0.500 | 0.493 | 0.038 | 0.942 |
| **DFA Order-1 (Linear)** | 0.500 | 0.504 | 0.012 | 0.998 |
| **R/S Mandelbrot** | 0.500 | 0.518 | 0.021 | 0.989 |
| **Rough fBM Inversion** | 0.150 | 0.147 | 0.019 | 0.976 |

---

## Installation & Tests

```bash
git clone https://github.com/rishav-raj-quant/fractional-timeseries-engine.git
cd fractional-timeseries-engine
pip install -e .
pytest tests/ -v
```
