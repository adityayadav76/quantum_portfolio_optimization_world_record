"""
Portfolio performance metrics and weight optimisation.

After the Quantum Annealer selects which assets to include (binary decision),
this module solves for the optimal continuous allocation weights using
standard mean-variance portfolio theory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ─── Weight optimisation ──────────────────────────────────────────────────────

WEIGHT_METHODS = {
    "Equal Weight":       "equal",
    "Minimum Variance":   "min_variance",
    "Maximum Sharpe":     "max_sharpe",
    "Inverse Volatility": "inverse_vol",
    "Risk Parity":        "risk_parity",
}


def compute_portfolio_weights(
    selected_indices: list[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free_rate: float = 0.0,
    method: str = "equal",
) -> np.ndarray:
    """
    Compute allocation weights for the quantum-selected assets.

    The quantum annealer chooses WHICH assets to hold (binary selection).
    This function then determines HOW MUCH to hold in each (continuous weights).

    Methods
    -------
    equal          Simple 1/K equal weight across selected assets.
    min_variance   Global Minimum Variance portfolio (long-only).
                   Minimises wᵀΣw subject to ∑w=1, w≥0.
    max_sharpe     Maximum Sharpe Ratio / Tangency portfolio (long-only).
                   Maximises (μᵀw − r_f) / √(wᵀΣw) subject to ∑w=1, w≥0.
    inverse_vol    Weight proportional to 1/σᵢ (fast, no optimisation).
    risk_parity    Equal Risk Contribution: each asset contributes equally
                   to total portfolio volatility (long-only).

    Args:
        selected_indices: indices of assets chosen by the quantum annealer
        mu:               (n,) annualised expected returns (full universe)
        sigma:            (n,n) annualised covariance matrix (full universe)
        risk_free_rate:   annualised risk-free rate (used for max_sharpe)
        method:           one of the keys in WEIGHT_METHODS

    Returns:
        (n,) weight array; non-selected assets have weight 0; sums to 1.
    """
    n = len(mu)
    weights = np.zeros(n)

    if not selected_indices:
        return weights

    K = len(selected_indices)
    idx = list(selected_indices)
    mu_s = mu[idx]
    sigma_s = sigma[np.ix_(idx, idx)]

    if K == 1 or method == "equal":
        w_s = np.ones(K) / K
    elif method == "min_variance":
        w_s = _min_variance(sigma_s)
    elif method == "max_sharpe":
        w_s = _max_sharpe(mu_s, sigma_s, risk_free_rate)
    elif method == "inverse_vol":
        vols = np.sqrt(np.maximum(np.diag(sigma_s), 1e-10))
        inv_v = 1.0 / vols
        w_s = inv_v / inv_v.sum()
    elif method == "risk_parity":
        w_s = _risk_parity(sigma_s)
    else:
        w_s = np.ones(K) / K

    for local_i, global_i in enumerate(idx):
        weights[global_i] = w_s[local_i]

    return weights


# ─── Convenience wrappers ─────────────────────────────────────────────────────

def compute_equal_weights(n_assets: int) -> np.ndarray:
    """1/n allocation across all n_assets (equal-weight benchmark)."""
    return np.ones(n_assets) / n_assets


# ─── Portfolio metrics ────────────────────────────────────────────────────────

def portfolio_metrics(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute annualised portfolio performance metrics.

    Returns:
        expected_return  weighted average annualised return
        variance         portfolio variance (wᵀΣw)
        std_dev          annualised volatility
        sharpe_ratio     (E[R] − r_f) / σ
    """
    expected_return = float(weights @ mu)
    variance = float(weights @ sigma @ weights)
    std_dev = float(np.sqrt(max(variance, 0.0)))
    sharpe = (
        (expected_return - risk_free_rate) / std_dev
        if std_dev > 1e-10 else 0.0
    )
    return {
        "expected_return": expected_return,
        "variance": variance,
        "std_dev": std_dev,
        "sharpe_ratio": sharpe,
    }


# ─── Historical back-test ─────────────────────────────────────────────────────

def compute_historical_performance(
    prices: pd.DataFrame,
    weights: np.ndarray,
    tickers: list[str],
) -> tuple[pd.Series, pd.Series]:
    """
    Cumulative portfolio value normalised to 1 at inception.

    Returns both the optimised portfolio and an equal-weight benchmark
    for side-by-side comparison.
    """
    norm = prices / prices.iloc[0]

    w_opt = pd.Series(weights, index=tickers)
    portfolio_curve = (norm * w_opt).sum(axis=1)

    w_eq = pd.Series(np.ones(len(tickers)) / len(tickers), index=tickers)
    benchmark_curve = (norm * w_eq).sum(axis=1)

    return portfolio_curve, benchmark_curve


# ─── Weight optimisation internals ───────────────────────────────────────────

_SLSQP_OPTS = {"ftol": 1e-12, "maxiter": 2000}
_EQ_CONSTRAINT = {"type": "eq", "fun": lambda w: w.sum() - 1.0}


def _min_variance(sigma: np.ndarray) -> np.ndarray:
    """Global Minimum Variance weights (long-only)."""
    n = len(sigma)
    result = minimize(
        fun=lambda w: float(w @ sigma @ w),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=_EQ_CONSTRAINT,
        options=_SLSQP_OPTS,
    )
    return result.x if result.success else np.ones(n) / n


def _max_sharpe(mu: np.ndarray, sigma: np.ndarray, rf: float) -> np.ndarray:
    """Maximum Sharpe Ratio (Tangency Portfolio) weights (long-only)."""
    n = len(sigma)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu) - rf
        vol = float(np.sqrt(max(w @ sigma @ w, 1e-12)))
        return -ret / vol

    result = minimize(
        fun=neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=_EQ_CONSTRAINT,
        options=_SLSQP_OPTS,
    )
    return result.x if result.success else np.ones(n) / n


def _risk_parity(sigma: np.ndarray) -> np.ndarray:
    """
    Equal Risk Contribution weights.

    Each asset contributes σ_portfolio / K to total portfolio volatility.
    """
    n = len(sigma)

    def rp_loss(w: np.ndarray) -> float:
        port_vol = float(np.sqrt(max(w @ sigma @ w, 1e-12)))
        rc = w * (sigma @ w) / port_vol          # risk contributions
        target = port_vol / n                     # equal target
        return float(np.sum((rc - target) ** 2))

    result = minimize(
        fun=rp_loss,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(1e-6, 1.0)] * n,                # small lower bound avoids zeros
        constraints=_EQ_CONSTRAINT,
        options={**_SLSQP_OPTS, "ftol": 1e-14},
    )
    if result.success:
        return result.x

    # Fallback: inverse-volatility is a close approximation of risk parity
    vols = np.sqrt(np.maximum(np.diag(sigma), 1e-10))
    inv_v = 1.0 / vols
    return inv_v / inv_v.sum()
