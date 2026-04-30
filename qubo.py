"""QUBO formulation for binary portfolio selection via dimod."""

from __future__ import annotations

import numpy as np
import dimod


def build_qubo(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_risk: float,
    num_assets: int | None = None,
    penalty: float | None = None,
) -> tuple[dimod.BinaryQuadraticModel, dict]:
    """
    Construct a QUBO for binary portfolio selection.

    Decision variables: x_i ∈ {0, 1}  (1 = include asset i)

    Objective (minimisation form):
        f(x) = −μᵀx + λ · xᵀΣx

    QUBO coefficients (upper-triangular dict, i ≤ j):
        Q[i,i] = −μᵢ + λ·Σᵢᵢ            (linear/diagonal term)
        Q[i,j] = λ·(Σᵢⱼ + Σⱼᵢ) = 2λΣᵢⱼ  (quadratic term, i < j)

    Optional cardinality constraint  ∑xᵢ = K:
        Adds penalty P·(∑xᵢ − K)²  expanded as (using xᵢ² = xᵢ):
            P·[(1 − 2K)·∑xᵢ  +  2·∑_{i<j} xᵢxⱼ]   (+constant K² ignored)

    Args:
        mu:          (n,) annualised mean returns
        sigma:       (n,n) annualised covariance matrix (must be symmetric)
        lambda_risk: risk-aversion coefficient λ ≥ 0
        num_assets:  K — enforce exactly K selected assets (None = unconstrained)
        penalty:     cardinality penalty strength (auto-estimated when None)

    Returns:
        bqm:    dimod.BinaryQuadraticModel ready for sampling
        Q_dict: raw QUBO coefficients {(i, j): value}
    """
    n = len(mu)
    Q: dict[tuple[int, int], float] = {}

    # ── Objective: linear terms ────────────────────────────────────────────────
    for i in range(n):
        Q[(i, i)] = float(-mu[i] + lambda_risk * sigma[i, i])

    # ── Objective: quadratic terms ─────────────────────────────────────────────
    # xᵀΣx = ∑ᵢ Σᵢᵢ xᵢ  +  2·∑_{i<j} Σᵢⱼ xᵢxⱼ  (sigma symmetric)
    for i in range(n):
        for j in range(i + 1, n):
            Q[(i, j)] = float(lambda_risk * (sigma[i, j] + sigma[j, i]))

    # ── Cardinality constraint P·(∑xᵢ − K)² ───────────────────────────────────
    if num_assets is not None:
        K = int(num_assets)
        if penalty is None:
            penalty = _auto_penalty(mu, sigma, lambda_risk)

        for i in range(n):
            Q[(i, i)] = Q.get((i, i), 0.0) + penalty * (1 - 2 * K)

        for i in range(n):
            for j in range(i + 1, n):
                Q[(i, j)] = Q.get((i, j), 0.0) + 2.0 * penalty

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    return bqm, Q


def _auto_penalty(
    mu: np.ndarray,
    sigma: np.ndarray,
    lambda_risk: float,
) -> float:
    """
    Estimate a penalty coefficient that makes the cardinality constraint binding.

    The penalty must dominate the objective scale so violating the cardinality
    by even one asset is costlier than any gain in the objective.  We use
    10× the max absolute value of either term as a conservative heuristic.
    """
    obj_scale = max(
        float(np.abs(mu).max()),
        float(np.abs(lambda_risk * sigma).max()),
        1e-4,
    )
    return 10.0 * obj_scale
