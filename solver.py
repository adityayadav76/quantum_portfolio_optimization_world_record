"""
Quantum annealing solver — Automatski Quantum Computer.

Sends the QUBO directly to Automatski's Quantum Annealer via HTTP.
Falls back to a local annealer if the quantum server is unreachable.
"""

from __future__ import annotations

import dimod
import neal

from AutomatskiInitium import AutomatskiInitiumSASolver

# ── Quantum annealer defaults (no user-facing controls needed) ────────────────
_QA_MAX_ITER    = 1000
_QA_TEMP        = 10.0
_QA_COOLING_RATE = 0.003
_QA_NUM_READS   = 10
_QA_TIMEOUT     = 3600   # 1 hour hard limit


def run_quantum_annealing(
    Q_dict: dict,
    n_variables: int,
    host: str = "localhost",
    port: int = 8080,
    api_key: str = "open",
) -> tuple[dict, float, str]:
    """
    Solve a QUBO using Automatski's Quantum Annealer.

    Args:
        Q_dict:      {(i, j): coefficient} QUBO dictionary (upper-triangular)
        n_variables: number of binary decision variables
        host:        Quantum computer server hostname
        port:        Quantum computer server port
        api_key:     Automatski API key (default 'open')

    Returns:
        sample:      {variable_index: 0|1}  best solution found
        energy:      objective value of the best solution
        solver_info: string describing which solver was used
    """
    # ── Primary: Automatski Quantum Annealer ──────────────────────────────────
    try:
        solver = AutomatskiInitiumSASolver(
            host=host,
            port=port,
            max_iter=_QA_MAX_ITER,
            temp=_QA_TEMP,
            cooling_rate=_QA_COOLING_RATE,
            num_reads=_QA_NUM_READS,
            timeout=_QA_TIMEOUT,
            apiKey=api_key,
        )
        sample, energy = solver.solve(Q_dict, silent=False)
        return sample, float(energy), "Automatski Quantum Annealer"

    except Exception as exc:
        # ── Fallback: local annealer ──────────────────────────────────────────
        sample, energy = _local_annealer_fallback(Q_dict, n_variables)
        info = f"Local fallback solver (Automatski unavailable: {exc})"
        return sample, energy, info


def extract_selected_assets(
    sample: dict,
    tickers: list[str],
) -> tuple[list[str], list[int]]:
    """
    Identify assets selected by the quantum annealer (x_i = 1).

    Args:
        sample:  {integer_index: 0|1} variable assignment from the solver
        tickers: asset names in index order 0 … n−1

    Returns:
        selected_tickers: names of assets with x_i = 1
        selected_indices: corresponding integer indices
    """
    selected_tickers: list[str] = []
    selected_indices: list[int] = []
    for i, ticker in enumerate(tickers):
        if sample.get(i, 0) == 1:
            selected_tickers.append(ticker)
            selected_indices.append(i)
    return selected_tickers, selected_indices


# ─── Internal fallback ────────────────────────────────────────────────────────

def _local_annealer_fallback(Q_dict: dict, n_variables: int) -> tuple[dict, float]:
    """Local neal-based annealer used when the quantum server is unavailable."""
    reads, sweeps = _adaptive_params(n_variables)
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(
        bqm,
        num_reads=reads,
        num_sweeps=sweeps,
        beta_range=(0.1, 10.0),
        beta_schedule_type="geometric",
    )
    best = sampleset.first
    return best.sample, best.energy


def _adaptive_params(n: int) -> tuple[int, int]:
    """Scale local annealer iterations with problem size."""
    if n <= 50:   return 1000, 1000
    if n <= 200:  return 500,  1000
    if n <= 500:  return 200,  1000
    return 100, 2000
