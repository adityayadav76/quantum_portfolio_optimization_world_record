"""
Portfolio Optimization — Powered by Automatski Quantum Computer
===============================================================
Run:  python app.py
Then open  http://localhost:7860  in your browser.

Dependencies:
    pip install gradio yfinance dimod dwave-neal numpy pandas matplotlib scipy lxml
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from data import (
    load_equity_universe_from_file,
    get_large_ticker_universe,
    fetch_prices,
    preprocess_prices,
    compute_returns,
    compute_statistics,
    FALLBACK_SP500,
)
from qubo import build_qubo
from solver import run_quantum_annealing, extract_selected_assets
from metrics import (
    WEIGHT_METHODS,
    compute_portfolio_weights,
    compute_equal_weights,
    portfolio_metrics,
    compute_historical_performance,
)

# ─── Load equity universe once at startup ─────────────────────────────────────
_EQUITY_UNIVERSE: list[str] = load_equity_universe_from_file()
if not _EQUITY_UNIVERSE:
    _EQUITY_UNIVERSE = list(FALLBACK_SP500)   # fallback while file is absent

EQUITY_COUNT = len(_EQUITY_UNIVERSE)
_EQUITY_FILE_STATUS = (
    f"{EQUITY_COUNT:,} equities loaded from equities.txt"
    if EQUITY_COUNT > len(FALLBACK_SP500)
    else f"{EQUITY_COUNT} equities loaded (run download_equities.py for the full universe)"
)


def _get_top_n_equities(n: int) -> list[str]:
    """Return the first n tickers from the pre-loaded universe."""
    return _EQUITY_UNIVERSE[:int(n)]


# ─── Core optimization pipeline ───────────────────────────────────────────────

def run_optimization(
    ticker_mode: str,
    custom_tickers: str,
    top_n: int,
    date_mode: str,
    start_date: str,
    end_date: str,
    period: str,
    lambda_risk: float,
    risk_free_rate: float,
    use_cardinality: bool,
    num_assets_to_select: int,
    weight_method_label: str,
    qa_host: str,
    qa_port: int,
    progress=gr.Progress(),
):
    """
    End-to-end quantum portfolio optimization.

    1. Resolve ticker universe (custom input or top-N US equities).
    2. Download historical prices and compute annualised μ and Σ.
    3. Formulate QUBO: minimise −μᵀx + λ·xᵀΣx with optional ∑xᵢ = K.
    4. Send QUBO to Automatski's Quantum Annealer for binary asset selection.
    5. Compute continuous allocation weights for selected assets.
    6. Return performance metrics and visualisations.
    """
    try:
        # ── 1. Tickers ─────────────────────────────────────────────────────────
        progress(0.04, desc="Resolving ticker universe…")

        if ticker_mode == "Custom Tickers":
            if not custom_tickers.strip():
                return _err("Please enter at least one ticker symbol.")
            tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
        else:
            tickers = _get_top_n_equities(int(top_n))

        if len(tickers) < 2:
            return _err("Please provide at least 2 ticker symbols.")

        # ── 2. Prices ──────────────────────────────────────────────────────────
        progress(0.12, desc=f"Fetching price history for {len(tickers)} tickers…")

        if date_mode == "Date Range":
            if not start_date.strip() or not end_date.strip():
                return _err("Please enter both a Start Date and an End Date.")
            prices_raw = fetch_prices(
                tickers, start=start_date.strip(), end=end_date.strip()
            )
        else:
            prices_raw = fetch_prices(tickers, period=period)

        # ── 3. Preprocess ──────────────────────────────────────────────────────
        progress(0.25, desc="Cleaning and validating data…")

        prices = preprocess_prices(prices_raw)
        valid_tickers = list(prices.columns)
        dropped = [t for t in tickers if t not in valid_tickers]
        n = len(valid_tickers)

        if n < 2:
            return _err(
                f"Only {n} valid ticker(s) after data cleaning. "
                "Use a longer date range or replace the problematic tickers."
            )

        # ── 4. Statistics ──────────────────────────────────────────────────────
        progress(0.38, desc="Computing annualised μ and Σ…")

        returns = compute_returns(prices)
        mu, sigma, tickers_ord = compute_statistics(returns)

        # ── 5. QUBO formulation ────────────────────────────────────────────────
        progress(0.50, desc="Formulating QUBO for Quantum Annealer…")

        K: int | None = int(num_assets_to_select) if use_cardinality else None
        if K is not None and (K <= 0 or K >= n):
            K = None

        bqm, Q_dict = build_qubo(mu, sigma, float(lambda_risk), num_assets=K)

        # ── 6. Quantum Annealing ───────────────────────────────────────────────
        n_qubits = bqm.num_variables
        n_clauses = len(Q_dict)
        progress(
            0.62,
            desc=f"Sending to Automatski Quantum Annealer "
                 f"({n_qubits} qubits · {n_clauses} clauses)…",
        )

        sample, energy, solver_info = run_quantum_annealing(
            Q_dict, n_qubits,
            host=qa_host.strip(),
            port=int(qa_port),
        )

        # ── 7. Portfolio construction ──────────────────────────────────────────
        progress(0.78, desc="Constructing portfolio…")

        sel_tickers, sel_indices = extract_selected_assets(sample, tickers_ord)

        if not sel_tickers:
            return _err(
                "The Quantum Annealer selected no assets. "
                "Try reducing the risk-aversion (λ) or disabling the cardinality constraint."
            )

        # Weight method
        method_key = WEIGHT_METHODS.get(weight_method_label, "equal")
        rf = float(risk_free_rate)

        w_opt = compute_portfolio_weights(sel_indices, mu, sigma, rf, method=method_key)
        w_eq  = compute_equal_weights(n)

        m_opt = portfolio_metrics(w_opt, mu, sigma, rf)
        m_eq  = portfolio_metrics(w_eq,  mu, sigma, rf)

        # ── 8. Results ─────────────────────────────────────────────────────────
        progress(0.90, desc="Generating charts…")

        # Allocation table (sorted by weight descending)
        sorted_pairs = sorted(
            zip(sel_tickers, [w_opt[i] for i in sel_indices]),
            key=lambda x: x[1], reverse=True,
        )
        alloc_df = pd.DataFrame({
            "Ticker":     [p[0] for p in sorted_pairs],
            "Weight (%)": [f"{p[1]*100:.2f}" for p in sorted_pairs],
        })

        # Metrics comparison table
        metrics_df = pd.DataFrame({
            "Metric": ["Expected Return", "Std Dev (Risk)", "Sharpe Ratio"],
            "Quantum Portfolio": [
                f"{m_opt['expected_return']*100:.2f}%",
                f"{m_opt['std_dev']*100:.2f}%",
                f"{m_opt['sharpe_ratio']:.4f}",
            ],
            "Equal-Weight Benchmark": [
                f"{m_eq['expected_return']*100:.2f}%",
                f"{m_eq['std_dev']*100:.2f}%",
                f"{m_eq['sharpe_ratio']:.4f}",
            ],
        })

        fig_alloc = _make_alloc_chart(
            [p[0] for p in sorted_pairs],
            [p[1] for p in sorted_pairs],
            weight_method_label,
        )
        fig_hist = _make_history_chart(prices, w_opt, tickers_ord)

        # Drop warning
        drop_note = (
            f"\n\n> ⚠️ **Dropped** (no/insufficient data): {', '.join(dropped)}"
            if dropped else ""
        )

        constraint_label = f"K = {K}" if K is not None else "unconstrained"
        opt_wins = m_opt["sharpe_ratio"] >= m_eq["sharpe_ratio"]
        opt_tag = " 🏆" if opt_wins else ""
        eq_tag  = " 🏆" if not opt_wins else ""

        summary = (
            f"## Quantum Portfolio Optimization Complete\n\n"
            f"**Solver:** {solver_info}  \n"
            f"**Universe:** {n} assets  |  "
            f"**Selected:** {len(sel_tickers)}  |  "
            f"**Cardinality:** {constraint_label}  |  "
            f"**Weights:** {weight_method_label}\n\n"
            f"| Metric | Quantum Portfolio{opt_tag} | Equal-Weight{eq_tag} |\n"
            f"|---|---|---|\n"
            f"| Expected Return | **{m_opt['expected_return']*100:.2f}%** "
            f"| {m_eq['expected_return']*100:.2f}% |\n"
            f"| Std Dev (Risk)  | **{m_opt['std_dev']*100:.2f}%** "
            f"| {m_eq['std_dev']*100:.2f}% |\n"
            f"| Sharpe Ratio    | **{m_opt['sharpe_ratio']:.4f}** "
            f"| {m_eq['sharpe_ratio']:.4f} |\n\n"
            f"*λ = {lambda_risk:.2f}  ·  r_f = {rf*100:.1f}%  ·  "
            f"QUBO energy = {energy:.4f}  ·  "
            f"{n_qubits} qubits · {n_clauses} clauses*"
            f"{drop_note}"
        )

        progress(1.0, desc="Done!")
        return summary, alloc_df, metrics_df, fig_alloc, fig_hist, solver_info

    except Exception as exc:
        return _err(str(exc))


# ─── Chart helpers ─────────────────────────────────────────────────────────────

def _make_alloc_chart(
    tickers: list[str],
    weights: list[float],
    method_label: str,
) -> plt.Figure:
    if not tickers:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No assets selected", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(max(7, len(tickers) * 0.65 + 2), 4.5))
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(tickers)))
    bars = ax.bar(
        tickers, [w * 100 for w in weights],
        color=palette, edgecolor="white", linewidth=0.7, zorder=3,
    )
    ax.set_ylabel("Weight (%)", fontsize=11)
    ax.set_title(
        f"Quantum Portfolio Allocation  ·  {method_label}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_ylim(0, max(w * 100 for w in weights) * 1.35)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, w in zip(bars, weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{w*100:.1f}%",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    return fig


def _make_history_chart(
    prices: pd.DataFrame,
    weights: np.ndarray,
    tickers: list[str],
) -> plt.Figure:
    port, bench = compute_historical_performance(prices, weights, tickers)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(port.index,  port.values,  label="Quantum Portfolio",
            color="#7b2d8b", linewidth=2.2, zorder=3)
    ax.plot(bench.index, bench.values, label="Equal-Weight Benchmark",
            color="#e8732a", linewidth=1.5, linestyle="--", zorder=2)
    ax.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle=":")

    ax.fill_between(port.index, port.values, bench.values,
                    where=(port.values >= bench.values),
                    alpha=0.12, color="#7b2d8b")
    ax.fill_between(port.index, port.values, bench.values,
                    where=(port.values < bench.values),
                    alpha=0.12, color="#e8732a")

    ax.set_ylabel("Cumulative Return (base = 1)", fontsize=11)
    ax.set_title(
        "Historical Back-Test  ·  Normalized to 1 at Start",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def _err(msg: str):
    """Return a consistent 6-tuple for all outputs in the error case."""
    empty = pd.DataFrame()
    return f"## ❌ Error\n\n{msg}", empty, empty, None, None, f"❌ {msg}"


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    weight_choices = list(WEIGHT_METHODS.keys())

    with gr.Blocks(
        title="Portfolio Optimization — Automatski's Quantum Annealer",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 100% !important; padding: 0 1.5rem; }
            .main { max-width: 100% !important; }
            footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# Portfolio Optimization\n"
            "### Powered by Automatski's Quantum Annealer\n"
            "Select a stock universe, configure risk parameters, and solve "
            "portfolio selection via **QUBO on Automatski's Quantum Annealer**."
        )

        with gr.Row(equal_height=False):

            # ── LEFT: inputs ─────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=360):

                # ── Quantum Computer Connection ──────────────────────────────
                with gr.Group():
                    gr.Markdown("### ⚛️ Quantum Computer")
                    with gr.Row():
                        qa_host = gr.Textbox(
                            value="localhost",
                            label="Server Host",
                            placeholder="localhost or IP address",
                        )
                        qa_port = gr.Number(
                            value=8000,
                            label="Port",
                            precision=0,
                        )

                # ── Asset Universe ───────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### 1 · Asset Universe")

                    ticker_mode = gr.Radio(
                        choices=["Custom Tickers", "Top N US Equities"],
                        value="Custom Tickers",
                        label="Ticker Source",
                    )
                    custom_tickers = gr.Textbox(
                        value="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,V,BRK-B",
                        label="Custom Tickers (comma-separated)",
                        placeholder="AAPL,MSFT,GOOGL,TSLA,…",
                        lines=2,
                        visible=True,
                    )
                    top_n = gr.Slider(
                        minimum=3,
                        maximum=max(EQUITY_COUNT, 3),
                        value=min(50, EQUITY_COUNT),
                        step=1,
                        label=f"Number of Equities  (max {EQUITY_COUNT:,} available)",
                        visible=False,
                    )
                    equity_status_md = gr.Markdown(
                        f"<small>📁 {_EQUITY_FILE_STATUS}</small>",
                        visible=False,
                    )

                # ── Historical Data ──────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### 2 · Historical Data")

                    date_mode = gr.Radio(
                        choices=["Predefined Period", "Date Range"],
                        value="Predefined Period",
                        label="Date Selection",
                    )
                    period = gr.Dropdown(
                        choices=["1y", "2y", "3y", "4y", "5y"],
                        value="2y",
                        label="Period",
                        visible=True,
                    )
                    with gr.Row(visible=False) as date_row:
                        start_date = gr.Textbox("2022-01-01", label="Start Date (YYYY-MM-DD)")
                        end_date   = gr.Textbox("2024-12-31", label="End Date (YYYY-MM-DD)")

                # ── Optimisation Parameters ──────────────────────────────────
                with gr.Group():
                    gr.Markdown("### 3 · Optimisation Parameters")

                    lambda_risk = gr.Slider(
                        0.0, 5.0, value=1.0, step=0.05,
                        label="Risk Aversion λ  (0 = max return  ·  5 = min risk)",
                    )
                    risk_free_rate = gr.Slider(
                        0.0, 0.10, value=0.04, step=0.005,
                        label="Annual Risk-Free Rate",
                    )

                    gr.Markdown("**Asset Selection (Quantum Annealer)**")
                    with gr.Row():
                        use_cardinality = gr.Checkbox(
                            label="Enforce exact number of assets",
                            value=False,
                        )
                        num_assets_to_select = gr.Slider(
                            1, 100, value=10, step=1,
                            label="K — assets to select",
                        )

                    gr.Markdown("**Weight Allocation (Post-Quantum Optimisation)**")
                    weight_method = gr.Dropdown(
                        choices=weight_choices,
                        value="Equal Weight",
                        label="Weight Allocation Method",
                    )
                    gr.Markdown(
                        "<small>Equal Weight · Minimum Variance · Maximum Sharpe · "
                        "Inverse Volatility · Risk Parity</small>",
                        visible=True,
                    )

                run_btn = gr.Button(
                    "⚛️  Run Quantum Optimization", variant="primary", size="lg"
                )
                status_box = gr.Textbox(
                    label="Solver Status", interactive=False, lines=2, max_lines=3
                )

            # ── RIGHT: outputs ───────────────────────────────────────────────
            with gr.Column(scale=2):

                summary_md = gr.Markdown(
                    "*Configure the parameters and click **Run Quantum Optimization**.*"
                )

                with gr.Tabs():
                    with gr.Tab("📊 Allocation"):
                        alloc_table = gr.DataFrame(
                            label="Selected Assets & Weights",
                            interactive=False, wrap=True,
                        )
                        fig_alloc = gr.Plot(label="Allocation Chart")

                    with gr.Tab("📈 Performance Metrics"):
                        metrics_table = gr.DataFrame(
                            label="Quantum Portfolio vs. Equal-Weight Benchmark",
                            interactive=False, wrap=True,
                        )

                    with gr.Tab("📉 Historical Back-Test"):
                        fig_hist = gr.Plot(
                            label="Cumulative Return (Normalized to 1 at Start)"
                        )

        # ── Visibility toggles ────────────────────────────────────────────────
        ticker_mode.change(
            fn=lambda m: (
                gr.update(visible=(m == "Custom Tickers")),
                gr.update(visible=(m == "Top N US Equities")),
                gr.update(visible=(m == "Top N US Equities")),
            ),
            inputs=ticker_mode,
            outputs=[custom_tickers, top_n, equity_status_md],
        )
        date_mode.change(
            fn=lambda m: (
                gr.update(visible=(m == "Predefined Period")),
                gr.update(visible=(m == "Date Range")),
            ),
            inputs=date_mode,
            outputs=[period, date_row],
        )

        # ── Main action ───────────────────────────────────────────────────────
        run_btn.click(
            fn=run_optimization,
            inputs=[
                ticker_mode, custom_tickers, top_n,
                date_mode, start_date, end_date, period,
                lambda_risk, risk_free_rate,
                use_cardinality, num_assets_to_select,
                weight_method,
                qa_host, qa_port,
            ],
            outputs=[
                summary_md, alloc_table, metrics_table,
                fig_alloc, fig_hist, status_box,
            ],
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
