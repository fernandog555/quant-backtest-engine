"""
Plotting utilities for backtest results. Kept separate from the engine
itself so the core library has no hard matplotlib dependency — only
scripts/notebooks that actually want charts need to import this.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.backtest.engine import BacktestResult


def plot_equity_curve(
    result: BacktestResult,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    save_path: str | None = None,
):
    """
    Plots the strategy's equity curve, optionally overlaid with a benchmark
    (e.g. buy-and-hold on the same capital base) for visual comparison.
    """
    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_equity.plot(result.equity_curve.index, result.equity_curve.values, label="Strategy", linewidth=1.5)
    if benchmark is not None:
        ax_equity.plot(benchmark.index, benchmark.values, label="Benchmark", linewidth=1.2, alpha=0.7, linestyle="--")

    ax_equity.set_title(title)
    ax_equity.set_ylabel("Equity ($)")
    ax_equity.legend(loc="upper left")
    ax_equity.grid(alpha=0.3)

    running_max = result.equity_curve.cummax()
    drawdown_pct = (result.equity_curve - running_max) / running_max * 100
    ax_drawdown.fill_between(drawdown_pct.index, drawdown_pct.values, 0, color="crimson", alpha=0.4)
    ax_drawdown.set_ylabel("Drawdown (%)")
    ax_drawdown.grid(alpha=0.3)

    ax_drawdown.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_strategy_comparison(
    results: dict[str, BacktestResult],
    title: str = "Strategy Comparison",
    save_path: str | None = None,
):
    """Overlay multiple strategies' equity curves on one chart — useful for
    the 'does this beat buy-and-hold' comparison the README calls for."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, result in results.items():
        ax.plot(result.equity_curve.index, result.equity_curve.values, label=name, linewidth=1.3)

    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_walk_forward_windows(report, title: str = "Walk-Forward Test Window Returns", save_path: str | None = None):
    """Bar chart of each walk-forward window's return — makes it immediately
    visible whether a strategy is consistently profitable or just got lucky
    on one window."""
    df = report.per_window_metrics
    fig, ax = plt.subplots(figsize=(11, 5))

    colors = ["seagreen" if v > 0 else "crimson" for v in df["total_return_pct"]]
    ax.bar(range(len(df)), df["total_return_pct"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in df["test_start"]], rotation=45, ha="right")
    ax.set_ylabel("Test Window Return (%)")
    ax.set_title(f"{title} ({report.combined_metrics.get('pct_windows_profitable', '?')}% profitable windows)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
