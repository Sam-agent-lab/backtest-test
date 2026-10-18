"""Equity curve plotting for backtest results (headless, PNG output)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from .engine import BacktestResult


def _long_spans(position: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return contiguous date ranges where the strategy is long."""
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start: pd.Timestamp | None = None
    for day, value in position.items():
        if value > 0 and start is None:
            start = day
        elif value <= 0 and start is not None:
            spans.append((start, day))
            start = None
    if start is not None:
        spans.append((start, position.index[-1]))
    return spans


def plot_equity_curves(
    result: BacktestResult,
    out_path: str = "output/equity_curves.png",
    title: str | None = None,
) -> str:
    """Save normalised strategy/benchmark curves plus price on a log scale."""
    start = result.equity.index[0].date()
    end = result.equity.index[-1].date()
    if title is None:
        title = (
            f"BTC-USD - SMA 50/200 vs Buy & Hold - {start} to {end} "
            f"- fees {result.fee_rate:.1%} per trade"
        )

    strategy = result.equity / result.equity.iloc[0] * 100.0
    benchmark = result.benchmark / result.benchmark.iloc[0] * 100.0
    price = benchmark  # buy & hold is proportional to close, so this is an indexed close

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for left, right in _long_spans(result.position):
        ax_top.axvspan(left, right, color="green", alpha=0.10, linewidth=0)

    ax_top.plot(strategy.index, strategy, label="SMA 50/200", color="#1f77b4", lw=1.6)
    ax_top.plot(benchmark.index, benchmark, label="Buy & hold", color="#ff7f0e", lw=1.4)
    ax_top.set_ylabel("Equity (start = 100)")
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)
    handles, labels = ax_top.get_legend_handles_labels()
    handles.append(Patch(facecolor="green", alpha=0.10, edgecolor="none"))
    labels.append("strategy in market (long)")
    ax_top.legend(handles=handles, labels=labels, loc="best")

    ax_bottom.plot(price.index, price, color="#555555", lw=1.1)
    ax_bottom.set_yscale("log")
    ax_bottom.set_ylabel("BTC-USD close\n(indexed, log)")
    ax_bottom.set_xlabel("Date")
    ax_bottom.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return out_path
