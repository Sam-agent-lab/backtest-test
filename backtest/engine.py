"""Backtest engine: fee-aware equity curves, benchmark, and trade log."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Trade:
    """A single long position, possibly still open."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    entry_price: float
    exit_price: float | None
    net_return: float | None


@dataclasses.dataclass
class BacktestResult:
    """Output of :func:`run_backtest`."""

    equity: pd.Series
    benchmark: pd.Series
    position: pd.Series
    trades: list[Trade]
    initial_capital: float
    fee_rate: float


def run_backtest(
    prices: pd.DataFrame,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
) -> BacktestResult:
    """Run the position series in ``prices`` against daily closes.

    Fees are charged on entry, exit, and any turnover at the close of the day
    the position changes; the equity recurrence is an explicit loop so no
    compounding detail is lost to vectorisation.
    """
    d0 = prices["position"].first_valid_index()
    if d0 is None:
        raise ValueError("No usable rows: position column is all NaN.")
    live = prices.loc[d0:]
    live = live[live["position"].notna()]
    if len(live) < 2:
        raise ValueError("Fewer than 2 usable rows in the live window.")

    positions = live["position"].to_numpy(dtype=float)
    closes = live["close"].to_numpy(dtype=float)
    size = len(live)

    equity = np.empty(size, dtype=float)
    benchmark = np.empty(size, dtype=float)
    equity[0] = initial_capital * (1 - fee_rate * abs(positions[0]))
    benchmark[0] = initial_capital * (1 - fee_rate)
    for i in range(1, size):
        ret = closes[i] / closes[i - 1] - 1
        turnover = abs(positions[i] - positions[i - 1])
        equity[i] = equity[i - 1] * (1 + positions[i - 1] * ret) * (1 - fee_rate * turnover)
        benchmark[i] = benchmark[i - 1] * (1 + ret)

    trades: list[Trade] = []
    open_idx: int | None = None
    for i, position in enumerate(positions):
        if open_idx is None and position > 0:
            open_idx = i
        elif open_idx is not None and position == 0:
            entry, exit_ = closes[open_idx], closes[i]
            trades.append(
                Trade(
                    entry_date=live.index[open_idx],
                    exit_date=live.index[i],
                    entry_price=float(entry),
                    exit_price=float(exit_),
                    net_return=float((exit_ / entry) * (1 - fee_rate) ** 2 - 1),
                )
            )
            open_idx = None
    if open_idx is not None:
        trades.append(
            Trade(
                entry_date=live.index[open_idx],
                exit_date=None,
                entry_price=float(closes[open_idx]),
                exit_price=None,
                net_return=None,
            )
        )

    return BacktestResult(
        equity=pd.Series(equity, index=live.index, name="equity"),
        benchmark=pd.Series(benchmark, index=live.index, name="benchmark"),
        position=live["position"].astype(float).rename("position"),
        trades=trades,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
    )
