"""Performance metrics computed from a :class:`~backtest.engine.BacktestResult`."""

from __future__ import annotations

import pandas as pd

from .engine import BacktestResult, Trade


def total_return(equity: pd.Series, base: float | None = None) -> float:
    """Return the compounded return over the whole equity curve.

    ``base`` is the capital the return is measured against; it defaults to the
    first equity value.
    """
    return float(equity.iloc[-1] / (base if base is not None else equity.iloc[0]) - 1)


def max_drawdown(equity: pd.Series) -> float:
    """Return the worst peak-to-trough drawdown as a negative fraction."""
    return float((equity / equity.cummax() - 1).min())


def n_trades(trades: list[Trade]) -> int:
    """Return the number of closed round trips."""
    return sum(1 for trade in trades if trade.exit_date is not None)


def has_open_position(trades: list[Trade]) -> bool:
    """Return whether any trade is still open at the end of the window."""
    return any(trade.exit_date is None for trade in trades)


def summarize(result: BacktestResult) -> dict:
    """Return a JSON-friendly summary of strategy and buy-and-hold performance."""
    start = result.equity.index[0].date().isoformat()
    end = result.equity.index[-1].date().isoformat()
    return {
        "strategy": {
            "total_return": total_return(result.equity, base=result.initial_capital),
            "max_drawdown": max_drawdown(result.equity),
            "n_trades": n_trades(result.trades),
            "open_position": has_open_position(result.trades),
        },
        "buy_hold": {
            "total_return": total_return(result.benchmark, base=result.initial_capital),
            "max_drawdown": max_drawdown(result.benchmark),
            "n_trades": 1,
            "open_position": True,
        },
        "fee_rate": result.fee_rate,
        "initial_capital": result.initial_capital,
        "start": start,
        "end": end,
    }
