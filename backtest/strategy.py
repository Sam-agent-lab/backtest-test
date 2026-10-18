"""SMA crossover signal layer (no engine, fees, or metrics)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma_crossover_signals(
    df: pd.DataFrame, fast: int = 50, slow: int = 200
) -> pd.DataFrame:
    """Return a copy of ``df`` with SMA fast/slow, signal, and position columns.

    ``position`` is already the look-ahead-free representation: the decision is
    taken on the close of day ``t`` and only the return of day ``t+1`` is earned
    from it, so no shifting is applied here.
    """
    result = df.copy()
    result["sma_fast"] = result["close"].rolling(window=fast, min_periods=fast).mean()
    result["sma_slow"] = result["close"].rolling(window=slow, min_periods=slow).mean()
    signal = pd.Series(
        np.where(result["sma_fast"] > result["sma_slow"], 1.0, 0.0),
        index=result.index,
    )
    result["signal"] = signal.where(result["sma_slow"].notna())
    result["position"] = result["signal"].copy()
    return result


def live_start(df: pd.DataFrame) -> pd.Timestamp:
    """Return the first timestamp with a non-NaN position (both SMAs exist)."""
    valid = df["position"].dropna()
    if valid.empty:
        raise ValueError("No live position found: position column is all NaN.")
    return valid.index[0]
