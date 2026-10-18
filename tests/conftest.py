"""Shared test data builders."""

from __future__ import annotations

import pandas as pd


def make_prices(closes: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Build a deterministic OHLCV frame on business days with the given closes."""
    index = pd.bdate_range(start=start, periods=len(closes), name="date")
    close = pd.Series([float(value) for value in closes], index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": pd.Series(
                [1000.0 + i for i in range(len(closes))], index=index
            ),
        },
        index=index,
    )


def make_trending_prices(n_up: int, n_down: int) -> pd.DataFrame:
    """Build prices that rise for ``n_up`` days then fall for ``n_down`` days.

    With a short ``fast`` window and a slightly longer ``slow`` window the fast
    SMA is above the slow SMA during the rise and below it after the fall, so
    the signal crosses down exactly once.
    """
    up = [100.0 + i for i in range(n_up)]
    peak = up[-1] if up else 100.0
    down = [peak - i for i in range(1, n_down + 1)]
    return make_prices(up + down)
