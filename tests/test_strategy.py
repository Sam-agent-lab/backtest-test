"""Tests for the SMA crossover signal layer."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.strategy import live_start, sma_crossover_signals
from tests.conftest import make_prices, make_trending_prices


def test_signals_are_nan_until_slow_sma_exists_and_never_zero():
    prices = make_prices([100.0 + i for i in range(10)])
    result = sma_crossover_signals(prices, fast=2, slow=4)

    slow_missing = result["sma_slow"].isna()
    assert slow_missing.iloc[:3].all()
    assert result.loc[slow_missing, "signal"].isna().all()
    assert (result.loc[slow_missing, "signal"] == 0).sum() == 0
    assert result.loc[~slow_missing, "signal"].isin([0.0, 1.0]).all()


def test_sma_columns_match_manual_and_pandas_rolling_means():
    prices = make_prices([10.0, 20.0, 30.0, 40.0, 50.0])
    result = sma_crossover_signals(prices, fast=2, slow=3)

    expected_fast = pd.Series(
        [float("nan"), 15.0, 25.0, 35.0, 45.0],
        index=prices.index,
        name="sma_fast",
    )
    expected_slow = pd.Series(
        [float("nan"), float("nan"), 20.0, 30.0, 40.0],
        index=prices.index,
        name="sma_slow",
    )

    pd.testing.assert_series_equal(result["sma_fast"], expected_fast)
    pd.testing.assert_series_equal(result["sma_slow"], expected_slow)
    pd.testing.assert_series_equal(
        result["sma_fast"],
        prices["close"].rolling(window=2, min_periods=2).mean().rename("sma_fast"),
    )
    pd.testing.assert_series_equal(
        result["sma_slow"],
        prices["close"].rolling(window=3, min_periods=3).mean().rename("sma_slow"),
    )


def test_signal_is_one_on_a_rising_series_and_zero_on_a_falling_one():
    rising = sma_crossover_signals(
        make_prices([100.0 + i for i in range(12)]), fast=3, slow=5
    )
    falling = sma_crossover_signals(
        make_prices([200.0 - i for i in range(12)]), fast=3, slow=5
    )

    rising_valid = rising["sma_slow"].notna()
    falling_valid = falling["sma_slow"].notna()
    assert (rising.loc[rising_valid, "signal"] == 1.0).all()
    assert (falling.loc[falling_valid, "signal"] == 0.0).all()

    expected_rising = (
        (rising["sma_fast"] > rising["sma_slow"])
        .astype(float)
        .where(rising_valid)
        .rename("signal")
    )
    pd.testing.assert_series_equal(rising["signal"], expected_rising)


def test_trending_prices_cross_down_exactly_once():
    result = sma_crossover_signals(make_trending_prices(20, 20), fast=2, slow=4)
    position = result["position"].dropna()

    assert position.iloc[0] == 1.0
    assert (position.diff() == 1.0).sum() == 0
    assert (position.diff() == -1.0).sum() == 1
    assert position.iloc[-1] == 0.0


def test_mutating_a_future_close_does_not_change_past_positions():
    closes = [100.0 + i for i in range(40)]
    prices = make_prices(closes)
    base = sma_crossover_signals(prices, fast=2, slow=4)["position"]

    for t in range(len(closes) - 5, len(closes)):
        perturbed = prices.copy()
        perturbed.loc[perturbed.index[t], "close"] = 1.0
        result = sma_crossover_signals(perturbed, fast=2, slow=4)["position"]

        pd.testing.assert_series_equal(result.iloc[:t], base.iloc[:t])
        assert result.iloc[t] != base.iloc[t]


def test_live_start_returns_first_position_timestamp():
    signals = sma_crossover_signals(
        make_prices([100.0 + i for i in range(10)]), fast=2, slow=4
    )

    assert live_start(signals) == signals.index[3]


def test_live_start_raises_when_shorter_than_slow_window():
    signals = sma_crossover_signals(make_prices([100.0, 101.0, 102.0]), fast=2, slow=5)

    with pytest.raises(ValueError):
        live_start(signals)
