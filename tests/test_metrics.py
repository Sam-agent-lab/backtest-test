"""Tests for performance metrics and the summary shape."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.metrics import max_drawdown, summarize, total_return
from tests.conftest import make_prices


def test_total_return_on_a_hand_made_series():
    equity = pd.Series([100.0, 150.0])

    assert total_return(equity) == pytest.approx(0.5)


def test_max_drawdown_for_a_peak_then_trough():
    equity = pd.Series([100.0, 150.0, 75.0])

    drawdown = max_drawdown(equity)

    assert drawdown == pytest.approx(-0.5)
    assert isinstance(drawdown, float)


def test_max_drawdown_is_zero_for_a_monotonically_rising_series():
    equity = pd.Series([100.0, 110.0, 120.0])

    drawdown = max_drawdown(equity)

    assert drawdown == 0.0
    assert isinstance(drawdown, float)
    assert drawdown <= 0.0


def test_summarize_returns_documented_shape():
    prices = make_prices([100.0, 150.0, 120.0])
    prices["position"] = [1.0, 1.0, 0.0]
    result = run_backtest(prices, initial_capital=1000.0, fee_rate=0.001)

    summary = summarize(result)

    assert set(summary) == {
        "strategy",
        "buy_hold",
        "fee_rate",
        "initial_capital",
        "start",
        "end",
    }
    metric_keys = {"total_return", "max_drawdown", "n_trades", "open_position"}
    assert set(summary["strategy"]) == metric_keys
    assert set(summary["buy_hold"]) == metric_keys

    for section in (summary["strategy"], summary["buy_hold"]):
        assert isinstance(section["total_return"], float)
        assert isinstance(section["max_drawdown"], float)
        assert isinstance(section["n_trades"], int)
        assert isinstance(section["open_position"], bool)

    assert summary["strategy"]["n_trades"] == 1
    assert summary["buy_hold"]["n_trades"] == 1
    assert summary["buy_hold"]["open_position"] is True
    assert summary["fee_rate"] == 0.001
    assert summary["initial_capital"] == 1000.0
    assert summary["start"] == result.equity.index[0].date().isoformat()
    assert summary["end"] == result.equity.index[-1].date().isoformat()


def test_total_return_uses_the_given_base():
    equity = pd.Series([99.0, 198.0])

    assert total_return(equity) == pytest.approx(1.0)
    assert total_return(equity, base=100.0) == pytest.approx(0.98)


def test_summarize_measures_both_returns_against_the_initial_capital():
    prices = make_prices([100.0, 100.0, 100.0])
    prices["position"] = [0.0, 0.0, 0.0]
    result = run_backtest(prices, initial_capital=1000.0, fee_rate=0.001)

    summary = summarize(result)

    assert summary["buy_hold"]["total_return"] == pytest.approx(-0.001)
    assert summary["strategy"]["total_return"] == pytest.approx(0.0)
