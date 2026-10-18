"""Tests for the fee-aware backtest engine."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.metrics import has_open_position, n_trades
from tests.conftest import make_prices

FEE = 0.001
CAPITAL = 1000.0


def engine_frame(closes: list[float], positions: list[float]) -> pd.DataFrame:
    prices = make_prices(closes)
    prices["position"] = positions
    return prices


@pytest.mark.parametrize(
    ("positions", "expected"),
    [([1.0, 1.0, 0.0], CAPITAL * (1 - FEE)), ([0.0, 1.0, 0.0], CAPITAL)],
)
def test_first_equity_value_only_charges_an_initial_entry(positions, expected):
    result = run_backtest(
        engine_frame([100.0, 110.0, 121.0], positions),
        initial_capital=CAPITAL,
        fee_rate=FEE,
    )

    assert result.equity.iloc[0] == pytest.approx(expected)


def test_two_day_hold_pays_entry_and_exit_fees():
    result = run_backtest(
        engine_frame([100.0, 110.0, 121.0], [1.0, 1.0, 0.0]),
        initial_capital=CAPITAL,
        fee_rate=FEE,
    )

    expected_first_step = CAPITAL * (1 - FEE) * (110.0 / 100.0)
    expected_final = CAPITAL * (1 - FEE) * (121.0 / 100.0) * (1 - FEE)

    assert result.equity.iloc[1] == pytest.approx(expected_first_step)
    assert result.equity.iloc[-1] == pytest.approx(expected_final)


def test_flat_position_series_keeps_equity_exactly_constant():
    result = run_backtest(
        engine_frame([100.0, 50.0, 200.0, 10.0], [0.0, 0.0, 0.0, 0.0]),
        initial_capital=CAPITAL,
        fee_rate=FEE,
    )

    assert (result.equity == CAPITAL).all()


def test_equity_prefix_does_not_depend_on_future_prices():
    closes = [100.0, 110.0, 121.0, 100.0, 90.0, 95.0]
    positions = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    base = run_backtest(
        engine_frame(closes, positions), initial_capital=CAPITAL, fee_rate=FEE
    )

    for cut in range(1, len(closes)):
        modified = closes.copy()
        modified[cut:] = [value * 2.0 + 7.0 for value in modified[cut:]]
        result = run_backtest(
            engine_frame(modified, positions),
            initial_capital=CAPITAL,
            fee_rate=FEE,
        )
        pd.testing.assert_series_equal(
            result.equity.iloc[:cut], base.equity.iloc[:cut]
        )


def test_flat_benchmark_pays_exactly_one_entry_fee():
    expected = CAPITAL * (1 - FEE)
    result = run_backtest(
        engine_frame([100.0] * 5, [0.0, 1.0, 1.0, 0.0, 1.0]),
        initial_capital=CAPITAL,
        fee_rate=FEE,
    )

    assert result.benchmark.iloc[0] == pytest.approx(expected)
    assert (result.benchmark == expected).all()


def test_closed_trades_match_round_trips_and_switching_closes():
    closes = [100.0, 110.0, 121.0, 100.0, 90.0, 95.0, 105.0]
    positions = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    prices = engine_frame(closes, positions)
    result = run_backtest(prices, initial_capital=CAPITAL, fee_rate=FEE)

    expected = [(1, 3, 110.0, 100.0), (4, 6, 90.0, 105.0)]
    assert len(result.trades) == 2
    assert n_trades(result.trades) == 2
    assert not has_open_position(result.trades)

    for trade, (entry_idx, exit_idx, entry_price, exit_price) in zip(
        result.trades, expected, strict=True
    ):
        assert trade.entry_date == prices.index[entry_idx]
        assert trade.exit_date == prices.index[exit_idx]
        assert trade.entry_price == pytest.approx(entry_price)
        assert trade.exit_price == pytest.approx(exit_price)
        expected_return = (exit_price / entry_price) * (1 - FEE) ** 2 - 1
        assert trade.net_return == pytest.approx(expected_return)


def test_open_trade_has_no_exit_and_is_not_counted():
    prices = engine_frame([100.0, 110.0, 121.0], [0.0, 1.0, 1.0])
    result = run_backtest(prices, initial_capital=CAPITAL, fee_rate=FEE)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == prices.index[1]
    assert trade.entry_price == pytest.approx(110.0)
    assert trade.exit_date is None
    assert trade.exit_price is None
    assert trade.net_return is None
    assert n_trades(result.trades) == 0
    assert has_open_position(result.trades)


@pytest.mark.parametrize(
    ("closes", "positions"),
    [
        ([100.0], [1.0]),
        ([100.0, 110.0], [float("nan"), float("nan")]),
    ],
)
def test_run_backtest_raises_with_fewer_than_two_usable_rows(closes, positions):
    with pytest.raises(ValueError):
        run_backtest(
            engine_frame(closes, positions),
            initial_capital=CAPITAL,
            fee_rate=FEE,
        )
