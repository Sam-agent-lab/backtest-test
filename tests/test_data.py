"""Tests for the CSV cache layer of the data module."""

from __future__ import annotations

import pandas as pd

from backtest import data
from tests.conftest import make_prices


def fail_fetch(*args, **kwargs):
    raise AssertionError("fetch_btc_usd must not be called when the cache is used")


def test_load_prices_reads_a_cache_that_covers_the_requested_range(
    tmp_path, monkeypatch
):
    path = tmp_path / "btc-usd.csv"
    prices = make_prices([100.0 + i for i in range(10)])
    prices.to_csv(path, index_label="date")
    monkeypatch.setattr(data, "fetch_btc_usd", fail_fetch)

    result = data.load_prices(
        "2020-01-03", "2020-01-08", cache_path=str(path)
    )

    expected = prices.loc["2020-01-03":"2020-01-08"]
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_load_prices_writes_the_cache_when_missing(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "btc-usd.csv"
    prices = make_prices([100.0 + i for i in range(10)])
    calls: list[tuple[str, str | None]] = []

    def fake_fetch(start, end=None):
        calls.append((start, end))
        return prices

    monkeypatch.setattr(data, "fetch_btc_usd", fake_fetch)

    result = data.load_prices(
        "2020-01-02", "2020-01-07", cache_path=str(path)
    )

    expected = prices.loc["2020-01-02":"2020-01-07"]
    assert path.exists()
    assert calls == [("2020-01-02", "2020-01-07")]
    pd.testing.assert_frame_equal(result, expected, check_freq=False)

    cached = pd.read_csv(path, parse_dates=["date"], index_col="date")
    assert list(cached.columns) == data.COLUMNS
    assert len(cached) == len(prices)


def test_cached_frame_is_sorted_unique_and_float_typed(tmp_path, monkeypatch):
    path = tmp_path / "cache.csv"
    raw = pd.DataFrame(
        {
            "date": [
                "2020-01-03",
                "2020-01-02",
                "2020-01-03",
                "2019-12-31",
                "2020-01-10",
                "2020-01-06",
            ],
            "open": [104.0, 101.0, 105.0, 98.0, 109.0, 106.0],
            "high": [106.0, 103.0, 107.0, 100.0, 111.0, 108.0],
            "low": [103.0, 100.0, 104.0, 97.0, 108.0, 105.0],
            "close": [105.0, 102.0, 106.0, 99.0, 110.0, None],
            "volume": [1.0, 2.0, 3.0, 4.0, 5.0, None],
        }
    )
    raw.to_csv(path, index=False)
    monkeypatch.setattr(data, "fetch_btc_usd", fail_fetch)

    result = data.load_prices(
        "2020-01-01", "2020-01-08", cache_path=str(path)
    )

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "date"
    assert result.index.is_monotonic_increasing
    assert not result.index.has_duplicates
    assert list(result.columns) == data.COLUMNS
    assert all(dtype == "float64" for dtype in result.dtypes)
    assert result["close"].notna().all()
    assert result.loc["2020-01-03", "close"] == 106.0
    assert len(result) == 2
