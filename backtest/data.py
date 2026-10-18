"""BTC-USD data download layer (no strategy or trading logic)."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = ["open", "high", "low", "close", "volume"]
HEADERS = {"User-Agent": "Mozilla/5.0"}

def _parse_date(value: str) -> date:
    return date.fromisoformat(value)

def _effective_end(end: str | None) -> date:
    return _parse_date(end) if end else datetime.now(UTC).date()

def _midnight(day: date) -> pd.Timestamp:
    return pd.Timestamp(day)

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS, dtype="float64",
                            index=pd.DatetimeIndex([], name="date"))
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df.dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df[COLUMNS]

def _slice(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (df.index >= _midnight(start)) & (df.index <= _midnight(end))
    return df.loc[mask].copy()

def _fetch_yahoo(start: date, end: date) -> pd.DataFrame:
    params = {
        "period1": int(_midnight(start).tz_localize("UTC").timestamp()),
        "period2": int((_midnight(end) + timedelta(days=1)).tz_localize("UTC").timestamp()),
        "interval": "1d", "events": "div,split",
    }
    response = requests.get(YAHOO_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    result = (response.json().get("chart") or {}).get("result") or []
    if not result:
        raise ValueError("Yahoo Finance returned no result")
    item = result[0]
    timestamps = item.get("timestamp") or []
    quote = ((item.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps or not quote.get("close"):
        raise ValueError("Yahoo Finance returned no usable rows")
    index = pd.to_datetime(timestamps, unit="s", utc=True)
    data = {}
    for c in COLUMNS:
        values = list(quote.get(c) or [])[: len(index)]
        values.extend([None] * (len(index) - len(values)))
        data[c] = values
    df = _normalize(pd.DataFrame(data, index=index))
    if df.empty:
        raise ValueError("Yahoo Finance returned no usable rows after normalization")
    return df

def _fetch_binance(start: date, end: date) -> pd.DataFrame:
    cursor = int(_midnight(start).tz_localize("UTC").timestamp() * 1000)
    end_ms = int((_midnight(end) + timedelta(days=1)).tz_localize("UTC").timestamp() * 1000) - 1
    rows: list[list] = []
    while cursor <= end_ms:
        params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000,
                  "startTime": cursor, "endTime": end_ms}
        response = requests.get(BINANCE_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        cursor = int(batch[-1][0]) + 86_400_000
    if not rows:
        raise ValueError("Binance returned no usable rows")
    index = pd.to_datetime([row[0] for row in rows], unit="ms", utc=True)
    data = {col: [float(row[i]) for row in rows] for i, col in enumerate(COLUMNS, 1)}
    return _normalize(pd.DataFrame(data, index=index))

def _try_fetch(name, func, start: date, end: date):
    try:
        df = func(start, end)
        if not df.empty:
            logger.info("Using %s source (%d rows)", name, len(df))
            return df, None
        return None, f"{name} returned no usable rows"
    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        logger.warning("%s failed: %s", name, exc)
        return None, f"{name}: {exc}"

def fetch_btc_usd(start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily BTC-USD OHLCV, trying Yahoo Finance then Binance."""
    start_day, end_day = _parse_date(start), _effective_end(end)
    if end_day < start_day:
        raise ValueError("end must be on or after start")
    errors: list[str] = []
    for name, func in (("Yahoo Finance", _fetch_yahoo), ("Binance", _fetch_binance)):
        df, error = _try_fetch(name, func, start_day, end_day)
        if df is not None:
            return df
        errors.append(error or name)
    raise RuntimeError("Could not download BTC-USD data: " + "; ".join(errors))

def load_prices(
    start: str,
    end: str | None = None,
    cache_path: str = "data/btc-usd.csv",
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load daily BTC-USD prices for [start, end], using a CSV cache if it covers the range."""
    start_day, end_day = _parse_date(start), _effective_end(end)
    if end_day < start_day:
        raise ValueError("end must be on or after start")
    path = Path(cache_path)
    if use_cache and not refresh and path.exists():
        cached = _normalize(pd.read_csv(path, parse_dates=["date"], index_col="date"))
        if (not cached.empty and cached.index.min() <= _midnight(start_day)
                and cached.index.max() >= _midnight(end_day)):
            logger.info("Using cached prices from %s", path)
            return _slice(cached, start_day, end_day)
    df = fetch_btc_usd(start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date")
    return _slice(df, start_day, end_day)

def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily BTC-USD prices.")
    parser.add_argument("--start", default="2022-07-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cache", default="data/btc-usd.csv")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = load_prices(args.start, args.end, cache_path=args.cache, refresh=args.refresh)
    print(f"Rows: {len(df)}")
    print(f"First date: {df.index.min().date() if len(df) else 'n/a'}")
    print(f"Last date: {df.index.max().date() if len(df) else 'n/a'}")
    print(f"Last close: {df['close'].iloc[-1] if len(df) else 'n/a'}")

if __name__ == "__main__":
    main()
