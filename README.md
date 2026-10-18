# BTC-USD SMA 50/200 crossover backtest

This project backtests a simple long-only SMA 50/200 crossover strategy on daily
BTC-USD prices and compares it with a buy & hold benchmark over roughly the last
three years. The strategy is all-in or all-out: it holds BTC when the 50-day SMA
is above the 200-day SMA and stays in cash otherwise. A 0.1% fee is charged on
every entry and every exit, so a full round trip costs about 0.2%. Both the
strategy and buy & hold start on the same day with 10,000 USD of initial capital.

## Data

- Source: the free Yahoo Finance chart API for `BTC-USD`, with no API key.
- Fallback: the Binance `BTCUSDT` klines API if Yahoo Finance fails or returns
  no usable rows.
- Download window: starts on `2022-07-01`, so the series includes the warm-up
  needed for the 200-day moving average before the live window begins.
- Cache: `data/btc-usd.csv`. The `data/` directory is gitignored, so the cache
  is not committed.

## Method

- The signal is computed on the close of day `t`: long when SMA(50) > SMA(200),
  flat otherwise.
- The position is held from the close of day `t` to the close of day `t+1`, so
  the return of day `t+1` is earned from a decision taken at the close of day
  `t`. There is no look-ahead.
- Long-only, no leverage: the portfolio is either all-in BTC or all-in cash.
- A 0.1% fee is charged on every entry and every exit (a round trip costs about
  0.2%).
- Both the strategy and buy & hold start on the same day with the same initial
  capital. That day is the first day where the 200-day average exists AND the
  day is inside the last 3 years.
- Initial capital: 10,000 USD.

## Results

Produced by `.venv/bin/python -m backtest.run` on the cached dataset:

```text
Period: 2023-10-19 -> 2026-10-18
Rows:   1096

Metric              SMA 50/200    Buy & hold
Total return           +74.18%      +183.44%
Max drawdown           -39.56%       -53.06%
Trades                       3             1

Fee rate: 0.10% per trade
Saved:    output/equity_curves.png
Summary:  output/summary.json
```

- Period used: `2023-10-19` to `2026-10-18` (1,096 daily rows).
- Total return: `+74.18%` for the SMA 50/200 strategy, `+183.44%` for buy & hold.
- Max drawdown: `-39.56%` for the strategy, `-53.06%` for buy & hold.
- Closed trades: `3` for the strategy, `1` for buy & hold.
- Fee rate: `0.10%` per trade.
- Chart: `output/equity_curves.png` (equity curves and the indexed BTC-USD close
  on a log scale). A machine-readable summary is written to
  `output/summary.json`.

### Trades taken by the strategy

| Entry | Exit | Entry price | Exit price | Net return |
|---|---|---|---|---|
| 2023-10-30 | 2024-08-10 | 34,502.36 | 60,945.81 | +76.29% |
| 2024-10-28 | 2025-04-07 | 69,907.76 | 79,235.34 | +13.12% |
| 2025-05-22 | 2025-11-16 | 111,673.28 | 94,177.08 | -15.84% |
| 2026-09-08 | still open | 78,438.58 | - | - |

Net returns already include the 0.1% entry and 0.1% exit fee. Over this window
buy & hold beat the strategy: the crossover filter cut the drawdown from -53.06%
to -39.56%, but it also missed a large part of the up move, and it was long
during the 2025-2026 decline while sitting out the recoveries.

## How to run

From the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m backtest.run
.venv/bin/python -m pytest -q
.venv/bin/ruff check backtest tests
```

`backtest.run` downloads the data on the first run (or when `--refresh` is
passed), writes the CSV cache, and regenerates the chart and summary.

## Assumptions and limitations

- Long-only: the strategy never shorts and never uses leverage.
- No slippage or market impact; fills happen at the daily close.
- Fees are the only trading cost, charged at 0.1% per entry and per exit.
- The last open position is marked to market in the equity curve but is not
  counted as a closed trade.
- When the Binance fallback is used, prices are in USDT rather than USD, so
  small differences from a true USD price series are possible.
- Past performance is not indicative of future results.

## Project layout

```text
.
├── backtest/
│   ├── __init__.py
│   ├── chart.py       # equity-curve plotting
│   ├── data.py        # Yahoo Finance / Binance download and CSV cache
│   ├── engine.py      # fee-aware equity curves, benchmark, trade log
│   ├── metrics.py     # total return, max drawdown, trade counts, summary
│   ├── run.py         # end-to-end CLI
│   └── strategy.py    # SMA 50/200 signal and live-start detection
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_engine.py
│   ├── test_metrics.py
│   └── test_strategy.py
├── output/            # generated chart and summary (gitignored)
│   ├── equity_curves.png
│   └── summary.json
├── data/              # local price cache (gitignored)
│   └── btc-usd.csv
├── requirements.txt
└── README.md
```
