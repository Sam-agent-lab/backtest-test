"""End-to-end SMA 50/200 vs buy-and-hold CLI for BTC-USD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .chart import plot_equity_curves
from .data import load_prices
from .engine import run_backtest
from .metrics import summarize
from .strategy import live_start, sma_crossover_signals


def _pct(value: float) -> str:
    """Format a fraction as a signed percentage with two decimals."""
    return f"{value:+.2%}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BTC-USD SMA crossover backtest.")
    parser.add_argument("--start", default="2022-07-01", help="download start (includes warm-up)")
    parser.add_argument("--end", default=None, help="download end (default: today)")
    parser.add_argument("--live-start", default="2023-10-19", help="first day of the live window")
    parser.add_argument("--fast", type=int, default=50)
    parser.add_argument("--slow", type=int, default=200)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--cache", default="data/btc-usd.csv")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", default="output/equity_curves.png")
    return parser.parse_args()


def main() -> None:
    """Load data, run the backtest, print a summary, and save chart and JSON."""
    args = _parse_args()
    prices = load_prices(args.start, args.end, cache_path=args.cache, refresh=args.refresh)
    signals = sma_crossover_signals(prices, fast=args.fast, slow=args.slow)

    effective = max(live_start(signals), pd.Timestamp(args.live_start))
    live = signals.loc[signals.index >= effective].copy()

    result = run_backtest(live, initial_capital=args.capital, fee_rate=args.fee)
    summary = summarize(result)
    title = (
        f"BTC-USD - SMA {args.fast}/{args.slow} vs Buy & Hold - "
        f"{summary['start']} to {summary['end']} - fees {args.fee:.1%} per trade"
    )
    out_path = plot_equity_curves(result, out_path=args.out, title=title)

    summary_path = Path("output/summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    strategy, buy_hold = summary["strategy"], summary["buy_hold"]
    print(f"Period: {summary['start']} -> {summary['end']}")
    print(f"Rows:   {len(result.equity)}")
    print()
    print(f"{'Metric':<16}{'SMA ' + str(args.fast) + '/' + str(args.slow):>14}{'Buy & hold':>14}")
    print(f"{'Total return':<16}{_pct(strategy['total_return']):>14}{_pct(buy_hold['total_return']):>14}")
    print(f"{'Max drawdown':<16}{_pct(strategy['max_drawdown']):>14}{_pct(buy_hold['max_drawdown']):>14}")
    print(f"{'Trades':<16}{strategy['n_trades']:>14}{buy_hold['n_trades']:>14}")
    print()
    print(f"Fee rate: {args.fee:.2%} per trade")
    print(f"Saved:    {out_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
