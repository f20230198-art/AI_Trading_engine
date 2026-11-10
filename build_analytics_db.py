"""
DOOM ENGINE — Build the analytics database for the Analyst agent.

Usage:
    python build_analytics_db.py                    # build data/analytics.db from committed snapshots
    python build_analytics_db.py --refresh-regime   # re-download SPY and recompute regime labels
    python build_analytics_db.py --trades results/trades_2y.csv   # refresh the trade snapshot first

The snapshots in results/analytics/ are committed so the database (and the
Analyst eval) is reproducible offline. Refreshing them needs network access.
"""
import argparse
import os
import shutil
import sys

from core.analytics_db import ANALYTICS_DIR, DEFAULT_DB_PATH, build_analytics_db, compute_regime_daily


def refresh_regime(start: str, end: str):
    """Download SPY and write results/analytics/spy_regime_daily.csv."""
    import pandas as pd
    import yfinance as yf

    # Pull extra history before `start` so the first labelled day has a full lookback.
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    fetch_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download("SPY", start=fetch_start, end=fetch_end, progress=False, auto_adjust=True)
    if raw.empty:
        raise SystemExit("No SPY data returned. Check your network connection.")
    close = raw["Close"]
    if hasattr(close, "columns"):  # newer yfinance returns a one-column frame
        close = close.iloc[:, 0]
    regimes = compute_regime_daily(close.dropna())
    regimes = regimes[(regimes["date"] >= start) & (regimes["date"] <= end)]
    out = os.path.join(ANALYTICS_DIR, "spy_regime_daily.csv")
    regimes.to_csv(out, index=False)
    print(f"Wrote {len(regimes)} regime rows to {out}")
    print(regimes["regime"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="Build the Analyst agent's analytics DB")
    parser.add_argument("--out", default=DEFAULT_DB_PATH)
    parser.add_argument("--refresh-regime", action="store_true",
                        help="re-download SPY and recompute daily regime labels")
    parser.add_argument("--start", default="2024-11-29", help="regime start date")
    parser.add_argument("--end", default="2026-09-18", help="regime end date")
    parser.add_argument("--trades", metavar="PATH",
                        help="copy a run_backtest.py --trades CSV in as the trade snapshot")
    args = parser.parse_args()

    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    if args.trades:
        shutil.copyfile(args.trades, os.path.join(ANALYTICS_DIR, "backtest_trades_2y.csv"))
        print(f"Copied {args.trades} into the trade snapshot")
    if args.refresh_regime:
        refresh_regime(args.start, args.end)

    counts = build_analytics_db(args.out)
    print(f"Built {args.out}")
    for table, n in counts.items():
        print(f"  {table:<28} {n:>5} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
