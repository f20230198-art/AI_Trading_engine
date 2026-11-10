"""
DOOM ENGINE — Backtest Runner

Measures the engine's strategies against real historical data.

    python run_backtest.py                  # 2 years, default costs
    python run_backtest.py --period 5y      # longer window
    python run_backtest.py --no-costs       # frictionless, for comparison
    python run_backtest.py --json out.json  # machine-readable results

Data comes from yfinance, so no API keys are required.
"""
import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import pandas as pd

from config import WATCHLIST_ALL
from core.backtest import BacktestConfig, BacktestEngine, buy_and_hold_benchmark

BENCHMARK = "SPY"


def fetch_prices(symbols, period):
    """Download daily OHLCV. auto_adjust handles splits and dividends."""
    import yfinance as yf

    print(f"Fetching {len(symbols)} symbols + {BENCHMARK} over {period}...")
    raw = yf.download(
        symbols + [BENCHMARK], period=period, progress=False, auto_adjust=True
    )
    if raw.empty:
        raise SystemExit("No data returned. Check your network connection.")

    out = {}
    for symbol in symbols + [BENCHMARK]:
        try:
            df = pd.DataFrame({
                "open": raw["Open"][symbol],
                "high": raw["High"][symbol],
                "low": raw["Low"][symbol],
                "close": raw["Close"][symbol],
                "volume": raw["Volume"][symbol],
            }).dropna()
        except KeyError:
            print(f"  warning: no data for {symbol}, skipping")
            continue
        if len(df) > 60:
            out[symbol] = df

    print(f"Loaded {len(out)} symbols, "
          f"{len(next(iter(out.values())))} bars "
          f"({next(iter(out.values())).index[0].date()} -> "
          f"{next(iter(out.values())).index[-1].date()})\n")
    return out


def _row(label, strategy, benchmark, suffix="", better="high"):
    """One comparison row, marked with whether the strategy beat the benchmark."""
    if benchmark is None:
        return f"  {label:<22} {strategy:>12}{suffix}"
    if better == "high":
        mark = "+" if strategy > benchmark else " "
    else:
        mark = "+" if strategy < benchmark else " "
    return (f"  {label:<22} {strategy:>12}{suffix} "
            f"{benchmark:>12}{suffix}  {mark}")


def print_report(metrics, benchmark, config, period):
    line = "=" * 68
    print(line)
    print("  DOOM ENGINE — BACKTEST RESULTS")
    print(line)
    print(f"  Period           : {period}")
    print(f"  Universe         : {len(WATCHLIST_ALL)} symbols, 3 sectors")
    print(f"  Trading days     : {metrics['trading_days']}")
    print(f"  Initial capital  : ${config.initial_capital:,.0f}")
    if config.commission_per_share > 0 or config.slippage_bps > 0:
        print(f"  Costs modeled    : ${config.commission_per_share}/share "
              f"(min ${config.min_commission}), {config.slippage_bps} bps slippage")
    else:
        print("  Costs modeled    : NONE (frictionless)")
    print()

    print(f"  {'':<22} {'STRATEGY':>12} {'SPY B&H':>12}")
    print(f"  {'-' * 62}")
    print(_row("Final equity", f"${metrics['final_equity']:,.0f}",
               f"${benchmark['final_equity']:,.0f}" if benchmark else None))
    print(_row("Total return", metrics["total_return_pct"],
               benchmark.get("total_return_pct") if benchmark else None, "%"))
    print(_row("CAGR", metrics["cagr_pct"],
               benchmark.get("cagr_pct") if benchmark else None, "%"))
    print(_row("Sharpe ratio", metrics["sharpe_ratio"],
               benchmark.get("sharpe_ratio") if benchmark else None))
    print(_row("Max drawdown", metrics["max_drawdown_pct"],
               benchmark.get("max_drawdown_pct") if benchmark else None, "%", better="low"))
    print()

    print("  RISK-ADJUSTED")
    print(f"    Sortino ratio       {metrics['sortino_ratio']:>10}")
    print(f"    Calmar ratio        {metrics['calmar_ratio']:>10}")
    print()

    print("  TRADE STATISTICS")
    print(f"    Total trades        {metrics['total_trades']:>10}")
    print(f"    Win rate            {metrics['win_rate_pct']:>10}%")
    pf = metrics["profit_factor"]
    print(f"    Profit factor       {pf if pf is not None else 'inf':>10}")
    print(f"    Average win         ${metrics['avg_win']:>9,.2f}")
    print(f"    Average loss        ${metrics['avg_loss']:>9,.2f}")
    print(f"    Avg holding period  {metrics['avg_holding_days']:>10} days")
    print()

    print("  COSTS")
    print(f"    Total costs paid    ${metrics['total_costs']:>9,.2f}")
    print(f"    Cost drag           {metrics['cost_drag_pct']:>10}% of capital")
    print()

    if metrics["by_strategy"]:
        print("  BY STRATEGY")
        for name, stats in sorted(metrics["by_strategy"].items()):
            print(f"    {name:<18} {stats['trades']:>4} trades  "
                  f"{stats['win_rate']:>5}% win  "
                  f"${stats['total_pnl']:>10,.2f}")
        print()

    if metrics["exit_reasons"]:
        print("  EXIT BREAKDOWN")
        total = sum(metrics["exit_reasons"].values())
        for reason, count in sorted(metrics["exit_reasons"].items(),
                                    key=lambda kv: -kv[1]):
            print(f"    {reason:<18} {count:>4}  ({count / total * 100:>5.1f}%)")
        print()

    print(line)
    if benchmark:
        alpha = metrics["total_return_pct"] - benchmark["total_return_pct"]
        verdict = "OUTPERFORMED" if alpha > 0 else "UNDERPERFORMED"
        print(f"  {verdict} SPY buy-and-hold by {abs(alpha):.2f} "
              f"percentage points")
    print(line)


def _ensure_dir(path):
    """Create the parent directory for an output path if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Backtest the DOOM ENGINE strategies")
    parser.add_argument("--period", default="2y", help="yfinance period (1y, 2y, 5y, max)")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--no-costs", action="store_true",
                        help="disable commission and slippage")
    parser.add_argument("--stop", type=float, default=None,
                        help="stop loss %% (default: BacktestConfig value)")
    parser.add_argument("--target", type=float, default=None,
                        help="take profit %% (default: BacktestConfig value)")
    parser.add_argument("--json", metavar="PATH", help="write results as JSON")
    parser.add_argument("--trades", metavar="PATH", help="write trade log as CSV")
    args = parser.parse_args()

    prices = fetch_prices(list(WATCHLIST_ALL), args.period)
    spy = prices.pop(BENCHMARK, None)

    config = BacktestConfig(initial_capital=args.capital)
    if args.stop is not None:
        config.stop_loss_pct = args.stop
    if args.target is not None:
        config.take_profit_pct = args.target
    if args.no_costs:
        config.commission_per_share = 0.0
        config.min_commission = 0.0
        config.slippage_bps = 0.0

    engine = BacktestEngine(config)
    print("Running backtest...")
    metrics = engine.run(prices)

    benchmark = (
        buy_and_hold_benchmark(spy, args.capital, config.warmup_bars)
        if spy is not None else None
    )

    print()
    print_report(metrics, benchmark, config, args.period)

    if args.json:
        _ensure_dir(args.json)
        payload = {
            "generated_at": datetime.utcnow().isoformat(),
            "period": args.period,
            "config": {
                "initial_capital": config.initial_capital,
                "commission_per_share": config.commission_per_share,
                "slippage_bps": config.slippage_bps,
                "stop_loss_pct": config.stop_loss_pct,
                "take_profit_pct": config.take_profit_pct,
            },
            "strategy": metrics,
            "benchmark": {k: v for k, v in (benchmark or {}).items()
                          if k != "equity_curve"},
            "equity_curve": [
                {"date": e["date"].strftime("%Y-%m-%d"), "equity": e["equity"]}
                for e in engine.equity_curve
            ],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote results to {args.json}")

    if args.trades:
        _ensure_dir(args.trades)
        rows = [{
            "symbol": t.symbol, "strategy": t.strategy,
            "entry_date": t.entry_date.strftime("%Y-%m-%d"),
            "exit_date": t.exit_date.strftime("%Y-%m-%d"),
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "qty": t.qty, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
            "costs": t.costs, "exit_reason": t.exit_reason,
            "holding_days": t.holding_days,
        } for t in engine.trades]
        pd.DataFrame(rows).to_csv(args.trades, index=False)
        print(f"Wrote {len(rows)} trades to {args.trades}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
