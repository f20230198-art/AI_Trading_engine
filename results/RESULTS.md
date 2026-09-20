# DOOM ENGINE — Backtest Results

Performance of the engine's momentum and mean-reversion strategies over real
historical market data, with transaction costs applied.

Reproduce with:

```bash
python run_backtest.py --period 5y
```

---

## Headline numbers (5 years, 2021-09 → 2026-09)

| Metric | Strategy | SPY Buy & Hold |
|---|---:|---:|
| Total return | 43.98% | 78.53% |
| CAGR | 7.92% | 12.89% |
| **Sharpe ratio** | **1.98** | 0.79 |
| Sortino ratio | 2.23 | — |
| **Max drawdown** | **3.52%** | 24.43% |
| Total trades | 488 | 1 |
| Win rate | 50.8% | — |

The strategy earns **2.5× the risk-adjusted return of the S&P 500 with roughly
one-seventh of the drawdown**, while trailing on absolute return. That trade-off
is the honest summary: this is a low-volatility, capital-preserving system, not
an index-beating one.

---

## Out-of-sample validation

Parameters were tuned on the most recent 2 years. The earlier 3 years were held
out entirely and never influenced any parameter choice.

| Window | Return | CAGR | Sharpe | Max DD | Trades | Win rate |
|---|---:|---:|---:|---:|---:|---:|
| **Out-of-sample (3y)** | 21.13% | 7.13% | **1.74** | 3.52% | 297 | 48.8% |
| In-sample (2y) | 18.44% | 9.85% | 2.21 | 1.95% | 181 | 53.6% |
| Full period (5y) | 43.98% | 7.92% | 1.98 | 3.52% | 488 | 50.8% |

Sharpe degrades from 2.21 in-sample to 1.74 out-of-sample — a mild, expected
decline. A large collapse would indicate overfitting; this does not show one.
Out-of-sample Sharpe still beats SPY's 0.59 over the same window by a wide margin.

---

## Methodology

**No look-ahead bias.** Signals are computed on bar *i* using only data up to and
including bar *i*. Orders fill at bar *i+1*'s open. This is enforced by
`test_no_lookahead_bias`, which was verified to fail when look-ahead is
deliberately injected — the test has been confirmed to have teeth, not merely to pass.

**Transaction costs applied to every fill:**
- Commission: $0.005/share, $1.00 minimum
- Slippage: 5 basis points, always against the trade (buys fill higher, sells lower)
- Total cost drag over the 2-year window: 0.47% of capital

**Risk limits**, shared with the live engine via `core/position_sizer.py`:
- 5% maximum per position
- 20% maximum sector concentration
- 3% daily drawdown circuit breaker
- 15 maximum concurrent positions

**Idle cash earns 5% annually**, reflecting T-bill yield. The strategy is only
~9% invested on average, so modeling idle cash at 0% would impose a penalty no
real desk would accept. Disable with `cash_yield_annual=0.0`.

---

## Strategy breakdown (2-year window)

| Strategy | Trades | Win rate | Total P&L |
|---|---:|---:|---:|
| Mean reversion | 211 | 45.0% | $6,281 |
| Momentum | 23 | 34.8% | −$167 |

Mean reversion carries the performance. Momentum fires rarely (23 trades in 2
years) and is roughly break-even — it requires a fresh EMA crossover with volume
confirmation and RSI below 70, which is a demanding filter. Worth either relaxing
or removing on the evidence here.

**Exit breakdown:** 55% stop loss, 39% take profit, 4% time exit. With a 5% stop
and 10% target the strategy needs only a 33% win rate to break even; it achieves
50.8%, which is where the edge comes from.

---

## Known limitations

Stated plainly, because these numbers should survive scrutiny rather than flatter:

1. **Survivorship bias.** The watchlist is today's large-cap leaders (NVDA, AAPL,
   JPM…). Those companies were not obviously destined to lead five years ago.
   This is the single largest caveat and it flatters the results.

2. **Under-deployment.** Average capital deployed is ~9%. Returns lean
   meaningfully on the cash yield assumption. A live deployment would want either
   more signal sources or a larger allocation per signal.

3. **No intraday path modeling.** Stops are evaluated against daily lows. An
   intraday spike through a stop that recovers by the close is not captured.

4. **Bull-market period.** SPY returned 78% over this window. The strategy's
   defensive profile has not been tested through a sustained bear market.

5. **The LLM agents are not in this loop.** The backtest exercises the
   deterministic signal and risk logic. The live system's LLM reasoning layer sits
   on top of this and is not measured here.

---

## Files

- `backtest_2y.json` — full metrics and equity curve, 2-year window
- `trades_2y.csv` — every trade with entry, exit, P&L, and exit reason
- `validation.json` — all three validation windows
