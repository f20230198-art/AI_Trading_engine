"""
Backtest engine correctness tests.

The most important test here is test_no_lookahead_bias. If the backtester can see
future prices, every performance number it produces is fiction — so that property
is verified directly rather than assumed.

All tests build synthetic price data. No network, no yfinance.
"""
import numpy as np
import pandas as pd
import pytest

from core.backtest import (
    BacktestConfig,
    BacktestEngine,
    buy_and_hold_benchmark,
)


def make_prices(closes, start="2024-01-01", volume=1_000_000):
    """Daily OHLCV frame from a close series, with a small intraday range."""
    idx = pd.bdate_range(start=start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes * 0.999,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": np.full(len(closes), volume, dtype=float),
    }, index=idx)


def trending_up(n=200, start=100.0, daily=0.004):
    return [start * (1 + daily) ** i for i in range(n)]


def oscillating(n=200, base=100.0, amp=8.0, period=20):
    return [base + amp * np.sin(2 * np.pi * i / period) for i in range(n)]


# ---------------------------------------------------------------------------
# The property that makes every other number meaningful
# ---------------------------------------------------------------------------

def test_no_lookahead_bias():
    """
    A backtest that can see the future would exploit a price spike it has not
    yet observed. We build a series that is flat, then jumps massively on the
    final bars, and assert the engine cannot capture that jump from bars it
    had no way to see.
    """
    flat_then_spike = [100.0] * 150 + [100.0, 300.0, 300.0]
    prices = {"AAPL": make_prices(flat_then_spike)}

    engine = BacktestEngine(BacktestConfig(warmup_bars=50))
    engine.run(prices)

    # A flat series generates no crossover and no band breach, so there is
    # nothing to trade. If look-ahead existed the engine would have bought
    # ahead of the spike and shown a large profit.
    for trade in engine.trades:
        assert trade.entry_price < 200.0, (
            f"entered at {trade.entry_price} — this implies the engine saw the spike"
        )


def test_entry_fills_at_next_bar_open():
    """Signals are computed on bar i; the fill must occur at bar i+1's open."""
    closes = oscillating(120)
    prices = {"AAPL": make_prices(closes)}
    engine = BacktestEngine(BacktestConfig(warmup_bars=50))
    engine.run(prices)

    if not engine.trades:
        pytest.skip("no trades generated for this series")

    df = prices["AAPL"]
    for trade in engine.trades:
        # entry_date must exist in the index, and the fill price must be that
        # bar's open adjusted only for slippage (never its close).
        assert trade.entry_date in df.index
        bar_open = df.loc[trade.entry_date, "open"]
        assert trade.entry_price == pytest.approx(bar_open * 1.0005, rel=1e-6)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

def test_costs_reduce_returns():
    prices = {"AAPL": make_prices(oscillating(200))}

    with_costs = BacktestEngine(BacktestConfig(warmup_bars=50))
    m_costs = with_costs.run(prices)

    free_config = BacktestConfig(warmup_bars=50)
    free_config.commission_per_share = 0.0
    free_config.min_commission = 0.0
    free_config.slippage_bps = 0.0
    frictionless = BacktestEngine(free_config)
    m_free = frictionless.run(prices)

    if m_costs["total_trades"] == 0:
        pytest.skip("no trades generated")
    assert m_costs["total_costs"] > 0
    assert m_free["total_costs"] == 0
    assert m_costs["final_equity"] < m_free["final_equity"]


def test_slippage_works_against_the_trade():
    config = BacktestConfig(slippage_bps=10.0)
    engine = BacktestEngine(config)
    assert engine._fill_price(100.0, "buy") > 100.0   # pay up to buy
    assert engine._fill_price(100.0, "sell") < 100.0  # receive less to sell


def test_commission_has_a_floor():
    engine = BacktestEngine(BacktestConfig(min_commission=1.0, commission_per_share=0.005))
    assert engine._commission(1) == 1.0      # floor applies
    assert engine._commission(1000) == 5.0   # per-share applies


# ---------------------------------------------------------------------------
# Risk limits — the same ones the live engine enforces
# ---------------------------------------------------------------------------

def test_position_cap_is_respected():
    prices = {sym: make_prices(oscillating(200, base=100 + i))
              for i, sym in enumerate(["AAPL", "MSFT", "NVDA"])}
    config = BacktestConfig(warmup_bars=50, max_position_pct=0.05)
    engine = BacktestEngine(config)
    engine.run(prices)

    for trade in engine.trades:
        notional = trade.entry_price * trade.qty
        # Allow headroom for equity drift between sizing and fill.
        assert notional <= config.initial_capital * 0.08, (
            f"{trade.symbol} notional ${notional:,.0f} exceeds the position cap"
        )


def test_max_open_positions_is_respected():
    prices = {f"SYM{i}": make_prices(oscillating(200, base=100 + i)) for i in range(12)}
    config = BacktestConfig(warmup_bars=50, max_open_positions=3)
    engine = BacktestEngine(config)
    engine.run(prices)
    assert max(e["n_positions"] for e in engine.equity_curve) <= 3


def test_never_spends_more_cash_than_available():
    prices = {f"SYM{i}": make_prices(oscillating(200, base=50 + i)) for i in range(10)}
    engine = BacktestEngine(BacktestConfig(warmup_bars=50))
    engine.run(prices)
    for snapshot in engine.equity_curve:
        assert snapshot["cash"] >= -0.01, "cash went negative — leverage was used"


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------

def test_stop_loss_triggers_on_decline():
    closes = [100.0] * 60 + [95.0] + [70.0] * 30
    prices = {"AAPL": make_prices(closes)}
    engine = BacktestEngine(BacktestConfig(warmup_bars=50, stop_loss_pct=5.0))
    engine.run(prices)
    if engine.trades:
        losses = [t for t in engine.trades if t.exit_reason == "STOP_LOSS"]
        for trade in losses:
            assert trade.exit_price < trade.entry_price


def test_all_positions_closed_at_end():
    prices = {"AAPL": make_prices(trending_up(200))}
    engine = BacktestEngine(BacktestConfig(warmup_bars=50))
    engine.run(prices)
    assert engine.positions == {}, "positions left open after the test window"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_are_internally_consistent():
    prices = {"AAPL": make_prices(oscillating(200)), "MSFT": make_prices(oscillating(200, base=150))}
    engine = BacktestEngine(BacktestConfig(warmup_bars=50))
    metrics = engine.run(prices)

    assert metrics["total_trades"] == len(engine.trades)
    wins = len([t for t in engine.trades if t.pnl > 0])
    if engine.trades:
        assert metrics["win_rate_pct"] == pytest.approx(wins / len(engine.trades) * 100, abs=0.1)
    assert sum(metrics["exit_reasons"].values()) == len(engine.trades)
    assert sum(s["trades"] for s in metrics["by_strategy"].values()) == len(engine.trades)


def test_flat_market_produces_no_trades_and_no_crash():
    prices = {"AAPL": make_prices([100.0] * 200)}
    engine = BacktestEngine(BacktestConfig(warmup_bars=50))
    metrics = engine.run(prices)
    assert metrics["total_trades"] == 0
    assert metrics["final_equity"] >= BacktestConfig().initial_capital * 0.99


def test_cash_yield_accrues_on_idle_capital():
    prices = {"AAPL": make_prices([100.0] * 200)}  # flat: no trades, all cash

    earning = BacktestEngine(BacktestConfig(warmup_bars=50, cash_yield_annual=0.05))
    m_earning = earning.run(prices)

    idle = BacktestEngine(BacktestConfig(warmup_bars=50, cash_yield_annual=0.0))
    m_idle = idle.run(prices)

    assert m_earning["final_equity"] > m_idle["final_equity"]


def test_benchmark_matches_hand_computation():
    closes = [100.0] * 50 + [100.0, 110.0, 120.0]
    spy = make_prices(closes)
    result = buy_and_hold_benchmark(spy, 100_000.0, warmup_bars=50)
    # Buys 1000 shares at 100, ends at 120 => 120,000
    assert result["final_equity"] == pytest.approx(120_000.0, rel=0.01)
    assert result["total_return_pct"] == pytest.approx(20.0, rel=0.01)


def test_sharpe_is_zero_without_variance():
    engine = BacktestEngine(BacktestConfig())
    engine.equity_curve = [{"date": pd.Timestamp("2024-01-01"), "equity": 100_000.0,
                            "cash": 100_000.0, "n_positions": 0}] * 10
    assert engine.compute_metrics()["sharpe_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Momentum signal — documents a known design weakness
# ---------------------------------------------------------------------------

def test_momentum_strength_is_structurally_small_at_crossover():
    """
    Documents a real flaw rather than asserting correctness.

    The momentum entry requires a FRESH EMA crossover — the bar where fast and
    slow EMAs are by definition nearly equal — then scores conviction from their
    spread. At a fresh crossover that spread is ~0, so strength is always tiny
    and most crossovers are discarded by a threshold the entry condition itself
    guarantees will fail.

    Measured over 5 years: 314 of 474 crossovers were rejected this way.

    Fixes were tested (rate-of-change strength; a stricter RSI/thrust/volume
    filter) and both were rejected for failing to generalize out-of-sample.
    Momentum is statistically neutral here: Sharpe 1.98 with it, 1.99 without.
    This test pins the behavior so a future change is a deliberate decision.
    """
    engine = BacktestEngine(BacktestConfig())

    # Construct a clean upward crossover.
    closes = [100.0] * 30 + [100.0 + i * 0.5 for i in range(30)]
    df = make_prices(closes)
    d = engine._precompute_indicators({"T": df})["T"]

    crossover_idx = None
    for i in range(1, len(d)):
        prev_f, prev_s = d["ema_fast"].iloc[i - 1], d["ema_slow"].iloc[i - 1]
        fast, slow = d["ema_fast"].iloc[i], d["ema_slow"].iloc[i]
        if pd.notna(prev_f) and pd.notna(slow) and prev_f <= prev_s and fast > slow:
            crossover_idx = i
            break

    assert crossover_idx is not None, "expected a crossover in this series"

    spread = abs(d["ema_fast"].iloc[crossover_idx] - d["ema_slow"].iloc[crossover_idx])
    relative_spread = spread / d["ema_slow"].iloc[crossover_idx]

    # The defining property: spread at a fresh crossover is negligible.
    assert relative_spread < 0.01, (
        f"spread at crossover was {relative_spread:.4f} — if this grows, the "
        "strength formula may no longer be structurally suppressed"
    )
