"""
BUG 3 regression tests — daily vs all-time drawdown.

Before the fix, risk_guardian read metrics["max_drawdown"] (computed across the
whole equity curve) and labelled it current_drawdown_pct, feeding the "3% max
daily drawdown" rule. All-time drawdown only ever grows, so once tripped the rule
latched on permanently and never reset the next day.
"""
from datetime import datetime, timedelta

import pytest

from config import TRADING_CONFIG
from core.performance_tracker import PerformanceTracker

NOW = datetime(2026, 9, 20, 15, 0, 0)
TODAY = NOW.date()
YESTERDAY = TODAY - timedelta(days=1)


def _tracker(snapshots):
    """A tracker with a hand-built equity curve — no sqlite, no initialize()."""
    tracker = PerformanceTracker()
    tracker._equity_curve = [
        {"timestamp": ts.isoformat(), "equity": eq, "cash": 0, "positions_value": 0}
        for ts, eq in snapshots
    ]
    return tracker


def _at(day, hour):
    return datetime(day.year, day.month, day.day, hour, 0, 0)


def test_no_snapshots_today_returns_zero():
    tracker = _tracker([(_at(YESTERDAY, 10), 100_000), (_at(YESTERDAY, 16), 95_000)])
    assert tracker.compute_daily_drawdown(now=NOW) == 0.0


def test_intraday_peak_to_current():
    tracker = _tracker([
        (_at(TODAY, 10), 100_000),
        (_at(TODAY, 12), 103_000),
        (_at(TODAY, 14), 100_000),
    ])
    # (103k - 100k) / 103k = 2.91%
    assert tracker.compute_daily_drawdown(now=NOW) == pytest.approx(2.91, abs=0.01)


def test_recovery_reduces_drawdown():
    """The regression that proves it is no longer monotonic."""
    at_trough = _tracker([(_at(TODAY, 10), 100_000), (_at(TODAY, 12), 97_000)])
    recovered = _tracker([
        (_at(TODAY, 10), 100_000),
        (_at(TODAY, 12), 97_000),
        (_at(TODAY, 14), 99_000),
    ])
    trough_dd = at_trough.compute_daily_drawdown(now=NOW)
    recovered_dd = recovered.compute_daily_drawdown(now=NOW)
    assert trough_dd == pytest.approx(3.0, abs=0.01)
    assert recovered_dd == pytest.approx(1.0, abs=0.01)
    assert recovered_dd < trough_dd


def test_ignores_prior_day_trough():
    """
    The exact bug-3 latch: a 20% crash yesterday, flat today.
    Daily drawdown must be 0 while all-time max drawdown stays at 20%.
    """
    tracker = _tracker([
        (_at(YESTERDAY, 10), 100_000),
        (_at(YESTERDAY, 14), 80_000),
        (_at(YESTERDAY, 16), 100_000),
        (_at(TODAY, 10), 100_000),
        (_at(TODAY, 14), 100_000),
    ])
    assert tracker.compute_daily_drawdown(now=NOW) == 0.0
    assert tracker.compute_max_drawdown() == pytest.approx(20.0, abs=0.01)


def test_seeds_peak_from_prior_close():
    """An overnight gap down is visible from the day's first snapshot."""
    tracker = _tracker([(_at(YESTERDAY, 16), 105_000), (_at(TODAY, 10), 102_000)])
    # (105k - 102k) / 105k = 2.86%
    assert tracker.compute_daily_drawdown(now=NOW) == pytest.approx(2.86, abs=0.01)


def test_breach_crosses_configured_limit():
    tracker = _tracker([(_at(TODAY, 10), 100_000), (_at(TODAY, 14), 96_500)])
    drawdown = tracker.compute_daily_drawdown(now=NOW)
    assert drawdown >= TRADING_CONFIG.max_daily_drawdown_pct * 100


def test_gains_are_not_negative_drawdown():
    tracker = _tracker([(_at(TODAY, 10), 100_000), (_at(TODAY, 14), 105_000)])
    assert tracker.compute_daily_drawdown(now=NOW) == 0.0


def test_malformed_snapshots_are_ignored():
    tracker = PerformanceTracker()
    tracker._equity_curve = [
        {"timestamp": None, "equity": 100_000},
        {"timestamp": "not-a-date", "equity": 100_000},
        {"equity": 100_000},
        {"timestamp": _at(TODAY, 10).isoformat(), "equity": 100_000},
        {"timestamp": _at(TODAY, 14).isoformat(), "equity": 98_000},
    ]
    assert tracker.compute_daily_drawdown(now=NOW) == pytest.approx(2.0, abs=0.01)


def test_empty_curve_returns_zero():
    assert PerformanceTracker().compute_daily_drawdown(now=NOW) == 0.0


def test_dashboard_exposes_both_metrics():
    tracker = _tracker([(_at(TODAY, 10), 100_000), (_at(TODAY, 14), 98_000)])
    metrics = tracker.get_dashboard_metrics()
    assert "daily_drawdown" in metrics
    assert "max_drawdown" in metrics  # kept so the frontend keeps working
