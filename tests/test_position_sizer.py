"""
BUG 1 regression tests — position sizing.

Before the fix, orchestrator.py read `qty` off the LLM response and defaulted to
1 share when it was missing, turning a 5% conviction position on a $100k book into
a ~$180 order. These tests pin the deterministic sizing that replaced it.
"""
import pytest

from config import TRADING_CONFIG
from core.position_sizer import (
    SKIP_NO_EQUITY,
    SKIP_NO_PRICE,
    SKIP_SIZE_TOO_SMALL,
    build_sized_trades,
    compute_qty,
    resolve_price,
    size_proposal,
)

EQUITY = 100_000.0


def test_qty_from_pct_and_price():
    # 2% of $100k = $2,000 at $200 => 10 shares
    assert compute_qty(EQUITY, 2.0, 200.0) == 10


def test_qty_floors_never_rounds_up():
    # 2% of $100k = $2,000 at $300 => 6.67 shares. Rounding up would breach the cap.
    assert compute_qty(EQUITY, 2.0, 300.0) == 6


def test_respects_max_position_pct():
    # A 20% request is clamped to the configured 5% cap => $5,000 / $200 = 25.
    assert compute_qty(EQUITY, 20.0, 200.0) == 25
    assert TRADING_CONFIG.max_position_pct == 0.05


def test_zero_and_negative_inputs_yield_zero():
    assert compute_qty(0, 2.0, 200.0) == 0
    assert compute_qty(EQUITY, 2.0, 0) == 0
    assert compute_qty(EQUITY, 0, 200.0) == 0
    assert compute_qty(-1, 2.0, 200.0) == 0


def test_uses_adjusted_size_pct_when_resized(make_proposal, make_review, quotes):
    proposal = make_proposal(pct=4.0)
    review = make_review(verdict="RESIZED", adjusted=1.0)
    trade, reason = size_proposal(proposal, review, EQUITY, quotes, "RANGING")
    assert reason is None
    # 1% of 100k at $200 => 5 shares, not the 20 the proposal asked for.
    assert trade["qty"] == 5


def test_missing_price_skips_rather_than_sizing_one_share(make_proposal, quotes):
    """The core of bug 1: an unpriceable symbol must never become a 1-share order."""
    proposal = make_proposal(symbol="NVDA")
    trade, reason = size_proposal(proposal, None, EQUITY, quotes, "RANGING")
    assert trade is None
    assert reason == SKIP_NO_PRICE


def test_unknown_equity_skips(make_proposal, quotes):
    proposal = make_proposal()
    trade, reason = size_proposal(proposal, None, None, quotes, "RANGING")
    assert trade is None
    assert reason == SKIP_NO_EQUITY


def test_price_too_high_for_slice_skips(make_proposal):
    """qty would floor to 0 — skip, don't round up to 1."""
    proposal = make_proposal(pct=1.0)
    small_quotes = {"AAPL": {"price": 500.0}}
    trade, reason = size_proposal(proposal, None, 10_000.0, small_quotes, "RANGING")
    assert trade is None
    assert reason == SKIP_SIZE_TOO_SMALL


def test_prev_close_fallback_for_after_hours(make_proposal, quotes):
    """Alpaca reports price 0 outside hours; prev_close keeps sizing sane."""
    assert resolve_price("XOM", quotes) == 110.0
    proposal = make_proposal(symbol="XOM", pct=2.0)
    trade, reason = size_proposal(proposal, None, EQUITY, quotes, "RANGING")
    assert reason is None
    assert trade["qty"] == 18  # floor(2000 / 110)


def test_resolve_price_falls_back_to_live_position():
    positions = [{"symbol": "GOOGL", "current_price": 150.0}]
    assert resolve_price("GOOGL", {}, positions) == 150.0
    assert resolve_price("MISSING", {}, positions) is None


def test_crisis_halves_when_unreviewed(make_proposal, quotes):
    proposal = make_proposal(pct=4.0)
    trade, _ = size_proposal(proposal, None, EQUITY, quotes, "CRISIS")
    # 4% halved to 2% => $2,000 / $200 = 10 shares
    assert trade["qty"] == 10


def test_crisis_does_not_double_halve_when_reviewed(make_proposal, make_review, quotes):
    """Risk Guardian owns the regime haircut; the sizer must not reapply it."""
    proposal = make_proposal(pct=4.0)
    review = make_review(verdict="RESIZED", adjusted=2.0)  # Guardian already halved
    trade, _ = size_proposal(proposal, review, EQUITY, quotes, "CRISIS")
    assert trade["qty"] == 10  # 2%, not 1%


def test_llm_qty_is_ignored(make_proposal, quotes):
    """Even when the LLM asserts a quantity, sizing is computed in code."""
    proposals = [make_proposal(pct=2.0)]
    decisions = [{"symbol": "AAPL", "decision": "EXECUTE", "action": "BUY", "qty": 999}]
    approved, _ = build_sized_trades(
        proposals, [], EQUITY, quotes, "RANGING", llm_decisions=decisions
    )
    assert len(approved) == 1
    assert approved[0]["qty"] == 10


def test_llm_can_still_reject(make_proposal, quotes):
    proposals = [make_proposal()]
    decisions = [{"symbol": "AAPL", "decision": "REJECT"}]
    approved, skipped = build_sized_trades(
        proposals, [], EQUITY, quotes, "RANGING", llm_decisions=decisions
    )
    assert approved == []
    assert skipped[0]["symbol"] == "AAPL"


def test_sizing_metadata_is_recorded(make_proposal, quotes):
    proposal = make_proposal(pct=2.0)
    trade, _ = size_proposal(proposal, None, EQUITY, quotes, "RANGING")
    assert trade["sizing"]["source"] == "deterministic"
    assert trade["sizing"]["notional"] == pytest.approx(2000.0)


def test_no_approved_trade_is_ever_one_share_by_default(make_proposal, quotes):
    """Whole-list guard: nothing slips through carrying the old default."""
    proposals = [
        make_proposal(symbol="AAPL", pct=2.0),
        make_proposal(symbol="NVDA", pct=2.0),  # unpriceable
        make_proposal(symbol="MSFT", pct=3.0),
    ]
    approved, skipped = build_sized_trades(proposals, [], EQUITY, quotes, "RANGING")
    assert {s["symbol"] for s in skipped} == {"NVDA"}
    for trade in approved:
        assert trade["qty"] > 1, f"{trade['symbol']} sized to {trade['qty']}"
