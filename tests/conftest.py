"""
Shared test fixtures for the DOOM ENGINE test suite.

Every test here is offline: no network, no sqlite, no agent initialize().
Tests drive pure functions directly, or patch _call_llm when an agent's
run() must be exercised.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def quotes():
    """Quote snapshots covering the realistic shapes the sizer must handle."""
    return {
        "AAPL": {"price": 200.0, "prev_close": 198.0, "high": 202.0, "low": 197.0, "vwap": 199.0},
        "MSFT": {"price": 400.0, "prev_close": 399.0, "high": 405.0, "low": 398.0, "vwap": 401.0},
        # After-hours: Alpaca reports latestTrade.p == 0 but prev_close is usable.
        "XOM": {"price": 0.0, "prev_close": 110.0, "high": 0.0, "low": 0.0, "vwap": 0.0},
        # Genuinely unpriceable — must be skipped, never sized.
        "NVDA": {"price": 0.0, "prev_close": 0.0, "high": 0.0, "low": 0.0, "vwap": 0.0},
    }


@pytest.fixture
def make_proposal():
    def _make(symbol="AAPL", pct=2.0, action="BUY", conviction=0.7, strategy="MOMENTUM"):
        return {
            "symbol": symbol,
            "action": action,
            "strategy": strategy,
            "conviction": conviction,
            "position_size_pct": pct,
            "entry_type": "MARKET",
            "stop_loss_pct": 2.0,
            "reasoning": "test proposal",
        }
    return _make


@pytest.fixture
def make_review():
    def _make(symbol="AAPL", verdict="APPROVED", adjusted=None):
        return {
            "symbol": symbol,
            "verdict": verdict,
            "adjusted_size_pct": adjusted,
            "reasoning": "test review",
            "risk_flags": [],
        }
    return _make
