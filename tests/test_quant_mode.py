"""
BUG 4 regression tests — the deterministic quant fallback.

base_agent.py documents that "each agent also has a _quant_fallback() method", but
no agent overrode it. The single generic implementation returned empty
trade_proposals and final_decisions, and wrote the key "reviewed" where
risk_guardian reads "reviewed_proposals". So the advertised "always works without
API keys" mode ran cycles and updated the dashboard while being structurally
incapable of placing a trade.
"""
import json

import pytest

from core.position_sizer import build_sized_trades

EQUITY = 100_000.0


def test_generic_fallback_uses_the_key_risk_guardian_reads():
    from agents.base_agent import BaseAgent

    class _Probe(BaseAgent):
        async def run(self, context):
            return {}

    payload = json.loads(_Probe("probe", "probe")._quant_fallback({}))
    assert "reviewed_proposals" in payload, "risk_guardian reads reviewed_proposals"


def test_strategy_fallback_generates_momentum_proposals(monkeypatch):
    from agents.strategy import StrategyAgent
    from core import shared_state as shared_state_mod

    monkeypatch.setattr(shared_state_mod.shared_state, "current_regime", "TRENDING")

    context = {
        "quotes": {
            # Above VWAP and up on the day -> momentum BUY
            "AAPL": {"price": 205.0, "change_pct": 2.0, "vwap": 200.0, "high": 206.0, "low": 199.0},
            # Below VWAP, flat -> no signal
            "MSFT": {"price": 395.0, "change_pct": 0.1, "vwap": 400.0, "high": 402.0, "low": 394.0},
        }
    }
    payload = json.loads(StrategyAgent()._quant_fallback(context))
    proposals = payload["trade_proposals"]

    assert len(proposals) == 1
    assert proposals[0]["symbol"] == "AAPL"
    assert proposals[0]["strategy"] == "MOMENTUM"
    assert 0 < proposals[0]["position_size_pct"] <= 5.0
    assert proposals[0]["rank"] == 1


def test_strategy_fallback_generates_mean_reversion_proposals(monkeypatch):
    from agents.strategy import StrategyAgent
    from core import shared_state as shared_state_mod

    monkeypatch.setattr(shared_state_mod.shared_state, "current_regime", "RANGING")

    context = {
        "quotes": {
            # Near the low of the day's range while ranging -> mean reversion BUY
            "XOM": {"price": 100.5, "change_pct": -2.0, "vwap": 103.0, "high": 105.0, "low": 100.0},
        }
    }
    payload = json.loads(StrategyAgent()._quant_fallback(context))
    proposals = payload["trade_proposals"]

    assert len(proposals) == 1
    assert proposals[0]["strategy"] == "MEAN_REVERSION"


def test_strategy_fallback_skips_unpriceable_symbols(monkeypatch):
    from agents.strategy import StrategyAgent
    from core import shared_state as shared_state_mod

    monkeypatch.setattr(shared_state_mod.shared_state, "current_regime", "TRENDING")
    context = {"quotes": {"NVDA": {"price": 0.0, "change_pct": 0.0, "vwap": 0.0}}}
    payload = json.loads(StrategyAgent()._quant_fallback(context))
    assert payload["trade_proposals"] == []


def test_strategy_fallback_handles_empty_context(monkeypatch):
    from agents.strategy import StrategyAgent
    from core import shared_state as shared_state_mod

    monkeypatch.setattr(shared_state_mod.shared_state, "current_regime", "RANGING")
    payload = json.loads(StrategyAgent()._quant_fallback({}))
    assert payload["trade_proposals"] == []
    assert "strategy_assessment" in payload


def test_quant_mode_produces_executable_trades(make_proposal, quotes):
    """
    The end-to-end shape of quant mode: no LLM reviews, no LLM decisions.
    Empty llm_decisions must mean "no opinion", not "reject everything".
    """
    proposals = [make_proposal(symbol="AAPL", pct=2.0), make_proposal(symbol="MSFT", pct=3.0)]
    approved, skipped = build_sized_trades(
        proposals=proposals,
        reviewed=[],        # quant fallback returns no reviews
        equity=EQUITY,
        quotes=quotes,
        regime="RANGING",
        llm_decisions=[],   # quant fallback returns no decisions
    )
    assert len(approved) == 2, "quant mode must be able to trade"
    assert skipped == []
    for trade in approved:
        assert trade["qty"] > 0
        assert trade["sizing"]["source"] == "deterministic"


def test_empty_decisions_is_not_reject_all(make_proposal, quotes):
    """Guards the single easiest thing to get backwards in the merge rule."""
    approved_none, _ = build_sized_trades(
        [make_proposal()], [], EQUITY, quotes, "RANGING", llm_decisions=None
    )
    approved_empty, _ = build_sized_trades(
        [make_proposal()], [], EQUITY, quotes, "RANGING", llm_decisions=[]
    )
    assert len(approved_none) == 1
    assert len(approved_empty) == 1
