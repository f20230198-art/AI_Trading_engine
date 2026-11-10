"""
BUG 2 regression tests — Risk Guardian veto enforcement.

Before the fix, the string "VETOED" appeared nowhere in the execution path. A veto
was only prompt text an LLM could ignore. These tests pin the code-level block.
"""
import asyncio

import pytest

from core.position_sizer import build_sized_trades, enforce_vetoes

EQUITY = 100_000.0


def _sized(symbol="AAPL", conviction=0.9, override=False):
    return {
        "symbol": symbol,
        "action": "BUY",
        "qty": 10,
        "conviction": conviction,
        "override_authorized": override,
    }


def test_vetoed_symbol_never_reaches_approved(make_proposal, make_review, quotes):
    proposals = [make_proposal(symbol="AAPL"), make_proposal(symbol="MSFT")]
    reviewed = [make_review(symbol="AAPL", verdict="VETOED")]
    approved, skipped = build_sized_trades(proposals, reviewed, EQUITY, quotes, "RANGING")
    assert [t["symbol"] for t in approved] == ["MSFT"]
    assert any(s["symbol"] == "AAPL" for s in skipped)


def test_crisis_veto_is_absolute(make_review):
    """Even maximum conviction plus an explicit override cannot pass in CRISIS."""
    trades = [_sized(conviction=0.99, override=True)]
    reviewed = [make_review(verdict="VETOED")]
    kept, blocked = enforce_vetoes(trades, reviewed, "CRISIS")
    assert kept == []
    assert blocked[0]["symbol"] == "AAPL"


def test_override_allowed_outside_crisis(make_review):
    trades = [_sized(conviction=0.9, override=True)]
    reviewed = [make_review(verdict="VETOED")]
    kept, blocked = enforce_vetoes(trades, reviewed, "TRENDING")
    assert len(kept) == 1
    assert blocked == []


def test_override_requires_high_conviction(make_review):
    trades = [_sized(conviction=0.5, override=True)]
    reviewed = [make_review(verdict="VETOED")]
    kept, blocked = enforce_vetoes(trades, reviewed, "TRENDING")
    assert kept == []
    assert len(blocked) == 1


def test_override_defaults_to_block(make_review):
    """The LLM omitting the override field must mean the veto stands."""
    trades = [_sized(conviction=0.99, override=False)]
    reviewed = [make_review(verdict="VETOED")]
    kept, blocked = enforce_vetoes(trades, reviewed, "TRENDING")
    assert kept == []
    assert len(blocked) == 1


def test_enforce_vetoes_is_a_last_line_of_defense(make_review):
    """A trade injected past sizing still cannot survive a veto."""
    trades = [_sized(symbol="AAPL"), _sized(symbol="MSFT")]
    reviewed = [make_review(symbol="AAPL", verdict="VETOED")]
    kept, blocked = enforce_vetoes(trades, reviewed, "RANGING")
    assert [t["symbol"] for t in kept] == ["MSFT"]
    assert [b["symbol"] for b in blocked] == ["AAPL"]


# ---------------------------------------------------------------------------
# Risk Guardian hard rules, exercised through run() with the LLM patched out.
# ---------------------------------------------------------------------------

CANNED_APPROVAL = """{
  "reviewed_proposals": [
    {"symbol": "AAPL", "verdict": "APPROVED", "adjusted_size_pct": 3.0, "reasoning": "ok"},
    {"symbol": "MSFT", "verdict": "APPROVED", "adjusted_size_pct": 3.0, "reasoning": "ok"}
  ],
  "portfolio_risk_score": 30,
  "risk_label": "LOW",
  "portfolio_risk_assessment": "fine",
  "active_risk_flags": [],
  "recommended_hedges": []
}"""


def _run_guardian(monkeypatch, proposals, daily_drawdown=0.0, regime="RANGING",
                  llm_response=CANNED_APPROVAL, positions=None):
    """Drive RiskGuardianAgent.run() offline — no HTTP client, no sqlite, no memory."""
    from agents.base_agent import BaseAgent
    from agents.risk_guardian import RiskGuardianAgent
    from core import shared_state as shared_state_mod
    from core.performance_tracker import performance_tracker

    async def fake_llm(self, sp, up, max_tokens=1500, context=None):
        return llm_response

    async def noop(*args, **kwargs):
        return None

    async def noop_str(*args, **kwargs):
        return ""

    monkeypatch.setattr(RiskGuardianAgent, "_call_llm", fake_llm)
    monkeypatch.setattr(BaseAgent, "_recall_memories", noop_str)
    monkeypatch.setattr(BaseAgent, "_store_decision", noop_str)
    monkeypatch.setattr(BaseAgent, "update_status", noop)
    monkeypatch.setattr(shared_state_mod.shared_state, "update_risk", noop)
    monkeypatch.setattr(shared_state_mod.shared_state, "add_agent_message", noop)
    monkeypatch.setattr(shared_state_mod.shared_state, "current_regime", regime)
    monkeypatch.setattr(
        performance_tracker, "get_dashboard_metrics",
        lambda: {"total_pnl": 0.0, "max_drawdown": 12.0},
    )
    monkeypatch.setattr(
        performance_tracker, "compute_daily_drawdown", lambda now=None: daily_drawdown
    )

    agent = RiskGuardianAgent()  # no initialize() — _call_llm is patched, so no client needed
    return asyncio.run(agent.run({
        "proposals": proposals,
        "equity": EQUITY,
        "positions": positions or [],
    }))


def test_drawdown_breach_vetoes_every_trade(monkeypatch, make_proposal):
    """The 3% daily limit is enforced in code, not left to the LLM."""
    proposals = [make_proposal(symbol="AAPL"), make_proposal(symbol="MSFT")]
    result = _run_guardian(monkeypatch, proposals, daily_drawdown=3.5)
    assert result["reviewed"], "expected review entries"
    assert all(r["verdict"] == "VETOED" for r in result["reviewed"])
    assert result["veto_all"] is True


def test_below_drawdown_limit_allows_trades(monkeypatch, make_proposal):
    proposals = [make_proposal(symbol="AAPL")]
    result = _run_guardian(monkeypatch, proposals, daily_drawdown=1.0)
    assert all(r["verdict"] != "VETOED" for r in result["reviewed"])


def test_quant_fallback_synthesizes_missing_reviews(monkeypatch, make_proposal):
    """BUG 4: with no LLM reviews, every proposal still gets a verdict."""
    proposals = [make_proposal(symbol="AAPL"), make_proposal(symbol="MSFT")]
    empty = '{"reviewed_proposals": [], "portfolio_risk_score": 30, "risk_label": "LOW"}'
    result = _run_guardian(monkeypatch, proposals, llm_response=empty)
    assert len(result["reviewed"]) == 2
    assert {r["symbol"] for r in result["reviewed"]} == {"AAPL", "MSFT"}
    assert all(r["verdict"] == "APPROVED" for r in result["reviewed"])


def test_position_cap_is_clamped_in_code(monkeypatch, make_proposal):
    """An LLM approving 30% is overruled down to the 5% cap."""
    oversized = """{
      "reviewed_proposals": [
        {"symbol": "AAPL", "verdict": "APPROVED", "adjusted_size_pct": 30.0, "reasoning": "yolo"}
      ],
      "portfolio_risk_score": 30, "risk_label": "LOW"
    }"""
    result = _run_guardian(monkeypatch, [make_proposal()], llm_response=oversized)
    entry = result["reviewed"][0]
    assert entry["adjusted_size_pct"] <= 5.0
    assert entry["verdict"] == "RESIZED"


def test_sector_cap_resizes_to_headroom(monkeypatch, make_proposal):
    """Technology already at 18% leaves only 2% of the 20% sector cap."""
    positions = [{"symbol": "AAPL", "market_value": 18_000.0}]
    result = _run_guardian(
        monkeypatch, [make_proposal(symbol="MSFT", pct=5.0)], positions=positions
    )
    entry = result["reviewed"][0]
    assert entry["adjusted_size_pct"] == pytest.approx(2.0)
    assert entry["verdict"] == "RESIZED"


def test_crisis_halves_position_size(monkeypatch, make_proposal):
    result = _run_guardian(monkeypatch, [make_proposal(pct=4.0)], regime="CRISIS")
    entry = result["reviewed"][0]
    assert entry["adjusted_size_pct"] == pytest.approx(1.5)  # LLM's 3.0, halved
    assert entry["verdict"] == "RESIZED"
