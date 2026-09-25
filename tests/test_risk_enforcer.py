"""
Deterministic risk enforcement — the plain-Python gate between the Orchestrator
and Execution. Limits come from config.py (5% position, 20% sector, 3% daily DD).
"""
import asyncio

from core.risk_enforcer import CLAMPED, REJECTED, apply_risk_enforcement, enforce_risk_limits

EQUITY = 100_000.0


def _trade(symbol="AAPL", qty=10, price=200.0, action="BUY", override=False, conviction=0.9):
    return {
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "conviction": conviction,
        "override_authorized": override,
        "sizing": {"price": price, "equity": EQUITY, "notional": qty * price},
    }


def test_valid_trade_passes_unchanged():
    trade = _trade(qty=20, price=200.0)  # $4k = 4% < 5%
    allowed, events = enforce_risk_limits([trade], [], EQUITY)
    assert allowed == [trade]
    assert events == []


def test_over_position_limit_is_clamped():
    allowed, events = enforce_risk_limits([_trade(qty=100, price=200.0)], [], EQUITY)  # $20k
    assert allowed[0]["qty"] == 25  # $5k / $200
    assert allowed[0]["sizing"]["notional"] == 5_000.0
    assert events[0]["verdict"] == CLAMPED
    assert "position" in events[0]["reason"]


def test_existing_position_at_limit_is_rejected():
    positions = [{"symbol": "AAPL", "market_value": 5_000.0}]
    allowed, events = enforce_risk_limits([_trade(qty=5)], [], EQUITY, positions)
    assert allowed == []
    assert events[0]["verdict"] == REJECTED


def test_over_sector_limit_is_clamped():
    # $18k already in technology; only $2k of the 20% sector cap left.
    positions = [
        {"symbol": "MSFT", "market_value": 5_000.0},
        {"symbol": "NVDA", "market_value": 5_000.0},
        {"symbol": "GOOGL", "market_value": 4_000.0},
        {"symbol": "META", "market_value": 4_000.0},
    ]
    allowed, events = enforce_risk_limits([_trade(qty=20, price=200.0)], [], EQUITY, positions)
    assert allowed[0]["qty"] == 10
    assert events[0]["verdict"] == CLAMPED
    assert "sector" in events[0]["reason"]


def test_sector_limit_counts_trades_in_same_batch():
    trades = [_trade(s, qty=25, price=200.0) for s in ("AAPL", "MSFT", "NVDA", "GOOGL", "META")]
    allowed, events = enforce_risk_limits(trades, [], EQUITY)
    assert [t["symbol"] for t in allowed] == ["AAPL", "MSFT", "NVDA", "GOOGL"]  # 4 x $5k = $20k
    assert events[0]["symbol"] == "META" and events[0]["verdict"] == REJECTED


def test_drawdown_breached_rejects_all_buys():
    trades = [_trade("AAPL", qty=1), _trade("XOM", qty=1, price=110.0)]
    allowed, events = enforce_risk_limits(trades, [], EQUITY, daily_drawdown_pct=3.0)
    assert allowed == []
    assert {e["verdict"] for e in events} == {REJECTED}
    assert "drawdown" in events[0]["reason"].lower()


def test_drawdown_breached_still_allows_sells():
    sell = _trade(action="SELL", qty=5)
    allowed, _ = enforce_risk_limits([sell], [], EQUITY, daily_drawdown_pct=10.0)
    assert allowed == [sell]


def test_veto_cannot_be_overridden_by_orchestrator():
    """Max conviction plus an explicit override flag, outside CRISIS — still blocked."""
    trade = _trade(qty=1, override=True, conviction=0.99)
    reviewed = [{"symbol": "AAPL", "verdict": "VETOED"}]
    allowed, events = enforce_risk_limits([trade], reviewed, EQUITY)
    assert allowed == []
    assert events[0]["verdict"] == REJECTED
    assert "veto" in events[0]["reason"].lower()


def test_llm_approval_does_not_bypass_limits():
    reviewed = [{"symbol": "AAPL", "verdict": "APPROVED"}]
    allowed, events = enforce_risk_limits([_trade(qty=1000)], reviewed, EQUITY)
    assert allowed[0]["qty"] == 25
    assert events[0]["verdict"] == CLAMPED


def test_missing_equity_fails_closed():
    allowed, events = enforce_risk_limits([_trade()], [], None)
    assert allowed == []
    assert events[0]["verdict"] == REJECTED


def test_every_event_is_written_to_audit_trail():
    class _FakeAudit:
        def __init__(self):
            self.rows = []

        async def log_conflict(self, **kwargs):
            self.rows.append(kwargs)

    audit = _FakeAudit()
    trades = [_trade("AAPL", qty=100), _trade("MSFT", qty=1, price=400.0)]
    reviewed = [{"symbol": "MSFT", "verdict": "VETOED"}]
    allowed = asyncio.run(apply_risk_enforcement(trades, reviewed, EQUITY, audit_logger=audit))

    assert [t["symbol"] for t in allowed] == ["AAPL"]
    assert {(r["symbol"], r["verdict"]) for r in audit.rows} == {("AAPL", CLAMPED), ("MSFT", REJECTED)}
    assert all(r["resolved_by"] == "risk_enforcer" and r["verdict_reasoning"] for r in audit.rows)
