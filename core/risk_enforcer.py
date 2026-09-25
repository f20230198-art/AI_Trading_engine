"""
DOOM ENGINE — Deterministic Risk Enforcement

The final gate between the Orchestrator and the Execution agent. Runs after the
Risk Guardian has responded and the Orchestrator has decided, and re-checks every
trade against the hard limits in config.py in plain Python:

    - Risk Guardian veto       -> REJECT (absolute; no orchestrator override)
    - daily drawdown breached  -> REJECT every new BUY
    - max_position_pct         -> CLAMP qty down, or REJECT if no room left
    - max_sector_pct           -> CLAMP qty down, or REJECT if no room left

Fails closed: a BUY with no usable equity or price is rejected. SELLs reduce
exposure, so only a veto can stop them.

Every rejection and clamp is written to the SQLite audit trail (conflicts.db).
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from config import TRADING_CONFIG
from core.position_sizer import sector_of

logger = logging.getLogger("doom.risk_enforcer")

REJECTED = "REJECTED"
CLAMPED = "CLAMPED"


def _position_value(position: Dict[str, Any]) -> float:
    """Absolute market value of a broker position."""
    try:
        value = float(position.get("market_value") or 0)
        if not value:
            value = float(position.get("qty") or 0) * float(position.get("current_price") or 0)
    except (TypeError, ValueError):
        return 0.0
    return abs(value)


def _trade_price(trade: Dict[str, Any]) -> float:
    try:
        return float((trade.get("sizing") or {}).get("price") or trade.get("price") or 0)
    except (TypeError, ValueError):
        return 0.0


def enforce_risk_limits(
    trades: List[Dict[str, Any]],
    reviewed: Optional[List[Dict[str, Any]]],
    equity: Optional[float],
    positions: Optional[List[Dict[str, Any]]] = None,
    daily_drawdown_pct: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pure, no I/O. `daily_drawdown_pct` is in percent (3.0 == 3%).

    Returns (allowed_trades, events). Each event is
    {"symbol", "action", "verdict": REJECTED|CLAMPED, "reason", "original_qty", "final_qty"}.
    """
    vetoed = {
        r.get("symbol") for r in (reviewed or [])
        if str(r.get("verdict", "")).upper() == "VETOED" and r.get("symbol")
    }
    try:
        equity = float(equity or 0)
    except (TypeError, ValueError):
        equity = 0.0

    position_cap = TRADING_CONFIG.max_position_pct * equity
    sector_cap = TRADING_CONFIG.max_sector_pct * equity
    drawdown_limit = TRADING_CONFIG.max_daily_drawdown_pct * 100
    drawdown_breached = float(daily_drawdown_pct or 0) >= drawdown_limit

    symbol_exposure: Dict[str, float] = {}
    sector_exposure: Dict[str, float] = {}
    for position in positions or []:
        symbol = position.get("symbol")
        if not symbol:
            continue
        value = _position_value(position)
        symbol_exposure[symbol] = symbol_exposure.get(symbol, 0.0) + value
        sector = sector_of(symbol)
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + value

    allowed: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []

    def _event(trade, verdict, reason, final_qty):
        events.append({
            "symbol": trade.get("symbol"),
            "action": str(trade.get("action", "")).upper(),
            "verdict": verdict,
            "reason": reason,
            "original_qty": trade.get("qty"),
            "final_qty": final_qty,
        })
        logger.warning("RiskEnforcer %s %s: %s", verdict, trade.get("symbol"), reason)

    for trade in trades or []:
        symbol = trade.get("symbol")
        action = str(trade.get("action", "")).upper()

        if symbol in vetoed:
            _event(trade, REJECTED, "Risk Guardian VETOED this symbol; veto cannot be overridden", 0)
            continue

        if action != "BUY":
            allowed.append(trade)
            continue

        if drawdown_breached:
            _event(trade, REJECTED,
                   f"Daily drawdown {daily_drawdown_pct:.2f}% >= {drawdown_limit:.2f}% limit", 0)
            continue

        price = _trade_price(trade)
        try:
            qty = int(trade.get("qty") or 0)
        except (TypeError, ValueError):
            qty = 0
        if equity <= 0 or price <= 0 or qty <= 0:
            _event(trade, REJECTED,
                   f"Cannot verify limits (equity={equity}, price={price}, qty={qty})", 0)
            continue

        sector = sector_of(symbol)
        room = min(
            position_cap - symbol_exposure.get(symbol, 0.0),
            sector_cap - sector_exposure.get(sector, 0.0),
        )
        max_qty = max(0, math.floor(room / price + 1e-9))

        if max_qty <= 0:
            limit = ("position" if position_cap - symbol_exposure.get(symbol, 0.0)
                     <= sector_cap - sector_exposure.get(sector, 0.0) else f"sector '{sector}'")
            _event(trade, REJECTED, f"No room under {limit} limit", 0)
            continue

        if qty > max_qty:
            limit = ("position" if position_cap - symbol_exposure.get(symbol, 0.0)
                     <= sector_cap - sector_exposure.get(sector, 0.0) else f"sector '{sector}'")
            _event(trade, CLAMPED,
                   f"qty {qty} -> {max_qty} to respect {limit} limit", max_qty)
            trade = {**trade, "qty": max_qty}
            if trade.get("sizing"):
                trade["sizing"] = {**trade["sizing"], "notional": round(max_qty * price, 2)}
            qty = max_qty

        notional = qty * price
        symbol_exposure[symbol] = symbol_exposure.get(symbol, 0.0) + notional
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + notional
        allowed.append(trade)

    return allowed, events


async def apply_risk_enforcement(
    trades: List[Dict[str, Any]],
    reviewed: Optional[List[Dict[str, Any]]],
    equity: Optional[float],
    positions: Optional[List[Dict[str, Any]]] = None,
    daily_drawdown_pct: float = 0.0,
    audit_logger=None,
) -> List[Dict[str, Any]]:
    """Run enforce_risk_limits and write every event to the SQLite audit trail."""
    allowed, events = enforce_risk_limits(trades, reviewed, equity, positions, daily_drawdown_pct)
    if audit_logger is None:
        from core.conflict_logger import conflict_logger as audit_logger
    for e in events:
        try:
            await audit_logger.log_conflict(
                agent_a_name="orchestrator",
                agent_a_position=f"{e['action']} {e['original_qty']}",
                agent_a_reasoning="Trade approved by orchestrator",
                agent_b_name="risk_enforcer",
                agent_b_position=f"{e['verdict']} (qty {e['final_qty']})",
                agent_b_reasoning=e["reason"],
                symbol=e["symbol"],
                verdict=e["verdict"],
                verdict_reasoning=e["reason"],
                resolved_by="risk_enforcer",
            )
        except Exception as exc:  # audit failure must never unblock a trade
            logger.error("RiskEnforcer: audit log write failed: %s", exc)
    return allowed
