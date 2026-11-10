"""
DOOM ENGINE — Deterministic Position Sizing and Veto Enforcement

Turns a strategy proposal's `position_size_pct` into an actual share count, in code.
Previously the Orchestrator read `qty` straight off the LLM response and defaulted to
1 share when it was missing, which silently reduced a 5% conviction position on a
$100k book to a ~$180 trade and made the whole risk framework a no-op.

Everything here is pure: no network, no I/O, no LLM. That is deliberate — the same
logic has to run identically in LLM mode and in the deterministic quant fallback,
and it has to be testable offline.

Sizing order of operations (each step only ever tightens):
    veto check -> size source (review override > proposal) -> regime haircut
    -> max_position_pct clamp -> share math -> min-size check
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from config import SECTOR_MAP, TRADING_CONFIG

logger = logging.getLogger("doom.sizer")

# Skip reasons — surfaced to the Orchestrator and logged, never silently swallowed.
SKIP_NO_PRICE = "NO_PRICE"
SKIP_NO_EQUITY = "NO_EQUITY"
SKIP_SIZE_TOO_SMALL = "SIZE_TOO_SMALL"
SKIP_VETOED = "VETOED"
SKIP_LLM_REJECTED = "LLM_REJECTED"

# Regime haircuts, matching the language in base_agent._get_regime_context().
_REGIME_HAIRCUT = {
    "CRISIS": 0.5,
    "HIGH_VOLATILITY": 0.75,
}


def sector_of(symbol: str) -> str:
    """Map a symbol to its configured sector, or 'other'."""
    for sector, symbols in SECTOR_MAP.items():
        if symbol in symbols:
            return sector
    return "other"


def resolve_price(
    symbol: str,
    quotes: Optional[Dict[str, Dict]],
    positions: Optional[List[Dict]] = None,
) -> Optional[float]:
    """
    Best available price for sizing, using only data already in the cycle context.

    Returns None when no usable price exists — callers must skip, not guess.

    Alpaca's `latestTrade.p` is legitimately 0 outside regular hours, so every
    candidate is validated as > 0 rather than merely present. `prev_close` is the
    last resort so an after-hours cycle still sizes sanely instead of skipping
    every symbol.

    Deliberately makes no network call: market_intel already fetched the whole
    watchlist in Phase 1, and this runs per-symbol inside the Orchestrator.
    """
    quote = (quotes or {}).get(symbol) or {}
    for key in ("price", "last", "ask", "prev_close"):
        try:
            value = float(quote.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value

    # Fall back to a live broker position, which carries its own current_price.
    for position in positions or []:
        if position.get("symbol") != symbol:
            continue
        for key in ("current_price", "avg_entry_price"):
            try:
                value = float(position.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value

    return None


def compute_qty(equity: float, size_pct: float, price: float) -> int:
    """
    Whole shares for `size_pct` of `equity` at `price`, clamped to max_position_pct.

    Floors rather than rounds: rounding up can breach the position cap.
    Returns 0 when the notional cannot cover a single share — execution submits
    `str(int(qty))` and does not support fractional shares.
    """
    if equity is None or price is None:
        return 0
    if equity <= 0 or price <= 0 or size_pct <= 0:
        return 0

    capped_pct = min(float(size_pct), TRADING_CONFIG.max_position_pct * 100)
    notional = equity * capped_pct / 100.0
    return int(math.floor(notional / price))


def size_proposal(
    proposal: Dict[str, Any],
    review: Optional[Dict[str, Any]],
    equity: Optional[float],
    quotes: Optional[Dict[str, Dict]],
    regime: str,
    positions: Optional[List[Dict]] = None,
    llm_decision: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Size a single proposal into an executable trade.

    Returns (trade, None) on success or (None, skip_reason) when the trade cannot
    be sized within the configured limits.

    On any missing input we skip and log rather than defaulting to a nominal size:
    with an unknown price max_position_pct cannot be honoured at all, so no order
    can be proven to be within limits. A 1-share consolation order would still
    incur slippage and would pollute per-agent P&L attribution.
    """
    symbol = proposal.get("symbol", "")
    review = review or {}

    # 1. Risk Guardian veto wins outright. Override handling lives in enforce_vetoes().
    if review.get("verdict") == "VETOED":
        return None, SKIP_VETOED

    # 2. Size source: a RESIZED review overrides the proposal's own request.
    size_pct = None
    if review.get("verdict") == "RESIZED":
        try:
            adjusted = review.get("adjusted_size_pct")
            if adjusted is not None and float(adjusted) > 0:
                size_pct = float(adjusted)
        except (TypeError, ValueError):
            size_pct = None
    if size_pct is None:
        try:
            size_pct = float(proposal.get("position_size_pct") or 0)
        except (TypeError, ValueError):
            size_pct = 0.0
    if size_pct <= 0:
        size_pct = 1.0  # conservative floor when the proposal omitted a size

    # 3. Regime haircut — applied here ONLY when Risk Guardian did not review this
    #    symbol. When a review exists the Guardian has already priced the regime in,
    #    and applying it twice would halve a CRISIS position to a quarter.
    if not review:
        size_pct *= _REGIME_HAIRCUT.get(regime, 1.0)

    # 4. Inputs required for share math.
    if equity is None or equity <= 0:
        return None, SKIP_NO_EQUITY

    price = resolve_price(symbol, quotes, positions)
    if price is None:
        return None, SKIP_NO_PRICE

    # 5. Share math, with the position cap applied inside compute_qty().
    qty = compute_qty(equity, size_pct, price)
    if qty <= 0:
        return None, SKIP_SIZE_TOO_SMALL

    decision = llm_decision or {}
    capped_pct = min(size_pct, TRADING_CONFIG.max_position_pct * 100)

    trade = {
        "symbol": symbol,
        "action": decision.get("action") or proposal.get("action", "BUY"),
        "qty": qty,
        "entry_type": decision.get("order_type") or proposal.get("entry_type", "MARKET"),
        "limit_price": decision.get("limit_price") or proposal.get("limit_price"),
        "stop_loss_pct": decision.get("stop_loss_pct") or proposal.get("stop_loss_pct"),
        "originating_agent": decision.get("originating_agent")
        or proposal.get("originating_agent", "strategy"),
        "strategy": decision.get("strategy") or proposal.get("strategy", "unknown"),
        "conviction": decision.get("conviction", proposal.get("conviction", 0)),
        # An override is only honoured when the LLM asked for it explicitly.
        "override_authorized": bool(decision.get("override_risk_veto", False)),
        "sizing": {
            "equity": round(float(equity), 2),
            "price": round(price, 4),
            "size_pct": round(capped_pct, 4),
            "notional": round(qty * price, 2),
            "source": "deterministic",
        },
    }
    return trade, None


def build_sized_trades(
    proposals: List[Dict[str, Any]],
    reviewed: List[Dict[str, Any]],
    equity: Optional[float],
    quotes: Optional[Dict[str, Dict]],
    regime: str,
    positions: Optional[List[Dict]] = None,
    llm_decisions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build the executable trade list from strategy proposals.

    Returns (approved_trades, skipped).

    Iterates PROPOSALS as the universe rather than the LLM's final_decisions. The
    LLM can still veto a trade down (REJECT / HOLD), but an empty llm_decisions
    list means "no opinion", not "reject everything" — otherwise the deterministic
    quant fallback, which returns no decisions at all, could never trade.

    `qty` from the LLM is ignored entirely; it is always computed here.
    """
    reviews_by_symbol = {r.get("symbol"): r for r in (reviewed or []) if r.get("symbol")}
    decisions_by_symbol = {
        d.get("symbol"): d for d in (llm_decisions or []) if d.get("symbol")
    }

    approved: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for proposal in proposals or []:
        symbol = proposal.get("symbol", "")
        if not symbol:
            continue

        decision = decisions_by_symbol.get(symbol)

        # The LLM may veto down, but only when it actually expressed an opinion.
        if decision is not None:
            verdict = str(decision.get("decision", "")).upper()
            action = str(decision.get("action", "")).upper()
            if verdict in ("REJECT", "REJECTED") or action == "HOLD":
                skipped.append({"symbol": symbol, "reason": SKIP_LLM_REJECTED})
                logger.info("Sizer: %s skipped — LLM decision %s/%s", symbol, verdict, action)
                continue

        trade, reason = size_proposal(
            proposal=proposal,
            review=reviews_by_symbol.get(symbol),
            equity=equity,
            quotes=quotes,
            regime=regime,
            positions=positions,
            llm_decision=decision,
        )

        if trade is None:
            skipped.append({"symbol": symbol, "reason": reason})
            logger.warning(
                "Sizer: %s skipped — %s (equity=%s, price=%s). No order placed.",
                symbol,
                reason,
                equity,
                resolve_price(symbol, quotes, positions),
            )
            continue

        approved.append(trade)
        logger.info(
            "Sizer: %s %s qty=%d @ ~$%.2f (%.2f%% of $%.2f = $%.2f)",
            trade["action"],
            symbol,
            trade["qty"],
            trade["sizing"]["price"],
            trade["sizing"]["size_pct"],
            trade["sizing"]["equity"],
            trade["sizing"]["notional"],
        )

    return approved, skipped


def enforce_vetoes(
    trades: List[Dict[str, Any]],
    reviewed: List[Dict[str, Any]],
    regime: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Last line of defense: strip any trade whose symbol carries a VETOED verdict.

    Runs after sizing as defense in depth, so even a trade injected by an LLM
    response cannot reach execution past a veto.

    In CRISIS the veto is absolute and cannot be overridden, as the README states.
    Outside CRISIS an override requires BOTH an explicit override_risk_veto flag
    from the Orchestrator AND conviction > 0.85 — so the default, where the LLM
    simply omits the field, is a hard block.

    Returns (kept, blocked).
    """
    vetoed = {
        r.get("symbol")
        for r in (reviewed or [])
        if r.get("verdict") == "VETOED" and r.get("symbol")
    }
    if not vetoed:
        return list(trades or []), []

    kept: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []

    for trade in trades or []:
        symbol = trade.get("symbol")
        if symbol not in vetoed:
            kept.append(trade)
            continue

        if regime == "CRISIS":
            logger.warning("CRISIS: Risk Guardian veto is absolute — blocking %s", symbol)
            blocked.append({"symbol": symbol, "reason": SKIP_VETOED})
            continue

        try:
            conviction = float(trade.get("conviction") or 0)
        except (TypeError, ValueError):
            conviction = 0.0

        if trade.get("override_authorized") and conviction > 0.85:
            logger.warning(
                "Orchestrator OVERRODE Risk Guardian veto on %s (conviction=%.2f)",
                symbol,
                conviction,
            )
            kept.append(trade)
            continue

        logger.info("Risk Guardian veto enforced on %s — blocked", symbol)
        blocked.append({"symbol": symbol, "reason": SKIP_VETOED})

    return kept, blocked
