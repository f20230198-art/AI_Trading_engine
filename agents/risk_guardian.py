"""
DOOM ENGINE — Risk Guardian Agent
Independently reviews every trade proposal before execution.
Can VETO or RESIZE any trade. Makes its own LLM calls to reason about risk.
"""
import json
import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from core.position_sizer import sector_of
from core.shared_state import shared_state
from core.performance_tracker import performance_tracker
from config import TRADING_CONFIG, SECTOR_MAP

logger = logging.getLogger("doom.agent.risk")


class RiskGuardianAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="risk_guardian",
            role="Risk Guardian — reviews and can VETO any trade proposal",
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Reviewing trade proposals")

        try:
            proposals = context.get("proposals", [])
            portfolio_equity = context.get("equity") or TRADING_CONFIG.initial_capital
            current_positions = context.get("positions", [])
            regime = shared_state.current_regime

            if not proposals:
                await self.update_status("NOMINAL", "No proposals to review")
                return {"agent": self.name, "reviewed": [], "risk_score": 0}

            # 1. Compute portfolio-level risk metrics
            risk_metrics = self._compute_risk_metrics(
                current_positions, portfolio_equity, proposals
            )

            # 2. LLM risk analysis
            regime_context = self._get_regime_context()
            memory_context = await self._recall_memories(
                f"Risk review of {len(proposals)} proposals in {regime} regime"
            )

            proposals_text = json.dumps(proposals[:5], indent=2)
            positions_text = self._format_positions(current_positions)

            system_prompt = """You are the Risk Guardian Agent in the DOOM ENGINE.
You are the last line of defense before any trade is executed.

HARD RULES (non-negotiable):
1. Max 5% of portfolio per single position
2. Max 20% sector concentration
3. Max 3% daily drawdown — if breached, VETO ALL new trades
4. In CRISIS regime, you have absolute VETO power and position sizes must be halved

Your job: Review each trade proposal and either APPROVE, RESIZE, or VETO it.
For each, explain your risk reasoning.

Also compute an overall portfolio risk score (0-100):
0-25 = LOW (safe)
26-50 = MODERATE (normal operations)
51-75 = ELEVATED (tighten stops, reduce size)
76-100 = CRITICAL (veto new longs, hedge existing)

Output valid JSON:
{
  "reviewed_proposals": [
    {
      "symbol": "TICKER",
      "original_action": "BUY|SELL",
      "original_size_pct": 3.0,
      "verdict": "APPROVED|RESIZED|VETOED",
      "adjusted_size_pct": 3.0 or null if vetoed,
      "reasoning": "why this verdict",
      "risk_flags": ["specific risks for this trade"]
    }
  ],
  "portfolio_risk_score": 0 to 100,
  "risk_label": "LOW|MODERATE|ELEVATED|CRITICAL",
  "portfolio_risk_assessment": "2-3 sentence portfolio risk view",
  "active_risk_flags": ["portfolio-level risk concerns"],
  "recommended_hedges": ["any hedging recommendations"]
}"""

            user_prompt = f"""{regime_context}

PORTFOLIO STATE:
  Equity: ${portfolio_equity:,.2f}
  Daily P&L: ${risk_metrics['daily_pnl']:+,.2f} ({risk_metrics['daily_pnl_pct']:+.2f}%)
  Daily Drawdown (today, peak->now): {risk_metrics['current_drawdown_pct']:.2f}% (limit {TRADING_CONFIG.max_daily_drawdown_pct * 100:.1f}%)
  All-time Max Drawdown: {risk_metrics['max_drawdown_pct']:.2f}%
  Position Count: {risk_metrics['position_count']}
  Sector Exposure: {json.dumps(risk_metrics['sector_exposure'])}

CURRENT POSITIONS:
{positions_text}

TRADE PROPOSALS TO REVIEW:
{proposals_text}

{memory_context}

Review each proposal against the hard rules and current portfolio state.
Be conservative — protecting capital is your primary mandate. JSON only."""

            response = await self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            analysis = self._parse_json(response)
            if not analysis:
                analysis = {
                    "reviewed_proposals": [],
                    "portfolio_risk_score": 50,
                    "risk_label": "MODERATE",
                    "portfolio_risk_assessment": response[:200],
                    "active_risk_flags": [],
                    "recommended_hedges": [],
                }

            # Enforce the hard limits in code. The LLM's review is advisory —
            # this pass guarantees one verdict per proposal and applies the
            # non-negotiable rules, so it works in quant fallback mode too.
            analysis["reviewed_proposals"] = self._enforce_hard_rules(
                proposals=proposals,
                reviewed=analysis.get("reviewed_proposals", []),
                risk_metrics=risk_metrics,
                regime=regime,
            )

            # Update risk in shared state
            risk_score = analysis.get("portfolio_risk_score", 50)
            risk_label = analysis.get("risk_label", "MODERATE")
            await shared_state.update_risk(risk_score, risk_label)

            # Store decision
            reviewed = analysis.get("reviewed_proposals", [])
            vetoed = [r for r in reviewed if r.get("verdict") == "VETOED"]
            approved = [r for r in reviewed if r.get("verdict") in ("APPROVED", "RESIZED")]

            await self._store_decision(
                situation=f"Reviewed {len(proposals)} proposals, regime={regime}, "
                          f"drawdown={risk_metrics['current_drawdown_pct']:.2f}%",
                decision=f"Approved {len(approved)}, Vetoed {len(vetoed)}, "
                         f"Risk score: {risk_score}",
                metadata={"risk_score": risk_score, "regime": regime},
            )

            await shared_state.add_agent_message(
                self.name, "orchestrator", "RISK_REVIEW",
                f"Risk={risk_score} ({risk_label}) | "
                f"Approved {len(approved)}, Vetoed {len(vetoed)} of {len(proposals)}"
            )

            await self.update_status(
                "ADVISORY" if risk_score > 50 else "NOMINAL",
                f"Risk: {risk_score} ({risk_label})"
            )

            return {
                "agent": self.name,
                "analysis": analysis,
                "reviewed": reviewed,
                "risk_score": risk_score,
                "risk_label": risk_label,
                "daily_drawdown_pct": risk_metrics["current_drawdown_pct"],
                "veto_all": bool(vetoed) and len(vetoed) == len(reviewed),
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"RiskGuardian error: {e}", exc_info=True)
            raise

    def _enforce_hard_rules(
        self,
        proposals: List[Dict],
        reviewed: List[Dict],
        risk_metrics: Dict,
        regime: str,
    ) -> List[Dict]:
        """
        Apply the non-negotiable risk limits in code rather than trusting the LLM
        to have applied them.

        Also guarantees exactly one review entry per proposal. That matters for
        the deterministic quant fallback, which returns no reviews at all — without
        a synthesized entry those proposals would carry no verdict downstream.

        Rules only ever tighten a position, never loosen one.
        """
        reviews_by_symbol = {r.get("symbol"): dict(r) for r in (reviewed or []) if r.get("symbol")}
        limit_pct = TRADING_CONFIG.max_daily_drawdown_pct * 100
        daily_dd = float(risk_metrics.get("current_drawdown_pct") or 0)
        drawdown_breached = daily_dd >= limit_pct

        if drawdown_breached:
            logger.warning(
                "HARD RULE: daily drawdown %.2f%% >= %.2f%% limit — vetoing ALL new trades",
                daily_dd, limit_pct,
            )

        sector_exposure = dict(risk_metrics.get("sector_exposure") or {})
        max_position_pct = TRADING_CONFIG.max_position_pct * 100
        max_sector_pct = TRADING_CONFIG.max_sector_pct * 100

        enforced: List[Dict] = []
        for proposal in proposals or []:
            symbol = proposal.get("symbol", "")
            if not symbol:
                continue

            try:
                proposed_pct = float(proposal.get("position_size_pct") or 0) or 1.0
            except (TypeError, ValueError):
                proposed_pct = 1.0

            review = reviews_by_symbol.get(symbol) or {
                "symbol": symbol,
                "original_action": proposal.get("action", "BUY"),
                "original_size_pct": proposed_pct,
                "verdict": "APPROVED",
                "adjusted_size_pct": proposed_pct,
                "reasoning": "No LLM review returned — deterministic default applied.",
                "risk_flags": [],
            }

            # Rule A: daily drawdown breach vetoes everything.
            if drawdown_breached:
                review["verdict"] = "VETOED"
                review["adjusted_size_pct"] = None
                review["reasoning"] = (
                    f"Daily drawdown {daily_dd:.2f}% breached the "
                    f"{limit_pct:.1f}% limit — all new trades vetoed."
                )
                enforced.append(review)
                continue

            if review.get("verdict") == "VETOED":
                enforced.append(review)
                continue

            # Current working size for this proposal.
            try:
                size_pct = float(review.get("adjusted_size_pct") or proposed_pct)
            except (TypeError, ValueError):
                size_pct = proposed_pct

            # Rule B: per-position cap.
            if size_pct > max_position_pct:
                logger.info(
                    "HARD RULE: %s resized %.2f%% -> %.2f%% (max position)",
                    symbol, size_pct, max_position_pct,
                )
                size_pct = max_position_pct
                review["verdict"] = "RESIZED"

            # Rule C: sector concentration cap, counting already-approved trades.
            sector = sector_of(symbol)
            current = float(sector_exposure.get(sector, 0) or 0)
            headroom = max_sector_pct - current
            if headroom <= 0:
                review["verdict"] = "VETOED"
                review["adjusted_size_pct"] = None
                review["reasoning"] = (
                    f"{sector} exposure {current:.1f}% is at the "
                    f"{max_sector_pct:.0f}% sector cap — no headroom."
                )
                logger.info("HARD RULE: %s vetoed — %s sector cap reached", symbol, sector)
                enforced.append(review)
                continue
            if size_pct > headroom:
                logger.info(
                    "HARD RULE: %s resized %.2f%% -> %.2f%% (%s sector headroom)",
                    symbol, size_pct, headroom, sector,
                )
                size_pct = headroom
                review["verdict"] = "RESIZED"

            # Rule D: CRISIS halving. The sizer defers to us whenever a review
            # exists, so this is the single place the regime haircut is applied.
            if regime == "CRISIS":
                size_pct *= 0.5
                review["verdict"] = "RESIZED"
                logger.info("HARD RULE: %s halved to %.2f%% (CRISIS regime)", symbol, size_pct)

            review["adjusted_size_pct"] = round(size_pct, 4)
            sector_exposure[sector] = current + size_pct
            enforced.append(review)

        return enforced

    def _compute_risk_metrics(
        self,
        positions: List[Dict],
        equity: float,
        proposals: List[Dict],
    ) -> Dict:
        """Compute real portfolio risk metrics."""
        # Daily P&L from performance tracker
        metrics = performance_tracker.get_dashboard_metrics()
        daily_pnl = metrics.get("total_pnl", 0)
        daily_pnl_pct = (daily_pnl / equity * 100) if equity > 0 else 0

        # Sector exposure from current positions
        sector_exposure = {"technology": 0, "energy": 0, "financials": 0, "other": 0}
        for pos in positions:
            symbol = pos.get("symbol", "")
            market_value = abs(float(pos.get("market_value", 0)))
            pct = (market_value / equity * 100) if equity > 0 else 0

            placed = False
            for sector, symbols in SECTOR_MAP.items():
                if symbol in symbols:
                    sector_exposure[sector] += pct
                    placed = True
                    break
            if not placed:
                sector_exposure["other"] += pct

        # Drawdown. The 3% hard rule is a DAILY limit, so it must use the
        # day-scoped figure — max_drawdown spans the whole curve and only grows,
        # which would latch the rule on permanently once tripped.
        daily_drawdown = performance_tracker.compute_daily_drawdown()
        max_drawdown = metrics.get("max_drawdown", 0)

        return {
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "current_drawdown_pct": daily_drawdown,
            "max_drawdown_pct": max_drawdown,
            "position_count": len(positions),
            "sector_exposure": sector_exposure,
        }

    def _format_positions(self, positions: List[Dict]) -> str:
        if not positions:
            return "  No current positions"
        lines = []
        for p in positions:
            lines.append(
                f"  {p.get('symbol', '?')}: {p.get('qty', 0)} shares @ "
                f"${float(p.get('avg_entry_price', 0)):.2f} | "
                f"P&L: ${float(p.get('unrealized_pl', 0)):+.2f}"
            )
        return "\n".join(lines)
