"""
DOOM ENGINE — The Orchestrator (DOOM Brain)
Master coordinator. Resolves conflicts, makes final go/no-go decisions,
generates "DOOM SPEAKS" briefings. The central nervous system.
"""
import json
import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from core.conflict_logger import conflict_logger
from core.position_sizer import build_sized_trades, enforce_vetoes
from core.shared_state import shared_state
from config import TRADING_CONFIG

logger = logging.getLogger("doom.orchestrator")


class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="orchestrator",
            role="DOOM Brain — master coordinator, conflict resolver, final decision maker",
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Synthesizing all agent outputs")

        try:
            # Gather all agent outputs
            market_result = context.get("market_intel", {})
            sector_results = {
                "tech": context.get("sector_tech", {}),
                "energy": context.get("sector_energy", {}),
                "financials": context.get("sector_financials", {}),
            }
            sentiment_result = context.get("sentiment", {})
            strategy_result = context.get("strategy", {})
            risk_result = context.get("risk_guardian", {})
            portfolio_result = context.get("portfolio", {})

            regime = shared_state.current_regime
            equity = context.get("equity") or TRADING_CONFIG.initial_capital

            # 1. Detect and log conflicts
            conflicts = self._detect_conflicts(strategy_result, risk_result, sector_results)
            for conflict in conflicts:
                await conflict_logger.log_conflict(**conflict)
                await shared_state.add_conflict(conflict)

            # 2. Build comprehensive decision context
            regime_context = self._get_regime_context()
            memory_context = await self._recall_memories(
                f"Orchestrator decision cycle in {regime}, "
                f"{len(conflicts)} conflicts detected"
            )

            # Format all outputs for LLM
            synthesis = self._build_synthesis(
                market_result, sector_results, sentiment_result,
                strategy_result, risk_result, portfolio_result, conflicts
            )

            system_prompt = """You are the DOOM BRAIN — the master Orchestrator of the DOOM ENGINE
autonomous trading system. You make the FINAL decision on every trade.

Your inputs come from 8+ specialized agents. Your job:
1. Synthesize all agent intelligence
2. Resolve conflicts (when Strategy says BUY but Risk says VETO, you decide)
3. Make final GO/NO-GO decision on each trade proposal
4. Generate a "DOOM SPEAKS" commander's briefing

Decision rules:
- In CRISIS regime, Risk Guardian's VETO is absolute — you cannot override it
- In other regimes, you CAN override Risk if conviction is very high (>0.85) and
  the risk concern is minor. To do so you MUST set "override_risk_veto": true on
  that decision. Omitting the field means the veto stands.
- Always explain your reasoning when overriding any agent
- Position sizes from Risk Guardian's adjustments are binding
- Do NOT output share quantities. Position sizing is computed deterministically
  from position_size_pct, live price, and portfolio equity. Your job is to decide
  WHETHER each trade executes, not how large it is.

Output valid JSON:
{
  "final_decisions": [
    {
      "symbol": "TICKER",
      "action": "BUY|SELL|HOLD",
      "order_type": "MARKET|LIMIT",
      "limit_price": null,
      "stop_loss_pct": 2.0,
      "originating_agent": "strategy",
      "strategy": "MOMENTUM|MEAN_REVERSION|SENTIMENT_SPIKE",
      "conviction": 0.0 to 1.0,
      "decision": "EXECUTE|HOLD|REJECT",
      "override_risk_veto": false,
      "reasoning": "full reasoning chain"
    }
  ],
  "conflict_verdicts": [
    {
      "conflict_id": 0,
      "verdict": "brief verdict",
      "reasoning": "why"
    }
  ],
  "doom_speaks": "2-3 sentence commander's briefing in authoritative tone. What did you decide and why. Written as if addressing a war room.",
  "cycle_assessment": "1 sentence overall cycle assessment",
  "next_cycle_focus": "what to watch for next cycle"
}"""

            user_prompt = f"""{regime_context}

PORTFOLIO EQUITY: ${equity:,.2f}
CYCLE: {context.get('cycle', 0)}

=== FULL AGENT SYNTHESIS ===
{synthesis}

{memory_context}

Make your final decisions. Resolve all conflicts. Generate your DOOM SPEAKS briefing.
Remember: you are the supreme commander. Every trade lives or dies by your verdict.
JSON only."""

            response = await self._call_llm(system_prompt, user_prompt, max_tokens=2000)

            analysis = self._parse_json(response)
            if not analysis:
                analysis = {
                    "final_decisions": [],
                    "conflict_verdicts": [],
                    "doom_speaks": f"Cycle {context.get('cycle', 0)} analysis complete. "
                                   f"Market regime: {regime}. Maintaining current posture.",
                    "cycle_assessment": response[:200],
                    "next_cycle_focus": "",
                }

            # 3. Post DOOM SPEAKS briefing
            briefing = analysis.get("doom_speaks", "Awaiting next cycle data.")
            await shared_state.add_doom_speaks(briefing)

            # 4. Build approved trades list for execution.
            #    Sizing is deterministic and computed in code — the LLM's `qty` is
            #    ignored entirely. We iterate proposals rather than final_decisions
            #    so the quant fallback (which returns no decisions) can still trade.
            proposals = strategy_result.get("proposals", [])
            reviewed = risk_result.get("reviewed", [])

            approved_trades, skipped = build_sized_trades(
                proposals=proposals,
                reviewed=reviewed,
                equity=context.get("equity"),
                quotes=context.get("quotes", {}),
                regime=regime,
                positions=context.get("positions", []),
                llm_decisions=analysis.get("final_decisions", []),
            )

            # Defense in depth: nothing carrying a veto reaches execution.
            approved_trades, vetoed_blocked = enforce_vetoes(approved_trades, reviewed, regime)
            skipped.extend(vetoed_blocked)

            if skipped:
                logger.warning(
                    "Orchestrator: %d trade(s) not executed — %s",
                    len(skipped),
                    ", ".join(f"{s['symbol']}({s['reason']})" for s in skipped),
                )

            # Store decision
            await self._store_decision(
                situation=f"Cycle {context.get('cycle', 0)}, regime={regime}, "
                          f"{len(conflicts)} conflicts, "
                          f"{len(strategy_result.get('proposals', []))} proposals",
                decision=f"Approved {len(approved_trades)} trades, "
                         f"Rejected {len(analysis.get('final_decisions', [])) - len(approved_trades)}",
                metadata={
                    "regime": regime,
                    "approved_symbols": [t["symbol"] for t in approved_trades],
                },
            )

            await shared_state.add_agent_message(
                self.name, "execution", "EXECUTE_ORDERS",
                f"Approved {len(approved_trades)} trades for execution"
            )

            await self.update_status(
                "ACTIVE",
                f"Approved {len(approved_trades)} trades | {len(skipped)} skipped "
                f"| {len(conflicts)} conflicts resolved"
            )

            return {
                "agent": self.name,
                "analysis": analysis,
                "approved_trades": approved_trades,
                "skipped_trades": skipped,
                "conflicts": conflicts,
                "briefing": briefing,
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            raise

    def _detect_conflicts(
        self,
        strategy_result: Dict,
        risk_result: Dict,
        sector_results: Dict[str, Dict],
    ) -> List[Dict]:
        """Detect real conflicts between agent outputs."""
        conflicts = []

        proposals = strategy_result.get("proposals", [])
        reviewed = risk_result.get("reviewed", [])

        # Strategy vs Risk Guardian conflicts
        reviewed_by_symbol = {r.get("symbol"): r for r in reviewed}
        for proposal in proposals:
            symbol = proposal.get("symbol", "")
            review = reviewed_by_symbol.get(symbol, {})

            if review.get("verdict") == "VETOED":
                conflicts.append({
                    "agent_a_name": "strategy",
                    "agent_a_position": f"{proposal.get('action', 'BUY')} {symbol}",
                    "agent_a_reasoning": proposal.get("reasoning", "Strategy signal")[:200],
                    "agent_b_name": "risk_guardian",
                    "agent_b_position": f"VETO {symbol}",
                    "agent_b_reasoning": review.get("reasoning", "Risk too high")[:200],
                    "symbol": symbol,
                    "verdict": "PENDING",
                    "verdict_reasoning": "Awaiting Orchestrator resolution",
                })

        # Cross-sector conflicts (e.g., one sector bullish, another bearish on same theme)
        tech_conviction = (
            sector_results.get("tech", {}).get("analysis", {}).get("sector_conviction", 0)
        )
        fin_conviction = (
            sector_results.get("financials", {}).get("analysis", {}).get("sector_conviction", 0)
        )

        # If tech and financials have opposite strong convictions, that's a conflict
        if abs(tech_conviction) > 0.5 and abs(fin_conviction) > 0.5:
            if (tech_conviction > 0) != (fin_conviction > 0):
                conflicts.append({
                    "agent_a_name": "sector_tech",
                    "agent_a_position": f"Tech {'BULLISH' if tech_conviction > 0 else 'BEARISH'} ({tech_conviction:.2f})",
                    "agent_a_reasoning": sector_results.get("tech", {}).get("analysis", {}).get(
                        "sector_summary", ""
                    )[:200],
                    "agent_b_name": "sector_financials",
                    "agent_b_position": f"Fin {'BULLISH' if fin_conviction > 0 else 'BEARISH'} ({fin_conviction:.2f})",
                    "agent_b_reasoning": sector_results.get("financials", {}).get("analysis", {}).get(
                        "sector_summary", ""
                    )[:200],
                    "symbol": None,
                    "verdict": "SECTOR_DIVERGENCE",
                    "verdict_reasoning": "Sector specialists disagree on market direction",
                })

        return conflicts

    def _build_synthesis(
        self,
        market: Dict,
        sectors: Dict,
        sentiment: Dict,
        strategy: Dict,
        risk: Dict,
        portfolio: Dict,
        conflicts: List[Dict],
    ) -> str:
        """Build a comprehensive synthesis of all agent outputs."""
        parts = []

        # Market Intel
        analysis = market.get("analysis", {})
        parts.append(f"MARKET INTEL: {analysis.get('market_condition', 'N/A')}")
        signals = analysis.get("signals", [])
        parts.append(f"  Signals: {len(signals)} detected")
        for s in signals[:5]:
            parts.append(f"    {s.get('symbol', '?')}: {s.get('signal_type', '?')} (str={s.get('strength', 0):.2f})")

        # Sectors
        for name, result in sectors.items():
            sa = result.get("analysis", {})
            parts.append(
                f"\n{name.upper()} SECTOR: conviction={sa.get('sector_conviction', 0):.2f}"
            )
            parts.append(f"  {sa.get('sector_summary', 'N/A')}")

        # Sentiment
        sa = sentiment.get("analysis", {})
        parts.append(
            f"\nSENTIMENT: overall={sa.get('overall_market_sentiment', 0):.2f}"
        )
        parts.append(f"  {sa.get('sentiment_summary', 'N/A')}")
        spikes = sa.get("sentiment_spikes", [])
        if spikes:
            parts.append(f"  Spikes: {', '.join(spikes)}")

        # Strategy proposals
        proposals = strategy.get("proposals", [])
        parts.append(f"\nSTRATEGY: {len(proposals)} trade proposals")
        for p in proposals[:5]:
            parts.append(
                f"  #{p.get('rank', '?')} {p.get('action', '?')} {p.get('symbol', '?')} "
                f"via {p.get('strategy', '?')} (conv={p.get('conviction', 0):.2f}, "
                f"size={p.get('position_size_pct', 0)}%)"
            )

        # Risk review
        parts.append(
            f"\nRISK GUARDIAN: score={risk.get('risk_score', '?')} ({risk.get('risk_label', '?')})"
        )
        for r in risk.get("reviewed", [])[:5]:
            parts.append(
                f"  {r.get('symbol', '?')}: {r.get('verdict', '?')} — {r.get('reasoning', '')[:80]}"
            )

        # Portfolio
        pa = portfolio.get("analysis", {})
        parts.append(
            f"\nPORTFOLIO: health={pa.get('portfolio_health', 'N/A')}, "
            f"rebalance={'YES' if portfolio.get('rebalance_needed') else 'NO'}"
        )

        # Conflicts
        if conflicts:
            parts.append(f"\nACTIVE CONFLICTS ({len(conflicts)}):")
            for i, c in enumerate(conflicts):
                parts.append(
                    f"  [{i}] {c['agent_a_name']}({c['agent_a_position']}) vs "
                    f"{c['agent_b_name']}({c['agent_b_position']})"
                )

        return "\n".join(parts)
