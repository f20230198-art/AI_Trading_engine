"""
DOOM ENGINE — Portfolio Rebalancing Agent
Runs every 30 minutes to review holdings vs target allocations.
Generates rebalance orders if drift exceeds threshold.
"""
import json
import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from core.data_feeds import market_data
from core.shared_state import shared_state
from config import TRADING_CONFIG, SECTOR_MAP

logger = logging.getLogger("doom.agent.portfolio")


class PortfolioAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="portfolio",
            role="Portfolio Rebalancing — monitors drift and generates rebalance orders",
        )
        self._last_rebalance_cycle = 0

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Checking portfolio drift")

        try:
            cycle = context.get("cycle", 0)
            positions = context.get("positions", [])
            equity = context.get("equity") or TRADING_CONFIG.initial_capital

            # Check if it's time to rebalance (every 30 cycles ~ 30 min at 60s cycles)
            cycles_since_last = cycle - self._last_rebalance_cycle
            rebalance_due = cycles_since_last >= 30 or cycle <= 1

            if not rebalance_due and positions:
                # Quick drift check — only trigger LLM if drift is significant
                drift = self._compute_max_drift(positions, equity)
                if drift < TRADING_CONFIG.rebalance_drift_threshold * 100:
                    await self.update_status("NOMINAL", f"Drift {drift:.1f}% < threshold")
                    return {
                        "agent": self.name,
                        "rebalance_needed": False,
                        "max_drift": drift,
                        "orders": [],
                    }

            # Full rebalance analysis
            regime_context = self._get_regime_context()
            memory_context = await self._recall_memories(
                f"Portfolio rebalance check with {len(positions)} positions, equity=${equity:,.0f}"
            )

            positions_text = self._format_positions(positions, equity)

            system_prompt = """You are the Portfolio Rebalancing Agent in the DOOM ENGINE.
You run every 30 minutes to maintain optimal portfolio allocation.

Rules:
1. No single position should exceed 5% of portfolio
2. No sector should exceed 20% of portfolio
3. If drift exceeds 5% from target, generate rebalance orders
4. In CRISIS regime, target allocation shifts to 50% cash
5. Coordinate with Risk Guardian — your rebalance orders need approval

Output valid JSON:
{
  "portfolio_assessment": "1-2 sentence portfolio state",
  "current_allocations": {
    "technology_pct": 0.0,
    "energy_pct": 0.0,
    "financials_pct": 0.0,
    "cash_pct": 0.0
  },
  "target_allocations": {
    "technology_pct": 0.0,
    "energy_pct": 0.0,
    "financials_pct": 0.0,
    "cash_pct": 0.0
  },
  "max_drift_pct": 0.0,
  "rebalance_needed": true|false,
  "rebalance_orders": [
    {
      "symbol": "TICKER",
      "action": "BUY|SELL",
      "qty": 10,
      "reason": "reduce overweight in tech",
      "urgency": "HIGH|MEDIUM|LOW"
    }
  ],
  "portfolio_health": "OPTIMAL|ACCEPTABLE|NEEDS_REBALANCE|CRITICAL"
}"""

            user_prompt = f"""{regime_context}

PORTFOLIO EQUITY: ${equity:,.2f}
POSITIONS:
{positions_text}

SECTOR MAPPING:
  Technology: {', '.join(SECTOR_MAP['technology'])}
  Energy: {', '.join(SECTOR_MAP['energy'])}
  Financials: {', '.join(SECTOR_MAP['financials'])}

{memory_context}

Analyze current portfolio allocations, compute drift from optimal targets,
and generate rebalance orders if needed. JSON only."""

            response = await self._call_llm(system_prompt, user_prompt)

            analysis = self._parse_json(response)
            if not analysis:
                analysis = {
                    "portfolio_assessment": response[:200],
                    "rebalance_needed": False,
                    "rebalance_orders": [],
                    "portfolio_health": "ACCEPTABLE",
                    "max_drift_pct": 0,
                }

            if analysis.get("rebalance_needed"):
                self._last_rebalance_cycle = cycle

            # Store decision
            await self._store_decision(
                situation=f"Rebalance check: {len(positions)} positions, equity=${equity:,.0f}",
                decision=f"Health: {analysis.get('portfolio_health', 'unknown')}, "
                         f"Orders: {len(analysis.get('rebalance_orders', []))}",
                metadata={"regime": shared_state.current_regime},
            )

            orders = analysis.get("rebalance_orders", [])
            if orders:
                await shared_state.add_agent_message(
                    self.name, "risk_guardian", "REBALANCE_REQUEST",
                    f"Rebalance: {len(orders)} orders proposed"
                )

            status = analysis.get("portfolio_health", "NOMINAL")
            await self.update_status("NOMINAL", f"Health: {status}")

            return {
                "agent": self.name,
                "analysis": analysis,
                "rebalance_needed": analysis.get("rebalance_needed", False),
                "orders": orders,
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"Portfolio error: {e}", exc_info=True)
            raise

    def _compute_max_drift(self, positions: List[Dict], equity: float) -> float:
        """Quick drift computation without LLM."""
        if not positions or equity <= 0:
            return 0

        sector_alloc = {"technology": 0, "energy": 0, "financials": 0}
        for pos in positions:
            symbol = pos.get("symbol", "")
            mv = abs(float(pos.get("market_value", 0)))
            pct = mv / equity * 100
            for sector, symbols in SECTOR_MAP.items():
                if symbol in symbols:
                    sector_alloc[sector] += pct

        # Target: roughly equal across sectors
        target = 100 / 3
        max_drift = max(abs(alloc - target) for alloc in sector_alloc.values())
        return max_drift

    def _format_positions(self, positions: List[Dict], equity: float) -> str:
        if not positions:
            return "  No positions — 100% cash"
        lines = []
        for p in positions:
            symbol = p.get("symbol", "?")
            qty = float(p.get("qty", 0))
            price = float(p.get("current_price", p.get("avg_entry_price", 0)))
            mv = abs(float(p.get("market_value", qty * price)))
            pct = (mv / equity * 100) if equity > 0 else 0
            pnl = float(p.get("unrealized_pl", 0))
            lines.append(
                f"  {symbol}: {qty} shares, MV=${mv:,.2f} ({pct:.1f}%), P&L=${pnl:+.2f}"
            )
        return "\n".join(lines)
