"""
DOOM ENGINE — Energy Sector Specialist Agent
Independently analyzes energy sector with oil/gas cycle awareness.
"""
import json
import logging
from typing import Any, Dict

from agents.base_agent import BaseAgent
from core.data_feeds import market_data
from core.shared_state import shared_state
from config import WATCHLIST_ENERGY

logger = logging.getLogger("doom.agent.sector_energy")


class SectorEnergyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="sector_energy",
            role="Energy Sector Specialist — analyzes XOM, CVX, COP, SLB, EOG, MPC, PSX",
        )
        self.sector = "energy"
        self.symbols = WATCHLIST_ENERGY

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Analyzing energy sector")

        try:
            quotes = context.get("quotes", {})
            energy_quotes = {s: quotes[s] for s in self.symbols if s in quotes}
            if not energy_quotes:
                energy_quotes = await market_data.get_latest_quotes(self.symbols)

            # Get indicators for top movers
            indicator_summary = ""
            top_movers = sorted(
                energy_quotes.items(),
                key=lambda x: abs(x[1].get("change_pct", 0)),
                reverse=True,
            )[:3]

            for symbol, _ in top_movers:
                try:
                    hist = await market_data.get_historical_bars(symbol, days=30)
                    if len(hist) > 20:
                        ema12 = market_data.compute_ema(hist["close"], 12).iloc[-1]
                        ema26 = market_data.compute_ema(hist["close"], 26).iloc[-1]
                        rsi = market_data.compute_rsi(hist["close"]).iloc[-1]
                        indicator_summary += (
                            f"\n{symbol}: EMA12={ema12:.2f} EMA26={ema26:.2f} "
                            f"{'BULL' if ema12 > ema26 else 'BEAR'} | RSI={rsi:.1f}"
                        )
                except Exception as e:
                    logger.warning(f"Indicators failed for {symbol}: {e}")

            sector_summary = self._build_sector_summary(energy_quotes)
            memory_context = await self._recall_memories(sector_summary[:400])
            regime_context = self._get_regime_context()

            system_prompt = """You are an Energy Sector Specialist Agent in the DOOM ENGINE.
Expert in: XOM, CVX, COP, SLB, EOG, MPC, PSX.
You understand oil/gas supply-demand dynamics, OPEC decisions, refining margins,
and energy transition impacts on traditional energy stocks.

Output valid JSON:
{
  "sector_conviction": -1.0 to 1.0,
  "sector_summary": "1-2 sentence view",
  "top_picks": [
    {
      "symbol": "TICKER",
      "action": "BUY|SELL|HOLD",
      "conviction": 0.0 to 1.0,
      "reasoning": "why",
      "target_allocation_pct": 1 to 5
    }
  ],
  "sector_risks": ["risk1"],
  "oil_outlook": "brief crude oil outlook"
}"""

            user_prompt = f"""{regime_context}

ENERGY SECTOR REAL-TIME DATA:
{sector_summary}

TECHNICAL INDICATORS:
{indicator_summary if indicator_summary else "Computing..."}

{memory_context}

Provide your independent energy sector analysis with top conviction trades. JSON only."""

            response = await self._call_llm(system_prompt, user_prompt)

            try:
                json_str = response
                if "```" in json_str:
                    json_str = json_str.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                analysis = json.loads(json_str.strip())
            except json.JSONDecodeError:
                analysis = {
                    "sector_conviction": 0,
                    "sector_summary": response[:200],
                    "top_picks": [],
                    "sector_risks": [],
                    "oil_outlook": "",
                }

            await self._store_decision(
                situation=sector_summary[:400],
                decision=json.dumps(analysis.get("top_picks", [])[:2]),
                metadata={"sector": self.sector, "regime": shared_state.current_regime},
            )

            await shared_state.add_agent_message(
                self.name, "orchestrator", "SECTOR_SIGNAL",
                f"Energy conviction: {analysis.get('sector_conviction', 0):.2f}"
            )

            await self.update_status("NOMINAL", f"Conviction: {analysis.get('sector_conviction', 0):.2f}")

            return {"agent": self.name, "sector": self.sector, "analysis": analysis}

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"SectorEnergy error: {e}", exc_info=True)
            raise

    def _build_sector_summary(self, quotes: Dict[str, Dict]) -> str:
        lines = ["ENERGY SECTOR:"]
        for symbol, q in sorted(quotes.items()):
            lines.append(
                f"  {symbol}: ${q['price']:.2f} ({q['change_pct']:+.2f}%) Vol={q['volume']:,.0f}"
            )
        changes = [q["change_pct"] for q in quotes.values()]
        if changes:
            lines.append(f"\n  Sector Avg: {sum(changes)/len(changes):+.2f}%")
        return "\n".join(lines)
