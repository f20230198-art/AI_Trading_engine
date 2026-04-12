"""
DOOM ENGINE — Combined Sector Agent
Analyzes all three sectors (tech, energy, financials) in a single LLM call
to conserve API quota on free-tier Gemini.
"""
import json
import logging
from typing import Any, Dict

from agents.base_agent import BaseAgent
from core.data_feeds import market_data
from core.shared_state import shared_state
from config import WATCHLIST_TECH, WATCHLIST_ENERGY, WATCHLIST_FINANCIALS

logger = logging.getLogger("doom.agent.sectors")

_EMPTY_SECTOR = {
    "sector_conviction": 0,
    "sector_summary": "No data",
    "top_picks": [],
    "sector_risks": [],
}


class SectorCombinedAgent(BaseAgent):
    """Single agent that covers tech, energy, and financials in one LLM call."""

    def __init__(self):
        super().__init__(
            name="sectors",
            role="Combined Sector Specialist — tech, energy, and financials in one analysis",
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Analyzing all sectors")

        try:
            quotes = context.get("quotes", {})

            tech_quotes = {s: quotes[s] for s in WATCHLIST_TECH if s in quotes}
            energy_quotes = {s: quotes[s] for s in WATCHLIST_ENERGY if s in quotes}
            fin_quotes = {s: quotes[s] for s in WATCHLIST_FINANCIALS if s in quotes}

            # Fetch any missing quotes
            if not tech_quotes:
                tech_quotes = await market_data.get_latest_quotes(WATCHLIST_TECH)
            if not energy_quotes:
                energy_quotes = await market_data.get_latest_quotes(WATCHLIST_ENERGY)
            if not fin_quotes:
                fin_quotes = await market_data.get_latest_quotes(WATCHLIST_FINANCIALS)

            # Build compact indicator summary (top 2 movers per sector)
            indicator_lines = []
            for sector_name, sector_quotes in [
                ("TECH", tech_quotes),
                ("ENERGY", energy_quotes),
                ("FINANCIALS", fin_quotes),
            ]:
                top = sorted(
                    sector_quotes.items(),
                    key=lambda x: abs(x[1].get("change_pct", 0)),
                    reverse=True,
                )[:2]
                for symbol, _ in top:
                    try:
                        hist = await market_data.get_historical_bars(symbol, days=30)
                        if len(hist) > 20:
                            ema12 = market_data.compute_ema(hist["close"], 12).iloc[-1]
                            ema26 = market_data.compute_ema(hist["close"], 26).iloc[-1]
                            rsi = market_data.compute_rsi(hist["close"]).iloc[-1]
                            indicator_lines.append(
                                f"[{sector_name}] {symbol}: EMA12={ema12:.2f} EMA26={ema26:.2f} "
                                f"{'BULL' if ema12 > ema26 else 'BEAR'} | RSI={rsi:.1f}"
                            )
                    except Exception as e:
                        logger.debug(f"Indicator skip {symbol}: {e}")

            sector_data = (
                self._fmt_sector("TECHNOLOGY", tech_quotes)
                + "\n" + self._fmt_sector("ENERGY", energy_quotes)
                + "\n" + self._fmt_sector("FINANCIALS", fin_quotes)
            )
            indicator_text = "\n".join(indicator_lines) if indicator_lines else "Computing..."
            regime_context = self._get_regime_context()
            memory_context = await self._recall_memories(sector_data[:400])

            system_prompt = """You are a Multi-Sector Analyst in the DOOM ENGINE covering Technology, Energy, and Financials.

Output valid JSON:
{
  "technology": {
    "sector_conviction": -1.0 to 1.0,
    "sector_summary": "1-2 sentences",
    "top_picks": [{"symbol":"TICK","action":"BUY|SELL|HOLD","conviction":0.0-1.0,"reasoning":"why","target_allocation_pct":1-5}],
    "sector_risks": ["risk"]
  },
  "energy": {
    "sector_conviction": -1.0 to 1.0,
    "sector_summary": "1-2 sentences",
    "top_picks": [{"symbol":"TICK","action":"BUY|SELL|HOLD","conviction":0.0-1.0,"reasoning":"why","target_allocation_pct":1-5}],
    "sector_risks": ["risk"],
    "oil_outlook": "brief"
  },
  "financials": {
    "sector_conviction": -1.0 to 1.0,
    "sector_summary": "1-2 sentences",
    "top_picks": [{"symbol":"TICK","action":"BUY|SELL|HOLD","conviction":0.0-1.0,"reasoning":"why","target_allocation_pct":1-5}],
    "sector_risks": ["risk"],
    "rate_sensitivity_note": "brief"
  }
}"""

            user_prompt = f"""{regime_context}

REAL-TIME SECTOR DATA:
{sector_data}

TECHNICAL INDICATORS:
{indicator_text}

{memory_context}

Analyze all three sectors and identify top conviction trades per sector. JSON only."""

            response = await self._call_llm(system_prompt, user_prompt)

            result = self._parse_json(response)
            if not result:
                result = {
                    "technology": {**_EMPTY_SECTOR, "sector_summary": response[:100]},
                    "energy": {**_EMPTY_SECTOR, "oil_outlook": ""},
                    "financials": {**_EMPTY_SECTOR, "rate_sensitivity_note": ""},
                }

            tech_analysis = result.get("technology", _EMPTY_SECTOR)
            energy_analysis = result.get("energy", _EMPTY_SECTOR)
            fin_analysis = result.get("financials", _EMPTY_SECTOR)

            await self._store_decision(
                situation=sector_data[:400],
                decision=json.dumps({
                    "tech_conviction": tech_analysis.get("sector_conviction", 0),
                    "energy_conviction": energy_analysis.get("sector_conviction", 0),
                    "fin_conviction": fin_analysis.get("sector_conviction", 0),
                }),
                metadata={"regime": shared_state.current_regime},
            )

            await shared_state.add_agent_message(
                self.name, "orchestrator", "SECTOR_SIGNAL",
                f"Tech={tech_analysis.get('sector_conviction', 0):.2f} | "
                f"Energy={energy_analysis.get('sector_conviction', 0):.2f} | "
                f"Fin={fin_analysis.get('sector_conviction', 0):.2f}"
            )

            convictions = [
                tech_analysis.get("sector_conviction", 0),
                energy_analysis.get("sector_conviction", 0),
                fin_analysis.get("sector_conviction", 0),
            ]
            await self.update_status("NOMINAL", f"Sectors analyzed, avg conv={sum(convictions)/3:.2f}")

            return {
                "agent": self.name,
                "sector_tech": {"agent": "sector_tech", "sector": "technology", "analysis": tech_analysis},
                "sector_energy": {"agent": "sector_energy", "sector": "energy", "analysis": energy_analysis},
                "sector_financials": {"agent": "sector_financials", "sector": "financials", "analysis": fin_analysis},
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"SectorCombined error: {e}", exc_info=True)
            raise

    def _fmt_sector(self, name: str, quotes: Dict[str, Dict]) -> str:
        lines = [f"{name} SECTOR:"]
        for symbol, q in sorted(quotes.items()):
            lines.append(
                f"  {symbol}: ${q['price']:.2f} ({q['change_pct']:+.2f}%) Vol={q['volume']:,.0f}"
            )
        changes = [q["change_pct"] for q in quotes.values()]
        if changes:
            lines.append(f"  Avg: {sum(changes)/len(changes):+.2f}%")
        return "\n".join(lines)
