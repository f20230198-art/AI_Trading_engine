"""
DOOM ENGINE — Market Intelligence Agent
Ingests real-time price data, monitors momentum, volume anomalies,
and macro signals across the full watchlist.
"""
import json
import logging
from typing import Any, Dict

from agents.base_agent import BaseAgent
from core.data_feeds import market_data, MarketDataFeed
from core.shared_state import shared_state
from config import WATCHLIST_ALL

logger = logging.getLogger("doom.agent.market_intel")


class MarketIntelAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="market_intel",
            role="Market Intelligence — monitors real-time prices, volume, and momentum",
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Fetching market data")

        try:
            # 1. Fetch real-time quotes
            quotes = await market_data.get_latest_quotes(WATCHLIST_ALL)
            await shared_state.update_market_data(quotes)

            # 2. Build market summary for LLM analysis
            market_summary = self._build_market_summary(quotes)

            # 3. Recall past similar market conditions
            memory_context = await self._recall_memories(market_summary[:500])

            # 4. Get regime context
            regime_context = self._get_regime_context()

            # 5. LLM call — real model analysis of current market
            system_prompt = """You are a Market Intelligence Agent in the DOOM ENGINE autonomous trading system.
Your job: analyze real-time market data and identify actionable signals.

You MUST output valid JSON with this exact structure:
{
  "market_condition": "brief 1-line market summary",
  "signals": [
    {
      "symbol": "TICKER",
      "signal_type": "MOMENTUM_BULLISH|MOMENTUM_BEARISH|VOLUME_SPIKE|BREAKOUT|BREAKDOWN|REVERSAL",
      "strength": 0.0 to 1.0,
      "reasoning": "why this signal matters"
    }
  ],
  "sector_heat": {
    "technology": -1.0 to 1.0,
    "energy": -1.0 to 1.0,
    "financials": -1.0 to 1.0
  },
  "top_opportunities": ["TICKER1", "TICKER2", "TICKER3"],
  "risk_flags": ["any market-wide concerns"]
}

Be precise. Base everything on the actual numbers provided. Do not hallucinate data."""

            user_prompt = f"""{regime_context}

CURRENT MARKET DATA:
{market_summary}

{memory_context}

Analyze the current market state. Identify the strongest signals, sector rotation patterns,
volume anomalies, and top 3 trading opportunities. Output JSON only."""

            response = await self._call_llm(system_prompt, user_prompt)

            # Parse LLM response
            analysis = self._parse_json(response)
            if not analysis:
                logger.warning("Market Intel LLM returned non-JSON, using raw text")
                analysis = {
                    "market_condition": response[:200],
                    "signals": [],
                    "sector_heat": {"technology": 0, "energy": 0, "financials": 0},
                    "top_opportunities": [],
                    "risk_flags": [],
                }

            # 6. Store decision in memory
            await self._store_decision(
                situation=market_summary[:500],
                decision=json.dumps(analysis.get("signals", [])[:3]),
                metadata={"regime": shared_state.current_regime},
            )

            # 7. Notify other agents via shared state
            await shared_state.add_agent_message(
                self.name, "orchestrator", "SIGNAL",
                f"Found {len(analysis.get('signals', []))} signals. "
                f"Top: {', '.join(analysis.get('top_opportunities', [])[:3])}"
            )

            await self.update_status("NOMINAL", f"Scanned {len(quotes)} symbols")

            return {
                "agent": self.name,
                "quotes": quotes,
                "analysis": analysis,
                "signal_count": len(analysis.get("signals", [])),
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"MarketIntel error: {e}", exc_info=True)
            raise

    def _build_market_summary(self, quotes: Dict[str, Dict]) -> str:
        """Build a concise market summary string for the LLM."""
        lines = []
        for symbol, q in sorted(quotes.items()):
            lines.append(
                f"{symbol}: ${q['price']:.2f} ({q['change_pct']:+.2f}%) "
                f"Vol={q['volume']:,.0f} H={q['high']:.2f} L={q['low']:.2f} "
                f"VWAP={q.get('vwap', 0):.2f}"
            )

        # Add sector aggregates
        from config import SECTOR_MAP
        for sector, symbols in SECTOR_MAP.items():
            sector_changes = [quotes[s]["change_pct"] for s in symbols if s in quotes]
            if sector_changes:
                avg_change = sum(sector_changes) / len(sector_changes)
                lines.append(f"\n{sector.upper()} SECTOR AVG: {avg_change:+.2f}%")

        return "\n".join(lines)
