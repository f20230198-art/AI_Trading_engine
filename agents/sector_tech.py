"""
DOOM ENGINE — Technology Sector Specialist Agent
Independently analyzes tech sector using real price data and sector-specific reasoning.
"""
import json
import logging
from typing import Any, Dict

from agents.base_agent import BaseAgent
from core.data_feeds import market_data
from core.shared_state import shared_state
from config import WATCHLIST_TECH

logger = logging.getLogger("doom.agent.sector_tech")


class SectorTechAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="sector_tech",
            role="Technology Sector Specialist — deep analysis of AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSM, AVGO",
        )
        self.sector = "technology"
        self.symbols = WATCHLIST_TECH

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Analyzing tech sector")

        try:
            # Get real quotes for tech stocks
            quotes = context.get("quotes", {})
            tech_quotes = {s: quotes[s] for s in self.symbols if s in quotes}

            if not tech_quotes:
                tech_quotes = await market_data.get_latest_quotes(self.symbols)

            # Get historical data for top movers to compute indicators
            top_movers = sorted(
                tech_quotes.items(),
                key=lambda x: abs(x[1].get("change_pct", 0)),
                reverse=True,
            )[:4]

            indicator_summary = ""
            for symbol, _ in top_movers:
                try:
                    hist = await market_data.get_historical_bars(symbol, days=30)
                    if len(hist) > 20:
                        ema12 = market_data.compute_ema(hist["close"], 12).iloc[-1]
                        ema26 = market_data.compute_ema(hist["close"], 26).iloc[-1]
                        rsi = market_data.compute_rsi(hist["close"]).iloc[-1]
                        bb = market_data.compute_bollinger(hist)
                        bb_width = bb["bb_width"].iloc[-1]
                        price = hist["close"].iloc[-1]

                        indicator_summary += (
                            f"\n{symbol} Indicators: EMA12={ema12:.2f} EMA26={ema26:.2f} "
                            f"{'BULLISH' if ema12 > ema26 else 'BEARISH'} crossover | "
                            f"RSI={rsi:.1f} | BB_Width={bb_width:.4f} | Price={price:.2f}"
                        )
                except Exception as e:
                    logger.warning(f"Could not get indicators for {symbol}: {e}")

            # Build sector summary
            sector_summary = self._build_sector_summary(tech_quotes)

            # Memory recall
            memory_context = await self._recall_memories(sector_summary[:400])
            regime_context = self._get_regime_context()

            # LLM call
            system_prompt = """You are a Technology Sector Specialist Agent in the DOOM ENGINE.
You have deep expertise in tech stocks: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSM, AVGO.
You understand AI/semiconductor cycles, cloud revenue trends, and tech earnings dynamics.

Output valid JSON:
{
  "sector_conviction": -1.0 to 1.0 (negative=bearish, positive=bullish),
  "sector_summary": "1-2 sentence sector view",
  "top_picks": [
    {
      "symbol": "TICKER",
      "action": "BUY|SELL|HOLD",
      "conviction": 0.0 to 1.0,
      "reasoning": "why",
      "target_allocation_pct": 1 to 5
    }
  ],
  "sector_risks": ["risk1", "risk2"],
  "catalyst_watch": ["upcoming event or catalyst"]
}"""

            user_prompt = f"""{regime_context}

TECH SECTOR REAL-TIME DATA:
{sector_summary}

TECHNICAL INDICATORS:
{indicator_summary if indicator_summary else "Computing..."}

{memory_context}

Provide your independent technology sector analysis. Identify your top conviction trades
and sector outlook. Output JSON only."""

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
                    "catalyst_watch": [],
                }

            # Store decision
            await self._store_decision(
                situation=sector_summary[:400],
                decision=json.dumps(analysis.get("top_picks", [])[:2]),
                metadata={"sector": self.sector, "regime": shared_state.current_regime},
            )

            await shared_state.add_agent_message(
                self.name, "orchestrator", "SECTOR_SIGNAL",
                f"Tech conviction: {analysis.get('sector_conviction', 0):.2f} | "
                f"Top: {', '.join(p.get('symbol', '') for p in analysis.get('top_picks', [])[:2])}"
            )

            await self.update_status("NOMINAL", f"Conviction: {analysis.get('sector_conviction', 0):.2f}")

            return {
                "agent": self.name,
                "sector": self.sector,
                "analysis": analysis,
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"SectorTech error: {e}", exc_info=True)
            raise

    def _build_sector_summary(self, quotes: Dict[str, Dict]) -> str:
        lines = ["TECHNOLOGY SECTOR:"]
        for symbol, q in sorted(quotes.items()):
            lines.append(
                f"  {symbol}: ${q['price']:.2f} ({q['change_pct']:+.2f}%) Vol={q['volume']:,.0f}"
            )
        changes = [q["change_pct"] for q in quotes.values()]
        if changes:
            lines.append(f"\n  Sector Avg Change: {sum(changes)/len(changes):+.2f}%")
            lines.append(f"  Best: {max(quotes, key=lambda s: quotes[s]['change_pct'])}")
            lines.append(f"  Worst: {min(quotes, key=lambda s: quotes[s]['change_pct'])}")
        return "\n".join(lines)
