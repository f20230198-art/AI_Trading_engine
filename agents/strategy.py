"""
DOOM ENGINE — Strategy Agent
Receives signals from all intel agents, runs multiple sub-strategies,
and generates ranked trade proposals with sizing.
"""
import json
import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from core.shared_state import shared_state
from config import TRADING_CONFIG

logger = logging.getLogger("doom.agent.strategy")


class StrategyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="strategy",
            role="Strategy Agent — synthesizes signals into trade proposals using multiple sub-strategies",
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Generating trade proposals")

        try:
            # Collect all inputs
            market_analysis = context.get("market_intel", {}).get("analysis", {})
            sector_analyses = {}
            for key in ["sector_tech", "sector_energy", "sector_financials"]:
                if key in context:
                    sector_analyses[key] = context[key].get("analysis", {})

            sentiment_analysis = context.get("sentiment", {}).get("analysis", {})
            regime = shared_state.current_regime
            portfolio_equity = context.get("equity", TRADING_CONFIG.initial_capital)

            # Build comprehensive input for the Strategy LLM
            regime_context = self._get_regime_context()
            memory_context = await self._recall_memories(
                f"Strategy generation in {regime} regime with "
                f"{len(market_analysis.get('signals', []))} market signals"
            )

            # Format all agent inputs
            input_summary = self._format_inputs(
                market_analysis, sector_analyses, sentiment_analysis
            )

            system_prompt = """You are the Strategy Agent in the DOOM ENGINE.
You synthesize signals from Market Intel, Sector Specialists, and Sentiment Scanner
to generate ranked trade proposals.

You run THREE sub-strategies simultaneously:
1. MOMENTUM: EMA 12/26 crossover with volume confirmation
2. MEAN_REVERSION: Bollinger Band squeeze reversals
3. SENTIMENT_SPIKE: High positive sentiment combined with price breakout

For each strategy, evaluate which symbols have the strongest signals.
Then produce a RANKED list of trade proposals with position sizing.

Position sizing rules:
- Max 5% of portfolio per position
- Reduce sizes by 25% in HIGH_VOLATILITY regime
- Reduce sizes by 50% in CRISIS regime
- Increase conviction in TRENDING for momentum trades

Output valid JSON:
{
  "strategy_assessment": "2-3 sentence overall strategy view",
  "sub_strategy_signals": {
    "momentum": [
      {"symbol": "TICK", "direction": "BUY|SELL", "strength": 0-1, "reasoning": "why"}
    ],
    "mean_reversion": [
      {"symbol": "TICK", "direction": "BUY|SELL", "strength": 0-1, "reasoning": "why"}
    ],
    "sentiment_spike": [
      {"symbol": "TICK", "direction": "BUY|SELL", "strength": 0-1, "reasoning": "why"}
    ]
  },
  "trade_proposals": [
    {
      "rank": 1,
      "symbol": "TICKER",
      "action": "BUY|SELL",
      "strategy": "MOMENTUM|MEAN_REVERSION|SENTIMENT_SPIKE",
      "conviction": 0.0 to 1.0,
      "position_size_pct": 1 to 5,
      "entry_type": "MARKET|LIMIT",
      "stop_loss_pct": 1 to 5,
      "take_profit_pct": 2 to 10,
      "reasoning": "complete reasoning for this trade"
    }
  ],
  "no_trade_symbols": ["symbols explicitly avoided and why"]
}"""

            user_prompt = f"""{regime_context}

PORTFOLIO EQUITY: ${portfolio_equity:,.2f}

ALL AGENT INPUTS:
{input_summary}

{memory_context}

Generate ranked trade proposals across all three sub-strategies.
Be selective — only propose high-conviction trades. JSON only."""

            response = await self._call_llm(system_prompt, user_prompt, max_tokens=1500)

            analysis = self._parse_json(response)
            if not analysis:
                analysis = {
                    "strategy_assessment": response[:300],
                    "sub_strategy_signals": {},
                    "trade_proposals": [],
                    "no_trade_symbols": [],
                }

            proposals = analysis.get("trade_proposals", [])

            # Store decision
            await self._store_decision(
                situation=f"Regime={regime}, Signals={len(market_analysis.get('signals', []))}, "
                          f"Sentiment={sentiment_analysis.get('overall_market_sentiment', 0):.2f}",
                decision=json.dumps([{
                    "symbol": p.get("symbol"),
                    "action": p.get("action"),
                    "conviction": p.get("conviction"),
                } for p in proposals[:3]]),
                metadata={"regime": regime, "proposal_count": len(proposals)},
            )

            await shared_state.add_agent_message(
                self.name, "orchestrator", "PROPOSALS",
                f"Generated {len(proposals)} trade proposals. "
                f"Top: {proposals[0].get('symbol', 'none') if proposals else 'none'}"
            )

            await self.update_status("NOMINAL", f"{len(proposals)} proposals generated")

            return {
                "agent": self.name,
                "analysis": analysis,
                "proposals": proposals,
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"Strategy error: {e}", exc_info=True)
            raise

    def _format_inputs(
        self,
        market: Dict,
        sectors: Dict[str, Dict],
        sentiment: Dict,
    ) -> str:
        """Format all agent inputs into a readable string for the LLM."""
        parts = []

        # Market Intel signals
        signals = market.get("signals", [])
        if signals:
            parts.append("MARKET INTEL SIGNALS:")
            for s in signals[:10]:
                parts.append(
                    f"  {s.get('symbol', '?')}: {s.get('signal_type', '?')} "
                    f"(strength={s.get('strength', 0):.2f}) — {s.get('reasoning', '')[:80]}"
                )

        # Sector analyses
        for sector_key, analysis in sectors.items():
            sector_name = sector_key.replace("sector_", "").upper()
            conviction = analysis.get("sector_conviction", 0)
            parts.append(f"\n{sector_name} SECTOR (conviction={conviction:.2f}):")
            for pick in analysis.get("top_picks", [])[:3]:
                parts.append(
                    f"  {pick.get('symbol', '?')}: {pick.get('action', '?')} "
                    f"(conv={pick.get('conviction', 0):.2f}) — {pick.get('reasoning', '')[:60]}"
                )

        # Sentiment
        overall = sentiment.get("overall_market_sentiment", 0)
        parts.append(f"\nSENTIMENT (overall={overall:.2f}):")
        parts.append(f"  Summary: {sentiment.get('sentiment_summary', 'N/A')}")
        spikes = sentiment.get("sentiment_spikes", [])
        if spikes:
            parts.append(f"  Spikes: {', '.join(spikes)}")

        return "\n".join(parts)
