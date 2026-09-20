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
            portfolio_equity = context.get("equity") or TRADING_CONFIG.initial_capital

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

            # Pass the cycle context through so the quant fallback can compute
            # real signals from live quotes when no LLM backend is available.
            response = await self._call_llm(
                system_prompt, user_prompt, max_tokens=1500, context=context
            )

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

    def _quant_fallback(self, context: Dict) -> str:
        """
        Deterministic strategy — real signals, no LLM, no network.

        The base implementation returns an empty proposal list, which meant the
        advertised "always works without API keys" mode could never place a trade.
        This computes momentum and mean-reversion signals from the quotes already
        fetched by market_intel this cycle.

        Momentum:       price above VWAP and up on the day -> BUY
        Mean reversion: stretched well below the day's range in a RANGING regime -> BUY
        Sizing is left as position_size_pct; the sizer converts it to shares.
        """
        import json

        regime = shared_state.current_regime or "RANGING"
        quotes = context.get("quotes") or {}
        sentiment_scores = context.get("sentiment", {}).get("analysis", {}).get(
            "symbol_sentiments", {}
        )

        momentum, mean_reversion, proposals = [], [], []

        for symbol, quote in quotes.items():
            try:
                price = float(quote.get("price") or 0)
                change_pct = float(quote.get("change_pct") or 0)
                vwap = float(quote.get("vwap") or 0)
                high = float(quote.get("high") or 0)
                low = float(quote.get("low") or 0)
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue

            # --- Momentum: trading above VWAP with positive daily change ---
            if vwap > 0 and price > vwap and change_pct > 0.5:
                strength = min(1.0, change_pct / 3.0)
                momentum.append({
                    "symbol": symbol,
                    "direction": "BUY",
                    "strength": round(strength, 3),
                    "reasoning": f"Price ${price:.2f} above VWAP ${vwap:.2f}, {change_pct:+.2f}% on day",
                })
                proposals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "strategy": "MOMENTUM",
                    "conviction": round(strength, 3),
                    "position_size_pct": round(1.0 + 2.0 * strength, 2),
                    "entry_type": "MARKET",
                    "stop_loss_pct": 2.0,
                    "take_profit_pct": 5.0,
                    "reasoning": f"Quant momentum: above VWAP, {change_pct:+.2f}% daily move",
                    "_score": strength,
                })

            # --- Mean reversion: near the low of the day's range while ranging ---
            elif regime in ("RANGING", "HIGH_VOLATILITY") and high > low > 0:
                position_in_range = (price - low) / (high - low)
                if position_in_range < 0.25 and change_pct < -0.5:
                    strength = min(1.0, abs(change_pct) / 3.0)
                    mean_reversion.append({
                        "symbol": symbol,
                        "direction": "BUY",
                        "strength": round(strength, 3),
                        "reasoning": f"At {position_in_range:.0%} of daily range, {change_pct:+.2f}%",
                    })
                    proposals.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "strategy": "MEAN_REVERSION",
                        "conviction": round(strength, 3),
                        "position_size_pct": round(1.0 + 1.5 * strength, 2),
                        "entry_type": "MARKET",
                        "stop_loss_pct": 2.5,
                        "take_profit_pct": 4.0,
                        "reasoning": (
                            f"Quant mean reversion: bottom of daily range "
                            f"({position_in_range:.0%}), {change_pct:+.2f}%"
                        ),
                        "_score": strength,
                    })

        # Rank by signal strength, nudged by sentiment when it is available.
        for proposal in proposals:
            score = proposal.pop("_score", 0)
            raw = sentiment_scores.get(proposal["symbol"])
            if isinstance(raw, dict):
                raw = raw.get("score", 0)
            try:
                score += 0.2 * float(raw or 0)
            except (TypeError, ValueError):
                pass
            proposal["_rank_score"] = score

        proposals.sort(key=lambda p: p.pop("_rank_score", 0), reverse=True)
        proposals = proposals[:5]  # stay selective, as the LLM prompt instructs
        for i, proposal in enumerate(proposals, 1):
            proposal["rank"] = i

        logger.info(
            "Strategy quant fallback: %d proposals from %d quotes (regime=%s)",
            len(proposals), len(quotes), regime,
        )

        return json.dumps({
            "strategy_assessment": (
                f"Quant mode: {len(momentum)} momentum and {len(mean_reversion)} "
                f"mean-reversion signals across {len(quotes)} symbols in {regime} regime."
            ),
            "sub_strategy_signals": {
                "momentum": momentum,
                "mean_reversion": mean_reversion,
                "sentiment_spike": [],
            },
            "trade_proposals": proposals,
            "no_trade_symbols": [],
        })

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
