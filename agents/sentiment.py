"""
DOOM ENGINE — Sentiment Scanner Agent
Pulls real financial news via NewsAPI / Alpaca News / RSS feeds.
Runs LLM sentiment classification on each headline.
"""
import json
import logging
from typing import Any, Dict, List

import aiohttp
import feedparser

from agents.base_agent import BaseAgent
from core.shared_state import shared_state
from config import NEWS_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, WATCHLIST_ALL

logger = logging.getLogger("doom.agent.sentiment")

# Financial RSS feeds (always available, no API key needed)
RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
    "https://www.investing.com/rss/news.rss",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
]


class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="sentiment",
            role="Sentiment Scanner — analyzes real financial news and social media sentiment",
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Scanning news sources")

        try:
            # 1. Collect real headlines from multiple sources
            headlines = []

            # Source 1: Alpaca News (if available)
            if ALPACA_API_KEY:
                alpaca_news = await self._fetch_alpaca_news()
                headlines.extend(alpaca_news)

            # Source 2: NewsAPI (if available)
            if NEWS_API_KEY:
                newsapi_headlines = await self._fetch_newsapi()
                headlines.extend(newsapi_headlines)

            # Source 3: RSS feeds (always available)
            rss_headlines = await self._fetch_rss_feeds()
            headlines.extend(rss_headlines)

            if not headlines:
                logger.warning("No headlines collected from any source")
                await self.update_status("WARNING", "No news sources available")
                return {
                    "agent": self.name,
                    "headline_count": 0,
                    "sentiment_scores": {},
                    "overall_sentiment": 0,
                }

            # Deduplicate by title
            seen = set()
            unique_headlines = []
            for h in headlines:
                title_key = h["title"].lower()[:50]
                if title_key not in seen:
                    seen.add(title_key)
                    unique_headlines.append(h)
            headlines = unique_headlines[:30]  # Cap at 30 for LLM context

            # 2. LLM sentiment classification
            regime_context = self._get_regime_context()
            memory_context = await self._recall_memories(
                f"Market news sentiment scan with {len(headlines)} headlines"
            )

            headlines_text = "\n".join(
                f"- [{h.get('source', 'unknown')}] {h['title']}"
                + (f" | Symbols: {','.join(h.get('symbols', []))}" if h.get('symbols') else "")
                for h in headlines
            )

            system_prompt = """You are a Sentiment Scanner Agent in the DOOM ENGINE.
You analyze real financial news headlines and classify sentiment.

For each headline, determine if it is BULLISH, BEARISH, or NEUTRAL for the
mentioned stocks. Then compute an aggregate sentiment score per symbol.

Output valid JSON:
{
  "headline_sentiments": [
    {
      "headline": "headline text",
      "sentiment": "BULLISH|BEARISH|NEUTRAL",
      "score": -1.0 to 1.0,
      "affected_symbols": ["TICKER1"],
      "impact": "HIGH|MEDIUM|LOW"
    }
  ],
  "symbol_sentiments": {
    "TICKER": {
      "score": -1.0 to 1.0,
      "headline_count": 3,
      "dominant_theme": "brief theme"
    }
  },
  "overall_market_sentiment": -1.0 to 1.0,
  "sentiment_summary": "1-2 sentence overall mood",
  "sentiment_spikes": ["symbols with unusually strong positive or negative sentiment"]
}"""

            user_prompt = f"""{regime_context}

REAL NEWS HEADLINES ({len(headlines)} collected):
{headlines_text}

WATCHLIST SYMBOLS: {', '.join(WATCHLIST_ALL)}

{memory_context}

Classify sentiment for each headline and compute per-symbol aggregate scores.
Focus on headlines mentioning our watchlist symbols. JSON only."""

            response = await self._call_llm(system_prompt, user_prompt)

            analysis = self._parse_json(response)
            if not analysis:
                analysis = {
                    "headline_sentiments": [],
                    "symbol_sentiments": {},
                    "overall_market_sentiment": 0,
                    "sentiment_summary": response[:200],
                    "sentiment_spikes": [],
                }

            # Update shared state with sentiment scores
            symbol_scores = {}
            for sym, data in analysis.get("symbol_sentiments", {}).items():
                if isinstance(data, dict):
                    symbol_scores[sym] = data.get("score", 0)
                else:
                    symbol_scores[sym] = float(data) if data else 0
            await shared_state.update_sentiment(symbol_scores)

            # Store decision
            await self._store_decision(
                situation=f"Scanned {len(headlines)} headlines",
                decision=json.dumps({
                    "overall": analysis.get("overall_market_sentiment", 0),
                    "spikes": analysis.get("sentiment_spikes", []),
                }),
                metadata={"headline_count": len(headlines), "regime": shared_state.current_regime},
            )

            await shared_state.add_agent_message(
                self.name, "orchestrator", "SENTIMENT",
                f"Overall: {analysis.get('overall_market_sentiment', 0):.2f} | "
                f"Headlines: {len(headlines)} | "
                f"Spikes: {', '.join(analysis.get('sentiment_spikes', [])[:3])}"
            )

            await self.update_status("NOMINAL", f"Scanned {len(headlines)} headlines")

            return {
                "agent": self.name,
                "headline_count": len(headlines),
                "analysis": analysis,
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"Sentiment error: {e}", exc_info=True)
            raise

    async def _fetch_alpaca_news(self) -> List[Dict]:
        """Fetch real news from Alpaca News API."""
        headlines = []
        try:
            url = "https://data.alpaca.markets/v1beta1/news"
            headers = {
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            }
            params = {"limit": 20, "sort": "desc"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for article in data.get("news", []):
                            headlines.append({
                                "title": article.get("headline", ""),
                                "source": article.get("source", "alpaca"),
                                "symbols": article.get("symbols", []),
                                "url": article.get("url", ""),
                                "created_at": article.get("created_at", ""),
                            })
                        logger.info(f"Fetched {len(headlines)} headlines from Alpaca News")
        except Exception as e:
            logger.warning(f"Alpaca News fetch failed: {e}")
        return headlines

    async def _fetch_newsapi(self) -> List[Dict]:
        """Fetch real news from NewsAPI."""
        headlines = []
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "category": "business",
                "language": "en",
                "pageSize": 20,
                "apiKey": NEWS_API_KEY,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for article in data.get("articles", []):
                            headlines.append({
                                "title": article.get("title", ""),
                                "source": article.get("source", {}).get("name", "newsapi"),
                                "symbols": [],
                                "url": article.get("url", ""),
                            })
                        logger.info(f"Fetched {len(headlines)} from NewsAPI")
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
        return headlines

    async def _fetch_rss_feeds(self) -> List[Dict]:
        """Fetch headlines from financial RSS feeds."""
        headlines = []

        # General financial RSS
        general_feeds = [
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US",
        ]

        for feed_url in general_feeds:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            feed = feedparser.parse(text)
                            for entry in feed.entries[:10]:
                                headlines.append({
                                    "title": entry.get("title", ""),
                                    "source": "rss",
                                    "symbols": [],
                                    "url": entry.get("link", ""),
                                })
            except Exception as e:
                logger.debug(f"RSS feed {feed_url} failed: {e}")

        # Symbol-specific Yahoo Finance RSS
        for symbol in WATCHLIST_ALL[:5]:  # Top 5 symbols
            try:
                url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            feed = feedparser.parse(text)
                            for entry in feed.entries[:3]:
                                headlines.append({
                                    "title": entry.get("title", ""),
                                    "source": f"yahoo_{symbol}",
                                    "symbols": [symbol],
                                    "url": entry.get("link", ""),
                                })
            except Exception:
                pass

        logger.info(f"Fetched {len(headlines)} from RSS feeds")
        return headlines
