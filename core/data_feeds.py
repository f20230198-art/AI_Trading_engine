"""
DOOM ENGINE — Real Market Data Feeds
Pulls live data from Alpaca Markets API with yfinance fallback for history.
No fake data. If APIs fail, we fail loudly.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
import numpy as np
import pandas as pd

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    ALPACA_DATA_URL,
    WATCHLIST_ALL,
)

logger = logging.getLogger("doom.data_feeds")


class MarketDataFeed:
    """Provides real-time and historical market data from Alpaca + yfinance."""

    def __init__(self):
        self._alpaca_headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        self._client: Optional[httpx.AsyncClient] = None
        self._price_cache: Dict[str, Dict] = {}
        self._history_cache: Dict[str, pd.DataFrame] = {}
        self._last_fetch: Dict[str, datetime] = {}

    async def initialize(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info("MarketDataFeed initialized")

    async def close(self):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Real-time quotes via Alpaca
    # ------------------------------------------------------------------
    async def get_latest_quotes(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Fetch latest quotes, preferring Alpaca and falling back to yfinance.

        Without a fallback the whole engine is unusable with no Alpaca key —
        market_intel and the sector agents fail every cycle and nothing can be
        demonstrated. yfinance needs no key, so the system stays fully
        observable; only order execution actually requires Alpaca.
        """
        symbols = symbols or WATCHLIST_ALL
        if not ALPACA_API_KEY:
            logger.info("No Alpaca key — using yfinance for quotes")
            return await self._yfinance_quotes(symbols)

        url = f"{ALPACA_DATA_URL}/v2/stocks/snapshots"
        params = {"symbols": ",".join(symbols), "feed": "iex"}

        try:
            resp = await self._client.get(url, headers=self._alpaca_headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            quotes = {}
            for symbol, snapshot in data.items():
                latest_trade = snapshot.get("latestTrade", {})
                latest_quote = snapshot.get("latestQuote", {})
                daily_bar = snapshot.get("dailyBar", {})
                prev_daily = snapshot.get("prevDailyBar", {})

                price = latest_trade.get("p", 0)
                prev_close = prev_daily.get("c", price)
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                quotes[symbol] = {
                    "symbol": symbol,
                    "price": price,
                    "bid": latest_quote.get("bp", 0),
                    "ask": latest_quote.get("ap", 0),
                    "volume": daily_bar.get("v", 0),
                    "high": daily_bar.get("h", 0),
                    "low": daily_bar.get("l", 0),
                    "open": daily_bar.get("o", 0),
                    "prev_close": prev_close,
                    "change_pct": round(change_pct, 2),
                    "timestamp": latest_trade.get("t", ""),
                    "vwap": daily_bar.get("vw", 0),
                }

            self._price_cache.update(quotes)
            return quotes

        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Alpaca quotes failed ({e.response.status_code}), falling back to yfinance"
            )
            return await self._yfinance_quotes(symbols)
        except Exception as e:
            logger.warning(f"Alpaca quotes failed ({e}), falling back to yfinance")
            return await self._yfinance_quotes(symbols)

    async def _yfinance_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Quote snapshots from yfinance, shaped exactly like the Alpaca path.

        Uses 2 days of daily bars so change_pct can be computed against the
        prior close. Runs in a thread since yfinance is synchronous.
        """
        import yfinance as yf

        def _fetch() -> Dict[str, Dict]:
            data = yf.download(
                symbols, period="2d", progress=False,
                auto_adjust=True, group_by="ticker",
            )
            if data.empty:
                return {}

            quotes: Dict[str, Dict] = {}
            for symbol in symbols:
                try:
                    df = data[symbol] if len(symbols) > 1 else data
                    df = df.dropna()
                    if df.empty:
                        continue
                    last = df.iloc[-1]
                    prev_close = float(df.iloc[-2]["Close"]) if len(df) > 1 else float(last["Close"])
                    price = float(last["Close"])
                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
                    high, low = float(last["High"]), float(last["Low"])
                    quotes[symbol] = {
                        "symbol": symbol,
                        "price": price,
                        "bid": price,
                        "ask": price,
                        "volume": float(last["Volume"]),
                        "high": high,
                        "low": low,
                        "open": float(last["Open"]),
                        "prev_close": prev_close,
                        "change_pct": round(change_pct, 2),
                        "timestamp": str(df.index[-1]),
                        # No true VWAP in daily bars; the typical price is the
                        # standard proxy and is what the momentum check needs.
                        "vwap": round((high + low + price) / 3, 4),
                        "source": "yfinance",
                    }
                except (KeyError, IndexError, ValueError):
                    continue
            return quotes

        loop = asyncio.get_event_loop()
        quotes = await loop.run_in_executor(None, _fetch)
        if not quotes:
            raise RuntimeError(
                f"No quote data for {len(symbols)} symbols from Alpaca or yfinance. "
                "Check your network connection."
            )
        self._price_cache.update(quotes)
        logger.info(f"Fetched {len(quotes)} quotes from yfinance")
        return quotes

    async def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a single symbol."""
        if symbol in self._price_cache:
            cache_time = self._last_fetch.get(symbol)
            if cache_time and (datetime.utcnow() - cache_time).seconds < 10:
                return self._price_cache[symbol]["price"]

        quotes = await self.get_latest_quotes([symbol])
        self._last_fetch[symbol] = datetime.utcnow()
        if symbol in quotes:
            return quotes[symbol]["price"]
        raise RuntimeError(f"No quote data for {symbol}")

    # ------------------------------------------------------------------
    # Historical bars via Alpaca
    # ------------------------------------------------------------------
    async def get_historical_bars(
        self,
        symbol: str,
        days: int = 60,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch historical bars from Alpaca."""
        cache_key = f"{symbol}_{days}_{timeframe}"
        if cache_key in self._history_cache:
            cache_time = self._last_fetch.get(cache_key)
            if cache_time and (datetime.utcnow() - cache_time).seconds < 300:
                return self._history_cache[cache_key]

        if not ALPACA_API_KEY:
            return await self._yfinance_fallback(symbol, days)

        end = datetime.utcnow()
        start = end - timedelta(days=days)

        url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
        params = {
            "start": start.strftime("%Y-%m-%dT00:00:00Z"),
            "end": end.strftime("%Y-%m-%dT00:00:00Z"),
            "timeframe": timeframe,
            "feed": "iex",
            "limit": 1000,
        }

        try:
            resp = await self._client.get(url, headers=self._alpaca_headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            bars = data.get("bars", [])
            if not bars:
                logger.warning(f"No bars for {symbol}, trying yfinance fallback")
                return await self._yfinance_fallback(symbol, days)

            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df = df.rename(columns={
                "t": "timestamp", "o": "open", "h": "high",
                "l": "low", "c": "close", "v": "volume", "vw": "vwap",
            })
            df = df.set_index("timestamp").sort_index()

            self._history_cache[cache_key] = df
            self._last_fetch[cache_key] = datetime.utcnow()
            return df

        except Exception as e:
            logger.warning(f"Alpaca bars failed for {symbol}: {e}, trying yfinance")
            return await self._yfinance_fallback(symbol, days)

    async def _yfinance_fallback(self, symbol: str, days: int) -> pd.DataFrame:
        """Fallback to yfinance for historical data."""
        import yfinance as yf

        def _fetch():
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            df.columns = [c.lower() for c in df.columns]
            return df

        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, _fetch)
        if df.empty:
            raise RuntimeError(
                f"No historical data for {symbol} from any source. "
                "Check API keys and network."
            )
        return df

    # ------------------------------------------------------------------
    # Computed indicators
    # ------------------------------------------------------------------
    @staticmethod
    def compute_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def compute_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        df = df.copy()
        df["bb_mid"] = df["close"].rolling(window=period).mean()
        df["bb_std"] = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_mid"] + std * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - std * df["bb_std"]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        return df

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_volume_anomaly(volume: pd.Series, lookback: int = 20) -> pd.Series:
        """Returns z-score of current volume vs rolling mean."""
        mean = volume.rolling(window=lookback).mean()
        std = volume.rolling(window=lookback).std()
        return (volume - mean) / std.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Alpaca account info
    # ------------------------------------------------------------------
    async def get_account(self) -> Dict:
        """Get Alpaca paper trading account info."""
        url = f"{ALPACA_BASE_URL}/v2/account"
        resp = await self._client.get(url, headers=self._alpaca_headers)
        resp.raise_for_status()
        return resp.json()

    async def get_positions(self) -> List[Dict]:
        """Get current Alpaca positions."""
        url = f"{ALPACA_BASE_URL}/v2/positions"
        resp = await self._client.get(url, headers=self._alpaca_headers)
        resp.raise_for_status()
        return resp.json()

    async def get_orders(self, status: str = "all", limit: int = 50) -> List[Dict]:
        """Get recent orders."""
        url = f"{ALPACA_BASE_URL}/v2/orders"
        params = {"status": status, "limit": limit, "direction": "desc"}
        resp = await self._client.get(url, headers=self._alpaca_headers, params=params)
        resp.raise_for_status()
        return resp.json()


# Singleton
market_data = MarketDataFeed()
