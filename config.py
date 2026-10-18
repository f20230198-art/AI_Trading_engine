"""
DOOM ENGINE — Configuration
All API keys loaded from environment variables. No hardcoded secrets.
"""
import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

# Load local .env so config works even when imported outside main.py.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# ---------------------------------------------------------------------------
# API Keys — set these as environment variables before running
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Ollama — local LLM, zero API cost, no rate limits
# Set LLM_PROVIDER=ollama in .env, or leave auto-detect on (checked at startup)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "mistral")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

# Groq keys — rotated automatically when one hits rate limits
GROQ_API_KEYS: list = [
    k for k in [
        os.environ.get("GROQ_API_KEY_1", ""),
        os.environ.get("GROQ_API_KEY_2", ""),
        os.environ.get("GROQ_API_KEY_3", ""),
    ] if k
]

# Gemini keys — used as fallback pool
GEMINI_API_KEYS: list = [
    k for k in [
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_API_KEY_2", ""),
        os.environ.get("GEMINI_API_KEY_3", ""),
    ] if k
]

# Alpaca base URLs (paper trading)
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# LLM Model
_DEFAULT_MODEL_BY_PROVIDER = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-sonnet-20241022",
    "ollama": OLLAMA_MODEL,
    "quant": "quant-fallback",
}
LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    _DEFAULT_MODEL_BY_PROVIDER.get(LLM_PROVIDER, _DEFAULT_MODEL_BY_PROVIDER["groq"]),
)
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))

# Analyst agent (text-to-SQL). Separate from LLM_MODEL because SQL generation
# benefits from a stronger model than the per-cycle trading agents need.
ANALYST_LLM_MODEL = os.environ.get("ANALYST_LLM_MODEL", "openai/gpt-oss-120b")

# Primary key for legacy single-key checks
ACTIVE_LLM_API_KEY = (GROQ_API_KEYS + GEMINI_API_KEYS + [ANTHROPIC_API_KEY])[0] if (GROQ_API_KEYS or GEMINI_API_KEYS or ANTHROPIC_API_KEY) else ""

# ---------------------------------------------------------------------------
# Watchlist — 20+ symbols across sectors
# ---------------------------------------------------------------------------
WATCHLIST_TECH = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSM", "AVGO"]
WATCHLIST_ENERGY = ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"]
WATCHLIST_FINANCIALS = ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "SCHW"]
WATCHLIST_ALL = WATCHLIST_TECH + WATCHLIST_ENERGY + WATCHLIST_FINANCIALS

# ---------------------------------------------------------------------------
# Trading Parameters
# ---------------------------------------------------------------------------
@dataclass
class TradingConfig:
    # Portfolio
    initial_capital: float = 100_000.0

    # Risk limits
    max_position_pct: float = 0.05       # 5% max per position
    max_sector_pct: float = 0.20          # 20% max sector concentration
    max_daily_drawdown_pct: float = 0.03  # 3% max daily drawdown

    # Rebalance
    rebalance_interval_minutes: int = 30
    rebalance_drift_threshold: float = 0.05  # 5% drift triggers rebalance

    # Cycle timing
    cycle_interval_seconds: int = int(os.environ.get("CYCLE_INTERVAL_SECONDS", "120"))

    # Strategy params
    ema_fast: int = 12
    ema_slow: int = 26
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # Regime detection
    regime_lookback_days: int = 60
    hmm_n_states: int = 4

    # Agent memory
    memory_top_k: int = 3  # retrieve top-3 similar past situations


TRADING_CONFIG = TradingConfig()

# ---------------------------------------------------------------------------
# Sector mappings
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    "technology": WATCHLIST_TECH,
    "energy": WATCHLIST_ENERGY,
    "financials": WATCHLIST_FINANCIALS,
}

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")

try:
    API_PORT = int(os.environ.get("API_PORT", "8000"))
except ValueError:
    API_PORT = 8000
