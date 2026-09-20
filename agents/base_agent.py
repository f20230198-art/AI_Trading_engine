"""
DOOM ENGINE — Base Agent
All agents inherit from this. Provides LLM calling, memory integration,
regime-awareness, and standard lifecycle.

LLM resolution order (automatic, no config needed):
  1. Ollama  — if running locally (http://localhost:11434), zero cost, no limits
  2. Groq    — free tier with key rotation across all configured keys
  3. Gemini  — free tier fallback pool
  4. Quant   — pure deterministic fallback, always works, no API needed

Each agent also has a _quant_fallback() method that returns a valid response
using only the market data already in context — no network calls at all.
"""
import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from config import (
    ACTIVE_LLM_API_KEY,
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    TRADING_CONFIG,
    GROQ_API_KEYS,
    GEMINI_API_KEYS,
    ANTHROPIC_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from core.memory_store import memory_store
from core.shared_state import shared_state

logger = logging.getLogger("doom.agent")

# ---------------------------------------------------------------------------
# Resolved active provider — set once at startup by _resolve_provider()
# Priority: ollama > groq > gemini > anthropic > quant
# ---------------------------------------------------------------------------
_active_provider: str = ""   # set by _resolve_provider() called from initialize()
_ollama_available: Optional[bool] = None  # cached after first check


async def _check_ollama() -> bool:
    """Ping Ollama once and cache the result."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{OLLAMA_BASE_URL}/api/tags")
            _ollama_available = r.status_code == 200
    except Exception:
        _ollama_available = False
    if _ollama_available:
        logger.info(f"Ollama detected at {OLLAMA_BASE_URL} — using local LLM ({OLLAMA_MODEL})")
    else:
        logger.info("Ollama not running — will use API keys or quant fallback")
    return _ollama_available


async def _resolve_provider() -> str:
    """
    Determine which backend to use, in priority order:
      ollama → groq → gemini → anthropic → quant
    Returns the resolved provider name.
    """
    global _active_provider
    if _active_provider:
        return _active_provider

    # If user explicitly set LLM_PROVIDER=ollama or quant, honour it
    if LLM_PROVIDER == "ollama":
        _active_provider = "ollama"
    elif LLM_PROVIDER == "quant":
        _active_provider = "quant"
    else:
        # Auto-detect: try Ollama first regardless of configured provider
        if await _check_ollama():
            _active_provider = "ollama"
        elif GROQ_API_KEYS:
            _active_provider = "groq"
        elif GEMINI_API_KEYS:
            _active_provider = "gemini"
        elif ANTHROPIC_API_KEY:
            _active_provider = "anthropic"
        else:
            _active_provider = "quant"

    logger.info(f"LLM backend resolved: {_active_provider.upper()}")
    return _active_provider


# ---------------------------------------------------------------------------
# Module-level rate limiter — shared across ALL agents
# Only applied to cloud API calls (Ollama and quant have no limits)
# ---------------------------------------------------------------------------
_rate_lock = asyncio.Lock()
_call_timestamps: list = []          # rolling window of call times
_WINDOW_SECONDS = 70                 # sliding window
_MAX_CALLS_PER_WINDOW = 10           # conservative: 10 calls per 70s across all keys


async def _rate_limited_call(coro):
    """Wrap any LLM call to enforce the global rate limit."""
    async with _rate_lock:
        now = time.monotonic()
        # Drop timestamps older than the window
        while _call_timestamps and now - _call_timestamps[0] > _WINDOW_SECONDS:
            _call_timestamps.pop(0)

        if len(_call_timestamps) >= _MAX_CALLS_PER_WINDOW:
            # Must wait until the oldest call falls out of the window
            wait = _WINDOW_SECONDS - (now - _call_timestamps[0]) + 0.5
            if wait > 0:
                logger.info(f"Rate limiter: {len(_call_timestamps)} calls in window, waiting {wait:.1f}s")
                await asyncio.sleep(wait)
            # Re-prune after sleep
            now = time.monotonic()
            while _call_timestamps and now - _call_timestamps[0] > _WINDOW_SECONDS:
                _call_timestamps.pop(0)

        _call_timestamps.append(time.monotonic())

    return await coro


# Module-level key rotation state (shared across all agents)
_groq_key_index = 0
_gemini_key_index = 0


def _next_groq_key() -> Optional[str]:
    global _groq_key_index
    if not GROQ_API_KEYS:
        return None
    key = GROQ_API_KEYS[_groq_key_index % len(GROQ_API_KEYS)]
    _groq_key_index = (_groq_key_index + 1) % len(GROQ_API_KEYS)
    return key


def _next_gemini_key() -> Optional[str]:
    global _gemini_key_index
    if not GEMINI_API_KEYS:
        return None
    key = GEMINI_API_KEYS[_gemini_key_index % len(GEMINI_API_KEYS)]
    _gemini_key_index = (_gemini_key_index + 1) % len(GEMINI_API_KEYS)
    return key


class BaseAgent(ABC):
    """
    Base class for all DOOM ENGINE agents.
    Every agent makes real LLM API calls, has memory, and adapts to regime.
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.status = "INITIALIZING"
        self.last_action = ""
        self.last_run_time = 0.0
        self.total_calls = 0
        self.errors = 0
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """
        Initialize the agent.
        Resolves the active LLM provider (Ollama / Groq / Gemini / Anthropic / Quant)
        and always succeeds — quant fallback means we never hard-fail.
        """
        provider = await _resolve_provider()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=20.0))
        self.status = "READY"
        await shared_state.update_agent_status(self.name, self._get_status())
        logger.info(f"Agent {self.name} initialized [{provider.upper()}]")

    def _get_status(self) -> Dict:
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "last_action": self.last_action,
            "last_run_time": self.last_run_time,
            "total_calls": self.total_calls,
            "errors": self.errors,
        }

    async def _call_llm(self, system_prompt: str, user_prompt: str,
                        max_tokens: int = LLM_MAX_TOKENS,
                        context: Optional[Dict] = None) -> str:
        """
        Smart LLM dispatcher with automatic fallback chain:
          Ollama (local) → Groq (free API) → Gemini (free API) → Quant (deterministic)

        Never raises — if all LLM backends fail, _quant_fallback(context) is called.
        """
        if not self._client:
            raise RuntimeError(f"Agent {self.name} not initialized")

        provider = await _resolve_provider()
        start = time.time()

        try:
            if provider == "ollama":
                text = await self._call_ollama(system_prompt, user_prompt, max_tokens)
            elif provider == "anthropic":
                text = await self._call_anthropic(system_prompt, user_prompt, max_tokens)
            elif provider in ("groq", "gemini"):
                text = await self._call_with_rotation(system_prompt, user_prompt, max_tokens)
            else:
                # quant — skip LLM entirely
                raise RuntimeError("quant-only mode")

            self.total_calls += 1
            self.last_run_time = round(time.time() - start, 2)
            return text

        except Exception as e:
            logger.warning(
                f"Agent {self.name}: LLM backend '{provider}' failed ({e}). "
                "Falling back to next available backend."
            )
            # Try the full rotation (covers groq+gemini together) if not already tried
            if provider == "ollama" and (GROQ_API_KEYS or GEMINI_API_KEYS):
                try:
                    text = await self._call_with_rotation(system_prompt, user_prompt, max_tokens)
                    self.total_calls += 1
                    self.last_run_time = round(time.time() - start, 2)
                    return text
                except Exception as e2:
                    logger.warning(f"Agent {self.name}: API rotation also failed ({e2}). Using quant fallback.")

            # Final fallback: deterministic quant logic
            logger.info(f"Agent {self.name}: activating quant fallback")
            return self._quant_fallback(context or {})

    async def _call_ollama(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """Call Ollama local API (OpenAI-compatible /api/chat endpoint)."""
        if not self._client:
            raise RuntimeError("HTTP client not initialized")
        response = await self._client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.3},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Ollama returned empty content")
        return content

    def _parse_json(self, response: str) -> dict:
        """
        Robustly extract JSON from an LLM response.
        Handles: raw JSON, ```json...```, ```...```, JSON embedded in prose.
        Returns empty dict on total failure instead of raising.
        """
        import json as _json

        # 1. Try fenced code block: ```json ... ``` or ``` ... ```
        fenced = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if fenced:
            try:
                return _json.loads(fenced.group(1))
            except _json.JSONDecodeError:
                pass

        # 2. Try the largest {...} block in the response
        brace_match = re.search(r'\{[\s\S]*\}', response)
        if brace_match:
            try:
                return _json.loads(brace_match.group(0))
            except _json.JSONDecodeError:
                # Try progressively shorter trailing substrings (handles truncation)
                text = brace_match.group(0)
                for end in range(len(text) - 1, len(text) // 2, -1):
                    if text[end] in ('}', ']'):
                        try:
                            return _json.loads(text[:end + 1])
                        except _json.JSONDecodeError:
                            continue

        logger.warning(f"Agent {self.name}: could not parse JSON from LLM response: {response[:120]!r}")
        return {}

    def _quant_fallback(self, context: Dict) -> str:
        """
        Deterministic quant fallback — returns a valid JSON string using only
        the market data already in context. No network calls. Always succeeds.
        Subclasses should override this with agent-specific logic.
        """
        import json, random
        regime = shared_state.current_regime or "RANGING"

        # Generic fallback — subclasses override for better responses
        result = {
            "market_condition": f"Quant mode active. Regime: {regime}.",
            "signals": [],
            "sector_heat": {"technology": 0.0, "energy": 0.0, "financials": 0.0},
            "top_opportunities": [],
            "risk_flags": ["Operating in quant-only fallback mode"],
            # Strategy fields
            "strategy_assessment": f"Quant fallback active in {regime} regime.",
            "sub_strategy_signals": {"momentum": [], "mean_reversion": [], "sentiment_spike": []},
            "trade_proposals": [],
            "no_trade_symbols": [],
            # Orchestrator fields
            "final_decisions": [],
            "conflict_verdicts": [],
            "doom_speaks": (
                f"QUANT MODE ACTIVE — {regime} regime detected. "
                "All signals computed deterministically. Maintaining risk-adjusted posture. "
                "Awaiting LLM restoration for full autonomous reasoning."
            ),
            "cycle_assessment": f"Quant fallback cycle. Regime={regime}.",
            "next_cycle_focus": "Monitor regime shifts and LLM availability.",
            # Sentiment fields
            "overall_market_sentiment": 0.0,
            "sentiment_summary": "Neutral — quant fallback mode.",
            "sentiment_spikes": [],
            "news_events": [],
            # Risk fields
            "risk_score": 30,
            "risk_label": "LOW",
            # Must match the key risk_guardian actually reads. Left empty on
            # purpose — _enforce_hard_rules() synthesizes one entry per proposal.
            "reviewed_proposals": [],
            "portfolio_risk_summary": "Quant risk assessment: conservative posture.",
            # Portfolio fields
            "portfolio_health": "NOMINAL",
            "sector_concentration": {},
            "rebalance_needed": False,
            "recommended_adjustments": [],
            "portfolio_assessment": "Quant fallback: no rebalancing triggered.",
        }
        return json.dumps(result)

    async def _call_with_rotation(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """
        Try all Groq keys first, then all Gemini keys.
        Goes through the global rate limiter before each attempt.
        On 429, waits briefly then tries the next key.
        """
        # Build ordered list: all Groq keys then all Gemini keys
        candidates = []
        for key in GROQ_API_KEYS:
            candidates.append(("groq", key))
        for key in GEMINI_API_KEYS:
            candidates.append(("gemini", key))

        if not candidates:
            raise RuntimeError("No LLM API keys configured")

        last_error = None
        for i, (provider, key) in enumerate(candidates):
            try:
                if provider == "groq":
                    result = await _rate_limited_call(
                        self._call_groq_key(system_prompt, user_prompt, max_tokens, key)
                    )
                else:
                    result = await _rate_limited_call(
                        self._call_gemini_key(system_prompt, user_prompt, max_tokens, key)
                    )
                return result
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning(
                        f"Agent {self.name}: {provider} key ...{key[-6:]} still rate limited "
                        f"(attempt {i+1}/{len(candidates)}), waiting 12s then trying next"
                    )
                    last_error = e
                    await asyncio.sleep(12)
                    continue
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"Agent {self.name}: {provider} key ...{key[-6:]} failed: {e}, trying next")
                continue

        raise RuntimeError(f"All LLM keys exhausted. Last error: {last_error}")

    async def _call_groq_key(self, system_prompt: str, user_prompt: str, max_tokens: int, key: str) -> str:
        """Call Groq OpenAI-compatible API with a specific key."""
        if not self._client:
            raise RuntimeError("LLM HTTP client not initialized")

        response = await self._client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("Groq returned no choices")
        return choices[0]["message"]["content"]

    async def _call_gemini_key(self, system_prompt: str, user_prompt: str, max_tokens: int, key: str) -> str:
        """Call Gemini API with a specific key."""
        if not self._client:
            raise RuntimeError("LLM HTTP client not initialized")

        gemini_model = "gemini-2.0-flash"
        response = await self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent",
            params={"key": key},
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            block_reason = payload.get("promptFeedback", {}).get("blockReason", "unknown")
            raise RuntimeError(f"Gemini returned no candidates (blockReason={block_reason})")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts if p.get("text")]
        if not text_parts:
            raise RuntimeError("Gemini returned candidate without text content")
        return "\n".join(text_parts)

    async def _call_anthropic(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        if not self._client:
            raise RuntimeError("LLM HTTP client not initialized")

        response = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )
        response.raise_for_status()
        payload = response.json()
        for block in payload.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                return block["text"]
        raise RuntimeError("Anthropic returned no text content")

    def _get_regime_context(self) -> str:
        """Build regime-aware context string for agent prompts."""
        regime = shared_state.current_regime
        context = f"CURRENT MARKET REGIME: {regime}\n"

        if regime == "TRENDING":
            context += (
                "The market is in a TRENDING regime. Momentum strategies should be prioritized. "
                "Trend-following signals carry higher conviction. Mean reversion signals should "
                "be treated with caution."
            )
        elif regime == "RANGING":
            context += (
                "The market is in a RANGING/SIDEWAYS regime. Mean reversion strategies are favored. "
                "Bollinger Band squeeze plays and range-bound strategies should be prioritized. "
                "Momentum signals may produce whipsaws — reduce position sizes."
            )
        elif regime == "HIGH_VOLATILITY":
            context += (
                "The market is in HIGH VOLATILITY mode. Increase caution. Reduce position sizes by 25%. "
                "Focus on liquid names only. Widen stop-losses to avoid being shaken out. "
                "Short-term momentum may work but risk management is critical."
            )
        elif regime == "CRISIS":
            context += (
                "CRISIS MODE ACTIVE. Maximum caution. Risk Guardian has veto authority on ALL trades. "
                "Position sizes should be halved. Only high-conviction trades with strong risk/reward. "
                "Consider defensive positioning and hedges."
            )
        else:
            context += "Regime is being computed. Use balanced default parameters."

        return context

    async def _recall_memories(self, situation: str) -> str:
        """Retrieve relevant past decisions for context."""
        memories = await memory_store.recall_similar(
            agent_name=self.name,
            current_situation=situation,
            top_k=TRADING_CONFIG.memory_top_k,
        )

        if not memories:
            return "No relevant past decisions found in memory."

        context = "RELEVANT PAST DECISIONS:\n"
        for i, mem in enumerate(memories, 1):
            context += f"\n--- Memory {i} ---\n{mem['content']}\n"
            if mem.get("metadata", {}).get("outcome"):
                context += f"Outcome: {mem['metadata']['outcome']}\n"

        return context

    async def _store_decision(self, situation: str, decision: str, metadata: Optional[Dict] = None) -> str:
        """Store this decision in memory for future recall."""
        return await memory_store.store_decision(
            agent_name=self.name,
            situation=situation,
            decision=decision,
            metadata=metadata,
        )

    @abstractmethod
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one cycle of this agent. Must be implemented by subclasses."""
        pass

    async def update_status(self, status: str, action: str = ""):
        self.status = status
        if action:
            self.last_action = action
        await shared_state.update_agent_status(self.name, self._get_status())
