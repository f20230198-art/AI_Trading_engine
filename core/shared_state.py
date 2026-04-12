"""
DOOM ENGINE — Shared State
Central state object that all agents read/write and the dashboard subscribes to.
Thread-safe, JSON-serializable snapshot for WebSocket broadcast.
"""
import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("doom.state")


class SharedState:
    """
    Single source of truth for the entire DOOM ENGINE.
    Agents write here; the backend reads and broadcasts to the dashboard.
    """

    def __init__(self):
        self._lock = asyncio.Lock()

        # Agent statuses
        self.agent_statuses: Dict[str, Dict] = {}

        # Current market data snapshot
        self.market_quotes: Dict[str, Dict] = {}

        # Live broker positions (for real-time unrealized P&L display)
        self.live_positions: List[Dict] = []

        # Regime
        self.current_regime: str = "INITIALIZING"
        self.regime_color: str = "#ffffff"
        self.regime_probs: Dict[str, float] = {}

        # Trade proposals and decisions
        self.pending_proposals: List[Dict] = []
        self.recent_decisions: deque = deque(maxlen=50)

        # DOOM SPEAKS — Orchestrator briefings
        self.doom_speaks: deque = deque(maxlen=20)
        self.latest_briefing: str = "DOOM ENGINE initializing. Awaiting first cycle..."

        # Conflict log
        self.recent_conflicts: deque = deque(maxlen=50)

        # Agent messages (inter-agent comms for the network visualization)
        self.agent_messages: deque = deque(maxlen=100)

        # Cycle counter
        self.cycle_count: int = 0
        self.last_cycle_time: Optional[str] = None
        self.system_status: str = "INITIALIZING"

        # Control plane state (dashboard commands)
        self.trading_paused: bool = False
        self.emergency_stop: bool = False
        self.manual_strike_requested: bool = False
        self.last_control_event: Optional[str] = None

        # Risk score (0-100)
        self.risk_score: float = 0.0
        self.risk_label: str = "LOW"

        # Sentiment scores
        self.sentiment_scores: Dict[str, float] = {}

        # Trade / metrics cache — updated by performance_tracker calls via api/main
        self.recent_trades: List[Dict] = []
        self.open_trades: List[Dict] = []
        self.metrics: Dict[str, Any] = {}
        self.equity_curve: List[Dict] = []

        # WebSocket subscribers
        self._subscribers: List[asyncio.Queue] = []

    async def update_agent_status(self, agent_name: str, status: Dict):
        async with self._lock:
            self.agent_statuses[agent_name] = {
                **status,
                "last_update": datetime.utcnow().isoformat(),
            }
        await self._notify()

    async def update_market_data(self, quotes: Dict[str, Dict]):
        async with self._lock:
            self.market_quotes = quotes
        await self._notify()

    async def update_live_positions(self, positions: List[Dict]):
        async with self._lock:
            self.live_positions = positions
        await self._notify()

    async def update_regime(self, regime: str, color: str, probs: Dict[str, float]):
        async with self._lock:
            self.current_regime = regime
            self.regime_color = color
            self.regime_probs = probs
        await self._notify()

    async def add_doom_speaks(self, briefing: str):
        async with self._lock:
            self.latest_briefing = briefing
            self.doom_speaks.append({
                "text": briefing,
                "timestamp": datetime.utcnow().isoformat(),
            })
        await self._notify()

    async def add_conflict(self, conflict: Dict):
        async with self._lock:
            self.recent_conflicts.append(conflict)
        await self._notify()

    async def add_agent_message(self, from_agent: str, to_agent: str, msg_type: str, summary: str):
        async with self._lock:
            self.agent_messages.append({
                "from": from_agent,
                "to": to_agent,
                "type": msg_type,
                "summary": summary,
                "timestamp": datetime.utcnow().isoformat(),
            })
        await self._notify()

    async def update_risk(self, score: float, label: str):
        async with self._lock:
            self.risk_score = score
            self.risk_label = label
        await self._notify()

    async def update_sentiment(self, scores: Dict[str, float]):
        async with self._lock:
            self.sentiment_scores.update(scores)
        await self._notify()

    async def increment_cycle(self):
        async with self._lock:
            self.cycle_count += 1
            self.last_cycle_time = datetime.utcnow().isoformat()
            if self.emergency_stop:
                self.system_status = "EMERGENCY_STOP"
            elif self.trading_paused:
                self.system_status = "PAUSED"
            else:
                self.system_status = "ACTIVE"
        await self._notify()

    async def set_trading_paused(self, paused: bool):
        async with self._lock:
            self.trading_paused = paused
            if self.emergency_stop:
                self.system_status = "EMERGENCY_STOP"
            elif paused:
                self.system_status = "PAUSED"
            else:
                self.system_status = "ACTIVE"
            self.last_control_event = datetime.utcnow().isoformat()
        await self._notify()

    async def set_emergency_stop(self, enabled: bool):
        async with self._lock:
            self.emergency_stop = enabled
            if enabled:
                self.trading_paused = True
                self.system_status = "EMERGENCY_STOP"
            else:
                self.system_status = "PAUSED" if self.trading_paused else "ACTIVE"
            self.last_control_event = datetime.utcnow().isoformat()
        await self._notify()

    async def request_manual_strike(self):
        async with self._lock:
            self.manual_strike_requested = True
            self.last_control_event = datetime.utcnow().isoformat()
        await self._notify()

    async def consume_manual_strike(self) -> bool:
        async with self._lock:
            if self.manual_strike_requested:
                self.manual_strike_requested = False
                return True
        return False

    async def update_trade_state(
        self,
        recent_trades: List[Dict],
        open_trades: List[Dict],
        metrics: Dict[str, Any],
        equity_curve: List[Dict],
    ):
        """Called after any trade event to push updated trade data via WebSocket."""
        async with self._lock:
            self.recent_trades = recent_trades
            self.open_trades = open_trades
            self.metrics = metrics
            self.equity_curve = equity_curve
        await self._notify()

    # ------------------------------------------------------------------
    # WebSocket subscription
    # ------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=10)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def _notify(self):
        """Push state snapshot to all WebSocket subscribers."""
        snapshot = self.get_snapshot()
        dead = []
        for q in self._subscribers:
            try:
                # Non-blocking put — drop if queue is full (slow consumer)
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass
            except Exception:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def get_snapshot(self) -> Dict[str, Any]:
        """Full JSON-serializable snapshot for the dashboard."""
        return {
            "type": "state_update",
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": self.system_status,
            "cycle_count": self.cycle_count,
            "last_cycle_time": self.last_cycle_time,
            "agents": self.agent_statuses,
            "controls": {
                "paused": self.trading_paused,
                "emergency_stop": self.emergency_stop,
                "manual_strike_requested": self.manual_strike_requested,
                "last_control_event": self.last_control_event,
            },
            "regime": {
                "current": self.current_regime,
                "color": self.regime_color,
                "probabilities": self.regime_probs,
            },
            "risk": {
                "score": self.risk_score,
                "label": self.risk_label,
            },
            "doom_speaks": {
                "latest": self.latest_briefing,
                "history": list(self.doom_speaks),
            },
            "conflicts": list(self.recent_conflicts)[-10:],
            "agent_messages": list(self.agent_messages)[-20:],
            "sentiment": self.sentiment_scores,
            "market_quotes": self.market_quotes,
            "live_positions": self.live_positions,
            "recent_trades": self.recent_trades,
            "open_trades": self.open_trades,
            "metrics": self.metrics,
            "equity_curve": self.equity_curve,
        }


# Singleton
shared_state = SharedState()
