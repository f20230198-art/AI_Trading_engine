"""
DOOM ENGINE — Main Entry Point
Starts the orchestration loop and the FastAPI backend.
Run: python main.py
"""
import asyncio
import logging
import os
import sys
import signal
import socket
import time
from datetime import datetime

import uvicorn
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment variables
load_dotenv()

from config import (
    ACTIVE_LLM_API_KEY, ALPACA_API_KEY, LLM_PROVIDER, TRADING_CONFIG,
    API_HOST, API_PORT, WATCHLIST_ALL, OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_MODEL,
)
from core.shared_state import shared_state
from core.data_feeds import market_data
from core.regime_detector import regime_detector
from core.memory_store import memory_store
from core.conflict_logger import conflict_logger
from core.performance_tracker import performance_tracker
from core.risk_enforcer import apply_risk_enforcement

from agents.market_intel import MarketIntelAgent
from agents.sector_combined import SectorCombinedAgent
from agents.sentiment import SentimentAgent
from agents.strategy import StrategyAgent
from agents.risk_guardian import RiskGuardianAgent
from agents.portfolio import PortfolioAgent
from agents.execution import ExecutionAgent
from agents.orchestrator import OrchestratorAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(_data_dir, exist_ok=True)

import io
_stream_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
_stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler(os.path.join(_data_dir, "doom.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("doom.main")


class DoomEngine:
    """The main orchestration loop for the DOOM ENGINE."""

    def __init__(self):
        # Initialize all agents
        self.market_intel = MarketIntelAgent()
        self.sectors = SectorCombinedAgent()
        self.sentiment = SentimentAgent()
        self.strategy = StrategyAgent()
        self.risk_guardian = RiskGuardianAgent()
        self.portfolio = PortfolioAgent()
        self.execution = ExecutionAgent()
        self.orchestrator = OrchestratorAgent()

        self._running = False
        self._cycle = 0

    async def initialize(self):
        """Initialize all subsystems."""
        logger.info("=" * 60)
        logger.info("  DOOM ENGINE — INITIALIZING")
        logger.info("=" * 60)

        # LLM backend selection — logged but never fatal.
        # Priority: Ollama (local) → Groq → Gemini → Anthropic → Quant fallback
        if not ACTIVE_LLM_API_KEY:
            logger.warning(
                "No cloud LLM API key found. "
                "Will auto-detect Ollama; if not running, quant fallback activates. "
                "Set GROQ_API_KEY_1, GEMINI_API_KEY, or ANTHROPIC_API_KEY for cloud LLM."
            )
        else:
            logger.info(f"LLM_PROVIDER={LLM_PROVIDER} — cloud API key found.")

        if not ALPACA_API_KEY:
            logger.warning(
                "ALPACA_API_KEY not set. Market data and trading will be unavailable. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY for full functionality."
            )

        # Create data directory
        os.makedirs(os.path.join(os.path.dirname(__file__), "data", "memory"), exist_ok=True)

        # Initialize core systems
        await market_data.initialize()
        await memory_store.initialize()
        await conflict_logger.initialize()
        await performance_tracker.initialize(TRADING_CONFIG.initial_capital)

        # Initialize all agents
        agents = [
            self.market_intel, self.sectors, self.sentiment, self.strategy,
            self.risk_guardian, self.portfolio, self.execution,
            self.orchestrator,
        ]
        for agent in agents:
            try:
                await agent.initialize()
                logger.info(f"  Agent {agent.name}: READY")
            except Exception as e:
                logger.error(f"  Agent {agent.name}: FAILED — {e}")

        await shared_state.add_doom_speaks(
            "DOOM ENGINE online. All systems initialized. "
            "Beginning autonomous trading operations. Stay sharp."
        )

        # Show resolved LLM backend after all agents have initialized
        from agents.base_agent import _active_provider
        logger.info("=" * 60)
        logger.info("  DOOM ENGINE — ALL SYSTEMS NOMINAL")
        logger.info(f"  LLM BACKEND  : {_active_provider.upper()}")
        if _active_provider == "ollama":
            logger.info(f"  OLLAMA MODEL : {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
        elif _active_provider == "quant":
            logger.info("  MODE         : Deterministic quant fallback (no LLM)")
        else:
            logger.info(f"  CLOUD MODEL  : {LLM_MODEL}")
        logger.info("=" * 60)

    async def run_cycle(self):
        """Execute one full trading cycle."""
        self._cycle += 1
        cycle_start = time.time()

        logger.info(f"\n{'='*50}")
        logger.info(f"  CYCLE {self._cycle} — {datetime.utcnow().isoformat()}")
        logger.info(f"{'='*50}")

        await shared_state.increment_cycle()

        # Equity starts unknown and is filled from the live Alpaca account below.
        # Seeding it with initial_capital would size real orders against a
        # fictional $100k book whenever the account could not be read.
        context = {"cycle": self._cycle, "equity": None}

        try:
            # Phase 1: Data Gathering (parallel)
            # Run market intel + sentiment + regime detection in parallel
            logger.info("Phase 1: Data gathering...")

            market_task = asyncio.create_task(self._safe_run(self.market_intel, context))
            sentiment_task = asyncio.create_task(self._safe_run(self.sentiment, context))

            market_result = await market_task
            sentiment_result = await sentiment_task

            # Update context with results
            context["market_intel"] = market_result or {}
            context["sentiment"] = sentiment_result or {}
            context["quotes"] = (market_result or {}).get("quotes", {})

            # Phase 1b: Regime detection (needs market data)
            logger.info("Phase 1b: Regime detection...")
            await self._detect_regime(context)

            # Phase 2: Sector Analysis (single combined call)
            logger.info("Phase 2: Sector analysis...")
            sectors_result = await self._safe_run(self.sectors, context)
            if sectors_result:
                context["sector_tech"] = sectors_result.get("sector_tech", {})
                context["sector_energy"] = sectors_result.get("sector_energy", {})
                context["sector_financials"] = sectors_result.get("sector_financials", {})
            else:
                context["sector_tech"] = {}
                context["sector_energy"] = {}
                context["sector_financials"] = {}

            # Phase 3: Strategy Generation
            logger.info("Phase 3: Strategy generation...")
            strategy_result = await self._safe_run(self.strategy, context)
            context["strategy"] = strategy_result or {}

            # Phase 4: Risk Review + Portfolio Check (parallel)
            logger.info("Phase 4: Risk review + portfolio check...")
            context["proposals"] = (strategy_result or {}).get("proposals", [])

            # Get current positions for risk assessment
            try:
                if ALPACA_API_KEY:
                    positions = await self.execution.get_positions()
                    context["positions"] = positions
                    await shared_state.update_live_positions(positions)
                    account = await self.execution.get_account_info()
                    context["equity"] = float(account.get("equity", TRADING_CONFIG.initial_capital))

                    # Snapshot equity — augment with estimated open-trade value
                    # when Alpaca positions_value is 0 but we have tracked open trades
                    alpaca_equity = context["equity"]
                    positions_value = float(account.get("long_market_value", 0))
                    cash = float(account.get("cash", 0))

                    # If Alpaca shows no positions but we have open trades, estimate
                    if positions_value == 0:
                        open_trades = performance_tracker.get_open_trades()
                        quotes = context.get("quotes", {})
                        estimated_pv = 0.0
                        for t in open_trades:
                            sym = t.get("symbol", "")
                            qty = float(t.get("qty") or 0)
                            ep = float(t.get("entry_price") or 0)
                            if qty > 0 and ep > 0:
                                # Use live quote if available, otherwise entry price
                                current_price = ep
                                q = quotes.get(sym)
                                if q and isinstance(q, dict):
                                    current_price = float(q.get("price") or q.get("last") or ep)
                                if t.get("side") == "buy":
                                    estimated_pv += current_price * qty
                        if estimated_pv > 0:
                            positions_value = estimated_pv
                            alpaca_equity = cash + estimated_pv

                    await performance_tracker.snapshot_equity(
                        equity=alpaca_equity,
                        cash=cash,
                        positions_value=positions_value,
                    )
                else:
                    # No broker connected. Fall back to the tracked paper book so
                    # the full pipeline — sizing, risk limits, veto enforcement —
                    # stays observable in a keyless demo. Orders are still never
                    # placed; the Execution agent requires real Alpaca credentials.
                    context["positions"] = []
                    metrics = performance_tracker.get_dashboard_metrics()
                    context["equity"] = metrics.get(
                        "current_equity", TRADING_CONFIG.initial_capital
                    )
                    context["simulated_equity"] = True
                    await shared_state.update_live_positions([])
            except Exception as e:
                logger.warning(f"Could not fetch positions: {e}")
                context["positions"] = []
                await shared_state.update_live_positions([])

            risk_task = asyncio.create_task(self._safe_run(self.risk_guardian, context))
            portfolio_task = asyncio.create_task(self._safe_run(self.portfolio, context))

            risk_result = await risk_task
            portfolio_result = await portfolio_task

            context["risk_guardian"] = risk_result or {}
            context["portfolio"] = portfolio_result or {}

            # Phase 5: Orchestrator — Final Decisions
            logger.info("Phase 5: Orchestrator decision...")
            orchestrator_result = await self._safe_run(self.orchestrator, context)

            # Phase 5b: Deterministic risk enforcement — config.py limits in plain
            # Python, after the Risk Guardian and before Execution.
            if orchestrator_result and orchestrator_result.get("approved_trades"):
                orchestrator_result["approved_trades"] = await apply_risk_enforcement(
                    trades=orchestrator_result["approved_trades"],
                    reviewed=context["risk_guardian"].get("reviewed", []),
                    equity=context.get("equity"),
                    positions=context.get("positions", []),
                    daily_drawdown_pct=max(
                        float(context["risk_guardian"].get("daily_drawdown_pct") or 0),
                        performance_tracker.compute_daily_drawdown(),
                    ),
                )

            # Phase 6: Execution
            if shared_state.emergency_stop:
                logger.warning("Emergency stop active. Skipping execution phase.")
            elif orchestrator_result and orchestrator_result.get("approved_trades"):
                logger.info("Phase 6: Executing trades...")
                context["approved_trades"] = orchestrator_result["approved_trades"]
                execution_result = await self._safe_run(self.execution, context)
            else:
                logger.info("Phase 6: No trades to execute")

            # Push updated trade/metrics state to all dashboard clients
            await shared_state.update_trade_state(
                recent_trades=performance_tracker.get_recent_trades(50),
                open_trades=performance_tracker.get_open_trades(),
                metrics=performance_tracker.get_dashboard_metrics(),
                equity_curve=performance_tracker.get_equity_curve(200),
            )

            cycle_time = time.time() - cycle_start
            logger.info(f"Cycle {self._cycle} completed in {cycle_time:.1f}s")

        except Exception as e:
            logger.error(f"Cycle {self._cycle} failed: {e}", exc_info=True)
            await shared_state.add_doom_speaks(
                f"WARNING: Cycle {self._cycle} encountered errors. "
                f"Maintaining defensive posture. Investigating: {str(e)[:100]}"
            )

    async def _safe_run(self, agent, context):
        """Run an agent with error handling."""
        try:
            return await agent.run(context)
        except Exception as e:
            logger.error(f"Agent {agent.name} failed: {e}", exc_info=True)
            await agent.update_status("ERROR", str(e)[:100])
            return None

    async def _detect_regime(self, context):
        """Run regime detection on market data."""
        try:
            # Use SPY as the broad market proxy for regime detection
            hist = await market_data.get_historical_bars("SPY", days=60)
            if len(hist) > 30:
                regime = regime_detector.detect(hist)
                state = regime_detector.get_state()
                await shared_state.update_regime(
                    regime=state["current_regime"],
                    color=state["color"],
                    probs=state.get("probabilities", {}),
                )
                logger.info(f"Regime: {regime}")
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")

    async def run_loop(self):
        """Main orchestration loop."""
        await self.initialize()
        self._running = True

        # Carries a strike consumed by the sleep loop into the next iteration,
        # so a strike requested while paused still runs exactly one cycle.
        pending_strike = False

        while self._running:
            try:
                if shared_state.emergency_stop:
                    logger.warning("Emergency stop active. Trading loop halted.")
                else:
                    manual_strike = pending_strike or await shared_state.consume_manual_strike()
                    pending_strike = False
                    if shared_state.trading_paused and not manual_strike:
                        logger.info("Trading paused. Waiting for resume or manual strike.")
                    else:
                        if manual_strike and shared_state.trading_paused:
                            logger.info("Manual strike received while paused. Running one cycle.")
                        await self.run_cycle()
            except Exception as e:
                logger.error(f"Fatal cycle error: {e}", exc_info=True)

            if not self._running:
                break

            # Wait for next cycle
            if shared_state.emergency_stop or shared_state.trading_paused:
                await asyncio.sleep(1)
            else:
                logger.info(f"Sleeping {TRADING_CONFIG.cycle_interval_seconds}s until next cycle...")
                elapsed = 0
                while self._running and elapsed < TRADING_CONFIG.cycle_interval_seconds:
                    # Wake early on control changes so War Room commands feel
                    # immediate. A strike is CONSUMED here rather than merely
                    # observed: the flag is otherwise cleared at the top of the
                    # loop, so breaking on it without clearing it re-triggers on
                    # every subsequent sleep and the engine free-runs, burning
                    # cycles and LLM quota.
                    if shared_state.emergency_stop or shared_state.trading_paused:
                        break
                    if await shared_state.consume_manual_strike():
                        logger.info("Manual strike received — running next cycle now.")
                        pending_strike = True
                        break
                    await asyncio.sleep(1)
                    elapsed += 1

    def stop(self):
        self._running = False
        logger.info("DOOM ENGINE shutting down...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _is_port_available(host: str, port: int) -> bool:
    """Check whether the API port is free before starting background tasks."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


async def main() -> int:
    if not _is_port_available(API_HOST, API_PORT):
        logger.error(
            f"Cannot start API server on {API_HOST}:{API_PORT} because the port is already in use. "
            "Set API_PORT in .env or stop the process currently bound to that port."
        )
        return 1

    engine = DoomEngine()
    engine_task = None
    server_task = None
    server = None

    # Handle shutdown
    def handle_shutdown(sig, frame):
        engine.stop()

    signal.signal(signal.SIGINT, handle_shutdown)

    # Start the engine loop and API server concurrently
    engine_task = asyncio.create_task(engine.run_loop())

    # Start FastAPI
    from backend.api import app
    config = uvicorn.Config(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve(), name="doom-api-server")

    logger.info(f"Dashboard available at http://localhost:{API_PORT}")

    try:
        done, _ = await asyncio.wait(
            {engine_task, server_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in done:
            exc = task.exception()
            if exc is None:
                continue
            if isinstance(exc, SystemExit):
                logger.error(
                    f"API server exited with status {exc.code}. "
                    f"Check whether port {API_PORT} is available."
                )
                return int(exc.code) if isinstance(exc.code, int) else 1
            raise exc

        return 0
    except Exception as e:
        logger.error(f"Fatal runtime error: {e}", exc_info=True)
        return 1
    finally:
        engine.stop()
        if server is not None:
            server.should_exit = True

        pending = [task for task in (engine_task, server_task) if task is not None and not task.done()]
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=10,
                )
            except asyncio.TimeoutError:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            await market_data.close()
        except Exception as e:
            logger.debug(f"Market data close skipped: {e}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
