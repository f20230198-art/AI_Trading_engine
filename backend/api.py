"""
DOOM ENGINE — FastAPI Backend
REST + WebSocket API for the war room dashboard.
All data is live from the running engine — zero static values.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict

import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from core.shared_state import shared_state
from core.performance_tracker import performance_tracker
from core.conflict_logger import conflict_logger
from core.regime_detector import regime_detector
from core.memory_store import memory_store

logger = logging.getLogger("doom.api")

app = FastAPI(
    title="DOOM ENGINE API",
    description="War Room Dashboard Backend — Live Autonomous Trading System",
    version="1.0.0",
)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ManualOrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=12)
    side: str = Field(..., min_length=3, max_length=4)
    qty: int = Field(default=1, ge=1, le=100000)


async def _wait_for_fill_price(order_id: str, timeout_seconds: int = 8):
    """Poll Alpaca briefly for fill price after market order submission."""
    if not order_id:
        return None

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(timeout_seconds):
            try:
                resp = await client.get(
                    f"{ALPACA_BASE_URL}/v2/orders/{order_id}",
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "")
                    if status == "filled":
                        price = data.get("filled_avg_price")
                        return float(price) if price else None
                    if status in ("canceled", "rejected", "expired"):
                        return None
            except Exception:
                pass
            await asyncio.sleep(1)

    return None


# ---------------------------------------------------------------------------
# WebSocket — Real-time state updates
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live dashboard updates."""
    await websocket.accept()
    queue = shared_state.subscribe()
    logger.info("Dashboard client connected via WebSocket")

    try:
        # Send initial full state
        initial = _build_full_state()
        await websocket.send_json(initial)

        while True:
            try:
                # Wait for state updates
                data = await asyncio.wait_for(queue.get(), timeout=5.0)
                await websocket.send_json(data)
            except asyncio.TimeoutError:
                # Send heartbeat / periodic full refresh
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_status": shared_state.system_status,
                })
    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        shared_state.unsubscribe(queue)


# ---------------------------------------------------------------------------
# REST Endpoints — Dashboard data
# ---------------------------------------------------------------------------
@app.get("/api/state")
async def get_full_state():
    """Get complete system state snapshot."""
    return _build_full_state()


@app.get("/api/metrics")
async def get_metrics():
    """Get live performance metrics."""
    return performance_tracker.get_dashboard_metrics()


@app.get("/api/equity-curve")
async def get_equity_curve():
    """Get equity curve data for the chart."""
    return {"curve": performance_tracker.get_equity_curve(200)}


@app.get("/api/trades")
async def get_trades():
    """Get recent trades."""
    return {
        "recent": performance_tracker.get_recent_trades(50),
        "open": performance_tracker.get_open_trades(),
    }


@app.get("/api/conflicts")
async def get_conflicts():
    """Get conflict log."""
    return {
        "recent": conflict_logger.get_recent(30),
    }


@app.get("/api/agents")
async def get_agents():
    """Get all agent statuses."""
    return {"agents": shared_state.agent_statuses}


@app.get("/api/regime")
async def get_regime():
    """Get current market regime."""
    return regime_detector.get_state()


@app.get("/api/doom-speaks")
async def get_doom_speaks():
    """Get DOOM SPEAKS briefings."""
    return {
        "latest": shared_state.latest_briefing,
        "history": list(shared_state.doom_speaks),
    }


@app.get("/api/attribution")
async def get_attribution():
    """Get performance attribution by agent and strategy."""
    return {
        "by_agent": performance_tracker.compute_agent_attribution(),
        "by_strategy": performance_tracker.compute_strategy_attribution(),
    }


@app.get("/api/agent-messages")
async def get_agent_messages():
    """Get recent inter-agent messages for network visualization."""
    return {"messages": list(shared_state.agent_messages)}


@app.get("/api/sentiment")
async def get_sentiment():
    """Get current sentiment scores."""
    return {"scores": shared_state.sentiment_scores}


@app.get("/api/memory-stats")
async def get_memory_stats():
    """Get agent memory statistics."""
    agents = ["market_intel", "sector_tech", "sector_energy", "sector_financials",
              "sentiment", "strategy", "risk_guardian", "portfolio", "orchestrator"]
    stats = {}
    for agent in agents:
        try:
            stats[agent] = await memory_store.get_agent_stats(agent)
        except Exception:
            stats[agent] = {"total_memories": 0}
    return {"memory_stats": stats}


@app.post("/api/control/strike")
async def control_strike():
    """Request one immediate cycle, even when paused."""
    if shared_state.emergency_stop:
        return {"ok": False, "message": "Emergency stop is active. Clear it before executing strike."}
    await shared_state.request_manual_strike()
    await shared_state.add_doom_speaks(
        "WAR ROOM COMMAND RECEIVED: Execute Strike. Running one immediate cycle."
    )
    return {"ok": True, "message": "Manual strike requested."}


@app.post("/api/control/pause")
async def control_pause():
    """Pause autonomous cycle execution."""
    if shared_state.emergency_stop:
        return {"ok": False, "message": "Emergency stop is active. Cannot pause separately."}
    await shared_state.set_trading_paused(True)
    await shared_state.add_doom_speaks("WAR ROOM COMMAND RECEIVED: Trading loop paused.")
    return {"ok": True, "message": "Trading paused."}


@app.post("/api/control/resume")
async def control_resume():
    """Resume autonomous cycle execution."""
    if shared_state.emergency_stop:
        return {"ok": False, "message": "Emergency stop is active. Clear it before resume."}
    await shared_state.set_trading_paused(False)
    await shared_state.add_doom_speaks("WAR ROOM COMMAND RECEIVED: Trading loop resumed.")
    return {"ok": True, "message": "Trading resumed."}


@app.post("/api/control/emergency-stop")
async def control_emergency_stop():
    """Immediately halt new execution and keep system in a safe paused state."""
    await shared_state.set_emergency_stop(True)
    await shared_state.add_doom_speaks(
        "WAR ROOM COMMAND RECEIVED: Emergency stop engaged. New executions are halted."
    )
    return {"ok": True, "message": "Emergency stop engaged."}


@app.post("/api/control/clear-emergency-stop")
async def control_clear_emergency_stop():
    """Clear emergency stop while keeping manual resume explicit."""
    await shared_state.set_emergency_stop(False)
    await shared_state.add_doom_speaks(
        "WAR ROOM COMMAND RECEIVED: Emergency stop cleared. Use resume to restart cycles."
    )
    return {"ok": True, "message": "Emergency stop cleared."}


@app.post("/api/control/manual-order")
async def control_manual_order(order: ManualOrderRequest):
    """Submit a manual BUY/SELL market order from the War Room UI."""
    if shared_state.emergency_stop:
        return {"ok": False, "message": "Emergency stop is active. Clear it before manual orders."}

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"ok": False, "message": "Alpaca keys are not configured."}

    symbol = order.symbol.strip().upper()
    side = order.side.strip().lower()
    if side not in ("buy", "sell"):
        return {"ok": False, "message": "Side must be BUY or SELL."}

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "symbol": symbol,
        "qty": str(int(order.qty)),
        "side": side,
        "type": "market",
        "time_in_force": "gtc",
        "extended_hours": False,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{ALPACA_BASE_URL}/v2/orders", headers=headers, json=payload)

        if resp.status_code not in (200, 201):
            err_text = resp.text[:240]
            logger.warning(f"Manual order rejected: {resp.status_code} {err_text}")
            return {"ok": False, "message": f"Order rejected ({resp.status_code}): {err_text}"}

        order_data = resp.json()
        order_id = order_data.get("id", "")
        order_status = order_data.get("status", "")
        filled_price = await _wait_for_fill_price(order_id)

        # Fallback: use filled_avg_price from order response
        if not filled_price:
            avg = order_data.get("filled_avg_price")
            filled_price = float(avg) if avg else 0.0

        # If still no price (after-hours), fetch current market price as estimate
        if not filled_price or filled_price <= 0:
            try:
                from core.data_feeds import market_data
                filled_price = await market_data.get_latest_price(symbol)
            except Exception:
                filled_price = 0.0

        # Always record the trade — use submitted qty, reconcile updates it on fill
        actual_qty = float(order_data.get("qty") or order.qty)
        await performance_tracker.record_trade_open(
            symbol=symbol,
            side=side,
            qty=actual_qty,
            entry_price=filled_price,
            originating_agent="manual_operator",
            originating_strategy="MANUAL_ORDER",
            order_id=order_id,
        )

        # Push updated trade state to all WebSocket clients immediately
        await shared_state.update_trade_state(
            recent_trades=performance_tracker.get_recent_trades(50),
            open_trades=performance_tracker.get_open_trades(),
            metrics=performance_tracker.get_dashboard_metrics(),
            equity_curve=performance_tracker.get_equity_curve(200),
        )

        await shared_state.add_agent_message(
            "war_room",
            "execution",
            "MANUAL_ORDER",
            f"{side.upper()} {order.qty} {symbol}",
        )
        await shared_state.add_doom_speaks(
            f"WAR ROOM MANUAL ORDER: {side.upper()} {order.qty} {symbol} submitted."
        )

        if filled_price > 0:
            message = f"{side.upper()} {order.qty} {symbol} submitted and filled."
        else:
            message = f"{side.upper()} {order.qty} {symbol} submitted ({order_status or 'pending fill'})."

        return {
            "ok": True,
            "message": message,
            "order_id": order_id,
            "filled_price": filled_price,
            "status": order_status,
        }
    except Exception as e:
        logger.error(f"Manual order failed: {e}", exc_info=True)
        return {"ok": False, "message": f"Manual order failed: {str(e)[:180]}"}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
import os
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    # Serve from the single-file HTML build
    FRONTEND_HTML = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")

    @app.get("/")
    async def serve_frontend():
        if os.path.exists(FRONTEND_HTML):
            return FileResponse(FRONTEND_HTML)
        return {"message": "DOOM ENGINE API is running. Frontend not built yet."}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _build_full_state() -> Dict:
    """Build complete state for initial WebSocket push or REST."""
    metrics = performance_tracker.get_dashboard_metrics()
    regime_state = regime_detector.get_state()

    return {
        "type": "full_state",
        "timestamp": datetime.utcnow().isoformat(),
        "system_status": shared_state.system_status,
        "cycle_count": shared_state.cycle_count,
        "last_cycle_time": shared_state.last_cycle_time,
        "agents": shared_state.agent_statuses,
        "controls": {
            "paused": shared_state.trading_paused,
            "emergency_stop": shared_state.emergency_stop,
            "manual_strike_requested": shared_state.manual_strike_requested,
            "last_control_event": shared_state.last_control_event,
        },
        "regime": {
            "current": regime_state["current_regime"],
            "color": regime_state["color"],
            "probabilities": regime_state.get("probabilities", {}),
        },
        "risk": {
            "score": shared_state.risk_score,
            "label": shared_state.risk_label,
        },
        "metrics": metrics,
        "equity_curve": performance_tracker.get_equity_curve(200),
        "recent_trades": performance_tracker.get_recent_trades(20),
        "open_trades": performance_tracker.get_open_trades(),
        "doom_speaks": {
            "latest": shared_state.latest_briefing,
            "history": list(shared_state.doom_speaks),
        },
        "conflicts": conflict_logger.get_recent(20),
        "agent_messages": list(shared_state.agent_messages),
        "sentiment": shared_state.sentiment_scores,
        "market_quotes": shared_state.market_quotes,
        "live_positions": shared_state.live_positions,
        "attribution": {
            "by_agent": performance_tracker.compute_agent_attribution(),
            "by_strategy": performance_tracker.compute_strategy_attribution(),
        },
    }
