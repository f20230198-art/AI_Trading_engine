"""
DOOM ENGINE — Execution Agent
Places real paper trades via Alpaca API.
Handles market/limit/stop orders, confirms fills, retries with backoff.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set

import httpx

from agents.base_agent import BaseAgent
from core.data_feeds import market_data
from core.performance_tracker import performance_tracker
from core.shared_state import shared_state
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

logger = logging.getLogger("doom.agent.execution")


class ExecutionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="execution",
            role="Execution — places real paper trades via Alpaca API",
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._alpaca_headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type": "application/json",
        }

    async def initialize(self):
        await super().initialize()
        self._http_client = httpx.AsyncClient(timeout=30.0)
        # Track open order IDs → trade IDs so we can reconcile fills
        self._pending_fills: Dict[str, int] = {}  # order_id → trade_id

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        await self.update_status("RUNNING", "Executing approved trades")

        try:
            # Reconcile any previously opened trades against live Alpaca orders
            if ALPACA_API_KEY:
                await self._reconcile_closed_trades()

            if shared_state.emergency_stop:
                await self.update_status("ADVISORY", "Emergency stop active")
                return {
                    "agent": self.name,
                    "executed": [],
                    "failed": [{"success": False, "error": "Emergency stop active"}],
                }

            approved_trades = context.get("approved_trades", [])

            if not approved_trades:
                await self.update_status("ACTIVE", "Awaiting orders")
                return {"agent": self.name, "executed": [], "failed": []}

            if not ALPACA_API_KEY:
                # Not an error condition. Everything upstream — signals, risk
                # review, veto enforcement, sizing — has run and is observable;
                # only order placement needs a broker. Report and skip rather
                # than raising, so a keyless demo shows a clean cycle.
                logger.info(
                    f"No Alpaca credentials — {len(approved_trades)} approved "
                    "trade(s) not placed. Set ALPACA_API_KEY to enable execution."
                )
                await self.update_status(
                    "ADVISORY", f"{len(approved_trades)} trades pending broker credentials"
                )
                await shared_state.add_agent_message(
                    self.name, "orchestrator", "EXECUTION_SKIPPED",
                    f"{len(approved_trades)} trades ready but no broker configured",
                )
                return {
                    "agent": self.name,
                    "executed": [],
                    "failed": [],
                    "skipped": approved_trades,
                    "reason": "no_broker_credentials",
                }

            executed = []
            failed = []

            for trade in approved_trades:
                result = await self._execute_trade(trade)
                if result.get("success"):
                    executed.append(result)

                    # Record in performance tracker and map order_id for later reconciliation
                    trade_id = await performance_tracker.record_trade_open(
                        symbol=trade["symbol"],
                        side=trade["action"].lower(),
                        qty=result.get("filled_qty", trade.get("qty", 0)),
                        entry_price=result.get("filled_price", 0),
                        originating_agent=trade.get("originating_agent", "strategy"),
                        originating_strategy=trade.get("strategy", "unknown"),
                        order_id=result.get("order_id", ""),
                    )
                    order_id = result.get("order_id", "")
                    if order_id and trade_id:
                        self._pending_fills[order_id] = trade_id
                else:
                    failed.append(result)

            # Store decision
            await self._store_decision(
                situation=f"Executing {len(approved_trades)} trades",
                decision=f"Executed {len(executed)}, Failed {len(failed)}",
                metadata={
                    "executed_symbols": [t["symbol"] for t in executed] or ["none"],
                    "regime": shared_state.current_regime,
                },
            )

            await shared_state.add_agent_message(
                self.name, "orchestrator", "EXECUTION_REPORT",
                f"Executed {len(executed)}/{len(approved_trades)} orders. "
                f"Failed: {len(failed)}"
            )

            status_msg = f"Executed {len(executed)}, Failed {len(failed)}"
            await self.update_status("ACTIVE", status_msg)

            return {
                "agent": self.name,
                "executed": executed,
                "failed": failed,
            }

        except Exception as e:
            self.errors += 1
            await self.update_status("ERROR", str(e)[:100])
            logger.error(f"Execution error: {e}", exc_info=True)
            raise

    async def _execute_trade(self, trade: Dict, max_retries: int = 3) -> Dict:
        """Place a single order via Alpaca with retry logic."""
        symbol = trade["symbol"]
        action = trade["action"].lower()  # buy or sell
        order_type = trade.get("entry_type", "market").lower()

        # Never fall back to a nominal size. A trade that reaches here without a
        # deterministic qty was not sized against portfolio equity, so it cannot
        # be shown to respect max_position_pct — refuse it rather than guess.
        try:
            qty = int(trade.get("qty") or 0)
        except (TypeError, ValueError):
            qty = 0
        if qty <= 0:
            logger.error(
                f"Refusing to place {action.upper()} {symbol}: qty={trade.get('qty')!r} "
                "— trade was not sized deterministically."
            )
            return {
                "success": False,
                "symbol": symbol,
                "error": "Invalid qty — trade not sized",
            }

        # Build order payload
        order = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": action,
            "type": order_type,
            "time_in_force": "gtc",
        }

        # Add limit price if limit order
        if order_type == "limit" and trade.get("limit_price"):
            order["limit_price"] = str(trade["limit_price"])

        # Add stop loss if specified
        stop_loss_pct = trade.get("stop_loss_pct")

        for attempt in range(max_retries):
            try:
                resp = await self._http_client.post(
                    f"{ALPACA_BASE_URL}/v2/orders",
                    headers=self._alpaca_headers,
                    json=order,
                )

                if resp.status_code in (200, 201):
                    order_data = resp.json()
                    order_id = order_data.get("id", "")
                    logger.info(
                        f"Order placed: {action.upper()} {qty} {symbol} "
                        f"(type={order_type}) — ID: {order_id}"
                    )

                    # Wait briefly for fill
                    filled_price = await self._wait_for_fill(order_id)

                    # If not filled yet (after-hours/weekend), get current market price
                    # as the best estimate for entry_price so the tracker isn't null
                    if not filled_price or filled_price <= 0:
                        try:
                            from core.data_feeds import market_data
                            filled_price = await market_data.get_latest_price(symbol)
                        except Exception:
                            filled_price = 0.0

                    # Use the submitted qty (always correct); filled_qty updates on reconcile
                    actual_qty = float(order_data.get("qty", qty))

                    # Place stop-loss order if we have a price
                    if stop_loss_pct and filled_price and action == "buy":
                        stop_price = filled_price * (1 - stop_loss_pct / 100)
                        await self._place_stop_loss(symbol, actual_qty, stop_price)

                    return {
                        "success": True,
                        "symbol": symbol,
                        "action": action,
                        "qty": actual_qty,
                        "order_id": order_id,
                        "order_type": order_type,
                        "filled_price": filled_price,
                        "filled_qty": actual_qty,
                        "status": order_data.get("status", ""),
                    }

                elif resp.status_code == 403:
                    logger.error(f"Alpaca rejected order for {symbol}: Forbidden")
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": f"Forbidden: {resp.text}",
                    }

                elif resp.status_code == 422:
                    error_msg = resp.json().get("message", resp.text)
                    logger.warning(f"Order validation failed for {symbol}: {error_msg}")
                    return {
                        "success": False,
                        "symbol": symbol,
                        "error": f"Validation: {error_msg}",
                    }

                else:
                    logger.warning(
                        f"Order attempt {attempt + 1} failed: "
                        f"{resp.status_code} — {resp.text}"
                    )

            except Exception as e:
                logger.warning(f"Order attempt {attempt + 1} exception: {e}")

            # Exponential backoff
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait}s...")
                await asyncio.sleep(wait)

        return {
            "success": False,
            "symbol": symbol,
            "error": f"Failed after {max_retries} attempts",
        }

    async def _wait_for_fill(self, order_id: str, timeout: int = 10) -> Optional[float]:
        """Wait for an order to be filled and return the fill price."""
        for _ in range(timeout):
            try:
                resp = await self._http_client.get(
                    f"{ALPACA_BASE_URL}/v2/orders/{order_id}",
                    headers=self._alpaca_headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "filled":
                        return float(data.get("filled_avg_price", 0))
                    if data.get("status") in ("canceled", "rejected", "expired"):
                        logger.warning(f"Order {order_id} ended with status: {data['status']}")
                        return None
            except Exception:
                pass
            await asyncio.sleep(1)

        return None

    async def _place_stop_loss(self, symbol: str, qty: float, stop_price: float):
        """Place a stop-loss order."""
        order = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": "sell",
            "type": "stop",
            "stop_price": str(round(stop_price, 2)),
            "time_in_force": "gtc",
        }
        try:
            resp = await self._http_client.post(
                f"{ALPACA_BASE_URL}/v2/orders",
                headers=self._alpaca_headers,
                json=order,
            )
            if resp.status_code in (200, 201):
                logger.info(f"Stop-loss placed: SELL {qty} {symbol} @ ${stop_price:.2f}")
            else:
                logger.warning(f"Stop-loss failed for {symbol}: {resp.text}")
        except Exception as e:
            logger.warning(f"Stop-loss exception for {symbol}: {e}")

    async def _reconcile_closed_trades(self):
        """
        Each cycle:
        1. Pull live Alpaca positions and all open orders.
        2. For tracker trades whose order is now filled → update entry_price/qty.
        3. For tracker trades whose symbol left positions → close them with exit price.
        """
        open_trades = performance_tracker.get_open_trades()
        if not open_trades:
            return

        try:
            # Get live positions
            pos_resp = await self._http_client.get(
                f"{ALPACA_BASE_URL}/v2/positions",
                headers=self._alpaca_headers,
            )
            # Get all open orders from Alpaca
            ord_resp = await self._http_client.get(
                f"{ALPACA_BASE_URL}/v2/orders",
                headers=self._alpaca_headers,
                params={"status": "all", "limit": 200, "direction": "desc"},
            )
            if pos_resp.status_code != 200:
                return

            live_positions = pos_resp.json()
            live_pos_map: Dict[str, Dict] = {p["symbol"]: p for p in live_positions}
            live_symbols: Set[str] = set(live_pos_map.keys())

            alpaca_orders: Dict[str, Dict] = {}
            if ord_resp.status_code == 200:
                for o in ord_resp.json():
                    alpaca_orders[o["id"]] = o

            for trade in open_trades:
                symbol = trade.get("symbol", "")
                trade_id = trade.get("id")
                order_id = trade.get("order_id", "")
                if not trade_id:
                    continue

                # Step 1: if we have the Alpaca order and it's filled, update entry
                alpaca_order = alpaca_orders.get(order_id)
                if alpaca_order:
                    status = alpaca_order.get("status", "")
                    filled_qty = float(alpaca_order.get("filled_qty") or 0)
                    filled_price = float(alpaca_order.get("filled_avg_price") or 0)
                    # If now filled and our record had qty=0 or null price, update it
                    if status == "filled" and filled_qty > 0 and filled_price > 0:
                        current_qty = float(trade.get("qty") or 0)
                        current_price = float(trade.get("entry_price") or 0)
                        if current_qty == 0 or current_price == 0:
                            await performance_tracker.update_trade_entry(
                                trade_id=trade_id,
                                qty=filled_qty,
                                entry_price=filled_price,
                            )
                            logger.info(
                                f"Updated fill: {symbol} qty={filled_qty} @ ${filled_price:.2f}"
                            )

                # Step 2: if symbol is no longer in live positions, close the trade
                if symbol not in live_symbols:
                    # Skip trades with qty=0 and no fill — they're ghost records
                    current_qty = float(trade.get("qty") or 0)
                    if current_qty == 0:
                        # Mark as cancelled/ghost
                        await performance_tracker.cancel_ghost_trade(trade_id)
                        continue

                    # If the Alpaca order is still pending/accepted, the position just
                    # hasn't filled yet (e.g. after-hours GTC). Do NOT close the trade.
                    if alpaca_order:
                        order_status = alpaca_order.get("status", "")
                        if order_status in ("accepted", "pending_new", "new", "held",
                                           "partially_filled", "pending_cancel",
                                           "pending_replace", "accepted_for_bidding"):
                            continue  # Order still live — keep trade open

                    exit_price = await self._get_last_fill_price(symbol)
                    if not exit_price or exit_price <= 0:
                        exit_price = float(trade.get("entry_price") or 0)

                    if exit_price > 0:
                        closed = await performance_tracker.record_trade_close(
                            trade_id=trade_id,
                            exit_price=exit_price,
                        )
                        if closed:
                            logger.info(
                                f"Reconciled closed: {symbol} P&L=${closed.get('pnl', 0):+.2f}"
                            )

                # Step 3: for positions we DO hold, sync unrealized P&L to shared state
                elif symbol in live_pos_map:
                    pos = live_pos_map[symbol]
                    # live_positions already pushed to shared_state in main loop — nothing extra needed

        except Exception as e:
            logger.warning(f"Trade reconciliation failed: {e}")

    async def _get_last_fill_price(self, symbol: str) -> Optional[float]:
        """Fetch the most recent filled order price for a symbol from Alpaca."""
        try:
            resp = await self._http_client.get(
                f"{ALPACA_BASE_URL}/v2/orders",
                headers=self._alpaca_headers,
                params={"status": "closed", "symbols": symbol, "limit": 5, "direction": "desc"},
            )
            if resp.status_code != 200:
                return None
            orders = resp.json()
            for order in orders:
                if (
                    order.get("symbol") == symbol
                    and order.get("status") == "filled"
                    and order.get("filled_avg_price")
                ):
                    return float(order["filled_avg_price"])
        except Exception:
            pass
        return None

    async def get_account_info(self) -> Dict:
        """Get Alpaca account info."""
        resp = await self._http_client.get(
            f"{ALPACA_BASE_URL}/v2/account",
            headers=self._alpaca_headers,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_positions(self) -> List[Dict]:
        """Get all current positions from Alpaca."""
        resp = await self._http_client.get(
            f"{ALPACA_BASE_URL}/v2/positions",
            headers=self._alpaca_headers,
        )
        resp.raise_for_status()
        return resp.json()
