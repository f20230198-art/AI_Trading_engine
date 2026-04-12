"""
DOOM ENGINE — Performance Attribution Tracker
Tracks per-agent and per-strategy P&L, Sharpe, win rate — all computed
from real trade history. No hardcoded metrics.
"""
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("doom.performance")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class PerformanceTracker:
    """
    Tracks real P&L attribution per agent/strategy.
    All metrics computed mathematically from actual trade history.
    """

    def __init__(self):
        self._trades: List[Dict] = []
        self._equity_curve: List[Dict] = []
        self._initial_equity: float = 100_000.0
        self._current_equity: float = 100_000.0
        self._peak_equity: float = 100_000.0
        self._daily_returns: List[float] = []
        self._db_path = os.path.join(DATA_DIR, "performance.db")
        self._initialized = False

    async def initialize(self, initial_equity: float = 100_000.0):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._initial_equity = initial_equity
        self._current_equity = initial_equity
        self._peak_equity = initial_equity

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    originating_agent TEXT,
                    originating_strategy TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    status TEXT DEFAULT 'open',
                    order_id TEXT,
                    metadata TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL,
                    positions_value REAL
                )
            """)
            await db.commit()

        # Load existing trades
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT * FROM trades ORDER BY id")
            rows = await cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
            self._trades = [dict(zip(cols, row)) for row in rows]

        self._initialized = True
        logger.info(f"PerformanceTracker initialized with ${initial_equity:,.2f}")

    async def record_trade_open(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        originating_agent: str,
        originating_strategy: str,
        order_id: str = "",
    ) -> int:
        """Record a new trade opening."""
        import aiosqlite

        entry_time = datetime.utcnow().isoformat()
        trade = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": None,
            "pnl": 0,
            "pnl_pct": 0,
            "originating_agent": originating_agent,
            "originating_strategy": originating_strategy,
            "entry_time": entry_time,
            "exit_time": None,
            "status": "open",
            "order_id": order_id,
        }

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """INSERT INTO trades (symbol, side, qty, entry_price, originating_agent,
                   originating_strategy, entry_time, status, order_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
                (symbol, side, qty, entry_price, originating_agent,
                 originating_strategy, entry_time, order_id),
            )
            trade_id = cursor.lastrowid
            await db.commit()

        trade["id"] = trade_id
        self._trades.append(trade)
        price_str = f"${entry_price:.2f}" if entry_price is not None else "market"
        logger.info(f"Trade opened: {side} {qty} {symbol} @ {price_str} by {originating_agent}")
        return trade_id

    async def update_trade_entry(self, trade_id: int, qty: float, entry_price: float):
        """Update qty and entry_price once an order's fill is confirmed."""
        import aiosqlite
        for t in self._trades:
            if t.get("id") == trade_id:
                t["qty"] = qty
                t["entry_price"] = entry_price
                break
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE trades SET qty=?, entry_price=? WHERE id=?",
                (qty, entry_price, trade_id),
            )
            await db.commit()

    async def cancel_ghost_trade(self, trade_id: int):
        """Remove a trade that was recorded before fill and never filled (qty=0)."""
        import aiosqlite
        self._trades = [t for t in self._trades if t.get("id") != trade_id]
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM trades WHERE id=? AND qty=0", (trade_id,))
            await db.commit()

    async def record_trade_close(
        self,
        trade_id: int,
        exit_price: float,
    ) -> Dict:
        """Record a trade closing and compute P&L."""
        import aiosqlite

        # Find trade
        trade = None
        for t in self._trades:
            if t.get("id") == trade_id:
                trade = t
                break

        if not trade:
            logger.error(f"Trade {trade_id} not found")
            return {}

        # Compute P&L
        if trade["side"] == "buy":
            pnl = (exit_price - trade["entry_price"]) * trade["qty"]
        else:  # sell/short
            pnl = (trade["entry_price"] - exit_price) * trade["qty"]

        pnl_pct = (pnl / (trade["entry_price"] * trade["qty"])) * 100

        exit_time = datetime.utcnow().isoformat()

        trade["exit_price"] = exit_price
        trade["pnl"] = pnl
        trade["pnl_pct"] = pnl_pct
        trade["exit_time"] = exit_time
        trade["status"] = "closed"

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """UPDATE trades SET exit_price = ?, pnl = ?, pnl_pct = ?,
                   exit_time = ?, status = 'closed' WHERE id = ?""",
                (exit_price, pnl, pnl_pct, exit_time, trade_id),
            )
            await db.commit()

        self._current_equity += pnl
        self._peak_equity = max(self._peak_equity, self._current_equity)

        logger.info(f"Trade closed: {trade['symbol']} P&L=${pnl:+.2f} ({pnl_pct:+.2f}%)")
        return trade

    async def snapshot_equity(self, equity: float, cash: float, positions_value: float):
        """Record an equity snapshot for the curve."""
        import aiosqlite

        timestamp = datetime.utcnow().isoformat()
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

        snapshot = {"timestamp": timestamp, "equity": equity, "cash": cash, "positions_value": positions_value}
        self._equity_curve.append(snapshot)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO equity_snapshots (timestamp, equity, cash, positions_value) VALUES (?, ?, ?, ?)",
                (timestamp, equity, cash, positions_value),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Computed metrics — all from real trade data
    # ------------------------------------------------------------------
    def compute_total_pnl(self) -> float:
        return sum(t.get("pnl", 0) for t in self._trades if t["status"] == "closed")

    def compute_win_rate(self) -> float:
        closed = [t for t in self._trades if t["status"] == "closed"]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
        return round(wins / len(closed) * 100, 1)

    def compute_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Compute Sharpe ratio from trade returns."""
        closed = [t for t in self._trades if t["status"] == "closed" and t.get("pnl_pct")]
        if len(closed) < 2:
            return 0.0
        returns = [t["pnl_pct"] / 100 for t in closed]
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret == 0:
            return 0.0
        # Annualize assuming ~252 trading days
        daily_rf = risk_free_rate / 252
        sharpe = (mean_ret - daily_rf) / std_ret * math.sqrt(252)
        return round(sharpe, 2)

    def compute_max_drawdown(self) -> float:
        """Compute max drawdown from equity curve."""
        if not self._equity_curve:
            return 0.0
        equities = [s["equity"] for s in self._equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        return round(max_dd * 100, 2)

    def compute_agent_attribution(self) -> Dict[str, Dict]:
        """Per-agent performance attribution."""
        agent_stats = defaultdict(lambda: {
            "trades": 0, "wins": 0, "total_pnl": 0.0,
            "returns": [], "agent": "",
        })

        for t in self._trades:
            if t["status"] != "closed":
                continue
            agent = t.get("originating_agent", "unknown")
            stats = agent_stats[agent]
            stats["agent"] = agent
            stats["trades"] += 1
            if t.get("pnl", 0) > 0:
                stats["wins"] += 1
            stats["total_pnl"] += t.get("pnl", 0)
            if t.get("pnl_pct"):
                stats["returns"].append(t["pnl_pct"] / 100)

        result = {}
        for agent, stats in agent_stats.items():
            win_rate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
            returns = stats["returns"]
            if len(returns) >= 2:
                sharpe = (np.mean(returns) / np.std(returns, ddof=1)) * math.sqrt(252)
            else:
                sharpe = 0.0
            result[agent] = {
                "agent": agent,
                "trades": stats["trades"],
                "wins": stats["wins"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "avg_return": round(np.mean(returns) * 100, 2) if returns else 0,
                "sharpe": round(sharpe, 2),
            }
        return result

    def compute_strategy_attribution(self) -> Dict[str, Dict]:
        """Per-strategy performance attribution."""
        strat_stats = defaultdict(lambda: {
            "trades": 0, "wins": 0, "total_pnl": 0.0, "returns": [],
        })

        for t in self._trades:
            if t["status"] != "closed":
                continue
            strat = t.get("originating_strategy", "unknown")
            stats = strat_stats[strat]
            stats["trades"] += 1
            if t.get("pnl", 0) > 0:
                stats["wins"] += 1
            stats["total_pnl"] += t.get("pnl", 0)
            if t.get("pnl_pct"):
                stats["returns"].append(t["pnl_pct"] / 100)

        result = {}
        for strat, stats in strat_stats.items():
            win_rate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
            result[strat] = {
                "strategy": strat,
                "trades": stats["trades"],
                "win_rate": round(win_rate, 1),
                "total_pnl": round(stats["total_pnl"], 2),
            }
        return result

    def get_open_trades(self) -> List[Dict]:
        return [t for t in self._trades if t["status"] == "open"]

    def get_recent_trades(self, n: int = 20) -> List[Dict]:
        return self._trades[-n:]

    def get_equity_curve(self, n: int = 100) -> List[Dict]:
        return self._equity_curve[-n:]

    def get_dashboard_metrics(self) -> Dict:
        """All metrics for the dashboard, computed live."""
        open_trades = self.get_open_trades()
        realized_pnl = round(self.compute_total_pnl(), 2)
        equity_pnl = round(self._current_equity - self._initial_equity, 2)

        # Unrealized P&L: sum from open trades with known entry prices
        unrealized = 0.0
        for t in open_trades:
            ep = t.get("entry_price") or 0
            qty = t.get("qty") or 0
            if ep > 0 and qty > 0:
                # We don't have live price here; rely on equity_pnl - realized_pnl
                pass
        unrealized = round(equity_pnl - realized_pnl, 2)

        return {
            "total_pnl": realized_pnl,
            "realized_pnl": realized_pnl,
            "equity_pnl": equity_pnl,
            "unrealized_estimate": unrealized,
            "sharpe_ratio": self.compute_sharpe_ratio(),
            "max_drawdown": self.compute_max_drawdown(),
            "win_rate": self.compute_win_rate(),
            "active_positions": len(open_trades),
            "total_trades": len(self._trades),
            "closed_trades": len([t for t in self._trades if t["status"] == "closed"]),
            "current_equity": round(self._current_equity, 2),
            "initial_equity": self._initial_equity,
            "return_pct": round(
                (self._current_equity - self._initial_equity) / self._initial_equity * 100, 2
            ) if self._initial_equity > 0 else 0,
        }


# Singleton
performance_tracker = PerformanceTracker()
