"""
DOOM ENGINE — Conflict Resolution Logger
Logs real disagreements between agents with full reasoning chains.
Feeds the dashboard conflict panel in real time.
"""
import json
import logging
import os
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("doom.conflict")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class ConflictLogger:
    """
    Records agent conflicts — when two agents produce contradictory signals
    and the Orchestrator must resolve.
    """

    def __init__(self, max_in_memory: int = 200):
        self._conflicts: deque = deque(maxlen=max_in_memory)
        self._db_path = os.path.join(DATA_DIR, "conflicts.db")
        self._initialized = False

    async def initialize(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_a_name TEXT NOT NULL,
                    agent_a_position TEXT NOT NULL,
                    agent_a_reasoning TEXT NOT NULL,
                    agent_b_name TEXT NOT NULL,
                    agent_b_position TEXT NOT NULL,
                    agent_b_reasoning TEXT NOT NULL,
                    symbol TEXT,
                    verdict TEXT NOT NULL,
                    verdict_reasoning TEXT NOT NULL,
                    resolved_by TEXT DEFAULT 'orchestrator'
                )
            """)
            await db.commit()
        self._initialized = True
        logger.info("ConflictLogger initialized")

    async def log_conflict(
        self,
        agent_a_name: str,
        agent_a_position: str,
        agent_a_reasoning: str,
        agent_b_name: str,
        agent_b_position: str,
        agent_b_reasoning: str,
        symbol: Optional[str],
        verdict: str,
        verdict_reasoning: str,
        resolved_by: str = "orchestrator",
    ) -> Dict:
        """Log a real conflict between two agents."""
        if not self._initialized:
            await self.initialize()

        timestamp = datetime.utcnow().isoformat()

        conflict = {
            "timestamp": timestamp,
            "agent_a": {
                "name": agent_a_name,
                "position": agent_a_position,
                "reasoning": agent_a_reasoning,
            },
            "agent_b": {
                "name": agent_b_name,
                "position": agent_b_position,
                "reasoning": agent_b_reasoning,
            },
            "symbol": symbol,
            "verdict": verdict,
            "verdict_reasoning": verdict_reasoning,
            "resolved_by": resolved_by,
        }

        self._conflicts.append(conflict)

        # Persist to SQLite
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO conflicts
                   (timestamp, agent_a_name, agent_a_position, agent_a_reasoning,
                    agent_b_name, agent_b_position, agent_b_reasoning,
                    symbol, verdict, verdict_reasoning, resolved_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp, agent_a_name, agent_a_position, agent_a_reasoning,
                    agent_b_name, agent_b_position, agent_b_reasoning,
                    symbol, verdict, verdict_reasoning, resolved_by,
                ),
            )
            await db.commit()

        logger.info(
            f"CONFLICT: {agent_a_name}({agent_a_position}) vs "
            f"{agent_b_name}({agent_b_position}) -> {verdict}"
        )
        return conflict

    def get_recent(self, n: int = 20) -> List[Dict]:
        """Get n most recent conflicts for the dashboard."""
        return list(self._conflicts)[-n:]

    async def get_all(self, limit: int = 100) -> List[Dict]:
        """Get conflicts from database."""
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """SELECT timestamp, agent_a_name, agent_a_position, agent_a_reasoning,
                          agent_b_name, agent_b_position, agent_b_reasoning,
                          symbol, verdict, verdict_reasoning, resolved_by
                   FROM conflicts ORDER BY id DESC LIMIT ?""",
                (limit,),
            )
            rows = await cursor.fetchall()

        return [
            {
                "timestamp": r[0],
                "agent_a": {"name": r[1], "position": r[2], "reasoning": r[3]},
                "agent_b": {"name": r[4], "position": r[5], "reasoning": r[6]},
                "symbol": r[7],
                "verdict": r[8],
                "verdict_reasoning": r[9],
                "resolved_by": r[10],
            }
            for r in rows
        ]

    async def get_stats(self) -> Dict:
        """Get conflict statistics."""
        import aiosqlite
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM conflicts")
            total = (await cursor.fetchone())[0]

            cursor = await db.execute(
                "SELECT resolved_by, COUNT(*) FROM conflicts GROUP BY resolved_by"
            )
            by_resolver = dict(await cursor.fetchall())

        return {
            "total_conflicts": total,
            "by_resolver": by_resolver,
            "recent_count": len(self._conflicts),
        }


# Singleton
conflict_logger = ConflictLogger()
