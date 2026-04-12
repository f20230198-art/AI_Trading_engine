"""
DOOM ENGINE — Agent Memory Store
Real vector similarity search using ChromaDB + sentence-transformers.
Each agent persists decisions and retrieves similar past situations.
"""
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("doom.memory")

MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "memory")


class AgentMemoryStore:
    """
    Persistent agent memory with real vector similarity search.
    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    Falls back to SQLite-based keyword search if ChromaDB unavailable.
    """

    def __init__(self):
        self._chroma_client = None
        self._collections: Dict = {}
        self._initialized = False
        self._use_chroma = False

    async def initialize(self):
        """Initialize the memory store."""
        os.makedirs(MEMORY_DIR, exist_ok=True)

        try:
            import chromadb
            from chromadb.config import Settings

            self._chroma_client = chromadb.PersistentClient(
                path=os.path.join(MEMORY_DIR, "chromadb"),
                settings=Settings(anonymized_telemetry=False),
            )
            self._use_chroma = True
            logger.info("ChromaDB memory store initialized with persistent storage")
        except ImportError:
            logger.warning(
                "ChromaDB not available, falling back to SQLite memory. "
                "Install chromadb and sentence-transformers for vector search."
            )
            self._use_chroma = False
            await self._init_sqlite_fallback()

        self._initialized = True

    async def _init_sqlite_fallback(self):
        """Initialize SQLite-based fallback memory."""
        import aiosqlite

        db_path = os.path.join(MEMORY_DIR, "agent_memory.db")
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            await db.commit()

    def _get_collection(self, agent_name: str):
        """Get or create a ChromaDB collection for an agent."""
        if agent_name not in self._collections:
            self._collections[agent_name] = self._chroma_client.get_or_create_collection(
                name=f"doom_agent_{agent_name}",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[agent_name]

    async def store_decision(
        self,
        agent_name: str,
        situation: str,
        decision: str,
        outcome: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Store an agent's decision with its context for future retrieval.

        Args:
            agent_name: Name of the agent
            situation: Description of market situation when decision was made
            decision: What the agent decided
            outcome: Result of the decision (added later)
            metadata: Additional context (regime, symbols, etc.)
        """
        if not self._initialized:
            await self.initialize()

        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        doc_text = f"Situation: {situation}\nDecision: {decision}"
        if outcome:
            doc_text += f"\nOutcome: {outcome}"

        meta = {
            "agent_name": agent_name,
            "timestamp": timestamp,
            "decision": decision,
            **(metadata or {}),
        }

        if self._use_chroma:
            collection = self._get_collection(agent_name)
            # ChromaDB will use its default embedding function (sentence-transformers)
            collection.add(
                documents=[doc_text],
                metadatas=[meta],
                ids=[memory_id],
            )
            logger.debug(f"Stored memory for {agent_name}: {memory_id}")
        else:
            await self._sqlite_store(memory_id, agent_name, doc_text, meta, timestamp)

        return memory_id

    async def recall_similar(
        self,
        agent_name: str,
        current_situation: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Retrieve the most similar past situations and their outcomes.
        Uses real vector similarity search — not hardcoded lookup.
        """
        if not self._initialized:
            await self.initialize()

        if self._use_chroma:
            collection = self._get_collection(agent_name)
            if collection.count() == 0:
                return []

            results = collection.query(
                query_texts=[current_situation],
                n_results=min(top_k, collection.count()),
            )

            memories = []
            for i in range(len(results["documents"][0])):
                memories.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })
            return memories
        else:
            return await self._sqlite_recall(agent_name, current_situation, top_k)

    async def update_outcome(
        self,
        memory_id: str,
        agent_name: str,
        outcome: str,
    ):
        """Update a stored decision with its outcome."""
        if self._use_chroma:
            collection = self._get_collection(agent_name)
            existing = collection.get(ids=[memory_id])
            if existing and existing["documents"]:
                updated_doc = existing["documents"][0] + f"\nOutcome: {outcome}"
                meta = existing["metadatas"][0]
                meta["outcome"] = outcome
                collection.update(
                    ids=[memory_id],
                    documents=[updated_doc],
                    metadatas=[meta],
                )

    async def get_agent_stats(self, agent_name: str) -> Dict:
        """Get memory statistics for an agent."""
        if self._use_chroma:
            collection = self._get_collection(agent_name)
            return {
                "agent": agent_name,
                "total_memories": collection.count(),
                "store_type": "chromadb_vector",
            }
        else:
            return {
                "agent": agent_name,
                "total_memories": await self._sqlite_count(agent_name),
                "store_type": "sqlite_fallback",
            }

    # ------------------------------------------------------------------
    # SQLite fallback methods
    # ------------------------------------------------------------------
    async def _sqlite_store(self, memory_id, agent_name, content, metadata, timestamp):
        import aiosqlite
        db_path = os.path.join(MEMORY_DIR, "agent_memory.db")
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO memories (id, agent_name, content, metadata, timestamp) VALUES (?, ?, ?, ?, ?)",
                (memory_id, agent_name, content, json.dumps(metadata), timestamp),
            )
            await db.commit()

    async def _sqlite_recall(self, agent_name, situation, top_k):
        """Simple keyword-based recall for SQLite fallback."""
        import aiosqlite
        db_path = os.path.join(MEMORY_DIR, "agent_memory.db")

        # Extract keywords from situation
        keywords = [w.lower() for w in situation.split() if len(w) > 3]

        async with aiosqlite.connect(db_path) as db:
            # Get all memories for this agent, ordered by recency
            cursor = await db.execute(
                "SELECT content, metadata FROM memories WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?",
                (agent_name, top_k * 3),
            )
            rows = await cursor.fetchall()

        # Score by keyword overlap
        scored = []
        for content, meta_str in rows:
            content_lower = content.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            scored.append((score, content, json.loads(meta_str)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": content, "metadata": meta, "distance": None}
            for _, content, meta in scored[:top_k]
        ]

    async def _sqlite_count(self, agent_name):
        import aiosqlite
        db_path = os.path.join(MEMORY_DIR, "agent_memory.db")
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM memories WHERE agent_name = ?",
                (agent_name,),
            )
            row = await cursor.fetchone()
            return row[0] if row else 0


# Singleton
memory_store = AgentMemoryStore()
