"""
Jarvis V1 — Database Manager
===============================
SQLite-based persistent storage for conversations, messages, and memories.

Architecture decisions:
    - WAL mode enabled for concurrent read performance
    - Schema auto-initialized on first connection
    - All timestamps stored as ISO 8601 strings for portability
    - UUIDs used for conversation and memory IDs (time-sortable)
    - JSON metadata fields for future extensibility without schema changes
    - Connection created per-call (SQLite handles this efficiently)
    - The `data/` directory is auto-created if missing

Future extensibility:
    - The memories table supports typed memory entries (fact, goal, project, etc.)
    - The session_metadata / metadata JSON fields allow schema-free extension
    - Indexes on timestamp and type support the retrieval patterns needed for
      behavioral analytics and longitudinal analysis
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Conversations: each chat session with the user
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    session_metadata TEXT DEFAULT '{}'
);

-- Messages: individual turns within a conversation
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    thinking TEXT,
    timestamp TEXT NOT NULL,
    token_count INTEGER,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Memories: extracted facts, summaries, goals, preferences
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL CHECK(memory_type IN (
        'fact', 'summary', 'preference', 'goal', 'project',
        'habit', 'routine', 'deadline', 'observation'
    )),
    importance REAL DEFAULT 0.5 CHECK(importance >= 0.0 AND importance <= 1.0),
    created_at TEXT NOT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp
    ON messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_memories_type
    ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance
    ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created
    ON memories(created_at DESC);
"""


# ---------------------------------------------------------------------------
# Database Manager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    Manages all SQLite operations for Jarvis.
    
    Handles connection lifecycle, schema initialization, and provides
    typed CRUD methods for conversations, messages, and memories.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self._db_path = settings.resolve_path(settings.database.path)
        self._wal_mode = settings.database.wal_mode
        self._ensure_directory()
        self.initialize()

    def _ensure_directory(self) -> None:
        """Create the database directory if it doesn't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """
        Create a new database connection with optimal settings.
        
        Each connection enables:
        - WAL mode (if configured) for concurrent reads
        - Foreign key enforcement
        - Row factory for dict-like access
        """
        conn = sqlite3.connect(
            str(self._db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if self._wal_mode:
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
        logger.info("Database initialized at %s", self._db_path)

    # -------------------------------------------------------------------
    # Conversation CRUD
    # -------------------------------------------------------------------

    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation and return its ID.
        
        Args:
            title: Optional human-readable title. Auto-generated if not provided.
        
        Returns:
            The UUID of the new conversation.
        """
        conversation_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        title = title or f"Conversation {now[:10]}"

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )

        logger.info("Created conversation: %s (%s)", conversation_id[:8], title)
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a single conversation by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 20) -> list[dict]:
        """List recent conversations, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_conversation_timestamp(self, conversation_id: str) -> None:
        """Touch the updated_at timestamp for a conversation."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    # -------------------------------------------------------------------
    # Message CRUD
    # -------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        thinking: Optional[str] = None,
        token_count: Optional[int] = None,
    ) -> int:
        """
        Store a message in a conversation.
        
        Args:
            conversation_id: Parent conversation UUID.
            role: One of 'user', 'assistant', 'system'.
            content: The message text (with <think> tags stripped for assistant).
            thinking: The raw <think> content, stored separately for debugging.
            token_count: Optional token count for budget tracking.
        
        Returns:
            The auto-incremented message ID.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, content, thinking, timestamp, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, role, content, thinking, now, token_count),
            )
            # Also update the conversation's updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

        logger.debug("Stored %s message in conversation %s", role, conversation_id[:8])
        return cursor.lastrowid

    def get_conversation_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> list[dict]:
        """
        Get messages for a conversation, ordered chronologically.
        
        Args:
            conversation_id: The conversation UUID.
            limit: If set, return only the N most recent messages.
        
        Returns:
            List of message dicts with keys: id, role, content, timestamp, etc.
        """
        query = (
            "SELECT * FROM messages WHERE conversation_id = ? "
            "ORDER BY timestamp ASC"
        )
        params: list = [conversation_id]

        if limit is not None:
            # Get the last N messages while maintaining chronological order
            query = (
                "SELECT * FROM ("
                "  SELECT * FROM messages WHERE conversation_id = ? "
                "  ORDER BY timestamp DESC LIMIT ?"
                ") ORDER BY timestamp ASC"
            )
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_recent_messages_across_conversations(
        self,
        limit: int = 50,
    ) -> list[dict]:
        """Get the most recent messages across all conversations."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT m.*, c.title as conversation_title "
                "FROM messages m "
                "JOIN conversations c ON m.conversation_id = c.id "
                "ORDER BY m.timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -------------------------------------------------------------------
    # Memory CRUD
    # -------------------------------------------------------------------

    def store_memory(
        self,
        content: str,
        memory_type: str,
        conversation_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store an extracted memory entry.
        
        Args:
            content: The memory text (fact, summary, goal, etc.).
            memory_type: Category — one of: fact, summary, preference, goal,
                        project, habit, routine, deadline, observation.
            conversation_id: Source conversation (nullable for manual entries).
            importance: Relevance score from 0.0 to 1.0.
            metadata: Optional JSON-serializable dict for extensibility.
        
        Returns:
            The UUID of the stored memory.
        """
        memory_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories "
                "(id, conversation_id, content, memory_type, importance, "
                "created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, conversation_id, content, memory_type,
                 importance, now, meta_json),
            )

        logger.info(
            "Stored memory [%s]: %.60s... (importance=%.2f)",
            memory_type, content, importance,
        )
        return memory_id

    def search_memories(
        self,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search memories with optional filters.
        
        Args:
            memory_type: Filter by type (e.g., 'goal', 'fact').
            min_importance: Minimum importance threshold.
            limit: Maximum results to return.
        
        Returns:
            List of memory dicts, ordered by importance then recency.
        """
        query = "SELECT * FROM memories WHERE 1=1"
        params: list = []

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        if min_importance is not None:
            query += " AND importance >= ?"
            params.append(min_importance)

        query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_all_memories(self, limit: int = 100) -> list[dict]:
        """Get all memories ordered by recency."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_memory_access(self, memory_id: str) -> None:
        """
        Record that a memory was accessed (used in context retrieval).
        
        This supports future memory decay/reinforcement algorithms:
        frequently accessed memories can be boosted, rarely accessed
        ones can be deprioritized.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 "
                "WHERE id = ?",
                (now, memory_id),
            )

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if a row was deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted memory: %s", memory_id[:8])
        return deleted

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get database statistics for health checks and observability."""
        with self._connect() as conn:
            conv_count = conn.execute(
                "SELECT COUNT(*) FROM conversations"
            ).fetchone()[0]
            msg_count = conn.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
            mem_count = conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]
            mem_by_type = conn.execute(
                "SELECT memory_type, COUNT(*) as count FROM memories "
                "GROUP BY memory_type ORDER BY count DESC"
            ).fetchall()

        return {
            "conversations": conv_count,
            "messages": msg_count,
            "memories": mem_count,
            "memories_by_type": {row["memory_type"]: row["count"] for row in mem_by_type},
            "db_path": str(self._db_path),
        }
