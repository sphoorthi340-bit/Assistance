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
import threading
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

-- =====================================================================
-- Phase 2: Personal State Modeling tables
-- =====================================================================

-- Goals: long-term objectives with progress tracking
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    category TEXT DEFAULT 'personal',
    target_type TEXT DEFAULT 'completion'
        CHECK(target_type IN ('streak', 'count', 'completion', 'progress')),
    target_value REAL,
    current_value REAL DEFAULT 0,
    status TEXT DEFAULT 'active'
        CHECK(status IN ('active', 'paused', 'completed', 'abandoned')),
    priority INTEGER DEFAULT 3 CHECK(priority >= 1 AND priority <= 5),
    start_date TEXT,
    deadline TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Habits: recurring behaviors to track
CREATE TABLE IF NOT EXISTS habits (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    frequency TEXT DEFAULT 'daily'
        CHECK(frequency IN ('daily', 'weekly', 'custom')),
    category TEXT DEFAULT 'personal',
    target_per_period REAL DEFAULT 1,
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL
);

-- Habit logs: individual completion records (powers streaks & analytics)
CREATE TABLE IF NOT EXISTS habit_logs (
    id TEXT PRIMARY KEY,
    habit_id TEXT NOT NULL,
    date TEXT NOT NULL,
    completed INTEGER DEFAULT 1,
    value REAL,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (habit_id) REFERENCES habits(id) ON DELETE CASCADE
);

-- Projects: active work items with progress tracking
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'active'
        CHECK(status IN ('active', 'paused', 'completed', 'archived')),
    progress_percentage REAL DEFAULT 0
        CHECK(progress_percentage >= 0 AND progress_percentage <= 100),
    current_blocker TEXT,
    next_step TEXT,
    priority INTEGER DEFAULT 3 CHECK(priority >= 1 AND priority <= 5),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_worked_at TEXT
);

-- Project tasks: granular work items within projects
CREATE TABLE IF NOT EXISTS project_tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending'
        CHECK(status IN ('pending', 'in_progress', 'completed', 'cancelled')),
    due_date TEXT,
    priority INTEGER DEFAULT 3 CHECK(priority >= 1 AND priority <= 5),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- =====================================================================
-- Phase 2.5: Action Engine & Knowledge Pipeline
-- =====================================================================

-- Action Log: audit trail for deterministic actions
CREATE TABLE IF NOT EXISTS action_log (
    id TEXT PRIMARY KEY,
    tool_name TEXT NOT NULL,
    parameters TEXT NOT NULL,       -- JSON
    result_status TEXT NOT NULL,     -- 'success', 'error', 'cancelled'
    result_message TEXT,
    confidence REAL,
    user_message TEXT,              -- original message that triggered this
    execution_time_ms INTEGER,
    created_at TEXT NOT NULL,
    undo_intent TEXT,               -- Intent for undo action
    undo_parameters TEXT,           -- JSON for undo action
    reversible INTEGER DEFAULT 0,   -- Boolean
    action_source TEXT,             -- 'user', 'scheduler', etc.
    trigger_type TEXT,              -- 'explicit_request', 'scheduled_task', etc.
    trigger_context TEXT            -- Optional JSON context
);

-- Documents: ingested knowledge sources for RAG
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    title TEXT,
    format TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    word_count INTEGER DEFAULT 0,
    file_hash TEXT,                -- SHA-256 for change detection
    ingested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

-- =====================================================================
-- Performance indexes
-- =====================================================================

-- Phase 1 indexes
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

-- Phase 2 indexes
CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_category ON goals(category);
CREATE INDEX IF NOT EXISTS idx_goals_deadline ON goals(deadline);
CREATE INDEX IF NOT EXISTS idx_habits_active ON habits(active);
CREATE INDEX IF NOT EXISTS idx_habit_logs_habit_id ON habit_logs(habit_id);
CREATE INDEX IF NOT EXISTS idx_habit_logs_date ON habit_logs(date DESC);
CREATE INDEX IF NOT EXISTS idx_habit_logs_habit_date
    ON habit_logs(habit_id, date);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);
CREATE INDEX IF NOT EXISTS idx_project_tasks_project ON project_tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_project_tasks_status ON project_tasks(status);
CREATE INDEX IF NOT EXISTS idx_project_tasks_due ON project_tasks(due_date);

-- Phase 2.5 indexes
CREATE INDEX IF NOT EXISTS idx_action_log_tool ON action_log(tool_name);
CREATE INDEX IF NOT EXISTS idx_action_log_created ON action_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(source_path);
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
        self._local = threading.local()
        self._ensure_directory()
        self.initialize()

    def _ensure_directory(self) -> None:
        """Create the database directory if it doesn't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """
        Create or retrieve a thread-local database connection with optimal settings.
        
        Each connection enables:
        - WAL mode (if configured) for concurrent reads
        - Foreign key enforcement
        - Row factory for dict-like access
        """
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(
                str(self._db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            if self._wal_mode:
                conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return self._local.conn

    def initialize(self) -> None:
        """Create tables and indexes if they don't exist, handle migrations."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            
        try:
            from backend.db_migrations import run_migrations
            run_migrations(str(self._db_path))
        except ImportError as e:
            logger.warning("Could not run Phase 3 migrations: %s", e)
            
            # Migration: Ensure action_log has new Phase 3 columns
            # In SQLite we can safely execute ALTER TABLE ADD COLUMN. If it fails due to existing, we ignore.
            try:
                conn.execute("ALTER TABLE action_log ADD COLUMN undo_intent TEXT")
                conn.execute("ALTER TABLE action_log ADD COLUMN undo_parameters TEXT")
                conn.execute("ALTER TABLE action_log ADD COLUMN reversible INTEGER DEFAULT 0")
                conn.execute("ALTER TABLE action_log ADD COLUMN action_source TEXT")
                conn.execute("ALTER TABLE action_log ADD COLUMN trigger_type TEXT")
                conn.execute("ALTER TABLE action_log ADD COLUMN trigger_context TEXT")
            except sqlite3.OperationalError:
                pass # Columns already exist
                
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
    # Goal CRUD
    # -------------------------------------------------------------------

    def create_goal(
        self,
        goal_id: str,
        title: str,
        description: Optional[str] = None,
        category: str = "personal",
        target_type: str = "completion",
        target_value: Optional[float] = None,
        priority: int = 3,
        start_date: Optional[str] = None,
        deadline: Optional[str] = None,
    ) -> None:
        """Insert a new goal row."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO goals "
                "(id, title, description, category, target_type, target_value, "
                "current_value, status, priority, start_date, deadline, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?, ?, ?)",
                (goal_id, title, description, category, target_type,
                 target_value, priority, start_date, deadline, now, now),
            )

    def get_goal(self, goal_id: str) -> Optional[dict]:
        """Get a single goal by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_goals(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[dict]:
        """List goals with optional filters."""
        query = "SELECT * FROM goals WHERE 1=1"
        params: list = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY priority ASC, created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_goal(self, goal_id: str, **fields) -> None:
        """Update arbitrary fields on a goal. Caller must validate fields."""
        if not fields:
            return
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [goal_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE goals SET {set_clause} WHERE id = ?", values
            )

    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal by ID. Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
        return cursor.rowcount > 0

    # -------------------------------------------------------------------
    # Habit CRUD
    # -------------------------------------------------------------------

    def create_habit(
        self,
        habit_id: str,
        name: str,
        description: Optional[str] = None,
        frequency: str = "daily",
        category: str = "personal",
        target_per_period: float = 1,
    ) -> None:
        """Insert a new habit row."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO habits "
                "(id, name, description, frequency, category, "
                "target_per_period, active, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (habit_id, name, description, frequency, category,
                 target_per_period, now),
            )

    def get_habit(self, habit_id: str) -> Optional[dict]:
        """Get a single habit by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM habits WHERE id = ?", (habit_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_habits(self, active_only: bool = True) -> list[dict]:
        """List habits, optionally filtering to active only."""
        query = "SELECT * FROM habits"
        params: list = []
        if active_only:
            query += " WHERE active = 1"
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_habit(self, habit_id: str, **fields) -> None:
        """Update arbitrary fields on a habit."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [habit_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE habits SET {set_clause} WHERE id = ?", values
            )

    def delete_habit(self, habit_id: str) -> bool:
        """Delete a habit and its logs (CASCADE). Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM habits WHERE id = ?", (habit_id,))
        return cursor.rowcount > 0

    # -------------------------------------------------------------------
    # Habit Log CRUD
    # -------------------------------------------------------------------

    def create_habit_log(
        self,
        log_id: str,
        habit_id: str,
        date: str,
        completed: bool = True,
        value: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Insert a new habit log entry."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO habit_logs "
                "(id, habit_id, date, completed, value, notes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (log_id, habit_id, date, int(completed), value, notes, now),
            )

    def get_habit_log_by_date(
        self, habit_id: str, date: str
    ) -> Optional[dict]:
        """Get a habit log for a specific habit and date."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM habit_logs WHERE habit_id = ? AND date = ?",
                (habit_id, date),
            ).fetchone()
        return dict(row) if row else None

    def update_habit_log(self, log_id: str, **fields) -> None:
        """Update fields on a habit log."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [log_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE habit_logs SET {set_clause} WHERE id = ?", values
            )

    def list_habit_logs(
        self,
        habit_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 30,
    ) -> list[dict]:
        """List habit logs for a habit with optional date range."""
        query = "SELECT * FROM habit_logs WHERE habit_id = ?"
        params: list = [habit_id]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_completed_habit_dates(self, habit_id: str) -> list[str]:
        """Get all dates where a habit was completed, sorted descending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date FROM habit_logs "
                "WHERE habit_id = ? AND completed = 1 "
                "ORDER BY date DESC",
                (habit_id,),
            ).fetchall()
        return [row["date"] for row in rows]

    # -------------------------------------------------------------------
    # Project CRUD
    # -------------------------------------------------------------------

    def create_project(
        self,
        project_id: str,
        name: str,
        description: Optional[str] = None,
        priority: int = 3,
    ) -> None:
        """Insert a new project row."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO projects "
                "(id, name, description, status, progress_percentage, "
                "priority, created_at, updated_at) "
                "VALUES (?, ?, ?, 'active', 0, ?, ?, ?)",
                (project_id, name, description, priority, now, now),
            )

    def get_project(self, project_id: str) -> Optional[dict]:
        """Get a single project by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_projects(self, status: Optional[str] = None) -> list[dict]:
        """List projects with optional status filter."""
        query = "SELECT * FROM projects"
        params: list = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY priority ASC, created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_project(self, project_id: str, **fields) -> None:
        """Update arbitrary fields on a project."""
        if not fields:
            return
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [project_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE projects SET {set_clause} WHERE id = ?", values
            )

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and its tasks (CASCADE). Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM projects WHERE id = ?", (project_id,)
            )
        return cursor.rowcount > 0

    # -------------------------------------------------------------------
    # Project Task CRUD
    # -------------------------------------------------------------------

    def create_project_task(
        self,
        task_id: str,
        project_id: str,
        title: str,
        description: Optional[str] = None,
        due_date: Optional[str] = None,
        priority: int = 3,
    ) -> None:
        """Insert a new project task."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO project_tasks "
                "(id, project_id, title, description, status, due_date, "
                "priority, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)",
                (task_id, project_id, title, description, due_date,
                 priority, now),
            )

    def get_project_task(self, task_id: str) -> Optional[dict]:
        """Get a single project task by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM project_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_project_tasks(
        self,
        project_id: str,
        status: Optional[str] = None,
    ) -> list[dict]:
        """List tasks for a project with optional status filter."""
        query = "SELECT * FROM project_tasks WHERE project_id = ?"
        params: list = [project_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY priority ASC, created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def update_project_task(self, task_id: str, **fields) -> None:
        """Update arbitrary fields on a project task."""
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [task_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE project_tasks SET {set_clause} WHERE id = ?", values
            )

    def get_project_task_counts(self, project_id: str) -> dict:
        """Get task count breakdown for a project."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM project_tasks "
                "WHERE project_id = ? GROUP BY status",
                (project_id,),
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM project_tasks WHERE project_id = ?",
                (project_id,),
            ).fetchone()[0]
        counts = {row["status"]: row["count"] for row in rows}
        counts["total"] = total
        return counts

    def delete_project_task(self, task_id: str) -> bool:
        """Delete a project task by ID. Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM project_tasks WHERE id = ?", (task_id,)
            )
        return cursor.rowcount > 0

    # -------------------------------------------------------------------
    # Document CRUD
    # -------------------------------------------------------------------

    def add_document(
        self,
        doc_id: str,
        source_path: str,
        format: str,
        title: Optional[str] = None,
        chunk_count: int = 0,
        word_count: int = 0,
        file_hash: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert a new document row."""
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO documents "
                "(id, source_path, title, format, chunk_count, word_count, file_hash, ingested_at, updated_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (doc_id, source_path, title, format, chunk_count, word_count, file_hash, now, now, meta_json),
            )

    def get_document_by_path(self, source_path: str) -> Optional[dict]:
        """Get a document by source_path."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE source_path = ?", (source_path,)
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        """List all documents."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID. Returns True if deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE id = ?", (doc_id,)
            )
        return cursor.rowcount > 0

    # -------------------------------------------------------------------
    # Action Log CRUD
    # -------------------------------------------------------------------

    def log_action(
        self,
        tool_name: str,
        parameters: dict,
        result_status: str,
        result_message: Optional[str] = None,
        confidence: Optional[float] = None,
        user_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        undo_intent: Optional[str] = None,
        undo_parameters: Optional[dict] = None,
        reversible: bool = False,
        action_source: Optional[str] = "user",
        trigger_type: Optional[str] = "explicit_request",
        trigger_context: Optional[dict] = None,
    ) -> str:
        """Log an executed action to the database with provenance."""
        action_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO action_log (
                    id, tool_name, parameters, result_status, result_message, 
                    confidence, user_message, execution_time_ms, created_at,
                    undo_intent, undo_parameters, reversible,
                    action_source, trigger_type, trigger_context
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    tool_name,
                    json.dumps(parameters),
                    result_status,
                    result_message,
                    confidence,
                    user_message,
                    execution_time_ms,
                    now,
                    undo_intent,
                    json.dumps(undo_parameters) if undo_parameters else None,
                    1 if reversible else 0,
                    action_source,
                    trigger_type,
                    json.dumps(trigger_context) if trigger_context else None,
                )
            )
        return action_id
        
    def get_last_action_log(self, successful_only: bool = True) -> Optional[dict]:
        """Fetch the most recent action executed."""
        with self._connect() as conn:
            query = "SELECT * FROM action_log"
            if successful_only:
                query += " WHERE result_status = 'success'"
            query += " ORDER BY created_at DESC LIMIT 1"
            
            row = conn.execute(query).fetchone()
            if not row:
                return None
                
            return dict(row)

    def get_action_log(self, action_id: str) -> Optional[dict]:
        """Get an action log by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM action_log WHERE id = ?", (action_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("parameters"):
            try:
                d["parameters"] = json.loads(d["parameters"])
            except Exception:
                pass
        return d

    def list_action_logs(self, limit: int = 50) -> list[dict]:
        """List recent action logs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM action_log ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            
        result = []
        for row in rows:
            d = dict(row)
            if d.get("parameters"):
                try:
                    d["parameters"] = json.loads(d["parameters"])
                except Exception:
                    pass
            result.append(d)
        return result

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

            # Phase 2 counts
            goal_count = conn.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'active'"
            ).fetchone()[0]
            habit_count = conn.execute(
                "SELECT COUNT(*) FROM habits WHERE active = 1"
            ).fetchone()[0]
            project_count = conn.execute(
                "SELECT COUNT(*) FROM projects WHERE status = 'active'"
            ).fetchone()[0]
            task_count = conn.execute(
                "SELECT COUNT(*) FROM project_tasks"
            ).fetchone()[0]

        return {
            "conversations": conv_count,
            "messages": msg_count,
            "memories": mem_count,
            "memories_by_type": {row["memory_type"]: row["count"] for row in mem_by_type},
            "goals": goal_count,
            "habits": habit_count,
            "projects": project_count,
            "tasks": task_count,
            "db_path": str(self._db_path),
        }

