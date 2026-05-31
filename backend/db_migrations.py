"""
Jarvis Phase 3 — Database Migrations
========================================
Creates all new tables required for Phase 3 components.
All statements use IF NOT EXISTS — safe to run repeatedly.

New tables:
    - provider_health:  Provider availability tracking
    - route_logs:       Model routing audit trail
    - cloud_usage:      Cloud API cost/token tracking
    - cloud_cache:      Semantic response caching
    - weekly_reports:   Generated weekly summaries
    - inbox:            Persistent notification storage
    - decision_trace:   Explainability audit trail

Also expands the memories.memory_type CHECK constraint to include
'academic' and 'insight' categories.
"""

import sqlite3
from pathlib import Path
from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Migration SQL
# ---------------------------------------------------------------------------

_PHASE3_TABLES_SQL = """
-- =====================================================================
-- Phase 3: Provider Manager
-- =====================================================================

CREATE TABLE IF NOT EXISTS provider_health (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unknown'
        CHECK(status IN ('healthy', 'unhealthy', 'disabled', 'unknown')),
    latency_ms REAL,
    last_check TEXT,
    failure_count INTEGER DEFAULT 0,
    quota_remaining INTEGER,
    daily_cost_estimate REAL DEFAULT 0.0,
    daily_calls INTEGER DEFAULT 0,
    last_reset_date TEXT,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_provider_health_name
    ON provider_health(provider_name);

-- =====================================================================
-- Phase 3: Model Router
-- =====================================================================

CREATE TABLE IF NOT EXISTS route_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query_summary TEXT,
    complexity TEXT CHECK(complexity IN ('low', 'medium', 'high', 'extreme')),
    privacy TEXT CHECK(privacy IN ('public', 'personal', 'sensitive')),
    latency_priority TEXT CHECK(latency_priority IN ('realtime', 'normal', 'deep_thinking')),
    cost_priority TEXT CHECK(cost_priority IN ('minimize_cost', 'balanced', 'quality_first')),
    selected_provider TEXT NOT NULL,
    selected_model TEXT NOT NULL,
    reason TEXT,
    response_time_ms INTEGER,
    confidence REAL,
    fallback_used INTEGER DEFAULT 0,
    conversation_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_route_logs_timestamp
    ON route_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_route_logs_provider
    ON route_logs(selected_provider);

-- =====================================================================
-- Phase 3: Cloud LLM
-- =====================================================================

CREATE TABLE IF NOT EXISTS cloud_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0.0,
    timestamp TEXT NOT NULL,
    query_hash TEXT,
    conversation_id TEXT,
    compressed INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cloud_usage_timestamp
    ON cloud_usage(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_cloud_usage_provider
    ON cloud_usage(provider);
CREATE INDEX IF NOT EXISTS idx_cloud_usage_hash
    ON cloud_usage(query_hash);

CREATE TABLE IF NOT EXISTS cloud_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash TEXT NOT NULL,
    query_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_hit_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_cloud_cache_hash
    ON cloud_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_cloud_cache_created
    ON cloud_cache(created_at DESC);

-- =====================================================================
-- Phase 3: Analytics Engine
-- =====================================================================

CREATE TABLE IF NOT EXISTS weekly_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start TEXT NOT NULL,
    week_end TEXT NOT NULL,
    report_json TEXT NOT NULL,
    insight TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_weekly_reports_week
    ON weekly_reports(week_start DESC);

-- =====================================================================
-- Phase 3: Proactive Layer
-- =====================================================================

CREATE TABLE IF NOT EXISTS inbox (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL DEFAULT 'notification'
        CHECK(type IN (
            'morning_briefing', 'evening_nudge', 'reminder',
            'alert', 'notification', 'weekly_report'
        )),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    read_status INTEGER DEFAULT 0,
    priority TEXT DEFAULT 'normal'
        CHECK(priority IN ('low', 'normal', 'high', 'urgent')),
    source TEXT DEFAULT 'system',
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_inbox_read_status
    ON inbox(read_status);
CREATE INDEX IF NOT EXISTS idx_inbox_timestamp
    ON inbox(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_inbox_type
    ON inbox(type);

-- =====================================================================
-- Phase 3: Explainability Engine
-- =====================================================================

CREATE TABLE IF NOT EXISTS decision_trace (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    decision_type TEXT NOT NULL
        CHECK(decision_type IN (
            'routing', 'provider_selection', 'memory_retrieval',
            'context_assembly', 'action_execution', 'cloud_escalation'
        )),
    input_summary TEXT,
    result_json TEXT,
    trace_json TEXT,
    response_time_ms INTEGER,
    conversation_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_decision_trace_timestamp
    ON decision_trace(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_trace_type
    ON decision_trace(decision_type);
"""


# ---------------------------------------------------------------------------
# Memory type expansion migration
# ---------------------------------------------------------------------------

_MEMORY_TYPE_MIGRATION_SQL = """
-- SQLite doesn't support ALTER TABLE to modify CHECK constraints.
-- We create a new table with the expanded constraint and migrate data.
-- This is wrapped in a transaction for safety.

CREATE TABLE IF NOT EXISTS memories_v2 (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL CHECK(memory_type IN (
        'fact', 'summary', 'preference', 'goal', 'project',
        'habit', 'routine', 'deadline', 'observation',
        'academic', 'insight'
    )),
    importance REAL DEFAULT 0.5 CHECK(importance >= 0.0 AND importance <= 1.0),
    created_at TEXT NOT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    category TEXT,
    related_goal_or_project TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
"""


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

def run_migrations(db_path: str = None) -> dict:
    """
    Run all Phase 3 database migrations.

    Args:
        db_path: Optional path to SQLite database.
                 If None, uses the configured path from settings.

    Returns:
        Dict with migration results: tables_created, errors, status.
    """
    if db_path is None:
        settings = get_settings()
        db_path = str(settings.resolve_path(settings.database.path))

    results = {
        "tables_created": [],
        "migrations_applied": [],
        "errors": [],
        "status": "success",
    }

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")

        # --- Phase 3 new tables ---
        logger.info("Running Phase 3 table migrations...")
        conn.executescript(_PHASE3_TABLES_SQL)
        results["tables_created"].extend([
            "provider_health", "route_logs", "cloud_usage",
            "cloud_cache", "weekly_reports", "inbox", "decision_trace",
        ])
        logger.info("Phase 3 tables created/verified: %s", results["tables_created"])

        # --- Memory type expansion ---
        # Check if migration already completed by looking for the backup table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_v1_backup'"
        )
        if cursor.fetchone() is None:
            logger.info("Expanding memories table with new categories...")
            conn.executescript(_MEMORY_TYPE_MIGRATION_SQL)

            # Check if the old memories table has data to migrate
            old_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if old_count > 0:
                # Copy data from old table to new
                conn.execute("""
                    INSERT OR IGNORE INTO memories_v2
                        (id, conversation_id, content, memory_type, importance,
                         created_at, last_accessed, access_count, metadata)
                    SELECT id, conversation_id, content, memory_type, importance,
                           created_at, last_accessed, access_count, metadata
                    FROM memories
                """)
                logger.info("Migrated %d memories to expanded schema", old_count)

            # Rename tables: old → backup, new → memories
            conn.execute("ALTER TABLE memories RENAME TO memories_v1_backup")
            conn.execute("ALTER TABLE memories_v2 RENAME TO memories")

            # Recreate indexes on the new table
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_memories_type
                    ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_importance
                    ON memories(importance DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_created
                    ON memories(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memories_category
                    ON memories(category);
            """)

            results["migrations_applied"].append("memories_type_expansion")
            logger.info("Memory type expansion migration completed")
        else:
            logger.info("Memory type expansion migration already applied")
            results["migrations_applied"].append("memories_type_expansion (already done)")
            
            # Clean up memories_v2 if it was left behind in a partially failed migration
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_v2'"
            )
            if cursor.fetchone() is not None:
                conn.execute("DROP TABLE memories_v2")
                logger.info("Cleaned up orphaned memories_v2 table")

        # --- Add new columns to existing tables if missing ---
        _safe_add_column(conn, "memories", "category", "TEXT")
        _safe_add_column(conn, "memories", "related_goal_or_project", "TEXT")

        conn.commit()
        conn.close()

        logger.info("All Phase 3 migrations completed successfully")

    except Exception as e:
        results["errors"].append(str(e))
        results["status"] = "error"
        logger.error("Migration failed: %s", str(e), exc_info=True)

    return results


def _safe_add_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> bool:
    """
    Safely add a column to a table. Returns True if added, False if already exists.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        logger.debug("Added column %s.%s (%s)", table, column, col_type)
        return True
    except sqlite3.OperationalError:
        # Column already exists
        return False


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_migrations(db_path: str = None) -> dict:
    """
    Verify that all Phase 3 tables exist and have the correct structure.

    Returns:
        Dict with table names and their column counts.
    """
    if db_path is None:
        settings = get_settings()
        db_path = str(settings.resolve_path(settings.database.path))

    expected_tables = [
        "provider_health", "route_logs", "cloud_usage",
        "cloud_cache", "weekly_reports", "inbox", "decision_trace",
    ]

    conn = sqlite3.connect(db_path)
    results = {}

    for table in expected_tables:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        if columns:
            results[table] = {
                "exists": True,
                "column_count": len(columns),
                "columns": [col[1] for col in columns],
            }
        else:
            results[table] = {"exists": False}

    # Check memories table has new columns
    cursor = conn.execute("PRAGMA table_info(memories)")
    mem_columns = [col[1] for col in cursor.fetchall()]
    results["memories_expanded"] = {
        "has_category": "category" in mem_columns,
        "has_related_goal": "related_goal_or_project" in mem_columns,
    }

    conn.close()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from backend.logger import initialize_logging
    initialize_logging()

    print("=== JARVIS Phase 3 Database Migrations ===\n")

    results = run_migrations()
    print(f"Status: {results['status']}")
    print(f"Tables created/verified: {results['tables_created']}")
    print(f"Migrations applied: {results['migrations_applied']}")

    if results["errors"]:
        print(f"Errors: {results['errors']}")

    print("\n=== Verification ===\n")
    verification = verify_migrations()
    for table, info in verification.items():
        if isinstance(info, dict) and info.get("exists"):
            print(f"  ✓ {table}: {info['column_count']} columns")
        elif isinstance(info, dict) and "has_category" in info:
            print(f"  {'✓' if info['has_category'] else '✗'} memories.category column")
            print(f"  {'✓' if info['has_related_goal'] else '✗'} memories.related_goal_or_project column")
        else:
            print(f"  ✗ {table}: MISSING")
