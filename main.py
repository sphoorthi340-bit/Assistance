"""
Jarvis V2 — Terminal Interface
=================================
Primary entry point for the Jarvis personal cognitive assistant.

This is a terminal-first chat loop that:
1. Initializes all subsystems (config, logging, database, LLM, memory, state)
2. Runs health checks (Ollama connectivity, database, ChromaDB)
3. Enters a persistent chat loop
4. Extracts and stores memories after each exchange
5. Retrieves relevant context before each response
6. Supports slash commands for system interaction

Phase 1 Commands (Memory & Chat):
    /new        — Start a new conversation
    /history    — Show conversation history
    /memories   — Show stored memories
    /remember   — Manually store a memory (e.g., /remember I prefer Python)
    /forget     — Delete a memory by ID
    /help       — Show available commands
    /quit       — Exit Jarvis (conversation is auto-saved)

Phase 2 Commands (Personal State):
    /goal add|list|update|complete|pause|resume|delete
    /habit add|list|log|stats|deactivate|activate|delete
    /project add|list|update|status|complete|delete
    /task add|list|complete|delete
    /stats      — Show system statistics (extended)
    /accountability — Run the accountability report

The terminal interface uses the `rich` library for clean, readable output
with panels, colors, and markdown rendering.
"""

import sys
import signal

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich import box

from backend.logger import initialize_logging, get_logger
from backend.database import DatabaseManager
from backend.llm import OllamaClient
from backend.context import ContextBuilder
from memory.vector_store import VectorStore
from memory.extractor import MemoryExtractor
from memory.manager import MemoryManager
from state.goal_manager import GoalManager
from state.habit_manager import HabitManager
from state.project_manager import ProjectManager
from state.analytics_manager import AnalyticsManager
from configs.settings import get_settings

# Initialize logging FIRST — before anything else
initialize_logging()
logger = get_logger(__name__)

# Rich console for terminal output
console = Console()


# ---------------------------------------------------------------------------
# Startup and initialization
# ---------------------------------------------------------------------------

def initialize_system():
    """
    Initialize all Jarvis subsystems in dependency order.

    Returns:
        Dict of all initialized subsystems.

    Raises:
        SystemExit if critical components fail to initialize.
    """
    console.print()
    console.print(
        Panel(
            "[bold cyan]JARVIS[/bold cyan] [dim]v2.0[/dim]\n"
            "[dim]Personal Cognitive Infrastructure[/dim]",
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 4),
        )
    )
    console.print()

    settings = get_settings()

    # --- Database ---
    console.print("  [dim]→[/dim] Initializing database...", end="")
    try:
        db = DatabaseManager(settings=settings)
        console.print(" [green]✓[/green]")
    except Exception as e:
        console.print(f" [red]✗[/red] {e}")
        logger.critical("Database initialization failed: %s", str(e))
        sys.exit(1)

    # --- LLM Client ---
    console.print("  [dim]→[/dim] Connecting to Ollama...", end="")
    try:
        llm = OllamaClient(settings=settings)
        health = llm.check_health()
        if health["status"] == "healthy":
            console.print(f" [green]✓[/green] [dim]({health['model']})[/dim]")
        else:
            console.print(f" [yellow]⚠[/yellow] {health.get('error', 'Unknown issue')}")
            console.print(
                "  [yellow]  Warning: LLM may not be available. "
                "Ensure Ollama is running and the model is pulled.[/yellow]"
            )
            logger.warning("LLM health check failed: %s", health)
    except Exception as e:
        console.print(f" [red]✗[/red] {e}")
        console.print("  [red]  Error: Cannot connect to Ollama. Is it running?[/red]")
        logger.error("LLM initialization failed: %s", str(e))
        # Don't exit — let the user try to fix Ollama while Jarvis is running
        llm = OllamaClient(settings=settings)

    # --- Vector Store ---
    console.print("  [dim]→[/dim] Initializing vector memory...", end="")
    try:
        vector_store = VectorStore(settings=settings)
        console.print(
            f" [green]✓[/green] [dim]({vector_store.get_count()} entries)[/dim]"
        )
    except Exception as e:
        console.print(f" [red]✗[/red] {e}")
        logger.critical("Vector store initialization failed: %s", str(e))
        sys.exit(1)

    # --- Memory System ---
    console.print("  [dim]→[/dim] Initializing memory system...", end="")
    try:
        extractor = MemoryExtractor(llm_client=llm)
        memory_manager = MemoryManager(
            db=db,
            vector_store=vector_store,
            extractor=extractor,
            settings=settings,
        )
        console.print(" [green]✓[/green]")
    except Exception as e:
        console.print(f" [red]✗[/red] {e}")
        logger.critical("Memory system initialization failed: %s", str(e))
        sys.exit(1)

    # --- Context Builder ---
    context_builder = ContextBuilder(
        db=db,
        memory_manager=memory_manager,
        settings=settings,
    )

    # --- Phase 2: State Managers ---
    console.print("  [dim]→[/dim] Initializing state systems...", end="")
    try:
        goal_manager = GoalManager(db=db)
        habit_manager = HabitManager(db=db)
        project_manager = ProjectManager(db=db)
        analytics_manager = AnalyticsManager(db=db)
        console.print(" [green]✓[/green]")
    except Exception as e:
        console.print(f" [red]✗[/red] {e}")
        logger.critical("State system initialization failed: %s", str(e))
        sys.exit(1)

    # --- Summary ---
    db_stats = db.get_stats()
    console.print()
    console.print(
        f"  [dim]Database: {db_stats['conversations']} conversations, "
        f"{db_stats['messages']} messages, "
        f"{db_stats['memories']} memories[/dim]"
    )
    console.print(
        f"  [dim]State: {db_stats['goals']} goals, "
        f"{db_stats['habits']} habits, "
        f"{db_stats['projects']} projects[/dim]"
    )
    console.print("  [dim]Type /help for commands. Type /quit to exit.[/dim]")
    console.print()

    logger.info("System fully initialized")
    return {
        "settings": settings,
        "db": db,
        "llm": llm,
        "memory_manager": memory_manager,
        "context_builder": context_builder,
        "goal_manager": goal_manager,
        "habit_manager": habit_manager,
        "project_manager": project_manager,
        "analytics_manager": analytics_manager,
    }


# ---------------------------------------------------------------------------
# Phase 1 slash command handlers
# ---------------------------------------------------------------------------

def handle_new_conversation(db: DatabaseManager) -> str:
    """Create a new conversation and return its ID."""
    conv_id = db.create_conversation()
    console.print(
        f"\n  [cyan]New conversation started.[/cyan] "
        f"[dim](ID: {conv_id[:8]}...)[/dim]\n"
    )
    return conv_id


def handle_history(db: DatabaseManager, conversation_id: str):
    """Display conversation history."""
    messages = db.get_conversation_messages(conversation_id, limit=20)
    if not messages:
        console.print("\n  [dim]No messages in this conversation yet.[/dim]\n")
        return

    console.print()
    table = Table(
        title="Conversation History",
        box=box.ROUNDED,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Role", style="bold", width=10)
    table.add_column("Message", ratio=1)
    table.add_column("Time", style="dim", width=19)

    for msg in messages:
        role_style = "cyan" if msg["role"] == "user" else "green"
        content = msg["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        table.add_row(
            f"[{role_style}]{msg['role']}[/{role_style}]",
            content,
            msg["timestamp"][:19],
        )

    console.print(table)
    console.print()


def handle_memories(memory_manager: MemoryManager):
    """Display all stored memories."""
    memories = memory_manager.get_all_memories(limit=30)
    if not memories:
        console.print("\n  [dim]No memories stored yet.[/dim]\n")
        return

    console.print()
    table = Table(
        title="Stored Memories",
        box=box.ROUNDED,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("ID", style="dim", width=10)
    table.add_column("Type", style="bold cyan", width=12)
    table.add_column("Content", ratio=1)
    table.add_column("Imp.", width=5, justify="center")
    table.add_column("Accessed", style="dim", width=5, justify="center")

    for mem in memories:
        content = mem["content"]
        if len(content) > 120:
            content = content[:120] + "..."
        table.add_row(
            mem["id"][:8] + "...",
            mem["memory_type"],
            content,
            f"{mem['importance']:.1f}",
            str(mem.get("access_count", 0)),
        )

    console.print(table)
    console.print()


def handle_remember(memory_manager: MemoryManager, text: str):
    """Manually store a memory."""
    if not text.strip():
        console.print("\n  [yellow]Usage: /remember <something to remember>[/yellow]\n")
        return

    result = memory_manager.store_manual_memory(content=text.strip())
    console.print(
        f"\n  [green]✓ Memory stored[/green] "
        f"[dim]([{result['type']}] {result['id'][:8]}...)[/dim]\n"
    )


def handle_forget(memory_manager: MemoryManager, memory_id: str):
    """Delete a memory by ID prefix."""
    if not memory_id.strip():
        console.print("\n  [yellow]Usage: /forget <memory-id-prefix>[/yellow]\n")
        return

    # Search for memories matching the prefix
    all_memories = memory_manager.get_all_memories(limit=200)
    matches = [m for m in all_memories if m["id"].startswith(memory_id.strip())]

    if not matches:
        console.print(f"\n  [yellow]No memory found matching '{memory_id.strip()}'[/yellow]\n")
        return

    if len(matches) > 1:
        console.print(f"\n  [yellow]Multiple matches — be more specific:[/yellow]")
        for m in matches:
            console.print(f"    {m['id'][:12]}  [{m['memory_type']}] {m['content'][:60]}")
        console.print()
        return

    memory = matches[0]
    deleted = memory_manager.delete_memory(memory["id"])
    if deleted:
        console.print(
            f"\n  [green]✓ Deleted memory:[/green] "
            f"[dim]{memory['content'][:60]}[/dim]\n"
        )
    else:
        console.print(f"\n  [red]Failed to delete memory.[/red]\n")


# ---------------------------------------------------------------------------
# Phase 2 slash command handlers — Goals
# ---------------------------------------------------------------------------

def handle_goal(goal_manager: GoalManager, subcommand: str, args: str):
    """Dispatch /goal subcommands."""
    if subcommand == "add":
        _goal_add(goal_manager, args)
    elif subcommand == "list":
        _goal_list(goal_manager, args)
    elif subcommand == "update":
        _goal_update(goal_manager, args)
    elif subcommand == "complete":
        _goal_complete(goal_manager, args)
    elif subcommand == "pause":
        _goal_pause(goal_manager, args)
    elif subcommand == "resume":
        _goal_resume(goal_manager, args)
    elif subcommand == "delete":
        _goal_delete(goal_manager, args)
    else:
        console.print(
            f"\n  [yellow]Unknown subcommand: /goal {subcommand}[/yellow]\n"
            "  [dim]Available: add, list, update, complete, pause, resume, delete[/dim]\n"
        )


def _goal_add(goal_manager: GoalManager, args: str):
    """Add a new goal."""
    title = args.strip()
    if not title:
        console.print("\n  [yellow]Usage: /goal add <title>[/yellow]\n")
        return

    goal = goal_manager.add_goal(title=title)
    console.print(
        f"\n  [green]✓ Goal created:[/green] {goal['title']} "
        f"[dim](ID: {goal['id'][:8]}...)[/dim]\n"
        f"  [dim]Use /goal update {goal['id'][:8]} <field> <value> to set details.[/dim]\n"
    )


def _goal_list(goal_manager: GoalManager, args: str):
    """List goals."""
    status = args.strip() if args.strip() else "active"
    goals = goal_manager.list_goals(status=status)

    if not goals:
        console.print(f"\n  [dim]No {status} goals found.[/dim]\n")
        return

    console.print()
    table = Table(
        title=f"Goals ({status})",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", ratio=1)
    table.add_column("Category", style="cyan", width=12)
    table.add_column("Priority", width=8, justify="center")
    table.add_column("Status", width=10)
    table.add_column("Deadline", style="dim", width=12)

    for g in goals:
        priority_style = "red" if g["priority"] <= 2 else "yellow" if g["priority"] == 3 else "dim"
        deadline = g["deadline"][:10] if g["deadline"] else "—"
        table.add_row(
            g["id"][:8] + "..",
            g["title"],
            g["category"],
            f"[{priority_style}]{g['priority']}[/{priority_style}]",
            g["status"],
            deadline,
        )

    console.print(table)
    console.print()


def _goal_update(goal_manager: GoalManager, args: str):
    """Update a goal field."""
    parts = args.strip().split(maxsplit=2)
    if len(parts) < 3:
        console.print(
            "\n  [yellow]Usage: /goal update <id> <field> <value>[/yellow]\n"
            "  [dim]Fields: title, description, category, target_type, "
            "target_value, priority, deadline[/dim]\n"
        )
        return

    goal_id_prefix, field, value = parts
    # Type conversion for numeric fields
    if field in ("priority",):
        try:
            value = int(value)
        except ValueError:
            console.print(f"\n  [red]{field} must be an integer.[/red]\n")
            return
    elif field in ("target_value", "current_value"):
        try:
            value = float(value)
        except ValueError:
            console.print(f"\n  [red]{field} must be a number.[/red]\n")
            return

    result = goal_manager.update_goal(goal_id_prefix, **{field: value})
    if result:
        console.print(f"\n  [green]✓ Goal updated:[/green] {field} → {value}\n")
    else:
        console.print(f"\n  [yellow]Goal not found: {goal_id_prefix}[/yellow]\n")


def _goal_complete(goal_manager: GoalManager, args: str):
    """Complete a goal."""
    goal_id = args.strip()
    if not goal_id:
        console.print("\n  [yellow]Usage: /goal complete <id>[/yellow]\n")
        return
    result = goal_manager.complete_goal(goal_id)
    if result:
        console.print(f"\n  [green]✓ Goal completed:[/green] {result['title']}\n")
    else:
        console.print(f"\n  [yellow]Goal not found: {goal_id}[/yellow]\n")


def _goal_pause(goal_manager: GoalManager, args: str):
    """Pause a goal."""
    goal_id = args.strip()
    if not goal_id:
        console.print("\n  [yellow]Usage: /goal pause <id>[/yellow]\n")
        return
    result = goal_manager.pause_goal(goal_id)
    if result:
        console.print(f"\n  [green]✓ Goal paused:[/green] {result['title']}\n")
    else:
        console.print(f"\n  [yellow]Goal not found: {goal_id}[/yellow]\n")


def _goal_resume(goal_manager: GoalManager, args: str):
    """Resume a paused goal."""
    goal_id = args.strip()
    if not goal_id:
        console.print("\n  [yellow]Usage: /goal resume <id>[/yellow]\n")
        return
    result = goal_manager.resume_goal(goal_id)
    if result:
        console.print(f"\n  [green]✓ Goal resumed:[/green] {result['title']}\n")
    else:
        console.print(f"\n  [yellow]Goal not found: {goal_id}[/yellow]\n")


def _goal_delete(goal_manager: GoalManager, args: str):
    """Delete a goal."""
    goal_id = args.strip()
    if not goal_id:
        console.print("\n  [yellow]Usage: /goal delete <id>[/yellow]\n")
        return
    deleted = goal_manager.delete_goal(goal_id)
    if deleted:
        console.print(f"\n  [green]✓ Goal deleted.[/green]\n")
    else:
        console.print(f"\n  [yellow]Goal not found: {goal_id}[/yellow]\n")


# ---------------------------------------------------------------------------
# Phase 2 slash command handlers — Habits
# ---------------------------------------------------------------------------

def handle_habit(habit_manager: HabitManager, subcommand: str, args: str):
    """Dispatch /habit subcommands."""
    if subcommand == "add":
        _habit_add(habit_manager, args)
    elif subcommand == "list":
        _habit_list(habit_manager)
    elif subcommand == "log":
        _habit_log(habit_manager, args)
    elif subcommand == "stats":
        _habit_stats(habit_manager, args)
    elif subcommand == "deactivate":
        _habit_deactivate(habit_manager, args)
    elif subcommand == "activate":
        _habit_activate(habit_manager, args)
    elif subcommand == "delete":
        _habit_delete(habit_manager, args)
    else:
        console.print(
            f"\n  [yellow]Unknown subcommand: /habit {subcommand}[/yellow]\n"
            "  [dim]Available: add, list, log, stats, deactivate, activate, delete[/dim]\n"
        )


def _habit_add(habit_manager: HabitManager, args: str):
    """Add a new habit."""
    name = args.strip()
    if not name:
        console.print("\n  [yellow]Usage: /habit add <name>[/yellow]\n")
        return

    habit = habit_manager.add_habit(name=name)
    console.print(
        f"\n  [green]✓ Habit created:[/green] {habit['name']} "
        f"[dim](ID: {habit['id'][:8]}..., frequency: {habit['frequency']})[/dim]\n"
    )


def _habit_list(habit_manager: HabitManager):
    """List active habits."""
    habits = habit_manager.list_habits(active_only=True)

    if not habits:
        console.print("\n  [dim]No active habits found.[/dim]\n")
        return

    console.print()
    table = Table(
        title="Active Habits",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("ID", style="dim", width=10)
    table.add_column("Name", ratio=1)
    table.add_column("Frequency", style="cyan", width=10)
    table.add_column("Category", width=12)
    table.add_column("Target", width=8, justify="center")

    for h in habits:
        table.add_row(
            h["id"][:8] + "..",
            h["name"],
            h["frequency"],
            h["category"],
            str(int(h["target_per_period"])),
        )

    console.print(table)
    console.print()


def _habit_log(habit_manager: HabitManager, args: str):
    """Log a habit completion."""
    parts = args.strip().split(maxsplit=1)
    if not parts:
        console.print("\n  [yellow]Usage: /habit log <name-or-id> [notes][/yellow]\n")
        return

    identifier = parts[0]
    notes = parts[1] if len(parts) > 1 else None

    result = habit_manager.log_habit(identifier, notes=notes)
    if result:
        console.print(
            f"\n  [green]✓ Habit logged:[/green] {result.get('habit_name', identifier)} "
            f"[dim]({result['date']})[/dim]\n"
        )
    else:
        console.print(f"\n  [yellow]Habit not found: {identifier}[/yellow]\n")


def _habit_stats(habit_manager: HabitManager, args: str):
    """Show habit stats (streak, completion rate)."""
    identifier = args.strip()

    if identifier:
        # Stats for a specific habit
        habit = habit_manager.get_habit(identifier)
        if not habit:
            console.print(f"\n  [yellow]Habit not found: {identifier}[/yellow]\n")
            return

        streak = habit_manager.get_streak(habit["id"])
        console.print()
        console.print(Panel(
            f"[bold]{habit['name']}[/bold] [dim]({habit['frequency']})[/dim]\n\n"
            f"  Current Streak:  [bold cyan]{streak['current_streak']}[/bold cyan] days\n"
            f"  Longest Streak:  [bold]{streak['longest_streak']}[/bold] days\n"
            f"  Last Logged:     {streak['last_logged'] or 'Never'}",
            title="Habit Stats",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        console.print()
    else:
        # Stats for all habits
        habits = habit_manager.list_habits(active_only=True)
        if not habits:
            console.print("\n  [dim]No active habits found.[/dim]\n")
            return

        console.print()
        table = Table(
            title="Habit Statistics",
            box=box.ROUNDED,
            border_style="dim",
        )
        table.add_column("Habit", ratio=1)
        table.add_column("Streak", width=8, justify="center")
        table.add_column("Best", width=8, justify="center")
        table.add_column("Last Log", style="dim", width=12)

        for h in habits:
            streak = habit_manager.get_streak(h["id"])
            streak_style = "green" if streak["current_streak"] > 0 else "red"
            table.add_row(
                h["name"],
                f"[{streak_style}]{streak['current_streak']}d[/{streak_style}]",
                f"{streak['longest_streak']}d",
                streak["last_logged"] or "Never",
            )

        console.print(table)
        console.print()


def _habit_deactivate(habit_manager: HabitManager, args: str):
    """Deactivate a habit."""
    identifier = args.strip()
    if not identifier:
        console.print("\n  [yellow]Usage: /habit deactivate <name-or-id>[/yellow]\n")
        return
    result = habit_manager.deactivate_habit(identifier)
    if result:
        console.print(f"\n  [green]✓ Habit deactivated:[/green] {result['name']}\n")
    else:
        console.print(f"\n  [yellow]Habit not found: {identifier}[/yellow]\n")


def _habit_activate(habit_manager: HabitManager, args: str):
    """Activate a habit."""
    identifier = args.strip()
    if not identifier:
        console.print("\n  [yellow]Usage: /habit activate <name-or-id>[/yellow]\n")
        return
    result = habit_manager.activate_habit(identifier)
    if result:
        console.print(f"\n  [green]✓ Habit activated:[/green] {result['name']}\n")
    else:
        console.print(f"\n  [yellow]Habit not found: {identifier}[/yellow]\n")


def _habit_delete(habit_manager: HabitManager, args: str):
    """Delete a habit."""
    identifier = args.strip()
    if not identifier:
        console.print("\n  [yellow]Usage: /habit delete <name-or-id>[/yellow]\n")
        return
    deleted = habit_manager.delete_habit(identifier)
    if deleted:
        console.print(f"\n  [green]✓ Habit deleted.[/green]\n")
    else:
        console.print(f"\n  [yellow]Habit not found: {identifier}[/yellow]\n")


# ---------------------------------------------------------------------------
# Phase 2 slash command handlers — Projects
# ---------------------------------------------------------------------------

def handle_project(project_manager: ProjectManager, subcommand: str, args: str):
    """Dispatch /project subcommands."""
    if subcommand == "add":
        _project_add(project_manager, args)
    elif subcommand == "list":
        _project_list(project_manager, args)
    elif subcommand == "update":
        _project_update(project_manager, args)
    elif subcommand == "status":
        _project_status(project_manager, args)
    elif subcommand == "complete":
        _project_complete(project_manager, args)
    elif subcommand == "delete":
        _project_delete(project_manager, args)
    else:
        console.print(
            f"\n  [yellow]Unknown subcommand: /project {subcommand}[/yellow]\n"
            "  [dim]Available: add, list, update, status, complete, delete[/dim]\n"
        )


def _project_add(project_manager: ProjectManager, args: str):
    """Add a new project."""
    name = args.strip()
    if not name:
        console.print("\n  [yellow]Usage: /project add <name>[/yellow]\n")
        return

    project = project_manager.add_project(name=name)
    console.print(
        f"\n  [green]✓ Project created:[/green] {project['name']} "
        f"[dim](ID: {project['id'][:8]}...)[/dim]\n"
    )


def _project_list(project_manager: ProjectManager, args: str):
    """List projects."""
    status = args.strip() if args.strip() else "active"
    projects = project_manager.list_projects(status=status)

    if not projects:
        console.print(f"\n  [dim]No {status} projects found.[/dim]\n")
        return

    console.print()
    table = Table(
        title=f"Projects ({status})",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("ID", style="dim", width=10)
    table.add_column("Name", ratio=1)
    table.add_column("Progress", width=10, justify="center")
    table.add_column("Priority", width=8, justify="center")
    table.add_column("Blocker", style="red", width=20)

    for p in projects:
        progress = f"{p['progress_percentage']:.0f}%"
        blocker = (p["current_blocker"] or "—")[:20]
        table.add_row(
            p["id"][:8] + "..",
            p["name"],
            progress,
            str(p["priority"]),
            blocker,
        )

    console.print(table)
    console.print()


def _project_update(project_manager: ProjectManager, args: str):
    """Update a project field."""
    parts = args.strip().split(maxsplit=2)
    if len(parts) < 3:
        console.print(
            "\n  [yellow]Usage: /project update <id> <field> <value>[/yellow]\n"
            "  [dim]Fields: name, description, status, progress_percentage, "
            "current_blocker, next_step, priority[/dim]\n"
        )
        return

    project_id_prefix, field, value = parts
    if field in ("priority",):
        try:
            value = int(value)
        except ValueError:
            console.print(f"\n  [red]{field} must be an integer.[/red]\n")
            return
    elif field in ("progress_percentage",):
        try:
            value = float(value)
        except ValueError:
            console.print(f"\n  [red]{field} must be a number.[/red]\n")
            return

    result = project_manager.update_project(project_id_prefix, **{field: value})
    if result:
        console.print(f"\n  [green]✓ Project updated:[/green] {field} → {value}\n")
    else:
        console.print(f"\n  [yellow]Project not found: {project_id_prefix}[/yellow]\n")


def _project_status(project_manager: ProjectManager, args: str):
    """Show detailed status for a project or all projects."""
    identifier = args.strip()

    if identifier:
        summary = project_manager.get_project_summary(identifier)
        if not summary:
            console.print(f"\n  [yellow]Project not found: {identifier}[/yellow]\n")
            return

        console.print()
        status_color = "green" if summary["status"] == "active" else "yellow"
        panel_content = (
            f"[bold]{summary['name']}[/bold] "
            f"[{status_color}]({summary['status']})[/{status_color}]\n\n"
            f"  Progress:      [{summary['progress_percentage']:.0f}%]\n"
            f"  Total Tasks:   {summary['total_tasks']}\n"
            f"  Completed:     {summary['completed_tasks']}\n"
            f"  In Progress:   {summary['in_progress_tasks']}\n"
            f"  Pending:       {summary['pending_tasks']}\n"
            f"  Overdue:       {summary['overdue_tasks']}\n"
        )
        if summary.get("current_blocker"):
            panel_content += f"\n  [red]Blocker: {summary['current_blocker']}[/red]\n"
        if summary.get("next_step"):
            panel_content += f"  [cyan]Next: {summary['next_step']}[/cyan]\n"

        console.print(Panel(
            panel_content,
            title="Project Status",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        console.print()
    else:
        # Show all active projects briefly
        _project_list(project_manager, "active")


def _project_complete(project_manager: ProjectManager, args: str):
    """Complete a project."""
    project_id = args.strip()
    if not project_id:
        console.print("\n  [yellow]Usage: /project complete <id>[/yellow]\n")
        return
    result = project_manager.complete_project(project_id)
    if result:
        console.print(f"\n  [green]✓ Project completed:[/green] {result['name']}\n")
    else:
        console.print(f"\n  [yellow]Project not found: {project_id}[/yellow]\n")


def _project_delete(project_manager: ProjectManager, args: str):
    """Delete a project."""
    project_id = args.strip()
    if not project_id:
        console.print("\n  [yellow]Usage: /project delete <id>[/yellow]\n")
        return
    deleted = project_manager.delete_project(project_id)
    if deleted:
        console.print(f"\n  [green]✓ Project deleted.[/green]\n")
    else:
        console.print(f"\n  [yellow]Project not found: {project_id}[/yellow]\n")


# ---------------------------------------------------------------------------
# Phase 2 slash command handlers — Tasks
# ---------------------------------------------------------------------------

def handle_task(project_manager: ProjectManager, subcommand: str, args: str):
    """Dispatch /task subcommands."""
    if subcommand == "add":
        _task_add(project_manager, args)
    elif subcommand == "list":
        _task_list(project_manager, args)
    elif subcommand == "complete":
        _task_complete(project_manager, args)
    elif subcommand == "delete":
        _task_delete(project_manager, args)
    else:
        console.print(
            f"\n  [yellow]Unknown subcommand: /task {subcommand}[/yellow]\n"
            "  [dim]Available: add, list, complete, delete[/dim]\n"
        )


def _task_add(project_manager: ProjectManager, args: str):
    """Add a task to a project."""
    parts = args.strip().split(maxsplit=1)
    if len(parts) < 2:
        console.print("\n  [yellow]Usage: /task add <project-id> <title>[/yellow]\n")
        return

    project_id_prefix, title = parts
    task = project_manager.add_task(project_id_prefix, title=title)
    if task:
        console.print(
            f"\n  [green]✓ Task added:[/green] {task['title']} "
            f"[dim](ID: {task['id'][:8]}...)[/dim]\n"
        )
    else:
        console.print(f"\n  [yellow]Project not found: {project_id_prefix}[/yellow]\n")


def _task_list(project_manager: ProjectManager, args: str):
    """List tasks for a project."""
    project_id = args.strip()
    if not project_id:
        console.print("\n  [yellow]Usage: /task list <project-id>[/yellow]\n")
        return

    tasks = project_manager.list_tasks(project_id)
    if tasks is None:
        console.print(f"\n  [yellow]Project not found: {project_id}[/yellow]\n")
        return

    if not tasks:
        console.print("\n  [dim]No tasks for this project.[/dim]\n")
        return

    console.print()
    table = Table(
        title="Project Tasks",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", ratio=1)
    table.add_column("Status", width=12)
    table.add_column("Priority", width=8, justify="center")
    table.add_column("Due", style="dim", width=12)

    for t in tasks:
        status_style = {
            "completed": "green",
            "in_progress": "yellow",
            "pending": "dim",
            "cancelled": "red",
        }.get(t["status"], "dim")
        due = t["due_date"][:10] if t["due_date"] else "—"
        table.add_row(
            t["id"][:8] + "..",
            t["title"],
            f"[{status_style}]{t['status']}[/{status_style}]",
            str(t["priority"]),
            due,
        )

    console.print(table)
    console.print()


def _task_complete(project_manager: ProjectManager, args: str):
    """Complete a task."""
    task_id = args.strip()
    if not task_id:
        console.print("\n  [yellow]Usage: /task complete <task-id>[/yellow]\n")
        return
    result = project_manager.complete_task(task_id)
    if result:
        console.print(
            f"\n  [green]✓ Task completed:[/green] {result['title']}\n"
        )
    else:
        console.print(f"\n  [yellow]Task not found: {task_id}[/yellow]\n")


def _task_delete(project_manager: ProjectManager, args: str):
    """Delete a task."""
    task_id = args.strip()
    if not task_id:
        console.print("\n  [yellow]Usage: /task delete <task-id>[/yellow]\n")
        return
    # Resolve prefix
    task = project_manager._resolve_task(task_id)
    if not task:
        console.print(f"\n  [yellow]Task not found: {task_id}[/yellow]\n")
        return
    deleted = project_manager.delete_task(task["id"])
    if deleted:
        console.print(f"\n  [green]✓ Task deleted.[/green]\n")
    else:
        console.print(f"\n  [yellow]Failed to delete task.[/yellow]\n")


# ---------------------------------------------------------------------------
# Phase 2 slash command handlers — Stats & Accountability
# ---------------------------------------------------------------------------

def handle_stats(
    db: DatabaseManager,
    memory_manager: MemoryManager,
    analytics_manager: AnalyticsManager,
):
    """Display system statistics — Phase 1 + Phase 2."""
    mem_stats = memory_manager.get_stats()
    db_stats = mem_stats["database"]
    vec_stats = mem_stats["vector_store"]

    console.print()

    # Phase 1 stats
    panel_content = (
        f"[bold]Database[/bold]\n"
        f"  Conversations: {db_stats['conversations']}\n"
        f"  Messages: {db_stats['messages']}\n"
        f"  Memories: {db_stats['memories']}\n"
    )

    if db_stats.get("memories_by_type"):
        panel_content += "  By type:\n"
        for mtype, count in db_stats["memories_by_type"].items():
            panel_content += f"    {mtype}: {count}\n"

    panel_content += (
        f"\n[bold]Vector Store[/bold]\n"
        f"  Entries: {vec_stats['total_entries']}\n"
        f"  Model: {vec_stats['embedding_model']}\n"
        f"  Collection: {vec_stats['collection_name']}\n"
    )

    # Phase 2 stats
    try:
        dashboard = analytics_manager.get_dashboard_stats()
        panel_content += (
            f"\n[bold]Personal State[/bold]\n"
            f"  Active Goals:    {dashboard['active_goals']}"
        )
        if dashboard.get("overdue_goals", 0) > 0:
            panel_content += f" [red]({dashboard['overdue_goals']} overdue)[/red]"
        panel_content += (
            f"\n  Active Habits:   {dashboard['active_habits']}\n"
            f"  Active Projects: {dashboard['active_projects']}\n"
            f"  Tasks:           {dashboard['completed_tasks']}/{dashboard['total_tasks']} completed"
        )
        if dashboard.get("overdue_tasks", 0) > 0:
            panel_content += f" [red]({dashboard['overdue_tasks']} overdue)[/red]"
        panel_content += f"\n  Consistency:     {dashboard['overall_consistency_score']:.0f}%\n"
    except Exception as e:
        logger.warning("Failed to get dashboard stats: %s", str(e))

    console.print(Panel(
        panel_content,
        title="System Statistics",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print()


def handle_accountability(analytics_manager: AnalyticsManager):
    """Run and display the accountability report."""
    report = analytics_manager.get_accountability_report()

    console.print()

    if not report["observations"]:
        console.print(
            "  [dim]No observations yet. Start tracking goals, habits, "
            "and projects to get insights.[/dim]\n"
        )
        return

    # Build the accountability panel
    content_lines = []

    # Observations
    for obs in report["observations"]:
        content_lines.append(f"  • {obs}")

    content = "\n".join(content_lines)

    # Summary line
    score = report.get("consistency_score", 0)
    if score >= 80:
        score_style = "green"
        score_label = "Strong"
    elif score >= 50:
        score_style = "yellow"
        score_label = "Moderate"
    else:
        score_style = "red"
        score_label = "Needs attention"

    content += f"\n\n  [{score_style}]Overall Consistency: {score:.0f}% ({score_label})[/{score_style}]"

    console.print(Panel(
        content,
        title="Accountability Report",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print()


# ---------------------------------------------------------------------------
# Help command
# ---------------------------------------------------------------------------

def handle_help():
    """Show available commands."""
    console.print()
    table = Table(
        title="Commands",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("Command", style="bold cyan", width=35)
    table.add_column("Description")

    commands = [
        # Memory & Chat
        ("", "[bold]Memory & Chat[/bold]"),
        ("/new", "Start a new conversation"),
        ("/history", "Show conversation history"),
        ("/memories", "Show all stored memories"),
        ("/remember <text>", "Manually store a memory"),
        ("/forget <id>", "Delete a memory by ID prefix"),
        # Goals
        ("", "[bold]Goals[/bold]"),
        ("/goal add <title>", "Add a new goal"),
        ("/goal list [status]", "List goals (default: active)"),
        ("/goal update <id> <field> <val>", "Update a goal field"),
        ("/goal complete <id>", "Mark a goal as completed"),
        ("/goal pause|resume <id>", "Pause or resume a goal"),
        ("/goal delete <id>", "Delete a goal"),
        # Habits
        ("", "[bold]Habits[/bold]"),
        ("/habit add <name>", "Add a new habit to track"),
        ("/habit list", "List active habits"),
        ("/habit log <name-or-id> [notes]", "Log a habit completion"),
        ("/habit stats [name-or-id]", "Show streaks & completion stats"),
        ("/habit deactivate|activate <id>", "Toggle habit active state"),
        ("/habit delete <id>", "Delete a habit"),
        # Projects
        ("", "[bold]Projects[/bold]"),
        ("/project add <name>", "Add a new project"),
        ("/project list [status]", "List projects (default: active)"),
        ("/project update <id> <field> <val>", "Update a project field"),
        ("/project status [id]", "Show project status details"),
        ("/project complete|delete <id>", "Complete or delete a project"),
        # Tasks
        ("", "[bold]Tasks[/bold]"),
        ("/task add <project-id> <title>", "Add a task to a project"),
        ("/task list <project-id>", "List tasks for a project"),
        ("/task complete <task-id>", "Mark a task as completed"),
        ("/task delete <task-id>", "Delete a task"),
        # System
        ("", "[bold]System[/bold]"),
        ("/stats", "Show system statistics"),
        ("/accountability", "Run the accountability report"),
        ("/help", "Show this help message"),
        ("/quit", "Exit Jarvis"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def chat_loop(systems: dict):
    """
    The primary interactive chat loop.

    Flow for each turn:
    1. User types a message
    2. ContextBuilder retrieves memories and builds the prompt
    3. OllamaClient sends to the LLM and streams the response
    4. Response is stored in SQLite
    5. MemoryExtractor processes the user message for memorizable facts
    6. Next turn begins
    """
    db = systems["db"]
    llm = systems["llm"]
    memory_manager = systems["memory_manager"]
    context_builder = systems["context_builder"]
    goal_manager = systems["goal_manager"]
    habit_manager = systems["habit_manager"]
    project_manager = systems["project_manager"]
    analytics_manager = systems["analytics_manager"]

    # Start with a new conversation (or resume if we add that later)
    conversation_id = db.create_conversation()
    console.print(f"  [dim]Session: {conversation_id[:8]}...[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not user_input:
                continue

            # --- Handle slash commands ---
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=2)
                command = parts[0].lower()
                subcommand = parts[1].lower() if len(parts) > 1 else ""
                args = parts[2] if len(parts) > 2 else ""

                # For single-arg commands, treat subcommand as args
                single_arg_commands = {
                    "/quit", "/new", "/history", "/memories",
                    "/remember", "/forget", "/stats",
                    "/accountability", "/help",
                }

                if command in single_arg_commands:
                    # Rejoin subcommand + args as the full argument
                    full_args = user_input.split(maxsplit=1)
                    cmd_args = full_args[1] if len(full_args) > 1 else ""

                if command == "/quit":
                    # Summarize conversation before exiting (if LLM available)
                    messages = db.get_conversation_messages(conversation_id)
                    if len(messages) >= 4:
                        console.print("  [dim]Saving conversation summary...[/dim]")
                        try:
                            memory_manager.summarize_conversation(conversation_id)
                        except Exception as e:
                            logger.warning("Failed to summarize on exit: %s", str(e))
                    console.print("\n  [cyan]Goodbye. Your memories persist.[/cyan]\n")
                    break

                elif command == "/new":
                    old_messages = db.get_conversation_messages(conversation_id)
                    if len(old_messages) >= 4:
                        try:
                            memory_manager.summarize_conversation(conversation_id)
                        except Exception:
                            pass
                    conversation_id = handle_new_conversation(db)

                elif command == "/history":
                    handle_history(db, conversation_id)

                elif command == "/memories":
                    handle_memories(memory_manager)

                elif command == "/remember":
                    handle_remember(memory_manager, cmd_args)

                elif command == "/forget":
                    handle_forget(memory_manager, cmd_args)

                elif command == "/stats":
                    handle_stats(db, memory_manager, analytics_manager)

                elif command == "/accountability":
                    handle_accountability(analytics_manager)

                elif command == "/help":
                    handle_help()

                # --- Phase 2 compound commands ---
                elif command == "/goal":
                    if not subcommand:
                        console.print(
                            "\n  [yellow]Usage: /goal <add|list|update|complete|pause|resume|delete>[/yellow]\n"
                        )
                    else:
                        handle_goal(goal_manager, subcommand, args)

                elif command == "/habit":
                    if not subcommand:
                        console.print(
                            "\n  [yellow]Usage: /habit <add|list|log|stats|deactivate|activate|delete>[/yellow]\n"
                        )
                    else:
                        handle_habit(habit_manager, subcommand, args)

                elif command == "/project":
                    if not subcommand:
                        console.print(
                            "\n  [yellow]Usage: /project <add|list|update|status|complete|delete>[/yellow]\n"
                        )
                    else:
                        handle_project(project_manager, subcommand, args)

                elif command == "/task":
                    if not subcommand:
                        console.print(
                            "\n  [yellow]Usage: /task <add|list|complete|delete>[/yellow]\n"
                        )
                    else:
                        handle_task(project_manager, subcommand, args)

                else:
                    console.print(
                        f"\n  [yellow]Unknown command: {command}. "
                        f"Type /help for available commands.[/yellow]\n"
                    )
                continue

            # --- Normal chat flow ---

            # Step 1: Store user message
            db.add_message(
                conversation_id=conversation_id,
                role="user",
                content=user_input,
            )

            # Step 2: Build context (retrieves memories + history)
            messages = context_builder.build_messages(
                user_message=user_input,
                conversation_id=conversation_id,
            )

            # Step 3: Get LLM response (streaming)
            console.print()
            console.print("[bold green]Jarvis:[/bold green] ", end="")

            try:
                full_response_parts = []
                thinking_content = ""

                stream = llm._client.chat(
                    model=llm._model,
                    messages=messages,
                    stream=True,
                    options={"temperature": llm._temperature},
                )

                # --- Streaming with think-token stripping ---
                raw_buffer = ""
                in_thinking = False
                thinking_buffer = []

                for chunk in stream:
                    token = chunk.message.content or ""
                    raw_buffer += token

                    if not llm._strip_thinking:
                        console.print(token, end="")
                        full_response_parts.append(token)
                        continue

                    # Process the buffer for think tags
                    while True:
                        if not in_thinking:
                            think_start = raw_buffer.find("<think>")
                            if think_start == -1:
                                # No think tag — check for partial tag at end
                                # Output everything except a potential partial tag
                                safe_end = len(raw_buffer)
                                for i in range(1, min(8, len(raw_buffer) + 1)):
                                    if raw_buffer.endswith("<think>"[:i]):
                                        safe_end = len(raw_buffer) - i
                                        break
                                if safe_end > 0:
                                    output = raw_buffer[:safe_end]
                                    console.print(output, end="")
                                    full_response_parts.append(output)
                                    raw_buffer = raw_buffer[safe_end:]
                                break
                            else:
                                # Output text before <think>
                                if think_start > 0:
                                    output = raw_buffer[:think_start]
                                    console.print(output, end="")
                                    full_response_parts.append(output)
                                raw_buffer = raw_buffer[think_start + 7:]
                                in_thinking = True
                        else:
                            think_end = raw_buffer.find("</think>")
                            if think_end == -1:
                                # Still inside thinking — accumulate
                                thinking_buffer.append(raw_buffer)
                                raw_buffer = ""
                                break
                            else:
                                # End of thinking block
                                thinking_buffer.append(raw_buffer[:think_end])
                                raw_buffer = raw_buffer[think_end + 8:]
                                in_thinking = False

                # Flush remaining buffer
                if raw_buffer and not in_thinking:
                    console.print(raw_buffer, end="")
                    full_response_parts.append(raw_buffer)

                thinking_content = "".join(thinking_buffer).strip()

                console.print()  # Newline after response
                console.print()

                # Build clean response
                clean_response = "".join(full_response_parts).strip()

            except Exception as e:
                console.print(f"\n  [red]Error: {str(e)}[/red]\n")
                logger.error("LLM request failed: %s", str(e), exc_info=True)
                continue

            # Step 4: Store assistant response
            db.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=clean_response,
                thinking=thinking_content if thinking_content else None,
            )

            # Step 5: Extract and store memories from user message
            stored = memory_manager.process_message(
                message=user_input,
                role="user",
                conversation_id=conversation_id,
            )

            # Show memory extraction feedback (subtle)
            if stored:
                for mem in stored:
                    console.print(
                        f"  [dim]📝 Memory stored: [{mem['type']}] "
                        f"{mem['content'][:60]}...[/dim]"
                    )
                console.print()

        except KeyboardInterrupt:
            console.print("\n\n  [cyan]Interrupted. Type /quit to exit properly.[/cyan]\n")
            continue

        except Exception as e:
            console.print(f"\n  [red]Unexpected error: {str(e)}[/red]\n")
            logger.error("Chat loop error: %s", str(e), exc_info=True)
            continue


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for Jarvis."""
    # Set up signal handling for graceful shutdown
    def signal_handler(sig, frame):
        console.print("\n\n  [cyan]Shutting down...[/cyan]\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize all subsystems
        systems = initialize_system()

        # Enter the chat loop
        chat_loop(systems)

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"\n  [red]Fatal error: {str(e)}[/red]\n")
        logger.critical("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
