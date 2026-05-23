"""
Jarvis V1 — Terminal Interface
=================================
Primary entry point for the Jarvis personal cognitive assistant.

This is a terminal-first chat loop that:
1. Initializes all subsystems (config, logging, database, LLM, memory)
2. Runs health checks (Ollama connectivity, database, ChromaDB)
3. Enters a persistent chat loop
4. Extracts and stores memories after each exchange
5. Retrieves relevant context before each response
6. Supports slash commands for system interaction

Commands:
    /new        — Start a new conversation
    /history    — Show conversation history
    /memories   — Show stored memories
    /stats      — Show system statistics
    /remember   — Manually store a memory (e.g., /remember I prefer Python)
    /forget     — Delete a memory by ID
    /help       — Show available commands
    /quit       — Exit Jarvis (conversation is auto-saved)

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
        Tuple of (settings, db, llm, memory_manager, context_builder).
    
    Raises:
        SystemExit if critical components fail to initialize.
    """
    console.print()
    console.print(
        Panel(
            "[bold cyan]JARVIS[/bold cyan] [dim]v1.0[/dim]\n"
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

    # --- Summary ---
    db_stats = db.get_stats()
    console.print()
    console.print(
        f"  [dim]Database: {db_stats['conversations']} conversations, "
        f"{db_stats['messages']} messages, "
        f"{db_stats['memories']} memories[/dim]"
    )
    console.print("  [dim]Type /help for commands. Type /quit to exit.[/dim]")
    console.print()

    logger.info("System fully initialized")
    return settings, db, llm, memory_manager, context_builder


# ---------------------------------------------------------------------------
# Slash command handlers
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


def handle_stats(db: DatabaseManager, memory_manager: MemoryManager):
    """Display system statistics."""
    stats = memory_manager.get_stats()
    db_stats = stats["database"]
    vec_stats = stats["vector_store"]

    console.print()
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

    console.print(Panel(
        panel_content,
        title="System Statistics",
        border_style="cyan",
        box=box.ROUNDED,
    ))
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


def handle_help():
    """Show available commands."""
    console.print()
    table = Table(
        title="Commands",
        box=box.ROUNDED,
        border_style="dim",
    )
    table.add_column("Command", style="bold cyan", width=20)
    table.add_column("Description")

    commands = [
        ("/new", "Start a new conversation"),
        ("/history", "Show conversation history"),
        ("/memories", "Show all stored memories"),
        ("/stats", "Show system statistics"),
        ("/remember <text>", "Manually store a memory"),
        ("/forget <id>", "Delete a memory by ID prefix"),
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

def chat_loop(
    db: DatabaseManager,
    llm: OllamaClient,
    memory_manager: MemoryManager,
    context_builder: ContextBuilder,
):
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
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

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
                    # Summarize old conversation before starting new one
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

                elif command == "/stats":
                    handle_stats(db, memory_manager)

                elif command == "/remember":
                    handle_remember(memory_manager, args)

                elif command == "/forget":
                    handle_forget(memory_manager, args)

                elif command == "/help":
                    handle_help()

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
        settings, db, llm, memory_manager, context_builder = initialize_system()

        # Enter the chat loop
        chat_loop(db, llm, memory_manager, context_builder)

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"\n  [red]Fatal error: {str(e)}[/red]\n")
        logger.critical("Fatal error: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
