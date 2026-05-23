# Jarvis V1 — Personal Cognitive Infrastructure

A local-first, persistent AI assistant built as foundational cognitive infrastructure.  
**Not a chatbot.** A system designed for continuity, memory, context preservation, and long-term behavioral intelligence.

## Philosophy

- The LLM is a **reasoning engine**, not memory
- All memory, context, and continuity live in **independent storage systems**
- Precision of retrieval > volume of context
- Clean memory flow > feature completeness
- Stability > sophistication

## Architecture

```
┌─────────────────────────────────────────────────┐
│                Terminal CLI (main.py)            │
├──────────┬──────────┬──────────┬────────────────┤
│ Context  │   LLM    │ Memory   │   Database     │
│ Builder  │ (Ollama) │ Manager  │   (SQLite)     │
│          │          │          │                │
│ Retrieves│ DeepSeek │ Extract  │ Conversations  │
│ memories │ R1 7B    │ Store    │ Messages       │
│ + history│          │ Retrieve │ Memories       │
├──────────┘──────────┤──────────┤────────────────┤
│                     │  Vector  │                │
│                     │  Store   │                │
│                     │ (ChromaDB)                │
└─────────────────────┴──────────┴────────────────┘
```

## Tech Stack

| Component     | Technology                  |
|---------------|-----------------------------|
| Language      | Python 3.11                 |
| LLM Runtime   | Ollama                      |
| LLM Model     | DeepSeek-R1 7B              |
| Database      | SQLite (WAL mode)           |
| Vector DB     | ChromaDB (persistent)       |
| Embeddings    | sentence-transformers (all-MiniLM-L6-v2) |
| Terminal UI   | Rich                        |
| Config        | Pydantic + YAML + .env      |

## Project Structure

```
Jarvis/
├── main.py                  # Terminal chat entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
├── configs/
│   ├── __init__.py
│   ├── config.yaml          # Primary configuration (version-controlled)
│   └── settings.py          # Pydantic settings with 3-tier loading
├── backend/
│   ├── __init__.py
│   ├── logger.py            # Rotating file + console logging
│   ├── database.py          # SQLite manager (conversations, messages, memories)
│   ├── llm.py               # Ollama wrapper with think-token parsing
│   └── context.py           # Prompt assembly with memory injection
├── memory/
│   ├── __init__.py
│   ├── vector_store.py      # ChromaDB semantic search
│   ├── extractor.py         # Heuristic + LLM memory extraction
│   └── manager.py           # Memory lifecycle orchestration
├── vector_db/               # ChromaDB persistent storage (auto-created)
├── data/                    # SQLite database (auto-created)
├── documents/               # Future: PDF/document storage for RAG
├── logs/                    # Rotating log files (auto-created)
└── frontend/                # Future: dashboard UI
```

## Setup

### Prerequisites

1. **Python 3.11+** installed
2. **Ollama** installed and running ([ollama.com](https://ollama.com))
3. **DeepSeek-R1 7B** model pulled:
   ```bash
   ollama pull deepseek-r1:7b
   ```

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Jarvis

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config (optional — defaults work out of the box)
copy .env.example .env
```

### First Run

```bash
python main.py
```

The system will:
1. Initialize SQLite database (`data/jarvis.db`)
2. Connect to Ollama and verify the model
3. Initialize ChromaDB vector store
4. Start the interactive chat loop

## Usage

### Chat

Type naturally. Jarvis will:
- Remember important facts you mention (goals, habits, preferences, projects)
- Retrieve relevant memories from past conversations
- Maintain continuity within and across sessions

### Commands

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation |
| `/history` | Show current conversation history |
| `/memories` | View all stored memories |
| `/stats` | System statistics (conversations, messages, memories) |
| `/remember <text>` | Manually store a memory |
| `/forget <id>` | Delete a memory by ID prefix |
| `/help` | List available commands |
| `/quit` | Exit (auto-saves conversation summary) |

## Configuration

Settings are loaded with this priority (highest wins):

1. **Environment variables** (`JARVIS_*` prefix)
2. **`.env` file** (local overrides, not committed)
3. **`configs/config.yaml`** (version-controlled defaults)
4. **Hardcoded defaults** (in `configs/settings.py`)

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `llm.model` | `deepseek-r1:7b` | Ollama model to use |
| `llm.strip_thinking_tokens` | `true` | Hide `<think>` reasoning |
| `memory.max_retrieved_memories` | `3` | Memories per retrieval |
| `memory.max_context_tokens` | `500` | Token budget for context |
| `memory.similarity_threshold` | `0.3` | Minimum relevance score |
| `logging.level` | `INFO` | Log verbosity |

## Memory System

### How It Works

1. **Extraction**: After each user message, heuristic patterns detect goals, habits, deadlines, preferences, projects, and routines
2. **Storage**: Extracted facts are stored in both SQLite (structured) and ChromaDB (semantic vectors)
3. **Retrieval**: Before each LLM response, semantically similar memories are found and injected into the prompt
4. **Summarization**: When exiting or starting a new conversation, the session is summarized and stored as a memory

### Memory Types

| Type | Example |
|------|---------|
| `goal` | "I want to complete a marathon" |
| `habit` | "I run 5km every morning" |
| `deadline` | "Project due by June 15th" |
| `preference` | "I prefer Python over JavaScript" |
| `project` | "I'm building a personal assistant" |
| `routine` | "My morning routine is meditation then coding" |
| `fact` | "I'm a computer science student" |
| `summary` | Auto-generated conversation summaries |

## Future Roadmap

This V1 is the foundation. Planned expansions:

- [ ] RAG pipeline (PDF/document ingestion and retrieval)
- [ ] Task tracking and management
- [ ] Behavioral analytics and pattern detection
- [ ] Habit tracking and accountability system
- [ ] Proactive reminders
- [ ] Longitudinal pattern analysis
- [ ] Multi-model orchestration
- [ ] Dashboard UI
- [ ] Local automation integrations

## Design Principles

- **Persistence over performance** — Data survives restarts, crashes, and interruptions
- **Precision over intelligence** — Few highly relevant memories, not giant history dumps
- **Modularity over monolith** — Each subsystem is independent and swappable
- **Observability over opacity** — Comprehensive logging at every layer
- **Maintainability over cleverness** — Clean code that's easy to extend

## License

Private project — not yet licensed for distribution.
