# Jarvis V3 — Personal Cognitive Infrastructure

A local-first, persistent AI assistant built as foundational cognitive infrastructure.  
**Not a chatbot.** A system designed for continuity, memory, context preservation, multi-provider intelligence routing, and long-term behavioral intelligence.

---

## Philosophy

- The LLM is a **reasoning engine**, not memory
- All memory, context, and continuity live in **independent storage systems**
- Precision of retrieval > volume of context
- Clean memory flow > feature completeness
- Stability > sophistication
- Deterministic systems separated from LLM reasoning
- **Privacy by default** — sensitive content never leaves local providers

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Terminal CLI (main.py)                          │
├─────────────┬────────────────┬──────────────┬────────────┬─────────────┤
│   Context   │  Model Router  │    Memory    │   State    │  Database   │
│   Builder   │  + Provider    │   Manager    │  Managers  │  (SQLite)   │
│             │   Manager      │              │            │             │
│  State      │  Gemini        │  Extract     │  Goals     │  Memories   │
│  Memories   │  OpenAI        │  Store       │  Habits    │  Goals      │
│  Knowledge  │  Anthropic     │  Retrieve    │  Projects  │  Habits     │
│  History    │  LM Studio     │  Decay       │  Analytics │  Projects   │
│             │  Ollama        │              │            │  Route Logs │
├─────────────┴────────────────┤──────────────┤────────────┤─────────────┤
│   Knowledge Pipeline (RAG)   │ Vector Store │  Scheduler │ Explainabi- │
│   DocumentLoader + Chunker   │  (ChromaDB)  │  APSched   │ lity Engine │
└──────────────────────────────┴──────────────┴────────────┴─────────────┘
```

---

## What's New in V3

### Multi-Provider Intelligence Routing
- Strict priority-based routing: `Gemini → OpenAI → Anthropic → LM Studio → Ollama`
- **Development mode** routes to cloud for quality; **Production mode** stays local for privacy
- Domain-specific tier routing: `fast`, `reasoning`, `coding`, `math`
- **Model alias layer**: each tier maps to provider-specific model names (e.g. `qwen2.5:7b` on Ollama, `qwen2.5-7b-instruct` on LM Studio)
- Zero false routing — routes only to actually available, validated models

### Authenticated Cloud Health Checks
- Providers are only marked `healthy` after a real lightweight API call succeeds
- OpenAI: `models.list()` | Gemini: `list_models()` | Anthropic: minimal completion
- Diagnostic error classification: *Authentication failed*, *SDK missing*, *Network error*

### LM Studio Integration
- Runs as primary local inference runtime (OpenAI-compatible API)
- Ollama retained as a mandatory fallback until LM Studio is proven stable

### Startup Alias Validation
- Every routing tier × provider combination is validated at boot
- Reports `FOUND`, `NOT FOUND`, or `PROVIDER OFFLINE` before the scheduler starts

### Explainability Layer
- `/why` — explains last routing decision (provider, model, confidence, reason)
- `/provider_trace` — shows all provider health + quota status
- `/router` — recent routing history
- `/memory_debug` — memory retrieval trace
- `/router_test <query>` — dry-run any query through the router
- `/db_health` — SQLite schema verification
- `/knowledge_health` — ChromaDB collection status

### Cloud LLM with Caching
- Response caching via SHA-256 query hash (avoids redundant API calls)
- Per-provider daily call budget enforcement
- Context compression before cloud submission (uses Ollama Llama for summarization)

### Analytics Engine
- Weekly behavioral reports generated from habit, study, and conversation data
- Correlation detection: burnout patterns, study-habit alignment
- LLM-generated insights (Llama 3B), with deterministic fallback

---

## Tech Stack

| Component         | Technology                                   |
|-------------------|----------------------------------------------|
| Language          | Python 3.11+                                 |
| Local Inference   | LM Studio (primary), Ollama (fallback)       |
| Cloud Providers   | Gemini, OpenAI, Anthropic                    |
| Local Models      | qwen2.5:7b, llama3.2:1b                      |
| Database          | SQLite (WAL mode)                            |
| Vector DB         | ChromaDB (persistent)                        |
| Embeddings        | sentence-transformers (all-MiniLM-L6-v2)    |
| Scheduler         | APScheduler                                  |
| Terminal UI       | Rich                                         |
| Config            | Pydantic + YAML + .env                       |

---

## Project Structure

```
Jarvis/
├── main.py                          # Terminal CLI entry point
├── system_validation.py             # Full system health validation suite
├── requirements.txt
├── .env.example                     # Environment variable template
├── configs/
│   ├── config.yaml                  # Primary configuration
│   └── settings.py                  # Pydantic settings with 3-tier loading
├── backend/
│   ├── database.py                  # SQLite manager
│   ├── db_migrations.py             # Phase 3 schema migrations (idempotent)
│   ├── llm.py                       # Ollama client
│   ├── lm_studio.py                 # LM Studio client (OpenAI-compatible)
│   ├── cloud_llm.py                 # Unified cloud provider wrapper
│   ├── provider_manager.py          # Provider health, quota, and routing
│   ├── model_router.py              # Multi-tier alias-aware model router
│   ├── context.py                   # Token-budgeted context builder
│   ├── context_ranker.py            # Relevance ranking for context items
│   ├── explainability_engine.py     # /why, /provider_trace, /router
│   ├── analytics_engine.py          # Weekly reports, correlations
│   ├── proactive_layer.py           # Morning briefings, evening nudges
│   ├── scheduler.py                 # Background job scheduler
│   ├── backup_manager.py            # Automated vector_db backups
│   └── action_engine/               # Natural language → deterministic actions
├── memory/
│   ├── vector_store.py              # ChromaDB semantic search
│   ├── extractor.py                 # Memory extraction pipeline
│   └── manager.py                   # Memory lifecycle orchestration
├── state/
│   ├── goal_manager.py
│   ├── habit_manager.py
│   ├── project_manager.py
│   └── analytics_manager.py
├── knowledge/
│   ├── knowledge_store.py           # Multi-collection ChromaDB store
│   ├── document_loader.py           # PDF, MD, TXT ingestion
│   ├── chunker.py                   # Metadata-aware chunker
│   └── ingestion_pipeline.py        # End-to-end ingestion pipeline
├── vector_db/                       # ChromaDB storage (auto-created)
├── data/                            # SQLite database (auto-created)
├── documents/                       # Ingest PDFs and notes here
├── logs/                            # Rotating log files (auto-created)
└── frontend/                        # Reserved for future dashboard
```

---

## Setup

### Prerequisites

1. **Python 3.11+**
2. **LM Studio** — download from [lmstudio.ai](https://lmstudio.ai), load a model and start the local server on port `1234`
3. **Ollama** — download from [ollama.com](https://ollama.com), used as fallback
4. Pull the required Ollama models:
   ```bash
   ollama pull llama3.2:1b
   ollama pull qwen2.5:7b
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/sphoorthi340-bit/Assistance
cd Assistance

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
copy .env.example .env
# Edit .env and add your API keys (optional — cloud providers are optional)
```

### Configuration

Set API keys in `.env` (all optional — Jarvis works fully offline with local models):

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
```

### First Run

```bash
python main.py
```

Jarvis will automatically:
1. Run database migrations (idempotent)
2. Validate all providers and model aliases
3. Display the `=== MODEL ALIAS VALIDATION ===` startup report
4. Start the background scheduler
5. Launch the interactive terminal interface

### System Validation

```bash
python system_validation.py
```

Runs the full diagnostic suite including:
- All subsystem health checks
- Authenticated cloud provider verification
- Router verification suite (4 test cases)
- Per-tier model alias validation

---

## Routing Modes

### Development Mode (default)
```
Priority: Gemini → OpenAI → Anthropic → LM Studio → Ollama
```
Cloud-first for maximum quality. Set `JARVIS_DEVELOPMENT_MODE=true`.

### Production Mode
```
Priority: LM Studio → Ollama → Gemini → OpenAI → Anthropic
```
Local-first for privacy and cost. Set `JARVIS_DEVELOPMENT_MODE=false`.

### Routing Tiers

| Tier | Use Case | Ollama Model | LM Studio Model |
|------|----------|--------------|-----------------|
| `fast` | Quick replies, reminders | `llama3.2:1b` | `llama-3.2-3b-instruct` |
| `reasoning` | Analysis, explanation | `qwen2.5:7b` | `qwen2.5-7b-instruct` |
| `coding` | Code generation | `qwen2.5:7b` | `qwen2.5-7b-instruct` |
| `math` | Mathematical reasoning | `qwen2.5:7b` | `qwen2.5-7b-instruct` |

---

## Commands Reference

### Intelligence & Explainability
| Command | Description |
|---------|-------------|
| `/why` | Explain last routing decision (provider, model, reason) |
| `/provider_trace` | All provider health, quota, and cost status |
| `/router` | Recent routing history |
| `/router_test <query>` | Dry-run a query through the router |
| `/memory_debug [query]` | Memory retrieval trace |
| `/db_health` | SQLite schema verification |
| `/knowledge_health` | ChromaDB collection status |

### Knowledge Base (RAG)
| Command | Description |
|---------|-------------|
| `/ingest <path>` | Ingest a PDF, MD, or TXT file |
| `/knowledge list` | List all ingested documents |
| `/knowledge search <query>` | Search the knowledge base |
| `/knowledge delete <id>` | Remove a document |

### Memory
| Command | Description |
|---------|-------------|
| `/memories` | View stored memories |
| `/remember <fact>` | Manually store a memory |
| `/forget <id>` | Delete a memory |

### Goals
| Command | Description |
|---------|-------------|
| `/goal add <title>` | Add a new goal |
| `/goal list [status]` | List goals |
| `/goal complete <id>` | Mark as completed |
| `/goal update <id> <field> <value>` | Update a goal |
| `/goal pause / resume / delete` | Manage goal lifecycle |

### Habits
| Command | Description |
|---------|-------------|
| `/habit add <name>` | Add a habit |
| `/habit log <name> [notes]` | Log a completion |
| `/habit stats [name]` | Show streak and stats |
| `/habit list / deactivate / activate / delete` | Manage habits |

### Projects & Tasks
| Command | Description |
|---------|-------------|
| `/project add <name>` | Add a project |
| `/project list / status / update / complete` | Manage projects |
| `/task add <project-id> <title>` | Add a task |
| `/task list / complete / delete` | Manage tasks |

### System
| Command | Description |
|---------|-------------|
| `/stats` | Full system dashboard |
| `/accountability` | Accountability report |
| `/health` / `/system_status` | Live system health |
| `/context` | Last assembled context items |
| `/undo` | Reverse last action |
| `/new` | Start a new conversation |
| `/history` | View current conversation |
| `/quit` | Exit and generate session summary |
| `/help` | Full command reference |

---

## Design Principles

- **Persistence over performance** — Data survives restarts, crashes, and interruptions
- **Precision over intelligence** — Few highly relevant memories, not giant history dumps
- **Modularity over monolith** — Each subsystem is independent and swappable
- **Observability over opacity** — `/why`, `/provider_trace`, and `/router` expose all decisions
- **Maintainability over cleverness** — Clean, readable code designed to extend
- **Determinism over magic** — Analytics are SQL queries, not LLM guesses
- **Privacy by default** — Sensitive content stays local; cloud is opt-in

---

## Roadmap

- [x] Phase 1 — Memory & Context Assembly
- [x] Phase 2 — Personal State Modeling (Goals, Habits, Projects)
- [x] Phase 2.5 — Natural Action Engine & Knowledge Pipeline (RAG)
- [x] Phase 3 — Multi-Provider Routing, LM Studio, Cloud LLM, Explainability
- [ ] Phase 4 — Dashboard UI (web frontend)
- [ ] Phase 4 — Voice interface
- [ ] Phase 4 — Calendar & local automation integrations
- [ ] Phase 4 — Long-term behavioral trend analysis

---

## License

Private project — not licensed for public distribution.
