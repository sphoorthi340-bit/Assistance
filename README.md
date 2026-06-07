# Jarvis V4 — Personal Cognitive Infrastructure

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
│   Context   │  System 4 (S4) │    Memory    │   State    │  Database   │
│   Builder   │  Role Manager  │   Manager    │  Managers  │  (SQLite)   │
│             │                │              │            │             │
│  State      │  Chief         │  Extract     │  Academic  │  Memories   │
│  Memories   │  Analyst       │  Store       │  Goals     │  Goals      │
│  Knowledge  │  Engineer      │  Retrieve    │  Habits    │  Habits     │
│  History    │  Mentor        │  Decay       │  Projects  │  Projects   │
│             │  Rapid         │              │  MS Abroad │  Route Logs │
├─────────────┴────────────────┤──────────────┤────────────┤─────────────┤
│   Knowledge Pipeline (RAG)   │ Vector Store │  Scheduler │ Explainabi- │
│   DocumentLoader + Chunker   │  (ChromaDB)  │  APSched   │ lity Engine │
└──────────────────────────────┴──────────────┴────────────┴─────────────┘
```

---

## What's New in V4 (System 4)

### The S4 Council of Experts
Jarvis is now powered by **System 4 (S4)**, a multi-agent architectural layer that routes requests to 5 distinct specialized roles, each backed by an optimized local model:
1. **Chief of Staff**: Strategic planning, task triage, and high-level decisions.
2. **Strategic Analyst**: Deep reasoning, research, paper analysis, MS profile audits.
3. **Software Engineer**: System architecture, coding, and debugging.
4. **Learning Mentor**: Concept explanation, exam prep, academic guidance.
5. **Rapid Assistant**: Instant retrieval, focus timer management, quick definitions.

### Collaboration Patterns
S4 features dynamic interaction patterns based on intent complexity:
- **Solo**: Single expert handles the task.
- **Verify**: Primary expert drafts an answer; secondary expert reviews and corrects it.
- **Pipeline**: Sequential handoff (e.g., Analyst reads a paper → Mentor explains it to the user).
- **Council**: All relevant experts generate perspectives, which the Chief synthesizes into a final recommendation.

### Academic & MS Abroad Engines
- **Academic Manager**: Tracks CGPA, semester progression, exam dates, and automatically triggers high-urgency **Exam Mode**.
- **MS Roadmap Manager**: Dedicated subsystem for tracking MS Fall 2028 prep, GRE/TOEFL scores, university shortlists, and research publications.

### Web Dashboard
- Lightweight background API server running on port `8080`.
- Feeds live data to a web dashboard for visualizing Academic Progress, Focus Sessions, and the MS Abroad Roadmap.

### Advanced Fallback & VRAM Protection
- Automatic fallback hierarchy: If an expert model fails, the task gracefully cascades to the `Chief` model to ensure zero downtime.
- Strict token budgeting per model to prevent infinite KV cache allocation and `0xC0000005` VRAM crashes on local hardware.

---

## What's Maintained from V3

### Multi-Provider Intelligence Routing
- Strict priority-based routing: `Gemini → OpenAI → Anthropic → LM Studio → Ollama`
- **Development mode** routes to cloud for quality; **Production mode** stays local for privacy

### Explainability Layer
- `/why` — explains last routing decision (provider, model, confidence, reason)
- `/s4` — trace S4 role delegation and collaboration patterns
- `/provider_trace` — shows all provider health + quota status
- `/memory_debug` — memory retrieval trace

---

## Tech Stack

| Component         | Technology                                   |
|-------------------|----------------------------------------------|
| Language          | Python 3.11+                                 |
| Local Inference   | LM Studio (primary), Ollama (fallback)       |
| Cloud Providers   | Gemini, OpenAI, Anthropic                    |
| Vector DB         | ChromaDB (persistent)                        |
| Database          | SQLite (WAL mode)                            |
| Embeddings        | sentence-transformers (all-MiniLM-L6-v2)    |
| Scheduler         | APScheduler                                  |
| Terminal UI       | Rich                                         |

---

## Project Structure

```
Jarvis/
├── main.py                          # Terminal CLI entry point
├── system_validation.py             # Full system health validation suite
├── configs/
│   ├── config.yaml                  # Primary configuration
│   └── s4_prompts/                  # Role-specific system prompts
├── backend/
│   ├── s4_dispatcher.py             # S4 Core routing logic
│   ├── s4_classifier.py             # S4 Intent and pattern classification
│   ├── s4_roles.py                  # S4 Role Manager & Fallback Engine
│   ├── dashboard_api.py             # Background web server
│   ├── focus_guard.py               # Pomodoro & distraction management
│   ├── research_workflow.py         # ArXiv paper fetch & analyze engine
│   ├── database.py                  # SQLite manager
│   ├── llm.py                       # Ollama client
│   └── lm_studio.py                 # LM Studio client
├── memory/
│   ├── s4_memory.py                 # S4 structured hot/warm/cold memory
│   └── vector_store.py              # ChromaDB semantic search
├── state/
│   ├── academic_manager.py          # CGPA & Exam tracking
│   ├── ms_roadmap.py                # Masters application roadmap
│   └── ...                          # Goal/Habit managers
├── vector_db/                       # ChromaDB storage
├── data/                            # SQLite database & JSON profiles
├── knowledge/                       # Ingested PDFs and raw papers
└── logs/                            # Rotating log files
```

---

## Setup

### Prerequisites

1. **Python 3.11+**
2. **LM Studio** — download from [lmstudio.ai](https://lmstudio.ai), load the target models (Qwen, Phi-4, Gemma) and start the local server on port `1234`
3. **Ollama** — download from [ollama.com](https://ollama.com), used as fallback (Llama 3.2, Qwen 3)

### Installation

```bash
# Clone the repository
git clone https://github.com/sphoorthi340-bit/Assistance
cd Assistance

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Run

```bash
python main.py
```

Jarvis will automatically:
1. Run database migrations
2. Validate all providers and S4 model aliases
3. Start the Dashboard API server (Port 8080)
4. Launch the interactive terminal interface

---

## System 4 (S4) Roles & Models

| Role | Responsibility | Provider | Default Model |
|------|----------------|----------|---------------|
| `Chief` | Triage, Planning, Synthesis | LM Studio | `qwen3-4b` |
| `Analyst` | Deep Reasoning, Research | LM Studio | `phi-4-mini-reasoning` |
| `Engineer` | Coding, System Architecture | LM Studio | `qwen2.5-7b-instruct` |
| `Mentor` | Academic Explanation | LM Studio | `gemma-3-4b` |
| `Rapid` | CLI Retrieval, Fast Tasks | Ollama | `qwen3-1.7b` |

---

## Roadmap

- [x] Phase 1 — Memory & Context Assembly
- [x] Phase 2 — Personal State Modeling (Goals, Habits, Projects)
- [x] Phase 2.5 — Natural Action Engine & Knowledge Pipeline (RAG)
- [x] Phase 3 — Multi-Provider Routing & Cloud Explainability
- [x] **Phase 4 — System 4 Multi-Agent Architecture (Chief, Analyst, Engineer, Mentor, Rapid)**
- [x] **Phase 4 — Academic & MS Roadmap Tracking Engines**
- [x] **Phase 4 — Local Web Dashboard**
- [ ] Phase 5 — Continuous System Integration (Calendar, Filesystem, Automation)
- [ ] Phase 5 — Voice Interface

---

## License

Private project — not licensed for public distribution.
