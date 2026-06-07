"""
Jarvis V2.5 — Unified Context Builder
=======================================
Assembles the complete prompt sent to the LLM by combining:
    1. System prompt (identity, rules)
    2. State snapshot (goals, habits, projects)
    3. Retrieved memories (semantic search results)
    4. Knowledge chunks (from documents)
    5. Recent conversation history

Architecture decisions:
    - Replaces old ContextBuilder with UnifiedContextBuilder (REV 3 & 4).
    - Uses ContextRanker to enforce a strict token budget across all sources.
    - All injected context is tagged with provenance metadata for transparency.
"""

from datetime import datetime, timezone
from typing import Optional

from backend.context_ranker import ContextItem, ContextRanker
from backend.database import DatabaseManager
from backend.logger import get_logger
from configs.settings import get_settings
from memory.manager import MemoryManager
from state.goal_manager import GoalManager
from state.habit_manager import HabitManager
from state.project_manager import ProjectManager

logger = get_logger(__name__)


class UnifiedContextBuilder:
    """
    Builds the complete message list sent to the LLM for each request,
    unifying multiple context sources into a budgeted prompt.
    """

    def __init__(
        self,
        db: DatabaseManager,
        memory_manager: MemoryManager,
        goal_manager: GoalManager,
        habit_manager: HabitManager,
        project_manager: ProjectManager,
        knowledge_store=None,  # Optional until knowledge pipeline is ready
        settings=None,
    ):
        if settings is None:
            settings = get_settings()

        self._db = db
        self._memory = memory_manager
        self._goals = goal_manager
        self._habits = habit_manager
        self._projects = project_manager
        self._knowledge = knowledge_store
        
        self._settings = settings
        self._ranker = ContextRanker()
        self.last_assembled_context: list[ContextItem] = []

        logger.info(
            "UnifiedContextBuilder initialized — total_budget=%d tokens",
            self._settings.context.total_budget_tokens
        )

    def build_messages(
        self,
        user_message: str,
        conversation_id: str,
    ) -> list[dict]:
        """
        Build the complete message list for an LLM request.
        """
        # 1. Gather ContextItems from all sources
        items: list[ContextItem] = []
        
        # We fetch items, but don't strictly enforce per-source budgets here.
        # We let the Ranker handle the global budget, though we limit what we fetch.
        
        # State
        state_items = self._get_state_snapshot()
        items.extend(state_items)
        
        # Memories
        memory_items = self._get_memories(user_message)
        items.extend(memory_items)
        
        # Knowledge
        knowledge_items = self._get_knowledge(user_message)
        items.extend(knowledge_items)
        
        # History
        history_items = self._get_history(conversation_id)
        # History is treated a bit specially — we rank it, but we also format it 
        # as alternating messages if possible, OR we inject it into the prompt.
        # Actually, for standard LLM APIs, history is best passed as actual message objects,
        # but to control tokens precisely, we can inject older history into the system prompt
        # and only keep the last few turns as actual messages.
        # For this implementation, we will pass history as actual alternating messages
        # like the old ContextBuilder did, but we use the ranker to decide HOW MANY to keep.
        
        # 2. Rank and truncate items (excluding history which we handle separately for formatting)
        ranked_context = self._ranker.rank_and_truncate(
            items, 
            budget_tokens=self._settings.context.total_budget_tokens - self._settings.context.history_budget_tokens
        )
        # TEMPORARY CONTEXT CLAMP
        ranked_context = ranked_context[:3]
        self.last_assembled_context = ranked_context

        # 3. Format the ranked items by source
        state_text = self._format_items([i for i in ranked_context if i.source == 'state'], "CURRENT SYSTEM STATE")
        memory_text = self._format_items([i for i in ranked_context if i.source == 'memory'], "RELEVANT MEMORIES")
        knowledge_text = self._format_items([i for i in ranked_context if i.source == 'knowledge'], "KNOWLEDGE BASE")
        
        # Format history note
        history_text = f"[Continuing conversation with {len(history_items)} previous turns available.]" if history_items else "[New conversation.]"

        # 4. Build system prompt
        system_prompt = self._settings.system.system_prompt.format(
            state_snapshot=state_text or "[No active state]",
            memories=memory_text or "[No relevant memories]",
            knowledge=knowledge_text or "[No relevant knowledge]",
            conversation_history=history_text
        )

        # 5. Assemble messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history (truncate to budget)
        hist_budget = self._settings.context.history_budget_tokens
        current_hist_tokens = 0
        accepted_history = []
        # Process history newest first to ensure immediate context is kept
        for msg in reversed(history_items):
            est_tokens = max(1, len(msg["content"]) // 4)
            if current_hist_tokens + est_tokens <= hist_budget:
                accepted_history.insert(0, msg)
                current_hist_tokens += est_tokens
            else:
                break
                
        for msg in accepted_history:
             messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        logger.info("Context built: %d context items injected, %d history turns included", 
                    len(ranked_context), len(accepted_history))
        return messages

    def get_s4_context(self, user_message: str) -> dict:
        """
        Export formatted context for JARVIS System 4 (S4) injection.
        Returns a dict matching the placeholders in S4 role prompts.
        """
        # Fetch items
        state_items = self._get_state_snapshot()
        memory_items = self._get_memories(user_message)
        knowledge_items = self._get_knowledge(user_message)
        
        # Rank them briefly to stay within budget
        all_items = state_items + memory_items + knowledge_items
        budget = self._settings.context.total_budget_tokens - self._settings.context.history_budget_tokens
        ranked = self._ranker.rank_and_truncate(all_items, budget_tokens=budget)
        
        return {
            "state_snapshot": self._format_items([i for i in ranked if i.source == 'state'], "CURRENT SYSTEM STATE"),
            "memories": self._format_items([i for i in ranked if i.source == 'memory'], "RELEVANT MEMORIES"),
            "knowledge": self._format_items([i for i in ranked if i.source == 'knowledge'], "KNOWLEDGE BASE")
        }

    # -----------------------------------------------------------------------
    # Source Gatherers
    # -----------------------------------------------------------------------

    def _get_state_snapshot(self) -> list[ContextItem]:
        """Fetch current state (goals, habits, projects)."""
        items = []
        now = datetime.now(timezone.utc).isoformat()
        
        # Active Goals
        goals = self._goals.list_goals(status="active")
        if goals:
            goal_lines = [f"- {g['id'][:4]}: {g['title']} ({g['target_type']})" for g in goals]
            items.append(ContextItem(
                content="Active Goals:\n" + "\n".join(goal_lines),
                source="state",
                source_id="goals_active",
                timestamp=now,
                relevance_score=1.0,
                confidence=1.0,
                retrieval_reason="Always injected to ground Jarvis in user objectives."
            ))
            
        # Active Projects
        projects = self._projects.list_projects(status="active")
        if projects:
            proj_lines = [f"- {p['id'][:4]}: {p['name']} ({p['progress_percentage']}%)" for p in projects]
            items.append(ContextItem(
                content="Active Projects:\n" + "\n".join(proj_lines),
                source="state",
                source_id="projects_active",
                timestamp=now,
                relevance_score=1.0,
                confidence=1.0,
                retrieval_reason="Always injected."
            ))
            
        # Habits (just a count or summary to save tokens)
        habits = self._habits.list_habits(active_only=True)
        if habits:
            items.append(ContextItem(
                content=f"Tracking {len(habits)} active habits.",
                source="state",
                source_id="habits_active",
                timestamp=now,
                relevance_score=0.8,
                confidence=1.0,
                retrieval_reason="General state awareness."
            ))
            
        return items

    def _get_memories(self, query: str) -> list[ContextItem]:
        """Fetch memories via semantic search."""
        q_lower = query.strip().lower()
        greetings = {"hello", "hi", "hey", "sup", "what's up", "how are you", "good morning", "good afternoon", "good evening", "morning"}
        if len(q_lower) < 25 and any(q_lower.startswith(g) for g in greetings) or q_lower in greetings:
            # Low complexity / greeting: prefer zero memory injection
            return []

        memories = self._memory.retrieve_relevant_memories(
            query=query, 
            n_results=self._settings.memory.max_retrieved_memories
        )
        
        items = []
        for mem in memories:
            items.append(ContextItem(
                content=mem.get("content", ""),
                source="memory",
                source_id=mem.get("id", "unknown"),
                timestamp=mem.get("created_at", datetime.now(timezone.utc).isoformat()),
                relevance_score=mem.get("similarity_score", 0.5),
                confidence=mem.get("importance", 0.5),
                retrieval_reason=f"Semantic match to query ({mem.get('memory_type', 'fact')})"
            ))
        return items

    def _get_knowledge(self, query: str) -> list[ContextItem]:
        """Fetch knowledge chunks if store is available."""
        if not self._knowledge:
            return []
            
        try:
            chunks = self._knowledge.search(
                query=query, 
                n_results=self._settings.knowledge.max_retrieved_chunks
            )
            items = []
            for chunk in chunks:
                items.append(ContextItem(
                    content=chunk["text"],
                    source="knowledge",
                    source_id=chunk["id"],
                    timestamp=chunk["metadata"].get("ingested_at", datetime.now(timezone.utc).isoformat()),
                    relevance_score=1.0 - chunk.get("distance", 0.5),  # Convert Chroma distance to similarity score
                    confidence=1.0,
                    retrieval_reason=f"From {chunk['metadata'].get('title', 'Document')}"
                ))
            return items
        except Exception as e:
            logger.error("Failed to retrieve knowledge: %s", str(e))
            return []

    def _get_history(self, conversation_id: str) -> list[dict]:
        """Get recent conversation messages directly."""
        messages = self._db.get_conversation_messages(
            conversation_id=conversation_id,
            limit=self._settings.memory.conversation_history_turns * 2,
        )
        # Filter to only user and assistant
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ("user", "assistant")
        ]

    # -----------------------------------------------------------------------
    # Formatting
    # -----------------------------------------------------------------------

    def _format_items(self, items: list[ContextItem], header: str) -> str:
        if not items:
            return ""
            
        lines = [f"{header}:"]
        for item in items:
            # Include provenance metadata subtly (REV 4)
            prov = f"[src:{item.source}|score:{item.combined_score:.2f}|reason:{item.retrieval_reason}]"
            lines.append(f"{prov} {item.content}")
            
        return "\n".join(lines)
