"""
Jarvis V1 — Context Builder
===============================
Assembles the complete prompt sent to the LLM by combining:
    1. System prompt (identity, rules)
    2. Retrieved memories (semantic search results)
    3. Recent conversation history (from SQLite)

Architecture decisions:
    The LLM has NO memory of its own. Every response is generated
    from a freshly assembled prompt that includes:
    
    - A system prompt defining Jarvis's identity and behavior
    - Relevant memories retrieved from ChromaDB (semantic search)
    - Recent conversation turns from SQLite (for continuity within session)
    
    Token budget management:
    - Memories: ~500 tokens max (configurable)
    - Conversation history: last N turns (configurable, default 10)
    - System prompt: fixed template
    - Total stays within the model's context window (8K for DeepSeek-R1 7B)

    This is the CORE of the context retrieval flow:
    user message → retrieve memories → build prompt → send to LLM
"""

from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger
from memory.manager import MemoryManager
from configs.settings import get_settings

logger = get_logger(__name__)


class ContextBuilder:
    """
    Builds the complete message list sent to the LLM for each request.
    
    Responsible for:
    - Retrieving relevant memories via semantic search
    - Formatting memories into a readable context block
    - Pulling recent conversation history
    - Assembling the final prompt with token budget awareness
    """

    def __init__(
        self,
        db: DatabaseManager,
        memory_manager: MemoryManager,
        settings=None,
    ):
        if settings is None:
            settings = get_settings()

        self._db = db
        self._memory = memory_manager
        self._system_prompt_template = settings.system.system_prompt
        self._max_context_tokens = settings.memory.max_context_tokens
        self._max_retrieved = settings.memory.max_retrieved_memories
        self._history_turns = settings.memory.conversation_history_turns

        logger.info(
            "ContextBuilder initialized — history_turns=%d, "
            "max_memories=%d, max_context_tokens=%d",
            self._history_turns, self._max_retrieved, self._max_context_tokens,
        )

    def build_messages(
        self,
        user_message: str,
        conversation_id: str,
    ) -> list[dict]:
        """
        Build the complete message list for an LLM request.
        
        Flow:
        1. Retrieve relevant memories for the user's message
        2. Format memories into a context block
        3. Get recent conversation history
        4. Assemble: system prompt → history → current user message
        
        Args:
            user_message: The current user input.
            conversation_id: The active conversation UUID.
        
        Returns:
            List of message dicts ready for OllamaClient.chat().
            Format: [{"role": "system", "content": ...}, 
                     {"role": "user/assistant", "content": ...}, ...]
        """
        # Step 1: Retrieve relevant memories
        memories = self._retrieve_and_format_memories(user_message)

        # Step 2: Get recent conversation history
        history = self._get_conversation_history(conversation_id)

        # Step 3: Build the system prompt with injected context
        system_prompt = self._build_system_prompt(memories, history)

        # Step 4: Assemble the final message list
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history as alternating user/assistant turns
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # Add the current user message
        messages.append({"role": "user", "content": user_message})

        logger.info(
            "Built context — %d messages total (%d history + system + current), "
            "%d memories injected",
            len(messages), len(history), len(memories) if isinstance(memories, list) else 0,
        )

        return messages

    def _retrieve_and_format_memories(self, query: str) -> str:
        """
        Retrieve relevant memories and format them into a readable block.
        
        Args:
            query: The user's message to search for relevant memories.
        
        Returns:
            Formatted string of relevant memories, or empty string if none found.
        """
        memories = self._memory.retrieve_relevant_memories(
            query=query,
            n_results=self._max_retrieved,
        )

        if not memories:
            return ""

        # Format memories into a clean, readable block
        formatted_lines = ["RELEVANT MEMORIES FROM PAST CONVERSATIONS:"]
        for i, mem in enumerate(memories, 1):
            mem_type = mem.get("memory_type", "unknown").upper()
            content = mem.get("content", "")
            similarity = mem.get("similarity_score", 0)

            # Truncate individual memories if they're too long
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = (self._max_context_tokens * 4) // self._max_retrieved
            if len(content) > max_chars:
                content = content[:max_chars] + "..."

            formatted_lines.append(
                f"  [{mem_type}] {content}"
            )

        formatted = "\n".join(formatted_lines)

        logger.debug(
            "Formatted %d memories into context block (%d chars)",
            len(memories), len(formatted),
        )

        return formatted

    def _get_conversation_history(
        self,
        conversation_id: str,
    ) -> list[dict]:
        """
        Get recent conversation turns for continuity.
        
        Only includes user and assistant messages (not system).
        Limited to the configured number of turns to stay within
        token budget.
        
        Args:
            conversation_id: The current conversation UUID.
        
        Returns:
            List of message dicts with 'role' and 'content'.
        """
        messages = self._db.get_conversation_messages(
            conversation_id=conversation_id,
            limit=self._history_turns * 2,  # Each "turn" is user + assistant
        )

        # Filter to only user and assistant messages
        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ("user", "assistant")
        ]

        return history

    def _build_system_prompt(
        self,
        memories_block: str,
        history: list[dict],
    ) -> str:
        """
        Construct the system prompt with injected context.
        
        The template has two injection points:
        - {memories} — relevant retrieved memories
        - {conversation_history} — note about conversation state
        
        Args:
            memories_block: Formatted memories string.
            history: Conversation history (used for the history note).
        
        Returns:
            The complete system prompt string.
        """
        # Build the memories section
        if memories_block:
            memories_section = memories_block
        else:
            memories_section = "[No relevant memories found for this query]"

        # Build a brief conversation state note
        if history:
            history_note = (
                f"[This is a continuing conversation with {len(history)} previous messages. "
                f"Maintain continuity with the discussion above.]"
            )
        else:
            history_note = "[This is a new conversation. No prior context available.]"

        # Inject into template
        system_prompt = self._system_prompt_template.format(
            memories=memories_section,
            conversation_history=history_note,
        )

        return system_prompt
