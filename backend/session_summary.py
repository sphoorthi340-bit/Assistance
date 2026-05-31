"""
Jarvis V2.5 — Session Summary
===============================
Structured session summarization (REV 9).

Architecture decisions:
    - Automatically triggered at the end of significant sessions
      (e.g., /quit or /new).
    - Summarizes work completed, goals updated, blockers, next actions.
    - Stores the result as a structured memory for future retrieval.
    - Subscribes to the event bus to optionally trigger automatically
      if long idle periods are detected.
"""

from typing import Optional

from backend.database import DatabaseManager
from backend.llm import OllamaClient
from backend.logger import get_logger
from backend.events.event_bus import get_event_bus, Event, SESSION_ENDED
from memory.manager import MemoryManager

logger = get_logger(__name__)


class SessionSummarizer:
    """Generates structured summaries of user sessions."""

    def __init__(
        self,
        db: DatabaseManager,
        llm: OllamaClient,
        memory: MemoryManager,
    ):
        self._db = db
        self._llm = llm
        self._memory = memory
        self._bus = get_event_bus()

        # Subscribe to session ending events
        self._bus.subscribe(SESSION_ENDED, self._handle_session_ended)
        logger.info("SessionSummarizer initialized")

    def _handle_session_ended(self, event: Event) -> None:
        """Handle SESSION_ENDED event from the bus."""
        conversation_id = event.data.get("conversation_id")
        if conversation_id:
            logger.info("Session ended for %s, generating summary...", conversation_id)
            self.summarize_session(conversation_id)

    def summarize_session(self, conversation_id: str) -> Optional[dict]:
        """
        Generate and store a structured session summary.
        
        Args:
            conversation_id: The conversation UUID.
            
        Returns:
            The stored memory dict, or None if skipped/failed.
        """
        # Fetch messages for this session
        messages = self._db.get_conversation_messages(conversation_id)
        
        # Don't summarize short conversations (less than 4 messages = 2 turns)
        if not messages or len(messages) < 4:
            logger.debug("Conversation %s too short to summarize (%d msgs)", 
                         conversation_id, len(messages) if messages else 0)
            return None

        # Prepare context for the LLM
        history_text = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            history_text += f"{role}: {content}\n"

        prompt = (
            "Analyze the following conversation session and generate a structured summary.\n"
            "Focus ONLY on factual outcomes. Do not use filler text.\n\n"
            "Provide the summary in the following format (omit empty sections):\n"
            "- Work completed:\n"
            "- Goals/Habits updated:\n"
            "- Blockers identified:\n"
            "- Next actions:\n"
            "- Important concepts discussed:\n\n"
            "CONVERSATION:\n"
            f"{history_text}"
        )

        logger.info("Requesting structured session summary from LLM...")
        try:
            response = self._llm.chat([
                {"role": "system", "content": "You are a concise, analytical summarizer."},
                {"role": "user", "content": prompt}
            ], stream=False)

            summary_text = response.content.strip()
            if summary_text:
                # Store it as a memory of type 'summary' with high importance
                stored = self._memory.store_manual_memory(
                    content=f"Session Summary:\n{summary_text}",
                    memory_type="summary",
                    importance=0.8
                )
                logger.info("Successfully stored structured session summary")
                
                # Emit event
                self._bus.emit_simple(
                    name="session_summarized",
                    data={"conversation_id": conversation_id, "memory_id": stored["id"]},
                    source="SessionSummarizer"
                )
                return stored
                
        except Exception as e:
            logger.error("Failed to generate session summary: %s", str(e), exc_info=True)
            
        return None
