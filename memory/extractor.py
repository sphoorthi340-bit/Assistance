"""
Jarvis V1 — Memory Extractor
================================
Lightweight hybrid extraction: heuristics first, optional LLM summarization.

Architecture decisions:
    V1 extraction strategy (intentionally conservative):
    
    1. RULE-BASED extraction for structured, predictable patterns:
       - Goals ("I want to...", "my goal is...")
       - Habits ("I do X daily", "every morning I...")
       - Deadlines ("due by...", "deadline is...")
       - Preferences ("I prefer...", "I like...")
       - Projects ("I'm working on...", "my project...")
       - Routines ("every day I...", "my routine is...")
    
    2. LLM-ASSISTED extraction for unstructured content:
       - Conversation summaries
       - Complex observations
       - Semantic compression of lengthy exchanges
    
    WHY NOT full LLM extraction:
       - Early systems hallucinate memories
       - Over-storing creates noisy databases
       - Precision > intelligence for foundational layer
       - Rule-based is deterministic and debuggable

    Future phases will add:
       - Behavioral pattern detection
       - Longitudinal analysis
       - Autonomous memory management
"""

import re
from dataclasses import dataclass
from typing import Optional

from backend.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Extracted memory data structure
# ---------------------------------------------------------------------------

@dataclass
class ExtractedMemory:
    """A single memory extracted from conversation text."""
    content: str
    memory_type: str        # fact, goal, preference, habit, routine, deadline, project, observation
    importance: float       # 0.0 to 1.0
    source_text: str = ""   # The original text this was extracted from


# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Each pattern maps to (memory_type, importance, compiled_regex)
# Patterns are checked in priority order — first match wins per sentence

_EXTRACTION_PATTERNS: list[tuple[str, float, re.Pattern]] = [
    # --- Goals (high importance) ---
    ("goal", 0.8, re.compile(
        r"(?:my goal is|i want to|i aim to|i need to|i plan to|"
        r"i'm trying to|i intend to|objective is|target is)\s+(.+)",
        re.IGNORECASE,
    )),

    # --- Deadlines (high importance) ---
    ("deadline", 0.9, re.compile(
        r"(?:due by|deadline is|due date is|must be done by|"
        r"need to finish by|submit by|complete by)\s+(.+)",
        re.IGNORECASE,
    )),

    # --- Projects (high importance) ---
    ("project", 0.7, re.compile(
        r"(?:i'm working on|i am working on|my project is|"
        r"i'm building|i am building|currently developing|"
        r"i started|i'm developing)\s+(.+)",
        re.IGNORECASE,
    )),

    # --- Habits (medium-high importance) ---
    ("habit", 0.7, re.compile(
        r"(?:i (?:do|run|read|exercise|meditate|study|practice|write|code|train)"
        r"\s+(?:every|daily|weekly|each)(?:\s+\w+)*"
        r"|every (?:day|morning|evening|night|week)\s+i\s+\w+)",
        re.IGNORECASE,
    )),

    # --- Routines (medium importance) ---
    ("routine", 0.6, re.compile(
        r"(?:my routine is|my schedule is|i usually|i typically|"
        r"every morning|every evening|my daily|my weekly)\s+(.+)",
        re.IGNORECASE,
    )),

    # --- Preferences (medium importance) ---
    ("preference", 0.5, re.compile(
        r"(?:i prefer|i like|i enjoy|i don't like|i hate|i dislike|"
        r"i love|my favorite|i always use|i usually use)\s+(.+)",
        re.IGNORECASE,
    )),

    # --- Facts / personal info (medium importance) ---
    ("fact", 0.5, re.compile(
        r"(?:my name is|i am a|i work (?:at|as|in)|i study|"
        r"i live in|i'm a|i am (?:a |an )?(?:student|developer|engineer|"
        r"researcher|teacher|designer|writer|artist))\s*(.+)?",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# Memory Extractor
# ---------------------------------------------------------------------------

class MemoryExtractor:
    """
    Extracts memorable facts from conversation turns using
    lightweight heuristics + optional LLM summarization.
    
    This is intentionally conservative for V1 — precision over recall.
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional OllamaClient for LLM-assisted extraction.
                       If None, only rule-based extraction is used.
        """
        self._llm = llm_client
        logger.info(
            "MemoryExtractor initialized — llm_assisted=%s",
            self._llm is not None,
        )

    def extract_from_message(
        self,
        message: str,
        role: str = "user",
    ) -> list[ExtractedMemory]:
        """
        Extract memories from a single message using heuristic patterns.
        
        Args:
            message: The message text to analyze.
            role: 'user' or 'assistant' — user messages are primary extraction targets.
        
        Returns:
            List of ExtractedMemory objects found in the message.
        """
        # Only extract from user messages (user's own statements matter most)
        if role != "user":
            return []

        memories = []
        # Split into sentences for more precise extraction
        sentences = self._split_sentences(message)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue

            for memory_type, importance, pattern in _EXTRACTION_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    # Use the captured group if available, otherwise the full sentence
                    content = match.group(0).strip()

                    memory = ExtractedMemory(
                        content=content,
                        memory_type=memory_type,
                        importance=importance,
                        source_text=sentence,
                    )
                    memories.append(memory)

                    logger.debug(
                        "Extracted [%s] (%.1f): %.60s...",
                        memory_type, importance, content,
                    )
                    break  # First match per sentence wins

        return memories

    def extract_from_conversation(
        self,
        messages: list[dict],
    ) -> list[ExtractedMemory]:
        """
        Extract memories from a list of conversation messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Returns:
            Deduplicated list of ExtractedMemory objects.
        """
        all_memories = []
        seen_contents = set()

        for msg in messages:
            extracted = self.extract_from_message(
                message=msg.get("content", ""),
                role=msg.get("role", "user"),
            )
            for memory in extracted:
                # Simple deduplication by normalized content
                normalized = memory.content.lower().strip()
                if normalized not in seen_contents:
                    seen_contents.add(normalized)
                    all_memories.append(memory)

        logger.info(
            "Extracted %d memories from %d messages",
            len(all_memories), len(messages),
        )
        return all_memories

    def summarize_conversation(
        self,
        messages: list[dict],
        max_words: int = 50,
    ) -> Optional[ExtractedMemory]:
        """
        Generate a summary memory for a conversation segment.
        
        Requires LLM client. Returns None if LLM is not available.
        
        Args:
            messages: List of message dicts to summarize.
            max_words: Target summary length.
        
        Returns:
            An ExtractedMemory of type 'summary', or None.
        """
        if self._llm is None:
            logger.debug("LLM not available — skipping conversation summary")
            return None

        if len(messages) < 4:
            logger.debug("Too few messages for summary (%d)", len(messages))
            return None

        # Build a condensed conversation string for the LLM
        conversation_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages[-20:]  # Last 20 messages max
        )

        try:
            summary = self._llm.generate_summary(conversation_text, max_words)
            if summary:
                return ExtractedMemory(
                    content=summary,
                    memory_type="summary",
                    importance=0.6,
                    source_text=f"[Summary of {len(messages)} messages]",
                )
        except Exception as e:
            logger.error("Failed to generate conversation summary: %s", str(e))

        return None

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentences.
        
        Simple but effective for V1 — handles periods, question marks,
        exclamation marks, and newlines as delimiters.
        """
        # Split on sentence-ending punctuation followed by space/newline
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        return [s for s in sentences if s.strip()]
