"""
Jarvis V1 — Memory Manager
==============================
Orchestrates the full memory lifecycle: extraction → storage → retrieval.

Architecture decisions:
    The MemoryManager is the SINGLE entry point for all memory operations.
    Other modules should never interact with DatabaseManager or VectorStore
    directly for memory purposes — always go through MemoryManager.
    
    This ensures:
    - Consistency between SQLite and ChromaDB
    - Single place to add logging/metrics
    - Clean interface for future behavioral analytics
    - Easy to swap storage backends later

    Memory flow:
    1. User sends a message
    2. After assistant responds, MemoryManager.process_message() runs
    3. Extractor pulls facts from the user's message (heuristics)
    4. Each extracted fact is stored in BOTH SQLite and ChromaDB
    5. Before the next response, MemoryManager.retrieve_context() finds
       relevant memories via semantic search
    6. These memories are injected into the prompt by ContextBuilder
"""

from typing import Optional

from backend.database import DatabaseManager
from backend.logger import get_logger
from memory.extractor import MemoryExtractor, ExtractedMemory
from memory.vector_store import VectorStore
from configs.settings import get_settings

logger = get_logger(__name__)


class MemoryManager:
    """
    Central orchestrator for the memory subsystem.
    
    Coordinates between:
    - MemoryExtractor (pulls facts from conversations)
    - DatabaseManager (structured storage in SQLite)
    - VectorStore (semantic storage in ChromaDB)
    """

    def __init__(
        self,
        db: DatabaseManager,
        vector_store: VectorStore,
        extractor: MemoryExtractor,
        settings=None,
    ):
        if settings is None:
            settings = get_settings()

        self._db = db
        self._vector = vector_store
        self._extractor = extractor
        self._max_retrieved = settings.memory.max_retrieved_memories
        self._importance_threshold = settings.memory.importance_threshold

        logger.info(
            "MemoryManager initialized — max_retrieved=%d, importance_threshold=%.2f",
            self._max_retrieved, self._importance_threshold,
        )

    # -------------------------------------------------------------------
    # Memory storage
    # -------------------------------------------------------------------

    def process_message(
        self,
        message: str,
        role: str,
        conversation_id: str,
    ) -> list[dict]:
        """
        Process a message for memory extraction and storage.
        
        This is called AFTER each user message. It:
        1. Runs heuristic extraction on the message
        2. Stores each extracted memory in both SQLite and ChromaDB
        3. Returns the list of stored memories for observability
        
        Args:
            message: The message text.
            role: 'user' or 'assistant'.
            conversation_id: The current conversation UUID.
        
        Returns:
            List of stored memory dicts (for logging/display).
        """
        # Extract memories using heuristics
        extracted = self._extractor.extract_from_message(message, role)

        stored_memories = []
        for memory in extracted:
            try:
                stored = self._store_memory(
                    memory=memory,
                    conversation_id=conversation_id,
                )
                stored_memories.append(stored)
            except Exception as e:
                logger.error(
                    "Failed to store memory: %s — %s",
                    memory.content[:50], str(e),
                )

        if stored_memories:
            logger.info(
                "Processed message — extracted and stored %d memories",
                len(stored_memories),
            )

        return stored_memories

    def _store_memory(
        self,
        memory: ExtractedMemory,
        conversation_id: Optional[str] = None,
    ) -> dict:
        """
        Store a single memory in both SQLite and ChromaDB.
        
        Ensures consistency: if SQLite write succeeds, the vector
        store write uses the same ID.
        """
        # Step 1: Store in SQLite (source of truth for structured data)
        memory_id = self._db.store_memory(
            content=memory.content,
            memory_type=memory.memory_type,
            conversation_id=conversation_id,
            importance=memory.importance,
            metadata={"source_text": memory.source_text},
        )

        # Step 2: Store in ChromaDB (for semantic retrieval)
        self._vector.add_memory(
            memory_id=memory_id,
            content=memory.content,
            memory_type=memory.memory_type,
            importance=memory.importance,
            conversation_id=conversation_id,
        )

        return {
            "id": memory_id,
            "content": memory.content,
            "type": memory.memory_type,
            "importance": memory.importance,
        }

    def store_manual_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.6,
    ) -> dict:
        """
        Manually store a memory (not from extraction).
        
        Useful for explicit user commands like '/remember X'.
        
        Args:
            content: The memory text.
            memory_type: Category of the memory.
            importance: Relevance score.
        
        Returns:
            Dict with stored memory details.
        """
        memory = ExtractedMemory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            source_text="[manual entry]",
        )
        return self._store_memory(memory)

    # -------------------------------------------------------------------
    # Memory retrieval
    # -------------------------------------------------------------------

    def retrieve_relevant_memories(
        self,
        query: str,
        n_results: Optional[int] = None,
    ) -> list[dict]:
        """
        Find memories semantically relevant to a query.
        
        This is called BEFORE each LLM response to inject context.
        
        Args:
            query: The user's current message (or derived query).
            n_results: Max results (defaults to config max_retrieved_memories).
        
        Returns:
            List of memory dicts ordered by relevance, each containing:
            id, content, memory_type, importance, similarity_score.
        """
        n = n_results or self._max_retrieved

        memories = self._vector.search_similar(
            query=query,
            n_results=n,
            min_importance=self._importance_threshold,
        )

        # Update access tracking in SQLite for retrieved memories
        for mem in memories:
            try:
                self._db.update_memory_access(mem["id"])
            except Exception as e:
                logger.debug("Failed to update memory access: %s", str(e))

        if memories:
            logger.info(
                "Retrieved %d relevant memories for query '%.40s...'",
                len(memories), query,
            )
            for mem in memories:
                logger.debug(
                    "  → [%s] (sim=%.3f, imp=%.1f): %.50s...",
                    mem.get("memory_type", "?"),
                    mem.get("similarity_score", 0),
                    mem.get("importance", 0),
                    mem.get("content", ""),
                )

        return memories

    def summarize_conversation(
        self,
        conversation_id: str,
    ) -> Optional[dict]:
        """
        Generate and store a summary of a conversation.
        
        Args:
            conversation_id: The conversation to summarize.
        
        Returns:
            The stored summary memory dict, or None if summarization fails.
        """
        messages = self._db.get_conversation_messages(conversation_id)
        if not messages:
            return None

        # Convert to the format expected by extractor
        msg_dicts = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

        summary = self._extractor.summarize_conversation(msg_dicts)
        if summary:
            return self._store_memory(summary, conversation_id=conversation_id)
        return None

    # -------------------------------------------------------------------
    # Memory inspection
    # -------------------------------------------------------------------

    def get_all_memories(self, limit: int = 50) -> list[dict]:
        """Get all stored memories from SQLite, ordered by recency."""
        return self._db.get_all_memories(limit=limit)

    def get_memories_by_type(
        self,
        memory_type: str,
        limit: int = 20,
    ) -> list[dict]:
        """Get memories filtered by type."""
        return self._db.search_memories(memory_type=memory_type, limit=limit)

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory from both SQLite and ChromaDB.
        
        Returns True if the memory was found and deleted.
        """
        # Delete from both stores
        db_deleted = self._db.delete_memory(memory_id)
        self._vector.delete_memory(memory_id)

        if db_deleted:
            logger.info("Deleted memory: %s", memory_id[:8])
        return db_deleted

    def get_stats(self) -> dict:
        """Get memory subsystem statistics."""
        db_stats = self._db.get_stats()
        vector_stats = self._vector.get_stats()
        return {
            "database": db_stats,
            "vector_store": vector_stats,
        }
