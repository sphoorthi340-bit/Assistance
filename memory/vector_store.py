"""
Jarvis V1 — Vector Store (ChromaDB)
======================================
Semantic memory storage and retrieval using ChromaDB with local embeddings.

Architecture decisions:
    - ChromaDB runs in persistent local mode (no server needed)
    - Embeddings generated via sentence-transformers (all-MiniLM-L6-v2)
    - 384-dimensional embeddings — fast, efficient, good quality
    - Each memory entry is stored as a document with metadata
    - Metadata supports filtering by type, importance, date
    - Similarity search returns scored results for ranking
    - Duplicate detection prevents storing the same memory twice

The vector store is the RETRIEVAL half of the memory system.
SQLite stores the structured data; ChromaDB enables semantic search
so we can find memories by meaning, not just keywords.

Future extensibility:
    - Swap embedding models without schema changes
    - Add collections for different memory domains
    - Integrate with RAG pipeline for document retrieval
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store for semantic memory search.
    
    Stores memory embeddings alongside metadata and supports
    similarity-based retrieval with filtering.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self._chromadb_path = settings.resolve_path(settings.memory.chromadb_path)
        self._collection_name = settings.memory.collection_name
        self._embedding_model = settings.memory.embedding_model
        self._similarity_threshold = settings.memory.similarity_threshold

        # Ensure the storage directory exists
        self._chromadb_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client in persistent mode
        self._client = chromadb.PersistentClient(
            path=str(self._chromadb_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,  # Respect user privacy
            ),
        )

        # Get or create the memories collection
        # Using the default embedding function initially;
        # we use sentence-transformers via ChromaDB's built-in support
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )

        # Lazy-loaded embedding function
        self._embed_fn = None

        logger.info(
            "VectorStore initialized — path=%s, collection=%s, "
            "embedding_model=%s, existing_entries=%d",
            self._chromadb_path,
            self._collection_name,
            self._embedding_model,
            self._collection.count(),
        )

    def _get_embedding_function(self):
        """
        Lazy-load the sentence-transformers embedding model.
        
        The model is loaded on first use to avoid slow startup when
        the vector store isn't immediately needed.
        """
        if self._embed_fn is None:
            from sentence_transformers import SentenceTransformer
            self._embed_fn = SentenceTransformer(self._embedding_model)
            logger.info("Loaded embedding model: %s", self._embedding_model)
        return self._embed_fn

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        model = self._get_embedding_function()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def add_memory(
        self,
        memory_id: str,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        conversation_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a memory entry to the vector store.
        
        Args:
            memory_id: UUID matching the SQLite memories.id.
            content: The memory text to embed.
            memory_type: Category (fact, goal, project, etc.).
            importance: Relevance score 0.0–1.0.
            conversation_id: Source conversation UUID.
            metadata: Additional metadata dict.
        """
        # Build metadata for ChromaDB
        meta = {
            "memory_type": memory_type,
            "importance": importance,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if conversation_id:
            meta["conversation_id"] = conversation_id
        if metadata:
            # ChromaDB metadata values must be str, int, float, or bool
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v

        # Generate embedding
        embedding = self._embed([content])[0]

        # Upsert (add or update) to handle potential duplicates
        self._collection.upsert(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[meta],
        )

        logger.debug(
            "Added memory to vector store: %s [%s] (%.2f importance)",
            memory_id[:8], memory_type, importance,
        )

    def search_similar(
        self,
        query: str,
        n_results: int = 3,
        memory_type: Optional[str] = None,
        min_importance: Optional[float] = None,
    ) -> list[dict]:
        """
        Search for semantically similar memories.
        
        Args:
            query: The text to find similar memories for.
            n_results: Maximum number of results.
            memory_type: Optional filter by memory type.
            min_importance: Optional minimum importance threshold.
        
        Returns:
            List of dicts with keys: id, content, memory_type, importance,
            similarity_score, metadata.
            Ordered by similarity (highest first).
        """
        if self._collection.count() == 0:
            logger.debug("Vector store is empty — no results to return")
            return []

        # Build where filter
        where_filter = None
        conditions = []

        if memory_type:
            conditions.append({"memory_type": {"$eq": memory_type}})
        if min_importance is not None:
            conditions.append({"importance": {"$gte": min_importance}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Generate query embedding
        query_embedding = self._embed([query])[0]

        # Query ChromaDB
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self._collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("Vector search failed: %s", str(e))
            return []

        # Parse results into clean dicts
        memories = []
        if results and results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                # ChromaDB returns cosine distance; convert to similarity
                distance = results["distances"][0][i]
                similarity = 1 - distance  # cosine similarity = 1 - cosine distance

                if similarity < self._similarity_threshold:
                    continue

                memory = {
                    "id": memory_id,
                    "content": results["documents"][0][i],
                    "similarity_score": round(similarity, 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
                # Extract common metadata fields to top level
                meta = memory["metadata"]
                memory["memory_type"] = meta.get("memory_type", "unknown")
                memory["importance"] = meta.get("importance", 0.5)

                memories.append(memory)

        logger.debug(
            "Vector search for '%.40s...' returned %d results (threshold=%.2f)",
            query, len(memories), self._similarity_threshold,
        )

        return memories

    def delete_memory(self, memory_id: str) -> None:
        """Remove a memory from the vector store by ID."""
        try:
            self._collection.delete(ids=[memory_id])
            logger.debug("Deleted memory from vector store: %s", memory_id[:8])
        except Exception as e:
            logger.error("Failed to delete from vector store: %s", str(e))

    def get_count(self) -> int:
        """Get the total number of entries in the vector store."""
        return self._collection.count()

    def get_stats(self) -> dict:
        """Get vector store statistics for health checks."""
        return {
            "total_entries": self._collection.count(),
            "collection_name": self._collection_name,
            "embedding_model": self._embedding_model,
            "chromadb_path": str(self._chromadb_path),
        }
