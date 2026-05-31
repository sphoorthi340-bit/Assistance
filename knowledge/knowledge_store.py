"""
Jarvis V3.0 — Knowledge Store
=============================
Manages persistent semantic storage for documents, notes, and academic materials.
Features multi-collection support, hybrid retrieval, and Obsidian Markdown support.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import re
import uuid

import chromadb
from chromadb.utils import embedding_functions

from backend.logger import get_logger
from configs.settings import get_settings
from knowledge.chunker import Chunk

logger = get_logger(__name__)


class KnowledgeStore:
    """
    Manages multi-collection semantic storage.
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        
        # Resolve ChromaDB path
        db_path = self._settings.resolve_path(self._settings.memory.chromadb_path)
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize client with matching settings to VectorStore
        self._client = chromadb.PersistentClient(
            path=str(db_path),
            settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        
        # Lazy loading for embedding model (to speed up startup if not used)
        self._embedding_function = None
        
        # Load collection names from settings
        self._collections_config = self._settings.knowledge.collections
        
        # Cache of initialized collections
        self._collections = {}
        
        logger.info("KnowledgeStore initialized")

    def _get_embedding_function(self):
        """Lazy load the embedding function."""
        if self._embedding_function is None:
            model_name = self._settings.memory.embedding_model
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        return self._embedding_function

    def _get_collection(self, collection_name: str):
        """Get or create a ChromaDB collection by name."""
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[collection_name]

    # -------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------

    def add_chunks(self, document_id: str, chunks: List[Chunk], collection_name: str = None) -> bool:
        """
        Add chunks to the specified collection (defaults to personal_memory).
        """
        if not chunks:
            return False
            
        col_name = collection_name or self._collections_config.personal_memory
        collection = self._get_collection(col_name)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk.text)
            
            # Combine chunk metadata with system metadata
            meta = {
                "document_id": document_id,
                "chunk_index": i,
                **chunk.metadata
            }
            # ChromaDB requires scalar metadata values (str, int, float, bool)
            safe_meta = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
            metadatas.append(safe_meta)

        try:
            embeddings = self._get_embedding_function()(documents)
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.debug("Added %d chunks to %s", len(chunks), col_name)
            return True
        except Exception as e:
            logger.error("Failed to add chunks to knowledge store: %s", e)
            return False

    def delete_document(self, document_id: str, collection_name: str = None) -> bool:
        """Delete all chunks for a document."""
        col_name = collection_name or self._collections_config.personal_memory
        collection = self._get_collection(col_name)
        
        try:
            collection.delete(
                where={"document_id": document_id}
            )
            logger.info("Deleted chunks for document %s from %s", document_id, col_name)
            return True
        except Exception as e:
            logger.error("Failed to delete document from knowledge store: %s", e)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get stats for all collections."""
        stats = {}
        try:
            for key, name in self._collections_config.model_dump().items():
                col = self._get_collection(name)
                stats[key] = {
                    "name": name,
                    "count": col.count()
                }
            return stats
        except Exception as e:
            logger.error("Failed to get knowledge store stats: %s", e)
            return {"error": str(e)}

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------

    def search(self, query: str, n_results: int = 5, filters: Optional[Dict] = None, collection_name: str = None) -> List[Dict]:
        """
        Standard vector search in a specific collection.
        """
        col_name = collection_name or self._collections_config.personal_memory
        collection = self._get_collection(col_name)
        
        try:
            query_embedding = self._get_embedding_function()([query])[0]
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            formatted_results = []
            
            # ChromaDB returns nested lists since you can query multiple texts at once
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })
                    
            return formatted_results
            
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []

    def search_hybrid(self, query: str, collections: List[str] = None, n_results: int = 5) -> List[Dict]:
        """
        Search across multiple collections and return merged results.
        Note: Currently a simple multi-collection query, not true hybrid (vector+keyword)
        as ChromaDB's native keyword search is limited. True hybrid would require
        ElasticSearch/BM25 integration which is out of scope for Phase 3.
        """
        if not collections:
            collections = [
                self._collections_config.personal_memory,
                self._collections_config.academic_knowledge,
                self._collections_config.project_docs
            ]
            
        all_results = []
        
        # Search each collection
        for col_name in collections:
            results = self.search(query, n_results=n_results, collection_name=col_name)
            # Annotate with source collection
            for r in results:
                r["collection"] = col_name
            all_results.extend(results)
            
        # Sort combined results by distance (lower is better for cosine distance in Chroma)
        all_results.sort(key=lambda x: x["distance"])
        
        # Deduplicate by ID
        seen_ids = set()
        deduped = []
        for r in all_results:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                deduped.append(r)
                
        # Return top N
        return deduped[:n_results]

    # -------------------------------------------------------------------
    # Phase 3: Obsidian Support
    # -------------------------------------------------------------------

    def ingest_obsidian(self, vault_path: str, collection_name: str = None) -> dict:
        """
        Ingests an Obsidian markdown vault, resolving wikilinks.
        """
        col_name = collection_name or self._collections_config.project_docs
        vault = Path(vault_path)
        
        if not vault.exists() or not vault.is_dir():
            return {"status": "error", "reason": "Vault path not found or not a directory"}
            
        md_files = list(vault.rglob("*.md"))
        results = {"processed": 0, "errors": 0, "chunks": 0}
        
        from knowledge.document_loader import DocumentLoader
        from knowledge.chunker import MetadataAwareChunker
        
        loader = DocumentLoader()
        chunker = MetadataAwareChunker(self._settings)
        
        for file_path in md_files:
            try:
                # Load document
                doc = loader.load(str(file_path))
                
                # Resolve Obsidian wikilinks [[Page Name|Alias]] -> Alias
                # and [[Page Name]] -> Page Name
                content = doc.content
                content = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', content)
                doc.content = content
                
                # Create chunks
                base_metadata = {
                    "source": "obsidian",
                    "file": file_path.name,
                    **doc.metadata
                }
                
                chunks = chunker.chunk_document(doc.content, base_metadata)
                
                # Add to store
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"obsidian://{file_path}"))
                success = self.add_chunks(doc_id, chunks, collection_name=col_name)
                
                if success:
                    results["processed"] += 1
                    results["chunks"] += len(chunks)
                else:
                    results["errors"] += 1
                    
            except Exception as e:
                logger.error("Failed to ingest obsidian file %s: %s", file_path, e)
                results["errors"] += 1
                
        return results
