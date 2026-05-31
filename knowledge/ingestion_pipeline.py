"""
Ingestion Pipeline for Knowledge Base
Orchestrates loading, chunking, and storing documents.
"""
import uuid
import hashlib
from pathlib import Path
from typing import Optional

from backend.logger import get_logger
from backend.database import DatabaseManager
from backend.events.event_bus import get_event_bus, DOCUMENT_INGESTED, DOCUMENT_DELETED
from configs.settings import get_settings

from knowledge.document_loader import DocumentLoader
from knowledge.chunker import MetadataAwareChunker
from knowledge.knowledge_store import KnowledgeStore

logger = get_logger(__name__)

class IngestionPipeline:
    def __init__(self, db: Optional[DatabaseManager] = None, settings=None):
        if settings is None:
            settings = get_settings()
            
        self.db = db or DatabaseManager(settings)
        self.settings = settings
        self.event_bus = get_event_bus()
        
        self.loader = DocumentLoader()
        self.chunker = MetadataAwareChunker(settings)
        self.store = KnowledgeStore(settings)

    def _compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def ingest_document(self, file_path: str | Path, document_id: Optional[str] = None) -> str:
        """
        Ingest a document into the knowledge base.
        Returns the document_id.
        """
        file_path = Path(file_path)
        
        existing = self.db.get_document_by_path(str(file_path.absolute()))
        file_hash = self._compute_file_hash(file_path)
        
        if existing and existing.get("file_hash") == file_hash:
            logger.info("Document %s is already up to date.", file_path.name)
            return existing["id"]
            
        logger.info("Loading document: %s", file_path)
        document = self.loader.load(file_path)
        
        logger.info("Chunking document...")
        base_meta = document.metadata.copy()
        if document.title:
            base_meta["title"] = document.title
            
        chunks = self.chunker.chunk_document(document.content, base_meta)
        
        doc_id = document_id or (existing["id"] if existing else str(uuid.uuid4()))
        
        logger.info("Storing chunks in vector DB...")
        if existing:
            self.store.delete_document(doc_id)
            
        self.store.add_chunks(doc_id, chunks)
        
        word_count = sum(len(c.text.split()) for c in chunks)
        
        if existing:
            self.db.delete_document(doc_id)
            
        self.db.add_document(
            doc_id=doc_id,
            source_path=document.source_path,
            format=document.format,
            title=document.title,
            chunk_count=len(chunks),
            word_count=word_count,
            file_hash=file_hash,
            metadata=document.metadata
        )
        
        self.event_bus.emit_simple(
            name=DOCUMENT_INGESTED,
            data={
                "document_id": doc_id,
                "source_path": document.source_path,
                "title": document.title,
                "chunk_count": len(chunks)
            },
            source="ingestion_pipeline"
        )
        
        logger.info("Successfully ingested document: %s (ID: %s, %d chunks)", document.title, doc_id, len(chunks))
        return doc_id
        
    def delete_document(self, document_id: str) -> bool:
        """Remove a document from knowledge base."""
        deleted = self.db.delete_document(document_id)
        if deleted:
            self.store.delete_document(document_id)
            self.event_bus.emit_simple(
                name=DOCUMENT_DELETED, 
                data={"document_id": document_id}, 
                source="ingestion_pipeline"
            )
            logger.info("Deleted document %s", document_id)
            return True
        return False
