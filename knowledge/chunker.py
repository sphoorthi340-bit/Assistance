"""
Document Chunker for Knowledge Pipeline (REV 8)
Implements metadata-rich chunking preserving section titles, headings, and page refs.
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from backend.logger import get_logger
from configs.settings import get_settings

logger = get_logger(__name__)

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]

class MetadataAwareChunker:
    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()
        self.chunk_size = settings.knowledge.chunk_size
        self.chunk_overlap = settings.knowledge.chunk_overlap
        self.min_chunk_size = settings.knowledge.min_chunk_size

    def chunk_document(self, content: str, base_metadata: Dict[str, Any] = None) -> List[Chunk]:
        base_metadata = base_metadata or {}
        
        # Regex patterns to find metadata
        page_pattern = re.compile(r"^---\s*Page\s+(\d+)\s*---$", re.MULTILINE)
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        
        paragraphs = content.split('\n\n')
        chunks: List[Chunk] = []
        
        current_chunk_words = []
        current_length = 0
        current_page = 1
        current_heading = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check for page changes
            page_match = page_pattern.search(para)
            if page_match:
                current_page = int(page_match.group(1))
                para = page_pattern.sub("", para).strip()
                if not para:
                    continue
                    
            # Check for headings
            heading_match = heading_pattern.search(para)
            if heading_match:
                current_heading = heading_match.group(2).strip()
                
            words = para.split()
            word_count = len(words)
            
            if current_length + word_count > self.chunk_size and current_length >= self.min_chunk_size:
                chunks.append(self._create_chunk(current_chunk_words, base_metadata, current_page, current_heading))
                
                # Handle overlap
                overlap_words = current_chunk_words[-self.chunk_overlap:] if self.chunk_overlap > 0 else []
                current_chunk_words = overlap_words + words
                current_length = len(current_chunk_words)
            else:
                current_chunk_words.extend(words)
                current_length += word_count
                
        # Emit final chunk
        if current_chunk_words:
            chunks.append(self._create_chunk(current_chunk_words, base_metadata, current_page, current_heading))
            
        logger.debug("Chunked document into %d chunks", len(chunks))
        return chunks

    def _create_chunk(self, words: List[str], base_metadata: Dict[str, Any], page: int, heading: str) -> Chunk:
        meta = base_metadata.copy()
        meta["page"] = page
        if heading:
            meta["heading"] = heading
            
        chunk_text = " ".join(words)
        
        # Inject context prefix
        prefix = []
        if "title" in meta:
            prefix.append(f"Document: {meta['title']}")
        if heading:
            prefix.append(f"Section: {heading}")
        prefix.append(f"Page: {page}")
        
        context_prefix = " | ".join(prefix) + "\n\n"
        
        return Chunk(text=context_prefix + chunk_text, metadata=meta)
