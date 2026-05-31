"""
Document Loader for Knowledge Pipeline
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import fitz  # PyMuPDF
from backend.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Document:
    source_path: str
    format: str
    content: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentLoader:
    def load(self, file_path: str | Path) -> Document:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext in [".md", ".markdown", ".txt"]:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported document format: {ext}")

    def _load_pdf(self, file_path: Path) -> Document:
        logger.debug("Loading PDF: %s", file_path)
        doc = fitz.open(file_path)
        text = ""
        # Preserve page references if we can, but simple approach is appending text.
        # But wait, chunking needs metadata-rich chunks (section titles, page refs).
        # We can pass page numbers in the text to help chunker, or keep it simple.
        # Actually, let's inject page boundaries.
        for i, page in enumerate(doc):
            text += f"\n\n--- Page {i+1} ---\n\n"
            text += page.get_text()
        
        title = doc.metadata.get("title") if doc.metadata else None  # pylint: disable=no-member
        if not title:
            title = file_path.stem
            
        page_count = len(doc)
        doc.close()
        return Document(
            source_path=str(file_path.absolute()),
            format="pdf",
            content=text,
            title=title,
            metadata={"pages": page_count}
        )

    def _load_text(self, file_path: Path) -> Document:
        import chardet
        logger.debug("Loading Text/Markdown: %s", file_path)
        raw_data = file_path.read_bytes()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        text = raw_data.decode(encoding, errors='replace')
        return Document(
            source_path=str(file_path.absolute()),
            format=file_path.suffix.lower()[1:],
            content=text,
            title=file_path.stem,
            metadata={"encoding": encoding}
        )
