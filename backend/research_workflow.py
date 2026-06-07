"""
JARVIS System 4 — Research Workflow Engine
============================================
Manages the research session pipeline:

  1. Paper selection (ArXiv API or manual entry)
  2. Deep analysis via Model B (Analyst)
  3. Concept explanation via Model D (Mentor) if needed
  4. Note saving to knowledge/papers/
  5. Weekly progress tracking
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional

from backend.logger import get_logger
from memory.s4_memory import S4MemoryManager

logger = get_logger(__name__)

try:
    import urllib.request
    import urllib.parse
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paper Data Structure
# ---------------------------------------------------------------------------

class PaperNote:
    """Represents a research paper that has been analyzed."""

    def __init__(
        self,
        title: str,
        authors: str = "",
        arxiv_id: str = "",
        domain: str = "",
        abstract: str = "",
        contribution: str = "",
        methodology: str = "",
        results: str = "",
        limitations: str = "",
        relevance: str = "",
        relevance_score: float = 0.5,
        notes: str = "",
    ):
        self.id = str(uuid.uuid4())[:8]
        self.title = title
        self.authors = authors
        self.arxiv_id = arxiv_id
        self.domain = domain
        self.abstract = abstract
        self.contribution = contribution
        self.methodology = methodology
        self.results = results
        self.limitations = limitations
        self.relevance = relevance
        self.relevance_score = relevance_score
        self.notes = notes
        self.date_read = datetime.now().strftime("%Y-%m-%d")

    def to_markdown(self) -> str:
        """Convert to a markdown note file."""
        return f"""# {self.title}

> **Date Read**: {self.date_read}  
> **Authors**: {self.authors or "Unknown"}  
> **ArXiv ID**: {self.arxiv_id or "N/A"}  
> **Domain**: {self.domain or "N/A"}  
> **Relevance Score**: {self.relevance_score:.1f}/1.0

---

## Abstract
{self.abstract or "_Not provided_"}

---

## Core Contribution
{self.contribution or "_Not analyzed_"}

## Methodology
{self.methodology or "_Not analyzed_"}

## Key Results
{self.results or "_Not analyzed_"}

## Limitations & Gaps
{self.limitations or "_Not analyzed_"}

## Relevance to AIoT / Edge AI Goals
{self.relevance or "_Not analyzed_"}

---

## Personal Notes
{self.notes or "_Add notes here_"}
"""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "arxiv_id": self.arxiv_id,
            "domain": self.domain,
            "contribution": self.contribution,
            "relevance_score": self.relevance_score,
            "date_read": self.date_read,
        }


# ---------------------------------------------------------------------------
# Research Workflow Engine
# ---------------------------------------------------------------------------

class ResearchWorkflow:
    """
    Manages research sessions for JARVIS System 4.

    Integrates with:
    - Model B (Analyst) for deep paper analysis
    - Model D (Mentor) for concept explanation
    - S4MemoryManager for progress tracking
    - knowledge/papers/ directory for note storage
    """

    # ArXiv categories relevant to the user's interests
    DEFAULT_ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.AR", "eess.SP"]

    def __init__(
        self,
        role_manager,
        s4_memory: S4MemoryManager,
        papers_dir: str = None,
    ):
        self._rm = role_manager
        self._mem = s4_memory
        self._papers_dir = papers_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "knowledge", "papers"
        )
        os.makedirs(self._papers_dir, exist_ok=True)
        logger.info("ResearchWorkflow initialized, papers dir: %s", self._papers_dir)

    # -------------------------------------------------------------------
    # ArXiv Integration
    # -------------------------------------------------------------------

    def fetch_recent_papers(
        self,
        query: str = "edge AI TinyML",
        max_results: int = 5,
        categories: list = None,
    ) -> list[dict]:
        """
        Fetch recent papers from ArXiv API.

        Args:
            query: Search query
            max_results: Number of results to return
            categories: ArXiv category filters (e.g. ["cs.AI", "cs.LG"])

        Returns:
            List of paper dicts with title, authors, abstract, arxiv_id
        """
        cats = categories or self.DEFAULT_ARXIV_CATEGORIES
        cat_filter = " OR ".join(f"cat:{c}" for c in cats)
        full_query = f"({query}) AND ({cat_filter})"

        encoded = urllib.parse.quote(full_query)
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query={encoded}"
            f"&start=0&max_results={max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                content = resp.read().decode("utf-8")
            return self._parse_arxiv_response(content)
        except Exception as e:
            logger.warning("ArXiv fetch failed: %s", e)
            return []

    def _parse_arxiv_response(self, xml_content: str) -> list[dict]:
        """Parse ArXiv API XML response into paper dicts."""
        import re
        papers = []
        entries = re.findall(r"<entry>(.*?)</entry>", xml_content, re.DOTALL)

        for entry in entries:
            def extract(tag):
                m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)
                return m.group(1).strip() if m else ""

            title = re.sub(r"\s+", " ", extract("title"))
            abstract = re.sub(r"\s+", " ", extract("summary"))

            # Authors
            author_names = re.findall(r"<name>(.*?)</name>", entry)
            authors = ", ".join(author_names[:3])
            if len(author_names) > 3:
                authors += " et al."

            # ArXiv ID
            id_raw = extract("id")
            arxiv_id = id_raw.split("/abs/")[-1] if "/abs/" in id_raw else id_raw

            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "arxiv_id": arxiv_id,
                "url": id_raw,
            })

        return papers

    # -------------------------------------------------------------------
    # Paper Analysis
    # -------------------------------------------------------------------

    def analyze_paper(
        self,
        title: str,
        abstract: str = "",
        full_text: str = "",
        authors: str = "",
        arxiv_id: str = "",
        domain: str = "",
    ) -> PaperNote:
        """
        Run the full paper analysis pipeline using Model B (Analyst).

        Args:
            title: Paper title
            abstract: Paper abstract
            full_text: Optional full paper text (for deeper analysis)
            authors: Author string
            arxiv_id: ArXiv identifier
            domain: Topical domain (e.g., "TinyML", "Edge AI")

        Returns:
            PaperNote with analysis filled in
        """
        note = PaperNote(
            title=title,
            authors=authors,
            arxiv_id=arxiv_id,
            domain=domain,
            abstract=abstract,
        )

        text_to_analyze = full_text or abstract
        if not text_to_analyze.strip():
            logger.warning("No text to analyze for paper: %s", title)
            return note

        analysis_prompt = f"""Analyze this research paper for an ECE student interested in AIoT and Edge AI.

PAPER: {title}
AUTHORS: {authors or "Unknown"}

TEXT:
{text_to_analyze[:3000]}

Provide a structured analysis:

CONTRIBUTION: [One sentence — what is the core novel contribution?]

METHOD: [2-3 sentences — what approach/technique do they use?]

RESULTS: [Key quantitative results or findings]

LIMITATIONS: [2-3 specific gaps or limitations]

EDGE AI RELEVANCE: [How does this connect to TinyML / Edge AI / AIoT / Embedded Systems?]

RELEVANCE SCORE: [0.0-1.0 — how relevant is this to the user's AIoT/Edge AI MS goals?]

Be precise. Use numbers from the paper where available."""

        try:
            result = self._rm.call_role("analyst", analysis_prompt)
            if result.success and result.content:
                self._parse_analysis_into_note(note, result.content)
        except Exception as e:
            logger.error("Paper analysis failed: %s", e)

        return note

    def _parse_analysis_into_note(self, note: PaperNote, analysis_text: str):
        """Extract structured sections from analyst response into note fields."""
        import re

        def extract_section(header: str) -> str:
            pattern = rf"{header}:?\s*(.*?)(?=\n[A-Z][A-Z ]+:|$)"
            m = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        note.contribution = extract_section("CONTRIBUTION")
        note.methodology = extract_section("METHOD")
        note.results = extract_section("RESULTS")
        note.limitations = extract_section("LIMITATIONS")
        note.relevance = extract_section("EDGE AI RELEVANCE")

        # Parse relevance score
        score_match = re.search(r"RELEVANCE SCORE:\s*([0-9.]+)", analysis_text, re.IGNORECASE)
        if score_match:
            try:
                note.relevance_score = min(1.0, max(0.0, float(score_match.group(1))))
            except ValueError:
                pass

    def explain_paper_concepts(self, paper_note: PaperNote, concept: str) -> str:
        """
        Ask Model D (Mentor) to explain a confusing concept from the paper.

        Args:
            paper_note: The analyzed paper
            concept: The specific concept to explain

        Returns:
            str — plain-language explanation
        """
        prompt = (
            f"Explain the concept of '{concept}' as it appears in this research context:\n\n"
            f"Paper: {paper_note.title}\n"
            f"Context: {paper_note.contribution}\n\n"
            f"The student is a III semester ECE student with basic ML knowledge. "
            f"Use an analogy and then give the technical definition."
        )
        try:
            result = self._rm.call_role("mentor", prompt)
            return result.content if result.success else f"Mentor unavailable. Concept: {concept}"
        except Exception as e:
            return f"Explanation failed: {e}"

    # -------------------------------------------------------------------
    # Note Storage
    # -------------------------------------------------------------------

    def save_paper_note(self, note: PaperNote) -> str:
        """
        Save the paper note to knowledge/papers/ directory.

        Returns:
            str — absolute path of saved file
        """
        safe_title = "".join(
            c if c.isalnum() or c in " -_" else "" for c in note.title
        )[:60].strip().replace(" ", "_")
        filename = f"{note.date_read}_{safe_title}.md"
        filepath = os.path.join(self._papers_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(note.to_markdown())
            logger.info("Paper note saved: %s", filepath)
        except Exception as e:
            logger.error("Failed to save paper note: %s", e)

        # Record in daily log
        self._mem.record_paper_read(note.title)

        return filepath

    def log_manual_paper(
        self,
        title: str,
        notes: str = "",
        authors: str = "",
        arxiv_id: str = "",
    ) -> PaperNote:
        """
        Quickly log a paper without full analysis (for papers already read).

        Returns:
            PaperNote — saved to disk
        """
        note = PaperNote(
            title=title,
            authors=authors,
            arxiv_id=arxiv_id,
            notes=notes,
        )
        self.save_paper_note(note)
        return note

    # -------------------------------------------------------------------
    # Progress Tracking
    # -------------------------------------------------------------------

    def get_weekly_stats(self) -> dict:
        """Return research stats for the current week."""
        weekly = self._mem.get_weekly_state()
        today = self._mem.get_daily_log()
        return {
            "papers_this_week": weekly.papers_read,
            "papers_today": len(today.papers_read),
            "target_per_week": 2,
            "on_track": weekly.papers_read >= 1,  # At least 1 by midweek
            "papers_today_titles": today.papers_read,
        }

    def list_saved_papers(self, limit: int = 20) -> list[dict]:
        """List recently saved paper notes."""
        papers = []
        try:
            files = sorted(
                [f for f in os.listdir(self._papers_dir) if f.endswith(".md")],
                reverse=True
            )[:limit]
            for f in files:
                # Parse date and title from filename
                parts = f.replace(".md", "").split("_", 1)
                date = parts[0] if len(parts) > 0 else "unknown"
                title = parts[1].replace("_", " ") if len(parts) > 1 else f
                papers.append({"filename": f, "date": date, "title": title})
        except Exception as e:
            logger.warning("Failed to list papers: %s", e)
        return papers

    def suggest_papers(self, focus_area: str = "edge AI TinyML") -> list[dict]:
        """
        Fetch and return paper suggestions from ArXiv.
        Model B (Analyst) selects the most relevant ones.
        """
        raw_papers = self.fetch_recent_papers(query=focus_area, max_results=8)
        if not raw_papers:
            return []

        if len(raw_papers) <= 3:
            return raw_papers

        # Ask Analyst to pick the top 3 most relevant
        paper_list = "\n".join(
            f"{i+1}. {p['title']} — {p['abstract'][:200]}"
            for i, p in enumerate(raw_papers)
        )
        prompt = (
            f"From this list of papers, select the 3 most relevant to "
            f"'Edge AI, TinyML, AIoT, Embedded ML' for an ECE student.\n\n"
            f"{paper_list}\n\n"
            f"Output only the numbers (e.g., 1, 3, 5)."
        )

        try:
            result = self._rm.call_role("rapid", prompt)
            if result.success:
                import re
                indices = [int(n) - 1 for n in re.findall(r"\d+", result.content)]
                selected = [raw_papers[i] for i in indices if 0 <= i < len(raw_papers)]
                return selected[:3] if selected else raw_papers[:3]
        except Exception:
            pass

        return raw_papers[:3]
