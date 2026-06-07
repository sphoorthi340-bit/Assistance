"""
JARVIS System 4 — Domain/Intent Classifier
============================================
Two-stage classification pipeline that maps any user message
to the S4 domain taxonomy, enabling precise role-based routing.

Stage 1: Keyword heuristics (fast, ~0ms)
Stage 2: LLM-based classification via RAPID (fallback, ~200ms)

Taxonomy:
  academic   → concept | math | exam_prep | study_plan
  research   → paper_fetch | paper_analysis | literature_review | research_gap
  coding     → code_gen | debug | refactor | architecture | jarvis_build
  career     → ms_roadmap | profile_audit | sop_lor | career_decision
  productivity→ daily_plan | weekly_review | task_triage | distraction | focus
  quick      → definition | reminder | status_check | summary
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional

from backend.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# S4 Intent Data Structure
# ---------------------------------------------------------------------------

@dataclass
class S4Intent:
    """Classified intent from a user message."""
    domain: str                         # academic|research|coding|career|productivity|quick
    subdomain: str                      # fine-grained sub-type
    primary_role: str                   # chief|analyst|engineer|mentor|rapid
    secondary_roles: list = field(default_factory=list)
    pattern: str = "solo"               # solo|verify|pipeline|council
    confidence: float = 0.5
    is_emergency: bool = False          # True when exam mode triggered externally
    requires_memory_update: bool = True
    metadata: dict = field(default_factory=dict)

    def to_routing_trace(self) -> str:
        """Human-readable routing explanation."""
        roles = [self.primary_role]
        if self.secondary_roles:
            roles += self.secondary_roles
        role_str = " → ".join(r.upper() for r in roles)
        return (
            f"Domain: {self.domain}.{self.subdomain} | "
            f"Pattern: {self.pattern.upper()} | "
            f"Roles: {role_str} | "
            f"Confidence: {self.confidence:.0%}"
        )


# ---------------------------------------------------------------------------
# Keyword Rules
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, domain, subdomain, primary_role, secondary_roles, pattern)
KEYWORD_RULES = [

    # ── QUICK (highest priority — fast path) ─────────────────────────────
    (re.compile(r"\bwhat (is|are|does)\b.{0,40}\?$", re.I), "quick", "definition", "rapid", [], "solo"),
    (re.compile(r"\bdefin(e|ition)\b", re.I), "quick", "definition", "rapid", [], "solo"),
    (re.compile(r"\bremind(er|me)?\b", re.I), "quick", "reminder", "rapid", [], "solo"),
    (re.compile(r"\b(how many|count|total|status)\b.{0,30}\?$", re.I), "quick", "status_check", "rapid", [], "solo"),
    (re.compile(r"\btl;?dr|summarize (this|that|it)\b", re.I), "quick", "summary", "rapid", [], "solo"),

    # ── CODING ────────────────────────────────────────────────────────────
    (re.compile(r"\b(jarvis|system 4|s4).{0,30}(build|feature|add|implement|refactor|fix|bug|pr|commit)\b", re.I),
     "coding", "jarvis_build", "engineer", ["analyst", "chief"], "pipeline"),
    (re.compile(r"\b(project decision|major decision|project direction|architecture decision)\b", re.I),
     "coding", "project_decision", "chief", ["analyst"], "verify"),
    (re.compile(r"\b(write|create|generate|implement).{0,20}(code|function|class|script|program)\b", re.I),
     "coding", "code_gen", "engineer", [], "solo"),
    (re.compile(r"\b(debug|fix|error|traceback|exception|bug|crash|not working|fails)\b", re.I),
     "coding", "debug", "engineer", [], "solo"),
    (re.compile(r"\b(refactor|clean up|restructure|improve).{0,20}code\b", re.I),
     "coding", "refactor", "engineer", [], "solo"),
    (re.compile(r"\b(architecture|design pattern|module structure|how to structure)\b", re.I),
     "coding", "architecture", "analyst", ["engineer"], "verify"),
    (re.compile(r"\b(python|c\+\+|cpp|javascript|typescript|sql|bash|shell script)\b", re.I),
     "coding", "code_gen", "engineer", [], "solo"),

    # ── ACADEMIC ─────────────────────────────────────────────────────────
    (re.compile(r"\b(explain|what is|how does|teach me|i don.?t understand).{0,40}"
                r"(dsp|signal|fourier|fft|filter|transform|laplace|z-transform|"
                r"analog|digital|circuit|ece|electronics|electromagnet|maxwell|"
                r"microprocessor|microcontroller|vhdl|verilog|fpga|semiconductor|"
                r"transistor|opamp|diode|capacitor|inductor|resonance|"
                r"network|protocol|tcp|ip|routing|subnet|"
                r"control system|feedback|bode|nyquist|pid)\b", re.I),
     "academic", "concept", "mentor", [], "solo"),
    (re.compile(r"\b(derive|derivation|prove|proof|integral|differential equation|"
                r"laplace|fourier series|eigenvalue|matrix|determinant|"
                r"solve for|find the value|calculate|theorem)\b", re.I),
     "academic", "math", "analyst", ["mentor"], "pipeline"),
    (re.compile(r"\b(exam|test|viva|quiz).{0,20}(prep|prepare|study|ready|tips|strategy)\b", re.I),
     "academic", "exam_prep", "mentor", ["chief"], "pipeline"),
    (re.compile(r"\b(study plan|revision schedule|how.{0,10}study for|allocate time|timetable)\b", re.I),
     "academic", "study_plan", "chief", ["analyst"], "verify"),
    (re.compile(r"\b(cgpa|gpa|marks|grade|attendance|backlog)\b", re.I),
     "academic", "study_plan", "chief", [], "solo"),

    # ── RESEARCH ─────────────────────────────────────────────────────────
    (re.compile(r"\b(arxiv|paper|research paper|publication|journal|conference|ieee|acm)\b", re.I),
     "research", "paper_analysis", "analyst", ["mentor"], "verify"),
    (re.compile(r"\b(literature review|state of the art|related work|survey)\b", re.I),
     "research", "literature_review", "analyst", ["mentor"], "pipeline"),
    (re.compile(r"\b(research gap|open problem|future work|what.{0,10}nobody|unsolved)\b", re.I),
     "research", "research_gap", "analyst", [], "solo"),
    (re.compile(r"\b(find papers|suggest papers|papers on|recent work on)\b", re.I),
     "research", "paper_fetch", "rapid", [], "solo"),
    (re.compile(r"\b(tinyml|edge (ai|computing)|iot|aiot|embedded (ai|ml)|"
                r"model compression|quantization|pruning|federated learning|"
                r"mcunet|mobilenet|efficientnet|yolo|bert|llm on device)\b", re.I),
     "research", "paper_analysis", "analyst", [], "solo"),

    # ── CAREER / MS ABROAD ───────────────────────────────────────────────
    (re.compile(r"\b(ms abroad|masters|graduate school|phd|university|"
                r"application|admit|admit|sop|statement of purpose|lor|recommendation)\b", re.I),
     "career", "ms_roadmap", "chief", ["analyst"], "verify"),
    (re.compile(r"\b(gre|toefl|ielts|english test|vocab|quantitative reasoning)\b", re.I),
     "career", "ms_roadmap", "analyst", [], "solo"),
    (re.compile(r"\b(internship|job|placement|career|linkedin|resume|cv|"
                r"cold email|professor outreach|networking)\b", re.I),
     "career", "profile_audit", "analyst", [], "solo"),
    (re.compile(r"\b(should i|which (is better|option|path)|career decision|choose between)\b", re.I),
     "career", "career_decision", "analyst", ["chief", "mentor"], "council"),

    # ── PRODUCTIVITY ─────────────────────────────────────────────────────
    (re.compile(r"\b(morning brief|good morning|what should i do today|daily plan|plan (my|the) day)\b", re.I),
     "productivity", "daily_plan", "chief", [], "solo"),
    (re.compile(r"\b(weekly review|this week|week in review|how did i do|weekly summary)\b", re.I),
     "productivity", "weekly_review", "chief", ["analyst"], "pipeline"),
    (re.compile(r"\b(prioritize|what.{0,10}next|most important|focus on|what should i work on)\b", re.I),
     "productivity", "task_triage", "chief", [], "solo"),
    (re.compile(r"\b(distracted|youtube|netflix|procrastinat|can.?t focus|losing focus|off track)\b", re.I),
     "productivity", "distraction", "rapid", ["chief"], "verify"),
    (re.compile(r"\b(pomodoro|focus (session|timer|mode)|start focus|begin work)\b", re.I),
     "productivity", "focus", "rapid", [], "solo"),
]

# Domain → primary_role default (fallback when no rule matches)
DOMAIN_DEFAULTS = {
    "academic": ("concept", "mentor", [], "solo"),
    "research": ("paper_analysis", "analyst", [], "solo"),
    "coding": ("code_gen", "engineer", [], "solo"),
    "career": ("ms_roadmap", "analyst", [], "solo"),
    "productivity": ("task_triage", "chief", [], "solo"),
    "quick": ("definition", "rapid", [], "solo"),
}


# ---------------------------------------------------------------------------
# S4 Classifier
# ---------------------------------------------------------------------------

class S4Classifier:
    """
    Two-stage intent classifier for JARVIS System 4.

    Attributes:
        _rapid_client: Optional LLM client for Stage 2 classification.
                       If None, Stage 2 is skipped (keyword-only mode).
        _confidence_threshold: Min Stage 1 confidence before triggering Stage 2.
    """

    def __init__(self, rapid_client=None, confidence_threshold: float = 0.65):
        """
        Args:
            rapid_client: An initialized OllamaClient (used for LLM classification)
            confidence_threshold: Stage 1 scores below this trigger Stage 2
        """
        self._rapid = rapid_client
        self._threshold = confidence_threshold
        logger.info("S4Classifier initialized (threshold=%.2f)", confidence_threshold)

    def classify(self, message: str, is_exam_mode: bool = False) -> S4Intent:
        """
        Classify a user message into an S4Intent.

        Args:
            message: Raw user message
            is_exam_mode: If True, academic domains are boosted and council patterns
                          are escalated to pipeline for speed.

        Returns:
            S4Intent with domain, subdomain, roles, and pattern
        """
        # Stage 1: keyword heuristics
        intent, confidence = self._keyword_classify(message)

        # Stage 2: LLM fallback if confidence is low
        if confidence < self._threshold and self._rapid:
            llm_intent = self._llm_classify(message)
            if llm_intent:
                intent = llm_intent
                confidence = 0.75  # LLM classification is treated as medium-high confidence

        intent.confidence = confidence
        intent.is_emergency = is_exam_mode

        # Emergency mode override: boost academic + speed up patterns
        if is_exam_mode and intent.domain == "academic":
            if intent.pattern == "council":
                intent.pattern = "pipeline"
            intent.metadata["exam_mode_active"] = True

        logger.debug(
            "Classified: '%s' → %s.%s [%s] (%.0f%% conf)",
            message[:60], intent.domain, intent.subdomain,
            intent.pattern, confidence * 100
        )
        return intent

    # -------------------------------------------------------------------
    # Stage 1: Keyword Classification
    # -------------------------------------------------------------------

    def _keyword_classify(self, message: str) -> tuple[S4Intent, float]:
        """Fast keyword-pattern matching. Returns (S4Intent, confidence)."""
        msg = message.strip()

        for pattern, domain, subdomain, primary, secondaries, collab_pattern in KEYWORD_RULES:
            if pattern.search(msg):
                return S4Intent(
                    domain=domain,
                    subdomain=subdomain,
                    primary_role=primary,
                    secondary_roles=secondaries,
                    pattern=collab_pattern,
                    confidence=0.80,
                ), 0.80

        # Length-based complexity heuristic for fallback
        length = len(msg)
        if length < 40:
            return S4Intent(
                domain="quick", subdomain="definition",
                primary_role="rapid", pattern="solo", confidence=0.55,
            ), 0.55
        elif length < 150:
            return S4Intent(
                domain="productivity", subdomain="task_triage",
                primary_role="chief", pattern="solo", confidence=0.45,
            ), 0.45
        else:
            return S4Intent(
                domain="academic", subdomain="concept",
                primary_role="mentor", pattern="solo", confidence=0.40,
            ), 0.40

    # -------------------------------------------------------------------
    # Stage 2: LLM Classification
    # -------------------------------------------------------------------

    def _llm_classify(self, message: str) -> Optional[S4Intent]:
        """Use Rapid model (LLM) to classify when keywords are ambiguous."""
        prompt = f"""Classify this message for an AI routing system.

Message: "{message[:400]}"

Output valid JSON only, no explanation:
{{
  "domain": "academic" | "research" | "coding" | "career" | "productivity" | "quick",
  "subdomain": "concept" | "math" | "exam_prep" | "study_plan" | "paper_analysis" |
               "paper_fetch" | "literature_review" | "research_gap" |
               "code_gen" | "debug" | "refactor" | "architecture" | "jarvis_build" |
               "ms_roadmap" | "profile_audit" | "sop_lor" | "career_decision" |
               "daily_plan" | "weekly_review" | "task_triage" | "distraction" | "focus" |
               "definition" | "reminder" | "status_check" | "summary",
  "primary_role": "chief" | "analyst" | "engineer" | "mentor" | "rapid",
  "secondary_roles": [],
  "pattern": "solo" | "verify" | "pipeline" | "council"
}}"""
        try:
            original_model = self._rapid._model
            self._rapid._model = "llama3.2:1b"  # Use fast classifier model
            resp = self._rapid._client.chat(
                model="llama3.2:1b",
                messages=[{"role": "user", "content": prompt}],
                format="json"
            )
            self._rapid._model = original_model
            data = json.loads(resp["message"]["content"])
            return S4Intent(
                domain=data.get("domain", "quick"),
                subdomain=data.get("subdomain", "definition"),
                primary_role=data.get("primary_role", "rapid"),
                secondary_roles=data.get("secondary_roles", []),
                pattern=data.get("pattern", "solo"),
                confidence=0.75,
            )
        except Exception as e:
            logger.warning("LLM classification failed: %s", e)
            return None
