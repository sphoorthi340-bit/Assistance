"""
JARVIS System 4 — Core Dispatcher
====================================
Orchestrates multi-model collaboration using 4 patterns:

  SOLO     — 1 model handles the request end-to-end
  VERIFY   — primary model responds; second model fact-checks
  PIPELINE — models run in sequence, each building on the last
  COUNCIL  — models run in parallel; Chief synthesizes the results

The Dispatcher is the brain of System 4. Every user message enters here.
The existing ModelRouter handles provider-level fallback INSIDE each role call.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backend.logger import get_logger
from backend.s4_classifier import S4Classifier, S4Intent
from backend.s4_roles import S4RoleManager, RoleCallResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# S4 Response
# ---------------------------------------------------------------------------

@dataclass
class S4Response:
    """The final response produced by the S4 Dispatcher."""
    content: str
    primary_role: str
    roles_consulted: list
    pattern_used: str
    domain: str
    subdomain: str
    confidence: float
    total_time_ms: int
    role_times_ms: dict = field(default_factory=dict)
    routing_trace: str = ""
    fallback_used: bool = False
    conversation_id: Optional[str] = None

    def debug_footer(self) -> str:
        """Rich-formatted debug footer for terminal display."""
        roles_str = ", ".join(r.upper() for r in self.roles_consulted)
        return (
            f"[dim]S4 · {self.domain}.{self.subdomain} · "
            f"{self.pattern_used.upper()} · {roles_str} · "
            f"{self.total_time_ms}ms[/dim]"
        )


# ---------------------------------------------------------------------------
# S4 Dispatcher
# ---------------------------------------------------------------------------

class S4Dispatcher:
    """
    Core orchestrator for JARVIS System 4.

    Takes a user message, classifies it with S4Classifier,
    then executes the appropriate collaboration pattern
    using S4RoleManager.

    Args:
        role_manager: S4RoleManager (handles actual LLM calls per role)
        classifier: S4Classifier (intent classification)
        db: DatabaseManager (for logging routing decisions)
        settings: App settings (optional)
    """

    def __init__(
        self,
        role_manager: S4RoleManager,
        classifier: S4Classifier,
        db=None,
        settings=None,
    ):
        self._rm = role_manager
        self._clf = classifier
        self._db = db
        self._is_exam_mode = False  # Externally set by AcademicManager
        logger.info("S4Dispatcher initialized")

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def dispatch(
        self,
        message: str,
        conversation_history: list = None,
        context_kwargs: dict = None,
        conversation_id: str = None,
    ) -> S4Response:
        """
        Main entry point. Classify and route a user message.

        Args:
            message: Raw user message
            conversation_history: Prior conversation turns
            context_kwargs: Context for system prompt injection
                            (state_snapshot, memories, knowledge)
            conversation_id: Optional conversation ID for logging

        Returns:
            S4Response with content and full routing metadata
        """
        start = time.time()
        history = conversation_history or []
        ctx = context_kwargs or {}

        # 1. Classify intent
        intent = self._clf.classify(message, is_exam_mode=self._is_exam_mode)
        logger.info("Dispatching: %s", intent.to_routing_trace())

        # 2. Execute collaboration pattern
        if intent.pattern == "solo":
            result = self._run_solo(intent, message, history, ctx)
        elif intent.pattern == "verify":
            result = self._run_verify(intent, message, history, ctx)
        elif intent.pattern == "pipeline":
            result = self._run_pipeline(intent, message, history, ctx)
        elif intent.pattern == "council":
            result = self._run_council(intent, message, history, ctx)
        else:
            result = self._run_solo(intent, message, history, ctx)

        total_ms = int((time.time() - start) * 1000)

        response = S4Response(
            content=result["content"],
            primary_role=intent.primary_role,
            roles_consulted=result["roles_consulted"],
            pattern_used=intent.pattern,
            domain=intent.domain,
            subdomain=intent.subdomain,
            confidence=intent.confidence,
            total_time_ms=total_ms,
            role_times_ms=result.get("role_times_ms", {}),
            routing_trace=intent.to_routing_trace(),
            fallback_used=result.get("fallback_used", False),
            conversation_id=conversation_id,
        )

        # 3. Log to database
        self._log_dispatch(response, conversation_id)
        return response

    def set_exam_mode(self, active: bool):
        """Enable/disable exam emergency mode (called by AcademicManager)."""
        self._is_exam_mode = active
        if active:
            logger.warning("S4 EXAM MODE ACTIVATED — academic routing boosted")

    def dry_run(self, message: str) -> str:
        """Classify only (no LLM call). Returns routing trace for /s4 route."""
        intent = self._clf.classify(message, is_exam_mode=self._is_exam_mode)
        return (
            f"Domain:    {intent.domain}.{intent.subdomain}\n"
            f"Pattern:   {intent.pattern.upper()}\n"
            f"Primary:   {intent.primary_role.upper()}\n"
            f"Secondary: {', '.join(r.upper() for r in intent.secondary_roles) or '—'}\n"
            f"Confidence:{intent.confidence:.0%}\n"
            f"Exam Mode: {'YES ⚠' if self._is_exam_mode else 'no'}"
        )

    # -------------------------------------------------------------------
    # Collaboration Patterns
    # -------------------------------------------------------------------

    def _run_solo(
        self, intent: S4Intent, message: str, history: list, ctx: dict
    ) -> dict:
        """SOLO: One model, one call, direct response."""
        result = self._rm.call_role(
            intent.primary_role, message, history, ctx
        )
        return {
            "content": result.content if result.success else self._error_response(result),
            "roles_consulted": [intent.primary_role],
            "role_times_ms": {intent.primary_role: result.duration_ms},
            "fallback_used": not result.success,
        }

    def _run_verify(
        self, intent: S4Intent, message: str, history: list, ctx: dict
    ) -> dict:
        """
        VERIFY: Primary model answers; secondary model reviews the answer.
        Used for high-stakes responses where accuracy matters.
        """
        # Step 1: Primary responds
        primary_result = self._rm.call_role(intent.primary_role, message, history, ctx)
        if not primary_result.success:
            return self._run_solo(intent, message, history, ctx)

        primary_answer = primary_result.content

        # Step 2: Verifier reviews (if available)
        verifier_role = intent.secondary_roles[0] if intent.secondary_roles else None
        if not verifier_role or not self._rm.is_available(verifier_role):
            return {
                "content": primary_answer,
                "roles_consulted": [intent.primary_role],
                "role_times_ms": {intent.primary_role: primary_result.duration_ms},
                "fallback_used": False,
            }

        verify_prompt = (
            f"The following response was given to this question:\n\n"
            f"QUESTION: {message}\n\n"
            f"RESPONSE:\n{primary_answer}\n\n"
            f"Review it. Output:\n"
            f"VERIFIED: [what is correct]\n"
            f"GAPS: [what is missing]\n"
            f"CORRECTIONS: [what is wrong, if anything]\n"
            f"FINAL ANSWER: [improved version if corrections exist, else 'Response is accurate.']"
        )

        verify_result = self._rm.call_role(verifier_role, verify_prompt, [], ctx)
        role_times = {
            intent.primary_role: primary_result.duration_ms,
            verifier_role: verify_result.duration_ms,
        }

        if verify_result.success and "FINAL ANSWER:" in verify_result.content:
            # Extract just the final answer block
            parts = verify_result.content.split("FINAL ANSWER:")
            final = parts[-1].strip() if len(parts) > 1 else primary_answer
            if final.lower().startswith("response is accurate"):
                final = primary_answer
        else:
            final = primary_answer

        return {
            "content": final,
            "roles_consulted": [intent.primary_role, verifier_role],
            "role_times_ms": role_times,
            "fallback_used": False,
        }

    def _run_pipeline(
        self, intent: S4Intent, message: str, history: list, ctx: dict
    ) -> dict:
        """
        PIPELINE: Models run in sequence.
        Each model's output becomes the next model's input.
        Used for multi-step tasks (e.g., Analyst analyzes → Mentor explains).
        """
        pipeline_roles = [intent.primary_role] + intent.secondary_roles
        role_times = {}
        current_output = message
        roles_consulted = []
        fallback_used = False

        for i, role_name in enumerate(pipeline_roles):
            if not self._rm.is_available(role_name):
                logger.warning("Pipeline role '%s' unavailable, skipping", role_name)
                continue

            if i == 0:
                # First role gets the original message
                prompt = message
            else:
                # Subsequent roles get a synthesized prompt
                prompt = self._pipeline_handoff_prompt(
                    original_message=message,
                    previous_role=pipeline_roles[i - 1],
                    current_role=role_name,
                    previous_output=current_output,
                )

            result = self._rm.call_role(role_name, prompt, history if i == 0 else [], ctx)
            role_times[role_name] = result.duration_ms
            roles_consulted.append(role_name)

            if result.success and result.content.strip():
                current_output = result.content
            else:
                fallback_used = True
                # Skip failed role, continue with current output

        return {
            "content": current_output,
            "roles_consulted": roles_consulted,
            "role_times_ms": role_times,
            "fallback_used": fallback_used,
        }

    def _run_council(
        self, intent: S4Intent, message: str, history: list, ctx: dict
    ) -> dict:
        """
        COUNCIL: All roles respond in parallel (sequential for now),
        then Chief synthesizes into one unified recommendation.
        Used for major decisions (career, MS strategy, semester planning).
        """
        council_roles = [intent.primary_role] + intent.secondary_roles
        role_times = {}
        role_outputs = {}
        roles_consulted = []

        # Each role gives its perspective
        for role_name in council_roles:
            if not self._rm.is_available(role_name):
                continue
            council_prompt = (
                f"Give your perspective on this decision from the viewpoint of your role:\n\n"
                f"QUESTION: {message}\n\n"
                f"Provide your analysis (max 200 words):"
            )
            result = self._rm.call_role(role_name, council_prompt, [], ctx)
            role_times[role_name] = result.duration_ms
            roles_consulted.append(role_name)
            if result.success:
                role_outputs[role_name] = result.content

        if not role_outputs:
            return self._run_solo(intent, message, history, ctx)

        # Chief synthesizes
        synthesis_prompt = self._council_synthesis_prompt(message, role_outputs)
        chief_result = self._rm.call_role("chief", synthesis_prompt, [], ctx)
        role_times["chief"] = chief_result.duration_ms
        if "chief" not in roles_consulted:
            roles_consulted.append("chief")

        final = chief_result.content if chief_result.success else list(role_outputs.values())[0]

        return {
            "content": final,
            "roles_consulted": roles_consulted,
            "role_times_ms": role_times,
            "fallback_used": not chief_result.success,
        }

    # -------------------------------------------------------------------
    # Prompt Helpers
    # -------------------------------------------------------------------

    def _pipeline_handoff_prompt(
        self,
        original_message: str,
        previous_role: str,
        current_role: str,
        previous_output: str,
    ) -> str:
        role_descriptions = {
            "chief": "Chief of Staff (planning/prioritization)",
            "analyst": "Strategic Analyst (deep reasoning)",
            "engineer": "Software Engineer (coding/architecture)",
            "mentor": "Learning Mentor (explanation/teaching)",
            "rapid": "Rapid Assistant (quick responses)",
        }
        prev_desc = role_descriptions.get(previous_role, previous_role)
        curr_desc = role_descriptions.get(current_role, current_role)

        return (
            f"The {prev_desc} has responded to the following question:\n\n"
            f"ORIGINAL QUESTION: {original_message}\n\n"
            f"PREVIOUS RESPONSE:\n{previous_output}\n\n"
            f"As {curr_desc}, build on this response. "
            f"Add your expertise, correct any gaps, and produce a complete final answer."
        )

    def _council_synthesis_prompt(self, question: str, perspectives: dict) -> str:
        perspectives_text = "\n\n".join(
            f"[{role.upper()}]:\n{content}"
            for role, content in perspectives.items()
        )
        return (
            f"Multiple advisors have given their perspectives on this question:\n\n"
            f"QUESTION: {question}\n\n"
            f"PERSPECTIVES:\n{perspectives_text}\n\n"
            f"As Chief of Staff, synthesize these into ONE clear recommendation. "
            f"Structure your response as:\n"
            f"SYNTHESIS: [merged insight in 2-3 sentences]\n"
            f"RECOMMENDATION: [the single best action]\n"
            f"RISKS: [top 2 risks to watch]\n"
            f"NEXT 3 ACTIONS:\n1. ...\n2. ...\n3. ..."
        )

    # -------------------------------------------------------------------
    # Error & Logging
    # -------------------------------------------------------------------

    def _error_response(self, result: RoleCallResult) -> str:
        return (
            f"[JARVIS S4 Error] Role '{result.role_name}' failed: {result.error}\n"
            f"Please ensure {result.provider} is running with model '{result.model_id}' loaded."
        )

    def _log_dispatch(self, response: S4Response, conversation_id: Optional[str]):
        """Persist routing decision to s4_route_logs table."""
        if not self._db:
            return
        try:
            import json
            with self._db._connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO s4_route_logs
                    (id, timestamp, domain, subdomain, pattern, primary_role,
                     secondary_roles, confidence, total_time_ms, role_times_ms,
                     outcome, conversation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        datetime.now(timezone.utc).isoformat(),
                        response.domain,
                        response.subdomain,
                        response.pattern_used,
                        response.primary_role,
                        json.dumps(response.roles_consulted),
                        response.confidence,
                        response.total_time_ms,
                        json.dumps(response.role_times_ms),
                        "fallback" if response.fallback_used else "ok",
                        conversation_id,
                    )
                )
        except Exception as e:
            logger.debug("S4 route log failed (table may not exist yet): %s", e)
