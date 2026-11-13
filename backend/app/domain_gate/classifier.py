"""Domain Gate Classifier for GeoPol Bot.

Implements structured tri-state query classification (IN_DOMAIN, OUT_OF_DOMAIN, UNCERTAIN)
using OpenAI's Structured Outputs API, reinforced with defensive injection heuristics
and fail-safe error handling.
"""

import time
import re
import logging
from typing import Optional, List
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, OpenAIError

from backend.app.core.config import Settings, get_settings
from backend.app.domain_gate.policy import DOMAIN_POLICY_SYSTEM_PROMPT
from backend.app.domain_gate.schemas import (
    DomainLabel,
    DomainStructuredClassification,
    DomainGateResponse,
)

logger = logging.getLogger("geopol_bot.domain_gate")

# Deterministic patterns for prompt injection & jailbreak attempts
ADVERSARIAL_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|rules|prompts)",
    r"you\s+are\s+now\s+(a|an|in)\s+(developer\s+mode|unrestricted|pirate|jailbreak|dan)",
    r"system\s+(override|prompt|reset)",
    r"disregard\s+the\s+above",
    r"act\s+as\s+a\s+(python\s+repl|linux\s+terminal|calculator)",
    r"print\s+your\s+(instructions|system\s+prompt)",
    r"forget\s+(everything|your\s+rules)",
]
ADVERSARIAL_REGEX = re.compile("|".join(ADVERSARIAL_PATTERNS), re.IGNORECASE)


class DomainGateClassifier:
    """Independent domain gate classifier evaluating if queries adhere to geopolitics scope."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[OpenAI] = None):
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of the official OpenAI client using backend credentials."""
        if self._client is None:
            if not self.settings.is_openai_configured:
                raise ValueError("OpenAI API key is not configured in backend settings.")
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout_seconds,
            )
        return self._client

    def _check_deterministic_heuristics(self, query: str) -> Optional[DomainStructuredClassification]:
        """Fast deterministic checks for trivial, empty, or adversarial inputs before LLM call."""
        cleaned = query.strip()

        # Check for empty or negligible input
        if len(cleaned) < 3:
            return DomainStructuredClassification(
                label=DomainLabel.UNCERTAIN,
                confidence=0.99,
                primary_category="ambiguous_short_input",
                reasoning="The input is too brief to convey a geopolitical inquiry.",
                clarification_prompt="Please specify your geopolitical question in more detail.",
            )

        # Check for overt prompt injection / instruction override attempts
        if ADVERSARIAL_REGEX.search(cleaned):
            logger.warning("Adversarial prompt injection pattern detected in input: %r", cleaned[:100])
            return DomainStructuredClassification(
                label=DomainLabel.OUT_OF_DOMAIN,
                confidence=0.99,
                primary_category="adversarial_injection",
                reasoning="Input contains an adversarial instruction-override or jailbreak pattern.",
                clarification_prompt=None,
            )

        return None

    def classify(
        self,
        query: str,
        conversation_context: Optional[List[str]] = None,
    ) -> DomainGateResponse:
        """Classify a query into IN_DOMAIN, OUT_OF_DOMAIN, or UNCERTAIN with structured validation.

        Guarantees that:
        - OUT_OF_DOMAIN and UNCERTAIN will have allowed_to_proceed = False.
        - API errors fail safely to UNCERTAIN with clear diagnostics.
        """
        start_time = time.perf_counter()

        # Step 1: Pre-filter heuristics (fast path)
        heuristic_result = self._check_deterministic_heuristics(query)
        if heuristic_result:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_response(query, heuristic_result, duration_ms, model_used="heuristic_rule_engine")

        # Step 2: Ensure API is configured
        if not self.settings.is_openai_configured:
            logger.warning("OpenAI API not configured; rejecting query via safe fail-closed posture.")
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            fallback = DomainStructuredClassification(
                label=DomainLabel.UNCERTAIN,
                confidence=0.0,
                primary_category="system_unconfigured",
                reasoning="OpenAI API key is missing or not configured on the backend.",
                clarification_prompt="System is currently in configuration mode. Please check backend settings.",
            )
            return self._build_response(query, fallback, duration_ms, model_used="fallback_standby")

        # Step 3: Invoke OpenAI Structured Outputs with policy
        try:
            # Build messages payload
            messages = [
                {"role": "system", "content": DOMAIN_POLICY_SYSTEM_PROMPT},
            ]

            # Append conversational context if available
            if conversation_context:
                formatted_context = "\n".join([f"- {ctx}" for ctx in conversation_context[-3:]])
                messages.append(
                    {
                        "role": "system",
                        "content": f"Prior conversation context for disambiguation only:\n{formatted_context}",
                    }
                )

            # Untrusted input containerization
            user_content = f"<UNTRUSTED_USER_INPUT>\n{query}\n</UNTRUSTED_USER_INPUT>"
            messages.append({"role": "user", "content": user_content})

            # Call OpenAI Structured Outputs parsing directly into Pydantic model
            completion = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,  # type: ignore[arg-type]
                response_format=DomainStructuredClassification,
                temperature=0.0,  # Deterministic classification
            )

            structured_output = completion.choices[0].message.parsed
            if structured_output is None:
                raise ValueError("OpenAI API did not return structured classification.")

            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_response(query, structured_output, duration_ms, model_used=self.settings.openai_model)

        except RateLimitError as exc:
            logger.error("OpenAI RateLimitError during classification: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            fallback = DomainStructuredClassification(
                label=DomainLabel.UNCERTAIN,
                confidence=0.0,
                primary_category="api_rate_limit",
                reasoning="OpenAI API rate limit encountered. Query held safely in UNCERTAIN state.",
                clarification_prompt="The classification service is temporarily rate limited. Please retry in a few moments.",
            )
            return self._build_response(query, fallback, duration_ms, model_used="fail_safe_rate_limit")

        except (APIConnectionError, APITimeoutError) as exc:
            logger.error("OpenAI Network/Timeout error during classification: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            fallback = DomainStructuredClassification(
                label=DomainLabel.UNCERTAIN,
                confidence=0.0,
                primary_category="api_connectivity_error",
                reasoning="Classification API unreachable or timed out. Failing safely to UNCERTAIN.",
                clarification_prompt="Network timeout contacting intelligence service. Please retry.",
            )
            return self._build_response(query, fallback, duration_ms, model_used="fail_safe_connectivity")

        except OpenAIError as exc:
            logger.error("OpenAI General Error during classification: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            fallback = DomainStructuredClassification(
                label=DomainLabel.UNCERTAIN,
                confidence=0.0,
                primary_category="api_error",
                reasoning=f"OpenAI service error: {str(exc)[:100]}",
                clarification_prompt="An error occurred while evaluating your question. Please retry.",
            )
            return self._build_response(query, fallback, duration_ms, model_used="fail_safe_openai_error")

        except Exception as exc:
            logger.exception("Unexpected error in domain gate classifier: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            fallback = DomainStructuredClassification(
                label=DomainLabel.UNCERTAIN,
                confidence=0.0,
                primary_category="system_error",
                reasoning="Internal processing error. Defaulting safely to UNCERTAIN.",
                clarification_prompt="An unexpected error occurred. Please rephrase or retry.",
            )
            return self._build_response(query, fallback, duration_ms, model_used="fail_safe_internal")

    def _build_response(
        self,
        query: str,
        classification: DomainStructuredClassification,
        duration_ms: float,
        model_used: str,
    ) -> DomainGateResponse:
        """Construct validated DomainGateResponse enforcing security invariant:

        INVARIANT: allowed_to_proceed is strictly True ONLY if label == IN_DOMAIN
        and confidence >= 0.5. OUT_OF_DOMAIN and UNCERTAIN must NEVER proceed.
        """
        allowed = (classification.label == DomainLabel.IN_DOMAIN) and (classification.confidence >= 0.5)

        return DomainGateResponse(
            query=query,
            label=classification.label,
            confidence=classification.confidence,
            primary_category=classification.primary_category,
            reasoning=classification.reasoning,
            allowed_to_proceed=allowed,
            clarification_prompt=classification.clarification_prompt,
            processing_time_ms=duration_ms,
            model_used=model_used,
        )
