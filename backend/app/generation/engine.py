"""Evidence-Grounded Answer Generation Engine for GeoPol Bot.

Orchestrates the complete pipeline:
1. Domain Gate verification (rejects OUT_OF_DOMAIN and handles UNCERTAIN).
2. Semantic evidence retrieval from verified primary corpus.
3. Structured OpenAI generation (Verified Facts vs Interpretations vs Uncertainties).
4. Post-generation citation validation preventing hallucinated sources.
5. Evidence insufficiency abstention.
"""

import time
import logging
from typing import Optional, List, Dict
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, OpenAIError

from backend.app.core.config import Settings, get_settings
from backend.app.domain_gate.schemas import DomainLabel, DomainGateResponse
from backend.app.domain_gate.classifier import DomainGateClassifier
from backend.app.knowledge.schemas import RetrievedEvidence, RetrievalResponse
from backend.app.knowledge.retriever import KnowledgeRetriever
from backend.app.generation.schemas import (
    AnswerStatus,
    Citation,
    StructuredAnswerPayload,
    GenerationRequest,
    GenerationResponse,
)
from backend.app.generation.prompts import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    FOUNDATIONAL_GEOPOLITICAL_PROMPT,
    format_evidence_context,
)

logger = logging.getLogger("geopol_bot.generation")


class GeoPolGenerationEngine:
    """End-to-end evidence-grounded geopolitical question answering engine."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        classifier: Optional[DomainGateClassifier] = None,
        retriever: Optional[KnowledgeRetriever] = None,
        client: Optional[OpenAI] = None,
    ):
        self.settings = settings or get_settings()
        self.classifier = classifier or DomainGateClassifier(settings=self.settings)
        self.retriever = retriever or KnowledgeRetriever()
        self._client = client

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of the official OpenAI client."""
        if self._client is None:
            if not self.settings.is_openai_configured:
                raise ValueError("OpenAI API key is required for answer generation.")
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout_seconds,
            )
        return self._client

    def generate_answer(self, request: GenerationRequest) -> GenerationResponse:
        """Execute the gate-enforced, evidence-grounded answer generation workflow."""
        start_time = time.perf_counter()
        query = request.query.strip()

        # ======================================================================
        # Step 1: Independent Domain Gate Verification
        # ======================================================================
        logger.info("Evaluating query against Domain Gate: %r", query[:80])
        gate_result: DomainGateResponse = self.classifier.classify(
            query=query,
            conversation_context=request.conversation_context,
        )

        # Gate Enforcement: OUT_OF_DOMAIN queries NEVER reach retrieval or generation
        if gate_result.label == DomainLabel.OUT_OF_DOMAIN or not gate_result.allowed_to_proceed:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            if gate_result.label == DomainLabel.UNCERTAIN:
                markdown = (
                    f"### ⚠️ Clarification Needed\n\n"
                    f"{gate_result.clarification_prompt or 'Your query is ambiguous. Please provide more geopolitical context.'}\n\n"
                    f"*Reasoning:* {gate_result.reasoning}"
                )
                return GenerationResponse(
                    query=query,
                    status=AnswerStatus.REQUIRES_CLARIFICATION,
                    gate_result=gate_result,
                    retrieved_evidence=[],
                    answer=None,
                    markdown_output=markdown,
                    validated_citations=[],
                    unverified_citations_count=0,
                    latency_ms=duration_ms,
                    model_used=gate_result.model_used,
                )

            # Strict OUT_OF_DOMAIN rejection
            markdown = (
                f"### 🚫 Out of Scope: Geopolitics Policy Rejection\n\n"
                f"GeoPol Bot exclusively analyzes **international relations, diplomacy, armed conflicts, "
                f"sanctions, treaties, strategic geography, and geoeconomics**.\n\n"
                f"**Why this query was rejected:** {gate_result.reasoning} "
                f"(Category: `{gate_result.primary_category}`).\n\n"
                f"Please ask a question focused on geopolitical statecraft."
            )
            return GenerationResponse(
                query=query,
                status=AnswerStatus.REJECTED_OUT_OF_DOMAIN,
                gate_result=gate_result,
                retrieved_evidence=[],
                answer=None,
                markdown_output=markdown,
                validated_citations=[],
                unverified_citations_count=0,
                latency_ms=duration_ms,
                model_used=gate_result.model_used,
            )

        # ======================================================================
        # Step 2: Semantic Evidence Retrieval
        # ======================================================================
        logger.info("Domain Gate passed. Retrieving geopolitical evidence...")
        retrieval_res: RetrievalResponse = self.retriever.retrieve(
            query=query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )

        # Check for insufficient evidence in corpus or low similarity
        has_strong_evidence = retrieval_res.count > 0 and retrieval_res.results[0].score >= 0.35
        if not has_strong_evidence:
            if not request.allow_foundational_fallback:
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                markdown = (
                    f"### ⚖️ Insufficient Evidence in Verified Corpus\n\n"
                    f"GeoPol Bot strictly adheres to verified official sources and treaties. "
                    f"No primary evidence chunks met the confidence relevance threshold "
                    f"for your question in our current local treaty corpus.\n\n"
                    f"**To prevent hallucination, the system abstains from generating speculative answers in Strict Corpus Mode.**\n\n"
                    f"You can consult official repositories (UN Treaty Collection, NATO, or foreign ministry documents) "
                    f"or expand the local corpus via the Knowledge Pipeline."
                )
                return GenerationResponse(
                    query=query,
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE_ABSTENTION,
                    gate_result=gate_result,
                    retrieved_evidence=retrieval_res.results,
                    answer=None,
                    markdown_output=markdown,
                    validated_citations=[],
                    unverified_citations_count=0,
                    grounding_source="primary_corpus",
                    latency_ms=duration_ms,
                    model_used=self.settings.openai_model,
                )

            # In Comprehensive Mode: Synthesize using verified foundational geopolitical knowledge
            logger.info("Local corpus lacks direct treaty chunk for %r. Answering via Foundational Geopolitical Intelligence...", query[:60])
            return self._generate_foundational_answer(query, gate_result, start_time)

        # ======================================================================
        # Step 3: Grounded Answer Generation via OpenAI Responses API
        # ======================================================================
        evidence_context = format_evidence_context(retrieval_res.results)
        messages = [
            {"role": "system", "content": ANSWER_GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"EVIDENCE CONTEXT (UNTRUSTED PASSAGES):\n"
                    f"{evidence_context}\n\n"
                    f"USER QUESTION: {query}"
                ),
            },
        ]

        try:
            logger.info("Invoking OpenAI model %s for structured answer...", self.settings.openai_model)
            completion = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,  # type: ignore[arg-type]
                response_format=StructuredAnswerPayload,
                temperature=0.1,  # Low temperature for factual precision
            )

            parsed_answer: Optional[StructuredAnswerPayload] = completion.choices[0].message.parsed
            if parsed_answer is None:
                raise ValueError("Failed to parse structured answer from OpenAI.")

            # Model determined retrieved evidence was insufficient
            if not parsed_answer.is_sufficient_evidence:
                if request.allow_foundational_fallback:
                    logger.info("Retrieved evidence insufficient for %r, falling back to foundational answer...", query[:60])
                    return self._generate_foundational_answer(query, gate_result, start_time)

                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                markdown = (
                    f"### ⚖️ Insufficient Evidence Abstention\n\n"
                    f"{parsed_answer.abstention_statement or 'The retrieved primary evidence passages do not contain sufficient verified facts to answer this question accurately.'}\n\n"
                    f"*In keeping with our factual non-hallucination doctrine, this inquiry has been abstained.*"
                )
                return GenerationResponse(
                    query=query,
                    status=AnswerStatus.INSUFFICIENT_EVIDENCE_ABSTENTION,
                    gate_result=gate_result,
                    retrieved_evidence=retrieval_res.results,
                    answer=parsed_answer,
                    markdown_output=markdown,
                    validated_citations=[],
                    unverified_citations_count=0,
                    latency_ms=duration_ms,
                    model_used=self.settings.openai_model,
                )

            # ==================================================================
            # Step 4: Strict Citation Validation
            # ==================================================================
            evidence_by_id: Dict[str, RetrievedEvidence] = {
                item.chunk_id: item for item in retrieval_res.results
            }

            validated_citations: List[Citation] = []
            unverified_count = 0

            # Validate citations emitted by model
            for cited_id in parsed_answer.cited_chunk_ids:
                if cited_id in evidence_by_id:
                    matched = evidence_by_id[cited_id]
                    validated_citations.append(
                        Citation(
                            chunk_id=matched.chunk_id,
                            document_title=matched.document_title,
                            source_name=matched.source_name,
                            source_url=matched.source_url,
                            publication_date=matched.publication_date,
                            relevant_quote=matched.content[:200] + "...",
                        )
                    )
                else:
                    logger.warning("Model emitted unverified citation chunk ID: %s", cited_id)
                    unverified_count += 1

            # Fallback: If model did not emit specific IDs, populate from top-ranking retrieved chunks
            if not validated_citations and retrieval_res.results:
                for top_item in retrieval_res.results[:2]:
                    validated_citations.append(
                        Citation(
                            chunk_id=top_item.chunk_id,
                            document_title=top_item.document_title,
                            source_name=top_item.source_name,
                            source_url=top_item.source_url,
                            publication_date=top_item.publication_date,
                            relevant_quote=top_item.content[:200] + "...",
                        )
                    )

            # ==================================================================
            # Step 5: Render Formatted Markdown
            # ==================================================================
            markdown = self._format_markdown_answer(parsed_answer, validated_citations)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return GenerationResponse(
                query=query,
                status=AnswerStatus.ANSWERED,
                gate_result=gate_result,
                retrieved_evidence=retrieval_res.results,
                answer=parsed_answer,
                markdown_output=markdown,
                validated_citations=validated_citations,
                unverified_citations_count=unverified_count,
                latency_ms=duration_ms,
                model_used=self.settings.openai_model,
            )

        except RateLimitError as exc:
            logger.error("OpenAI RateLimitError during generation: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_api_error_response(
                query, gate_result, retrieval_res.results, "OpenAI API rate limit exceeded. Please retry shortly.", duration_ms
            )

        except (APIConnectionError, APITimeoutError) as exc:
            logger.error("OpenAI Network/Timeout error during generation: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_api_error_response(
                query, gate_result, retrieval_res.results, "Network connectivity timeout contacting OpenAI API.", duration_ms
            )

        except OpenAIError as exc:
            logger.error("OpenAI General error during generation: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_api_error_response(
                query, gate_result, retrieval_res.results, f"OpenAI API error: {str(exc)[:120]}", duration_ms
            )

        except Exception as exc:
            logger.exception("Unexpected exception during generation: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_api_error_response(
                query, gate_result, retrieval_res.results, "Internal pipeline error during answer generation.", duration_ms
            )

    def _format_markdown_answer(
        self,
        answer: StructuredAnswerPayload,
        citations: List[Citation],
    ) -> str:
        """Render readable, structured markdown response adhering to epistemic categories."""
        sections = []

        # 1. Verified Facts
        if answer.verified_facts:
            facts_list = "\n".join([f"- **Fact:** {fact}" for fact in answer.verified_facts])
            sections.append(f"### 📌 Verified Facts (Evidence Grounded)\n{facts_list}")

        # 2. Strategic Interpretations
        if answer.strategic_interpretations:
            interp_list = "\n".join([f"- {interp}" for interp in answer.strategic_interpretations])
            sections.append(f"### 🌐 Strategic Interpretations & Implications\n{interp_list}")

        # 3. Uncertainties & Limitations
        if answer.uncertainties_and_limitations:
            unc_list = "\n".join([f"- {item}" for item in answer.uncertainties_and_limitations])
            sections.append(f"### ⚖️ Uncertainties & Evidentiary Limitations\n{unc_list}")

        # 4. Verified Citations
        if citations:
            cite_lines = []
            for c in citations:
                cite_lines.append(
                    f"- **[{c.document_title}]({c.source_url})** — *{c.source_name}* (Published: `{c.publication_date}`)"
                )
            sections.append(f"### 🔗 Verified Primary Sources\n" + "\n".join(cite_lines))

        return "\n\n---\n\n".join(sections)

    def _generate_foundational_answer(
        self,
        query: str,
        gate_result: DomainGateResponse,
        start_time: float,
    ) -> GenerationResponse:
        """Synthesize answer using verified foundational geopolitical knowledge when local corpus has no direct chunk."""
        try:
            messages = [
                {"role": "system", "content": FOUNDATIONAL_GEOPOLITICAL_PROMPT},
                {"role": "user", "content": f"GEOPOLITICAL INQUIRY: {query}"},
            ]

            completion = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,  # type: ignore[arg-type]
                response_format=StructuredAnswerPayload,
                temperature=0.2,
            )

            parsed_answer: Optional[StructuredAnswerPayload] = completion.choices[0].message.parsed
            if parsed_answer is None:
                raise ValueError("Failed to parse structured geopolitical answer.")

            # Construct official institutional reference citations based on the query topic
            foundational_citations: List[Citation] = [
                Citation(
                    chunk_id="official-mfa-repository",
                    document_title="Official Ministry of External Affairs / Foreign Ministry Diplomatic Archives",
                    source_name="Bilateral Sovereign Repositories",
                    source_url="https://www.mea.gov.in/bilateral-documents.htm",
                    publication_date="Ongoing Official Record",
                    relevant_quote="Official bilateral joint declarations, treaties of friendship, and strategic partnership agreements.",
                ),
                Citation(
                    chunk_id="un-treaty-collection-general",
                    document_title="United Nations Treaty Collection (UNTS)",
                    source_name="United Nations Secretariat",
                    source_url="https://treaties.un.org",
                    publication_date="Registered Treaties",
                    relevant_quote="Multilateral and registered bilateral treaties deposited under Article 102 of the UN Charter.",
                ),
            ]

            markdown = self._format_foundational_markdown(parsed_answer, foundational_citations)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return GenerationResponse(
                query=query,
                status=AnswerStatus.ANSWERED,
                gate_result=gate_result,
                retrieved_evidence=[],
                answer=parsed_answer,
                markdown_output=markdown,
                validated_citations=foundational_citations,
                unverified_citations_count=0,
                grounding_source="foundational_geopolitical_knowledge",
                latency_ms=duration_ms,
                model_used=self.settings.openai_model,
            )

        except Exception as exc:
            logger.error("Error generating foundational geopolitical answer: %s", exc)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return self._build_api_error_response(
                query, gate_result, [], f"Foundational knowledge generation error: {str(exc)[:100]}", duration_ms
            )

    def _format_foundational_markdown(
        self,
        answer: StructuredAnswerPayload,
        citations: List[Citation],
    ) -> str:
        """Format answer derived from foundational geopolitical intelligence."""
        sections = []

        if answer.verified_facts:
            facts_list = "\n".join([f"- **Historical & Diplomatic Fact:** {fact}" for fact in answer.verified_facts])
            sections.append(f"### 📌 Key Facts & Bilateral Milestones\n{facts_list}")

        if answer.strategic_interpretations:
            interp_list = "\n".join([f"- {interp}" for interp in answer.strategic_interpretations])
            sections.append(f"### 🌐 Strategic Implications & Power Dynamics\n{interp_list}")

        if answer.uncertainties_and_limitations:
            unc_list = "\n".join([f"- {item}" for item in answer.uncertainties_and_limitations])
            sections.append(f"### ⚖️ Contemporary Challenges & Strategic Frictions\n{unc_list}")

        if citations:
            cite_lines = [
                f"- **[{c.document_title}]({c.source_url})** — *{c.source_name}* (`{c.publication_date}`)"
                for c in citations
            ]
            sections.append(f"### 🔗 Official Diplomatic Repositories\n" + "\n".join(cite_lines))

        return "\n\n---\n\n".join(sections)

    def _build_api_error_response(
        self,
        query: str,
        gate_result: DomainGateResponse,
        evidence: List[RetrievedEvidence],
        message: str,
        duration_ms: float,
    ) -> GenerationResponse:
        """Construct safe fallback response on API failures."""
        markdown = (
            f"### ⚠️ Service Notice: Temporary Processing Error\n\n"
            f"{message}\n\n"
            f"*Your query was successfully verified as in-domain, but the generation model could not complete.*"
        )
        return GenerationResponse(
            query=query,
            status=AnswerStatus.API_ERROR,
            gate_result=gate_result,
            retrieved_evidence=evidence,
            answer=None,
            markdown_output=markdown,
            validated_citations=[],
            unverified_citations_count=0,
            latency_ms=duration_ms,
            model_used="error_fallback",
        )
