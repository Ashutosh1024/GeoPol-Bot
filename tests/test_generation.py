"""Comprehensive Automated Test Suite for Answer Generation Pipeline.

Verifies:
1. Domain Gate rejection invariant: OUT_OF_DOMAIN never invokes retrieval or generation.
2. Domain Gate clarification invariant: UNCERTAIN routes to clarification.
3. Insufficient evidence abstention: Queries lacking evidence abstain without hallucinating.
4. Strict citation validation: Fabricated citation chunk IDs are detected and stripped.
5. End-to-end grounded generation: Live OpenAI model produces verified facts,
   strategic interpretations, uncertainties, and valid clickable citations.
6. FastAPI HTTP endpoint integration (/api/v1/chat/generate).
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import Settings
from backend.app.domain_gate.schemas import DomainLabel, DomainStructuredClassification, DomainGateResponse
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
from backend.app.generation.engine import GeoPolGenerationEngine

client = TestClient(app)


# ==============================================================================
# 1. Pipeline Invariant Tests (Mocked / Offline)
# ==============================================================================

def test_out_of_domain_never_calls_retrieval_or_generation():
    """Verify that OUT_OF_DOMAIN queries are rejected at the gate without invoking RAG or generation."""
    mock_classifier = MagicMock(spec=DomainGateClassifier)
    mock_classifier.classify.return_value = DomainGateResponse(
        query="Write Python quicksort",
        label=DomainLabel.OUT_OF_DOMAIN,
        confidence=0.99,
        primary_category="programming",
        reasoning="Software engineering request",
        allowed_to_proceed=False,
        clarification_prompt=None,
        processing_time_ms=5.0,
        model_used="mock_gate",
    )

    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_openai = MagicMock()

    engine = GeoPolGenerationEngine(
        settings=Settings(openai_api_key="mock-key"),
        classifier=mock_classifier,
        retriever=mock_retriever,
        client=mock_openai,
    )

    request = GenerationRequest(query="Write Python quicksort")
    response = engine.generate_answer(request)

    # Invariants
    assert response.status == AnswerStatus.REJECTED_OUT_OF_DOMAIN
    assert "Out of Scope" in response.markdown_output
    assert mock_retriever.retrieve.call_count == 0
    assert mock_openai.beta.chat.completions.parse.call_count == 0


def test_uncertain_query_routes_to_clarification():
    """Verify that UNCERTAIN queries route to clarification without invoking RAG or generation."""
    mock_classifier = MagicMock(spec=DomainGateClassifier)
    mock_classifier.classify.return_value = DomainGateResponse(
        query="Sanctions?",
        label=DomainLabel.UNCERTAIN,
        confidence=0.85,
        primary_category="ambiguous",
        reasoning="Too vague",
        allowed_to_proceed=False,
        clarification_prompt="Which sanctions regime or state are you inquiring about?",
        processing_time_ms=5.0,
        model_used="mock_gate",
    )

    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_openai = MagicMock()

    engine = GeoPolGenerationEngine(
        settings=Settings(openai_api_key="mock-key"),
        classifier=mock_classifier,
        retriever=mock_retriever,
        client=mock_openai,
    )

    request = GenerationRequest(query="Sanctions?")
    response = engine.generate_answer(request)

    assert response.status == AnswerStatus.REQUIRES_CLARIFICATION
    assert "Clarification Needed" in response.markdown_output
    assert mock_retriever.retrieve.call_count == 0
    assert mock_openai.beta.chat.completions.parse.call_count == 0


def test_insufficient_evidence_triggers_clean_abstention():
    """Verify that an in-domain query with zero retrieved evidence results in immediate abstention."""
    mock_classifier = MagicMock(spec=DomainGateClassifier)
    mock_classifier.classify.return_value = DomainGateResponse(
        query="What was the bilateral border agreement between Paraguay and Bolivia in 1938?",
        label=DomainLabel.IN_DOMAIN,
        confidence=0.95,
        primary_category="diplomacy_and_treaties",
        reasoning="Historic diplomatic border treaty",
        allowed_to_proceed=True,
        clarification_prompt=None,
        processing_time_ms=10.0,
        model_used="mock_gate",
    )

    # Empty retrieval response (no chunks in corpus meet threshold)
    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.retrieve.return_value = RetrievalResponse(
        query="What was the bilateral border agreement between Paraguay and Bolivia in 1938?",
        results=[],
        count=0,
        latency_ms=12.0,
    )

    mock_openai = MagicMock()

    engine = GeoPolGenerationEngine(
        settings=Settings(openai_api_key="mock-key"),
        classifier=mock_classifier,
        retriever=mock_retriever,
        client=mock_openai,
    )

    request = GenerationRequest(
        query="What was the bilateral border agreement between Paraguay and Bolivia in 1938?",
        allow_foundational_fallback=False,
    )
    response = engine.generate_answer(request)

    assert response.status == AnswerStatus.INSUFFICIENT_EVIDENCE_ABSTENTION
    assert "Insufficient Evidence" in response.markdown_output
    # Generation model should never be called when evidence count is zero in strict mode
    assert mock_openai.beta.chat.completions.parse.call_count == 0


def test_foundational_fallback_when_corpus_lacks_chunk():
    """Verify that when allow_foundational_fallback=True, a broad geopolitical query is synthesized rather than abstained."""
    mock_classifier = MagicMock(spec=DomainGateClassifier)
    mock_classifier.classify.return_value = DomainGateResponse(
        query="Analyze diplomatic relations between India and Russia",
        label=DomainLabel.IN_DOMAIN,
        confidence=0.98,
        primary_category="diplomacy_and_treaties",
        reasoning="Bilateral diplomatic statecraft",
        allowed_to_proceed=True,
        clarification_prompt=None,
        processing_time_ms=10.0,
        model_used="mock_gate",
    )

    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.retrieve.return_value = RetrievalResponse(
        query="Analyze diplomatic relations between India and Russia",
        results=[],
        count=0,
        latency_ms=10.0,
    )

    mock_openai = MagicMock()
    mock_parsed_payload = StructuredAnswerPayload(
        is_sufficient_evidence=True,
        abstention_statement=None,
        verified_facts=["The 1971 Indo-Soviet Treaty established mutual consultations."],
        strategic_interpretations=["India pursues strategic autonomy while maintaining robust defense ties with Moscow."],
        uncertainties_and_limitations=["Secondary sanctions and payment mechanisms present ongoing trade frictions."],
        cited_chunk_ids=[],
    )
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_parsed_payload
    mock_openai.beta.chat.completions.parse.return_value = MagicMock(choices=[mock_choice])

    engine = GeoPolGenerationEngine(
        settings=Settings(openai_api_key="mock-key"),
        classifier=mock_classifier,
        retriever=mock_retriever,
        client=mock_openai,
    )

    request = GenerationRequest(
        query="Analyze diplomatic relations between India and Russia",
        allow_foundational_fallback=True,
    )
    response = engine.generate_answer(request)

    assert response.status == AnswerStatus.ANSWERED
    assert response.grounding_source == "foundational_geopolitical_knowledge"
    assert "Bilateral Milestones" in response.markdown_output
    assert len(response.validated_citations) > 0


def test_citation_validation_strips_hallucinated_chunk_ids():
    """Verify that if the model emits a non-existent chunk ID, it is detected and flagged."""
    mock_classifier = MagicMock(spec=DomainGateClassifier)
    mock_classifier.classify.return_value = DomainGateResponse(
        query="NATO Article 5",
        label=DomainLabel.IN_DOMAIN,
        confidence=0.95,
        primary_category="military_alliances",
        reasoning="NATO query",
        allowed_to_proceed=True,
        clarification_prompt=None,
        processing_time_ms=5.0,
        model_used="mock_gate",
    )

    real_chunk = RetrievedEvidence(
        chunk_id="nato#0",
        document_title="The North Atlantic Treaty",
        source_name="NATO",
        source_url="https://www.nato.int/official",
        publication_date="1949-04-04",
        topic="military_alliances",
        content="Article 5 collective self defense.",
        score=0.88,
        rank=1,
    )

    mock_retriever = MagicMock(spec=KnowledgeRetriever)
    mock_retriever.retrieve.return_value = RetrievalResponse(
        query="NATO Article 5",
        results=[real_chunk],
        count=1,
        latency_ms=10.0,
    )

    # Model hallucinates a citation "fake_doc#99" alongside "nato#0"
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(
            message=MagicMock(
                parsed=StructuredAnswerPayload(
                    is_sufficient_evidence=True,
                    verified_facts=["Article 5 establishes collective self-defense."],
                    strategic_interpretations=["Deters external aggression."],
                    uncertainties_and_limitations=["Does not mandate kinetic force."],
                    cited_chunk_ids=["nato#0", "fake_doc#99"],  # fake_doc#99 is hallucinated
                )
            )
        )
    ]

    mock_openai = MagicMock()
    mock_openai.beta.chat.completions.parse.return_value = mock_completion

    engine = GeoPolGenerationEngine(
        settings=Settings(openai_api_key="mock-key"),
        classifier=mock_classifier,
        retriever=mock_retriever,
        client=mock_openai,
    )

    request = GenerationRequest(query="NATO Article 5")
    response = engine.generate_answer(request)

    assert response.status == AnswerStatus.ANSWERED
    assert len(response.validated_citations) == 1
    assert response.validated_citations[0].chunk_id == "nato#0"
    assert response.unverified_citations_count == 1  # Successfully detected and stripped fake_doc#99


# ==============================================================================
# 2. Live End-to-End Grounded Generation Test
# ==============================================================================

def test_live_grounded_answer_generation():
    """Verify live OpenAI Responses API produces structured answers with verified facts and citations."""
    engine = GeoPolGenerationEngine()
    if not engine.settings.is_openai_configured:
        pytest.skip("OpenAI API key not configured for live generation test.")

    request = GenerationRequest(
        query="Under the 1982 UNCLOS convention, what are the primary rights and obligations for ships exercising transit passage through international straits?",
        similarity_threshold=0.20,
        top_k=3,
    )

    response = engine.generate_answer(request)

    assert response.status == AnswerStatus.ANSWERED
    assert response.gate_result.allowed_to_proceed is True
    assert response.answer is not None
    assert response.answer.is_sufficient_evidence is True

    # Tripartite verification
    assert len(response.answer.verified_facts) > 0
    assert len(response.answer.strategic_interpretations) > 0
    assert len(response.answer.uncertainties_and_limitations) > 0

    # Citations verification
    assert len(response.validated_citations) > 0
    top_cite = response.validated_citations[0]
    assert "UNCLOS" in top_cite.document_title or "Straits" in top_cite.document_title
    assert top_cite.source_url.startswith("http")
    assert response.unverified_citations_count == 0

    # Markdown output check
    assert "Verified Facts" in response.markdown_output
    assert "Strategic Interpretations" in response.markdown_output
    assert "Verified Primary Sources" in response.markdown_output


# ==============================================================================
# 3. FastAPI Endpoint HTTP Test
# ==============================================================================

def test_api_chat_generate_endpoint():
    """Verify POST /api/v1/chat/generate responds with valid GenerationResponse schema."""
    payload = {
        "query": "What are the core commitments under Article 5 of the North Atlantic Treaty?",
        "similarity_threshold": 0.20,
        "top_k": 3,
    }
    response = client.post("/api/v1/chat/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "gate_result" in data
    assert "markdown_output" in data
    assert "validated_citations" in data
    assert "latency_ms" in data
    assert data["status"] in ["ANSWERED", "INSUFFICIENT_EVIDENCE_ABSTENTION"]
    if data["status"] == "ANSWERED":
        assert len(data["validated_citations"]) > 0
        assert data["validated_citations"][0]["source_url"].startswith("http")
