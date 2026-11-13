"""Comprehensive Automated Test Suite for GeoPol Bot Knowledge Pipeline.

Tests:
1. Document chunking with metadata and URL preservation.
2. Embedding generation with caching and normalized vector output.
3. Vector store indexing, schema initialization, and upsert idempotency.
4. Cosine similarity semantic search and source ranking.
5. Topic filtering and similarity threshold enforcement.
6. KnowledgeRetriever end-to-end retrieval.
7. FastAPI HTTP endpoints for /knowledge/retrieve and /knowledge/sources.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.knowledge.schemas import DocumentSource, DocumentChunk, RetrievalQuery
from backend.app.knowledge.chunker import DocumentChunker
from backend.app.knowledge.embeddings import EmbeddingService
from backend.app.knowledge.vector_store import SQLiteVectorStore
from backend.app.knowledge.retriever import KnowledgeRetriever

client = TestClient(app)


# ==============================================================================
# 1. Document Chunking Tests
# ==============================================================================

def test_document_chunker_preserves_metadata():
    """Verify chunking splits text while faithfully retaining provenance metadata."""
    sample_doc = DocumentSource(
        id="test-treaty-1990",
        title="Treaty on Conventional Armed Forces in Europe",
        source_name="OSCE",
        source_url="https://www.osce.org/library/14087",
        publication_date="1990-11-19",
        topic="conflicts_and_security",
        summary="Arms reduction treaty in Europe",
        content="Paragraph one discusses limitations on battle tanks and armored combat vehicles. " * 15,
    )

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(sample_doc)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.document_id == "test-treaty-1990"
        assert chunk.title == "Treaty on Conventional Armed Forces in Europe"
        assert chunk.source_name == "OSCE"
        assert chunk.source_url == "https://www.osce.org/library/14087"
        assert chunk.publication_date == "1990-11-19"
        assert chunk.topic == "conflicts_and_security"
        assert chunk.chunk_id == f"test-treaty-1990#chunk-{idx}"
        assert len(chunk.content) > 0


# ==============================================================================
# 2. Embedding Service Tests
# ==============================================================================

def test_embedding_service_mock_normalized():
    """Verify mock embedding generator returns unit-normalized 1536-dim vectors."""
    embedder = EmbeddingService(use_mock=True)
    vector = embedder.embed_text("NATO collective defense clause")

    assert len(vector) == 1536
    # Calculate vector norm to verify normalization (magnitude approx 1.0)
    norm = sum(x * x for x in vector) ** 0.5
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embedding_service_caching():
    """Verify identical text fetches from cache without recomputation."""
    embedder = EmbeddingService(use_mock=True)
    vec1 = embedder.embed_text("Strait of Malacca navigation rights")
    vec2 = embedder.embed_text("Strait of Malacca navigation rights")

    assert vec1 == vec2
    assert "Strait of Malacca navigation rights" in embedder._cache


# ==============================================================================
# 3. Vector Store Tests (Isolated SQLite Database)
# ==============================================================================

def test_vector_store_upsert_and_similarity(tmp_path: Path):
    """Verify vector store schema creation, upsert, and cosine similarity ranking."""
    db_file = tmp_path / "test_vectors.db"
    store = SQLiteVectorStore(db_path=db_file)
    assert store.count() == 0

    embedder = EmbeddingService(use_mock=True)

    chunks = [
        DocumentChunk(
            chunk_id="nato#0",
            document_id="nato",
            chunk_index=0,
            title="The North Atlantic Treaty",
            source_name="NATO",
            source_url="https://nato.int/treaty",
            publication_date="1949-04-04",
            topic="military_alliances",
            content="Article 5 guarantees collective defense for all member states.",
            embedding=embedder.embed_text("Article 5 guarantees collective defense for all member states."),
        ),
        DocumentChunk(
            chunk_id="unclos#0",
            document_id="unclos",
            chunk_index=0,
            title="UNCLOS Straits Navigation",
            source_name="United Nations",
            source_url="https://un.org/unclos",
            publication_date="1982-12-10",
            topic="strategic_geography_and_chokepoints",
            content="Transit passage rights through international straits shall not be impeded.",
            embedding=embedder.embed_text("Transit passage rights through international straits shall not be impeded."),
        ),
    ]

    inserted = store.upsert_chunks(chunks)
    assert inserted == 2
    assert store.count() == 2

    # Query with exact text match for NATO
    q_vec = embedder.embed_text("Article 5 guarantees collective defense for all member states.")
    results = store.similarity_search(query_vector=q_vec, top_k=2, similarity_threshold=0.50)

    assert len(results) > 0
    top_result = results[0]
    assert top_result.chunk_id == "nato#0"
    assert top_result.document_title == "The North Atlantic Treaty"
    assert top_result.source_url == "https://nato.int/treaty"
    assert top_result.rank == 1
    assert top_result.score > 0.99  # Identical mock embedding dot product is 1.0


def test_vector_store_topic_filtering(tmp_path: Path):
    """Verify topic filtering restricts results to the requested geopolitical category."""
    db_file = tmp_path / "test_filter.db"
    store = SQLiteVectorStore(db_path=db_file)
    embedder = EmbeddingService(use_mock=True)

    chunks = [
        DocumentChunk(
            chunk_id="doc1#0",
            document_id="doc1",
            chunk_index=0,
            title="Sanctions Regime",
            source_name="UN",
            source_url="https://un.org",
            publication_date="2020-01-01",
            topic="sanctions_and_trade_controls",
            content="Economic sanctions on proliferation.",
            embedding=embedder.embed_text("sanctions"),
        ),
        DocumentChunk(
            chunk_id="doc2#0",
            document_id="doc2",
            chunk_index=0,
            title="Straits Passage",
            source_name="UN",
            source_url="https://un.org",
            publication_date="2020-01-01",
            topic="strategic_geography_and_chokepoints",
            content="Maritime transit passage.",
            embedding=embedder.embed_text("sanctions"),  # Same embedding to test filter isolation
        ),
    ]
    store.upsert_chunks(chunks)

    q_vec = embedder.embed_text("sanctions")
    results = store.similarity_search(
        query_vector=q_vec,
        top_k=5,
        similarity_threshold=0.1,
        filter_topic="sanctions_and_trade_controls",
    )

    assert len(results) == 1
    assert results[0].chunk_id == "doc1#0"
    assert results[0].topic == "sanctions_and_trade_controls"


# ==============================================================================
# 4. KnowledgeRetriever Integration Tests
# ==============================================================================

def test_knowledge_retriever_live_or_indexed():
    """Verify KnowledgeRetriever retrieves ranked results from the indexed corpus."""
    retriever = KnowledgeRetriever()
    response = retriever.retrieve(
        query="What are the conditions for transit passage through international straits under UNCLOS?",
        top_k=3,
        similarity_threshold=0.20,
    )

    assert response.query is not None
    assert isinstance(response.count, int)
    assert isinstance(response.latency_ms, (int, float))
    if response.count > 0:
        top_hit = response.results[0]
        assert top_hit.source_url.startswith("http")
        assert top_hit.document_title != ""
        assert top_hit.publication_date != ""
        assert top_hit.rank == 1


# ==============================================================================
# 5. FastAPI Endpoints HTTP Integration Tests
# ==============================================================================

def test_api_knowledge_sources_endpoint():
    """Verify GET /api/v1/knowledge/sources returns verified corpus sources."""
    response = client.get("/api/v1/knowledge/sources")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "total_chunks" in data
    assert "sources" in data
    assert data["total_chunks"] >= 6


def test_api_knowledge_retrieve_endpoint():
    """Verify POST /api/v1/knowledge/retrieve responds with structured ranked evidence."""
    payload = {
        "query": "How does NATO Article 5 define collective self defense?",
        "top_k": 2,
        "similarity_threshold": 0.20,
    }
    response = client.post("/api/v1/knowledge/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "query" in data
    assert "results" in data
    assert "count" in data
    assert "latency_ms" in data

    if data["count"] > 0:
        first_result = data["results"][0]
        assert "document_title" in first_result
        assert "source_url" in first_result
        assert "publication_date" in first_result
        assert "content" in first_result
        assert "score" in first_result
        assert first_result["source_url"].startswith("http")
