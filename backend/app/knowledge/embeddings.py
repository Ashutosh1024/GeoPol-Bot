"""Embedding generation service for GeoPol Bot knowledge pipeline.

Uses the official OpenAI Python SDK with the text-embedding-3-small model.
Includes local in-memory caching and deterministic mock support for offline testing.
"""

import hashlib
import logging
from typing import List, Optional
import numpy as np
from openai import OpenAI

from backend.app.core.config import Settings, get_settings

logger = logging.getLogger("geopol_bot.embeddings")

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


class EmbeddingService:
    """Service responsible for generating normalized vector embeddings."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[OpenAI] = None,
        use_mock: bool = False,
    ):
        self.settings = settings or get_settings()
        self._client = client
        self.use_mock = use_mock
        self._cache: dict[str, List[float]] = {}

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of the OpenAI client."""
        if self._client is None:
            if not self.settings.is_openai_configured:
                raise ValueError("OpenAI API key is required for embedding generation.")
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout_seconds,
            )
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text string."""
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embedding vectors for a batch of text strings."""
        if not texts:
            return []

        # Check cache
        embeddings_to_return: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, text in enumerate(texts):
            clean_text = text.strip()
            if clean_text in self._cache:
                embeddings_to_return[idx] = self._cache[clean_text]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(clean_text)

        if not uncached_texts:
            return [e for e in embeddings_to_return if e is not None]

        # If mock mode or unconfigured, generate deterministic synthetic embeddings
        if self.use_mock or not self.settings.is_openai_configured:
            logger.info("Generating deterministic mock embeddings (mock_mode=%s)", self.use_mock)
            for idx, text in zip(uncached_indices, uncached_texts):
                mock_vector = self._generate_deterministic_mock(text)
                self._cache[text] = mock_vector
                embeddings_to_return[idx] = mock_vector
            return [e for e in embeddings_to_return if e is not None]

        # Call OpenAI Embeddings API
        try:
            response = self.client.embeddings.create(
                model=DEFAULT_EMBEDDING_MODEL,
                input=uncached_texts,
            )
            for uncached_idx, item in zip(uncached_indices, response.data):
                vector = item.embedding
                embeddings_to_return[uncached_idx] = vector
                self._cache[texts[uncached_idx].strip()] = vector

            return [e for e in embeddings_to_return if e is not None]

        except Exception as exc:
            logger.error("Error generating embeddings with OpenAI: %s. Falling back to deterministic mock.", exc)
            for idx, text in zip(uncached_indices, uncached_texts):
                fallback_vector = self._generate_deterministic_mock(text)
                embeddings_to_return[idx] = fallback_vector
            return [e for e in embeddings_to_return if e is not None]

    def _generate_deterministic_mock(self, text: str) -> List[float]:
        """Generate normalized deterministic 1536-dim vector based on text hash for offline testing."""
        hasher = hashlib.sha256(text.encode("utf-8"))
        seed = int.from_bytes(hasher.digest()[:4], byteorder="big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(EMBEDDING_DIMENSIONS)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
