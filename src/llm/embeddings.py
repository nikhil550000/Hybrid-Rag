"""Embedding client protocol and implementations.

Implements: HLD 3.3, 3.5 (embedding)
Satisfies: FR-04, FR-09
"""
from typing import Protocol

from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingClient(Protocol):
    """All embedding providers implement this interface."""

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Local embedding via sentence-transformers. No API key needed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. More efficient than calling embed() in a loop."""
        embeddings = self._model.encode(texts, show_progress_bar=len(texts) > 50)
        return [e.tolist() for e in embeddings]
