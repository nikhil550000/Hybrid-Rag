"""
Dense vector retrieval via ChromaDB.
"""
import time

from llm.embeddings import EmbeddingClient
from store.vector import VectorStore, RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """Embeds a query and retrieves top-k chunks by vector similarity from ChromaDB."""

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingClient, top_k: int):
        self._vector_store = vector_store
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        timings_ms: dict[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed query → query ChromaDB → return top_k RetrievedChunks."""
        stage_start = time.perf_counter()
        query_embedding = self._embedder.embed(query)
        if timings_ms is not None:
            timings_ms["query_embedding"] = (time.perf_counter() - stage_start) * 1000

        stage_start = time.perf_counter()
        results = self._vector_store.query(query_embedding, self._top_k)
        if timings_ms is not None:
            timings_ms["dense_retrieval"] = (time.perf_counter() - stage_start) * 1000

        logger.info(f"Dense retrieval: {len(results)} chunks for query '{query[:50]}...'")
        return results
