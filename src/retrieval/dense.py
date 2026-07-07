"""Dense vector retrieval via ChromaDB.

Implements: HLD 3.5
Satisfies: FR-09
"""
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

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Embed query → query ChromaDB → return top_k RetrievedChunks."""
        query_embedding = self._embedder.embed(query)
        results = self._vector_store.query(query_embedding, self._top_k)
        logger.info(f"Dense retrieval: {len(results)} chunks for query '{query[:50]}...'")
        return results
