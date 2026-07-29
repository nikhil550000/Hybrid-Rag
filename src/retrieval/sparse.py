"""
Sparse keyword retrieval via BM25.
"""
from store.bm25 import BM25Store
from store.vector import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """Retrieves top-k chunks by BM25 keyword score."""

    def __init__(self, bm25_store: BM25Store, top_k: int):
        self._bm25_store = bm25_store
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Query BM25 index → return top_k RetrievedChunks."""
        results = self._bm25_store.query(query, self._top_k)
        logger.info(f"Sparse retrieval: {len(results)} chunks for query '{query[:50]}...'")
        return results
