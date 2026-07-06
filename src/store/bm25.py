"""BM25 index loader and query-time retrieval.

Implements: HLD 3.4 (read side)
Satisfies: FR-10
"""
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from store.vector import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class BM25Store:
    """Loads the serialized BM25 index and handles query-time retrieval."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadata: list[dict] = []

    def load(self, path: Path) -> None:
        """
        Load BM25 index + chunk data from disk into memory.

        Raises:
            FileNotFoundError: If index doesn't exist (user needs to run ingest first)
        """
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run: python scripts/ingest.py"
            )

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        self._chunk_texts = data["chunk_texts"]
        self._chunk_metadata = data["chunk_metadata"]

        logger.info(f"Loaded BM25 index from {path} ({len(self._chunk_ids)} chunks)")

    def query(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        """
        Tokenize query, score all chunks, return top_k.
        retrieval_method is set to "sparse" on returned chunks.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "BM25 index not loaded. Call load() first or run: python scripts/ingest.py"
            )

        tokenized_query = query_text.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top_k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        retrieved: list[RetrievedChunk] = []
        for idx in top_indices:
            metadata = self._chunk_metadata[idx]
            retrieved.append(RetrievedChunk(
                chunk_id=self._chunk_ids[idx],
                text=self._chunk_texts[idx],
                source=metadata["source"],
                page=metadata["page"],
                score=float(scores[idx]),
                retrieval_method="sparse",
            ))

        return retrieved
