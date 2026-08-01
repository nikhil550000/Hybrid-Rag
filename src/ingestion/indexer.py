"""BM25 index builder and serializer.

Implements: HLD 3.4 (build side)
Satisfies: FR-10, FR-11
"""
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from ingestion.chunker import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)


class BM25Indexer:
    """Builds a BM25 index over all chunks and serializes it to disk."""

    def build(self, chunks: list[Chunk]) -> tuple[BM25Okapi | None, list[str]]:
        """
        Tokenize chunk texts and build BM25Okapi index.

        Returns:
            (bm25_index, chunk_ids) — chunk_ids[i] matches the i-th document in the index
        """
        chunk_ids = [chunk.id for chunk in chunks]
        tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]

        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        logger.info(f"Built BM25 index over {len(chunk_ids)} chunks")
        return bm25, chunk_ids

    def save(
        self, bm25: BM25Okapi | None, chunk_ids: list[str], chunks: list[Chunk], path: Path
    ) -> None:
        """Serialize index + chunk data to disk using pickle.

        Stores chunk texts and metadata alongside the index so BM25Store
        can reconstruct RetrievedChunk objects at query time without
        depending on VectorStore.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "bm25": bm25,
            "chunk_ids": chunk_ids,
            "chunk_texts": [c.text for c in chunks],
            "chunk_metadata": [{"source": c.source, "page": c.page} for c in chunks],
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved BM25 index to {path}")
