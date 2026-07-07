"""Cross-encoder re-ranking of candidate chunks.

Implements: HLD 3.8
Satisfies: FR-12
"""
from sentence_transformers import CrossEncoder

from store.vector import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs with a cross-encoder and returns top_n."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ):
        self._model = CrossEncoder(model_name)
        self._top_n = top_n
        logger.info(f"Loaded reranker model: {model_name}")

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Score all (query, chunk.text) pairs, sort by score, return top_n.
        retrieval_method is set to "reranked" on returned chunks.

        Args:
            query: Original user query
            chunks: Merged hybrid candidate list (up to 20)
        Returns:
            Top_n chunks by cross-encoder score
        """
        if not chunks:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self._model.predict(pairs)

        # Pair chunks with their cross-encoder scores
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Take top_n and update scores + retrieval_method
        reranked: list[RetrievedChunk] = []
        for chunk, score in scored_chunks[: self._top_n]:
            reranked.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                score=float(score),
                retrieval_method="reranked",
            ))

        logger.info(
            f"Reranked {len(chunks)} candidates → top {len(reranked)}"
        )
        return reranked
