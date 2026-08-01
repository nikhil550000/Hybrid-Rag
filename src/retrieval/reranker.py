"""
Cross-encoder re-ranking of candidate chunks.
"""
import re

from sentence_transformers import CrossEncoder
from store.vector import RetrievedChunk, clone_retrieval_provenance
from utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _jaccard_overlap(text_a: str, text_b: str) -> tuple[float, int]:
    tokens_a = _token_set(text_a)
    tokens_b = _token_set(text_b)
    if not tokens_a or not tokens_b:
        return 0.0, 0
    shared = tokens_a & tokens_b
    return len(shared) / len(tokens_a | tokens_b), len(shared)


class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs with a cross-encoder and returns top_n."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        diversity_jaccard_threshold: float = 0.65,
        diversity_min_shared_terms: int = 40,
    ):
        self._model = CrossEncoder(model_name)
        self._top_n = top_n
        self._diversity_jaccard_threshold = diversity_jaccard_threshold
        self._diversity_min_shared_terms = diversity_min_shared_terms
        logger.info(f"Loaded reranker model: {model_name}")

    def _is_redundant(
        self,
        candidate: RetrievedChunk,
        selected: list[RetrievedChunk],
    ) -> bool:
        """Return True for candidates that are highly overlapping with selected chunks."""
        for chunk in selected:
            jaccard, shared_terms = _jaccard_overlap(candidate.text, chunk.text)
            if (
                jaccard >= self._diversity_jaccard_threshold
                and shared_terms >= self._diversity_min_shared_terms
            ):
                return True
        return False

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Score all (query, chunk.text) pairs, sort by score, return top_n.
        retrieval_method is set to "reranked" on returned chunks.
        Prior retrieval provenance is preserved and rerank_score is added.
        A conservative lexical diversity pass avoids selecting near-duplicate
        chunks when enough non-redundant candidates are available.

        Args:
            query: Original user query
            chunks: Merged hybrid candidate list (up to 20)
        Returns:
            Top_n chunks by cross-encoder score after conservative de-duplication
        """
        if not chunks:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self._model.predict(pairs)

        # Pair chunks with their cross-encoder scores
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        selected: list[RetrievedChunk] = []
        skipped_redundant: list[RetrievedChunk] = []

        for chunk, score in scored_chunks:
            provenance = clone_retrieval_provenance(chunk.provenance)
            provenance.rerank_score = float(score)
            reranked_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                score=float(score),
                retrieval_method="reranked",
                provenance=provenance,
            )

            if self._is_redundant(reranked_chunk, selected):
                skipped_redundant.append(reranked_chunk)
                continue

            selected.append(reranked_chunk)
            if len(selected) >= self._top_n:
                break

        if len(selected) < self._top_n:
            needed = self._top_n - len(selected)
            selected.extend(skipped_redundant[:needed])

        logger.info(
            f"Reranked {len(chunks)} candidates → top {len(selected)} "
            f"(diversity-skipped={len(skipped_redundant)})"
        )
        return selected
