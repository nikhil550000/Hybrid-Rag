"""
Reciprocal Rank Fusion for hybrid retrieval.
"""
from store.vector import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    results_a: list[RetrievedChunk],
    results_b: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    Formula: score(chunk) = sum(1 / (k + rank_i)) across all lists.
    Chunks appearing in both lists are naturally boosted.
    Output is sorted descending by RRF score.
    retrieval_method is set to "hybrid" on returned chunks.

    Args:
        results_a: Ranked list from dense retriever
        results_b: Ranked list from sparse retriever
        k: RRF constant (default 60, standard empirical value)
    Returns:
        Deduplicated, RRF-ranked list of RetrievedChunks
    """
    # Build score dict: chunk_id → cumulative RRF score
    scores: dict[str, float] = {}
    # Keep first occurrence of each chunk for metadata
    chunk_map: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(results_a, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        if chunk.chunk_id not in chunk_map:
            chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(results_b, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        if chunk.chunk_id not in chunk_map:
            chunk_map[chunk.chunk_id] = chunk

    # Sort by RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    # Build result list with updated scores and retrieval_method
    merged: list[RetrievedChunk] = []
    for chunk_id in sorted_ids:
        original = chunk_map[chunk_id]
        merged.append(RetrievedChunk(
            chunk_id=original.chunk_id,
            text=original.text,
            source=original.source,
            page=original.page,
            score=scores[chunk_id],
            retrieval_method="hybrid",
        ))

    logger.info(
        f"RRF fusion: {len(results_a)} dense + {len(results_b)} sparse → "
        f"{len(merged)} merged candidates"
    )
    return merged
