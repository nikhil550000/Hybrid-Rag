import sys
import types

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
chromadb_stub.EphemeralClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

sentence_transformers_stub = types.ModuleType("sentence_transformers")
sentence_transformers_stub.CrossEncoder = object
sentence_transformers_stub.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers_stub)

from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from store.vector import RetrievalProvenance, RetrievedChunk


def _chunk(
    chunk_id: str,
    text: str = "chunk text",
    score: float = 1.0,
    method: str = "dense",
    provenance: RetrievalProvenance | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="paper.pdf",
        page=0,
        score=score,
        retrieval_method=method,
        provenance=provenance or RetrievalProvenance(),
    )


def test_rrf_preserves_dense_sparse_ranks_scores_and_sources():
    dense = [
        _chunk("shared", score=0.91, method="dense"),
        _chunk("dense-only", score=0.82, method="dense"),
    ]
    sparse = [
        _chunk("shared", score=3.5, method="sparse"),
        _chunk("sparse-only", score=1.2, method="sparse"),
    ]

    merged = reciprocal_rank_fusion(dense, sparse, k=60)

    assert [chunk.chunk_id for chunk in merged] == [
        "shared",
        "dense-only",
        "sparse-only",
    ]
    assert len([chunk for chunk in merged if chunk.chunk_id == "shared"]) == 1

    shared = next(chunk for chunk in merged if chunk.chunk_id == "shared")
    assert shared.retrieval_method == "hybrid"
    assert shared.score == shared.provenance.rrf_score
    assert shared.provenance.dense_rank == 1
    assert shared.provenance.dense_score == 0.91
    assert shared.provenance.sparse_rank == 1
    assert shared.provenance.sparse_score == 3.5
    assert shared.provenance.sources == ["dense", "sparse"]

    sparse_only = next(chunk for chunk in merged if chunk.chunk_id == "sparse-only")
    assert sparse_only.provenance.dense_rank is None
    assert sparse_only.provenance.sparse_rank == 2
    assert sparse_only.provenance.sources == ["sparse"]


class FixedScoreModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs):
        assert len(pairs) == len(self._scores)
        return self._scores


def _reranker(scores: list[float], top_n: int = 2) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker._model = FixedScoreModel(scores)
    reranker._top_n = top_n
    reranker._diversity_jaccard_threshold = 0.65
    reranker._diversity_min_shared_terms = 40
    return reranker


def test_reranker_preserves_provenance_and_filters_highly_redundant_chunks():
    shared_terms = " ".join(f"term{i}" for i in range(50))
    chunk_a = _chunk(
        "a",
        text=f"{shared_terms} alpha unique",
        score=0.04,
        method="hybrid",
        provenance=RetrievalProvenance(
            dense_rank=1,
            dense_score=0.91,
            rrf_score=0.04,
            sources=["dense"],
        ),
    )
    chunk_b = _chunk(
        "b",
        text=f"{shared_terms} beta unique",
        score=0.03,
        method="hybrid",
        provenance=RetrievalProvenance(
            sparse_rank=1,
            sparse_score=3.5,
            rrf_score=0.03,
            sources=["sparse"],
        ),
    )
    chunk_c = _chunk(
        "c",
        text=" ".join(f"other{i}" for i in range(50)),
        score=0.02,
        method="hybrid",
        provenance=RetrievalProvenance(
            dense_rank=2,
            dense_score=0.8,
            rrf_score=0.02,
            sources=["dense"],
        ),
    )

    reranked = _reranker([0.9, 0.8, 0.7], top_n=2).rerank(
        "query",
        [chunk_a, chunk_b, chunk_c],
    )

    assert [chunk.chunk_id for chunk in reranked] == ["a", "c"]
    assert reranked[0].retrieval_method == "reranked"
    assert reranked[0].score == 0.9
    assert reranked[0].provenance.rerank_score == 0.9
    assert reranked[0].provenance.dense_rank == 1
    assert reranked[0].provenance.rrf_score == 0.04
    assert reranked[1].provenance.rerank_score == 0.7
