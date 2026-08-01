import sys
import time
import types
from dataclasses import dataclass, field

fitz_stub = types.ModuleType("fitz")
sys.modules.setdefault("fitz", fitz_stub)

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
chromadb_stub.EphemeralClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

sentence_transformers_stub = types.ModuleType("sentence_transformers")
sentence_transformers_stub.SentenceTransformer = object
sentence_transformers_stub.CrossEncoder = object
sys.modules.setdefault("sentence_transformers", sentence_transformers_stub)

rank_bm25_stub = types.ModuleType("rank_bm25")
rank_bm25_stub.BM25Okapi = object
sys.modules.setdefault("rank_bm25", rank_bm25_stub)

from pipeline.query import QueryPipeline
from store.vector import RetrievedChunk


@dataclass
class GeneratorResponse:
    answer: str
    citations: list = field(default_factory=list)
    refused: bool = False
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0


def _chunk(chunk_id: str, method: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        source="paper.pdf",
        page=0,
        score=score,
        retrieval_method=method,
    )


class SlowDenseRetriever:
    def __init__(self) -> None:
        self.results = [_chunk("dense-a", "dense", 0.9)]

    def retrieve(self, query: str, timings_ms: dict[str, float] | None = None):
        time.sleep(0.2)
        if timings_ms is not None:
            timings_ms["query_embedding"] = 25.0
            timings_ms["dense_retrieval"] = 175.0
        return self.results


class SlowSparseRetriever:
    def __init__(self) -> None:
        self.results = [_chunk("sparse-a", "sparse", 3.0)]

    def retrieve(self, query: str):
        time.sleep(0.2)
        return self.results


class RecordingReranker:
    def __init__(self) -> None:
        self.seen_chunk_ids: list[str] = []

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        self.seen_chunk_ids = [chunk.chunk_id for chunk in chunks]
        return chunks


class StubGenerator:
    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        timings_ms: dict[str, float] | None = None,
    ) -> GeneratorResponse:
        if timings_ms is not None:
            timings_ms["prompt_building"] = 1.0
            timings_ms["llm_call"] = 1.0
            timings_ms["citation_validation"] = 1.0
        return GeneratorResponse(answer="answer")


def test_dense_and_sparse_retrieval_run_in_parallel_and_keep_rrf_inputs_ordered():
    dense = SlowDenseRetriever()
    sparse = SlowSparseRetriever()
    reranker = RecordingReranker()
    pipeline = QueryPipeline(
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
        generator=StubGenerator(),
    )

    start = time.perf_counter()
    result = pipeline.run("What does the paper say about retrieval?")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.35
    assert reranker.seen_chunk_ids == ["dense-a", "sparse-a"]
    assert [chunk.chunk_id for chunk in result.pre_rerank_chunks] == ["dense-a", "sparse-a"]
    assert [chunk.chunk_id for chunk in result.post_rerank_chunks] == ["dense-a", "sparse-a"]
    assert result.timings_ms["query_embedding"] == 25.0
    assert result.timings_ms["dense_retrieval"] == 175.0
    assert result.timings_ms["sparse_retrieval"] >= 190.0
