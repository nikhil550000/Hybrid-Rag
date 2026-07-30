import sys
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

from pipeline.memory import InMemoryConversationStore
from pipeline.query import QueryPipeline
from pipeline.rewriter import RewriteResult
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


class RecordingDenseRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str, timings_ms: dict[str, float] | None = None):
        self.queries.append(query)
        if timings_ms is not None:
            timings_ms["query_embedding"] = 1.0
            timings_ms["dense_retrieval"] = 1.0
        return [_chunk("dense-a", "dense", 0.9)]


class RecordingSparseRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str):
        self.queries.append(query)
        return [_chunk("sparse-a", "sparse", 3.0)]


class RecordingReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        self.queries.append(query)
        return chunks


class RecordingGenerator:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        timings_ms: dict[str, float] | None = None,
    ) -> GeneratorResponse:
        self.queries.append(query)
        if timings_ms is not None:
            timings_ms["prompt_building"] = 1.0
            timings_ms["llm_call"] = 1.0
            timings_ms["citation_validation"] = 1.0
        return GeneratorResponse(answer=f"answer for {query}", cost_usd=0.01)


class RecordingRewriter:
    def __init__(self, rewritten_query: str) -> None:
        self.rewritten_query = rewritten_query
        self.calls: list[tuple[str, int]] = []

    def rewrite(self, query: str, history: list, timings_ms: dict[str, float] | None = None):
        self.calls.append((query, len(history)))
        if timings_ms is not None:
            timings_ms["query_rewriting"] = 2.0
        return RewriteResult(query=self.rewritten_query, rewritten=True, cost_usd=0.02)


def _pipeline(rewriter: RecordingRewriter):
    dense = RecordingDenseRetriever()
    sparse = RecordingSparseRetriever()
    reranker = RecordingReranker()
    generator = RecordingGenerator()
    pipeline = QueryPipeline(
        dense_retriever=dense,
        sparse_retriever=sparse,
        reranker=reranker,
        generator=generator,
        conversation_store=InMemoryConversationStore(),
        query_rewriter=rewriter,
    )
    return pipeline, dense, sparse, reranker, generator


def test_follow_up_with_history_uses_rewritten_query_for_rag_stages():
    rewriter = RecordingRewriter(
        "What are the limitations of the FlashAttention method?"
    )
    pipeline, dense, sparse, reranker, generator = _pipeline(rewriter)

    first = pipeline.run(
        "What method does the FlashAttention paper propose?",
        conversation_id="default:conv-1",
    )
    follow_up = pipeline.run(
        "What about its limitations?",
        conversation_id="default:conv-1",
    )

    assert first.query_rewritten is False
    assert follow_up.query_rewritten is True
    assert follow_up.retrieval_query == rewriter.rewritten_query
    assert rewriter.calls == [("What about its limitations?", 1)]
    assert dense.queries[-1] == rewriter.rewritten_query
    assert sparse.queries[-1] == rewriter.rewritten_query
    assert reranker.queries[-1] == rewriter.rewritten_query
    assert generator.queries[-1] == rewriter.rewritten_query
    assert follow_up.cost_usd == 0.03
    assert follow_up.timings_ms["query_rewriting"] == 2.0


def test_follow_up_without_conversation_id_does_not_rewrite():
    rewriter = RecordingRewriter("rewritten query should not be used")
    pipeline, dense, sparse, reranker, generator = _pipeline(rewriter)

    result = pipeline.run("What about its limitations?")

    assert result.query_rewritten is False
    assert result.retrieval_query == "What about its limitations?"
    assert rewriter.calls == []
    assert dense.queries == ["What about its limitations?"]
    assert sparse.queries == ["What about its limitations?"]
    assert reranker.queries == ["What about its limitations?"]
    assert generator.queries == ["What about its limitations?"]
    assert result.timings_ms["query_rewriting"] == 0.0
