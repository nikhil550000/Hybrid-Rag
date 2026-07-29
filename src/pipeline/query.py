"""
Query pipeline orchestrator.
"""
import time
from dataclasses import dataclass, field

from generation.citations import Citation
from generation.generator import Generator
from observability.tracer import TracerProtocol, NullTracer
from observability.metrics import MetricsCollector
from retrieval.dense import DenseRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse import SparseRetriever
from utils.logger import get_logger

logger = get_logger(__name__)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _rounded_timings(timings_ms: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 1) for key, value in timings_ms.items()}


@dataclass
class QueryResult:
    """End-to-end result for a user query."""
    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)


class QueryPipeline:
    """Orchestrates: retrieve → fuse → rerank → generate → trace."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        reranker: CrossEncoderReranker,
        generator: Generator,
        rrf_k: int = 60,
        tracer: TracerProtocol | None = None,
        metrics_collector: MetricsCollector | None = None,
    ):
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._reranker = reranker
        self._generator = generator
        self._rrf_k = rrf_k
        self._tracer = tracer or NullTracer()
        self._metrics = metrics_collector

    def run(self, query: str) -> QueryResult:
        """
        Full query pipeline:
        1. Start trace
        2. Dense retrieval (vector similarity)
        3. Sparse retrieval (BM25 keywords)
        4. RRF fusion (merge + deduplicate)
        5. Cross-encoder reranking (precision)
        6. Log retrieval to tracer
        7. LLM generation with citation enforcement
        8. Log generation to tracer
        9. End trace + record metrics

        Args:
            query: User's natural language question
        Returns:
            QueryResult with answer, citations, latency, cost
        """
        start = time.perf_counter()
        timings_ms: dict[str, float] = {}

        # Step 1: Start trace
        ctx = self._tracer.start_trace(query)

        # Step 2 & 3: Parallel retrieval
        dense_results = self._dense.retrieve(query, timings_ms=timings_ms)

        stage_start = time.perf_counter()
        sparse_results = self._sparse.retrieve(query)
        timings_ms["sparse_retrieval"] = _elapsed_ms(stage_start)

        # Step 4: RRF fusion
        stage_start = time.perf_counter()
        merged = reciprocal_rank_fusion(dense_results, sparse_results, k=self._rrf_k)
        timings_ms["rrf"] = _elapsed_ms(stage_start)

        # Step 5: Rerank
        stage_start = time.perf_counter()
        reranked = self._reranker.rerank(query, merged)
        timings_ms["reranking"] = _elapsed_ms(stage_start)

        # Step 6: Log retrieval
        self._tracer.log_retrieval(
            ctx,
            pre_rerank=merged,
            post_rerank=reranked,
            timings_ms=_rounded_timings(timings_ms),
        )

        # Step 7: Generate
        gen_response = self._generator.generate(query, reranked, timings_ms=timings_ms)

        timings_ms["total_latency"] = _elapsed_ms(start)
        rounded_timings_ms = _rounded_timings(timings_ms)
        elapsed_ms = rounded_timings_ms["total_latency"]

        # Step 8: Log generation (cost + tokens flow from GeneratorResponse)
        self._tracer.log_generation(
            ctx=ctx,
            prompt=f"[query: {query}]",
            response=gen_response.answer[:500],
            tokens_in=gen_response.tokens_input,
            tokens_out=gen_response.tokens_output,
            latency_ms=elapsed_ms,
            cost_usd=gen_response.cost_usd,
            timings_ms=rounded_timings_ms,
        )

        # Step 9: End trace
        self._tracer.end_trace(
            ctx,
            refused=gen_response.refused,
            timings_ms=rounded_timings_ms,
        )

        result = QueryResult(
            query=query,
            answer=gen_response.answer,
            citations=gen_response.citations,
            refused=gen_response.refused,
            latency_ms=elapsed_ms,
            cost_usd=gen_response.cost_usd,
            timings_ms=rounded_timings_ms,
        )

        # Record metrics
        if self._metrics:
            self._metrics.record_request(
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                has_citations=len(result.citations) > 0,
                failed=result.refused,
                timings_ms=result.timings_ms,
            )

        logger.info(
            f"Query completed in {result.latency_ms}ms | "
            f"citations={len(result.citations)} | refused={result.refused} | "
            f"timings_ms={result.timings_ms}"
        )
        return result
