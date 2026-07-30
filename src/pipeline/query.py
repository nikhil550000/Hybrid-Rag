"""
Query pipeline orchestrator.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from generation.citations import Citation
from generation.generator import Generator
from observability.tracer import TracerProtocol, NullTracer
from observability.metrics import MetricsCollector
from pipeline.memory import ConversationTurn, InMemoryConversationStore
from pipeline.rewriter import NoOpQueryRewriter, QueryRewriter, RewriteResult
from pipeline.router import QueryRoute, RuleBasedQueryRouter
from retrieval.dense import DenseRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse import SparseRetriever
from store.vector import RetrievedChunk
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
    route: str = QueryRoute.RAG_FACTUAL.value
    retrieval_query: str = ""
    query_rewritten: bool = False


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
        query_router: RuleBasedQueryRouter | None = None,
        conversation_store: InMemoryConversationStore | None = None,
        query_rewriter: QueryRewriter | None = None,
    ):
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._reranker = reranker
        self._generator = generator
        self._rrf_k = rrf_k
        self._tracer = tracer or NullTracer()
        self._metrics = metrics_collector
        self._query_router = query_router or RuleBasedQueryRouter()
        self._conversation_store = conversation_store
        self._query_rewriter = query_rewriter or NoOpQueryRewriter()

    def _retrieve_hybrid_candidates(
        self,
        query: str,
        timings_ms: dict[str, float],
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """Run independent dense and sparse retrieval stages concurrently."""
        dense_timings_ms: dict[str, float] = {}
        sparse_timings_ms: dict[str, float] = {}

        def retrieve_dense() -> list[RetrievedChunk]:
            return self._dense.retrieve(query, timings_ms=dense_timings_ms)

        def retrieve_sparse() -> list[RetrievedChunk]:
            stage_start = time.perf_counter()
            results = self._sparse.retrieve(query)
            sparse_timings_ms["sparse_retrieval"] = _elapsed_ms(stage_start)
            return results

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-retrieval") as executor:
            dense_future = executor.submit(retrieve_dense)
            sparse_future = executor.submit(retrieve_sparse)

            dense_results = dense_future.result()
            sparse_results = sparse_future.result()

        timings_ms.update(dense_timings_ms)
        timings_ms.update(sparse_timings_ms)

        return dense_results, sparse_results

    def run(self, query: str, conversation_id: str | None = None) -> QueryResult:
        """
        Full query pipeline:
        1. Start trace
        2. Dense + sparse retrieval concurrently
        3. RRF fusion (merge + deduplicate)
        4. Cross-encoder reranking (precision)
        5. Log retrieval to tracer
        6. LLM generation with citation enforcement
        7. Log generation to tracer
        8. End trace + record metrics

        Args:
            query: User's natural language question
            conversation_id: Optional process-local conversation memory key
        Returns:
            QueryResult with answer, citations, latency, cost
        """
        start = time.perf_counter()
        timings_ms: dict[str, float] = {}
        retrieval_query = query
        rewrite_result = RewriteResult(query=query)

        # Step 1: Start trace
        ctx = self._tracer.start_trace(query)

        # Step 2: Route obvious non-RAG queries away from expensive retrieval.
        stage_start = time.perf_counter()
        route_decision = self._query_router.route(query)
        timings_ms["query_routing"] = _elapsed_ms(stage_start)

        if not route_decision.should_run_rag:
            timings_ms["total_latency"] = _elapsed_ms(start)
            rounded_timings_ms = _rounded_timings(timings_ms)

            self._tracer.end_trace(
                ctx,
                refused=route_decision.refused,
                timings_ms=rounded_timings_ms,
                metadata={
                    "query_route": route_decision.route.value,
                    "route_reason": route_decision.reason,
                    "short_circuited": True,
                },
            )

            result = QueryResult(
                query=query,
                answer=route_decision.answer,
                refused=route_decision.refused,
                latency_ms=rounded_timings_ms["total_latency"],
                cost_usd=0.0,
                timings_ms=rounded_timings_ms,
                route=route_decision.route.value,
                retrieval_query=query,
                query_rewritten=False,
            )

            if self._metrics:
                self._metrics.record_request(
                    latency_ms=result.latency_ms,
                    cost_usd=result.cost_usd,
                    has_citations=False,
                    failed=result.refused,
                    timings_ms=result.timings_ms,
                    route=result.route,
                )

            logger.info(
                f"Query routed without RAG | route={result.route} | "
                f"reason={route_decision.reason} | latency={result.latency_ms}ms"
            )
            return result

        if conversation_id and route_decision.route == QueryRoute.FOLLOW_UP:
            history = (
                self._conversation_store.get_recent(conversation_id)
                if self._conversation_store is not None
                else []
            )
            rewrite_result = self._query_rewriter.rewrite(
                query=query,
                history=history,
                timings_ms=timings_ms,
            )
            retrieval_query = rewrite_result.query
            ctx.query = retrieval_query
        elif route_decision.route == QueryRoute.FOLLOW_UP:
            timings_ms["query_rewriting"] = 0.0

        # Step 2 & 3: Parallel retrieval
        dense_results, sparse_results = self._retrieve_hybrid_candidates(
            retrieval_query,
            timings_ms,
        )

        # Step 4: RRF fusion
        stage_start = time.perf_counter()
        merged = reciprocal_rank_fusion(dense_results, sparse_results, k=self._rrf_k)
        timings_ms["rrf"] = _elapsed_ms(stage_start)

        # Step 5: Rerank
        stage_start = time.perf_counter()
        reranked = self._reranker.rerank(retrieval_query, merged)
        timings_ms["reranking"] = _elapsed_ms(stage_start)

        # Step 6: Log retrieval
        self._tracer.log_retrieval(
            ctx,
            pre_rerank=merged,
            post_rerank=reranked,
            timings_ms=_rounded_timings(timings_ms),
        )

        # Step 7: Generate
        gen_response = self._generator.generate(retrieval_query, reranked, timings_ms=timings_ms)

        timings_ms["total_latency"] = _elapsed_ms(start)
        rounded_timings_ms = _rounded_timings(timings_ms)
        elapsed_ms = rounded_timings_ms["total_latency"]

        # Step 8: Log generation (cost + tokens flow from GeneratorResponse)
        self._tracer.log_generation(
            ctx=ctx,
            prompt=f"[query: {retrieval_query}]",
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
            metadata={
                "query_route": route_decision.route.value,
                "route_reason": route_decision.reason,
                "short_circuited": False,
                "original_query": query,
                "retrieval_query": retrieval_query,
                "query_rewritten": rewrite_result.rewritten,
                "rewrite_error": rewrite_result.error,
            },
        )

        total_cost_usd = gen_response.cost_usd + rewrite_result.cost_usd

        result = QueryResult(
            query=query,
            answer=gen_response.answer,
            citations=gen_response.citations,
            refused=gen_response.refused,
            latency_ms=elapsed_ms,
            cost_usd=total_cost_usd,
            timings_ms=rounded_timings_ms,
            route=route_decision.route.value,
            retrieval_query=retrieval_query,
            query_rewritten=rewrite_result.rewritten,
        )

        if conversation_id and self._conversation_store is not None and not result.refused:
            self._conversation_store.append(
                conversation_id,
                ConversationTurn(
                    user_query=query,
                    retrieval_query=retrieval_query,
                    answer=result.answer,
                    citation_chunk_ids=[citation.chunk_id for citation in result.citations],
                ),
            )

        # Record metrics
        if self._metrics:
            self._metrics.record_request(
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                has_citations=len(result.citations) > 0,
                failed=result.refused,
                timings_ms=result.timings_ms,
                route=result.route,
            )

        logger.info(
            f"Query completed in {result.latency_ms}ms | "
            f"route={result.route} | citations={len(result.citations)} | "
            f"refused={result.refused} | "
            f"query_rewritten={result.query_rewritten} | "
            f"timings_ms={result.timings_ms}"
        )
        return result
