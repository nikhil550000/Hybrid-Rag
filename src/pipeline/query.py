"""Query pipeline orchestrator.

Implements: HLD 3.5 → 3.11
Satisfies: FR-08 → FR-19
"""
import time
from dataclasses import dataclass, field

from generation.citations import Citation, CitationValidator
from generation.generator import Generator, GeneratorResponse
from generation.prompt import PromptBuilder
from llm.client import LLMClient
from retrieval.dense import DenseRetriever
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse import SparseRetriever
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """End-to-end result for a user query."""
    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    latency_ms: float = 0.0


class QueryPipeline:
    """Orchestrates: retrieve → fuse → rerank → generate."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        reranker: CrossEncoderReranker,
        generator: Generator,
        rrf_k: int = 60,
    ):
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._reranker = reranker
        self._generator = generator
        self._rrf_k = rrf_k

    def run(self, query: str) -> QueryResult:
        """
        Full query pipeline:
        1. Dense retrieval (vector similarity)
        2. Sparse retrieval (BM25 keywords)
        3. RRF fusion (merge + deduplicate)
        4. Cross-encoder reranking (precision)
        5. LLM generation with citation enforcement

        Args:
            query: User's natural language question
        Returns:
            QueryResult with answer, citations, latency
        """
        start = time.perf_counter()

        # Step 1 & 2: Parallel retrieval
        dense_results = self._dense.retrieve(query)
        sparse_results = self._sparse.retrieve(query)

        # Step 3: RRF fusion
        merged = reciprocal_rank_fusion(dense_results, sparse_results, k=self._rrf_k)

        # Step 4: Rerank
        reranked = self._reranker.rerank(query, merged)

        # Step 5: Generate
        gen_response = self._generator.generate(query, reranked)

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = QueryResult(
            query=query,
            answer=gen_response.answer,
            citations=gen_response.citations,
            refused=gen_response.refused,
            latency_ms=round(elapsed_ms, 1),
        )

        logger.info(
            f"Query completed in {result.latency_ms}ms | "
            f"citations={len(result.citations)} | refused={result.refused}"
        )
        return result
