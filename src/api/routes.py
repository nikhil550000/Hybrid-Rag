"""FastAPI server — thin wrapper around QueryPipeline.

Implements: HLD 3.12
Satisfies: FR-WEB-01 → FR-WEB-04

Design: Zero pipeline logic in the API layer. The server constructs a
QueryPipeline from config at startup and calls pipeline.run(query) per
request. CLI and API exercise identical code paths.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import QueryRequest, QueryResponse, CitationSchema, ErrorResponse, HealthResponse
from config.settings import load_settings
from generation.citations import CitationValidator
from generation.generator import Generator
from generation.prompt import PromptBuilder
from llm.client import get_llm_client
from llm.embeddings import SentenceTransformerEmbedder
from observability.tracer import LangfuseTracer, NullTracer
from observability.metrics import MetricsCollector
from pipeline.query import QueryPipeline
from retrieval.dense import DenseRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse import SparseRetriever
from store.bm25 import BM25Store
from store.vector import VectorStore
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

# Module-level references set during lifespan
_pipeline: QueryPipeline | None = None
_metrics: MetricsCollector | None = None
_chunk_count: int = 0


def _build_tracer(backend: str):
    """Construct tracer based on settings."""
    if backend == "langfuse":
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if not public_key or not secret_key:
            logger.warning("Langfuse keys not set — using NullTracer")
            return NullTracer()
        return LangfuseTracer(public_key=public_key, secret_key=secret_key, host=host)
    return NullTracer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline once at startup, tear down on shutdown."""
    global _pipeline, _metrics, _chunk_count

    settings = load_settings()
    setup_logging(log_level=settings.log_level)
    logger.info("Starting RAG Scholar API server...")

    # Observability
    tracer = _build_tracer(settings.observability_backend)
    _metrics = MetricsCollector()

    # Retrieval components
    embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
    vector_store = VectorStore(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path,
    )
    _chunk_count = vector_store._collection.count()

    bm25_store = BM25Store()
    bm25_store.load(Path(settings.bm25_index_path))

    dense_retriever = DenseRetriever(vector_store, embedder, top_k=settings.retrieval_top_k)
    sparse_retriever = SparseRetriever(bm25_store, top_k=settings.retrieval_top_k)
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        top_n=settings.reranked_top_n,
    )

    # Generation components
    llm_client = get_llm_client(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    prompt_builder = PromptBuilder(prompt_version=settings.prompt_version)
    citation_validator = CitationValidator()
    generator = Generator(llm_client, prompt_builder, citation_validator)

    # Wire pipeline
    _pipeline = QueryPipeline(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
        generator=generator,
        rrf_k=settings.rrf_k,
        tracer=tracer,
        metrics_collector=_metrics,
    )

    logger.info(f"Pipeline ready — {_chunk_count} chunks indexed")
    yield
    logger.info("Shutting down RAG Scholar API server")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Scholar API",
    description="Production-grade RAG over ML research papers",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for monitoring."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        chunks_indexed=_chunk_count,
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Execute a RAG query — thin wrapper around Pipeline.run().

    Satisfies: FR-WEB-01 (POST /query), FR-WEB-02 (thin wrapper),
               FR-WEB-03 (structured JSON), FR-WEB-04 (error handling)
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = _pipeline.run(request.query)

        return QueryResponse(
            answer=result.answer,
            citations=[
                CitationSchema(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    passage=c.passage,
                )
                for c in result.citations
            ],
            refused=result.refused,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="PIPELINE_ERROR",
                message=str(e),
            ).model_dump(),
        )


@app.get("/metrics")
async def metrics_endpoint():
    """Return session-level aggregated metrics."""
    if _metrics is None:
        return {"error": "Metrics not available"}

    summary = _metrics.get_summary()
    return {
        "total_requests": summary.total_requests,
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "avg_latency_ms": summary.avg_latency_ms,
        "total_cost_usd": summary.total_cost_usd,
        "avg_cost_usd": summary.avg_cost_usd,
        "citation_coverage_pct": summary.citation_coverage_pct,
        "failure_rate_pct": summary.failure_rate_pct,
    }
