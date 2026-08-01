"""FastAPI server — thin wrapper around QueryPipeline.

Implements: HLD 3.12
Satisfies: FR-WEB-01 → FR-WEB-04

Design: Zero pipeline logic in the API layer. The server constructs a
QueryPipeline from config at startup and calls pipeline.run(query) per
request. CLI and API exercise identical code paths.
"""
import os
import re
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uuid
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import QueryRequest, QueryResponse, CitationSchema, ErrorResponse, HealthResponse, UploadResponse
from config.settings import load_settings
from generation.citations import CitationValidator
from generation.generator import Generator
from generation.prompt import PromptBuilder
from llm.client import get_llm_client
from llm.embeddings import SentenceTransformerEmbedder
from observability.tracer import LangfuseTracer, NullTracer, TracerProtocol
from observability.metrics import MetricsCollector
from pipeline.query import QueryPipeline
from pipeline.ingest import IngestPipeline
from pipeline.memory import InMemoryConversationStore
from pipeline.rewriter import LLMQueryRewriter, QueryRewriter
from ingestion.loader import PDFLoader
from ingestion.chunker import Chunker
from ingestion.indexer import BM25Indexer
from retrieval.dense import DenseRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse import SparseRetriever
from store.bm25 import BM25Store
from store.vector import VectorStore
from store.manifest import manifest_path_for, validate_manifest
from utils.logger import setup_logging, get_logger, new_correlation_id, get_correlation_id, set_correlation_id

logger = get_logger(__name__)

MAX_UPLOAD_FILES = 5
MAX_UPLOAD_FILE_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 50 * 1024 * 1024
UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024
_SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass
class DocumentSession:
    """Process-local state for an uploaded-document query session."""

    pipeline: QueryPipeline


# Module-level references set during lifespan
_pipeline: QueryPipeline | None = None
_metrics: MetricsCollector | None = None
_chunk_count: int = 0
_active_document_sessions: dict[str, DocumentSession] = {}
_shared_embedder: SentenceTransformerEmbedder | None = None
_shared_reranker: CrossEncoderReranker | None = None
_shared_generator: Generator | None = None
_shared_tracer: TracerProtocol | None = None
_shared_conversation_store: InMemoryConversationStore | None = None
_shared_query_rewriter: QueryRewriter | None = None


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


def _sanitize_pdf_filename(filename: str | None, index: int, seen_names: set[str]) -> str:
    """Return a basename-only, lowercase-.pdf filename safe for temp storage."""
    basename = Path(filename or "").name
    if not basename or basename in {".", ".."}:
        basename = f"upload_{index}.pdf"

    path = Path(basename)
    if path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    stem = _SAFE_FILENAME_PATTERN.sub("_", path.stem).strip(" ._")
    if not stem:
        stem = f"upload_{index}"

    candidate = f"{stem}.pdf"
    counter = 2
    while candidate in seen_names:
        candidate = f"{stem}_{counter}.pdf"
        counter += 1

    seen_names.add(candidate)
    return candidate


def _copy_upload_with_limits(
    upload: UploadFile,
    destination: Path,
    remaining_total_bytes: int,
) -> int:
    """Copy one upload to disk while enforcing per-file and total byte limits."""
    bytes_written = 0
    with open(destination, "wb") as out_file:
        while True:
            chunk = upload.file.read(UPLOAD_COPY_CHUNK_BYTES)
            if not chunk:
                break

            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_FILE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Each uploaded PDF must be {MAX_UPLOAD_FILE_BYTES // (1024 * 1024)}MB or smaller",
                )
            if bytes_written > remaining_total_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Total upload size must be {MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)}MB or smaller",
                )

            out_file.write(chunk)

    return bytes_written


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline once at startup, tear down on shutdown."""
    global _pipeline, _metrics, _chunk_count
    global _shared_embedder, _shared_reranker, _shared_generator, _shared_tracer
    global _shared_conversation_store, _shared_query_rewriter

    settings = load_settings()
    setup_logging(log_level=settings.log_level)
    logger.info("Starting RAG Scholar API server...")

    # Observability
    _shared_tracer = _build_tracer(settings.observability_backend)
    _metrics = MetricsCollector()

    # Retrieval components
    _shared_embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
    vector_store = VectorStore(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path,
    )
    _chunk_count = vector_store._collection.count()

    bm25_store = BM25Store()
    bm25_store.load(Path(settings.bm25_index_path))
    validate_manifest(
        path=manifest_path_for(Path(settings.bm25_index_path)),
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        papers_dir=Path("data/papers"),
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    dense_retriever = DenseRetriever(vector_store, _shared_embedder, top_k=settings.retrieval_top_k)
    sparse_retriever = SparseRetriever(bm25_store, top_k=settings.retrieval_top_k)
    _shared_reranker = CrossEncoderReranker(
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
    _shared_generator = Generator(llm_client, prompt_builder, citation_validator)
    _shared_conversation_store = InMemoryConversationStore()
    _shared_query_rewriter = LLMQueryRewriter(llm_client)

    # Wire pipeline
    _pipeline = QueryPipeline(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=_shared_reranker,
        generator=_shared_generator,
        rrf_k=settings.rrf_k,
        tracer=_shared_tracer,
        metrics_collector=_metrics,
        conversation_store=_shared_conversation_store,
        query_rewriter=_shared_query_rewriter,
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


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach a correlation ID to every API request and response."""
    correlation_id = request.headers.get("X-Request-ID") or new_correlation_id()
    set_correlation_id(correlation_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


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

    The optional request.session_id selects an uploaded-document session only.
    Conversation memory is intentionally modeled separately from this value.

    Satisfies: FR-WEB-01 (POST /query), FR-WEB-02 (thin wrapper),
               FR-WEB-03 (structured JSON), FR-WEB-04 (error handling)
    """
    if request.session_id:
        document_session = _active_document_sessions.get(request.session_id)
        if document_session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        pipeline_to_use = document_session.pipeline
    else:
        if _pipeline is None:
            raise HTTPException(status_code=503, detail="Default pipeline not initialized")
        pipeline_to_use = _pipeline

    conversation_id = request.conversation_id or str(uuid.uuid4())
    conversation_scope = request.session_id or "default"
    scoped_conversation_id = f"{conversation_scope}:{conversation_id}"

    try:
        result = await run_in_threadpool(
            pipeline_to_use.run,
            request.query,
            conversation_id=scoped_conversation_id,
        )

        return QueryResponse(
            answer=result.answer,
            citations=[
                CitationSchema(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    page=c.page + 1,
                    passage=c.passage,
                )
                for c in result.citations
            ],
            refused=result.refused,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            timings_ms=result.timings_ms,
            route=result.route,
            conversation_id=conversation_id,
            retrieval_query=result.retrieval_query,
            query_rewritten=result.query_rewritten,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error="PIPELINE_ERROR",
                message="Query processing failed. Use the request_id when checking backend logs.",
                request_id=get_correlation_id(),
            ).model_dump(),
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(files: list[UploadFile] = File(...)):
    """Accept PDFs and return an uploaded-document session_id."""
    settings = load_settings()

    if (
        _shared_embedder is None
        or _shared_reranker is None
        or _shared_generator is None
        or _shared_tracer is None
        or _shared_conversation_store is None
        or _shared_query_rewriter is None
    ):
        raise HTTPException(status_code=503, detail="Shared pipeline components not initialized")
    
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_UPLOAD_FILES} files allowed per session")

    session_id = str(uuid.uuid4())
    logger.info(f"Starting ephemeral session {session_id} for {len(files)} files")

    # Use a temporary directory to save the uploaded PDFs for ingestion
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        seen_names: set[str] = set()
        total_bytes = 0
        for index, f in enumerate(files, start=1):
            safe_filename = _sanitize_pdf_filename(f.filename, index, seen_names)
            remaining_total_bytes = MAX_UPLOAD_TOTAL_BYTES - total_bytes
            file_path = temp_dir_path / safe_filename
            total_bytes += await run_in_threadpool(
                _copy_upload_with_limits,
                f,
                file_path,
                remaining_total_bytes,
            )

        # Build ephemeral components
        ephemeral_vector_store = VectorStore(
            collection_name=f"session_{session_id}", 
            ephemeral=True
        )
        ephemeral_bm25_store = BM25Store()
        
        ingest_pipeline = IngestPipeline(
            loader=PDFLoader(),
            chunker=Chunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
            embedder=_shared_embedder,
            bm25_indexer=BM25Indexer(),
            vector_store=ephemeral_vector_store,
            bm25_index_path=Path("/dev/null") # Won't be written in ephemeral mode
        )
        ingest_pipeline.set_bm25_store(ephemeral_bm25_store)
        
        ingest_result = await run_in_threadpool(
            ingest_pipeline.run,
            temp_dir_path,
            ephemeral=True,
        )
        
        if ingest_result.chunks_created == 0:
            raise HTTPException(status_code=400, detail="Could not extract any text from uploaded PDFs")

        # Build ephemeral QueryPipeline
        dense_retriever = DenseRetriever(ephemeral_vector_store, _shared_embedder, top_k=settings.retrieval_top_k)
        sparse_retriever = SparseRetriever(ephemeral_bm25_store, top_k=settings.retrieval_top_k)

        session_pipeline = QueryPipeline(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker=_shared_reranker,
            generator=_shared_generator,
            rrf_k=settings.rrf_k,
            tracer=_shared_tracer,
            metrics_collector=_metrics,
            conversation_store=_shared_conversation_store,
            query_rewriter=_shared_query_rewriter,
        )

        _active_document_sessions[session_id] = DocumentSession(
            pipeline=session_pipeline,
        )
        
        return UploadResponse(
            session_id=session_id,
            files_processed=ingest_result.files_processed,
            chunks_created=ingest_result.chunks_created
        )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear an uploaded-document session and free its process-local memory."""
    if session_id in _active_document_sessions:
        del _active_document_sessions[session_id]
        logger.info(f"Cleared uploaded-document session {session_id}")
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Session not found")


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
        "avg_stage_timings_ms": summary.avg_stage_timings_ms,
        "requests_by_route": summary.requests_by_route,
        "total_cost_usd": summary.total_cost_usd,
        "avg_cost_usd": summary.avg_cost_usd,
        "citation_coverage_pct": summary.citation_coverage_pct,
        "failure_rate_pct": summary.failure_rate_pct,
    }
