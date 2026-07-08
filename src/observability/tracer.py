"""Observability tracer — dependency-injected Langfuse integration.

Implements: HLD 3.10
Satisfies: FR-26, FR-27, FR-28, NFR-08

Design:
    TracerProtocol defines the interface. LangfuseTracer sends spans to Langfuse.
    NullTracer does nothing (used in tests and when backend="none").
    Every LangfuseTracer method is wrapped in try/except — observability failure
    MUST NOT break query serving (FR-28).

    Uses the Langfuse Python SDK v4 API (get_client, start_as_current_observation,
    start_observation, update, flush). Follows the official instrumentation guide:
    https://langfuse.com/docs/observability/sdk/instrumentation
"""
import uuid
from dataclasses import dataclass, field
from typing import Protocol, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TraceContext:
    """Holds state for a single trace (one query request)."""
    trace_id: str
    query: str
    _root_span: Any = None    # LangfuseSpan from start_as_current_observation
    _generation: Any = None   # LangfuseGeneration child span


# ─── Protocol ─────────────────────────────────────────────────────────────────

class TracerProtocol(Protocol):
    """All tracers implement this interface. NullTracer used in tests."""

    def start_trace(self, query: str) -> TraceContext:
        """Open a new root trace for this query request."""
        ...

    def log_retrieval(
        self,
        ctx: TraceContext,
        pre_rerank: list,
        post_rerank: list,
    ) -> None:
        """Log retrieval results (pre- and post-rerank candidate lists)."""
        ...

    def log_generation(
        self,
        ctx: TraceContext,
        prompt: str,
        response: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Log the LLM generation step with full metadata."""
        ...

    def end_trace(self, ctx: TraceContext, refused: bool) -> None:
        """Close the root trace and flush to backend."""
        ...


# ─── Langfuse v4 implementation ──────────────────────────────────────────────

class LangfuseTracer:
    """Sends structured spans to Langfuse v4 using the Python SDK.

    Best practices followed (per Langfuse skill / instrumentation.md):
    - Uses get_client() for singleton Langfuse client
    - Uses start_observation() with as_type for proper observation types
    - Nested hierarchy: root span → retrieval span + generation span
    - Calls flush() in short-lived/script contexts
    - Descriptive trace names, not generic IDs
    - Generation observations capture model, usage_details, cost_details
    - All calls wrapped in try/except — tracing MUST NOT break serving
    """

    def __init__(self, public_key: str, secret_key: str, host: str):
        try:
            from langfuse import get_client
            # get_client() reads from env vars by default, but we can also
            # pass credentials explicitly via env vars set before import
            import os
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
            os.environ.setdefault("LANGFUSE_HOST", host)

            self._langfuse = get_client()
            logger.info(f"LangfuseTracer initialized (host={host})")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse client: {e}. Tracing disabled.")
            self._langfuse = None

    def start_trace(self, query: str) -> TraceContext:
        """Create a new root span for this query using Langfuse v4 SDK."""
        trace_id = str(uuid.uuid4())
        ctx = TraceContext(trace_id=trace_id, query=query)

        if self._langfuse is None:
            return ctx

        try:
            # Create root span using start_observation (manual lifecycle)
            # We use start_observation instead of start_as_current_observation
            # because our pipeline is method-based, not context-manager-based.
            ctx._root_span = self._langfuse.start_observation(
                name="rag-query",
                input={"query": query},
                metadata={"trace_id": trace_id},
            )
            logger.debug(f"Trace started: {trace_id}")
        except Exception as e:
            logger.warning(f"Failed to start Langfuse trace: {e}")

        return ctx

    def log_retrieval(
        self,
        ctx: TraceContext,
        pre_rerank: list,
        post_rerank: list,
    ) -> None:
        """Log retrieval candidates as a child span of the root span."""
        if ctx._root_span is None:
            return

        try:
            retrieval_output = {
                "pre_rerank_count": len(pre_rerank),
                "post_rerank_count": len(post_rerank),
                "pre_rerank_ids": [c.chunk_id for c in pre_rerank[:10]],
                "post_rerank_ids": [c.chunk_id for c in post_rerank],
            }
            # Create child span on the root span object for proper nesting
            retrieval_span = ctx._root_span.start_observation(
                name="retrieval",
                as_type="span",
            )
            retrieval_span.update(
                input={"query": ctx.query},
                output=retrieval_output,
                metadata=retrieval_output,
            )
            retrieval_span.end()
        except Exception as e:
            logger.warning(f"Failed to log retrieval span: {e}")

    def log_generation(
        self,
        ctx: TraceContext,
        prompt: str,
        response: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Log the LLM generation as a generation observation (child of root)."""
        if ctx._root_span is None:
            return

        try:
            # Create generation child on the root span for proper nesting
            gen = ctx._root_span.start_observation(
                name="llm-generation",
                as_type="generation",
            )
            gen.update(
                input=prompt[:500],
                output=response[:500],
                usage_details={
                    "input": tokens_in,
                    "output": tokens_out,
                    "total": tokens_in + tokens_out,
                },
                cost_details={
                    "total": cost_usd,
                },
                metadata={
                    "latency_ms": latency_ms,
                },
            )
            gen.end()
            ctx._generation = gen
        except Exception as e:
            logger.warning(f"Failed to log generation span: {e}")

    def end_trace(self, ctx: TraceContext, refused: bool) -> None:
        """Close the root span and flush pending events to Langfuse."""
        if ctx._root_span is None:
            return

        try:
            ctx._root_span.update(
                output={"refused": refused},
                metadata={"status": "refused" if refused else "answered"},
            )
            ctx._root_span.end()
        except Exception as e:
            logger.warning(f"Failed to end Langfuse trace: {e}")

        # Flush pending events (critical for scripts / short-lived processes)
        try:
            self._langfuse.flush()
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse: {e}")


# ─── Null implementation (tests + backend=none) ──────────────────────────────

class NullTracer:
    """No-op tracer for unit tests. Satisfies TracerProtocol, does nothing."""

    def start_trace(self, query: str) -> TraceContext:
        return TraceContext(trace_id="null", query=query)

    def log_retrieval(self, ctx: TraceContext, pre_rerank: list, post_rerank: list) -> None:
        pass

    def log_generation(
        self, ctx: TraceContext, prompt: str, response: str,
        tokens_in: int, tokens_out: int, latency_ms: float, cost_usd: float,
    ) -> None:
        pass

    def end_trace(self, ctx: TraceContext, refused: bool) -> None:
        pass
