"""Pydantic schemas for the FastAPI endpoints.

Implements: HLD 3.12 / LLD Section 4
Satisfies: FR-WEB-01, FR-WEB-03
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query from the frontend."""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    session_id: str | None = Field(
        None,
        description=(
            "Optional uploaded-document session ID returned by /upload. "
            "This is not a conversation or chat-memory ID."
        ),
    )
    conversation_id: str | None = Field(
        None,
        description=(
            "Optional conversation ID for compact follow-up query rewriting. "
            "This is separate from uploaded-document session_id."
        ),
    )


class CitationSchema(BaseModel):
    """A single citation reference in the response."""
    chunk_id: str
    source: str
    page: int
    passage: str


class QueryResponse(BaseModel):
    """Successful query response with answer and citations."""
    answer: str
    citations: list[CitationSchema]
    refused: bool
    latency_ms: float
    cost_usd: float
    timings_ms: dict[str, float] = Field(default_factory=dict)
    route: str = "RAG_FACTUAL"
    conversation_id: str | None = None
    retrieval_query: str = ""
    query_rewritten: bool = False


class ErrorResponse(BaseModel):
    """Structured error — never expose raw stack traces (FR-WEB-04)."""
    error: str      # e.g. "PIPELINE_ERROR", "VALIDATION_ERROR"
    message: str    # human-readable description


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    chunks_indexed: int

class UploadResponse(BaseModel):
    """Response returned when successfully uploading PDFs for a session."""
    session_id: str = Field(
        ...,
        description="Uploaded-document session ID for querying the uploaded files",
    )
    files_processed: int
    chunks_created: int
