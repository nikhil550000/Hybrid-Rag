import sys
import types

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


fitz_stub = types.ModuleType("fitz")
sys.modules.setdefault("fitz", fitz_stub)

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
chromadb_stub.EphemeralClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

sentence_transformers_stub = types.ModuleType("sentence_transformers")
sentence_transformers_stub.CrossEncoder = object
sentence_transformers_stub.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers_stub)

langchain_splitters_stub = types.ModuleType("langchain_text_splitters")
langchain_splitters_stub.RecursiveCharacterTextSplitter = object
sys.modules.setdefault("langchain_text_splitters", langchain_splitters_stub)

rank_bm25_stub = types.ModuleType("rank_bm25")
rank_bm25_stub.BM25Okapi = object
sys.modules.setdefault("rank_bm25", rank_bm25_stub)

import api.routes as routes
from api.routes import app
from api.schemas import QueryRequest
from observability.metrics import MetricsCollector


def test_query_request_rejects_empty_and_too_long_queries():
    with pytest.raises(ValidationError):
        QueryRequest(query="")

    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001)


def test_query_request_accepts_valid_query():
    request = QueryRequest(query="What does the paper say about retrieval?")

    assert request.query == "What does the paper say about retrieval?"


def test_correlation_id_middleware_returns_inbound_request_id():
    client = TestClient(app)

    response = client.get("/health", headers={"X-Request-ID": "request-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"


def test_correlation_id_middleware_generates_request_id_when_missing():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert response.headers["X-Request-ID"] != "no-correlation"


class FailingPipeline:
    def run(self, query: str, conversation_id: str | None = None):
        raise RuntimeError("simulated pipeline failure")


def test_query_exception_records_failure_metrics(monkeypatch):
    metrics = MetricsCollector()
    monkeypatch.setattr(routes, "_pipeline", FailingPipeline())
    monkeypatch.setattr(routes, "_metrics", metrics)

    client = TestClient(app)
    response = client.post("/query", json={"query": "What does the paper say about retrieval?"})

    assert response.status_code == 500
    summary = metrics.get_summary()
    assert summary.total_requests == 1
    assert summary.failure_rate_pct == 100.0
    assert summary.requests_by_route == {"PIPELINE_ERROR": 1}
    assert "total_latency" in summary.avg_stage_timings_ms


def test_health_reports_not_ready_when_components_are_missing(monkeypatch):
    monkeypatch.setattr(routes, "_pipeline", None)
    monkeypatch.setattr(routes, "_metrics", None)
    monkeypatch.setattr(routes, "_chunk_count", 0)
    monkeypatch.setattr(routes, "_settings_ready", False)
    monkeypatch.setattr(routes, "_vector_store_ready", False)
    monkeypatch.setattr(routes, "_bm25_store_ready", False)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["ready"] is False
    assert body["checks"]["pipeline"] is False
    assert body["checks"]["chunks_indexed"] is False


def test_health_reports_ready_when_startup_components_are_present(monkeypatch):
    ready_object = object()
    monkeypatch.setattr(routes, "_settings_ready", True)
    monkeypatch.setattr(routes, "_vector_store_ready", True)
    monkeypatch.setattr(routes, "_bm25_store_ready", True)
    monkeypatch.setattr(routes, "_pipeline", ready_object)
    monkeypatch.setattr(routes, "_metrics", ready_object)
    monkeypatch.setattr(routes, "_chunk_count", 7)
    monkeypatch.setattr(routes, "_shared_embedder", ready_object)
    monkeypatch.setattr(routes, "_shared_reranker", ready_object)
    monkeypatch.setattr(routes, "_shared_generator", ready_object)
    monkeypatch.setattr(routes, "_shared_tracer", ready_object)
    monkeypatch.setattr(routes, "_shared_conversation_store", ready_object)
    monkeypatch.setattr(routes, "_shared_query_rewriter", ready_object)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["chunks_indexed"] == 7
    assert body["ready"] is True
    assert all(body["checks"].values())
