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

from api.routes import app
from api.schemas import QueryRequest


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
