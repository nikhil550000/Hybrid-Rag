import sys
import types


# Keep these tests runnable in lightweight environments without the optional
# runtime libraries used by the full ingestion and vector-store modules.
fitz_stub = types.ModuleType("fitz")
sys.modules.setdefault("fitz", fitz_stub)

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
chromadb_stub.EphemeralClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

langchain_splitters_stub = types.ModuleType("langchain_text_splitters")
langchain_splitters_stub.RecursiveCharacterTextSplitter = object
sys.modules.setdefault("langchain_text_splitters", langchain_splitters_stub)

rank_bm25_stub = types.ModuleType("rank_bm25")
rank_bm25_stub.BM25Okapi = object
sys.modules.setdefault("rank_bm25", rank_bm25_stub)

from store.bm25 import BM25Store
from utils.tokenization import tokenize


class FixedScoresBM25:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.queries: list[list[str]] = []

    def get_scores(self, query: list[str]):
        self.queries.append(query)
        return self._scores


def test_tokenize_preserves_research_terms_and_removes_punctuation():
    assert tokenize("BERT-base, GPT-4; F1=0.92, p-value, Table 2.") == [
        "bert-base",
        "gpt-4",
        "f1",
        "0.92",
        "p-value",
        "table",
        "2",
    ]


def test_bm25_query_excludes_zero_score_documents():
    bm25 = FixedScoresBM25([0.0, 2.5, 0.0, 1.0])
    store = BM25Store()
    store.load_from_memory(
        bm25=bm25,
        chunk_ids=["zero-a", "match-a", "zero-b", "match-b"],
        chunk_texts=["a", "b", "c", "d"],
        chunk_metadata=[
            {"source": "paper.pdf", "page": 0},
            {"source": "paper.pdf", "page": 1},
            {"source": "paper.pdf", "page": 2},
            {"source": "paper.pdf", "page": 3},
        ],
    )

    results = store.query("unseen term", top_k=4)

    assert [result.chunk_id for result in results] == ["match-a", "match-b"]
    assert [result.score for result in results] == [2.5, 1.0]
    assert bm25.queries == [["unseen", "term"]]
