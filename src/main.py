"""CLI entry point for querying the RAG system.

Usage:
    python src/main.py "What is the attention mechanism?"
"""
import sys
from pathlib import Path

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import load_settings
from generation.citations import CitationValidator
from generation.generator import Generator
from generation.prompt import PromptBuilder
from llm.client import get_llm_client
from llm.embeddings import SentenceTransformerEmbedder
from pipeline.query import QueryPipeline
from retrieval.dense import DenseRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse import SparseRetriever
from store.bm25 import BM25Store
from store.vector import VectorStore


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/main.py \"Your question here\"")
        sys.exit(1)

    query = sys.argv[1]
    settings = load_settings()

    # --- Retrieval components ---
    embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
    vector_store = VectorStore(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path,
    )
    bm25_store = BM25Store()
    bm25_store.load(Path(settings.bm25_index_path))

    dense_retriever = DenseRetriever(vector_store, embedder, top_k=settings.retrieval_top_k)
    sparse_retriever = SparseRetriever(bm25_store, top_k=settings.retrieval_top_k)
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        top_n=settings.reranked_top_n,
    )

    # --- Generation components ---
    llm_client = get_llm_client(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    prompt_builder = PromptBuilder(prompt_version=settings.prompt_version)
    citation_validator = CitationValidator()
    generator = Generator(llm_client, prompt_builder, citation_validator)

    # --- Wire pipeline ---
    pipeline = QueryPipeline(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
        generator=generator,
        rrf_k=settings.rrf_k,
    )

    # --- Run ---
    result = pipeline.run(query)

    # --- Display ---
    print(f"\n{'='*80}")
    print(f"Query: {result.query}")
    print(f"{'='*80}\n")

    if result.refused:
        print(result.answer)
    else:
        print(result.answer)
        print(f"\n{'─'*80}")
        print(f"Citations ({len(result.citations)}):")
        for i, cit in enumerate(result.citations, 1):
            print(f"\n  [{i}] {cit.chunk_id}")
            print(f"      Source: {cit.source}")
            print(f"      Passage: {cit.passage[:150]}...")

    print(f"\n{'─'*80}")
    print(f"Latency: {result.latency_ms}ms")


if __name__ == "__main__":
    main()
