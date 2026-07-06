"""One-time ingestion CLI — run this before querying.

Usage:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

# Add src/ to Python path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config.settings import load_settings
from ingestion.loader import PDFLoader
from ingestion.chunker import Chunker
from ingestion.indexer import BM25Indexer
from store.vector import VectorStore
from llm.embeddings import SentenceTransformerEmbedder
from pipeline.ingest import IngestPipeline


def main() -> None:
    settings = load_settings()

    # Construct components from settings
    loader = PDFLoader()
    chunker = Chunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = SentenceTransformerEmbedder(model_name=settings.embedding_model)
    bm25_indexer = BM25Indexer()
    vector_store = VectorStore(
        collection_name=settings.collection_name,
        persist_directory=settings.vector_store_path,
    )

    # Wire pipeline
    pipeline = IngestPipeline(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
        bm25_index_path=Path(settings.bm25_index_path),
    )

    # Run
    result = pipeline.run(Path("data/papers"))

    print(
        f"Ingested {result.chunks_created} chunks from "
        f"{result.files_processed} files. "
        f"Failed: {result.failed_files or 'none'}"
    )


if __name__ == "__main__":
    main()
