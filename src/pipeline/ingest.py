"""Ingestion pipeline orchestrator.

Implements: HLD 3.1 → 3.4
Satisfies: FR-01 → FR-07
"""
from dataclasses import dataclass, field
from pathlib import Path

from ingestion.loader import PDFLoader
from ingestion.chunker import Chunker
from ingestion.indexer import BM25Indexer
from store.vector import VectorStore
from store.bm25 import BM25Store
from llm.embeddings import EmbeddingClient
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestResult:
    """Summary of ingestion run."""
    files_processed: int
    files_failed: int
    chunks_created: int
    failed_files: list[str] = field(default_factory=list)


class IngestPipeline:
    """Orchestrates the full ingestion flow: PDF → chunks → ChromaDB + BM25 index."""

    def __init__(
        self,
        loader: PDFLoader,
        chunker: Chunker,
        embedder: EmbeddingClient,
        bm25_indexer: BM25Indexer,
        vector_store: VectorStore,
        bm25_index_path: Path,
    ):
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._bm25_indexer = bm25_indexer
        self._vector_store = vector_store
        self._bm25_index_path = bm25_index_path
        self._bm25_store = None

    def set_bm25_store(self, store: BM25Store) -> None:
        """Inject BM25Store for ephemeral in-memory ingestion."""
        self._bm25_store = store

    def run(self, papers_dir: Path, ephemeral: bool = False) -> IngestResult:
        """
        Load all PDFs, chunk, embed, and persist.
        Skips individual PDF failures without halting (FR-07).
        Idempotent: skips chunks already in the vector store (FR-06).

        Args:
            papers_dir: Path to directory containing .pdf files
        Returns:
            IngestResult summarising what was processed and what failed
        """
        # Step 1: Load PDFs
        logger.info(f"Loading PDFs from {papers_dir}")
        pages, failed_files = self._loader.load_all(papers_dir)

        if not pages:
            logger.warning("No pages extracted — nothing to ingest")
            return IngestResult(
                files_processed=0,
                files_failed=len(failed_files),
                chunks_created=0,
                failed_files=failed_files,
            )

        # Step 2: Chunk
        chunks = self._chunker.chunk(pages)
        logger.info(f"Chunked into {len(chunks)} chunks")

        # Step 3: Embed
        logger.info("Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedder.embed_batch(texts)

        # Step 4: Store in ChromaDB (idempotent — skips duplicates)
        chunks_inserted = self._vector_store.add(chunks, embeddings)

        # Step 5: Build BM25 index
        bm25, chunk_ids = self._bm25_indexer.build(chunks)
        
        if ephemeral and self._bm25_store is not None:
            self._bm25_store.load_from_memory(
                bm25=bm25,
                chunk_ids=chunk_ids,
                chunk_texts=[c.text for c in chunks],
                chunk_metadata=[{"source": c.source, "page": c.page} for c in chunks],
            )
            logger.info("Skipped saving BM25 index to disk (ephemeral mode)")
        else:
            self._bm25_indexer.save(bm25, chunk_ids, chunks, self._bm25_index_path)

        # Count unique source files that succeeded
        source_files = {page.source for page in pages}

        result = IngestResult(
            files_processed=len(source_files),
            files_failed=len(failed_files),
            chunks_created=chunks_inserted,
            failed_files=failed_files,
        )

        logger.info(
            f"Ingestion complete: {result.files_processed} files, "
            f"{result.chunks_created} new chunks, "
            f"{result.files_failed} failed"
        )
        return result
