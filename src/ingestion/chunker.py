"""Text chunking with overlapping token-aware splits.

Implements: HLD 3.2
Satisfies: FR-02, FR-03, FR-05
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.loader import RawPage
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with stable ID and source metadata."""
    id: str          # "{source}_p{page}_c{chunk_index}" — stable, unique
    text: str
    source: str      # paper filename
    page: int
    chunk_index: int  # sequential index within the document


class Chunker:
    """Splits raw pages into overlapping chunks using LangChain's token-aware splitter."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between consecutive chunks in tokens
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, pages: list[RawPage]) -> list[Chunk]:
        """
        Split all pages into chunks.
        chunk_index is global per document (resets per source file).

        Returns:
            Flat list of Chunk objects across all pages
        """
        chunks: list[Chunk] = []
        source_chunk_counter: dict[str, int] = {}

        for page in pages:
            splits = self._splitter.split_text(page.text)

            for split_text in splits:
                source = page.source

                if source not in source_chunk_counter:
                    source_chunk_counter[source] = 0

                chunk_index = source_chunk_counter[source]
                chunk_id = f"{source}_p{page.page}_c{chunk_index}"

                chunks.append(Chunk(
                    id=chunk_id,
                    text=split_text,
                    source=source,
                    page=page.page,
                    chunk_index=chunk_index,
                ))

                source_chunk_counter[source] += 1

        logger.info(
            f"Created {len(chunks)} chunks from {len(source_chunk_counter)} documents"
        )
        return chunks
