"""PDF text extraction using PyMuPDF (fitz).

Implements: HLD 3.1
Satisfies: FR-01, FR-05, FR-07
"""
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RawPage:
    """A single page of extracted text with source metadata."""
    text: str
    source: str   # filename without path
    page: int     # 0-indexed page number


class PDFLoader:
    """Extracts text page-by-page from PDFs using PyMuPDF."""

    def load(self, pdf_path: Path) -> list[RawPage]:
        """
        Extract all pages from a single PDF.

        Args:
            pdf_path: Path to a .pdf file

        Returns:
            List of RawPage objects, one per page

        Raises:
            RuntimeError: If the file cannot be opened (caller handles skip logic)
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise RuntimeError(f"Cannot open {pdf_path.name}: {e}") from e

        pages: list[RawPage] = []
        source = pdf_path.name

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Skip pages with no extractable text
            if text.strip():
                pages.append(RawPage(
                    text=text,
                    source=source,
                    page=page_num,
                ))

        doc.close()
        logger.info(f"Loaded {len(pages)} pages from {source}")
        return pages

    def load_all(self, papers_dir: Path) -> tuple[list[RawPage], list[str]]:
        """
        Load all .pdf files in papers_dir.

        Failures are caught per-file: failed filename logged + added to failed list.
        Never raises — always returns what succeeded (FR-07).

        Args:
            papers_dir: Path to directory containing .pdf files

        Returns:
            (all_pages, failed_filenames)
        """
        all_pages: list[RawPage] = []
        failed_filenames: list[str] = []

        pdf_files = sorted(papers_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {papers_dir}")
            return all_pages, failed_filenames

        for pdf_path in pdf_files:
            try:
                pages = self.load(pdf_path)
                all_pages.extend(pages)
            except (RuntimeError, Exception) as e:
                logger.warning(f"Skipping {pdf_path.name}: {e}")
                failed_filenames.append(pdf_path.name)

        logger.info(
            f"Loaded {len(all_pages)} total pages from "
            f"{len(pdf_files) - len(failed_filenames)} files. "
            f"Failed: {failed_filenames or 'none'}"
        )
        return all_pages, failed_filenames
