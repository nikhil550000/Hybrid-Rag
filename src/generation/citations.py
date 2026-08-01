"""
Citation extraction and validation.
"""
import re
from dataclasses import dataclass

from store.vector import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

# Matches [SOURCE: chunk_id] tags in LLM output, tolerating harmless spacing/case variants.
CITATION_PATTERN = re.compile(r"\[\s*SOURCE\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)


@dataclass
class Citation:
    """A validated citation with full source metadata."""
    chunk_id: str
    source: str     # paper filename
    page: int        # 0-indexed page number from the retrieved chunk
    passage: str    # text from the cited chunk


class CitationValidator:
    """Extracts [SOURCE: chunk_id] tags and validates them against provided chunks."""

    def validate(
        self, llm_text: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[Citation], list[str]]:
        """
        Extract and validate all citations in the LLM output.

        Args:
            llm_text: Raw text from the LLM containing [SOURCE: chunk_id] tags
            chunks: The chunks that were passed to the LLM as context

        Returns:
            (valid_citations, invalid_ids)
            - valid_citations: Citations with full source and passage metadata
            - invalid_ids: chunk_ids cited by the LLM but not in the context
        """
        # Build lookup: chunk_id → chunk metadata
        chunk_lookup: dict[str, RetrievedChunk] = {
            chunk.chunk_id: chunk for chunk in chunks
        }

        cited_ids = [match.group(1).strip() for match in CITATION_PATTERN.finditer(llm_text)]

        valid_citations: list[Citation] = []
        invalid_ids: list[str] = []
        seen_ids: set[str] = set()

        for chunk_id in cited_ids:
            # Deduplicate — same chunk_id cited multiple times
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            if chunk_id in chunk_lookup:
                chunk = chunk_lookup[chunk_id]
                valid_citations.append(Citation(
                    chunk_id=chunk_id,
                    source=chunk.source,
                    page=chunk.page,
                    passage=chunk.text,
                ))
            else:
                invalid_ids.append(chunk_id)

        if invalid_ids:
            logger.warning(
                f"Hallucinated citations detected: {invalid_ids}. "
                f"These chunk_ids were not in the context."
            )

        logger.info(
            f"Citations: {len(valid_citations)} valid, "
            f"{len(invalid_ids)} invalid out of "
            f"{len(seen_ids)} unique cited"
        )
        return valid_citations, invalid_ids

    def remove_invalid_citations(self, llm_text: str, invalid_ids: list[str]) -> str:
        """Remove invalid citation tags using the same parser used for validation."""
        invalid_id_set = set(invalid_ids)
        if not invalid_id_set:
            return llm_text

        def replace_invalid(match: re.Match[str]) -> str:
            chunk_id = match.group(1).strip()
            if chunk_id in invalid_id_set:
                return ""
            return match.group(0)

        return CITATION_PATTERN.sub(replace_invalid, llm_text)
