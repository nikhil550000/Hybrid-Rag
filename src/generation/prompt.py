"""
Prompt builder — loads versioned prompt templates.
"""
from pathlib import Path

from store.vector import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Loads prompt templates from disk and formats them with runtime data."""

    def __init__(self, prompt_version: str = "v1"):
        prompts_dir = Path(f"prompts/{prompt_version}")
        self._system_prompt = (prompts_dir / "system.txt").read_text().strip()
        self._query_template = (prompts_dir / "query.txt").read_text().strip()
        logger.info(f"Loaded prompts: version={prompt_version}")

    def build(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
        """
        Format the system and user prompts with the retrieved chunks.

        Returns:
            (system_prompt, user_prompt)
        """
        formatted_chunks = self._format_chunks(chunks)
        user_prompt = self._query_template.format(
            formatted_chunks=formatted_chunks,
            user_query=query,
        )
        return self._system_prompt, user_prompt

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        """Format chunks into numbered passages with chunk_id headers."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[Passage {i} | ID: {chunk.chunk_id} | "
                f"Source: {chunk.source} | Page: {chunk.page}]\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)
