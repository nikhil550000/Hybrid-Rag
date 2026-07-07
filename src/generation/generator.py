"""LLM generation with citation enforcement.

Implements: HLD 3.10, 3.11
Satisfies: FR-15, FR-17, FR-18, FR-19
"""
from dataclasses import dataclass, field

from generation.citations import Citation, CitationValidator
from generation.prompt import PromptBuilder
from llm.client import LLMClient
from store.vector import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

REFUSAL_PREFIX = "INSUFFICIENT_CONTEXT"


@dataclass
class GeneratorResponse:
    """Final response from the generator with citations and metadata."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""


class Generator:
    """Builds prompt → calls LLM → validates citations → returns structured response."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        citation_validator: CitationValidator,
    ):
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._citation_validator = citation_validator

    def generate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> GeneratorResponse:
        """
        Generate a cited answer from the LLM.

        Flow:
        1. Build prompt with chunks
        2. Call LLM
        3. Check for INSUFFICIENT_CONTEXT refusal
        4. Validate citations (two-layer: prompt instruction + post-processing)
        5. Strip invalid citations from answer

        Args:
            query: User's question
            chunks: Reranked chunks to use as context

        Returns:
            GeneratorResponse with answer, valid citations, and refusal status
        """
        # Step 1: Build prompts
        system_prompt, user_prompt = self._prompt_builder.build(query, chunks)

        # Step 2: Call LLM
        llm_response = self._llm.complete(system_prompt, user_prompt)
        raw_answer = llm_response.text

        logger.info(
            f"LLM response: {llm_response.tokens_input} in, "
            f"{llm_response.tokens_output} out, model={llm_response.model}"
        )

        # Step 3: Check for refusal
        if raw_answer.strip().startswith(REFUSAL_PREFIX):
            return GeneratorResponse(
                answer=raw_answer,
                refused=True,
                refusal_reason=raw_answer,
            )

        # Step 4: Validate citations
        valid_citations, invalid_ids = self._citation_validator.validate(
            raw_answer, chunks
        )

        # Step 5: Strip invalid citations from the answer
        clean_answer = raw_answer
        for invalid_id in invalid_ids:
            clean_answer = clean_answer.replace(f"[SOURCE: {invalid_id}]", "")

        return GeneratorResponse(
            answer=clean_answer.strip(),
            citations=valid_citations,
            refused=False,
        )
