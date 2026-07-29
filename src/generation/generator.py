"""
LLM generation with citation enforcement.
"""
import time
from dataclasses import dataclass, field

from generation.citations import Citation, CitationValidator
from generation.prompt import PromptBuilder
from llm.client import LLMClient
from store.vector import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

REFUSAL_PREFIX = "INSUFFICIENT_CONTEXT"


def _record_timing(
    timings_ms: dict[str, float] | None,
    key: str,
    start: float,
) -> None:
    if timings_ms is not None:
        timings_ms[key] = (time.perf_counter() - start) * 1000


@dataclass
class GeneratorResponse:
    """Final response from the generator with citations and metadata."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    model: str = ""


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
        self,
        query: str,
        chunks: list[RetrievedChunk],
        timings_ms: dict[str, float] | None = None,
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
        stage_start = time.perf_counter()
        system_prompt, user_prompt = self._prompt_builder.build(query, chunks)
        _record_timing(timings_ms, "prompt_building", stage_start)

        # Step 2: Call LLM
        stage_start = time.perf_counter()
        llm_response = self._llm.complete(system_prompt, user_prompt)
        _record_timing(timings_ms, "llm_call", stage_start)
        raw_answer = llm_response.text

        logger.info(
            f"LLM response: {llm_response.tokens_input} in, "
            f"{llm_response.tokens_output} out, model={llm_response.model}"
        )

        # Step 3: Check for refusal
        if raw_answer.strip().startswith(REFUSAL_PREFIX):
            if timings_ms is not None:
                timings_ms["citation_validation"] = 0.0
            return GeneratorResponse(
                answer=raw_answer,
                refused=True,
                refusal_reason=raw_answer,
                tokens_input=llm_response.tokens_input,
                tokens_output=llm_response.tokens_output,
                cost_usd=llm_response.cost_usd,
                model=llm_response.model,
            )

        # Step 4: Validate citations
        stage_start = time.perf_counter()
        valid_citations, invalid_ids = self._citation_validator.validate(
            raw_answer, chunks
        )
        _record_timing(timings_ms, "citation_validation", stage_start)

        # Step 5: Strip invalid citations from the answer
        clean_answer = raw_answer
        for invalid_id in invalid_ids:
            clean_answer = clean_answer.replace(f"[SOURCE: {invalid_id}]", "")

        return GeneratorResponse(
            answer=clean_answer.strip(),
            citations=valid_citations,
            refused=False,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            cost_usd=llm_response.cost_usd,
            model=llm_response.model,
        )
