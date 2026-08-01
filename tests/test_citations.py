import sys
import types

fitz_stub = types.ModuleType("fitz")
sys.modules.setdefault("fitz", fitz_stub)

langchain_splitters_stub = types.ModuleType("langchain_text_splitters")
langchain_splitters_stub.RecursiveCharacterTextSplitter = object
sys.modules.setdefault("langchain_text_splitters", langchain_splitters_stub)

chromadb_stub = types.ModuleType("chromadb")
chromadb_stub.PersistentClient = object
chromadb_stub.EphemeralClient = object
sys.modules.setdefault("chromadb", chromadb_stub)

from generation.citations import CitationValidator
from generation.generator import Generator, NO_VALID_CITATIONS_REFUSAL
from llm.client import LLMResponse
from store.vector import RetrievedChunk


class FixedLLM:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        return LLMResponse(
            text=self.text,
            tokens_input=10,
            tokens_output=5,
            model="test-model",
            cost_usd=0.01,
        )


class FixedPromptBuilder:
    def build(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
        return "system", f"query: {query}"


def _chunk(chunk_id: str = "chunk-a", page: int = 2) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"passage for {chunk_id}",
        source="paper.pdf",
        page=page,
        score=0.9,
        retrieval_method="reranked",
    )


def test_validator_accepts_valid_citations_and_keeps_zero_indexed_page():
    validator = CitationValidator()

    citations, invalid_ids = validator.validate(
        "A supported claim [ source : chunk-a ]. Bad cite [ SOURCE : missing ].",
        [_chunk("chunk-a", page=3)],
    )

    assert invalid_ids == ["missing"]
    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-a"
    assert citations[0].page == 3
    assert citations[0].passage == "passage for chunk-a"


def test_invalid_citation_cleanup_uses_regex_variants():
    validator = CitationValidator()

    cleaned = validator.remove_invalid_citations(
        "Keep this [SOURCE: chunk-a]. Remove this [ source : missing ].",
        ["missing"],
    )

    assert "[SOURCE: chunk-a]" in cleaned
    assert "missing" not in cleaned


def test_generator_refuses_non_refusal_answer_with_zero_valid_citations():
    generator = Generator(
        llm_client=FixedLLM("This answer has no citation."),
        prompt_builder=FixedPromptBuilder(),
        citation_validator=CitationValidator(),
    )

    result = generator.generate("question", [_chunk("chunk-a")])

    assert result.refused is True
    assert result.answer == NO_VALID_CITATIONS_REFUSAL
    assert result.citations == []


def test_generator_removes_invalid_citations_but_keeps_valid_answer():
    generator = Generator(
        llm_client=FixedLLM(
            "Supported claim [SOURCE: chunk-a]. Unsupported claim [ source : missing ]."
        ),
        prompt_builder=FixedPromptBuilder(),
        citation_validator=CitationValidator(),
    )

    result = generator.generate("question", [_chunk("chunk-a", page=4)])

    assert result.refused is False
    assert len(result.citations) == 1
    assert result.citations[0].page == 4
    assert "[SOURCE: chunk-a]" in result.answer
    assert "missing" not in result.answer
