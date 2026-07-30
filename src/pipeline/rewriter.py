"""
Follow-up query rewriting for conversational RAG.
"""
import time
from dataclasses import dataclass
from typing import Protocol

from llm.client import LLMClient
from pipeline.memory import ConversationTurn
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RewriteResult:
    """Result of attempting to produce a standalone retrieval query."""

    query: str
    rewritten: bool = False
    cost_usd: float = 0.0
    error: str = ""


class QueryRewriter(Protocol):
    """Produces standalone retrieval queries from compact conversation history."""

    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn],
        timings_ms: dict[str, float] | None = None,
    ) -> RewriteResult:
        ...


class NoOpQueryRewriter:
    """Fallback rewriter used when no LLM rewriter is configured."""

    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn],
        timings_ms: dict[str, float] | None = None,
    ) -> RewriteResult:
        if timings_ms is not None:
            timings_ms["query_rewriting"] = 0.0
        return RewriteResult(query=query, rewritten=False)


class LLMQueryRewriter:
    """LLM-backed standalone query rewriter for follow-up questions."""

    _SYSTEM_PROMPT = (
        "You rewrite conversational follow-up questions into standalone search "
        "queries for a retrieval-augmented generation system over research "
        "papers. Do not answer the question. Return only one standalone search "
        "query. Preserve technical terms, paper names, datasets, metrics, and "
        "constraints from the conversation. If the latest question is already "
        "standalone, return it unchanged."
    )

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def rewrite(
        self,
        query: str,
        history: list[ConversationTurn],
        timings_ms: dict[str, float] | None = None,
    ) -> RewriteResult:
        if not history:
            if timings_ms is not None:
                timings_ms["query_rewriting"] = 0.0
            return RewriteResult(query=query, rewritten=False)

        stage_start = time.perf_counter()
        try:
            response = self._llm.complete(
                self._SYSTEM_PROMPT,
                self._build_user_prompt(query, history),
            )
            rewritten_query = self._clean_query(response.text)
            if not rewritten_query:
                logger.warning("Query rewrite returned an empty query; using original query")
                return RewriteResult(
                    query=query,
                    rewritten=False,
                    cost_usd=response.cost_usd,
                    error="empty_rewrite",
                )

            return RewriteResult(
                query=rewritten_query,
                rewritten=rewritten_query.strip() != query.strip(),
                cost_usd=response.cost_usd,
            )
        except Exception as e:
            logger.warning(f"Query rewrite failed; using original query: {e}")
            return RewriteResult(query=query, rewritten=False, error=str(e))
        finally:
            if timings_ms is not None:
                timings_ms["query_rewriting"] = (time.perf_counter() - stage_start) * 1000

    def _build_user_prompt(self, query: str, history: list[ConversationTurn]) -> str:
        formatted_history = []
        for i, turn in enumerate(history, start=1):
            citations = ", ".join(turn.citation_chunk_ids) or "none"
            formatted_history.append(
                f"Turn {i}\n"
                f"User question: {turn.user_query}\n"
                f"Retrieval query used: {turn.retrieval_query}\n"
                f"Assistant answer summary: {turn.answer}\n"
                f"Cited chunk IDs: {citations}"
            )

        history_text = "\n\n".join(formatted_history)
        return (
            "Recent conversation:\n"
            f"{history_text}\n\n"
            f"Latest user question:\n{query}\n\n"
            "Standalone search query:"
        )

    @staticmethod
    def _clean_query(raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1].strip()
        return " ".join(cleaned.split())
