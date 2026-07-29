"""
Rule-based query routing before expensive RAG work.
"""
import re
from dataclasses import dataclass
from enum import Enum


class QueryRoute(str, Enum):
    """Coarse route selected before retrieval."""

    GREETING = "GREETING"
    APP_HELP = "APP_HELP"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    BASIC_NON_RAG = "BASIC_NON_RAG"
    RAG_FACTUAL = "RAG_FACTUAL"
    RAG_SUMMARY = "RAG_SUMMARY"
    RAG_EXACT_KEYWORD_TABLE = "RAG_EXACT_KEYWORD_TABLE"
    FOLLOW_UP = "FOLLOW_UP"


@dataclass(frozen=True)
class RouteDecision:
    """Routing result for a user query."""

    route: QueryRoute
    should_run_rag: bool
    answer: str = ""
    refused: bool = False
    reason: str = ""


class RuleBasedQueryRouter:
    """Conservative pre-RAG router.

    This router only short-circuits queries that are clearly not research-paper
    questions. Ambiguous queries default to full RAG to preserve answer quality.
    """

    _GREETING_PATTERNS = (
        re.compile(r"^(hi|hello|hey|yo|good morning|good afternoon|good evening)[!. ]*$"),
        re.compile(r"^(thanks|thank you|thx)[!. ]*$"),
    )

    _APP_HELP_PATTERNS = (
        re.compile(r"\b(help|how do i use|how to use|what can you do)\b"),
        re.compile(r"\b(what can i ask|what is this app|what is this tool)\b"),
    )

    _FOLLOW_UP_PATTERNS = (
        re.compile(r"^(what about|how about|tell me more|go on|continue|elaborate)\b"),
        re.compile(r"^(why|how so|explain that|explain it)[?!. ]*$"),
        re.compile(r"\b(that|this|it|they|them|those|second one|first one)\b"),
    )

    _SUMMARY_PATTERNS = (
        re.compile(r"\b(summarize|summary|overview|main idea|key points|takeaways)\b"),
    )

    _EXACT_KEYWORD_PATTERNS = (
        re.compile(r"\b(table|figure|equation|algorithm|dataset|benchmark|score|accuracy|f1|auc|bleu|rouge)\b"),
        re.compile(r"\b(page|section|appendix)\b"),
    )

    _OUT_OF_SCOPE_PATTERNS = (
        re.compile(r"\b(weather|forecast)\b"),
        re.compile(r"\b(tell me a joke|write a poem|write a song)\b"),
        re.compile(r"\b(recipe|cook|cooking)\b"),
        re.compile(r"\b(stock price|share price|crypto price)\b"),
        re.compile(r"\b(who won|sports score|game score)\b"),
        re.compile(r"\b(book a flight|hotel booking|restaurant reservation)\b"),
    )

    def route(self, query: str) -> RouteDecision:
        normalized = self._normalize(query)

        if not normalized:
            return RouteDecision(
                route=QueryRoute.BASIC_NON_RAG,
                should_run_rag=False,
                answer="Please ask a specific question about the indexed research papers.",
                refused=True,
                reason="empty_query",
            )

        if self._matches_any(normalized, self._GREETING_PATTERNS):
            return RouteDecision(
                route=QueryRoute.GREETING,
                should_run_rag=False,
                answer=(
                    "Hi. Ask me a question about the indexed research papers, "
                    "and I will answer with citations when there is enough context."
                ),
                refused=False,
                reason="greeting",
            )

        if self._matches_any(normalized, self._APP_HELP_PATTERNS):
            return RouteDecision(
                route=QueryRoute.APP_HELP,
                should_run_rag=False,
                answer=(
                    "Ask a question about the indexed research papers. I can retrieve "
                    "relevant passages, answer from that context, and return citations."
                ),
                refused=False,
                reason="app_help",
            )

        if self._is_too_weak(normalized):
            return RouteDecision(
                route=QueryRoute.BASIC_NON_RAG,
                should_run_rag=False,
                answer="Please ask a more specific question about the indexed research papers.",
                refused=True,
                reason="too_weak",
            )

        if self._matches_any(normalized, self._OUT_OF_SCOPE_PATTERNS):
            return RouteDecision(
                route=QueryRoute.OUT_OF_SCOPE,
                should_run_rag=False,
                answer=(
                    "I can only answer questions about the indexed research papers. "
                    "Please ask a question grounded in that corpus."
                ),
                refused=True,
                reason="out_of_scope",
            )

        if self._matches_any(normalized, self._FOLLOW_UP_PATTERNS):
            return RouteDecision(
                route=QueryRoute.FOLLOW_UP,
                should_run_rag=True,
                reason="follow_up_shape",
            )

        if self._matches_any(normalized, self._SUMMARY_PATTERNS):
            return RouteDecision(
                route=QueryRoute.RAG_SUMMARY,
                should_run_rag=True,
                reason="summary_shape",
            )

        if self._matches_any(normalized, self._EXACT_KEYWORD_PATTERNS):
            return RouteDecision(
                route=QueryRoute.RAG_EXACT_KEYWORD_TABLE,
                should_run_rag=True,
                reason="exact_keyword_or_table_shape",
            )

        return RouteDecision(
            route=QueryRoute.RAG_FACTUAL,
            should_run_rag=True,
            reason="default_rag",
        )

    @staticmethod
    def _normalize(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    @staticmethod
    def _matches_any(query: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
        return any(pattern.search(query) for pattern in patterns)

    @staticmethod
    def _is_too_weak(query: str) -> bool:
        tokens = re.findall(r"[a-z0-9]+", query)
        if len(tokens) == 1 and len(tokens[0]) <= 3:
            return True
        return query in {"?", "??", "...", "paper", "papers", "research"}
