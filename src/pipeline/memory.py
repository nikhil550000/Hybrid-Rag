"""
Compact process-local conversation memory for query rewriting.
"""
from dataclasses import dataclass, field
from threading import Lock


@dataclass(frozen=True)
class ConversationTurn:
    """Small record of one answered RAG turn."""

    user_query: str
    retrieval_query: str
    answer: str
    citation_chunk_ids: list[str] = field(default_factory=list)


class InMemoryConversationStore:
    """Bounded in-memory store keyed by conversation ID.

    This is intentionally process-local. It keeps only compact recent turns so
    follow-up rewriting has context without growing final answer prompts.
    """

    def __init__(self, max_turns: int = 6, max_answer_chars: int = 600):
        self._max_turns = max_turns
        self._max_answer_chars = max_answer_chars
        self._conversations: dict[str, list[ConversationTurn]] = {}
        self._lock = Lock()

    def get_recent(self, conversation_id: str) -> list[ConversationTurn]:
        """Return a copy of recent turns for a conversation."""
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def append(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Append one compact turn, trimming old turns."""
        compact_turn = ConversationTurn(
            user_query=turn.user_query,
            retrieval_query=turn.retrieval_query,
            answer=self._compact_answer(turn.answer),
            citation_chunk_ids=list(turn.citation_chunk_ids),
        )

        with self._lock:
            turns = self._conversations.setdefault(conversation_id, [])
            turns.append(compact_turn)
            if len(turns) > self._max_turns:
                del turns[:-self._max_turns]

    def clear(self, conversation_id: str) -> None:
        """Delete a conversation if present."""
        with self._lock:
            self._conversations.pop(conversation_id, None)

    def _compact_answer(self, answer: str) -> str:
        normalized = " ".join(answer.split())
        if len(normalized) <= self._max_answer_chars:
            return normalized
        return normalized[: self._max_answer_chars].rstrip()
