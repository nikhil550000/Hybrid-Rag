"""Shared tokenization helpers for lexical retrieval."""
import re


# Keep compound research terms together while dropping surrounding punctuation.
# Examples: ``bert-base``, ``gpt-4``, ``p-value``, ``f1``, and ``3.14``.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 while preserving useful technical terms."""
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]
