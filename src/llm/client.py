"""LLM client protocol and factory.

Implements: HLD 3.9 (LLM call)
Satisfies: FR-19, NFR-07
"""
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """Raw response from an LLM provider."""
    text: str
    tokens_input: int
    tokens_output: int
    model: str


class LLMClient(Protocol):
    """All LLM providers implement this interface."""
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


def get_llm_client(provider: str, model: str, temperature: float) -> LLMClient:
    """
    Factory — returns the right client based on settings.yaml provider value.

    Raises:
        ValueError: If provider is not "anthropic" or "google"
    """
    from llm.providers import AnthropicClient, GoogleClient

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        return AnthropicClient(model=model, api_key=api_key, temperature=temperature)
    elif provider == "google":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        return GoogleClient(model=model, api_key=api_key, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Must be 'anthropic' or 'google'")
