"""Concrete LLM provider implementations.

Implements: HLD 3.9
Satisfies: FR-19
"""
import time

import anthropic
from google import genai

from llm.client import LLMResponse
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]

# Pricing per million tokens (USD) — update when switching models
MODEL_PRICING = {
    # Anthropic
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-0": {"input": 3.00, "output": 15.00},
    "claude-opus-4-0": {"input": 15.00, "output": 75.00},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro-preview-05-06": {"input": 1.25, "output": 10.00},
}


def _compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Compute cost in USD from token counts and model pricing."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    return (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000


class AnthropicClient:
    """Anthropic Claude client with retry logic."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.0):
        self._model = model
        self._temperature = temperature
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call Anthropic API with retry (3 attempts, exponential backoff)."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=2048,
                    temperature=self._temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                tokens_in = response.usage.input_tokens
                tokens_out = response.usage.output_tokens
                return LLMResponse(
                    text=response.content[0].text,
                    tokens_input=tokens_in,
                    tokens_output=tokens_out,
                    model=self._model,
                    cost_usd=_compute_cost(self._model, tokens_in, tokens_out),
                )
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_SECONDS[attempt]
                    logger.warning(
                        f"Anthropic API error (attempt {attempt + 1}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM API unavailable after {MAX_RETRIES} retries: {e}"
                    ) from e


class GoogleClient:
    """Google Gemini client using the new google-genai SDK with retry logic."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.0):
        self._model_name = model
        self._temperature = temperature
        self._client = genai.Client(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call Google Gemini API with retry (3 attempts, exponential backoff)."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=full_prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=self._temperature,
                        max_output_tokens=2048,
                    ),
                )
                # Extract token counts from usage metadata
                tokens_in = (
                    response.usage_metadata.prompt_token_count
                    if response.usage_metadata
                    else 0
                )
                tokens_out = (
                    response.usage_metadata.candidates_token_count
                    if response.usage_metadata
                    else 0
                )

                return LLMResponse(
                    text=response.text,
                    tokens_input=tokens_in,
                    tokens_output=tokens_out,
                    model=self._model_name,
                    cost_usd=_compute_cost(self._model_name, tokens_in, tokens_out),
                )
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_SECONDS[attempt]
                    logger.warning(
                        f"Google API error (attempt {attempt + 1}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LLM API unavailable after {MAX_RETRIES} retries: {e}"
                    ) from e

