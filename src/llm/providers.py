"""Concrete LLM provider implementations.

Implements: HLD 3.9
Satisfies: FR-19
"""
import time

import anthropic
import google.generativeai as genai

from llm.client import LLMResponse
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 2, 4]


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
                return LLMResponse(
                    text=response.content[0].text,
                    tokens_input=response.usage.input_tokens,
                    tokens_output=response.usage.output_tokens,
                    model=self._model,
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
    """Google Gemini client with retry logic."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.0):
        self._model_name = model
        self._temperature = temperature
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call Google Gemini API with retry (3 attempts, exponential backoff)."""
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        for attempt in range(MAX_RETRIES):
            try:
                response = self._model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self._temperature,
                        max_output_tokens=2048,
                    ),
                )
                # Estimate token counts (Gemini doesn't always return exact counts)
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
