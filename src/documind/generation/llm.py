"""LLM provider clients for answer generation."""

import os
import time
from typing import Protocol

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.genai import types

from documind.config.settings import get_settings


class BaseLLMClient(Protocol):
    """Interface for text generation models."""

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        """Generate a grounded text response given system and user prompts."""
        ...


class GeminiLLMClient:
    """Google Gemini client using the official google-genai SDK with retries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        resolved_key = api_key
        if resolved_key is None and settings.gemini_api_key is not None:
            resolved_key = settings.gemini_api_key.get_secret_value()
        if not resolved_key:
            resolved_key = os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Gemini API key missing. Set DOCUMIND_GEMINI_API_KEY in .env."
            )

        self.model = model or settings.llm_model
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self.max_retries = (
            max_retries if max_retries is not None else settings.max_retries
        )
        self._client = genai.Client(api_key=resolved_key)

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        """Call Gemini with exponential backoff on transient API errors."""
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=config,
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned an empty response.")
                return text
            except genai_errors.APIError as exc:
                last_exc = exc
                if (
                    exc.code not in {429, 500, 502, 503, 504}
                    or attempt >= self.max_retries
                ):
                    raise
                time.sleep(2**attempt)

        raise RuntimeError("Gemini generation failed after retries.") from last_exc


class FakeLLMClient:
    """Deterministic test double that records calls without hitting any network."""

    def __init__(self, canned_response: str = "Mocked answer [1].") -> None:
        self.canned_response = canned_response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        self.calls.append((system_instruction, user_prompt))
        return self.canned_response
