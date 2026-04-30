from __future__ import annotations

import logging
import os

from rag.scorer import AbstractScorer, ScorerRegistry

logger = logging.getLogger(__name__)


@ScorerRegistry.register("openai")
class OpenAIScorer(AbstractScorer):

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    @property
    def name(self) -> str:
        return "openai"

    def _ensure_client(self) -> None:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            logger.info("[%s] client initialized — model=%s", self.name, self._model)

    def _call_llm(self, system: str, user: str, *, json_mode: bool = True) -> str:
        self._ensure_client()
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
