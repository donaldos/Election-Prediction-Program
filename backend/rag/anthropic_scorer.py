from __future__ import annotations

import logging
import os

from rag.scorer import AbstractScorer, ScorerRegistry

logger = logging.getLogger(__name__)


@ScorerRegistry.register("anthropic")
class AnthropicScorer(AbstractScorer):

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.1,
        max_tokens: int = 8000,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    def _ensure_client(self) -> None:
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            logger.info("[%s] client initialized — model=%s", self.name, self._model)

    def _call_llm(self, system: str, user: str, *, json_mode: bool = True) -> str:
        self._ensure_client()
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.content[0].text
