from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class ComponentRegistry(Generic[T]):
    """Strategy 패턴용 범용 레지스트리. Scraper / Chunker / Embedder 공용."""

    def __init__(self, base_class: type[T], label: str) -> None:
        self._base_class = base_class
        self._label = label
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str):
        """클래스 데코레이터. config.yaml의 type 값과 name을 일치시켜 등록."""

        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                logger.warning(
                    "[%s] '%s' 이미 등록됨 — %s → %s 로 덮어씀",
                    self._label,
                    name,
                    self._registry[name].__name__,
                    cls.__name__,
                )
            self._registry[name] = cls
            return cls

        return decorator

    def create(self, name: str, **kwargs: Any) -> T:
        """등록된 이름으로 인스턴스를 생성."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise ValueError(
                f"[{self._label}] '{name}' 미등록. 사용 가능: [{available}]"
            )
        cls = self._registry[name]
        return cls(**kwargs)

    @property
    def registered_names(self) -> list[str]:
        return list(self._registry.keys())
