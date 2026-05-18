"""DI 컨테이너 — config, VectorDB 등 싱글톤 관리."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"


@lru_cache
def get_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reload_config() -> dict:
    get_config.cache_clear()
    return get_config()


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    get_config.cache_clear()


def get_poll_store(config: dict | None = None):
    if config is None:
        config = get_config()

    from rag.poll_store import create_poll_store

    return create_poll_store(config)


def get_vector_repo(config: dict | None = None):
    if config is None:
        config = get_config()

    import vectordb  # noqa: F401
    from vectordb.base import VectorRepositoryRegistry

    cfg = config.get("vectordb", {})
    repo = VectorRepositoryRegistry.create(
        cfg.get("type", "chroma"),
        collection=cfg.get("collection", "election_chunks"),
        **cfg.get("params", {}),
    )
    repo.load()
    return repo
