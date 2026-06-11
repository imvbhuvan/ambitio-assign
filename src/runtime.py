"""Process-wide runtime context shared across parallel LangGraph Send branches.

The OpenAlex client is a singleton so its rate limiter (<= 8 req/s) is *global*
across all concurrent area workers, and so the diskcache + connection pool are shared.
run.py initializes this before invoking the graph and closes it afterwards.
"""
from __future__ import annotations

from src.clients.openalex import OpenAlexClient
from src.feedback import store

_client: OpenAlexClient | None = None
_excluded_ids: set[str] | None = None


def init(use_cache: bool = True) -> None:
    global _client, _excluded_ids
    _client = OpenAlexClient(use_cache=use_cache)
    _excluded_ids = store.excluded_ids()


def get_client() -> OpenAlexClient:
    if _client is None:
        raise RuntimeError("runtime.init() was not called before use")
    return _client


def excluded_ids() -> set[str]:
    return _excluded_ids or set()


async def shutdown() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
