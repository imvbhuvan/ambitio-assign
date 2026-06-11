"""OpenAI LLM/embedding factory with config-driven model tiering and fallback.

Call sites ask for a *role* model ID (e.g. ``config.MODEL_JUDGE``); this module
constructs the LangChain chat model and, on a model-unavailable error at first use,
transparently falls back down the chain defined in ``config.MODEL_FALLBACKS``.

A process-wide requests-per-minute limiter (``config.LLM_MAX_RPM``) paces all chat
calls so concurrent area workers share one budget — useful under tier rate limits.
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src import config

log = logging.getLogger(__name__)

# Markers of an error where switching to a different model can help.
_FALLBACK_MARKERS = (
    "404", "not found", "does not exist", "model_not_found", "unsupported", "permission",
)


def _is_model_unavailable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _FALLBACK_MARKERS)


class _RpmLimiter:
    """Process-wide minimum-interval limiter shared across all chat calls.

    Lazily binds to the running event loop. ``LLM_MAX_RPM <= 0`` disables it.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._last = 0.0

    @property
    def _interval(self) -> float:
        return 60.0 / config.LLM_MAX_RPM if config.LLM_MAX_RPM > 0 else 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = time.monotonic()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


_RPM = _RpmLimiter()


def _make_chat(model_id: str) -> ChatOpenAI:
    # Temperature intentionally left at the model default — GPT-5-class models accept
    # only the default temperature, and shape determinism comes from structured output.
    return ChatOpenAI(
        model=model_id,
        api_key=config.require_env("OPENAI_API_KEY"),
        max_retries=2,
    )


class TieredChatModel:
    """A chat model bound to a role that falls back across model IDs on demand."""

    def __init__(self, role_model_id: str):
        self._chain = [role_model_id, *config.MODEL_FALLBACKS.get(role_model_id, [])]

    def structured(self, pydantic_model: type) -> "_StructuredRunnable":
        return _StructuredRunnable(self._chain, pydantic_model)


class _StructuredRunnable:
    def __init__(self, model_chain: list[str], pydantic_model: type):
        self._model_chain = model_chain
        self._pydantic_model = pydantic_model
        self._active_idx = 0

    def _runnable(self) -> Any:
        model_id = self._model_chain[self._active_idx]
        return _make_chat(model_id).with_structured_output(self._pydantic_model)

    def _advance_or_raise(self, exc: Exception) -> None:
        if _is_model_unavailable(exc) and self._active_idx + 1 < len(self._model_chain):
            bad = self._model_chain[self._active_idx]
            self._active_idx += 1
            good = self._model_chain[self._active_idx]
            log.warning("Model %s unavailable (%s); falling back to %s", bad, exc, good)
            return
        raise exc

    async def ainvoke(self, messages: Any) -> Any:
        while True:
            try:
                await _RPM.acquire()
                return await self._runnable().ainvoke(messages)
            except Exception as exc:  # noqa: BLE001 - inspect then re-raise
                self._advance_or_raise(exc)

    async def abatch(self, message_lists: list[Any]) -> list[Any]:
        while True:
            try:
                return await self._runnable().abatch(message_lists)
            except Exception as exc:  # noqa: BLE001
                self._advance_or_raise(exc)


@lru_cache(maxsize=8)
def get_chat(role_model_id: str) -> TieredChatModel:
    return TieredChatModel(role_model_id)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.MODEL_EMBED,
        api_key=config.require_env("OPENAI_API_KEY"),
    )
