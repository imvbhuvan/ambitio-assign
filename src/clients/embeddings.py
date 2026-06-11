"""Disk-cached topic/area embeddings via MODEL_EMBED.

Embeddings are cached on disk keyed by the exact text string (topic display name or
area query), so reruns are reproducible and cheap. Author topic centroids are the
mean of their top-5 topic embeddings.
"""
from __future__ import annotations

import logging

import numpy as np
from diskcache import Cache

from src import config
from src.clients.llm import get_embeddings

log = logging.getLogger(__name__)

_EMBED_CACHE = Cache(str(config.CACHE_DIR.parent / "embeddings"))


def _cache_key(text: str) -> str:
    return f"{config.MODEL_EMBED}::{text}"


async def embed_texts(texts: list[str]) -> dict[str, list[float]]:
    """Return {text: embedding}. Only uncached texts hit the API (batched)."""
    out: dict[str, list[float]] = {}
    missing: list[str] = []
    for t in texts:
        cached = _EMBED_CACHE.get(_cache_key(t))
        if cached is not None:
            out[t] = cached
        else:
            missing.append(t)

    # de-dup missing while preserving order
    missing = list(dict.fromkeys(missing))
    if missing:
        vectors = await get_embeddings().aembed_documents(missing)
        for t, v in zip(missing, vectors):
            _EMBED_CACHE.set(_cache_key(t), v, expire=config.CACHE_TTL_SECONDS)
            out[t] = v
    return out


async def embed_text(text: str) -> list[float]:
    return (await embed_texts([text]))[text]


async def area_embedding(area_name: str, keywords: list[str]) -> list[float]:
    """Embedding of an area, represented as its name + keywords."""
    text = f"{area_name}: {', '.join(keywords)}"
    return await embed_text(text)


async def author_topic_centroid(topic_names: list[str]) -> list[float] | None:
    """Mean embedding of an author's top topic display names."""
    if not topic_names:
        return None
    embs = await embed_texts(topic_names)
    vecs = [embs[t] for t in topic_names if t in embs]
    if not vecs:
        return None
    return list(np.mean(np.asarray(vecs, dtype=float), axis=0))
