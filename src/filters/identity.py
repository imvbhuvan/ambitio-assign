"""Identity / topic-overlap gate (challenge 6.1: same-name collisions).

Identity is anchored on OpenAlex author IDs throughout — name strings are display
only. This gate additionally requires the author's research *topics* to overlap the
student's research area above a cosine floor, so an off-topic same-name author (e.g.
a materials scientist sharing a name with an ML researcher) is dropped before judging.
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def top_topic_names(author: dict, n: int = 5) -> list[str]:
    """Top-N OpenAlex topic display names for an author (already sorted by count)."""
    topics = author.get("topics") or []
    names = [t.get("display_name") for t in topics[:n] if t.get("display_name")]
    return names


def passes_identity_gate(
    area_embedding: list[float],
    author_topic_centroid: list[float] | None,
    floor: float,
) -> tuple[bool, float]:
    if not author_topic_centroid:
        return False, 0.0
    sim = cosine_similarity(area_embedding, author_topic_centroid)
    return sim >= floor, sim
