"""rank_and_tier (§7.3): dedupe by author ID, global sort, coverage guarantee, tiering."""
from __future__ import annotations

import logging

from src import config
from src.schemas import PICandidate, ProfileSpec

log = logging.getLogger(__name__)


def _dedupe(candidates: list[PICandidate]) -> list[PICandidate]:
    """Keep the highest-scoring instance per author ID; merge matched_area provenance."""
    best: dict[str, PICandidate] = {}
    for c in candidates:
        prev = best.get(c.openalex_author_id)
        if prev is None:
            best[c.openalex_author_id] = c
            continue
        merged_areas = sorted(set(prev.matched_areas) | set(c.matched_areas))
        winner = c if c.score > prev.score else prev
        winner.matched_areas = merged_areas
        best[c.openalex_author_id] = winner
    return list(best.values())


def _assign_tiers(candidates: list[PICandidate]) -> None:
    """Citation-tercile tiering by institution mean citedness (simplified; see DECISIONS)."""
    if not candidates:
        return
    ranked = sorted(
        candidates,
        key=lambda c: (c.institution_mean_citedness or 0.0),
        reverse=True,
    )
    n = len(ranked)
    third = max(1, n // 3)
    for i, c in enumerate(ranked):
        if i < third:
            c.tier = "reach"
        elif i < 2 * third:
            c.tier = "target"
        else:
            c.tier = "safety"


def rank_and_tier(state: dict) -> dict:
    spec: ProfileSpec = state["spec"]
    candidates: list[PICandidate] = _dedupe(state.get("candidates", []))

    candidates.sort(key=lambda c: c.score, reverse=True)

    # Coverage guarantee: ensure MIN_PER_AREA per stated area by promotion.
    area_names = [a.name for a in spec.research_areas]
    selected: list[PICandidate] = candidates[: config.MAX_RECS]
    selected_ids = {c.openalex_author_id for c in selected}
    coverage_warnings: dict[str, int] = {}

    for area in area_names:
        in_area = [c for c in selected if area in c.matched_areas]
        if len(in_area) >= config.MIN_PER_AREA:
            continue
        # Promote next-best judged candidates for this area not already selected.
        pool = [
            c for c in candidates
            if area in c.matched_areas and c.openalex_author_id not in selected_ids
        ]
        need = config.MIN_PER_AREA - len(in_area)
        promoted = pool[:need]
        for c in promoted:
            selected.append(c)
            selected_ids.add(c.openalex_author_id)
        still_short = config.MIN_PER_AREA - (len(in_area) + len(promoted))
        if still_short > 0:
            coverage_warnings[area] = still_short
            log.warning("coverage: area %r short by %d (no relaxation of judge)", area, still_short)

    # Re-truncate (promotions may exceed MAX_RECS) and re-sort.
    selected.sort(key=lambda c: c.score, reverse=True)
    selected = selected[: config.MAX_RECS]

    _assign_tiers(selected)

    log.info("rank_and_tier: %d unique candidates -> %d selected", len(candidates), len(selected))
    return {
        "ranked_candidates": selected,
        "coverage_warnings": coverage_warnings,
    }
