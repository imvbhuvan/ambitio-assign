"""finalize (§7.5) — output assertions, then JSON serialization.

Assertions raise before any file is written: a country violation or empty/ungrounded
output crashes rather than emitting (hard rules §14.3, §14.6). Too-few recs exits with
a clear, actionable message.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src import PIPELINE_VERSION, config
from src.schemas import ProfileSpec, Recommendation, Shortlist

log = logging.getLogger(__name__)


class CoverageError(RuntimeError):
    """Raised when fewer than MIN_RECS survive the funnel."""


def _build_counts(state: dict, recs: list[Recommendation], spec: ProfileSpec) -> dict:
    per_area = Counter(r.matched_area for r in recs)
    tiers = Counter(r.tier for r in recs)
    return {
        "total_recommendations": len(recs),
        "per_area": {a.name: per_area.get(a.name, 0) for a in spec.research_areas},
        "per_tier": dict(tiers),
        "area_drops": state.get("area_drops", {}),
        "coverage_warnings": state.get("coverage_warnings", {}),
        "why_match_dropped": state.get("why_match_dropped", 0),
    }


def finalize(state: dict) -> dict:
    spec: ProfileSpec = state["spec"]
    recs: list[Recommendation] = list(state.get("recommendations", []))

    # Final global sort + rank assignment.
    recs.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(recs, start=1):
        r.rank = i

    # ----- Assertions (crash, don't emit) -----
    violations = [r for r in recs if r.country not in spec.target_countries]
    assert not violations, (
        f"COUNTRY VIOLATION: {len(violations)} recs outside {spec.target_countries}: "
        f"{[(r.name, r.country) for r in violations[:5]]}"
    )
    for r in recs:
        assert r.evidence and any(e.url for e in r.evidence), f"rec {r.supervisor_id} has no evidence URL"
        assert r.why_match.strip(), f"rec {r.supervisor_id} has empty why_match"

    if len(recs) < config.MIN_RECS:
        raise CoverageError(
            f"Only {len(recs)} recommendations survived (need >= {config.MIN_RECS}). "
            f"A funnel layer was too aggressive. Drop stats: {state.get('area_drops', {})}. "
            f"Suggestion: lower TOPIC_SIM_FLOOR (currently {config.TOPIC_SIM_FLOOR}) "
            f"or MIN_LAST_AUTHOR_SHARE, or widen WORKS_PER_AREA."
        )
    assert len(recs) <= config.MAX_RECS, f"{len(recs)} exceeds MAX_RECS {config.MAX_RECS}"

    counts = _build_counts(state, recs, spec)
    shortlist = Shortlist(
        student_id=spec.student_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        target_countries=spec.target_countries,
        pipeline_version=PIPELINE_VERSION,
        counts=counts,
        recommendations=recs,
    )

    out_dir = Path(state.get("output_path") or config.SAMPLE_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{spec.student_id}.json"
    out_file.write_text(
        shortlist.model_dump_json(indent=2),
        encoding="utf-8",
    )
    log.info("finalize: wrote %d recommendations -> %s", len(recs), out_file)
    return {"output_file": str(out_file), "shortlist": shortlist, "counts": counts}
