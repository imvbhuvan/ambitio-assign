"""generate_why_match (§7.4) — grounded blurb generation with a hard grounding guard.

A hallucinated why_match is worse than a missing entry: on guard failure we retry ONCE
with the failure reason appended, then DROP the recommendation (hard rule §14.6).
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.clients.llm import get_chat
from src.prompts import WHY_MATCH_SYSTEM, why_match_user
from src.schemas import PICandidate, ProfileSpec, Recommendation, WhyMatch

log = logging.getLogger(__name__)


def grounded(wm: WhyMatch, evidence_ids: set[str]) -> bool:
    return bool(wm.cited_evidence_ids) and set(wm.cited_evidence_ids) <= evidence_ids


def _has_title_mention(blurb: str, titles: list[str]) -> bool:
    """True if the blurb contains the exact title or >=5 consecutive words from one."""
    low = blurb.lower()
    for title in titles:
        t = (title or "").lower()
        if not t:
            continue
        if t in low:
            return True
        words = t.split()
        for i in range(0, max(0, len(words) - 4)):
            window = " ".join(words[i : i + 5])
            if window and window in low:
                return True
    return False


def _validate(wm: WhyMatch, cand: PICandidate) -> tuple[bool, str]:
    evidence_ids = {e.source_id for e in cand.evidence}
    if not grounded(wm, evidence_ids):
        return False, "cited_evidence_ids must be a non-empty subset of the provided source_ids"
    cited_titles = [e.title for e in cand.evidence if e.source_id in set(wm.cited_evidence_ids)]
    if not _has_title_mention(wm.blurb, cited_titles):
        return False, "blurb must quote at least one cited evidence title"
    return True, ""


def _evidence_lines(cand: PICandidate) -> list[str]:
    return [
        f"{e.source_id} | {e.kind} | {e.title} | {e.year or 'n/a'}"
        for e in cand.evidence
    ]


def _messages(cand: PICandidate, spec: ProfileSpec, failure_reason: str | None):
    return [
        SystemMessage(content=WHY_MATCH_SYSTEM),
        HumanMessage(
            content=why_match_user(
                spec.profile_summary,
                spec.notable_outputs,
                cand.display_name,
                cand.institution,
                _evidence_lines(cand),
                failure_reason,
            )
        ),
    ]


async def generate_why_match(state: dict) -> dict:
    spec: ProfileSpec = state["spec"]
    candidates: list[PICandidate] = state["ranked_candidates"]
    model = get_chat(config.MODEL_WRITER).structured(WhyMatch)
    sem = asyncio.Semaphore(config.WHY_MATCH_CONCURRENCY)

    async def write_one(cand: PICandidate) -> tuple[PICandidate, WhyMatch | None]:
        async with sem:
            try:
                wm = await model.ainvoke(_messages(cand, spec, None))
            except Exception as exc:  # noqa: BLE001
                log.warning("why_match error for %s: %s", cand.openalex_author_id, exc)
                return cand, None
            ok, reason = _validate(wm, cand)
            if ok:
                return cand, wm
            # ONE retry with failure reason appended.
            try:
                wm2 = await model.ainvoke(_messages(cand, spec, reason))
            except Exception as exc:  # noqa: BLE001
                log.warning("why_match retry error for %s: %s", cand.openalex_author_id, exc)
                return cand, None
            ok2, _ = _validate(wm2, cand)
            return (cand, wm2) if ok2 else (cand, None)

    results = await asyncio.gather(*(write_one(c) for c in candidates))

    recs: list[Recommendation] = []
    dropped = 0
    for cand, wm in results:
        if wm is None:
            dropped += 1
            continue
        recs.append(_to_recommendation(cand, wm))

    log.info("generate_why_match: %d recs, %d dropped (ungrounded)", len(recs), dropped)
    return {"recommendations": recs, "why_match_dropped": dropped}


def _to_recommendation(cand: PICandidate, wm: WhyMatch) -> Recommendation:
    focus = cand.verdict.discipline if cand.verdict else ""
    region = cand.verdict.region_of_study if cand.verdict else None
    research_focus = f"{focus}{f' — {region}' if region else ''}".strip(" —") or cand.matched_area
    return Recommendation(
        rank=0,  # assigned in finalize after final sort
        supervisor_id=cand.openalex_author_id,
        name=cand.display_name,
        institution=cand.institution,
        country=cand.country_code,
        contact=cand.homepage_or_email,
        research_focus=research_focus,
        matched_area=cand.matched_area,
        tier=cand.tier,
        evidence=cand.evidence,
        why_match=wm.blurb,
        linked_programs=[],
        score=cand.score,
    )
