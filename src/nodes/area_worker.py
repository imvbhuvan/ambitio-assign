"""area_worker — the contamination funnel (§7.2).

Sequential precision-first layers. Each layer logs its drop count; those counts feed
DECISIONS.md and Shortlist.counts. Cheap deterministic filters run before the LLM
judge so the judge only ever sees survivors (hard rule §14.2).
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage

from src import config, runtime
from src.clients.embeddings import area_embedding, author_topic_centroid
from src.clients.llm import get_chat
from src.clients.openalex import _short_id, reconstruct_abstract
from src.filters import career_stage, country, identity
from src.prompts import JUDGE_SYSTEM, judge_user
from src.schemas import EvidenceItem, PICandidate, ProfileSpec, ResearchArea, Verdict

log = logging.getLogger(__name__)


def _doi_url(work: dict) -> str | None:
    doi = work.get("doi")
    if not doi:
        return None
    return doi if doi.startswith("http") else f"https://doi.org/{doi}"


def _candidate_author_ids(work: dict) -> list[str]:
    """Candidate supervisor authors from a work (§6.1):

    last-author always; first-author only when the work carries a non-personal grant.
    """
    has_nonpersonal_grant = any(
        not career_stage.is_personal_fellowship(g) for g in (work.get("awards") or [])
    )
    ids: list[str] = []
    for a in work.get("authorships", []):
        pos = a.get("author_position")
        aid = (a.get("author") or {}).get("id")
        if not aid:
            continue
        if pos == "last" or (pos == "first" and has_nonpersonal_grant):
            ids.append(_short_id(aid))
    return ids


async def area_worker(state: dict) -> dict:
    area: ResearchArea = state["area"]
    spec: ProfileSpec = state["spec"]
    client = runtime.get_client()
    drops: dict[str, int] = defaultdict(int)

    # ----- Layer 1: Source (works-first) -----
    works_by_id: dict[str, dict] = {}
    per_query = max(1, config.WORKS_PER_AREA // max(1, len(area.openalex_queries)))
    for q in area.openalex_queries:
        works = await client.search_works(q, spec.target_countries, per_query)
        for w in works:
            works_by_id[_short_id(w["id"])] = w
    works = list(works_by_id.values())

    # Aggregate works per candidate author.
    author_works: dict[str, list[dict]] = defaultdict(list)
    for w in works:
        for aid in _candidate_author_ids(w):
            author_works[aid].append(w)
    log.info("[%s] sourced %d works -> %d candidate authors", area.name, len(works), len(author_works))

    if not author_works:
        return {"candidates": []}

    # ----- Hydrate authors (batched) -----
    authors = await client.hydrate_authors(list(author_works.keys()))

    # Pre-compute the area embedding once.
    area_emb = await area_embedding(area.name, area.keywords)
    excluded = runtime.excluded_ids()

    survivors: list[dict] = []  # carries author dict + derived metrics + works
    for aid, a_works in author_works.items():
        author = authors.get(aid)
        if author is None:
            drops["no_hydration"] += 1
            continue

        # ----- Layer 2: Country (hard) -----
        if not country.passes_country(author, spec.target_countries):
            drops["country"] += 1
            continue

        # ----- Layer 5 (early, cheap): blacklist/suppression -----
        if aid in excluded:
            drops["blacklist"] += 1
            continue

        # ----- Layer 3: Career stage -----
        sourced = a_works
        if len(sourced) < 3:
            # borderline: one extra works call to compute share properly.
            sourced = await client.author_recent_works(aid) or a_works
        passes_cs, metrics = career_stage.passes_career_stage(author, sourced, aid)
        if not passes_cs:
            drops["career_stage"] += 1
            continue

        survivors.append({"aid": aid, "author": author, "works": a_works, "metrics": metrics})

    # ----- Layer 4: Identity gate (topic overlap) — needs centroids -----
    gated: list[dict] = []
    for s in survivors:
        topic_names = identity.top_topic_names(s["author"], n=5)
        centroid = await author_topic_centroid(topic_names)
        ok, sim = identity.passes_identity_gate(area_emb, centroid, config.TOPIC_SIM_FLOOR)
        if not ok:
            drops["identity_gate"] += 1
            continue
        s["topic_sim"] = sim
        gated.append(s)

    # Latency cap: judge only top-K by topic similarity (§15).
    gated.sort(key=lambda s: s["topic_sim"], reverse=True)
    if len(gated) > config.JUDGE_TOP_K_PER_AREA:
        drops["judge_cap"] += len(gated) - config.JUDGE_TOP_K_PER_AREA
        gated = gated[: config.JUDGE_TOP_K_PER_AREA]

    # ----- Layer 6: LLM judge (batched, concurrency-capped) -----
    judged = await _run_judge(gated, area, spec)

    # ----- Layers 7-8: evidence assembly + provisional score -----
    candidates: list[PICandidate] = []
    for s, verdict in judged:
        if verdict is None:
            drops["judge_error"] += 1
            continue
        if not (verdict.matches_student_area == config.JUDGE_KEEP and verdict.is_active_supervisor):
            drops["judge_reject"] += 1
            continue

        evidence = _assemble_evidence(s["works"], s["aid"])
        if not any(e.kind == "paper" for e in evidence):
            drops["no_evidence"] += 1
            continue

        cand = _build_candidate(s, verdict, evidence, area)
        candidates.append(cand)

    log.info(
        "[%s] funnel: %d authors -> %d candidates | drops=%s",
        area.name,
        len(author_works),
        len(candidates),
        dict(drops),
    )
    # Per-area drop stats flow into Shortlist.counts via the area_drops reducer.
    return {"candidates": candidates, "area_drops": {area.name: dict(drops)}}


async def _run_judge(gated: list[dict], area: ResearchArea, spec: ProfileSpec):
    """Batch the judge with a concurrency cap; return list of (survivor, Verdict|None)."""
    if not gated:
        return []
    model = get_chat(config.MODEL_JUDGE).structured(Verdict)
    sem = asyncio.Semaphore(config.JUDGE_CONCURRENCY)

    async def judge_one(s: dict):
        abstracts = _abstract_snippets(s["works"], n=3)
        messages = [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=judge_user(
                    area.name, area.keywords, spec.profile_summary,
                    s["author"]["display_name"],
                    _institution_name(s["author"]),
                    abstracts,
                )
            ),
        ]
        async with sem:
            try:
                return s, await model.ainvoke(messages)
            except Exception as exc:  # noqa: BLE001 - one bad judge call shouldn't kill the area
                log.warning("judge error for %s: %s", s["aid"], exc)
                return s, None

    return await asyncio.gather(*(judge_one(s) for s in gated))


def _abstract_snippets(works: list[dict], n: int) -> list[str]:
    top = sorted(works, key=lambda w: w.get("cited_by_count", 0), reverse=True)[:n]
    out = []
    for w in top:
        abs = reconstruct_abstract(w.get("abstract_inverted_index"), config.ABSTRACT_TRUNCATE_CHARS)
        out.append(f"{w.get('title', '(untitled)')} ({w.get('publication_year')}) — {abs or 'no abstract'}")
    return out


def _assemble_evidence(works: list[dict], author_id: str) -> list[EvidenceItem]:
    """Top 2-3 sourced works with DOIs as papers + non-personal grants."""
    evidence: list[EvidenceItem] = []
    top = sorted(works, key=lambda w: w.get("cited_by_count", 0), reverse=True)
    for w in top:
        url = _doi_url(w)
        if not url:
            continue  # skip works without DOIs (no linkable evidence)
        evidence.append(
            EvidenceItem(
                kind="paper",
                title=w.get("title") or "(untitled)",
                year=w.get("publication_year"),
                url=url,
                source_id=_short_id(w["id"]),
                venue_or_funder=((w.get("primary_topic") or {}).get("display_name")),
            )
        )
        if len([e for e in evidence if e.kind == "paper"]) >= 3:
            break

    # Awards (grants) from the sourced works (non-personal only), linked to the funded
    # paper DOI. OpenAlex `awards` carry funder_award_id + funder_display_name.
    seen_awards: set[str] = set()
    for w in top:
        url = _doi_url(w)
        for g in w.get("awards") or []:
            if career_stage.is_personal_fellowship(g):
                continue
            award = g.get("funder_award_id") or g.get("funder_display_name") or ""
            if not award or award in seen_awards or not url:
                continue
            seen_awards.add(award)
            funder = g.get("funder_display_name", "Grant")
            evidence.append(
                EvidenceItem(
                    kind="grant",
                    title=f"{funder} {g.get('funder_award_id', '')}".strip(),
                    year=w.get("publication_year"),
                    url=url,
                    source_id=str(award),
                    venue_or_funder=funder,
                )
            )
    return evidence


def _build_candidate(s: dict, verdict: Verdict, evidence: list[EvidenceItem], area: ResearchArea) -> PICandidate:
    author = s["author"]
    metrics = s["metrics"]
    inst = (author.get("last_known_institutions") or [{}])[0]
    stats = author.get("summary_stats") or {}
    cand = PICandidate(
        openalex_author_id=s["aid"],
        display_name=author["display_name"],
        institution=inst.get("display_name", "Unknown"),
        institution_id=_short_id(inst.get("id", "")) if inst.get("id") else "",
        country_code=inst.get("country_code", ""),
        orcid=author.get("orcid"),
        matched_area=area.name,
        matched_areas=[area.name],
        topic_similarity=round(s["topic_sim"], 4),
        last_author_share=round(metrics["last_author_share"], 4),
        years_active=metrics["years_active"],
        works_count=metrics["works_count"],
        h_index=stats.get("h_index"),
        institution_type=metrics["institution_type"],
        institution_mean_citedness=stats.get("2yr_mean_citedness") or stats.get("i10_index"),
        evidence=evidence,
        verdict=verdict,
    )
    _score_candidate(cand)
    return cand


def _score_candidate(cand: PICandidate) -> None:
    """Provisional score (§7.2 step 8); components normalized to [0,1]."""
    from datetime import datetime

    cy = datetime.utcnow().year
    paper_years = [e.year for e in cand.evidence if e.kind == "paper" and e.year]
    recency = (
        sum(1 for y in paper_years if (cy - y) <= 3) / len(paper_years) if paper_years else 0.0
    )
    n_grants = sum(1 for e in cand.evidence if e.kind == "grant")
    n_papers = sum(1 for e in cand.evidence if e.kind == "paper")
    evidence_strength = 1.0 if n_grants >= 1 else (0.6 if n_papers >= 2 else 0.4)
    seniority = min(cand.years_active / 20.0, 1.0)
    topic_sim = max(0.0, min(cand.topic_similarity, 1.0))

    cand.score_topic_sim = topic_sim
    cand.score_recency = recency
    cand.score_evidence = evidence_strength
    cand.score_seniority = seniority
    cand.score = round(
        config.W_TOPIC_SIM * topic_sim
        + config.W_RECENCY * recency
        + config.W_EVIDENCE * evidence_strength
        + config.W_SENIORITY * seniority,
        4,
    )


def _institution_name(author: dict) -> str:
    insts = author.get("last_known_institutions") or []
    return insts[0].get("display_name", "Unknown") if insts else "Unknown"
