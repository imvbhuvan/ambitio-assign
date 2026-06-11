"""All Pydantic v2 models for the system (normative — implement exactly).

Extended only with optional fields beyond §5 of the spec.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #
class ResearchArea(BaseModel):
    name: str
    keywords: list[str] = Field(description="Expanded by parser LLM, 5-10 per area")
    openalex_queries: list[str] = Field(description="2-3 search strings per area")


class ProfileSpec(BaseModel):
    student_id: str
    degrees: list[str]
    research_areas: list[ResearchArea]              # 3-5
    target_countries: list[str]                     # ISO 3166-1 alpha-2, e.g. ["US","GB","AU"]
    target_intake: str                              # e.g. "Fall 2027"
    nationality: str | None = None                  # for eligibility filtering
    profile_summary: str                            # 150-word synthesis for judge/why_match
    notable_outputs: list[str]                      # publications/projects worth citing


# --------------------------------------------------------------------------- #
# Candidate pipeline
# --------------------------------------------------------------------------- #
class EvidenceItem(BaseModel):
    kind: Literal["paper", "grant"]
    title: str
    year: int | None = None
    url: str                                        # DOI URL or funder award URL — REQUIRED
    source_id: str                                  # OpenAlex work ID (W...) or award ID
    venue_or_funder: str | None = None


class Verdict(BaseModel):
    discipline: Literal["stem", "medical", "social_science", "humanities", "other"]
    region_of_study: str | None = None              # geographic focus of the RESEARCH, if any
    is_active_supervisor: bool
    matches_student_area: Literal["yes", "no", "uncertain"]
    # Bounded but generous: weaker judge models (e.g. flash-lite) overrun a tight 300-char
    # cap, which would fail structured-output validation and drop an otherwise-valid verdict.
    reason: str = Field(max_length=1000)


class PICandidate(BaseModel):
    openalex_author_id: str                         # canonical identity — A5XXXXXXXXX
    display_name: str
    institution: str
    institution_id: str
    country_code: str
    orcid: str | None = None
    homepage_or_email: str | None = None            # only if surfaced by source; never guessed
    matched_area: str                               # which ResearchArea sourced this candidate
    matched_areas: list[str] = Field(default_factory=list)  # all areas after dedupe merge
    topic_similarity: float
    last_author_share: float
    years_active: int
    works_count: int
    h_index: int | None = None
    institution_type: str | None = None
    institution_mean_citedness: float | None = None
    evidence: list[EvidenceItem]                    # ≥1 paper required; grants optional
    verdict: Verdict | None = None
    tier: Literal["reach", "target", "safety"] = "target"
    score: float = 0.0
    # score components (kept for feedback-loop feature extraction)
    score_topic_sim: float = 0.0
    score_recency: float = 0.0
    score_evidence: float = 0.0
    score_seniority: float = 0.0


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
class WhyMatch(BaseModel):
    blurb: str = Field(max_length=600)
    cited_evidence_ids: list[str]                   # MUST be subset of candidate's evidence ids


class Recommendation(BaseModel):
    rank: int
    supervisor_id: str                              # OpenAlex author ID
    name: str
    institution: str
    country: str
    contact: str | None
    research_focus: str
    matched_area: str
    tier: Literal["reach", "target", "safety"]
    evidence: list[EvidenceItem]
    why_match: str
    linked_programs: list[str] = []                 # program/position URLs if obtained; empty OK
    score: float


class Shortlist(BaseModel):
    student_id: str
    generated_at: str                               # ISO 8601
    target_countries: list[str]
    pipeline_version: str
    counts: dict                                    # per-area counts + totals + drop statistics
    recommendations: list[Recommendation]


# --------------------------------------------------------------------------- #
# Eligibility (challenge 6.4 hook — not wired into v1 pipeline)
# --------------------------------------------------------------------------- #
class Eligibility(BaseModel):
    citizenship_restriction: list[str] = Field(default_factory=list)
    fee_status: str | None = None
    notes: str = ""
