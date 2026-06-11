"""All prompt templates, verbatim from §8 of the spec.

Kept as module-level constants so call sites never inline prompt text.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# 8.1 Profile parser
# --------------------------------------------------------------------------- #
PARSER_SYSTEM = """\
You convert messy student application data into a structured research
profile for PhD supervisor matching. Be faithful to the student's stated
interests; do not invent areas. Countries: ISO 3166-1 alpha-2.
For each research area produce 5-10 specific keywords and 2-3 OpenAlex
title/abstract search queries (3-6 words each, no boolean operators).
profile_summary: 150 words max, third person, covering background, methods
skills, and research direction."""


def parser_user(raw_profile_json: str, resume_text: str, intro_call_summary: str) -> str:
    return (
        f"PROFILE JSON:\n{raw_profile_json}\n\n"
        f"RESUME TEXT:\n{resume_text or '(none provided)'}\n\n"
        f"INTRO CALL SUMMARY:\n{intro_call_summary or '(none provided)'}"
    )


# --------------------------------------------------------------------------- #
# 8.2 Judge (the contamination killer)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """\
You verify whether a researcher is a suitable PhD supervisor match
for a student. You are a skeptical gatekeeper: surfacing a wrong match is
far worse than rejecting a borderline one. Answer "uncertain" whenever the
abstracts do not clearly establish the match.

Classify:
- discipline: the researcher's actual field based on the abstracts (a
  "trauma-informed" project may be literary history, not psychology; "DNA
  barcoding" may be human chromatin methods, not plant biology; "biodegradable
  cartridges" may be munitions R&D, not biomaterials).
- region_of_study: geographic focus OF THE RESEARCH if any (a grant on
  "high-elevation social-ecological systems" may concern the Pacific
  Northwest, not the Himalaya).
- is_active_supervisor: false if the evidence suggests a PhD student,
  postdoc, or industry researcher.
- matches_student_area: "yes" only if discipline AND specific topic AND
  (where relevant) region align with the student's stated area."""


def judge_user(
    area_name: str,
    area_keywords: list[str],
    profile_summary: str,
    display_name: str,
    institution: str,
    abstracts: list[str],
) -> str:
    kw = ", ".join(area_keywords)
    abstract_block = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(abstracts))
    return (
        f"STUDENT AREA: {area_name} — {kw}\n"
        f"STUDENT SUMMARY: {profile_summary}\n"
        f"RESEARCHER: {display_name}, {institution}\n"
        f"RECENT WORK ABSTRACTS:\n{abstract_block}"
    )


# --------------------------------------------------------------------------- #
# 8.3 why_match writer
# --------------------------------------------------------------------------- #
WHY_MATCH_SYSTEM = """\
Write a 2-3 sentence why_match for a PhD cold-email. It must
reference SPECIFIC work by the supervisor (paper title or grant) and connect
it to a SPECIFIC element of the student's background. No generic praise
("renowned", "world-class", "perfect fit" are banned). cited_evidence_ids
must list the source_id of every evidence item you referenced — only IDs
from the provided list."""


def why_match_user(
    profile_summary: str,
    notable_outputs: list[str],
    name: str,
    institution: str,
    evidence_lines: list[str],
    failure_reason: str | None = None,
) -> str:
    outputs = "; ".join(notable_outputs) if notable_outputs else "(none)"
    evidence_block = "\n".join(evidence_lines)
    base = (
        f"STUDENT: {profile_summary} | NOTABLE OUTPUTS: {outputs}\n"
        f"SUPERVISOR: {name}, {institution}\n"
        f"EVIDENCE (id | kind | title | year):\n{evidence_block}"
    )
    if failure_reason:
        base += (
            f"\n\nYOUR PREVIOUS ATTEMPT FAILED VALIDATION: {failure_reason}\n"
            f"Fix it: cite only the source_ids listed above, and quote at least "
            f"one evidence title in the blurb."
        )
    return base


# --------------------------------------------------------------------------- #
# Eligibility extractor (challenge 6.4 hook)
# --------------------------------------------------------------------------- #
ELIGIBILITY_SYSTEM = """\
You extract PhD-position eligibility constraints from a job/position
advertisement. Report only what the text states; do not infer. citizenship_restriction:
list of nationalities/residencies the position is limited to (empty if open to all).
fee_status: e.g. "home", "international", "EU", or null if unstated. notes: any
other eligibility-relevant detail (funding source restrictions, visa conditions)."""


def eligibility_user(ad_text: str) -> str:
    return f"ADVERTISEMENT TEXT:\n{ad_text}"
