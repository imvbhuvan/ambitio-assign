"""Career-stage / PI filters (challenge 6.2: filter out students, postdocs, industry).

Determines whether an author looks like an active Principal Investigator based on:
- last-author share over recently sourced works,
- years active,
- total works count,
- institution type,
and excludes personal-fellowship works from counting as supervision evidence.
"""
from __future__ import annotations

from datetime import datetime

from src import config


def first_pub_year(author: dict) -> int | None:
    """Earliest year in ``counts_by_year`` with works > 0."""
    counts = author.get("counts_by_year") or []
    years = [c["year"] for c in counts if c.get("works_count", 0) > 0]
    return min(years) if years else None


def years_active(author: dict, current_year: int | None = None) -> int:
    cy = current_year or datetime.utcnow().year
    fy = first_pub_year(author)
    return (cy - fy) if fy is not None else 0


def institution_type(author: dict) -> str | None:
    insts = author.get("last_known_institutions") or []
    return insts[0].get("type") if insts else None


def is_personal_fellowship(grant: dict) -> bool:
    """True if an award matches a personal-fellowship pattern (not supervision evidence).

    Accepts both the legacy OpenAlex grant shape (``award_id``) and the current
    award shape (``funder_award_id``).
    """
    haystack = " ".join(
        str(grant.get(k, "")) for k in ("funder_award_id", "award_id", "display_name", "funder_display_name")
    ).upper()
    return any(p.upper() in haystack for p in config.PERSONAL_FELLOWSHIP_PATTERNS)


def author_position_in_work(work: dict, author_id: str) -> str | None:
    """Return this author's authorship position ('first'|'middle'|'last') in a work."""
    for a in work.get("authorships", []):
        aid = (a.get("author") or {}).get("id") or ""
        if aid.rstrip("/").rsplit("/", 1)[-1] == author_id:
            return a.get("author_position")
    return None


def last_author_share(works: list[dict], author_id: str) -> float:
    """Fraction of the given works in which this author is in the 'last' position."""
    if not works:
        return 0.0
    last = sum(1 for w in works if author_position_in_work(w, author_id) == "last")
    return last / len(works)


def passes_career_stage(
    author: dict,
    sourced_works: list[dict],
    author_id: str,
    current_year: int | None = None,
) -> tuple[bool, dict]:
    """Return (passes, derived-metrics). All of the criteria must hold."""
    share = last_author_share(sourced_works, author_id)
    ya = years_active(author, current_year)
    wc = author.get("works_count", 0) or 0
    itype = institution_type(author)

    passes = (
        share >= config.MIN_LAST_AUTHOR_SHARE
        and ya >= config.MIN_YEARS_ACTIVE
        and wc >= config.MIN_WORKS_COUNT
        and itype in config.ALLOWED_INSTITUTION_TYPES
    )
    metrics = {
        "last_author_share": share,
        "years_active": ya,
        "works_count": wc,
        "institution_type": itype,
    }
    return passes, metrics
