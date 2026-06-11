"""Hard country filter (challenge: country adherence = 100%, hard fail).

Applied at source in ``area_worker`` AND re-asserted in ``finalize``.
"""
from __future__ import annotations


def country_of(author: dict) -> str | None:
    """Country code from the author's first last-known institution."""
    insts = author.get("last_known_institutions") or []
    if not insts:
        return None
    return insts[0].get("country_code")


def passes_country(author: dict, target_countries: list[str]) -> bool:
    cc = country_of(author)
    return cc is not None and cc in target_countries
