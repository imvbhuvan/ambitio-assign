"""LangGraph StateGraph wiring (§7)."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.nodes.area_worker import area_worker
from src.nodes.finalize import finalize
from src.nodes.parse_profile import parse_profile
from src.nodes.rank_and_tier import rank_and_tier
from src.nodes.why_match import generate_why_match
from src.schemas import PICandidate, ProfileSpec, Recommendation, Shortlist


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer for fan-in dict channels (per-area drop stats keyed by area name)."""
    return {**(a or {}), **(b or {})}


class ShortlistState(TypedDict, total=False):
    profile_raw: dict
    spec: ProfileSpec
    candidates: Annotated[list[PICandidate], operator.add]   # fan-in reducer
    area_drops: Annotated[dict, _merge_dicts]                # per-area funnel drop stats
    ranked_candidates: list[PICandidate]
    coverage_warnings: dict
    recommendations: list[Recommendation]
    why_match_dropped: int
    output_path: str
    output_file: str
    counts: dict
    shortlist: Shortlist


class AreaWorkerState(TypedDict):     # private Send-branch state
    area: object
    spec: ProfileSpec


def fan_out(state: ShortlistState):
    return [
        Send("area_worker", {"area": a, "spec": state["spec"]})
        for a in state["spec"].research_areas
    ]


def build_graph():
    g = StateGraph(ShortlistState)
    g.add_node("parse_profile", parse_profile)
    g.add_node("area_worker", area_worker)
    g.add_node("rank_and_tier", rank_and_tier)
    g.add_node("generate_why_match", generate_why_match)
    g.add_node("finalize", finalize)
    g.add_edge(START, "parse_profile")
    g.add_conditional_edges("parse_profile", fan_out, ["area_worker"])
    g.add_edge("area_worker", "rank_and_tier")    # fan-in via operator.add reducer
    g.add_edge("rank_and_tier", "generate_why_match")
    g.add_edge("generate_why_match", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
