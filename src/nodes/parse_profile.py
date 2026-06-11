"""parse_profile node — 1 call to MODEL_PARSER with structured output (§7.1)."""
from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.clients.llm import get_chat
from src.prompts import PARSER_SYSTEM, parser_user
from src.schemas import ProfileSpec

log = logging.getLogger(__name__)


async def parse_profile(state: dict) -> dict:
    """Convert raw profile JSON into a validated ProfileSpec.

    Input state key: ``profile_raw`` (dict). Output: ``{"spec": ProfileSpec}``.
    """
    raw = state["profile_raw"]
    resume_text = raw.get("resume_text", "")
    intro_call_summary = raw.get("intro_call_summary", "")
    raw_json = json.dumps(raw, ensure_ascii=False, indent=2)

    model = get_chat(config.MODEL_PARSER).structured(ProfileSpec)
    messages = [
        SystemMessage(content=PARSER_SYSTEM),
        HumanMessage(content=parser_user(raw_json, resume_text, intro_call_summary)),
    ]
    spec: ProfileSpec = await model.ainvoke(messages)

    # Validation (§7.1): >=1 area, >=1 country.
    if not spec.research_areas:
        raise ValueError("parse_profile produced zero research areas")
    if not spec.target_countries:
        raise ValueError("parse_profile produced zero target countries")
    # Clamp to 3-5 areas as a guardrail.
    if len(spec.research_areas) > 5:
        log.warning("Parser returned %d areas; truncating to 5", len(spec.research_areas))
        spec.research_areas = spec.research_areas[:5]

    log.info(
        "parse_profile: student=%s areas=%d countries=%s",
        spec.student_id,
        len(spec.research_areas),
        spec.target_countries,
    )
    return {"spec": spec}
