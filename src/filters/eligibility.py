"""Challenge 6.4 hook (stub, documented) — eligibility extraction from position ads.

NOT wired into the v1 pipeline (there is no position-ad source yet). Provided so the
trade-off is concrete and the mechanism is testable. See DECISIONS.md §6.4.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src import config
from src.clients.llm import get_chat
from src.prompts import ELIGIBILITY_SYSTEM, eligibility_user
from src.schemas import Eligibility


async def extract_eligibility(ad_text: str) -> Eligibility:
    """Extract structured eligibility constraints from an advertisement's text."""
    model = get_chat(config.MODEL_JUDGE).structured(Eligibility)
    messages = [
        SystemMessage(content=ELIGIBILITY_SYSTEM),
        HumanMessage(content=eligibility_user(ad_text)),
    ]
    return await model.ainvoke(messages)
