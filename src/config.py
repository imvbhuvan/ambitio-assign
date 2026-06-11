"""Central configuration.

ALL tunables (model IDs, thresholds, weights, paths) live here — never inline at
call sites (hard rule §14.8). Learned weights from the feedback loop, if present at
``.data/weights.json``, override the hand-tuned ranking weights at import time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "openalex"
DATA_DIR = ROOT / ".data"
SAMPLE_OUTPUT_DIR = ROOT / "sample_output"
WEIGHTS_PATH = DATA_DIR / "weights.json"
FEEDBACK_DB_PATH = DATA_DIR / "feedback.db"

# --------------------------------------------------------------------------- #
# OpenAI model tiering (config-driven, never hardcoded at call sites).
# Fallback chains are tried in order on model-unavailable (404 / not found) errors.
# --------------------------------------------------------------------------- #
# All chat roles use GPT-5.4 Mini. (Override any role via env var, e.g. MODEL_PARSER.)
MODEL_PARSER = os.environ.get("MODEL_PARSER", "gpt-5.4-mini")
MODEL_JUDGE = os.environ.get("MODEL_JUDGE", "gpt-5.4-mini")
MODEL_WRITER = os.environ.get("MODEL_WRITER", "gpt-5.4-mini")
MODEL_EMBED = os.environ.get("MODEL_EMBED", "text-embedding-3-small")

# Optional per-role fallback chains (model IDs tried in order). Empty by default.
MODEL_FALLBACKS: dict[str, list[str]] = {}

# Global requests-per-minute cap across ALL chat calls (judge + writer + parser),
# enforced process-wide so concurrent area workers share one budget. 0 = unlimited.
# Set LLM_MAX_RPM to pace under a tier rate limit; paid tiers can leave it at 0.
LLM_MAX_RPM = _env_int("LLM_MAX_RPM", 0)

# --------------------------------------------------------------------------- #
# Sourcing
# --------------------------------------------------------------------------- #
WORKS_FROM_DATE = "2021-01-01"      # recency window for sourcing works
WORKS_PER_AREA = _env_int("WORKS_PER_AREA", 600)  # max works pulled per research area
QUERIES_PER_AREA = 3                # LLM-expanded keyword queries per area
PER_PAGE = 200
OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_MAX_RPS = 8.0              # global rate limit, under OpenAlex's 10/s polite-pool
OPENALEX_MAX_IDS_PER_REQUEST = 50   # author hydration batch size
CACHE_TTL_SECONDS = 7 * 24 * 3600   # 7 days

# --------------------------------------------------------------------------- #
# Deterministic PI filters (career stage / identity)
# --------------------------------------------------------------------------- #
MIN_LAST_AUTHOR_SHARE = 0.40        # share of recent works as last author
MIN_YEARS_ACTIVE = 6                # current_year - first_publication_year
MIN_WORKS_COUNT = 15
ALLOWED_INSTITUTION_TYPES = {"education", "healthcare", "facility"}
# Cosine floor before a candidate reaches the judge. Calibrated for OpenAI
# text-embedding-3-small, whose topic/area cosines run lower than the original spec's
# 0.55 assumption. This is a cheap PRE-filter to save judge calls — the LLM judge is the
# real domain gate — so a lower floor trades a few more judge calls for better recall,
# not contamination. Env-overridable via TOPIC_SIM_FLOOR.
TOPIC_SIM_FLOOR = float(os.environ.get("TOPIC_SIM_FLOOR", "0.40"))
PERSONAL_FELLOWSHIP_PATTERNS = ["F31", "F32", "MSCA-PF", "studentship"]  # not supervision evidence

# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
JUDGE_CONCURRENCY = _env_int("JUDGE_CONCURRENCY", 20)
JUDGE_KEEP = "yes"                  # drop "no" AND "uncertain" — precision-first
JUDGE_TOP_K_PER_AREA = _env_int("JUDGE_TOP_K_PER_AREA", 80)  # latency cap: judge top-K/area
ABSTRACT_TRUNCATE_CHARS = 1200

# --------------------------------------------------------------------------- #
# Ranking weights (hand-tuned prior; feedback loop may override)
# --------------------------------------------------------------------------- #
W_TOPIC_SIM = 0.40
W_RECENCY = 0.20
W_EVIDENCE = 0.25                   # grant + papers > papers only
W_SENIORITY = 0.15

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
MIN_RECS = _env_int("MIN_RECS", 50)
MAX_RECS = _env_int("MAX_RECS", 200)
MIN_PER_AREA = _env_int("MIN_PER_AREA", 8)  # coverage guarantee per stated research area
TOP_GRANT_ENRICH = 30               # lazily enrich grants for top-N only
WHY_MATCH_CONCURRENCY = _env_int("WHY_MATCH_CONCURRENCY", 20)

# --------------------------------------------------------------------------- #
# Learned-weight override (feedback loop bonus). Loaded once at import.
# --------------------------------------------------------------------------- #
def _load_learned_weights() -> None:
    """If ``.data/weights.json`` exists, override the hand-tuned ranking weights."""
    global W_TOPIC_SIM, W_RECENCY, W_EVIDENCE, W_SENIORITY
    if not WEIGHTS_PATH.exists():
        return
    try:
        data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    W_TOPIC_SIM = float(data.get("W_TOPIC_SIM", W_TOPIC_SIM))
    W_RECENCY = float(data.get("W_RECENCY", W_RECENCY))
    W_EVIDENCE = float(data.get("W_EVIDENCE", W_EVIDENCE))
    W_SENIORITY = float(data.get("W_SENIORITY", W_SENIORITY))


_load_learned_weights()


def require_env(name: str) -> str:
    """Fetch a required env var or raise a clear error."""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return val
