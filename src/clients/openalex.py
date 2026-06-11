"""Async, cached, rate-limited OpenAlex client.

Hard rules honoured:
- Works-first sourcing; NEVER author-name search (§14.1).
- ``mailto=`` on every request (polite pool).
- httpx.AsyncClient pooling + tenacity retry (429/5xx/timeouts).
- Global rate limit <= 8 req/s (semaphore + min-interval).
- diskcache layer keyed on full URL (TTL 7 days) for reproducible/fast reruns.
- ``select=`` on every request; cursor pagination, per-page=200.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from diskcache import Cache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure helper: abstract reconstruction (unit-tested, no network)
# --------------------------------------------------------------------------- #
def reconstruct_abstract(inverted_index: dict[str, list[int]] | None, max_chars: int | None = None) -> str:
    """Rebuild plain text from OpenAlex ``abstract_inverted_index``.

    The index maps token -> list of positions. We invert it to position -> token,
    sort by position, and join. Truncates to ``max_chars`` if given.
    """
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, token))
    positions.sort(key=lambda p: p[0])
    text = " ".join(tok for _, tok in positions)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class RateLimiter:
    """Global concurrency cap + minimum inter-request interval."""

    def __init__(self, max_rps: float):
        self._min_interval = 1.0 / max_rps
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._last + self._min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class OpenAlexClient:
    def __init__(self, mailto: str | None = None, use_cache: bool = True):
        self._mailto = mailto or config.require_env("OPENALEX_MAILTO")
        self._client = httpx.AsyncClient(
            base_url=config.OPENALEX_BASE_URL,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            headers={"User-Agent": f"phd-shortlist (mailto:{self._mailto})"},
        )
        self._limiter = RateLimiter(config.OPENALEX_MAX_RPS)
        self._use_cache = use_cache
        self._cache: Cache | None = Cache(str(config.CACHE_DIR)) if use_cache else None

    async def __aenter__(self) -> "OpenAlexClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._cache is not None:
            self._cache.close()

    # ----------------------------------------------------------------- #
    # Core GET: cache -> rate-limit -> retry
    # ----------------------------------------------------------------- #
    def _build_url(self, path: str, params: dict[str, Any]) -> str:
        params = {**params, "mailto": self._mailto}
        # Sort for stable cache keys; drop None values.
        clean = {k: v for k, v in sorted(params.items()) if v is not None}
        return f"{path}?{urlencode(clean, safe=':|,*@.')}"

    @retry(
        retry=retry_if_exception_type(
            (httpx.TransportError, httpx.HTTPStatusError, httpx.TimeoutException)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    async def _get_live(self, url: str) -> dict[str, Any]:
        await self._limiter.acquire()
        resp = await self._client.get(url)
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()

    async def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self._build_url(path, params)
        if self._cache is not None:
            cached = self._cache.get(url)
            if cached is not None:
                return cached
        data = await self._get_live(url)
        if self._cache is not None:
            self._cache.set(url, data, expire=config.CACHE_TTL_SECONDS)
        return data

    async def _paginate(
        self, path: str, params: dict[str, Any], max_results: int
    ) -> list[dict[str, Any]]:
        """Cursor pagination; follows ``meta.next_cursor`` until exhausted or capped."""
        out: list[dict[str, Any]] = []
        cursor = "*"
        while cursor and len(out) < max_results:
            page = await self.get(path, {**params, "cursor": cursor, "per-page": config.PER_PAGE})
            results = page.get("results", [])
            out.extend(results)
            cursor = page.get("meta", {}).get("next_cursor")
            if not results:
                break
        return out[:max_results]

    # ----------------------------------------------------------------- #
    # 6.1 Sourcing — works-first
    # ----------------------------------------------------------------- #
    async def search_works(
        self, query: str, country_codes: list[str], max_results: int
    ) -> list[dict[str, Any]]:
        countries = "|".join(country_codes)
        filt = (
            f"title_and_abstract.search:{query},"
            f"from_publication_date:{config.WORKS_FROM_DATE},"
            f"authorships.countries:{countries},"
            f"type:article"
        )
        # OpenAlex renamed the old `grants` field to `awards` (+ a separate `funders`).
        select = (
            "id,doi,title,publication_year,cited_by_count,authorships,"
            "awards,funders,primary_topic,abstract_inverted_index"
        )
        return await self._paginate(
            "/works",
            {"filter": filt, "select": select, "sort": "cited_by_count:desc"},
            max_results,
        )

    # ----------------------------------------------------------------- #
    # 6.2 Author hydration — batched by ID (max 50/request)
    # ----------------------------------------------------------------- #
    async def hydrate_authors(self, author_ids: list[str]) -> dict[str, dict[str, Any]]:
        select = (
            "id,display_name,orcid,last_known_institutions,topics,counts_by_year,"
            "summary_stats,works_count,affiliations"
        )
        out: dict[str, dict[str, Any]] = {}
        for chunk in _chunk(author_ids, config.OPENALEX_MAX_IDS_PER_REQUEST):
            ids = "|".join(chunk)
            page = await self.get(
                "/authors",
                {"filter": f"ids.openalex:{ids}", "select": select, "per-page": config.PER_PAGE},
            )
            for author in page.get("results", []):
                out[_short_id(author["id"])] = author
        return out

    async def author_recent_works(self, author_id: str, per_page: int = 25) -> list[dict[str, Any]]:
        """One extra call for a borderline author (<3 sourced works) to compute share."""
        filt = f"author.id:{author_id},from_publication_date:{config.WORKS_FROM_DATE}"
        page = await self.get(
            "/works",
            {
                "filter": filt,
                "select": "id,publication_year,authorships,awards",
                "per-page": per_page,
                "sort": "publication_year:desc",
            },
        )
        return page.get("results", [])


def _short_id(openalex_url_or_id: str | None) -> str:
    """Normalize 'https://openalex.org/A5...' -> 'A5...' (idempotent; None -> '')."""
    if not openalex_url_or_id:
        return ""
    return openalex_url_or_id.rstrip("/").rsplit("/", 1)[-1]
