# PhD Shortlist Builder

Ingests a student profile and produces a ranked shortlist of 50–200 PhD supervisor
recommendations, each with paper/grant evidence and a personalized, grounded
`why_match` blurb. Built as a precision-first contamination funnel: cheap deterministic
filters eliminate ~80% of bad candidates before any LLM call, and a skeptical
OpenAI judge (GPT-5.4 Mini) gates everything that remains.

## What it optimizes for

Contamination (wrong-person / wrong-domain / non-PI) is weighted **heavier than
coverage**. When uncertain, the pipeline drops the candidate. Country adherence is a
hard constraint — a violation crashes the run rather than emitting it.

## Quick start

```bash
# 1. Python 3.11+ and a virtual environment
python -m venv .venv && .venv\Scripts\activate      # Windows
# source .venv/bin/activate                          # macOS/Linux
pip install -r requirements.txt

# 2. Configure secrets
copy .env.example .env        # Windows  (cp on *nix)
#   OPENAI_API_KEY  = your OpenAI API key
#   OPENALEX_MAILTO = any valid email (OpenAlex polite pool)

# 3. Run end-to-end
python run.py --profile sample/student_profile.json
```

Output lands at `sample_output/{student_id}.json`. A run-summary table (per-layer
drop counts, per-area counts, wall-clock) prints to stdout. See `schema.md` for the
output format.

### Options

```
python run.py --profile <path> [--output <dir>] [--no-cache]
python run.py ingest-feedback --csv outcomes.csv     # feedback loop (bonus)
```

- `--no-cache` bypasses the OpenAlex disk cache (`.cache/openalex/`).
- With a warm cache, reruns are fast and the **candidate set + ranks are stable**.
  (LLM-written `why_match` *text* may vary slightly between runs since structured
  generation is non-deterministic in wording; the candidate IDs, scores, and ranks
  are deterministic given the cached API responses.)

## Data sources

- **[OpenAlex](https://openalex.org)** — works, authors, topics, grants. Queried
  works-first (never by author name) and accessed via the polite pool (`mailto=`).
- **[OpenAI](https://platform.openai.com)** — profile parsing, domain/identity
  judging, `why_match` writing (all **GPT-5.4 Mini**), and topic embeddings
  (`text-embedding-3-small`). Model IDs are configured in `src/config.py` (env-overridable);
  an optional per-role fallback chain handles model-unavailable errors.

## Architecture — the contamination funnel

A LangGraph `StateGraph` (`src/graph.py`):

```
parse_profile ──► fan-out (Send per research area) ──► area_worker × N
                                                          │ (operator.add fan-in)
                                                          ▼
                              rank_and_tier ──► generate_why_match ──► finalize
```

Each `area_worker` runs the precision-first funnel, logging drop counts per layer:

1. **Source** — works-first OpenAlex search (title/abstract), country-filtered at the
   query, then aggregate to candidate author IDs (last-author, or first-author on a
   non-personal grant). Identity is anchored on `A5…` IDs throughout.
2. **Country (hard)** — keep only authors at an institution in `target_countries`.
3. **Career stage** — require last-author share ≥ 0.40, ≥ 6 years active, ≥ 15 works,
   and an allowed institution type; exclude personal-fellowship works (F31/F32/MSCA-PF)
   as supervision evidence. Filters out students/postdocs/industry.
4. **Identity gate** — cosine(area embedding, author topic centroid) ≥ 0.55. Drops
   off-topic same-name authors.
5. **Blacklist join** — drop IDs the feedback store marks WRONG_PERSON/BOUNCE or
   suppresses (NOT_RECRUITING within TTL).
6. **LLM judge** — a skeptical GPT-5.4 Mini gatekeeper classifies discipline, region of
   study, active-supervisor status, and area match. Keeps **only** `matches=="yes"`
   AND `is_active_supervisor`. Drops "uncertain".
7. **Evidence assembly** — attach top papers (DOI required) + non-personal grants.
8. **Score** — weighted topic-similarity / recency / evidence-strength / seniority.

`rank_and_tier` dedupes by author ID, enforces a per-area coverage minimum, and assigns
reach/target/safety tiers by institution citation tercile. `generate_why_match`
produces grounded blurbs behind a guard (drops any blurb that cites evidence outside
the candidate's set or fails to quote a title). `finalize` asserts country adherence,
evidence presence, and minimum count before writing.

## Known limitations

- **No position-ad ingestion in v1.** Eligibility (citizenship/fee status) is
  implemented as a tested hook (`src/filters/eligibility.py`) but not wired in — there
  is no ad source yet. See `DECISIONS.md` §6.4.
- **Contact coverage is limited** to what OpenAlex/ORCID surface. Emails are never
  guessed; `contact` is usually null.
- **Tiering is a simplification** — institution citation terciles within the shortlist,
  not admissions selectivity. Defensible and transparent; noted in `DECISIONS.md`.
- **Grant enrichment is shallow** in v1: grants come from the sourced works' `grants`
  field, linked to the funded paper's DOI. No funder-site scraping.

## Tests

```bash
pytest          # 28 tests, no live network — fixtures only
```

Covers the trap cases (fresh postdoc, F31 fellowship, foreign country, same-name
collision), the grounding guard, the OpenAlex client (abstract reconstruction, 50-ID
batching, cursor pagination), the eligibility extractor, the feedback store, and a full
end-to-end smoke test with OpenAlex + the LLM stubbed to deterministic fakes.
