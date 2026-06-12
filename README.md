# PhD Shortlist Builder

Ingests a student profile JSON and produces a ranked shortlist of 50–200 PhD supervisor
recommendations, each with paper/grant evidence and a personalized, grounded
`why_match` blurb. Built as a precision-first contamination funnel: cheap deterministic
filters eliminate ~80% of bad candidates before any LLM call, and a skeptical
OpenAI judge (GPT-5.4 Mini) gates everything that remains.

## What it optimizes for

Contamination (wrong-person / wrong-domain / non-PI) is weighted **heavier than
coverage**. When uncertain, the pipeline drops the candidate. Country adherence is a
hard constraint — a violation crashes the run rather than emitting it.

---

## How to run (3 steps)

### Step 1 — Install

Requires **Python 3.11+**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### Step 2 — Set up the `.env` file

Copy the template and fill in the two required values:

```bash
copy .env.example .env          # Windows  (cp .env.example .env on macOS/Linux)
```

| Variable | What to put there | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (`sk-...`) | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `OPENALEX_MAILTO` | Any email address you control | No signup needed — OpenAlex's free "polite pool" just asks that requests identify a contact email |

Example `.env`:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
OPENALEX_MAILTO=you@example.com
```

That's all that's required. Optional overrides (models, rate limits, thresholds) are
documented in `.env.example` and default sensibly from `src/config.py`.

### Step 3 — Run

```bash
python run.py --profile sample/student_profile.json
```

- Takes **~4 minutes** on a cold cache (OpenAlex responses are disk-cached, so reruns
  are much faster).
- The shortlist is written to **`sample_output/{student_id}.json`** — see `schema.md`
  for the field-by-field output format. A committed example output from a real run is
  already in `sample_output/`.
- A run-summary table prints to stdout: total recommendations, per-area counts,
  reach/target/safety tier split, and per-layer funnel drop counts (how many candidates
  each filter eliminated, per research area).

### CLI options

```
python run.py --profile <path> [--output <dir>] [--no-cache]
python run.py ingest-feedback --csv sample/outcomes.csv     # feedback loop (bonus)
```

- `--output` — write the shortlist to a different directory.
- `--no-cache` — bypass the OpenAlex disk cache (`.cache/openalex/`).
- With a warm cache, reruns are fast and the **candidate set + ranks are stable**.
  (LLM-written `why_match` *text* may vary slightly between runs since generation is
  non-deterministic in wording; candidate IDs, scores, and ranks are deterministic
  given the cached API responses.)

---

## The input: student profile JSON

A ready-to-run sample is provided at **`sample/student_profile.json`** — a synthetic
but realistic applicant (Indian student, MSc Cognitive Neuroscience from Edinburgh,
three research interests spanning PTSD neuroimaging / computational psychiatry /
affective computing, targeting the US, UK, and Australia for Fall 2027).

To test with your own student, create a JSON file with these fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `student_id` | string | yes | Any unique ID — names the output file |
| `degrees` | list[string] | yes | Degrees with institution and year |
| `stated_interests` | list[string] | yes | Research interests in the student's own words — the parser LLM normalizes these into 3–5 search-ready research areas |
| `target_countries_freeform` | string | yes | Countries in plain English (e.g. `"US, UK and Australia"`) — mapped to ISO codes by the parser |
| `target_intake` | string | yes | e.g. `"Fall 2027"` |
| `nationality` | string | no | Used for eligibility context |
| `resume_text` | string | no | Raw resume text — improves keyword expansion and `why_match` specificity |
| `intro_call_summary` | string | no | Free-text notes from a counselor call — preferences, constraints, funding needs |
| `notable_outputs` | list[string] | no | Publications/posters/projects worth citing in cold emails |

The input is deliberately tolerant of mess: the first pipeline stage is an LLM parser
that converts whatever combination of these fields is present into a structured
`ProfileSpec` (see `src/schemas.py`). Field names don't need to match exactly — extra
fields are passed through to the parser as context.

```bash
python run.py --profile path/to/your_student.json
```

---

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
4. **Identity gate** — cosine(area embedding, author topic centroid) above a
   configured floor. Drops off-topic same-name authors.
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
