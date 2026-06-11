"""CLI entrypoint (§10).

    python run.py --profile sample/student_profile.json [--output sample_output/] [--no-cache]
    python run.py ingest-feedback --csv outcomes.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src import config, runtime  # noqa: E402  (after load_dotenv)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


log = logging.getLogger("run")


async def _run_shortlist(profile_path: Path, output_path: Path | None, use_cache: bool) -> int:
    from src.graph import build_graph

    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    runtime.init(use_cache=use_cache)
    started = time.monotonic()
    try:
        graph = build_graph()
        initial = {"profile_raw": raw}
        if output_path:
            initial["output_path"] = str(output_path)
        # Generous recursion limit for fan-out; areas run in parallel via Send.
        final_state = await graph.ainvoke(initial, config={"recursion_limit": 50})
    finally:
        await runtime.shutdown()

    elapsed = time.monotonic() - started
    _print_summary(final_state, elapsed)
    return 0


def _print_summary(state: dict, elapsed: float) -> None:
    counts = state.get("counts", {})
    print("\n" + "=" * 64)
    print("RUN SUMMARY")
    print("=" * 64)
    print(f"Output file        : {state.get('output_file')}")
    print(f"Total recs         : {counts.get('total_recommendations', 0)}")
    print(f"Per tier           : {counts.get('per_tier', {})}")
    print(f"why_match dropped  : {counts.get('why_match_dropped', 0)}")
    print(f"Wall-clock         : {elapsed:.1f}s")
    print("\nPer-area counts:")
    for area, n in counts.get("per_area", {}).items():
        print(f"  - {area}: {n}")
    print("\nPer-area funnel drops:")
    for area, drops in counts.get("area_drops", {}).items():
        print(f"  - {area}: {drops}")
    cov = counts.get("coverage_warnings", {})
    if cov:
        print(f"\nCoverage warnings  : {cov}")
    print("=" * 64)


def _ingest_feedback(csv_path: Path) -> int:
    from src.feedback.ingest import ingest_csv

    report = ingest_csv(csv_path)
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="run.py", description="PhD Shortlist Builder")
    sub = parser.add_subparsers(dest="command")

    fb = sub.add_parser("ingest-feedback", help="Ingest an outcomes CSV into the feedback store")
    fb.add_argument("--csv", required=True, type=Path)

    parser.add_argument("--profile", type=Path, help="Path to a student profile JSON")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the OpenAlex disk cache")

    args = parser.parse_args(argv)

    if args.command == "ingest-feedback":
        return _ingest_feedback(args.csv)

    if not args.profile:
        parser.error("--profile is required (or use the ingest-feedback subcommand)")
    if not args.profile.exists():
        parser.error(f"profile not found: {args.profile}")

    try:
        return asyncio.run(_run_shortlist(args.profile, args.output, use_cache=not args.no_cache))
    except Exception as exc:  # noqa: BLE001
        log.error("run failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
