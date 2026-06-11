"""Feedback ingest (§9): outcomes CSV -> blacklist/suppression + learned weights.

Confound (documented in DECISIONS.md): NO_REPLY != bad match; students email a biased
subset. So we learn *per-feature* reweights and apply hard negatives only — never
per-supervisor scores beyond the blacklist.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from src import config
from src.feedback import store

log = logging.getLogger(__name__)

# Outcome label mapping.
POSITIVES = {"ADMIT", "INTERVIEW", "POSITIVE_REPLY"}
WEAK_NEGATIVES = {"REJECT", "NO_REPLY"}
BLACKLIST_OUTCOMES = {"WRONG_PERSON", "BOUNCE"}
SUPPRESSION_OUTCOMES = {"NOT_RECRUITING"}
DROP_FROM_TRAINING = {"OUT_OF_OFFICE", "BOUNCE", "WRONG_PERSON", "NOT_RECRUITING"}

TIERS = ["reach", "target", "safety"]
HAND_WEIGHTS = {
    "W_TOPIC_SIM": config.W_TOPIC_SIM,
    "W_RECENCY": config.W_RECENCY,
    "W_EVIDENCE": config.W_EVIDENCE,
    "W_SENIORITY": config.W_SENIORITY,
}


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _upsert_store(rows: list[dict]) -> None:
    store.init_db()
    for r in rows:
        store.append_outcome(r)
        outcome = (r.get("outcome") or "").strip().upper()
        sid = r.get("supervisor_id")
        if not sid:
            continue
        if outcome in BLACKLIST_OUTCOMES:
            store.add_blacklist(sid, reason=outcome)
        elif outcome in SUPPRESSION_OUTCOMES:
            store.add_suppression(sid)


def _learn_weights(rows: list[dict]) -> dict | None:
    """Fit LogisticRegression on score components + tier one-hot; blend with hand weights."""
    train = [
        r for r in rows
        if (r.get("outcome") or "").strip().upper() not in DROP_FROM_TRAINING
        and (r.get("outcome") or "").strip().upper() in (POSITIVES | WEAK_NEGATIVES)
    ]
    n = len(train)
    if n == 0:
        log.warning("no trainable outcomes; keeping hand weights")
        return None

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        log.warning("scikit-learn unavailable; skipping weight learning")
        return None

    feat_cols = ["score_topic_sim", "score_recency", "score_evidence", "score_seniority"]
    X, y = [], []
    for r in train:
        try:
            row = [float(r.get(c, 0) or 0) for c in feat_cols]
        except ValueError:
            continue
        tier = (r.get("tier") or "target").lower()
        row += [1.0 if tier == t else 0.0 for t in TIERS]
        X.append(row)
        y.append(1 if (r.get("outcome") or "").strip().upper() in POSITIVES else 0)

    if len(set(y)) < 2:
        log.warning("only one class present in outcomes; keeping hand weights")
        return None

    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.asarray(X), np.asarray(y))
    coefs = clf.coef_[0][: len(feat_cols)]
    # Normalize the four component coefficients into positive weights summing to 1.
    pos = np.clip(coefs, a_min=0.0, a_max=None)
    learned = pos / pos.sum() if pos.sum() > 0 else np.array([0.25] * 4)
    w_learned = dict(zip(["W_TOPIC_SIM", "W_RECENCY", "W_EVIDENCE", "W_SENIORITY"], learned.tolist()))

    # Blend toward learned as data grows (lambda capped at 0.8).
    lam = min(n / 500.0, 0.8)
    w_final = {k: lam * w_learned[k] + (1 - lam) * HAND_WEIGHTS[k] for k in HAND_WEIGHTS}
    config.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.WEIGHTS_PATH.write_text(json.dumps(w_final, indent=2), encoding="utf-8")
    log.info("learned weights (lambda=%.2f) written to %s", lam, config.WEIGHTS_PATH)
    return {"lambda": lam, "n_train": n, "w_learned": w_learned, "w_final": w_final}


def _calibration_report(rows: list[dict]) -> dict:
    """Positive rate by tier, by area, by score decile."""
    def pos_rate(group: list[dict]) -> float:
        if not group:
            return 0.0
        p = sum(1 for r in group if (r.get("outcome") or "").upper() in POSITIVES)
        return round(p / len(group), 3)

    by_tier = defaultdict(list)
    by_area = defaultdict(list)
    by_decile = defaultdict(list)
    for r in rows:
        by_tier[(r.get("tier") or "?").lower()].append(r)
        by_area[r.get("area") or "?"].append(r)
        try:
            score = float(r.get("score", 0) or 0)
            by_decile[min(9, int(score * 10))].append(r)
        except ValueError:
            pass
    return {
        "n_outcomes": len(rows),
        "outcome_counts": dict(Counter((r.get("outcome") or "?").upper() for r in rows)),
        "positive_rate_by_tier": {k: pos_rate(v) for k, v in by_tier.items()},
        "positive_rate_by_area": {k: pos_rate(v) for k, v in by_area.items()},
        "positive_rate_by_score_decile": {k: pos_rate(v) for k, v in sorted(by_decile.items())},
    }


def ingest_csv(csv_path: Path) -> dict:
    rows = _read_rows(csv_path)
    _upsert_store(rows)
    weights = _learn_weights(rows)
    report = _calibration_report(rows)
    report["weights"] = weights
    return report
