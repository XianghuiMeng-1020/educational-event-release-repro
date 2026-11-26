#!/usr/bin/env python3
"""Verify the headline chronology-valid ASSISTments numbers against the stored result tables."""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "assistments"

GRU_SHA256 = "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
ATTENTION_SHA256 = "8728fb05b33399df24daaaa48a76b3c6d6870aa2cee8419909cdeda5f88c0aaa"

# (consumer, schema) -> (stable, set_valued, unsupported, false_stable)
FRONTIER = {
    ("TransparentActionPFA", "S0"): (916845, 2566, 20408, 45),
    ("TransparentActionPFA", "S1"): (861927, 15, 77877, 0),
    ("TransparentActionPFA", "S2"): (859025, 4, 80790, 0),
    ("TransparentActionPFA", "S3"): (839131, 0, 100688, 0),
    ("ActionTokenGRU", "S0"): (810576, 108835, 20408, 1077),
    ("ActionTokenGRU", "S1"): (846772, 15170, 77877, 252),
    ("ActionTokenGRU", "S2"): (844157, 14872, 80790, 250),
    ("ActionTokenGRU", "S3"): (839131, 0, 100688, 0),
}
SOURCE_EVENTS, VALID_TARGETS, INVALID_INTERVALS = 939910, 939819, 91

AUROC = {
    "ActionTokenGRU (action-conditioned)": ("chronological_replication/predictive_diagnostics.csv", "ActionTokenGRU", 0.979096),
    "HIST": ("matched_information/confirmatory_predictive_metrics.csv", "HIST", 0.674077),
    "LEN": ("matched_information/confirmatory_predictive_metrics.csv", "LEN", 0.970937),
    "ACT": ("matched_information/confirmatory_predictive_metrics.csv", "ACT", 0.977092),
    "DKT": ("kt_baselines/confirmatory_predictive_metrics.csv", "DKT", 0.704543),
    "SAKT-KC": ("kt_baselines/confirmatory_predictive_metrics.csv", "SAKT-KC", 0.692357),
    "AKT-NonRasch": ("kt_baselines/confirmatory_predictive_metrics.csv", "AKT-NonRasch", 0.702789),
    "BKT-40": ("kt_baselines/confirmatory_predictive_metrics.csv", "BKT-40", 0.624614),
}


def read_csv(rel: str) -> list[dict]:
    with (RES / rel).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"[{'MATCH' if ok else 'FAIL'}] {name}: {detail}")

    for label, path, expected in (
        ("chronological GRU checkpoint", RES / "chronological_replication/gru_chronological_seed17.pt", GRU_SHA256),
        ("causal attention checkpoint", RES / "attention_consumer/causal_token_attention_seed17.pt", ATTENTION_SHA256),
    ):
        digest = sha256(path)
        check(f"SHA-256 {label}", digest == expected, digest[:16] + "...")

    rows = read_csv("chronological_replication/event_certificate_frontier.csv")
    index = {(r["model"], r["schema"]): r for r in rows}
    for (model, schema), (stable, set_valued, unsupported, false_stable) in FRONTIER.items():
        r = index[(model, schema)]
        got = (int(r["stable"]), int(r["set_valued"]), int(r["unsupported"]), int(r["false_stable"]))
        check(
            f"frontier {model}/{schema}",
            got == (stable, set_valued, unsupported, false_stable)
            and int(r["source_events"]) == SOURCE_EVENTS
            and int(r["valid_targets"]) == VALID_TARGETS
            and int(r["invalid_excluded"]) == INVALID_INTERVALS
            and int(r["stable"]) + int(r["set_valued"]) + int(r["unsupported"]) == VALID_TARGETS,
            f"Stable={got[0]:,} Set-valued={got[1]:,} Unsupported={got[2]:,} False-Stable={got[3]:,}",
        )

    for name, (rel, model, expected) in AUROC.items():
        row = next(r for r in read_csv(rel) if (r.get("model") or r.get("estimand") or "").split(" ")[0] == model)
        value = float(row["roc_auc"] if "roc_auc" in row else row["auroc"])
        check(f"AUROC {name}", abs(value - expected) < 5e-7, f"{value:.6f}")

    print(f"\n{len(FRONTIER) + len(AUROC) + 2 - failures} checks matched, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
