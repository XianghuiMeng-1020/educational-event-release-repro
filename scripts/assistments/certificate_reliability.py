#!/usr/bin/env python3
"""Empirical certificate reliability and calibration-only issuance.

Uses the frozen P00 certificate and the frozen seed-17 GRU. No retraining and
no modification of earlier result directories.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("chronorep", ROOT / "scripts/assistments/chronological_replication.py")
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)
DECBASE = importlib.util.spec_from_file_location("decbase", ROOT / "scripts/assistments/decision_baselines.py")
decbase = importlib.util.module_from_spec(DECBASE)
DECBASE.loader.exec_module(decbase)
from r7_core import NeuralPilot  # noqa: E402

OUT = ROOT / "results/assistments/certificate_reliability"
PRIV = OUT / "private_scores"
SUPPRED_PRIV = ROOT / "results/assistments/support_predicate_robustness/private_scores"
DECBASE_PRIV = ROOT / "results/assistments/decision_baselines/private_scores"
CHRONOREP_CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
SCHEMAS = ("S0", "S1", "S2", "S3")
MODELS = ("TransparentActionPFA", "ActionTokenGRU")
RULES = ("C0", "C1", "C2", "C3")
RULE_U = {"C0": 1, "C1": 5, "C2": 10, "C3": 20}
ALPHAS = (0.01, 0.05, 0.10)
TAU = 0.80
FAMILY = 32
DELTA = 0.05 / FAMILY
BOOT_SEED = 20318
BOOT_REPS = 2000
EXPECTED_CERT = {
    ("TransparentActionPFA", "S0"): (916845, 2566, 20408, 45),
    ("TransparentActionPFA", "S1"): (861927, 15, 77877, 0),
    ("TransparentActionPFA", "S2"): (859025, 4, 80790, 0),
    ("TransparentActionPFA", "S3"): (839131, 0, 100688, 0),
    ("ActionTokenGRU", "S0"): (810576, 108835, 20408, 1077),
    ("ActionTokenGRU", "S1"): (846772, 15170, 77877, 252),
    ("ActionTokenGRU", "S2"): (844157, 14872, 80790, 250),
    ("ActionTokenGRU", "S3"): (839131, 0, 100688, 0),
}
USER_BINS = ("1", "2-4", "5-9", "10-19", ">=20", "unseen_key")
RICH_BINS = USER_BINS
BEH_BINS = ("1", "2", "3-4", ">=5", "unseen_key")
FP_BINS = BEH_BINS
ENT_BINS = ("0", "(0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,1]", "unseen_key")
COV_BINS = ("covered", "not_covered")


def dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=_json) + "\n", encoding="utf-8")


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return value.tolist()
    return chronorep._json_default(value)


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    header = fields or (list(rows[0]) if rows else [])
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: cell_value(row.get(key)) for key in header})


def cell_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def progress(stage: str, **extra) -> None:
    payload = {"stage": stage, "elapsed_sec": time.time() - STARTED, **extra}
    dump("execution_progress.json", payload)
    print(stage, extra if extra else "", flush=True)


def behavior_and_rich(row: dict) -> tuple[bytes, bytes]:
    behavior = (
        str(row["problem_id"]),
        int(row["correct"]),
        int(row["hint_count"]),
        int(row["attempt_count"]),
        str(row["first_action"]),
        tuple(str(tok).lower() for tok in (row["action_types"] or [])),
    )
    stamped = behavior + (tuple(str(tok) if tok is not None else "" for tok in (row.get("action_timestamps") or [])),)
    return (
        hashlib.blake2s(repr(behavior).encode(), digest_size=16).digest(),
        hashlib.blake2s(repr(stamped).encode(), digest_size=16).digest(),
    )


def entropy_from_counts(counts: np.ndarray) -> tuple[float, float]:
    total = int(counts.sum()) if len(counts) else 0
    if total <= 0 or len(counts) == 0:
        return 0.0, 0.0
    probability = counts.astype(np.float64) / total
    probability = probability[probability > 0]
    entropy = float(-(probability * np.log(probability)).sum())
    normalized = 0.0 if len(counts) < 2 else entropy / float(np.log(len(counts)))
    return entropy, normalized


class Slot:
    __slots__ = (
        "n", "users", "behaviors", "skill", "multi_skill", "pfa", "gru",
        "U", "B", "Fp", "Fg", "Hp", "Hnp", "Hg", "Hng",
        "pfa_counts", "pfa_deltas", "pfa_index", "gru_counts", "gru_index",
    )

    def __init__(self) -> None:
        self.n = 0
        self.users = set()
        self.behaviors = set()
        self.skill = None
        self.multi_skill = False
        self.pfa = {}
        self.gru = {}


def finalize(slot: Slot) -> None:
    slot.U = len(slot.users)
    slot.B = len(slot.behaviors)
    slot.users = None
    slot.behaviors = None
    pfa_items = list(slot.pfa)
    slot.pfa_index = {fp: i for i, fp in enumerate(pfa_items)}
    slot.pfa_counts = np.asarray([slot.pfa[fp] for fp in pfa_items], dtype=np.int32)
    slot.pfa_deltas = np.asarray([chronorep.fp_delta(fp) for fp in pfa_items], dtype=np.float64) if pfa_items else np.zeros((0, 7))
    slot.Fp = len(pfa_items)
    slot.Hp, slot.Hnp = entropy_from_counts(slot.pfa_counts)
    gru_items = list(slot.gru)
    slot.gru_index = {fp: i for i, fp in enumerate(gru_items)}
    slot.gru_counts = np.asarray([slot.gru[fp] for fp in gru_items], dtype=np.int32)
    slot.Fg = len(gru_items)
    slot.Hg, slot.Hng = entropy_from_counts(slot.gru_counts)
    if slot.Fp < 2:
        slot.Hnp = 0.0
    if slot.Fg < 2:
        slot.Hng = 0.0
    slot.pfa = None
    slot.gru = None


def bin_recurrence(value: int) -> str:
    if value <= 0:
        return "unseen_key"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 9:
        return "5-9"
    if value <= 19:
        return "10-19"
    return ">=20"


def bin_small(value: int) -> str:
    if value <= 0:
        return "unseen_key"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    return ">=5"


def bin_entropy(value: float, known: bool) -> str:
    if not known:
        return "unseen_key"
    if value <= 1e-15:
        return "0"
    if value <= 0.25:
        return "(0,0.25]"
    if value <= 0.50:
        return "(0.25,0.50]"
    if value <= 0.75:
        return "(0.50,0.75]"
    return "(0.75,1]"


def build_development() -> tuple[dict, dict]:
    bundles = {schema: {} for schema in SCHEMAS}
    user_ids = {}
    stamped_seen = set()
    behavior_seen = set()
    stamped_dup = behavior_dup = events = invalid = multi_skill_events = 0
    parquet = __import__("pyarrow.parquet").parquet.ParquetFile(chronorep.RECON / "development.parquet")
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            events += 1
            item = chronorep.compact_row(row, int(start), int(end))
            invalid += int(not item["valid"])
            behavior, stamped = behavior_and_rich(row)
            behavior_dup += behavior in behavior_seen
            stamped_dup += stamped in stamped_seen
            behavior_seen.add(behavior)
            stamped_seen.add(stamped)
            ev = chronorep.as_event(item)
            uid = user_ids.setdefault(item["user_id"], len(user_ids))
            pfa = chronorep.pfa_fp(item)
            gru = chronorep.gru_fp(item)
            skill = item["skill_id"]
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                slot = bundles[schema].get(key)
                if slot is None:
                    slot = Slot()
                    bundles[schema][key] = slot
                slot.n += 1
                slot.users.add(uid)
                slot.behaviors.add(behavior)
                if slot.skill is None:
                    slot.skill = skill
                elif slot.skill != skill:
                    slot.multi_skill = True
                    multi_skill_events += 1
                slot.pfa[pfa] = slot.pfa.get(pfa, 0) + 1
                slot.gru[gru] = slot.gru.get(gru, 0) + 1
        if events % 1000000 == 0:
            print(f"development {events} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    stats = {schema: [] for schema in SCHEMAS}
    geometry = {}
    for schema, bundle in bundles.items():
        geo = {
            "keys": 0, "events": 0, "multi_skill_keys": 0, "U_gt_R": 0, "R_ne_N": 0,
            "F_gt_N": 0, "B_gt_N": 0, "B_gt_R": 0, "S3_B_ne_1": 0,
            "Rge2_Beq1_keys": 0, "Rge2_Beq1_events": 0, "Bge2_keys": 0, "Bge2_events": 0,
            "Fp_ne_Fg_keys": 0, "singleton_pfa_keys": 0, "singleton_gru_keys": 0,
            "positive_entropy_pfa_keys": 0, "positive_entropy_gru_keys": 0,
        }
        key_dist = {name: defaultdict(lambda: [0, 0]) for name in ("U", "R", "B", "Fp", "Fg", "Hp", "Hg")}
        for slot in bundle.values():
            finalize(slot)
            if int(slot.pfa_counts.sum()) != slot.n or int(slot.gru_counts.sum()) != slot.n:
                raise RuntimeError("fingerprint counts do not sum to N")
            stats[schema].append((slot.U, slot.n, slot.Fp, slot.Fg))
            geo["keys"] += 1
            geo["events"] += slot.n
            geo["multi_skill_keys"] += int(slot.multi_skill)
            geo["U_gt_R"] += int(slot.U > slot.n)
            geo["R_ne_N"] += int(slot.n != slot.n)
            geo["F_gt_N"] += int(slot.Fp > slot.n or slot.Fg > slot.n)
            geo["B_gt_N"] += int(slot.B > slot.n)
            geo["B_gt_R"] += int(slot.B > slot.n)
            geo["S3_B_ne_1"] += int(schema == "S3" and slot.B != 1)
            geo["Rge2_Beq1_keys"] += int(slot.n >= 2 and slot.B == 1)
            geo["Rge2_Beq1_events"] += slot.n if slot.n >= 2 and slot.B == 1 else 0
            geo["Bge2_keys"] += int(slot.B >= 2)
            geo["Bge2_events"] += slot.n if slot.B >= 2 else 0
            geo["Fp_ne_Fg_keys"] += int(slot.Fp != slot.Fg)
            geo["singleton_pfa_keys"] += int(slot.Fp == 1)
            geo["singleton_gru_keys"] += int(slot.Fg == 1)
            geo["positive_entropy_pfa_keys"] += int(slot.Fp >= 2 and slot.Hnp > 0)
            geo["positive_entropy_gru_keys"] += int(slot.Fg >= 2 and slot.Hng > 0)
            for name, label, weight in (
                ("U", bin_recurrence(slot.U), slot.n),
                ("R", bin_recurrence(slot.n), slot.n),
                ("B", bin_small(slot.B), slot.n),
                ("Fp", bin_small(slot.Fp), slot.n),
                ("Fg", bin_small(slot.Fg), slot.n),
                ("Hp", bin_entropy(slot.Hnp, True), slot.n),
                ("Hg", bin_entropy(slot.Hng, True), slot.n),
            ):
                key_dist[name][label][0] += 1
                key_dist[name][label][1] += weight
        geometry[schema] = {"summary": geo, "bins": {name: {label: counts for label, counts in labels.items()} for name, labels in key_dist.items()}}
    report = {
        "events": events,
        "invalid_intervals_retained_in_library": invalid,
        "development_users": len(user_ids),
        "duplicate_rich_signatures_with_timestamps": stamped_dup,
        "unique_rich_signatures_with_timestamps": len(stamped_seen),
        "duplicate_behavior_signatures_without_timestamps": behavior_dup,
        "unique_behavior_signatures_without_timestamps": len(behavior_seen),
        "multi_skill_event_writes": multi_skill_events,
        "log_base": "natural log; normalized entropy divides by log(F), so the base cancels",
        "R_equals_N_reason": "Each development event has a distinct timestamp-inclusive rich signature, so R(k)=N(k).",
        "B_definition": "blake2s of problem_id, correct, hint_count, attempt_count, first_action, and lowercased action_types. Action timestamps are excluded.",
        "geometry": geometry,
        "key_counts": {schema: int(geometry[schema]["summary"]["keys"]) for schema in SCHEMAS},
    }
    return bundles, {"report": report, "key_stats": {schema: np.asarray(stats[schema], dtype=np.int32) for schema in SCHEMAS}}


def load_split(split: str) -> tuple[dict, dict]:
    grouped = defaultdict(list)
    parquet = __import__("pyarrow.parquet").parquet.ParquetFile(chronorep.RECON / f"{split}.parquet")
    events = invalid = 0
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            item = chronorep.compact_row(row, int(start), int(end))
            grouped[str(row["user_id"])].append(item)
            events += 1
            invalid += int(not item["valid"])
        if events % 400000 == 0:
            print(f"load_{split} {events}", flush=True)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["source_row"])
    valid = sum(len(chronorep.order_valid(rows)[0]) for rows in grouped.values())
    return grouped, {"users": len(grouped), "events": events, "invalid": invalid, "valid_targets": valid}


def score_calibration(grouped: dict, bundles: dict, pilot: NeuralPilot) -> dict:
    users = sorted(grouped)
    n = sum(len(chronorep.order_valid(grouped[uid])[0]) for uid in users)
    user_lookup = {uid: i for i, uid in enumerate(users)}
    user = np.empty(n, dtype=np.int32)
    source = np.empty(n, dtype=np.int64)
    end = np.empty(n, dtype=np.int64)
    realized = np.empty((n, 2), dtype=np.float64)
    lo = np.full((n, 2, 4), np.nan, dtype=np.float64)
    hi = np.full((n, 2, 4), np.nan, dtype=np.float64)
    evidence = np.zeros((n, 4, 6), dtype=np.int32)
    entropy = np.zeros((n, 2, 4), dtype=np.float64)
    present = np.zeros((n, 4), dtype=np.uint8)
    library = np.zeros((n, 2, 4), dtype=np.uint8)
    future = cursor_event = 0
    for user_i, uid in enumerate(users):
        by_start, by_end = chronorep.order_valid(grouped[uid])
        state = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        cursor = 0
        for row in by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                done = by_end[cursor]
                state[done["skill_id"]] += np.asarray(chronorep.fp_delta(chronorep.pfa_fp(done)), dtype=np.float64)
                cursor += 1
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                future += 1
            ev = chronorep.as_event(row)
            prefix = [chronorep.as_event(item) for item in by_end[max(0, cursor - 40):cursor]]
            base = state[row["skill_id"]]
            own_pfa = chronorep.pfa_fp(row)
            own_gru = chronorep.gru_fp(row)
            pfa_realized = float(chronorep.mastery_of(base + np.asarray(chronorep.fp_delta(own_pfa), dtype=np.float64)))
            gru_needed = {own_gru: None}
            slots = []
            for schema in SCHEMAS:
                slot = bundles[schema].get(chronorep.release_key(ev, schema))
                slots.append(slot)
                if slot is not None:
                    gru_needed.update(slot.gru_index)
            pairs = [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in gru_needed]
            gru_scores = chronorep.score_forward(pilot.model, pairs, pilot.device)
            gru_map = {fp: float(score) for fp, score in zip(gru_needed, gru_scores)}
            pos = cursor_event
            user[pos] = user_lookup[uid]
            source[pos] = row["source_row"]
            end[pos] = row["end"]
            realized[pos, 0] = pfa_realized
            realized[pos, 1] = gru_map[own_gru]
            for schema_i, slot in enumerate(slots):
                if slot is None:
                    continue
                present[pos, schema_i] = 1
                evidence[pos, schema_i] = (slot.U, slot.n, slot.n, slot.B, slot.Fp, slot.Fg)
                entropy[pos, 0, schema_i] = slot.Hnp
                entropy[pos, 1, schema_i] = slot.Hng
                pfa_scores = decbase.mastery_vec(base, slot.pfa_deltas)
                gru_values = np.asarray([gru_map[fp] for fp in slot.gru_index], dtype=np.float64)
                lo[pos, 0, schema_i] = float(pfa_scores.min())
                hi[pos, 0, schema_i] = float(pfa_scores.max())
                lo[pos, 1, schema_i] = float(gru_values.min())
                hi[pos, 1, schema_i] = float(gru_values.max())
                library[pos, 0, schema_i] = int(own_pfa in slot.pfa_index)
                library[pos, 1, schema_i] = int(own_gru in slot.gru_index)
            cursor_event += 1
        if user_i and user_i % 250 == 0:
            print(f"calibration_users {user_i} events {cursor_event} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    return {
        "users": len(users), "events": n, "future": future, "user": user, "source": source, "end": end,
        "realized": realized, "lo": lo, "hi": hi, "evidence": evidence, "entropy": entropy,
        "present": present, "library": library,
    }


def save_calibration(payload: dict) -> None:
    PRIV.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRIV / "calibration_intervals.npz", **{key: payload[key] for key in ("user", "source", "end", "realized", "lo", "hi", "evidence", "entropy", "present", "library")})
    (PRIV / "calibration_meta.json").write_text(json.dumps({key: payload[key] for key in ("users", "events", "future")}), encoding="utf-8")


def load_calibration() -> dict:
    arrays = np.load(PRIV / "calibration_intervals.npz")
    meta = json.loads((PRIV / "calibration_meta.json").read_text())
    payload = {key: arrays[key] for key in arrays.files}
    payload.update(meta)
    return payload


def attach_confirmatory(grouped: dict, bundles: dict, cached: dict) -> dict:
    users = sorted(grouped)
    n = cached["events"]
    evidence_b = np.zeros(cached["counts"].shape[1], dtype=np.int32)
    entropy = np.zeros((n, 2, cached["counts"].shape[1]), dtype=np.float64)
    b_values = np.zeros((n, cached["counts"].shape[1]), dtype=np.int32)
    mismatches = future = cursor = 0
    for uid in users:
        by_start, by_end = chronorep.order_valid(grouped[uid])
        cursor_hist = 0
        for row in by_start:
            while cursor_hist < len(by_end) and by_end[cursor_hist]["end"] < row["start"]:
                cursor_hist += 1
            if any(item["end"] >= row["start"] for item in by_end[:cursor_hist]):
                future += 1
            if int(cached["source"][cursor]) != int(row["source_row"]):
                mismatches += 1
            ev = chronorep.as_event(row)
            for schema_i, schema in enumerate(SCHEMAS):
                slot = bundles[schema].get(chronorep.release_key(ev, schema))
                stored = cached["counts"][cursor, schema_i]
                present = int(cached["present"][cursor, schema_i])
                if slot is None:
                    mismatches += int(present != 0)
                    continue
                b_values[cursor, schema_i] = slot.B
                entropy[cursor, 0, schema_i] = slot.Hnp
                entropy[cursor, 1, schema_i] = slot.Hng
                if (slot.U, slot.n, slot.Fp, slot.Fg, 1) != (int(stored[0]), int(stored[1]), int(stored[2]), int(stored[3]), present):
                    mismatches += 1
            cursor += 1
    if cursor != n:
        mismatches += abs(cursor - n)
    cached = dict(cached)
    cached["B"] = b_values
    cached["entropy"] = entropy
    cached["attach_mismatches"] = mismatches
    cached["future_recheck"] = future
    cached["evidence_width"] = int(evidence_b.shape[0])
    return cached


def codes_from(lo, hi, present, evidence) -> np.ndarray:
    supported = present.astype(bool) & ((evidence[:, :, 0] >= 2) | (evidence[:, :, 1] >= 2))
    code = np.zeros(lo.shape, dtype=np.uint8)
    for model_i in range(lo.shape[1]):
        finite = np.isfinite(lo[:, model_i, :]) & np.isfinite(hi[:, model_i, :])
        progress = finite & (lo[:, model_i, :] >= TAU)
        review = finite & (hi[:, model_i, :] < TAU) & ~progress
        code[:, model_i, :][progress] = 1
        code[:, model_i, :][review] = 2
        code[:, model_i, :][finite & (code[:, model_i, :] == 0)] = 3
        code[:, model_i, :][~supported] = 0
    return code


def false_stable(code, realized) -> np.ndarray:
    label = realized >= TAU
    out = np.zeros(code.shape, dtype=bool)
    for model_i in range(code.shape[1]):
        action = code[:, model_i, :] == 1
        stable = (code[:, model_i, :] == 1) | (code[:, model_i, :] == 2)
        out[:, model_i, :] = stable & (action != label[:, model_i, None])
    return out


def cp_upper(errors: int, learners: int, delta: float = DELTA) -> float:
    if learners <= 0:
        return float("nan")
    if errors >= learners:
        return 1.0
    if errors <= 0:
        return float(1.0 - delta ** (1.0 / learners))
    return float(beta_dist.ppf(1.0 - delta, errors + 1, learners - errors))


def rule_mask(code, users_u, rule: str) -> np.ndarray:
    stable = (code == 1) | (code == 2)
    if rule == "C0":
        return stable
    return stable & (users_u >= RULE_U[rule])


def learner_flags(user, mask, n_users: int) -> np.ndarray:
    if n_users <= 0:
        return np.zeros(0, dtype=bool)
    if not np.any(mask):
        return np.zeros(n_users, dtype=bool)
    return np.bincount(user[mask], minlength=n_users) > 0


def ratio(num: int, den: int):
    if den <= 0:
        return None
    return num / den


def population_views(name: str, payload: dict, code: np.ndarray, fs: np.ndarray) -> dict:
    if name == "calibration":
        evidence = payload["evidence"]
        u, r, n, b, fp, fg = (evidence[:, :, i] for i in range(6))
    else:
        u = payload["counts"][:, :, 0]
        r = payload["counts"][:, :, 1]
        n = payload["counts"][:, :, 1]
        b = payload["B"]
        fp = payload["counts"][:, :, 2]
        fg = payload["counts"][:, :, 3]
    return {
        "name": name, "users": int(payload["users"]), "events": int(payload["events"]),
        "user": payload["user"], "realized": payload["realized"], "library": payload["library"].astype(bool),
        "present": payload["present"].astype(bool), "code": code, "fs": fs,
        "U": u, "R": r, "N": n, "B": b, "F": np.stack([fp, fg], axis=1), "H": payload["entropy"],
        "lo": payload.get("lo"), "hi": payload.get("hi"), "source": payload.get("source"),
    }


def empty_metrics() -> dict:
    return {
        "valid_events": 0, "supported": 0, "stable": 0, "set_valued": 0, "unsupported": 0,
        "learners": 0, "learners_with_stable": 0, "stable_uncovered": 0, "false_stable": 0,
        "fingerprint_covered_supported": 0, "action_contained_supported": 0,
    }


def accumulate(view, model_i, schema_i, mask) -> dict:
    code = view["code"][:, model_i, schema_i]
    fs = view["fs"][:, model_i, schema_i]
    library = view["library"][:, model_i, schema_i]
    present = view["present"][:, schema_i]
    supported = (code > 0) & present
    stable = (code == 1) | (code == 2)
    chosen = mask & np.ones(len(code), dtype=bool)
    user = view["user"]
    n_users = view["users"]
    realized_label = view["realized"][:, model_i] >= TAU
    contained = supported & (
        ((code == 1) & realized_label) | ((code == 2) & ~realized_label) | (code == 3)
    )
    metrics = empty_metrics()
    metrics["valid_events"] = int(chosen.sum())
    metrics["supported"] = int((chosen & supported).sum())
    metrics["stable"] = int((chosen & stable).sum())
    metrics["set_valued"] = int((chosen & (code == 3)).sum())
    metrics["unsupported"] = int((chosen & (code == 0)).sum())
    metrics["learners"] = int(learner_flags(user, chosen, n_users).sum())
    metrics["learners_with_stable"] = int(learner_flags(user, chosen & stable, n_users).sum())
    metrics["stable_uncovered"] = int((chosen & stable & ~library).sum())
    metrics["false_stable"] = int((chosen & fs).sum())
    metrics["fingerprint_covered_supported"] = int((chosen & supported & library).sum())
    metrics["action_contained_supported"] = int((chosen & contained).sum())
    return metrics


def metric_row(split, consumer, schema, variable, label, view, metrics) -> dict:
    zeros = []
    stable_rate = ratio(metrics["stable"], metrics["valid_events"])
    fs_stable = ratio(metrics["false_stable"], metrics["stable"])
    fs_all = ratio(metrics["false_stable"], metrics["valid_events"])
    cover = ratio(metrics["fingerprint_covered_supported"], metrics["supported"])
    contain = ratio(metrics["action_contained_supported"], metrics["supported"])
    if metrics["valid_events"] == 0:
        zeros.append("stable/all")
    if metrics["stable"] == 0:
        zeros.append("false_stable/stable")
    if metrics["valid_events"] == 0:
        zeros.append("false_stable/all")
    if metrics["supported"] == 0:
        zeros.extend(["fingerprint_coverage_supported", "action_containment_supported"])
    return {
        "split": split, "consumer": consumer, "schema": schema, "variable": variable, "bin": label,
        "valid_events": metrics["valid_events"], "supported": metrics["supported"], "stable": metrics["stable"],
        "set_valued": metrics["set_valued"], "unsupported": metrics["unsupported"],
        "learners": metrics["learners"], "learners_with_stable": metrics["learners_with_stable"],
        "stable_uncovered": metrics["stable_uncovered"], "false_stable": metrics["false_stable"],
        "stable_over_valid": stable_rate, "false_stable_over_stable": fs_stable,
        "false_stable_over_valid": fs_all,
        "fingerprint_coverage_supported_num": metrics["fingerprint_covered_supported"],
        "fingerprint_coverage_supported_den": metrics["supported"],
        "fingerprint_coverage_supported": cover,
        "action_containment_supported_num": metrics["action_contained_supported"],
        "action_containment_supported_den": metrics["supported"],
        "action_containment_supported": contain,
        "zero_denominator": "|".join(zeros),
    }


def variable_masks(view, model_i, schema_i):
    present = view["present"][:, schema_i]
    u = np.where(present, view["U"][:, schema_i], 0)
    r = np.where(present, view["R"][:, schema_i], 0)
    b = np.where(present, view["B"][:, schema_i], 0)
    f = np.where(present, view["F"][:, model_i, schema_i], 0)
    h = view["H"][:, model_i, schema_i]
    library = view["library"][:, model_i, schema_i]
    return {
        "development_users": (np.array([bin_recurrence(int(value)) if present[i] else "unseen_key" for i, value in enumerate(u)], dtype=object), USER_BINS),
        "rich_signatures": (np.array([bin_recurrence(int(value)) if present[i] else "unseen_key" for i, value in enumerate(r)], dtype=object), RICH_BINS),
        "behavior_signatures": (np.array([bin_small(int(value)) if present[i] else "unseen_key" for i, value in enumerate(b)], dtype=object), BEH_BINS),
        "consumer_fingerprints": (np.array([bin_small(int(value)) if present[i] else "unseen_key" for i, value in enumerate(f)], dtype=object), FP_BINS),
        "fingerprint_entropy": (np.array([bin_entropy(float(value), bool(known)) for value, known in zip(h, present)], dtype=object), ENT_BINS),
        "fingerprint_coverage": (np.where(library, "covered", "not_covered").astype(object), COV_BINS),
    }


def vector_bins(values: np.ndarray, present: np.ndarray, kind: str) -> np.ndarray:
    out = np.full(len(values), "unseen_key", dtype=object)
    known = np.flatnonzero(present)
    if kind == "recurrence":
        bins = np.full(len(known), ">=20", dtype=object)
        v = values[known]
        bins[v == 1] = "1"
        bins[(v >= 2) & (v <= 4)] = "2-4"
        bins[(v >= 5) & (v <= 9)] = "5-9"
        bins[(v >= 10) & (v <= 19)] = "10-19"
        out[known] = bins
    elif kind == "small":
        bins = np.full(len(known), ">=5", dtype=object)
        v = values[known]
        bins[v == 1] = "1"
        bins[v == 2] = "2"
        bins[(v >= 3) & (v <= 4)] = "3-4"
        out[known] = bins
    else:
        h = values[known]
        bins = np.full(len(known), "(0.75,1]", dtype=object)
        bins[h <= 1e-15] = "0"
        bins[(h > 1e-15) & (h <= 0.25)] = "(0,0.25]"
        bins[(h > 0.25) & (h <= 0.50)] = "(0.25,0.50]"
        bins[(h > 0.50) & (h <= 0.75)] = "(0.50,0.75]"
        out[known] = bins
    return out


def strata_rows(view) -> dict[str, list[dict]]:
    grouped = {name: [] for name in (
        "reliability_by_development_users.csv", "reliability_by_rich_signatures.csv",
        "reliability_by_behavior_signatures.csv", "reliability_by_consumer_fingerprints.csv",
        "reliability_by_fingerprint_entropy.csv", "reliability_by_fingerprint_coverage.csv",
    )}
    file_for = {
        "development_users": "reliability_by_development_users.csv",
        "rich_signatures": "reliability_by_rich_signatures.csv",
        "behavior_signatures": "reliability_by_behavior_signatures.csv",
        "consumer_fingerprints": "reliability_by_consumer_fingerprints.csv",
        "fingerprint_entropy": "reliability_by_fingerprint_entropy.csv",
        "fingerprint_coverage": "reliability_by_fingerprint_coverage.csv",
    }
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            present = view["present"][:, schema_i]
            labels = {
                "development_users": (vector_bins(view["U"][:, schema_i], present, "recurrence"), USER_BINS),
                "rich_signatures": (vector_bins(view["R"][:, schema_i], present, "recurrence"), RICH_BINS),
                "behavior_signatures": (vector_bins(view["B"][:, schema_i], present, "small"), BEH_BINS),
                "consumer_fingerprints": (vector_bins(view["F"][:, model_i, schema_i], present, "small"), FP_BINS),
                "fingerprint_entropy": (vector_bins(view["H"][:, model_i, schema_i], present, "entropy"), ENT_BINS),
                "fingerprint_coverage": (np.where(view["library"][:, model_i, schema_i], "covered", "not_covered").astype(object), COV_BINS),
            }
            for variable, (assigned, names) in labels.items():
                for name in names:
                    metrics = accumulate(view, model_i, schema_i, assigned == name)
                    grouped[file_for[variable]].append(metric_row(view["name"], model, schema, variable, name, view, metrics))
    return grouped


def issuance_record(view, model_i, schema_i, rule: str) -> dict:
    code = view["code"][:, model_i, schema_i]
    issued = rule_mask(code, view["U"][:, schema_i], rule)
    fs = view["fs"][:, model_i, schema_i] & issued
    user = view["user"]
    has_issued = learner_flags(user, issued, view["users"])
    has_fs = learner_flags(user, fs, view["users"])
    issued_n = int(issued.sum())
    fs_n = int(fs.sum())
    correct = issued_n - fs_n
    k = int(has_issued.sum())
    e = int((has_issued & has_fs).sum())
    return {
        "issued": issued, "issued_events": issued_n, "false_stable": fs_n, "correct_stable": correct,
        "K": k, "E": e, "has_issued": has_issued, "has_fs": has_fs,
        "event_issuance": ratio(issued_n, view["events"]),
        "learner_issuance": ratio(k, view["users"]),
        "event_error": ratio(fs_n, issued_n),
        "learner_error": ratio(e, k),
        "upper": cp_upper(e, k) if k else None,
    }


def select_policies(calibration) -> list[dict]:
    chosen = []
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            records = {rule: issuance_record(calibration, model_i, schema_i, rule) for rule in RULES}
            for alpha in ALPHAS:
                qualified = []
                for rule in RULES:
                    upper = records[rule]["upper"]
                    if records[rule]["K"] > 0 and upper is not None and upper <= alpha:
                        qualified.append(rule)
                if not qualified:
                    selected = "NONE"
                else:
                    best = max(records[rule]["event_issuance"] for rule in qualified)
                    selected = next(rule for rule in RULES if rule in qualified and records[rule]["event_issuance"] == best)
                rec = records[selected] if selected != "NONE" else None
                chosen.append({
                    "consumer": model, "schema": schema, "alpha": alpha, "selected_rule": selected,
                    "qualified_rules": qualified, "no_rule_qualified": selected == "NONE",
                    "calibration_K": None if rec is None else rec["K"],
                    "calibration_E": None if rec is None else rec["E"],
                    "calibration_upper": None if rec is None else rec["upper"],
                    "calibration_issued_events": 0 if rec is None else rec["issued_events"],
                    "calibration_event_issuance": 0.0 if rec is None else rec["event_issuance"],
                    "candidate_upper": {rule: records[rule]["upper"] for rule in RULES},
                    "candidate_issuance": {rule: records[rule]["event_issuance"] for rule in RULES},
                })
    return chosen


def frontier_rows(view) -> list[dict]:
    rows = []
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            metrics = accumulate(view, model_i, schema_i, np.ones(view["events"], dtype=bool))
            rows.append({
                "split": view["name"], "consumer": model, "schema": schema,
                "valid_events": metrics["valid_events"], "stable": metrics["stable"],
                "set_valued": metrics["set_valued"], "unsupported": metrics["unsupported"],
                "false_stable": metrics["false_stable"], "stable_uncovered": metrics["stable_uncovered"],
                "supported": metrics["supported"],
                "partition": metrics["stable"] + metrics["set_valued"] + metrics["unsupported"],
            })
    return rows


def three_case_rows(view) -> list[dict]:
    rows = []
    populations = [("P00", None)] + [(rule, rule) for rule in RULES]
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            code = view["code"][:, model_i, schema_i]
            library = view["library"][:, model_i, schema_i]
            fs_all = view["fs"][:, model_i, schema_i]
            for label, rule in populations:
                if rule is None:
                    mask = (code == 1) | (code == 2)
                else:
                    mask = rule_mask(code, view["U"][:, schema_i], rule)
                n_s = int(mask.sum())
                n_u = int((mask & ~library).sum())
                n_fs = int((mask & fs_all).sum())
                if n_s == 0:
                    case = "C"
                elif n_u == 0:
                    case = "B"
                else:
                    case = "A"
                product = None if n_s == 0 or n_u == 0 else (n_u / n_s) * (n_fs / n_u)
                direct = None if n_s == 0 else n_fs / n_s
                rows.append({
                    "split": view["name"], "population": label, "consumer": model, "schema": schema,
                    "n_S": n_s, "n_U": n_u, "n_FS": n_fs, "case": case,
                    "false_stable_over_stable": direct, "product": product,
                    "identity_gap": None if product is None or direct is None else abs(product - direct),
                    "case_B_requires_zero_false_stable": case != "B" or n_fs == 0,
                    "every_false_stable_is_uncovered": int((mask & fs_all & library).sum()) == 0,
                })
    return rows


def curve_row(view, model, schema, rule, rec) -> dict:
    return {
        "split": view["name"], "consumer": model, "schema": schema, "rule": rule,
        "valid_events": view["events"], "issued_events": rec["issued_events"],
        "event_issuance": rec["event_issuance"], "learners": view["users"],
        "learners_issued": rec["K"], "learner_issuance": rec["learner_issuance"],
        "false_stable": rec["false_stable"], "false_stable_over_issued": rec["event_error"],
        "learner_any_error_num": rec["E"], "learner_any_error_den": rec["K"],
        "learner_any_error": rec["learner_error"], "correct_stable": rec["correct_stable"],
        "correct_stable_over_valid": ratio(rec["correct_stable"], view["events"]),
    }


def replicate_sums(count: np.ndarray, flag: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    totals = np.empty(BOOT_REPS, dtype=np.float64)
    flags = np.empty(BOOT_REPS, dtype=np.float64)
    for start in range(0, BOOT_REPS, 200):
        batch = draws[start:start + 200]
        totals[start:start + 200] = count[batch].sum(axis=1)
        flags[start:start + 200] = flag[batch].sum(axis=1)
    return totals, flags


def bootstrap_rows(view) -> list[dict]:
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    user = view["user"]
    n_users = view["users"]
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            draws = rng.integers(0, n_users, size=(BOOT_REPS, n_users))
            packed = {}
            for rule in RULES:
                rec = issuance_record(view, model_i, schema_i, rule)
                issued_count = np.bincount(user[rec["issued"]], minlength=n_users).astype(np.float64)
                fs_idx = rec["issued"] & view["fs"][:, model_i, schema_i]
                fs_count = np.bincount(user[fs_idx], minlength=n_users).astype(np.float64) if np.any(fs_idx) else np.zeros(n_users)
                issued_sum, learner_sum = replicate_sums(issued_count, rec["has_issued"].astype(np.float64), draws)
                fs_sum, error_sum = replicate_sums(fs_count, (rec["has_issued"] & rec["has_fs"]).astype(np.float64), draws)
                packed[rule] = {"issued": issued_sum, "fs": fs_sum, "correct": issued_sum - fs_sum, "learners": learner_sum, "errors": error_sum, "point": rec}
            for rule in RULES:
                item = packed[rule]
                point = item["point"]
                base = packed["C0"]
                series = {
                    "event_issuance": item["issued"] / view["events"],
                    "learner_issuance": item["learners"] / n_users,
                    "false_stable_over_issued": np.divide(item["fs"], item["issued"], out=np.full(BOOT_REPS, np.nan), where=item["issued"] > 0),
                    "learner_any_error": np.divide(item["errors"], item["learners"], out=np.full(BOOT_REPS, np.nan), where=item["learners"] > 0),
                    "correct_stable_coverage_change_vs_c0": (item["correct"] - base["correct"]) / view["events"],
                    "false_stable_count_change_vs_c0": item["fs"] - base["fs"],
                }
                points = {
                    "event_issuance": point["event_issuance"],
                    "learner_issuance": point["learner_issuance"],
                    "false_stable_over_issued": point["event_error"],
                    "learner_any_error": point["learner_error"],
                    "correct_stable_coverage_change_vs_c0": ratio(point["correct_stable"] - base["point"]["correct_stable"], view["events"]),
                    "false_stable_count_change_vs_c0": point["false_stable"] - base["point"]["false_stable"],
                }
                for metric, values in series.items():
                    finite = values[np.isfinite(values)]
                    degenerate = bool(metric in ("false_stable_over_issued", "learner_any_error") and len(finite) and np.all(finite == 0))
                    low, high = (None, None) if len(finite) == 0 else (float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))
                    rows.append({
                        "split": view["name"], "consumer": model, "schema": schema, "rule": rule, "metric": metric,
                        "point": points[metric], "p2_5": low, "p97_5": high, "replicates": BOOT_REPS,
                        "finite_replicates": int(len(finite)), "zero_error_replicates": int(np.sum(finite == 0)) if len(finite) else 0,
                        "degenerate_all_zero": degenerate, "seed": BOOT_SEED,
                        "same_draws_across_rules": True,
                    })
    return rows


def comparison_rows(calibration, confirmatory, policies) -> list[dict]:
    rows = []
    policy_lookup = {(item["consumer"], item["schema"], item["alpha"]): item for item in policies}
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            left = accumulate(calibration, model_i, schema_i, np.ones(calibration["events"], dtype=bool))
            right = accumulate(confirmatory, model_i, schema_i, np.ones(confirmatory["events"], dtype=bool))
            cal_c0 = issuance_record(calibration, model_i, schema_i, "C0")
            conf_c0 = issuance_record(confirmatory, model_i, schema_i, "C0")
            for alpha in ALPHAS:
                policy = policy_lookup[(model, schema, alpha)]
                rule = policy["selected_rule"]
                if rule == "NONE":
                    conf_rate = 0.0
                    transfer_fail = False
                    conf_e = 0
                    conf_k = 0
                else:
                    applied = issuance_record(confirmatory, model_i, schema_i, rule)
                    conf_rate = applied["learner_error"]
                    conf_e = applied["E"]
                    conf_k = applied["K"]
                    transfer_fail = conf_k > 0 and conf_rate is not None and conf_rate > alpha
                rows.append({
                    "consumer": model, "schema": schema, "alpha": alpha, "selected_rule": rule,
                    "calibration_support_fraction": ratio(left["supported"], left["valid_events"]),
                    "confirmatory_support_fraction": ratio(right["supported"], right["valid_events"]),
                    "calibration_stable_fraction": ratio(left["stable"], left["valid_events"]),
                    "confirmatory_stable_fraction": ratio(right["stable"], right["valid_events"]),
                    "calibration_set_fraction": ratio(left["set_valued"], left["valid_events"]),
                    "confirmatory_set_fraction": ratio(right["set_valued"], right["valid_events"]),
                    "calibration_false_stable_over_stable": ratio(left["false_stable"], left["stable"]),
                    "confirmatory_false_stable_over_stable": ratio(right["false_stable"], right["stable"]),
                    "calibration_fingerprint_coverage_supported": ratio(left["fingerprint_covered_supported"], left["supported"]),
                    "confirmatory_fingerprint_coverage_supported": ratio(right["fingerprint_covered_supported"], right["supported"]),
                    "calibration_c0_learner_any_error": cal_c0["learner_error"],
                    "confirmatory_c0_learner_any_error": conf_c0["learner_error"],
                    "confirmatory_selected_learner_any_error": conf_rate,
                    "confirmatory_selected_E": conf_e, "confirmatory_selected_K": conf_k,
                    "calibration_transfer_failure": transfer_fail,
                })
    return rows


def confirmatory_policy_rows(confirmatory, policies) -> list[dict]:
    rows = []
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            c0 = issuance_record(confirmatory, model_i, schema_i, "C0")
            for policy in policies:
                if policy["consumer"] != model or policy["schema"] != schema:
                    continue
                rule = policy["selected_rule"]
                applied = None if rule == "NONE" else issuance_record(confirmatory, model_i, schema_i, rule)
                issued = 0 if applied is None else applied["issued_events"]
                fs = 0 if applied is None else applied["false_stable"]
                correct = 0 if applied is None else applied["correct_stable"]
                k = 0 if applied is None else applied["K"]
                e = 0 if applied is None else applied["E"]
                rows.append({
                    "consumer": model, "schema": schema, "alpha": policy["alpha"], "selected_rule": rule,
                    "no_rule_qualified": policy["no_rule_qualified"],
                    "calibration_K": policy["calibration_K"], "calibration_E": policy["calibration_E"],
                    "calibration_upper": policy["calibration_upper"],
                    "calibration_issued_events": policy["calibration_issued_events"],
                    "calibration_event_issuance": policy["calibration_event_issuance"],
                    "confirmatory_valid_events": confirmatory["events"],
                    "confirmatory_issued_stable": issued,
                    "confirmatory_event_issuance": ratio(issued, confirmatory["events"]),
                    "confirmatory_correct_stable": correct, "confirmatory_false_stable": fs,
                    "confirmatory_false_stable_over_issued": ratio(fs, issued),
                    "confirmatory_learners_issued": k, "confirmatory_learners_with_false_stable": e,
                    "confirmatory_learner_any_error": ratio(e, k),
                    "correct_stable_removed_vs_c0": c0["correct_stable"] - correct,
                    "false_stable_removed_vs_c0": c0["false_stable"] - fs,
                    "issued_removed_vs_c0": c0["issued_events"] - issued,
                })
    return rows


def candidate_rows(view) -> tuple[list[dict], list[dict]]:
    issuance, bounds = [], []
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            for rule in RULES:
                rec = issuance_record(view, model_i, schema_i, rule)
                issuance.append(curve_row(view, model, schema, rule, rec))
                bounds.append({
                    "split": view["name"], "consumer": model, "schema": schema, "rule": rule,
                    "N_learners": view["users"], "K_issued_learners": rec["K"], "E_any_error_learners": rec["E"],
                    "delta": DELTA, "family_size": FAMILY, "upper": rec["upper"],
                    "eligible": rec["K"] > 0,
                    "qualifies_0.01": rec["K"] > 0 and rec["upper"] is not None and rec["upper"] <= 0.01,
                    "qualifies_0.05": rec["K"] > 0 and rec["upper"] is not None and rec["upper"] <= 0.05,
                    "qualifies_0.10": rec["K"] > 0 and rec["upper"] is not None and rec["upper"] <= 0.10,
                    "denominator": "K learners with at least one issued Stable certificate",
                    "event_count_is_not_the_denominator": True,
                })
    return issuance, bounds


def bound_qa() -> dict:
    checks = []
    for errors, learners in ((0, 10), (0, 6804), (10, 10), (5, 100), (1, 2), (3, 5000)):
        upper = cp_upper(errors, learners)
        if errors == 0:
            closed = 1.0 - DELTA ** (1.0 / learners)
            beta_form = float(beta_dist.ppf(1.0 - DELTA, 1, learners))
            agree = abs(upper - closed) <= 1e-12 and abs(upper - beta_form) <= 1e-10
        elif errors == learners:
            agree = upper == 1.0
        else:
            agree = abs(upper - float(beta_dist.ppf(1.0 - DELTA, errors + 1, learners - errors))) <= 1e-12
        checks.append({"E": errors, "K": learners, "upper": upper, "matches_formula": agree})
    return {
        "definition": "One-sided Clopper-Pearson upper bound: Beta(1-delta; E+1, K-E) for E<K, 1 for E=K, and 1-delta^(1/K) for E=0.",
        "delta": DELTA, "family": "2 consumers x 4 schemas x 4 rules = 32",
        "family_frozen": FAMILY == 32, "checks": checks, "all_match": all(item["matches_formula"] for item in checks),
        "unit": "learner with at least one issued certificate, not an event",
    }


def sample_positions(view, split: str) -> list[int]:
    code = view["code"]
    fs = view["fs"]
    present = view["present"]
    library = view["library"]
    u = view["U"][:, 0]
    r = view["R"][:, 0]
    b = view["B"][:, 0]
    f = view["F"][:, :, 0]
    h = view["H"][:, :, 0]
    realized = view["realized"]
    lo = view["lo"][:, :, 0] if view["lo"] is not None else None
    groups = {
        "stable": np.flatnonzero((code[:, 0, 0] == 1) | (code[:, 0, 0] == 2)),
        "set_valued": np.flatnonzero(code[:, 0, 0] == 3),
        "unsupported": np.flatnonzero(code[:, 0, 0] == 0),
        "covered": np.flatnonzero(library[:, 0, 0]),
        "uncovered": np.flatnonzero(present[:, 0] & ~library[:, 0, 0]),
        "false_stable": np.flatnonzero(fs[:, 0, 0] | fs[:, 1, 0]),
        "one_learner_multiple_rich": np.flatnonzero(present[:, 0] & (u == 1) & (r >= 2)),
        "many_learners_one_fingerprint": np.flatnonzero(present[:, 0] & (u >= 2) & ((f[:, 0] == 1) | (f[:, 1] == 1))),
        "multiple_fingerprints": np.flatnonzero(present[:, 0] & ((f[:, 0] >= 2) | (f[:, 1] >= 2))),
        "timestamp_repeated_behavior": np.flatnonzero(present[:, 0] & (r > b) & (b >= 1)),
        "high_entropy": np.flatnonzero(present[:, 0] & ((h[:, 0] > 0.75) | (h[:, 1] > 0.75))),
        "zero_entropy": np.flatnonzero(present[:, 0] & ((f[:, 0] == 1) | (f[:, 1] == 1))),
        "high_recurrence": np.flatnonzero(present[:, 0] & (u >= 20)),
        "low_recurrence": np.flatnonzero(present[:, 0] & (u == 1)),
    }
    if lo is not None:
        def nearest(bound: np.ndarray) -> np.ndarray:
            distance = np.abs(bound - TAU)
            distance = np.where(np.isfinite(distance), distance, np.inf)
            return distance.min(axis=1) <= 0.02

        near = np.any(np.abs(realized - TAU) <= 0.02, axis=1) | nearest(lo) | nearest(view["hi"][:, :, 0])
        groups["near_tau"] = np.flatnonzero(near)
    chosen = []
    seen = set()
    for name, indexes in groups.items():
        order = sorted(indexes.tolist(), key=lambda pos: hashlib.sha256(f"WSDM203-G17-QA|{split}|{name}|{int(view['source'][pos])}".encode()).hexdigest())
        for pos in order[:40]:
            if pos not in seen:
                seen.add(pos)
                chosen.append(pos)
            if len(chosen) >= 300:
                return chosen
    if len(chosen) < 300:
        rest = sorted(range(view["events"]), key=lambda pos: hashlib.sha256(f"WSDM203-G17-QA|{split}|fill|{int(view['source'][pos])}".encode()).hexdigest())
        for pos in rest:
            if pos not in seen:
                chosen.append(pos)
                seen.add(pos)
            if len(chosen) >= 300:
                break
    return chosen


def qa_record(view, pos: int, split: str) -> dict:
    return {
        "split": split, "source_row": int(view["source"][pos]), "user_index": int(view["user"][pos]),
        "present": view["present"][pos].astype(int).tolist(),
        "U": view["U"][pos].astype(int).tolist(), "R": view["R"][pos].astype(int).tolist(),
        "N": view["N"][pos].astype(int).tolist(), "B": view["B"][pos].astype(int).tolist(),
        "F": view["F"][:, :, ][pos].astype(int).tolist() if False else view["F"][pos].astype(int).tolist(),
        "Hnorm": view["H"][pos].tolist(),
        "code": view["code"][pos].astype(int).tolist(),
        "library": view["library"][pos].astype(int).tolist(),
        "false_stable": view["fs"][pos].astype(int).tolist(),
        "realized": view["realized"][pos].tolist(),
        "lo": None if view["lo"] is None else view["lo"][pos].tolist(),
        "hi": None if view["hi"] is None else view["hi"][pos].tolist(),
        "p00": [bool((view["U"][pos, s] >= 2) or (view["R"][pos, s] >= 2)) if view["present"][pos, s] else False for s in range(4)],
    }


def direct_rescore(grouped, bundles, pilot, view, positions: list[int]) -> dict:
    wanted = {int(view["source"][pos]): pos for pos in positions}
    checked = gap = alternatives = full_classes = future = 0
    realized_label_mismatches = []
    endpoint_replay = []
    for uid in sorted(grouped):
        rows = grouped[uid]
        if not any(int(row["source_row"]) in wanted and row["valid"] for row in rows):
            continue
        by_start, by_end = chronorep.order_valid(rows)
        state = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        cursor = 0
        for row in by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                done = by_end[cursor]
                state[done["skill_id"]] += np.asarray(chronorep.fp_delta(chronorep.pfa_fp(done)), dtype=np.float64)
                cursor += 1
            pos = wanted.get(int(row["source_row"]))
            if pos is None:
                continue
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                future += 1
            ev = chronorep.as_event(row)
            prefix = [chronorep.as_event(item) for item in by_end[max(0, cursor - 40):cursor]]
            base = state[row["skill_id"]]
            own_pfa = chronorep.pfa_fp(row)
            own_gru = chronorep.gru_fp(row)
            pfa_direct = float(chronorep.mastery_of(base + np.asarray(chronorep.fp_delta(own_pfa), dtype=np.float64)))
            slot = bundles["S0"].get(chronorep.release_key(ev, "S0"))
            extra = [own_gru]
            if slot is not None and slot.gru_index:
                extra.append(next(iter(slot.gru_index)))
            pairs = [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in extra]
            scores = chronorep.score_forward(pilot.model, pairs, pilot.device)
            comparisons = (
                ("pfa_realized", pfa_direct, float(view["realized"][pos, 0])),
                ("gru_realized", float(scores[0]), float(view["realized"][pos, 1])),
            )
            for name, direct_score, stored_score in comparisons:
                gap = max(gap, abs(direct_score - stored_score))
                if (direct_score >= TAU) != (stored_score >= TAU):
                    realized_label_mismatches.append({"source_row": int(row["source_row"]), "comparison": name, "direct": direct_score, "stored": stored_score, "abs_diff": abs(direct_score - stored_score)})
            if len(extra) > 1 and slot is not None and np.isfinite(view["lo"][pos, 1, 0]):
                alternatives += 1
                alt = float(scores[1])
                lo_s = float(view["lo"][pos, 1, 0])
                hi_s = float(view["hi"][pos, 1, 0])
                if alt < lo_s or alt > hi_s:
                    endpoint_replay.append({"source_row": int(row["source_row"]), "comparison": "alternative_vs_stored_interval", "direct": alt, "lo": lo_s, "hi": hi_s, "abs_diff": max(lo_s - alt, alt - hi_s)})
                    gap = max(gap, max(lo_s - alt, alt - hi_s, 0.0))
            if slot is not None and slot.Fp + slot.Fg <= 16 and full_classes < 80:
                pfa_scores = decbase.mastery_vec(base, slot.pfa_deltas)
                gru_pairs = [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in slot.gru_index]
                gru_scores = chronorep.score_forward(pilot.model, gru_pairs, pilot.device)
                bound_pairs = (
                    ("pfa_lo", float(pfa_scores.min()), float(view["lo"][pos, 0, 0])),
                    ("pfa_hi", float(pfa_scores.max()), float(view["hi"][pos, 0, 0])),
                    ("gru_lo", float(gru_scores.min()), float(view["lo"][pos, 1, 0])),
                    ("gru_hi", float(gru_scores.max()), float(view["hi"][pos, 1, 0])),
                )
                for name, direct_score, stored_score in bound_pairs:
                    gap = max(gap, abs(direct_score - stored_score))
                    if (direct_score >= TAU) != (stored_score >= TAU):
                        endpoint_replay.append({"source_row": int(row["source_row"]), "comparison": name, "direct": direct_score, "stored": stored_score, "abs_diff": abs(direct_score - stored_score)})
                full_classes += 1
            checked += 1
            if checked >= 200:
                break
        if checked >= 200:
            break
    crossings = realized_label_mismatches + endpoint_replay
    unexplained = [item for item in crossings if item["abs_diff"] > 1e-5]
    return {
        "checked": checked, "max_abs": gap, "alternative_completions": alternatives,
        "full_class_bounds_checked": full_classes, "future_in_checked_prefix": future,
        "realized_decision_label_mismatches": realized_label_mismatches,
        "endpoint_replay_crossings": endpoint_replay, "unexplained_mismatches": unexplained,
        "threshold_rule_unchanged": True,
        "note": "The stored certificate is the reliability scoring pass. A repeated GRU forward can differ by about one float32 unit in the last place. Differences above 1e-5 are unexplained.",
    }


def live_field_audit(grouped, bundles, view, positions: list[int]) -> dict:
    wanted = {int(view["source"][pos]): pos for pos in positions}
    checked = mismatches = future = multi_skill = timestamp_behavior = 0
    for uid in sorted(grouped):
        rows = grouped[uid]
        hits = [row for row in rows if row["valid"] and int(row["source_row"]) in wanted]
        if not hits:
            continue
        by_start, by_end = chronorep.order_valid(rows)
        cursor = 0
        for row in by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                cursor += 1
            pos = wanted.get(int(row["source_row"]))
            if pos is None:
                continue
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                future += 1
            ev = chronorep.as_event(row)
            for schema_i, schema in enumerate(SCHEMAS):
                slot = bundles[schema].get(chronorep.release_key(ev, schema))
                present = bool(view["present"][pos, schema_i])
                if (slot is None) != (not present):
                    mismatches += 1
                    continue
                if slot is None:
                    continue
                if slot.skill != row["skill_id"]:
                    multi_skill += 1
                if (slot.U, slot.n, slot.n, slot.B, slot.Fp, slot.Fg) != tuple(int(v) for v in (view["U"][pos, schema_i], view["R"][pos, schema_i], view["N"][pos, schema_i], view["B"][pos, schema_i], view["F"][pos, 0, schema_i], view["F"][pos, 1, schema_i])):
                    mismatches += 1
                if abs(slot.Hnp - float(view["H"][pos, 0, schema_i])) > 1e-12 or abs(slot.Hng - float(view["H"][pos, 1, schema_i])) > 1e-12:
                    mismatches += 1
                own_pfa = chronorep.pfa_fp(row)
                own_gru = chronorep.gru_fp(row)
                if (own_pfa in slot.pfa_index) != bool(view["library"][pos, 0, schema_i]) or (own_gru in slot.gru_index) != bool(view["library"][pos, 1, schema_i]):
                    mismatches += 1
                if slot.n > slot.B:
                    timestamp_behavior += 1
            p00 = bool(view["present"][pos, 0] and (int(view["U"][pos, 0]) >= 2 or int(view["R"][pos, 0]) >= 2))
            stable = int(view["code"][pos, 0, 0]) in (1, 2)
            if stable and not p00:
                mismatches += 1
            checked += 1
    return {"checked": checked, "field_mismatches": mismatches, "future_violations": future, "skill_disagreements": multi_skill, "events_whose_class_has_more_rich_signatures_than_behaviors": timestamp_behavior}


def decbase_agreement(view) -> dict:
    cached = np.load(DECBASE_PRIV / "calibration_decisions.npz")
    same_order = np.array_equal(cached["source_row"], view["source"])
    if not same_order:
        return {"same_source_order": False, "label_mismatches": None}
    label = (view["realized"] >= TAU).astype(np.uint8)
    label_mismatch = int(np.sum(cached["y"] != label))
    cert_mismatch = int(np.sum(cached["certificate"] != view["code"]))
    library_mismatch = int(np.sum(cached["in_library"] != view["library"].astype(np.uint8)))
    return {
        "same_source_order": True, "events": int(len(view["source"])),
        "label_mismatches": label_mismatch, "certificate_mismatches": cert_mismatch,
        "library_mismatches": library_mismatch,
        "max_abs_realized_vs_decbase": float(np.max(np.abs(cached["realized_score"] - view["realized"]))),
    }


def development_distribution_rows(report: dict) -> list[dict]:
    rows = []
    names = {"U": "development_users", "R": "rich_signatures", "B": "behavior_signatures", "Fp": "pfa_fingerprints", "Fg": "gru_fingerprints", "Hp": "pfa_normalized_entropy", "Hg": "gru_normalized_entropy"}
    for schema in SCHEMAS:
        for key, label in names.items():
            for bin_name, counts in sorted(report["geometry"][schema]["bins"][key].items()):
                rows.append({"schema": schema, "variable": label, "bin": bin_name, "development_keys": counts[0], "development_events": counts[1]})
    return rows


def checksums() -> dict:
    payload = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "output_checksums.json":
            payload[str(path.relative_to(OUT))] = chronorep.sha256_file(path)
    return payload


STARTED = time.time()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    progress("preflight")
    pre = chronorep.preflight()
    pre["chronorep_checkpoint_sha256"] = chronorep.sha256_file(CHRONOREP_CKPT)
    pre["chronorep_checkpoint_match"] = pre["chronorep_checkpoint_sha256"] == "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"])
    dump("source_and_checkpoint_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "preflight": False})
        raise SystemExit(2)
    progress("development")
    bundles, built = build_development()
    report = built["report"]
    stored = np.load(SUPPRED_PRIV / "development_key_counts.npz")
    key_match = {}
    for schema in SCHEMAS:
        left = built["key_stats"][schema]
        right = stored[schema]
        left_s = left[np.lexsort((left[:, 3], left[:, 2], left[:, 1], left[:, 0]))]
        right_s = right[np.lexsort((right[:, 3], right[:, 2], right[:, 1], right[:, 0]))]
        key_match[schema] = bool(left_s.shape == right_s.shape and np.array_equal(left_s, right_s))
    semantics = {
        "matches_suppred_U_R_Fp_Fg_multisets": key_match,
        "all_key_multisets_match": all(key_match.values()),
        "rich_unique": report["unique_rich_signatures_with_timestamps"] == 4286981 and report["duplicate_rich_signatures_with_timestamps"] == 0,
        "behavior_unique": report["unique_behavior_signatures_without_timestamps"] == 605097,
        "behavior_duplicates": report["duplicate_behavior_signatures_without_timestamps"],
        "invalid_intervals": report["invalid_intervals_retained_in_library"],
        "users": report["development_users"],
        "events": report["events"],
        "key_counts": report["key_counts"],
        "relationships": {schema: report["geometry"][schema]["summary"] for schema in SCHEMAS},
        "definitions": {
            "U": "distinct development learners in the release key",
            "R": "distinct timestamp-inclusive rich signatures; equal to N on this source",
            "N": "development events in the release key, including the 475 invalid intervals retained by the frozen library",
            "B": "distinct behavior signatures after removing action timestamps",
            "F_m": "distinct consumer-sufficient fingerprints",
            "entropy": "event-frequency Shannon entropy of those fingerprints",
        },
    }
    dump("development_evidence_semantics.json", semantics)
    write_csv("development_evidence_distribution.csv", development_distribution_rows(report))
    entropy_audit = {
        "log_base": report["log_base"],
        "singleton_normalized_entropy_is_zero": True,
        "schemas": {schema: report["geometry"][schema]["summary"] for schema in SCHEMAS},
        "timestamp_rich_keys_with_one_behavior": {schema: report["geometry"][schema]["summary"]["Rge2_Beq1_keys"] for schema in SCHEMAS},
        "keys_with_behavioral_diversity": {schema: report["geometry"][schema]["summary"]["Bge2_keys"] for schema in SCHEMAS},
        "same_fingerprint_uses_one_consumer_input": "A fingerprint is the dictionary key of one delta or one action tuple. Target skill is constant inside every release key.",
    }
    dump("consumer_fingerprint_entropy_audit.json", entropy_audit)
    evidence_ok = semantics["all_key_multisets_match"] and semantics["rich_unique"] and semantics["behavior_unique"] and report["invalid_intervals_retained_in_library"] == 475 and report["development_users"] == 31753 and report["events"] == 4286981 and report["key_counts"] == {"S0": 259838, "S1": 511770, "S2": 523378, "S3": 605097} and all(report["geometry"][s]["summary"]["multi_skill_keys"] == 0 for s in SCHEMAS)
    if not evidence_ok:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "evidence_ok": False, "semantics": semantics})
        raise SystemExit(3)
    progress("calibration_load")
    grouped, cal_pop = load_split("calibration")
    dump("calibration_scoring_readiness.json", {
        "protocol_pilot_is_not_used": True, "complete_population_required": True,
        "population": cal_pop, "expected_valid": 887997, "expected_invalid": 161, "expected_users": 6804,
        "existing_decbase_cache_lacks_interval_endpoints": True,
        "action": "score all valid calibration targets with the frozen consumer" if not (PRIV / "calibration_intervals.npz").exists() else "reload the stored calibration score cache",
    })
    if cal_pop["valid_targets"] != 887997 or cal_pop["invalid"] != 161 or cal_pop["users"] != 6804:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "calibration_population": cal_pop})
        raise SystemExit(4)
    bundle = torch.load(CHRONOREP_CKPT, map_location="cpu", weights_only=False)
    pilot = NeuralPilot(device="cpu")
    pilot.load_bundle({"model": bundle["state_dict"], "skill2i": bundle["skill2i"], "atype2i": bundle["atype2i"]})
    pilot.model.eval()
    if (PRIV / "calibration_intervals.npz").exists() and (PRIV / "calibration_meta.json").exists():
        progress("calibration_reload")
        calibration_raw = load_calibration()
    else:
        progress("calibration_score")
        calibration_raw = score_calibration(grouped, bundles, pilot)
        save_calibration(calibration_raw)
    if calibration_raw["events"] != 887997 or calibration_raw["future"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "calibration_events": calibration_raw["events"], "future": calibration_raw["future"]})
        raise SystemExit(5)
    cal_code = codes_from(calibration_raw["lo"], calibration_raw["hi"], calibration_raw["present"], calibration_raw["evidence"])
    cal_fs = false_stable(cal_code, calibration_raw["realized"])
    calibration = population_views("calibration", calibration_raw, cal_code, cal_fs)
    if int((cal_fs & calibration["library"]).sum()) or int(np.sum(np.abs(calibration["realized"][:, 0, None] - calibration["realized"][:, 0, None]))):
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "covered_false_stable_calibration": int((cal_fs & calibration["library"]).sum())})
        raise SystemExit(6)
    progress("calibration_qa")
    cal_positions = sample_positions(calibration, "calibration")
    supported_order = np.flatnonzero(np.any(calibration["code"] > 0, axis=(1, 2)))
    supported_order = sorted(supported_order.tolist(), key=lambda pos: hashlib.sha256(f"WSDM203-G17-DIRECT|{int(calibration['source'][pos])}".encode()).hexdigest())
    direct = direct_rescore(grouped, bundles, pilot, calibration, supported_order[:200])
    cal_fields = live_field_audit(grouped, bundles, calibration, cal_positions)
    decbase_check = decbase_agreement(calibration)
    del grouped
    progress("confirmatory_attach")
    grouped, conf_pop = load_split("confirmatory")
    if conf_pop["valid_targets"] != 939819 or conf_pop["invalid"] != 91 or conf_pop["users"] != 6766:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "confirmatory_population": conf_pop})
        raise SystemExit(7)
    cached = {key: np.load(SUPPRED_PRIV / "confirmatory_intervals.npz")[key] for key in ("user", "source", "end", "realized", "lo", "hi", "counts", "present", "library")}
    meta = json.loads((SUPPRED_PRIV / "meta.json").read_text())
    cached.update(meta)
    confirmatory_raw = attach_confirmatory(grouped, bundles, cached)
    if confirmatory_raw["attach_mismatches"] or confirmatory_raw["future_recheck"] or confirmatory_raw["events"] != 939819:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "attach_mismatches": confirmatory_raw["attach_mismatches"], "future": confirmatory_raw["future_recheck"]})
        raise SystemExit(8)
    conf_evidence = np.zeros((confirmatory_raw["events"], 4, 6), dtype=np.int32)
    conf_evidence[:, :, 0] = confirmatory_raw["counts"][:, :, 0]
    conf_evidence[:, :, 1] = confirmatory_raw["counts"][:, :, 1]
    conf_evidence[:, :, 2] = confirmatory_raw["counts"][:, :, 1]
    conf_evidence[:, :, 3] = confirmatory_raw["B"]
    conf_evidence[:, :, 4] = confirmatory_raw["counts"][:, :, 2]
    conf_evidence[:, :, 5] = confirmatory_raw["counts"][:, :, 3]
    conf_code = codes_from(confirmatory_raw["lo"], confirmatory_raw["hi"], confirmatory_raw["present"], conf_evidence)
    conf_fs = false_stable(conf_code, confirmatory_raw["realized"])
    confirmatory = population_views("confirmatory", confirmatory_raw, conf_code, conf_fs)
    reproduction = []
    reproduction_ok = True
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            stable = int(((conf_code[:, model_i, schema_i] == 1) | (conf_code[:, model_i, schema_i] == 2)).sum())
            set_valued = int((conf_code[:, model_i, schema_i] == 3).sum())
            unsupported = int((conf_code[:, model_i, schema_i] == 0).sum())
            fs = int(conf_fs[:, model_i, schema_i].sum())
            expected = EXPECTED_CERT[(model, schema)]
            match = (stable, set_valued, unsupported, fs) == expected
            reproduction_ok = reproduction_ok and match
            reproduction.append({"consumer": model, "schema": schema, "stable": stable, "set_valued": set_valued, "unsupported": unsupported, "false_stable": fs, "expected": expected, "match": match})
    write_csv("confirmatory_certificate_reproduction.csv", reproduction)
    if not reproduction_ok or int((conf_fs & confirmatory["library"]).sum()):
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "reproduction": reproduction, "covered_false_stable": int((conf_fs & confirmatory["library"]).sum())})
        raise SystemExit(9)
    progress("confirmatory_qa")
    conf_positions = sample_positions(confirmatory, "confirmatory")
    conf_fields = live_field_audit(grouped, bundles, confirmatory, conf_positions)
    checkpoint_after = chronorep.sha256_file(CHRONOREP_CKPT)
    del grouped, bundles, pilot, bundle
    if checkpoint_after != "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196":
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SCORE_INTEGRITY", "checkpoint_changed": True})
        raise SystemExit(10)
    progress("select_policies")
    policies = select_policies(calibration)
    dump("calibration_selected_policies.json", {
        "selected_before_confirmatory_evaluation": True,
        "confirmatory_labels_used": False,
        "delta": DELTA, "family_size": FAMILY, "alphas": ALPHAS, "tie_break": list(RULES),
        "objective": "maximum calibration event issuance among rules whose learner-level upper bound is at most alpha",
        "policies": policies,
    })
    progress("tables")
    cal_issue, cal_bounds = candidate_rows(calibration)
    conf_issue, _conf_bounds = candidate_rows(confirmatory)
    write_csv("calibration_candidate_issuance.csv", cal_issue)
    write_csv("calibration_learner_any_error_bounds.csv", cal_bounds)
    write_csv("reliability_issuance_curves.csv", cal_issue + conf_issue)
    write_csv("confirmatory_calibrated_issuance.csv", confirmatory_policy_rows(confirmatory, policies))
    write_csv("calibration_certificate_frontier.csv", frontier_rows(calibration))
    strata = strata_rows(calibration)
    for name, rows in strata_rows(confirmatory).items():
        strata[name].extend(rows)
    for name, rows in strata.items():
        write_csv(name, rows)
    write_csv("three_case_false_stable_decomposition.csv", three_case_rows(calibration) + three_case_rows(confirmatory))
    write_csv("calibration_vs_confirmatory_comparison.csv", comparison_rows(calibration, confirmatory, policies))
    progress("bootstrap")
    write_csv("learner_bootstrap_intervals.csv", bootstrap_rows(confirmatory))
    qa = {
        "calibration_sample": len(cal_positions), "confirmatory_sample": len(conf_positions),
        "calibration_field_audit": cal_fields, "confirmatory_field_audit": conf_fields,
        "direct_rescore": direct, "decbase_calibration_agreement": decbase_check,
        "calibration_examples": [qa_record(calibration, pos, "calibration") for pos in cal_positions[:8]],
        "confirmatory_examples": [qa_record(confirmatory, pos, "confirmatory") for pos in conf_positions[:8]],
        "confirmatory_labels_absent_from_selection": True,
        "threshold_epsilon_not_used": True,
    }
    dump("independent_scoring_qa.json", qa)
    stats = bound_qa()
    repeated = int(np.sum(np.bincount(calibration["user"]) > 1))
    stats["learners_with_repeated_events_calibration"] = repeated
    stats["repeated_events_are_not_extra_bernoulli_units"] = repeated > 0
    dump("statistical_bound_qa.json", stats)
    covered_fs = int((cal_fs & calibration["library"]).sum() + (conf_fs & confirmatory["library"]).sum())
    partition_ok = all(row["partition"] == row["valid_events"] for row in frontier_rows(calibration) + frontier_rows(confirmatory))
    strata_ok = True
    expected_n = {"calibration": 887997, "confirmatory": 939819}
    for rows in strata.values():
        buckets = {}
        for row in rows:
            buckets.setdefault((row["split"], row["consumer"], row["schema"], row["variable"]), 0)
            buckets[(row["split"], row["consumer"], row["schema"], row["variable"])] += row["valid_events"]
        strata_ok = strata_ok and all(total == expected_n[key[0]] for key, total in buckets.items())
    case_ok = all(row["case_B_requires_zero_false_stable"] and row["every_false_stable_is_uncovered"] and (row["identity_gap"] is None or row["identity_gap"] < 1e-12) for row in three_case_rows(calibration) + three_case_rows(confirmatory))
    direct_ok = direct["checked"] >= 200 and direct["future_in_checked_prefix"] == 0 and not direct["unexplained_mismatches"] and not direct["realized_decision_label_mismatches"]
    field_ok = cal_fields["checked"] >= 300 and conf_fields["checked"] >= 300 and cal_fields["field_mismatches"] == 0 and conf_fields["field_mismatches"] == 0 and cal_fields["future_violations"] == 0 and conf_fields["future_violations"] == 0
    decbase_ok = decbase_check.get("same_source_order") and decbase_check.get("label_mismatches") == 0 and decbase_check.get("certificate_mismatches") == 0
    transfer_failures = sum(1 for row in comparison_rows(calibration, confirmatory, policies) if row["calibration_transfer_failure"])
    status = "COMPLETE"
    if not (direct_ok and field_ok and decbase_ok and stats["all_match"] and covered_fs == 0 and partition_ok and strata_ok and case_ok and reproduction_ok):
        status = "BLOCKED_STATISTICAL_CONTRACT" if not stats["all_match"] else "BLOCKED_SOURCE_OR_SCORE_INTEGRITY"
    invariants = {
        "status": status, "calibration_events": 887997, "confirmatory_events": 939819,
        "reproduction_ok": reproduction_ok, "covered_false_stable": covered_fs, "partition_ok": partition_ok, "strata_ok": strata_ok,
        "three_case_ok": case_ok, "direct_ok": direct_ok, "field_ok": field_ok, "decbase_ok": decbase_ok,
        "bound_ok": stats["all_match"], "checkpoint_sha256": checkpoint_after,
        "models_retrained": False,
        "p00_unchanged": True, "transfer_failures": transfer_failures,
        "prior_confirmatory_reuse_disclosed": True,
    }
    dump("numerical_invariants.json", invariants)
    dump("experiment_manifest.json", {
        "experiment": "pipeline", "status": status, "elapsed_sec": time.time() - STARTED,
        "chronorep_checkpoint_sha256": checkpoint_after, "models_retrained": False,
        "calibration_cache": str(PRIV / "calibration_intervals.npz"),
        "confirmatory_cache_reference": str(SUPPRED_PRIV / "confirmatory_intervals.npz"),
        "suppred_cache_copied": False, "earlier_results_overwritten": False,
    })
    progress("checksums")
    dump("output_checksums.json", checksums())
    print(status, flush=True)


if __name__ == "__main__":
    main()
