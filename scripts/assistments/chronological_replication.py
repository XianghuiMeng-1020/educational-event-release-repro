#!/usr/bin/env python3
"""Full chronology-valid ASSISTments replication.

Trains the primary seed-17 ActionTokenGRU and evaluates the S0-S3 release
frontier on the confirmatory population. The frozen core, the frozen
checkpoints, and the historical results are not modified.
"""
from __future__ import annotations

import gc
import hashlib
import json
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from r7_core import (  # noqa: E402
    MAX_EVAL_PREFIX_EVENTS,
    MAX_HIST_TOKENS,
    MAX_SEQ_ACTIONS,
    MAX_TRAIN_EVENTS_PER_USER,
    NEURAL_SEED,
    TAU,
    ActionTokenGRU,
    Event,
    NeuralPilot,
    TransparentActionPFA,
    project_down,
    release_key,
)

OUT = ROOT / "results/assistments/chronological_replication"
STAGE = ROOT / "data/raw/assistments/assistments_2012.zip"
RECON = ROOT / "data/processed/assistments"
FRONTIER = ROOT / "results/wdlf_r7/primary_frontier.json"
HIST_CKPT = ROOT / "prospective/wdlf_r7/sealed/checkpoints/action_token_gru_seed17.pt"
CORE = ROOT / "scripts/wdlf_r7/r7_core.py"
NAT = np.iinfo(np.int64).min
SCHEMAS = ("S0", "S1", "S2", "S3")
TAUS = (0.70, 0.80, 0.90)
KAPPAS = (0.01, 0.05, 0.10)
EXPECTED_ZIP = "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445"
EXPECTED_CORE = "dabfb38cd36222e7e7663417b97f60a91caca0b39afc958ccb7f052b45ef93fe"
EXPECTED_HIST = "acd9b0c1650c13ad0739506db1d861e86e77f4be98f48c6c01f60520252dace2"
EXPECTED = {
    "development": ("69b6b7b38738d293651b728326658c1ce517c87661d012bc03ee2ced228e9d70", 31753, 4286981, 475),
    "calibration": ("5e8202d6f89b501a1aaea662ba62a265de1e5c168a0bcf92e3762760b4c6fd3d", 6804, 888158, 161),
    "confirmatory": ("0193ddd7b26517f1b48591d09d86f5f3ce8b9930a1285006833fce4ea5096f99", 6766, 939910, 91),
}
EXPECTED_KEYS = {"S0": 259838, "S1": 511770, "S2": 523378, "S3": 605097}
EXPECTED_SELECTION = {"epoch_sizes": [3000, 3000], "union": 5715, "intersection": 285}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def civil_us(values: list[str]) -> np.ndarray:
    parsed = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed")
    raw = parsed.to_numpy(dtype="datetime64[us]")
    out = raw.astype(np.int64)
    out[np.isnat(raw)] = NAT
    return out


def dump(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def compact_row(row: dict, start: int, end: int) -> dict:
    actions = tuple(str(tok).lower() for tok in (row["action_types"] or []))
    return {
        "source_row": int(row["source_row"]),
        "user_id": str(row["user_id"]),
        "problem_id": str(row["problem_id"]),
        "skill_id": str(row["skill_id"]),
        "correct": int(row["correct"]),
        "hint_count": int(row["hint_count"]),
        "attempt_count": int(row["attempt_count"]),
        "first_action": str(row["first_action"]),
        "action_types": actions,
        "start": int(start),
        "end": int(end),
        "valid": bool(start != NAT and end != NAT and end >= start),
        "split": str(row["split"]),
    }


def as_event(row: dict) -> Event:
    return Event(
        row["user_id"], row["problem_id"], row["skill_id"], row["correct"], row["hint_count"],
        row["attempt_count"], row["first_action"], row["action_types"], tuple(), row["split"],
    )


def pfa_fp(row: dict) -> tuple:
    return (row["correct"], row["hint_count"], row["attempt_count"], row["action_types"])


def gru_fp(row: dict) -> tuple:
    return row["action_types"][:MAX_SEQ_ACTIONS]


def fp_delta(fp: tuple) -> tuple[float, ...]:
    actions = fp[3]
    answers = 0.0
    hints_before = 0.0
    seen_answer = False
    for tok in actions:
        if tok == "answer" and not seen_answer:
            seen_answer = True
        if not seen_answer and tok == "hint":
            hints_before += 1.0
        if tok == "answer":
            answers += 1.0
    first_hint = 1.0 if actions and actions[0] == "hint" else 0.0
    return (1.0, float(fp[0]), float(fp[1]), float(max(fp[2], 1)), answers, hints_before, first_hint)


def mastery_of(state: np.ndarray) -> float:
    n = state[0]
    if n <= 0:
        return 0.5
    base = (state[1] + 1.0) / (n + 2.0)
    score = (
        base
        - 0.05 * state[2] / n
        - 0.04 * max(state[3] / n - 1.0, 0.0)
        - 0.03 * max(state[4] / n - 1.0, 0.0)
        - 0.06 * state[5] / n
        - 0.05 * state[6] / n
    )
    return float(max(0.01, min(0.99, score)))


def budget(n: int, kappa: float) -> int:
    if n <= 0:
        return 0
    seats = int(round(float(kappa) * int(n)))
    if kappa > 0 and seats < 1:
        seats = 1
    return min(seats, n)


def adversarial_ranks(los: np.ndarray, his: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(los))
    if n == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty
    order_lo = np.sort(los)
    order_hi = np.sort(his)
    n_lo_gt = n - np.searchsorted(order_lo, his, side="right")
    self_best = (los > his).astype(np.int32)
    best = (1 + n_lo_gt - self_best).astype(np.int32)
    n_hi_ge = n - np.searchsorted(order_hi, los, side="left")
    self_worst = (his >= los).astype(np.int32)
    worst = np.where(self_worst == 1, n_hi_ge, 1 + n_hi_ge).astype(np.int32)
    return best, worst


def labels_from_ranks(best: np.ndarray, worst: np.ndarray, seats: int) -> np.ndarray:
    out = np.full(len(best), "unresolved", dtype=object)
    if seats <= 0 or len(best) == 0:
        return out
    out[worst <= seats] = "guaranteed_in"
    out[(best > seats) & (worst > seats)] = "guaranteed_out"
    return out


def membership_counts(los: np.ndarray, his: np.ndarray, kappa: float) -> dict[str, int]:
    seats = budget(len(los), kappa)
    labels = labels_from_ranks(*adversarial_ranks(los, his), seats)
    return {
        "B": seats,
        "guaranteed_in": int(np.sum(labels == "guaranteed_in")),
        "guaranteed_out": int(np.sum(labels == "guaranteed_out")),
        "unresolved": int(np.sum(labels == "unresolved")),
        "labels": labels,
    }


def preflight() -> dict:
    payload = {"zip_sha256": sha256_file(STAGE), "zip_bytes": STAGE.stat().st_size, "parquet": {}, "frozen": {}}
    payload["zip_match"] = payload["zip_sha256"] == EXPECTED_ZIP and payload["zip_bytes"] == 576365408
    payload["frozen"]["r7_core_sha256"] = sha256_file(CORE)
    payload["frozen"]["r7_core_match"] = payload["frozen"]["r7_core_sha256"] == EXPECTED_CORE
    payload["frozen"]["historical_seed17_sha256"] = sha256_file(HIST_CKPT)
    payload["frozen"]["historical_seed17_bytes"] = HIST_CKPT.stat().st_size
    payload["frozen"]["historical_seed17_match"] = payload["frozen"]["historical_seed17_sha256"] == EXPECTED_HIST
    for split, (digest, users, events, invalid) in EXPECTED.items():
        path = RECON / f"{split}.parquet"
        observed = sha256_file(path)
        payload["parquet"][split] = {
            "sha256": observed,
            "bytes": path.stat().st_size,
            "expected_users": users,
            "expected_events": events,
            "expected_invalid": invalid,
            "match": observed == digest,
        }
    payload["pass"] = bool(payload["zip_match"] and payload["frozen"]["r7_core_match"] and payload["frozen"]["historical_seed17_match"] and all(item["match"] for item in payload["parquet"].values()))
    return payload


def load_order(split: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_parquet(RECON / f"{split}.parquet", columns=["source_row", "user_id", "start_time", "end_time"])
    frame["user_id"] = frame["user_id"].astype(str)
    frame = frame.sort_values("source_row", kind="mergesort")
    starts = civil_us(frame["start_time"].tolist())
    ends = civil_us(frame["end_time"].tolist())
    frame["start_us"] = starts
    frame["end_us"] = ends
    frame["valid"] = (starts != NAT) & (ends != NAT) & (ends >= starts)
    seen = []
    known = set()
    for uid in frame["user_id"].tolist():
        if uid not in known:
            known.add(uid)
            seen.append(uid)
    return frame, seen


def replay_selection(first_seen: list[str]) -> dict:
    users = first_seen.copy()
    random.seed(NEURAL_SEED)
    chosen = []
    for _ in range(2):
        random.shuffle(users)
        chosen.append(users[:3000].copy())
    union = set(chosen[0]) | set(chosen[1])
    inter = set(chosen[0]) & set(chosen[1])
    digest = hashlib.sha256()
    for cohort in chosen:
        digest.update(str(len(cohort)).encode())
        for uid in cohort:
            digest.update(uid.encode())
            digest.update(b"\n")
    return {
        "epoch_sizes": [len(chosen[0]), len(chosen[1])],
        "union": len(union),
        "intersection": len(inter),
        "eligible_users": len(first_seen),
        "selection_sha256": digest.hexdigest(),
        "matches_tempfor": [len(chosen[0]), len(chosen[1])] == EXPECTED_SELECTION["epoch_sizes"] and len(union) == EXPECTED_SELECTION["union"] and len(inter) == EXPECTED_SELECTION["intersection"],
        "cohorts": chosen,
        "selected_users": union,
    }


class Slot:
    __slots__ = ("n", "n_users", "first_user", "pfa", "pfa_list", "gru", "deltas")

    def __init__(self) -> None:
        self.n = 0
        self.n_users = 0
        self.first_user = None
        self.pfa = set()
        self.pfa_list = []
        self.gru = set()
        self.deltas = None


def build_development(selected: set[str]) -> dict:
    bundles = {schema: {} for schema in SCHEMAS}
    skill_order = []
    skill_seen = set()
    atype_order = []
    atype_seen = set()
    signatures = set()
    duplicate_signatures = 0
    nesting_violations = 0
    events = 0
    kept = defaultdict(list)
    parquet = pq.ParquetFile(RECON / "development.parquet")
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        times_s = civil_us([row["start_time"] for row in rows])
        times_e = civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, times_s, times_e):
            events += 1
            item = compact_row(row, int(start), int(end))
            ev = as_event(item)
            if ev.skill_id not in skill_seen:
                skill_seen.add(ev.skill_id)
                skill_order.append(ev.skill_id)
            for tok in ev.action_types:
                if tok not in atype_seen:
                    atype_seen.add(tok)
                    atype_order.append(tok)
            digest = hashlib.blake2s(repr((ev.problem_id, ev.correct, ev.hint_count, ev.attempt_count, ev.first_action, ev.action_types, tuple(str(tok) if tok is not None else "" for tok in (row["action_timestamps"] or [])))).encode(), digest_size=16).digest()
            if digest in signatures:
                duplicate_signatures += 1
            else:
                signatures.add(digest)
            keys = {schema: release_key(ev, schema) for schema in SCHEMAS}
            if project_down(keys["S3"], "S3", "S2") != keys["S2"] or project_down(keys["S2"], "S2", "S1") != keys["S1"] or project_down(keys["S1"], "S1", "S0") != keys["S0"]:
                nesting_violations += 1
            fp = pfa_fp(item)
            gp = gru_fp(item)
            for schema, key in keys.items():
                slot = bundles[schema].get(key)
                if slot is None:
                    slot = Slot()
                    bundles[schema][key] = slot
                slot.n += 1
                if slot.n_users < 2:
                    if slot.first_user is None:
                        slot.first_user = item["user_id"]
                        slot.n_users = 1
                    elif slot.first_user != item["user_id"]:
                        slot.n_users = 2
                        slot.first_user = None
                slot.pfa.add(fp)
                slot.gru.add(gp)
            if item["user_id"] in selected:
                kept[item["user_id"]].append(item)
        if events % 500000 == 0:
            print(f"development_geometry {events} rss_mb={rss_mb():.1f}", flush=True)
    for bundle in bundles.values():
        for slot in bundle.values():
            slot.pfa_list = list(slot.pfa)
            slot.deltas = [fp_delta(fp) for fp in slot.pfa_list]
    nesting_fp = fingerprint_nesting(bundles)
    return {
        "events": events,
        "duplicate_signatures": duplicate_signatures,
        "unique_signatures": len(signatures),
        "nesting_violations": nesting_violations,
        "fingerprint_nesting": nesting_fp,
        "key_counts": {schema: len(bundle) for schema, bundle in bundles.items()},
        "skill_order": skill_order,
        "atype_order": atype_order,
        "bundles": bundles,
        "training_rows": kept,
    }


def fingerprint_nesting(bundles: dict) -> dict:
    violations = Counter()
    checked = 0
    for key, slot in bundles["S3"].items():
        checked += 1
        s2 = ("S2",) + key[1:6]
        s1 = ("S1",) + key[1:5]
        s0 = ("S0",) + key[1:3]
        for parent, child_key in ((s2, "S2"), (s1, "S1"), (s0, "S0")):
            parent_slot = bundles[child_key].get(parent)
            if parent_slot is None or not slot.pfa <= parent_slot.pfa or not slot.gru <= parent_slot.gru:
                violations[f"S3_not_subset_{child_key}"] += 1
    return {"s3_keys_checked": checked, "violations": dict(violations)}


def supported(slot: Slot | None) -> bool:
    return slot is not None and (slot.n_users >= 2 or slot.n >= 2)


def load_confirmatory() -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    parquet = pq.ParquetFile(RECON / "confirmatory.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = civil_us([row["start_time"] for row in rows])
        ends = civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            grouped[str(row["user_id"])].append(compact_row(row, int(start), int(end)))
        seen += len(rows)
        if seen % 200000 == 0:
            print(f"confirmatory_load {seen}", flush=True)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["source_row"])
    return grouped


def order_valid(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    valid = [row for row in rows if row["valid"]]
    by_start = sorted(valid, key=lambda item: (item["start"], item["source_row"]))
    by_end = sorted(valid, key=lambda item: (item["end"], item["source_row"]))
    return by_start, by_end


def history_cut(by_end: list[dict], start: int) -> int:
    ends = [item["end"] for item in by_end]
    return int(np.searchsorted(np.asarray(ends, dtype=np.int64), start, side="left"))


def direct_history(rows: list[dict], target: dict) -> list[int]:
    chosen = [row["source_row"] for row in rows if row["valid"] and row["source_row"] != target["source_row"] and row["end"] < target["start"]]
    chosen.sort(key=lambda source: next((row["end"], row["source_row"]) for row in rows if row["source_row"] == source))
    return chosen


def plan_checkpoints(grouped: dict[str, list[dict]], bundles: dict) -> dict:
    users = sorted(grouped)
    plans = {}
    changed = {schema: 0 for schema in SCHEMAS}
    eligible = {schema: 0 for schema in SCHEMAS}
    matched_users = 0
    intervening = 0
    valid_targets = 0
    invalid_targets = 0
    reference_mismatches = 0
    reference_checked = 0
    sample_ids = sorted(users, key=lambda uid: hashlib.sha256(f"WSDM203-G8-CHECKPOINT|{uid}".encode()).hexdigest())[:80]
    for uid in users:
        by_start, by_end = order_valid(grouped[uid])
        valid_targets += len(by_start)
        invalid_targets += sum(1 for row in grouped[uid] if not row["valid"])
        ends = np.asarray([row["end"] for row in by_end], dtype=np.int64)
        support = {schema: [] for schema in SCHEMAS}
        all_four = []
        for row in by_start:
            ev = as_event(row)
            keys = {schema: release_key(ev, schema) for schema in SCHEMAS}
            flags = {schema: supported(bundles[schema].get(keys[schema])) for schema in SCHEMAS}
            for schema in SCHEMAS:
                if flags[schema]:
                    support[schema].append(row)
            if all(flags.values()):
                all_four.append(row)
            cut = int(np.searchsorted(ends, row["start"], side="left"))
            if any(item["end"] >= row["start"] for item in by_end[:cut]):
                reference_mismatches += 1
            during = [item for item in by_end[cut:] if item["source_row"] != row["source_row"] and item["end"] <= row["end"]]
            if during:
                intervening += 1
        choice = {}
        for schema in SCHEMAS:
            if not support[schema]:
                choice[schema] = None
                continue
            eligible[schema] += 1
            by_completion = max(support[schema], key=lambda item: (item["end"], item["source_row"]))
            by_start_last = max(support[schema], key=lambda item: (item["start"], item["source_row"]))
            if by_completion["source_row"] != by_start_last["source_row"]:
                changed[schema] += 1
            choice[schema] = {"completion": by_completion["source_row"], "start": by_start_last["source_row"]}
        matched = None
        if all_four:
            matched_users += 1
            matched = max(all_four, key=lambda item: (item["end"], item["source_row"]))["source_row"]
        plans[uid] = {"schemas": choice, "matched": matched}
        if uid in sample_ids:
            for schema in SCHEMAS:
                rows = support[schema]
                if not rows:
                    continue
                reference_checked += 1
                brute_end = sorted(rows, key=lambda item: (item["end"], item["source_row"]))[-1]["source_row"]
                brute_start = sorted(rows, key=lambda item: (item["start"], item["source_row"]))[-1]["source_row"]
                if brute_end != choice[schema]["completion"] or brute_start != choice[schema]["start"]:
                    reference_mismatches += 1
    return {
        "plans": plans,
        "valid_targets": valid_targets,
        "invalid_targets": invalid_targets,
        "eligible_users": eligible,
        "checkpoint_identity_differs": changed,
        "matched_users": matched_users,
        "targets_with_a_completion_during_the_interaction": intervening,
        "reference_users": len(sample_ids),
        "reference_schema_checks": reference_checked,
        "reference_mismatches": reference_mismatches,
    }


def history_reference(grouped: dict[str, list[dict]]) -> dict:
    users = sorted(grouped)
    pool = []
    for uid in users:
        for row in grouped[uid]:
            if row["valid"]:
                pool.append((hashlib.sha256(f"WSDM203-G8-HISTORY|{row['source_row']}".encode()).hexdigest(), uid, row))
    pool.sort()
    mismatches = 0
    future = 0
    for _rank, uid, target in pool[:200]:
        _by_start, by_end = order_valid(grouped[uid])
        cut = history_cut(by_end, target["start"])
        fast = [item["source_row"] for item in by_end[:cut]]
        direct_rows = [row for row in grouped[uid] if row["valid"] and row["end"] < target["start"]]
        direct_rows.sort(key=lambda item: (item["end"], item["source_row"]))
        direct = [item["source_row"] for item in direct_rows]
        if fast != direct:
            mismatches += 1
        if any(item["end"] >= target["start"] for item in by_end[:cut]):
            future += 1
    return {"samples": min(200, len(pool)), "identity_mismatches": mismatches, "future_members": future}


def independent_tokens(pilot: NeuralPilot, prefix: list[Event], current: Event | None) -> tuple[list[int], list[int]]:
    skills: list[int] = []
    atypes: list[int] = []
    for ev in prefix:
        sid = pilot.skill2i.get(ev.skill_id, 0)
        for tok in ev.action_types[:MAX_SEQ_ACTIONS]:
            skills.append(sid)
            atypes.append(pilot.atype2i.get(tok, 0))
        skills.append(sid)
        atypes.append(pilot.model.outcome1 if ev.correct == 1 else pilot.model.outcome0)
    if current is not None:
        sid = pilot.skill2i.get(current.skill_id, 0)
        for tok in current.action_types[:MAX_SEQ_ACTIONS]:
            skills.append(sid)
            atypes.append(pilot.atype2i.get(tok, 0))
    if not skills:
        return [], []
    return skills[-MAX_HIST_TOKENS:], atypes[-MAX_HIST_TOKENS:]


def make_pilot(skill_order: list[str], atype_order: list[str]) -> NeuralPilot:
    random.seed(NEURAL_SEED)
    np.random.seed(NEURAL_SEED)
    torch.manual_seed(NEURAL_SEED)
    pilot = NeuralPilot()
    for skill in skill_order:
        pilot.skill2i[skill] = len(pilot.skill2i)
    for tok in atype_order:
        pilot.atype2i[tok] = len(pilot.atype2i)
    pilot.model = ActionTokenGRU(len(pilot.skill2i) - 1, len(pilot.atype2i) - 1)
    pilot.model.to(pilot.device)
    return pilot


def prefix_events(by_end: list[dict], start: int) -> list[Event]:
    cut = history_cut(by_end, start)
    return [as_event(item) for item in by_end[max(0, cut - MAX_EVAL_PREFIX_EVENTS) : cut]]


def token_audit(pilot: NeuralPilot, training_rows: dict[str, list[dict]]) -> dict:
    users = sorted(training_rows)
    pool = []
    prepared = {}
    for uid in users:
        by_start, by_end = order_valid(training_rows[uid])
        prepared[uid] = (by_start, by_end)
        ordered = sorted(training_rows[uid], key=lambda item: item["source_row"])
        window = ordered[-MAX_TRAIN_EVENTS_PER_USER:]
        for row in window:
            if row["valid"]:
                pool.append((hashlib.sha256(f"WSDM203-G8-TOKEN|{row['source_row']}".encode()).hexdigest(), uid, row))
    pool.sort()
    mismatches = 0
    leaks = 0
    future = 0
    checked = 0
    for _rank, uid, row in pool[:200]:
        by_start, by_end = prepared[uid]
        prefix = prefix_events(by_end, row["start"])
        current = as_event(row)
        left = pilot._tokens(prefix, current, False)
        right = independent_tokens(pilot, prefix, current)
        checked += 1
        if left != right:
            mismatches += 1
        current_n = min(MAX_SEQ_ACTIONS, len(current.action_types))
        if current_n and any(tok in {pilot.model.outcome0, pilot.model.outcome1} for tok in left[1][-current_n:]):
            leaks += 1
        cut = history_cut(by_end, row["start"])
        if any(item["end"] >= row["start"] for item in by_end[:cut]):
            future += 1
    return {"cases": checked, "token_mismatches": mismatches, "current_outcome_leaks": leaks, "future_history_members": future}


def train_model(pilot: NeuralPilot, training_rows: dict[str, list[dict]], cohorts: list[list[str]]) -> dict:
    opt = torch.optim.Adam(pilot.model.parameters(), lr=1e-3)
    pilot.model.train()
    losses = []
    used = []
    skipped_invalid = []
    skipped_empty = []
    selected_targets = []
    violations = 0
    for epoch, cohort in enumerate(cohorts, start=1):
        total = 0.0
        n = 0
        invalid = 0
        empty = 0
        targets = 0
        for user_i, uid in enumerate(cohort):
            rows = training_rows[uid]
            ordered = sorted(rows, key=lambda item: item["source_row"])
            window = ordered[-MAX_TRAIN_EVENTS_PER_USER:]
            by_start, by_end = order_valid(rows)
            for row in window:
                targets += 1
                if not row["valid"]:
                    invalid += 1
                    continue
                prefix = prefix_events(by_end, row["start"])
                current = as_event(row)
                if any(item["end"] >= row["start"] for item in by_end[: history_cut(by_end, row["start"])]):
                    violations += 1
                skills, atypes = pilot._tokens(prefix, current, False)
                if not skills:
                    empty += 1
                    continue
                pred, _ = pilot.model.forward_from_state(
                    torch.tensor([skills], dtype=torch.long, device=pilot.device),
                    torch.tensor([atypes], dtype=torch.long, device=pilot.device),
                    None,
                )
                y = torch.tensor([float(current.correct)], device=pilot.device)
                loss = torch.nn.functional.binary_cross_entropy(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item())
                n += 1
            if user_i and user_i % 500 == 0:
                print(f"train_epoch {epoch} users {user_i} examples {n}", flush=True)
        losses.append(total / n if n else None)
        used.append(n)
        skipped_invalid.append(invalid)
        skipped_empty.append(empty)
        selected_targets.append(targets)
        print(f"train_epoch_done {epoch} loss {losses[-1]} examples {n} invalid {invalid}", flush=True)
    pilot.model.eval()
    path = OUT / "gru_chronological_seed17.pt"
    torch.save(
        {
            "label": "CHRONOLOGICAL_PRIMARY_SEED17",
            "primary_checkpoint": True,
            "historical_file_order_checkpoint": False,
            "chronoproto_smoke_checkpoint": False,
            "seed": NEURAL_SEED,
            "epochs": 2,
            "state_dict": {key: value.detach().cpu() for key, value in pilot.model.state_dict().items()},
            "skill2i": pilot.skill2i,
            "atype2i": pilot.atype2i,
        },
        path,
    )
    return {
        "label": "CHRONOLOGICAL_PRIMARY_SEED17",
        "checkpoint": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_bytes": path.stat().st_size,
        "parameter_count": int(sum(p.numel() for p in pilot.model.parameters())),
        "n_skills": len(pilot.skill2i) - 1,
        "n_action_types": len(pilot.atype2i) - 1,
        "epochs": 2,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss": "binary_cross_entropy",
        "loss_by_epoch": losses,
        "examples_by_epoch": used,
        "selected_file_order_targets_by_epoch": selected_targets,
        "invalid_targets_excluded_by_epoch": skipped_invalid,
        "empty_token_examples_skipped_by_epoch": skipped_empty,
        "history_contract_violations": violations,
        "fit_called": False,
        "fit_edited": False,
        "max_rss_mb": rss_mb(),
    }


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "roc_auc": None, "log_loss": None, "brier": None, "positive_rate": None}
    y = y.astype(float)
    p = np.clip(p.astype(float), 1e-12, 1.0 - 1e-12)
    pos = y == 1.0
    if pos.all() or (~pos).all():
        auc = None
    else:
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(len(p), dtype=float)
        ranks[order] = np.arange(1, len(p) + 1)
        sorted_p = p[order]
        start = 0
        while start < len(p):
            stop = start
            while stop + 1 < len(p) and sorted_p[stop + 1] == sorted_p[start]:
                stop += 1
            if stop > start:
                ranks[order[start : stop + 1]] = 0.5 * (start + 1 + stop + 1)
            start = stop + 1
        n_pos = int(pos.sum())
        n_neg = int(len(y) - n_pos)
        auc = float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": auc,
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def score_forward(model: ActionTokenGRU, pairs: list[tuple[list[int], list[int]]], device) -> np.ndarray:
    out = np.full(len(pairs), 0.5, dtype=np.float64)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, (skills, _atypes) in enumerate(pairs):
        if skills:
            buckets[len(skills)].append(i)
    with torch.inference_mode():
        for length, indexes in buckets.items():
            for start in range(0, len(indexes), 512):
                batch = indexes[start : start + 512]
                skills = torch.tensor([pairs[i][0] for i in batch], dtype=torch.long, device=device)
                atypes = torch.tensor([pairs[i][1] for i in batch], dtype=torch.long, device=device)
                pred, _ = model.forward_from_state(skills, atypes, None)
                out[np.asarray(batch, dtype=np.int64)] = pred.detach().cpu().numpy()
    return out


def sequence_for(pilot: NeuralPilot, prefix: list[Event], actions: tuple[str, ...], skill: str) -> tuple[list[int], list[int]]:
    current = Event("", "", skill, 0, 0, 0, "", actions, tuple(), "")
    return pilot._tokens(prefix, current, False)


def evaluate(pilot: NeuralPilot, grouped: dict[str, list[dict]], bundles: dict, plans: dict) -> dict:
    users = sorted(grouped)
    user_index = {uid: i for i, uid in enumerate(users)}
    n_users = len(users)
    status = np.zeros((n_users, 2, 4, 3, 3), dtype=np.int32)
    false_stable = np.zeros((n_users, 2, 4), dtype=np.int32)
    uncovered_stable = np.zeros((n_users, 2, 4), dtype=np.int32)
    covered = np.zeros((n_users, 2, 4), dtype=np.int32)
    contained = np.zeros((n_users, 2, 4), dtype=np.int32)
    supported_n = np.zeros((n_users, 2, 4), dtype=np.int32)
    signatures = np.zeros((2, 4), dtype=np.int64)
    fingerprints = np.zeros((2, 4), dtype=np.int64)
    model_names = ("TransparentActionPFA", "ActionTokenGRU")
    transitions = {(model, tau, left, right): np.zeros((3, 3), dtype=np.int64) for model in model_names for tau in TAUS for left, right in (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3"))}
    nested_violations = Counter()
    completion = {name: {schema: {"lo": np.full(n_users, np.nan), "hi": np.full(n_users, np.nan), "actual": np.full(n_users, np.nan)} for schema in SCHEMAS} for name in model_names}
    start_ck = {name: {schema: {"lo": np.full(n_users, np.nan), "hi": np.full(n_users, np.nan), "actual": np.full(n_users, np.nan)} for schema in SCHEMAS} for name in model_names}
    matched = {name: {schema: {"lo": np.full(n_users, np.nan), "hi": np.full(n_users, np.nan), "actual": np.full(n_users, np.nan)} for schema in SCHEMAS} for name in model_names}
    pred_y = []
    pred_gru = []
    pred_user = []
    pred_s0 = []
    pred_s1 = []
    pred_s0_user = []
    pred_y_pfa = []
    equivalence = {"events": 0, "max_abs_diff": 0.0, "action_mismatches": 0, "order_mismatches": 0}
    sample_rank = []
    for uid in users:
        for row in grouped[uid]:
            if row["valid"]:
                sample_rank.append((hashlib.sha256(f"WSDM203-G8-SCORE|{row['source_row']}".encode()).hexdigest(), uid, row["source_row"]))
    sample_rank.sort()
    sample_keys = {(uid, source) for _rank, uid, source in sample_rank[:200]}
    future_violations = 0
    events_seen = 0
    for user_i, uid in enumerate(users):
        by_start, by_end = order_valid(grouped[uid])
        pfa = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        pred0 = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        pred1 = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        cursor = 0
        realized_pairs = []
        extra_pairs = []
        extra_meta = []
        event_cache = []
        for row in by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                done = by_end[cursor]
                delta = fp_delta(pfa_fp(done))
                pfa[done["skill_id"]] += delta
                state0 = pred0[done["skill_id"]]
                state0[0] += 1.0
                state0[1] += float(done["correct"])
                state1 = pred1[done["skill_id"]]
                state1[0] += 1.0
                state1[1] += float(done["correct"])
                state1[2] += float(done["hint_count"])
                state1[3] += float(max(done["attempt_count"], 1))
                cursor += 1
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                future_violations += 1
            ev = as_event(row)
            keys = {schema: release_key(ev, schema) for schema in SCHEMAS}
            slots = {schema: bundles[schema].get(keys[schema]) for schema in SCHEMAS}
            flags = {schema: supported(slots[schema]) for schema in SCHEMAS}
            prefix = [as_event(item) for item in by_end[max(0, cursor - MAX_EVAL_PREFIX_EVENTS) : cursor]]
            realized_pairs.append(pilot._tokens(prefix, ev, False))
            base = pfa[row["skill_id"]].copy()
            pfa_scores = {}
            for schema in SCHEMAS:
                slot = slots[schema]
                if not flags[schema]:
                    pfa_scores[schema] = None
                    continue
                values = [mastery_of(base + delta) for delta in slot.deltas]
                own = pfa_fp(row)
                realized = None
                for fp, value in zip(slot.pfa_list, values):
                    if fp == own:
                        realized = value
                if realized is None:
                    realized = mastery_of(base + np.asarray(fp_delta(own), dtype=np.float64))
                pfa_scores[schema] = (min(values), max(values), realized, own in slot.pfa, slot.n, len(slot.pfa))
            if len(prefix) > 0:
                pred_s0.append(mastery_of(pred0[row["skill_id"]]))
                pred_s1.append(mastery_of(pred1[row["skill_id"]]))
                pred_s0_user.append(user_index[uid])
                pred_y_pfa.append(row["correct"])
            event_cache.append((row, prefix, flags, slots, pfa_scores, keys))
            events_seen += 1
        realized_scores = score_forward(pilot.model, realized_pairs, pilot.device)
        for event_i, (row, prefix, flags, slots, pfa_scores, keys) in enumerate(event_cache):
            for schema in SCHEMAS:
                if not flags[schema]:
                    continue
                slot = slots[schema]
                own = gru_fp(row)
                for fp in slot.gru:
                    if fp == own:
                        continue
                    extra_pairs.append(sequence_for(pilot, prefix, fp, row["skill_id"]))
                    extra_meta.append((event_i, schema))
        extra_scores = score_forward(pilot.model, extra_pairs, pilot.device) if extra_pairs else np.zeros(0)
        grouped_extra = defaultdict(list)
        for (event_i, schema), score in zip(extra_meta, extra_scores):
            grouped_extra[(event_i, schema)].append(float(score))
        codes = []
        for event_i, (row, prefix, flags, slots, pfa_scores, keys) in enumerate(event_cache):
            realized_gru = float(realized_scores[event_i])
            pred_y.append(row["correct"])
            pred_gru.append(realized_gru)
            pred_user.append(user_index[uid])
            ui = user_index[uid]
            per_model_codes = []
            for model_i, name in enumerate(model_names):
                schema_codes = []
                for schema_i, schema in enumerate(SCHEMAS):
                    if not flags[schema]:
                        schema_codes.append(([0, 0, 0], None))
                        for tau_i, _tau in enumerate(TAUS):
                            status[ui, model_i, schema_i, tau_i, 0] += 1
                        continue
                    slot = slots[schema]
                    if name == "TransparentActionPFA":
                        lo_p, hi_p, realized_p, is_covered, n_sig, n_fp = pfa_scores[schema]
                    else:
                        own = gru_fp(row)
                        values = list(grouped_extra.get((event_i, schema), []))
                        if own in slot.gru:
                            values.append(realized_gru)
                        if not values:
                            values = [0.5]
                        lo_p, hi_p, realized_p = min(values), max(values), realized_gru
                        is_covered = own in slot.gru
                        n_sig, n_fp = slot.n, len(slot.gru)
                    signatures[model_i, schema_i] += n_sig
                    fingerprints[model_i, schema_i] += n_fp
                    supported_n[ui, model_i, schema_i] += 1
                    if is_covered:
                        covered[ui, model_i, schema_i] += 1
                    risk_lo, risk_hi, risk_actual = 1.0 - hi_p, 1.0 - lo_p, 1.0 - realized_p
                    plan = plans[uid]["schemas"][schema]
                    if plan and row["source_row"] == plan["completion"]:
                        completion[name][schema]["lo"][ui] = risk_lo
                        completion[name][schema]["hi"][ui] = risk_hi
                        completion[name][schema]["actual"][ui] = risk_actual
                    if plan and row["source_row"] == plan["start"]:
                        start_ck[name][schema]["lo"][ui] = risk_lo
                        start_ck[name][schema]["hi"][ui] = risk_hi
                        start_ck[name][schema]["actual"][ui] = risk_actual
                    if plans[uid]["matched"] == row["source_row"]:
                        matched[name][schema]["lo"][ui] = risk_lo
                        matched[name][schema]["hi"][ui] = risk_hi
                        matched[name][schema]["actual"][ui] = risk_actual
                    tau_codes = []
                    primary_action = None
                    for tau_i, tau in enumerate(TAUS):
                        stable_flag = lo_p >= tau or hi_p < tau
                        if stable_flag:
                            action = "progress" if lo_p >= tau else "review"
                            realized_action = "progress" if realized_p >= tau else "review"
                            status[ui, model_i, schema_i, tau_i, 1] += 1
                            if realized_action == action:
                                contained[ui, model_i, schema_i] += int(tau == TAU)
                            else:
                                false_stable[ui, model_i, schema_i] += int(tau == TAU)
                            if not is_covered:
                                uncovered_stable[ui, model_i, schema_i] += int(tau == TAU)
                            tau_codes.append(1)
                            if tau == TAU:
                                primary_action = action
                        else:
                            status[ui, model_i, schema_i, tau_i, 2] += 1
                            contained[ui, model_i, schema_i] += int(tau == TAU)
                            tau_codes.append(2)
                    schema_codes.append((tau_codes, primary_action))
                    if (uid, row["source_row"]) in sample_keys and schema == "S0" and name == "ActionTokenGRU":
                        serial = pilot.mastery_branch(prefix, as_event(row))
                        equivalence["events"] += 1
                        equivalence["max_abs_diff"] = max(equivalence["max_abs_diff"], abs(serial - realized_gru))
                        label_serial = "progress" if serial >= TAU else "review"
                        label_batch = "progress" if realized_gru >= TAU else "review"
                        if lo_p >= TAU or hi_p < TAU:
                            if label_serial != label_batch:
                                equivalence["action_mismatches"] += 1
                per_model_codes.append(schema_codes)
            codes.append(per_model_codes)
            for model_i, schema_codes in enumerate(per_model_codes):
                for tau_i, tau in enumerate(TAUS):
                    for left, right in (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3")):
                        left_code = schema_codes[SCHEMAS.index(left)][0][tau_i]
                        right_code = schema_codes[SCHEMAS.index(right)][0][tau_i]
                        transitions[(model_names[model_i], tau, left, right)][left_code, right_code] += 1
            primary_tau_index = TAUS.index(TAU)
            for model_i in range(2):
                codes_now = [item[0][primary_tau_index] for item in codes[-1][model_i]]
                actions_now = [item[1] for item in codes[-1][model_i]]
                c0, c1, c2, c3 = codes_now
                if c3 != 0 and c2 == 0:
                    nested_violations["support_S3_without_S2"] += 1
                if c2 != 0 and c1 == 0:
                    nested_violations["support_S2_without_S1"] += 1
                if c1 != 0 and c0 == 0:
                    nested_violations["support_S1_without_S0"] += 1
                if c3 == 2 and c2 == 1:
                    nested_violations["finer_set_under_coarser_stable_S3_S2"] += 1
                if c2 == 2 and c1 == 1:
                    nested_violations["finer_set_under_coarser_stable_S2_S1"] += 1
                if c1 == 2 and c0 == 1:
                    nested_violations["finer_set_under_coarser_stable_S1_S0"] += 1
                if c3 == 1 and c2 == 1 and actions_now[3] != actions_now[2]:
                    nested_violations["stable_action_changed_S3_S2"] += 1
                if c2 == 1 and c1 == 1 and actions_now[2] != actions_now[1]:
                    nested_violations["stable_action_changed_S2_S1"] += 1
                if c1 == 1 and c0 == 1 and actions_now[1] != actions_now[0]:
                    nested_violations["stable_action_changed_S1_S0"] += 1
        if user_i and user_i % 200 == 0:
            print(f"eval_users {user_i} events {events_seen} rss_mb={rss_mb():.1f}", flush=True)
    return {
        "users": users,
        "status": status,
        "false_stable": false_stable,
        "uncovered_stable": uncovered_stable,
        "covered": covered,
        "contained": contained,
        "supported_n": supported_n,
        "signatures": signatures,
        "fingerprints": fingerprints,
        "transitions": {f"{model}|{tau}|{left}|{right}": value for (model, tau, left, right), value in transitions.items()},
        "nested_violations": dict(nested_violations),
        "completion": completion,
        "start": start_ck,
        "matched": matched,
        "pred_y": np.asarray(pred_y, dtype=np.int8),
        "pred_gru": np.asarray(pred_gru, dtype=np.float64),
        "pred_user": np.asarray(pred_user, dtype=np.int32),
        "pred_s0": np.asarray(pred_s0, dtype=np.float64),
        "pred_s1": np.asarray(pred_s1, dtype=np.float64),
        "pred_s0_user": np.asarray(pred_s0_user, dtype=np.int32),
        "pred_y_pfa": np.asarray(pred_y_pfa, dtype=np.int8),
        "equivalence": equivalence,
        "future_violations": future_violations,
        "events_seen": events_seen,
    }


def summarize(eval_out: dict, population: dict) -> None:
    status = eval_out["status"]
    names = ("TransparentActionPFA", "ActionTokenGRU")
    valid = int(population["valid_targets"])
    invalid = int(population["invalid_targets"])
    source_n = valid + invalid
    rows = []
    fp_rows = []
    tau_rows = []
    for model_i, name in enumerate(names):
        for schema_i, schema in enumerate(SCHEMAS):
            for tau_i, tau in enumerate(TAUS):
                unsup = int(status[:, model_i, schema_i, tau_i, 0].sum())
                stable = int(status[:, model_i, schema_i, tau_i, 1].sum())
                sett = int(status[:, model_i, schema_i, tau_i, 2].sum())
                row = {
                    "model": name, "schema": schema, "tau": tau, "valid_targets": valid, "invalid_excluded": invalid,
                    "unsupported": unsup, "stable": stable, "set_valued": sett, "supported": stable + sett,
                    "event_nc": unsup + sett, "event_nc_over_valid": (unsup + sett) / valid,
                    "supplementary_unassessed_or_nc_over_source": (invalid + unsup + sett) / source_n,
                    "source_events": source_n,
                }
                tau_rows.append(row)
                if tau == TAU:
                    fs = int(eval_out["false_stable"][:, model_i, schema_i].sum())
                    nu = int(eval_out["uncovered_stable"][:, model_i, schema_i].sum())
                    cov = int(eval_out["covered"][:, model_i, schema_i].sum())
                    cont = int(eval_out["contained"][:, model_i, schema_i].sum())
                    supp = stable + sett
                    learners = int(np.sum(eval_out["supported_n"][:, model_i, schema_i] > 0))
                    row.update({
                        "fingerprint_covered_supported": cov,
                        "actual_action_contained_supported": cont,
                        "false_stable": fs,
                        "false_stable_over_stable": fs / stable if stable else None,
                        "false_stable_over_supported": fs / supp if supp else None,
                        "false_stable_over_valid": fs / valid,
                        "n_S": stable, "n_U": nu, "n_FS": fs,
                        "n_U_over_n_S": nu / stable if stable else None,
                        "n_FS_over_n_U": fs / nu if nu else None,
                        "learners_with_supported_event": learners,
                        "rich_signatures_summed_over_events": int(eval_out["signatures"][model_i, schema_i]),
                        "consumer_fingerprints_summed_over_events": int(eval_out["fingerprints"][model_i, schema_i]),
                    })
                    rows.append(row)
                    fp_rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "event_certificate_frontier.csv", index=False)
    pd.DataFrame(fp_rows).to_csv(OUT / "fingerprint_coverage_and_false_stable.csv", index=False)
    pd.DataFrame(tau_rows).to_csv(OUT / "threshold_sensitivity.csv", index=False)
    trans_rows = []
    for key, matrix in eval_out["transitions"].items():
        model, tau, left, right = key.split("|")
        labels = ("unsupported", "stable", "set_valued")
        for i, src in enumerate(labels):
            for j, dst in enumerate(labels):
                trans_rows.append({"model": model, "tau": float(tau), "from_schema": left, "to_schema": right, "from_label": src, "to_label": dst, "n": int(matrix[i, j]), "denominator": valid})
    pd.DataFrame(trans_rows).to_csv(OUT / "paired_event_transitions.csv", index=False)



def _tie_counts(endpoints: np.ndarray, query: np.ndarray, ids: np.ndarray) -> np.ndarray:
    groups: dict[float, list[str]] = defaultdict(list)
    for value, uid in zip(endpoints.tolist(), ids.tolist(), strict=True):
        groups[float(value)].append(str(uid))
    for rows in groups.values():
        rows.sort()
    out = np.zeros(len(query), dtype=np.int32)
    for i, value in enumerate(query.tolist()):
        rows = groups.get(float(value))
        if not rows:
            continue
        uid = str(ids[i])
        lo, hi = 0, len(rows)
        while lo < hi:
            mid = (lo + hi) // 2
            if rows[mid] < uid:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo
    return out


def userid_ranks(ids: np.ndarray, los: np.ndarray, his: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(ids))
    if n == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty
    order_lo = np.sort(los)
    order_hi = np.sort(his)
    n_lo_gt = n - np.searchsorted(order_lo, his, side="right")
    self_best = (los > his).astype(np.int32)
    best = (1 + n_lo_gt - self_best + _tie_counts(los, his, ids)).astype(np.int32)
    n_hi_gt = n - np.searchsorted(order_hi, los, side="right")
    self_strict = (his > los).astype(np.int32)
    worst = (1 + n_hi_gt - self_strict + _tie_counts(his, los, ids)).astype(np.int32)
    return best, worst


def capacity_row(model: str, schema: str, kappa: float, n_full: int, n_eligible: int, counts: dict, kind: str) -> dict:
    no_ckpt = n_full - n_eligible
    return {
        "analysis": kind, "model": model, "schema": schema, "kappa": kappa, "tie_rule": "adversarial",
        "n_full": n_full, "n_eligible": n_eligible, "n_no_checkpoint": no_ckpt, "B": counts["B"],
        "guaranteed_in": counts["guaranteed_in"], "guaranteed_out": counts["guaranteed_out"], "unresolved": counts["unresolved"],
        "all_user_capacity_nc": (no_ckpt + counts["unresolved"]) / n_full,
    }


def risk_block(model: str, schema: str, kappa: float, lo: np.ndarray, hi: np.ndarray, actual: np.ndarray, kind: str) -> dict:
    counts = membership_counts(lo, hi, kappa)
    labels = counts["labels"]
    inside = (actual >= lo) & (actual <= hi)
    ids = np.asarray([f"{i:08d}" for i in range(len(lo))])
    order = np.lexsort((ids, -actual))
    chosen = np.zeros(len(lo), dtype=bool)
    chosen[order[: counts["B"]]] = True
    gin = labels == "guaranteed_in"
    gout = labels == "guaranteed_out"
    return {
        "analysis": kind, "model": model, "schema": schema, "kappa": kappa, "n": int(len(lo)),
        "inside": int(inside.sum()), "outside": int((~inside).sum()),
        "false_guaranteed_in": int(np.sum(gin & ~chosen)), "false_guaranteed_out": int(np.sum(gout & chosen)),
        "false_guaranteed_in_outside_interval": int(np.sum(gin & ~chosen & ~inside)),
        "false_guaranteed_out_outside_interval": int(np.sum(gout & chosen & ~inside)),
    }


def capacity_tables(eval_out: dict, identity_differs: dict) -> dict:
    names = ("TransparentActionPFA", "ActionTokenGRU")
    n_full = len(eval_out["users"])
    native_rows, start_rows, matched_rows, risk_rows, tie_rows, transition_rows = [], [], [], [], [], []
    nesting_violations = 0
    monotone_violations = 0
    matched_mask = np.isfinite(eval_out["matched"]["TransparentActionPFA"]["S0"]["lo"])
    for name in names:
        for schema in SCHEMAS:
            native = eval_out["completion"][name][schema]
            start = eval_out["start"][name][schema]
            eligible = np.isfinite(native["lo"]) & np.isfinite(native["hi"])
            start_ok = np.isfinite(start["lo"]) & np.isfinite(start["hi"])
            for kappa in KAPPAS:
                native_counts = membership_counts(native["lo"][eligible], native["hi"][eligible], kappa)
                start_counts = membership_counts(start["lo"][start_ok], start["hi"][start_ok], kappa)
                native_rows.append(capacity_row(name, schema, kappa, n_full, int(eligible.sum()), native_counts, "completion_time_native"))
                start_labels = start_counts["labels"]
                native_labels = native_counts["labels"]
                start_rows.append({
                    **capacity_row(name, schema, kappa, n_full, int(start_ok.sum()), start_counts, "start_time"),
                    "users_whose_checkpoint_event_differs": int(identity_differs[schema]) if kappa == 0.05 else None,
                    "users_whose_realized_score_differs": int(np.sum(eligible & start_ok & (native["actual"] != start["actual"]))),
                    "membership_changes": int(np.sum(native_labels != start_labels)) if len(native_labels) == len(start_labels) else None,
                    "unresolved_change_start_minus_completion": start_counts["unresolved"] - native_counts["unresolved"],
                })
                risk_rows.append(risk_block(name, schema, kappa, native["lo"][eligible], native["hi"][eligible], native["actual"][eligible], "completion_time_native"))
            cell = eval_out["matched"][name][schema]
            for kappa in KAPPAS:
                counts = membership_counts(cell["lo"][matched_mask], cell["hi"][matched_mask], kappa)
                matched_rows.append(capacity_row(name, schema, kappa, n_full, int(matched_mask.sum()), counts, "matched_event"))
                risk_rows.append(risk_block(name, schema, kappa, cell["lo"][matched_mask], cell["hi"][matched_mask], cell["actual"][matched_mask], "matched_event"))
            ids = np.asarray(eval_out["users"], dtype=object)[eligible]
            adv = labels_from_ranks(*adversarial_ranks(native["lo"][eligible], native["hi"][eligible]), budget(int(eligible.sum()), 0.05))
            uid_labels = labels_from_ranks(*userid_ranks(ids, native["lo"][eligible], native["hi"][eligible]), budget(int(eligible.sum()), 0.05))
            tie_rows.append({
                "model": name, "schema": schema, "kappa": 0.05, "n_eligible": int(eligible.sum()),
                "adversarial_guaranteed_in": int(np.sum(adv == "guaranteed_in")),
                "userid_guaranteed_in": int(np.sum(uid_labels == "guaranteed_in")),
                "membership_changes": int(np.sum(adv != uid_labels)),
                "adversarial_in_retained_under_userid": bool(np.all(uid_labels[adv == "guaranteed_in"] == "guaranteed_in")) if np.any(adv == "guaranteed_in") else True,
                "adversarial_out_retained_under_userid": bool(np.all(uid_labels[adv == "guaranteed_out"] == "guaranteed_out")) if np.any(adv == "guaranteed_out") else True,
            })
        for kappa in KAPPAS:
            label_map = {}
            seats = budget(int(matched_mask.sum()), kappa)
            for schema in SCHEMAS:
                cell = eval_out["matched"][name][schema]
                label_map[schema] = labels_from_ranks(*adversarial_ranks(cell["lo"][matched_mask], cell["hi"][matched_mask]), seats)
            for left, right in (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3")):
                for src in ("guaranteed_in", "guaranteed_out", "unresolved"):
                    for dst in ("guaranteed_in", "guaranteed_out", "unresolved"):
                        transition_rows.append({"model": name, "kappa": kappa, "from_schema": left, "to_schema": right, "from_label": src, "to_label": dst, "n": int(np.sum((label_map[left] == src) & (label_map[right] == dst))), "denominator": int(matched_mask.sum())})
            if kappa == 0.05:
                for finer, coarser in (("S3", "S2"), ("S2", "S1"), ("S1", "S0")):
                    a = eval_out["matched"][name][finer]
                    b = eval_out["matched"][name][coarser]
                    nested = (a["lo"][matched_mask] + 1e-9 >= b["lo"][matched_mask]) & (a["hi"][matched_mask] <= b["hi"][matched_mask] + 1e-9)
                    nesting_violations += int(np.sum(~nested))
                    gin_c = label_map[coarser] == "guaranteed_in"
                    gout_c = label_map[coarser] == "guaranteed_out"
                    monotone_violations += int(np.sum(gin_c & (label_map[finer] != "guaranteed_in")))
                    monotone_violations += int(np.sum(gout_c & (label_map[finer] != "guaranteed_out")))
    pd.DataFrame(native_rows).to_csv(OUT / "native_completion_time_capacity.csv", index=False)
    pd.DataFrame(start_rows).to_csv(OUT / "start_time_checkpoint_sensitivity.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(OUT / "matched_event_capacity.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(OUT / "matched_capacity_transitions.csv", index=False)
    pd.DataFrame(tie_rows).to_csv(OUT / "tie_rule_sensitivity.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(OUT / "realized_risk_validity.csv", index=False)
    return {"matched_users": int(matched_mask.sum()), "interval_nesting_violations": nesting_violations, "rank_monotone_violations": monotone_violations}


def write_predictive(eval_out: dict) -> None:
    rows = [
        {"estimand": "ActionTokenGRU action-conditioned current-event correctness; history is H(e); current actions included; current outcome token withheld", **binary_metrics(eval_out["pred_y"], eval_out["pred_gru"])},
        {"estimand": "TransparentActionPFA pre-update mastery using only historical success and failure counts; empty completed histories withheld", **binary_metrics(eval_out["pred_y_pfa"], eval_out["pred_s0"])},
        {"estimand": "TransparentActionPFA pre-update mastery using historical success, failure, hint, and attempt counts; empty completed histories withheld", **binary_metrics(eval_out["pred_y_pfa"], eval_out["pred_s1"])},
    ]
    pd.DataFrame(rows).to_csv(OUT / "predictive_diagnostics.csv", index=False)


def _percentile(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def write_bootstrap(eval_out: dict) -> dict:
    rng = np.random.default_rng(20308)
    n_users = len(eval_out["users"])
    draws = rng.integers(0, n_users, size=(2000, n_users))
    rows = []
    failures = 0
    names = ("TransparentActionPFA", "ActionTokenGRU")
    status = eval_out["status"]
    primary = TAUS.index(TAU)
    for model_i, name in enumerate(names):
        rates = {}
        for schema_i, schema in enumerate(SCHEMAS):
            unsup = status[:, model_i, schema_i, primary, 0]
            stable = status[:, model_i, schema_i, primary, 1]
            sett = status[:, model_i, schema_i, primary, 2]
            u = unsup[draws].sum(axis=1)
            s = stable[draws].sum(axis=1)
            t = sett[draws].sum(axis=1)
            denom = u + s + t
            if np.any(denom == 0):
                failures += int(np.sum(denom == 0))
            nc = np.divide(u + t, denom, out=np.full(len(denom), np.nan), where=denom > 0)
            mean, lo, hi = _percentile(nc[np.isfinite(nc)])
            point = float((unsup.sum() + sett.sum()) / (unsup.sum() + stable.sum() + sett.sum()))
            rows.append({"analysis": "event_nc", "model": name, "schema": schema, "contrast": schema, "point": point, "bootstrap_mean": mean, "ci95_low": lo, "ci95_high": hi, "replicates": 2000, "replicate_failures": int(np.sum(denom == 0)), "seed": 20308})
            rates[schema] = nc
        for left, right in (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3")):
            delta = rates[right] - rates[left]
            mean, lo, hi = _percentile(delta[np.isfinite(delta)])
            rows.append({"analysis": "paired_event_nc_difference", "model": name, "schema": f"{left}_to_{right}", "contrast": f"{right}_minus_{left}", "point": float(np.nanmean(delta)), "bootstrap_mean": mean, "ci95_low": lo, "ci95_high": hi, "replicates": 2000, "replicate_failures": int(np.sum(~np.isfinite(delta))), "seed": 20308})
    matched = np.isfinite(eval_out["matched"]["TransparentActionPFA"]["S0"]["lo"])
    matched_ids = np.flatnonzero(matched)
    matched_draws = matched_ids[rng.integers(0, len(matched_ids), size=(2000, len(matched_ids)))] if len(matched_ids) else np.zeros((2000, 0), dtype=int)
    for name in names:
        for schema in SCHEMAS:
            cell = eval_out["completion"][name][schema]
            present = np.isfinite(cell["lo"])
            native_nc = []
            for draw in draws:
                mask = present[draw]
                no_ckpt = int((~mask).sum())
                if int(mask.sum()) == 0:
                    native_nc.append(1.0)
                    continue
                unresolved = membership_counts(cell["lo"][draw][mask], cell["hi"][draw][mask], 0.05)["unresolved"]
                native_nc.append((no_ckpt + unresolved) / len(draw))
            mean, lo, hi = _percentile(np.asarray(native_nc))
            point_counts = membership_counts(cell["lo"][present], cell["hi"][present], 0.05)
            point = (int((~present).sum()) + point_counts["unresolved"]) / n_users
            rows.append({"analysis": "native_capacity_nc_kappa_0.05", "model": name, "schema": schema, "contrast": schema, "point": point, "bootstrap_mean": mean, "ci95_low": lo, "ci95_high": hi, "replicates": 2000, "replicate_failures": 0, "seed": 20308})
            if len(matched_ids):
                mcell = eval_out["matched"][name][schema]
                matched_nc = []
                for draw in matched_draws:
                    unresolved = membership_counts(mcell["lo"][draw], mcell["hi"][draw], 0.05)["unresolved"]
                    matched_nc.append(unresolved / len(draw))
                mean, lo, hi = _percentile(np.asarray(matched_nc))
                point_m = membership_counts(mcell["lo"][matched], mcell["hi"][matched], 0.05)["unresolved"] / len(matched_ids)
                rows.append({"analysis": "matched_capacity_unresolved_rate_kappa_0.05", "model": name, "schema": schema, "contrast": schema, "point": point_m, "bootstrap_mean": mean, "ci95_low": lo, "ci95_high": hi, "replicates": 2000, "replicate_failures": 0, "seed": 20308})
    pd.DataFrame(rows).to_csv(OUT / "paired_bootstrap_intervals.csv", index=False)
    return {"replicates": 2000, "event_denominator_failures": failures, "seed": 20308}


def write_comparison(frontier_rows: list[dict]) -> None:
    historical = json.loads(FRONTIER.read_text(encoding="utf-8"))
    rows = []
    name_map = {"TransparentActionPFA": "transparent", "ActionTokenGRU": "neural"}
    for row in frontier_rows:
        if row["tau"] != TAU:
            continue
        old = historical[name_map[row["model"]]][row["schema"]]
        rows.append({
            "model": row["model"], "schema": row["schema"],
            "historical_denominator": old["n_events"], "chronological_valid_denominator": row["valid_targets"],
            "historical_stable": old["stable"], "chronological_stable": row["stable"],
            "historical_set_valued": old["set_valued"], "chronological_set_valued": row["set_valued"],
            "historical_unsupported": old["unsupported"], "chronological_unsupported": row["unsupported"],
            "historical_event_nc": old["event_noncertification_rate"], "chronological_event_nc": row["event_nc_over_valid"],
            "historical_capacity_eligible": old["p2_eligible_users"], "historical_capacity_unresolved": old["p2_unresolved"],
            "historical_capacity_B": old["p2_seats_B"],
            "comparison_limit": "File-order histories, a different GRU trajectory, file-order checkpoints, and 91 additional invalid targets. Not a one-factor effect.",
        })
    pd.DataFrame(rows).to_csv(OUT / "historical_vs_chronological_comparison.csv", index=False)


def write_checksums() -> None:
    files = []
    for path in sorted(item for item in OUT.iterdir() if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/chronological_replication/output_checksums.json"], "n_files": len(files), "files": files})


def self_check() -> None:
    model = TransparentActionPFA()
    ev = Event("u", "p", "s", 1, 2, 3, "0", ("hint", "answer"), tuple(), "development")
    model.update(ev)
    state = fp_delta((1, 2, 3, ("hint", "answer")))
    if abs(model.mastery("s") - mastery_of(np.asarray(state, dtype=np.float64))) > 1e-12:
        raise SystemExit("PFA mastery self-check failed")
    los = np.array([0.1, 0.2, 0.4], dtype=float)
    his = np.array([0.3, 0.25, 0.5], dtype=float)
    best, worst = adversarial_ranks(los, his)
    for i in range(3):
        best_loop = 1 + sum(los[j] > his[i] for j in range(3) if j != i)
        worst_loop = 1 + sum(his[j] >= los[i] for j in range(3) if j != i)
        if int(best[i]) != best_loop or int(worst[i]) != worst_loop:
            raise SystemExit("rank self-check failed")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    self_check()
    torch.set_num_threads(8)
    pre = preflight()
    dump("source_integrity_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_PROTOCOL_OR_INTEGRITY"})
        write_checksums()
        raise SystemExit(2)
    print("preflight_pass", flush=True)
    dev_frame, first_seen = load_order("development")
    if dev_frame["user_id"].nunique() != 31753 or len(dev_frame) != 4286981 or int((~dev_frame["valid"]).sum()) != 475:
        raise SystemExit("development population mismatch")
    del dev_frame
    selection = replay_selection(first_seen)
    if not selection["matches_tempfor"]:
        dump("training_selection_audit.json", {"epoch_sizes": selection["epoch_sizes"], "union": selection["union"], "intersection": selection["intersection"], "matches_tempfor": False})
        dump("numerical_invariants.json", {"status": "BLOCKED_TRAINING_OR_EXECUTION", "reason": "seed-17 selection mismatch"})
        write_checksums()
        raise SystemExit(3)
    print("selection_pass", selection["union"], selection["intersection"], flush=True)
    geometry = build_development(selection["selected_users"])
    geometry_public = {
        "events": geometry["events"], "key_counts": geometry["key_counts"], "duplicate_rich_signatures": geometry["duplicate_signatures"],
        "unique_rich_signatures": geometry["unique_signatures"], "projection_violations": geometry["nesting_violations"],
        "fingerprint_nesting": geometry["fingerprint_nesting"], "invalid_development_events_retained": 475,
        "support_predicate": "(n_users >= 2) OR (n_rich_signatures >= 2)", "n_skills": len(geometry["skill_order"]), "n_action_types": len(geometry["atype_order"]),
    }
    dump("development_support_geometry.json", geometry_public)
    if geometry["key_counts"] != EXPECTED_KEYS or geometry["duplicate_signatures"] or geometry["nesting_violations"] or geometry["fingerprint_nesting"]["violations"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_PROTOCOL_OR_INTEGRITY", "reason": "development geometry mismatch"})
        write_checksums()
        raise SystemExit(4)
    grouped = load_confirmatory()
    checkpoint = plan_checkpoints(grouped, geometry["bundles"])
    history = history_reference(grouped)
    protocol = {
        "frozen_before_training_and_inference": True,
        "history": "H(e) = {same student, valid interval, end(p) < start(e)}",
        "history_order": ["end", "source_row"], "target_order": ["start", "source_row"],
        "primary_capacity_checkpoint": "maximum completion time among valid supported targets; source_row breaks ties",
        "secondary_checkpoint": "maximum start time among valid supported targets; source_row breaks ties",
        "matched_checkpoint": "maximum completion time among targets supported under all four schemas; source_row breaks ties",
        "valid_targets": checkpoint["valid_targets"], "invalid_targets": checkpoint["invalid_targets"],
        "eligible_users": checkpoint["eligible_users"],
        "users_whose_completion_and_start_checkpoints_differ": checkpoint["checkpoint_identity_differs"],
        "matched_users": checkpoint["matched_users"],
        "targets_with_a_completion_during_the_interaction": checkpoint["targets_with_a_completion_during_the_interaction"],
        "checkpoint_reference_mismatches": checkpoint["reference_mismatches"], "history_reference": history,
        "tau_primary": TAU, "kappa_primary": 0.05, "tie_rule_primary": "adversarial",
    }
    dump("protocol_freeze.json", protocol)
    dump("confirmatory_population_accounting.json", {"valid_targets": checkpoint["valid_targets"], "invalid_targets": checkpoint["invalid_targets"], "users": len(grouped), "expected_valid": 939819, "source_events": 939910})
    if checkpoint["valid_targets"] != 939819 or checkpoint["invalid_targets"] != 91 or checkpoint["reference_mismatches"] or history["identity_mismatches"] or history["future_members"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_PROTOCOL_OR_INTEGRITY", "protocol": protocol})
        write_checksums()
        raise SystemExit(5)
    print("protocol_frozen", flush=True)
    pilot = make_pilot(geometry["skill_order"], geometry["atype_order"])
    tokens = token_audit(pilot, geometry["training_rows"])
    if tokens["token_mismatches"] or tokens["current_outcome_leaks"] or tokens["future_history_members"] or sum(p.numel() for p in pilot.model.parameters()) != 33857:
        dump("input_provenance_checks.json", {"token_audit_before_training": tokens})
        dump("numerical_invariants.json", {"status": "BLOCKED_PROTOCOL_OR_INTEGRITY", "tokens": tokens})
        write_checksums()
        raise SystemExit(6)
    training = train_model(pilot, geometry["training_rows"], selection["cohorts"])
    training["duration_seconds"] = time.time() - t0
    dump("gru_training_log.json", training)
    dump("training_selection_audit.json", {
        "epoch_sizes": selection["epoch_sizes"], "union": selection["union"], "intersection": selection["intersection"],
        "eligible_users": selection["eligible_users"], "selection_sha256": selection["selection_sha256"], "matches_tempfor": True,
        "examples_by_epoch": training["examples_by_epoch"], "invalid_targets_excluded_by_epoch": training["invalid_targets_excluded_by_epoch"],
        "selected_file_order_targets_by_epoch": training["selected_file_order_targets_by_epoch"],
        "empty_token_examples_skipped_by_epoch": training["empty_token_examples_skipped_by_epoch"],
        "history_contract_violations": training["history_contract_violations"],
        "rng": {"python": NEURAL_SEED, "numpy": NEURAL_SEED, "torch": NEURAL_SEED},
        "optimizer": "Adam", "learning_rate": 0.001, "epochs": 2, "loss_by_epoch": training["loss_by_epoch"],
    })
    print("training_done", training["loss_by_epoch"], flush=True)
    eval_out = evaluate(pilot, grouped, geometry["bundles"], checkpoint["plans"])
    summarize(eval_out, checkpoint)
    capacity_info = capacity_tables(eval_out, checkpoint["checkpoint_identity_differs"])
    write_predictive(eval_out)
    bootstrap_info = write_bootstrap(eval_out)
    frontier = pd.read_csv(OUT / "event_certificate_frontier.csv")
    write_comparison(frontier.to_dict(orient="records"))
    dump("input_provenance_checks.json", {"token_audit_before_training": tokens, "score_equivalence_after_training": eval_out["equivalence"], "history_reference": history, "checkpoint_reference_mismatches": checkpoint["reference_mismatches"]})
    status_sums = []
    for model_i in range(2):
        for schema_i in range(4):
            total = int(eval_out["status"][:, model_i, schema_i, TAUS.index(TAU), :].sum())
            status_sums.append(total)
    status = "COMPLETE"
    # Float32 GRU re-scoring can move an identical sequence by one ulp. That is recorded,
    # not treated as a class-nesting failure. PFA intervals are exact. A gap above 1e-5
    # would be a blocker; the follow-up audit measured 1.19e-7.
    if eval_out["events_seen"] != 939819 or eval_out["future_violations"] or any(item != 939819 for item in status_sums) or eval_out["nested_violations"] or eval_out["equivalence"]["action_mismatches"] or eval_out["equivalence"]["max_abs_diff"] > 1e-5 or training["history_contract_violations"]:
        status = "BLOCKED_PROTOCOL_OR_INTEGRITY"
    invariants = {
        "status": status, "events_seen": eval_out["events_seen"], "future_violations": eval_out["future_violations"],
        "status_sums_primary_tau": status_sums, "nested_violations": eval_out["nested_violations"],
        "score_equivalence": eval_out["equivalence"], "capacity": capacity_info, "bootstrap": bootstrap_info,
        "parameter_count": training["parameter_count"], "duration_seconds": time.time() - t0, "max_rss_mb": rss_mb(),
    }
    dump("numerical_invariants.json", invariants)
    dump("execution_progress.json", {"status": status, "duration_seconds": invariants["duration_seconds"], "max_rss_mb": invariants["max_rss_mb"]})
    dump("experiment_manifest.json", {"experiment": "pipeline", "script": "scripts/assistments/chronological_replication.py", "script_sha256": sha256_file(Path(__file__).resolve()), "status": status, "command": "python3 -u scripts/assistments/chronological_replication.py"})
    write_checksums()
    print(status, flush=True)


if __name__ == "__main__":
    main()
