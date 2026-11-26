#!/usr/bin/env python3
"""Hint-count and attempt-count factorial release ablation.

Scores the frozen PFA, chronological GRU seed 17, and causal attention seed 17.
No training is performed and no checkpoint is written.
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

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "attncons", ROOT / "scripts/assistments/attention_consumer.py"
)
attncons = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attncons)
chronorep = attncons.chronorep

from r7_core import MAX_EVAL_PREFIX_EVENTS, MAX_HIST_TOKENS, MAX_SEQ_ACTIONS, ActionTokenGRU  # noqa: E402

OUT = ROOT / "results/assistments/release_field_ablation"
CHRONOREP_CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
ATT_CKPT = ROOT / "results/assistments/attention_consumer/causal_token_attention_seed17.pt"
RELEASES = ("R00", "R10", "R01", "R11")
EDGES = (("R00", "R10"), ("R00", "R01"), ("R10", "R11"), ("R01", "R11"), ("R00", "R11"))
MODELS = ("TransparentActionPFA", "ActionTokenGRU", "CausalTokenAttentionKT")
TAUS = (0.70, 0.80, 0.90)
PRIMARY = 0.80
VALID = 939819
STATUS_NAME = ("unsupported", "stable", "set_valued")
GRU_SHA = "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
ATT_SHA = "8728fb05b33399df24daaaa48a76b3c6d6870aa2cee8419909cdeda5f88c0aaa"
EXPECTED = {
    ("TransparentActionPFA", "R00"): (20408, 916845, 2566, 45),
    ("TransparentActionPFA", "R11"): (77877, 861927, 15, 0),
    ("ActionTokenGRU", "R00"): (20408, 810576, 108835, 1077),
    ("ActionTokenGRU", "R11"): (77877, 846772, 15170, 252),
    ("CausalTokenAttentionKT", "R00"): (20408, 763342, 156069, 1579),
    ("CausalTokenAttentionKT", "R11"): (77877, 808049, 53893, 697),
}


def dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=_json) + "\n", encoding="utf-8")


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return chronorep._json_default(value)


def num(value) -> str:
    if value == "N/A":
        return "N/A"
    return format(float(value), ".17g")


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["empty"]
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: num(value) if isinstance(value, float) else value for key, value in row.items()})


class Slot:
    __slots__ = ("n", "n_users", "first_user", "n_rich", "rich", "pfa", "gru", "pfa_list", "delta_matrix")

    def __init__(self) -> None:
        self.n = 0
        self.n_users = 0
        self.first_user = None
        self.n_rich = 0
        self.rich = set()
        self.pfa = set()
        self.gru = set()
        self.pfa_list = []
        self.delta_matrix = None


def release_of(row: dict, release: str) -> tuple:
    problem = str(row["problem_id"])
    correct = str(int(row["correct"]))
    hint = str(int(row["hint_count"]))
    attempt = str(int(row["attempt_count"]))
    if release == "R00":
        return ("R00", problem, correct)
    if release == "R10":
        return ("R10", problem, correct, hint)
    if release == "R01":
        return ("R01", problem, correct, attempt)
    if release == "R11":
        return ("R11", problem, correct, hint, attempt)
    raise KeyError(release)


def supported_event(slot: Slot | None) -> bool:
    return slot is not None and (slot.n_users >= 2 or slot.n >= 2)


def supported_rich(slot: Slot | None) -> bool:
    return slot is not None and (slot.n_users >= 2 or slot.n_rich >= 2)


def rich_hash(item: dict, timestamps) -> bytes:
    payload = repr(
        (
            item["problem_id"],
            item["correct"],
            item["hint_count"],
            item["attempt_count"],
            item["first_action"],
            item["action_types"],
            tuple("" if tok is None else str(tok) for tok in (timestamps or [])),
        )
    ).encode()
    return hashlib.blake2s(payload, digest_size=8).digest()


def mastery_vec(state: np.ndarray) -> np.ndarray:
    n = state[..., 0]
    pos = n > 0
    safe = np.where(pos, n, 1.0)
    base = (state[..., 1] + 1.0) / (safe + 2.0)
    score = (
        base
        - 0.05 * state[..., 2] / safe
        - 0.04 * np.maximum(state[..., 3] / safe - 1.0, 0.0)
        - 0.03 * np.maximum(state[..., 4] / safe - 1.0, 0.0)
        - 0.06 * state[..., 5] / safe
        - 0.05 * state[..., 6] / safe
    )
    return np.where(pos, np.clip(score, 0.01, 0.99), 0.5)


def parent_keys(key: tuple) -> list[tuple]:
    if key[0] == "R11":
        return [("R10", key[1], key[2], key[3]), ("R01", key[1], key[2], key[4]), ("R00", key[1], key[2])]
    if key[0] == "R10":
        return [("R00", key[1], key[2])]
    if key[0] == "R01":
        return [("R00", key[1], key[2])]
    return []


def build_development() -> dict:
    bundles = {release: {} for release in RELEASES}
    events = 0
    unique_rich = set()
    duplicate_rich = 0
    key_mismatches = 0
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(chronorep.RECON / "development.parquet")
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            events += 1
            item = chronorep.compact_row(row, int(start), int(end))
            ev = chronorep.as_event(item)
            if release_of(item, "R00")[1:] != chronorep.release_key(ev, "S0")[1:] or release_of(item, "R11")[1:] != chronorep.release_key(ev, "S1")[1:]:
                key_mismatches += 1
            digest = rich_hash(item, row.get("action_timestamps"))
            if digest in unique_rich:
                duplicate_rich += 1
            else:
                unique_rich.add(digest)
            fp = chronorep.pfa_fp(item)
            gp = chronorep.gru_fp(item)
            for release in RELEASES:
                key = release_of(item, release)
                slot = bundles[release].get(key)
                if slot is None:
                    slot = Slot()
                    bundles[release][key] = slot
                slot.n += 1
                if slot.n_users < 2:
                    if slot.first_user is None:
                        slot.first_user = item["user_id"]
                        slot.n_users = 1
                    elif slot.first_user != item["user_id"]:
                        slot.n_users = 2
                        slot.first_user = None
                if digest not in slot.rich:
                    slot.rich.add(digest)
                    slot.n_rich += 1
                slot.pfa.add(fp)
                slot.gru.add(gp)
        if events % 1000000 == 0:
            print(f"development_geometry {events} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    subset_violations = 0
    checked = 0
    for release in ("R10", "R01", "R11"):
        for key, slot in bundles[release].items():
            checked += 1
            for parent in parent_keys(key):
                other = bundles[parent[0]].get(parent)
                if other is None or not slot.pfa <= other.pfa or not slot.gru <= other.gru:
                    subset_violations += 1
    geometry_rows = []
    predicate_disagreements = {}
    for release, bundle in bundles.items():
        frozen_keys = rich_keys = differ = 0
        pfa_mass = gru_mass = 0
        for slot in bundle.values():
            frozen = supported_event(slot)
            rich = supported_rich(slot)
            frozen_keys += int(frozen)
            rich_keys += int(rich)
            differ += int(frozen != rich)
            pfa_mass += len(slot.pfa)
            gru_mass += len(slot.gru)
            slot.rich = None
            if frozen:
                slot.pfa_list = list(slot.pfa)
                if slot.pfa_list:
                    slot.delta_matrix = np.asarray([chronorep.fp_delta(fp) for fp in slot.pfa_list], dtype=np.float64)
                else:
                    slot.delta_matrix = np.zeros((0, 7), dtype=np.float64)
            else:
                slot.pfa = set()
                slot.gru = set()
                slot.pfa_list = []
                slot.delta_matrix = np.zeros((0, 7), dtype=np.float64)
        predicate_disagreements[release] = differ
        geometry_rows.append({
            "release": release,
            "keys": len(bundle),
            "supported_keys_event_predicate": frozen_keys,
            "supported_keys_rich_predicate": rich_keys,
            "predicate_disagreements": differ,
            "development_events": events,
            "unique_rich_signatures_global": len(unique_rich),
            "duplicate_rich_signatures": duplicate_rich,
            "pfa_fingerprint_occurrences": pfa_mass,
            "gru_fingerprint_occurrences": gru_mass,
        })
    return {
        "events": events,
        "unique_rich": len(unique_rich),
        "duplicate_rich": duplicate_rich,
        "key_mismatches": key_mismatches,
        "subset_checks": checked,
        "subset_violations": subset_violations,
        "predicate_disagreements": predicate_disagreements,
        "geometry_rows": geometry_rows,
        "bundles": bundles,
    }


def classify(lo: float, hi: float, realized: float, tau: float) -> tuple[int, str | None, bool]:
    if lo >= tau:
        action = "progress"
    elif hi < tau:
        action = "review"
    else:
        return 2, None, True
    realized_action = "progress" if realized >= tau else "review"
    return 1, action, realized_action == action


def case_of(n_s: int, n_u: int) -> str:
    if n_s == 0:
        return "C"
    if n_u == 0:
        return "B"
    return "A"


def ratio_pair(numer: int, denom: int, allow_zero_numer: bool = True):
    if denom <= 0:
        return "N/A"
    if numer == 0 and not allow_zero_numer:
        return "N/A"
    return numer / denom


def checkpoint_plans(grouped: dict, bundles: dict) -> dict:
    users = sorted(grouped)
    native = -np.ones((len(users), 4), dtype=np.int64)
    matched = -np.ones(len(users), dtype=np.int64)
    valid = invalid = 0
    for user_i, uid in enumerate(users):
        by_start, _by_end = chronorep.order_valid(grouped[uid])
        valid += len(by_start)
        invalid += sum(1 for row in grouped[uid] if not row["valid"])
        chosen = {release: None for release in RELEASES}
        r11 = []
        for row in by_start:
            flags = {release: supported_event(bundles[release].get(release_of(row, release))) for release in RELEASES}
            for release_i, release in enumerate(RELEASES):
                if not flags[release]:
                    continue
                current = chosen[release]
                if current is None or (row["end"], row["source_row"]) > (current["end"], current["source_row"]):
                    chosen[release] = row
            if flags["R11"]:
                r11.append(row)
        for release_i, release in enumerate(RELEASES):
            if chosen[release] is not None:
                native[user_i, release_i] = chosen[release]["source_row"]
        if r11:
            matched[user_i] = max(r11, key=lambda item: (item["end"], item["source_row"]))["source_row"]
    return {"users": users, "native": native, "matched": matched, "valid": valid, "invalid": invalid}


def score_population(pilot, gru, attention, grouped, bundles, plans) -> dict:
    cache = OUT / "private_event_cache"
    cache.mkdir(parents=True, exist_ok=True)
    progress_path = OUT / "execution_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    resume = int(progress.get("next_user", 0)) if progress.get("phase") == "score" else 0
    create = resume == 0
    source = np.memmap(cache / "source_row.i8", dtype=np.int64, mode="w+" if create else "r+", shape=(VALID,))
    scores = np.memmap(cache / "scores.f8", dtype=np.float64, mode="w+" if create else "r+", shape=(VALID, 3, 4, 3))
    state_path = cache / "running_state.npz"
    n_users = len(plans["users"])
    if resume and state_path.exists():
        previous = np.load(state_path)
        status = previous["status"]
        false_stable = previous["false_stable"]
        uncovered = previous["uncovered"]
        covered = previous["covered"]
        contained = previous["contained"]
        supported_n = previous["supported_n"]
        transitions = previous["transitions"]
        disagreements = previous["disagreements"]
        native_prob = previous["native_prob"]
        matched_prob = previous["matched_prob"]
        n_valid = previous["n_valid"]
        signature_mass = previous["signature_mass"]
        fingerprint_mass = previous["fingerprint_mass"]
        rich_mass = previous["rich_mass"]
        counters = json.loads((cache / "counters.json").read_text())
        event_i0 = int(progress["next_event"])
    else:
        status = np.zeros((n_users, 3, 4, 3, 3), dtype=np.int32)
        false_stable = np.zeros((n_users, 3, 4, 3), dtype=np.int32)
        uncovered = np.zeros((n_users, 3, 4, 3), dtype=np.int32)
        covered = np.zeros((n_users, 3, 4), dtype=np.int32)
        contained = np.zeros((n_users, 3, 4, 3), dtype=np.int32)
        supported_n = np.zeros((n_users, 3, 4), dtype=np.int32)
        transitions = np.zeros((3, 3, 5, 3, 3), dtype=np.int64)
        disagreements = np.zeros((3, 3, 5), dtype=np.int64)
        native_prob = np.full((n_users, 3, 4, 3), np.nan, dtype=np.float64)
        matched_prob = np.full((n_users, 3, 4, 3), np.nan, dtype=np.float64)
        n_valid = np.zeros(n_users, dtype=np.int32)
        signature_mass = np.zeros((3, 4), dtype=np.int64)
        fingerprint_mass = np.zeros((3, 4), dtype=np.int64)
        rich_mass = np.zeros((3, 4), dtype=np.int64)
        counters = {"future": 0, "implication": 0, "predicate_disagreement": 0, "empty_class": 0, "support_violation": 0, "set_violation": 0, "action_violation": 0, "plan_mismatch": 0}
        event_i0 = 0
        resume = 0
    users = plans["users"]
    event_i = event_i0
    for user_i, uid in enumerate(users):
        by_start, by_end = chronorep.order_valid(grouped[uid])
        if user_i < resume:
            continue
        n_valid[user_i] = len(by_start)
        state = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        cursor = 0
        realized_pairs = []
        records = []
        for row in by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                done = by_end[cursor]
                state[done["skill_id"]] += np.asarray(chronorep.fp_delta(chronorep.pfa_fp(done)), dtype=np.float64)
                cursor += 1
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                counters["future"] += 1
            flags = {}
            slots = {}
            for release in RELEASES:
                slot = bundles[release].get(release_of(row, release))
                slots[release] = slot
                flags[release] = supported_event(slot)
                if slot is not None and flags[release] != supported_rich(slot):
                    counters["predicate_disagreement"] += 1
            for left, right in EDGES:
                if flags[right] and not flags[left]:
                    counters["support_violation"] += 1
            prefix = [chronorep.as_event(item) for item in by_end[max(0, cursor - MAX_EVAL_PREFIX_EVENTS) : cursor]]
            realized_pairs.append(pilot._tokens(prefix, chronorep.as_event(row), False))
            records.append((row, prefix, flags, slots, state[row["skill_id"]].copy()))
        gru_real = chronorep.score_forward(gru, realized_pairs, pilot.device)
        att_real = chronorep.score_forward(attention, realized_pairs, pilot.device)
        extra_pairs = []
        event_fps = []
        for row, prefix, flags, slots, _base in records:
            if not flags["R00"]:
                event_fps.append([])
                continue
            own = chronorep.gru_fp(row)
            fps = [fp for fp in slots["R00"].gru if fp != own]
            event_fps.append(fps)
            for fp in fps:
                extra_pairs.append(chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]))
        extra_gru = chronorep.score_forward(gru, extra_pairs, pilot.device) if extra_pairs else np.zeros(0)
        extra_att = chronorep.score_forward(attention, extra_pairs, pilot.device) if extra_pairs else np.zeros(0)
        fp_cursor = 0
        for local_i, (row, _prefix, flags, slots, base) in enumerate(records):
            source[event_i] = row["source_row"]
            own_g = chronorep.gru_fp(row)
            own_p = chronorep.pfa_fp(row)
            fps = event_fps[local_i]
            gru_map = {fp: float(extra_gru[fp_cursor + j]) for j, fp in enumerate(fps)}
            att_map = {fp: float(extra_att[fp_cursor + j]) for j, fp in enumerate(fps)}
            fp_cursor += len(fps)
            if flags["R00"] and own_g in slots["R00"].gru:
                gru_map[own_g] = float(gru_real[local_i])
                att_map[own_g] = float(att_real[local_i])
            pfa_map = {}
            if flags["R00"] and slots["R00"].delta_matrix.shape[0]:
                values = mastery_vec(base + slots["R00"].delta_matrix)
                pfa_map = {fp: float(value) for fp, value in zip(slots["R00"].pfa_list, values)}
            if own_p in pfa_map:
                pfa_real = pfa_map[own_p]
            else:
                own_delta = np.asarray(chronorep.fp_delta(own_p), dtype=np.float64)
                pfa_real = float(mastery_vec((base + own_delta)[None])[0])
            realized = (pfa_real, float(gru_real[local_i]), float(att_real[local_i]))
            maps = (pfa_map, gru_map, att_map)
            codes = np.zeros((3, 4, 3), dtype=np.int8)
            actions = [[[None] * 4 for _ in range(3)] for _ in TAUS]
            for model_i in range(3):
                for release_i, release in enumerate(RELEASES):
                    if not flags[release]:
                        scores[event_i, model_i, release_i, :] = np.nan
                        for tau_i in range(3):
                            status[user_i, model_i, release_i, tau_i, 0] += 1
                        continue
                    slot = slots[release]
                    if model_i == 0:
                        vals = [pfa_map[fp] for fp in slot.pfa_list]
                        is_covered = own_p in slot.pfa
                        mass_fp = len(slot.pfa_list)
                    else:
                        vals = [maps[model_i][fp] for fp in slot.gru]
                        is_covered = own_g in slot.gru
                        mass_fp = len(slot.gru)
                    if not vals:
                        vals = [0.5]
                        counters["empty_class"] += 1
                    lo, hi = min(vals), max(vals)
                    realized_p = realized[model_i]
                    scores[event_i, model_i, release_i, :] = (lo, hi, realized_p)
                    supported_n[user_i, model_i, release_i] += 1
                    signature_mass[model_i, release_i] += slot.n
                    rich_mass[model_i, release_i] += slot.n_rich
                    fingerprint_mass[model_i, release_i] += mass_fp
                    if is_covered:
                        covered[user_i, model_i, release_i] += 1
                    for tau_i, tau in enumerate(TAUS):
                        code, action, contained_flag = classify(lo, hi, realized_p, tau)
                        codes[model_i, release_i, tau_i] = code
                        actions[tau_i][model_i][release_i] = action
                        status[user_i, model_i, release_i, tau_i, code] += 1
                        if code == 1:
                            if contained_flag:
                                contained[user_i, model_i, release_i, tau_i] += 1
                            else:
                                false_stable[user_i, model_i, release_i, tau_i] += 1
                                if is_covered:
                                    counters["implication"] += 1
                            if not is_covered:
                                uncovered[user_i, model_i, release_i, tau_i] += 1
                        else:
                            contained[user_i, model_i, release_i, tau_i] += 1
                    if plans["native"][user_i, release_i] == row["source_row"]:
                        native_prob[user_i, model_i, release_i, :] = (lo, hi, realized_p)
                    if plans["matched"][user_i] == row["source_row"]:
                        matched_prob[user_i, model_i, release_i, :] = (lo, hi, realized_p)
            for model_i in range(3):
                for tau_i, _tau in enumerate(TAUS):
                    for edge_i, (left, right) in enumerate(EDGES):
                        left_i = RELEASES.index(left)
                        right_i = RELEASES.index(right)
                        left_code = int(codes[model_i, left_i, tau_i])
                        right_code = int(codes[model_i, right_i, tau_i])
                        transitions[model_i, tau_i, edge_i, left_code, right_code] += 1
                        same_tau_actions = actions[tau_i][model_i]
                        if left_code == 1 and right_code == 1 and same_tau_actions[left_i] != same_tau_actions[right_i]:
                            disagreements[model_i, tau_i, edge_i] += 1
                            counters["action_violation"] += 1
                        if right_code == 2 and left_code == 1:
                            counters["set_violation"] += 1
            for release_i, release in enumerate(RELEASES):
                if plans["native"][user_i, release_i] == row["source_row"] and not flags[release]:
                    counters["plan_mismatch"] += 1
                if plans["matched"][user_i] == row["source_row"] and not flags[release]:
                    counters["plan_mismatch"] += 1
            event_i += 1
        if user_i and user_i % 200 == 0:
            _flush_state(state_path, status, false_stable, uncovered, covered, contained, supported_n, transitions, disagreements, native_prob, matched_prob, n_valid, signature_mass, fingerprint_mass, rich_mass)
            source.flush()
            scores.flush()
            (cache / "counters.json").write_text(json.dumps(counters))
            dump("execution_progress.json", {"phase": "score", "next_user": user_i + 1, "next_event": event_i, "complete": False})
            print(f"users {user_i} events {event_i} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    _flush_state(state_path, status, false_stable, uncovered, covered, contained, supported_n, transitions, disagreements, native_prob, matched_prob, n_valid, signature_mass, fingerprint_mass, rich_mass)
    source.flush()
    scores.flush()
    (cache / "counters.json").write_text(json.dumps(counters))
    dump("execution_progress.json", {"phase": "score", "next_user": n_users, "next_event": event_i, "complete": True})
    return _pack(status, false_stable, uncovered, covered, contained, supported_n, transitions, disagreements, native_prob, matched_prob, n_valid, signature_mass, fingerprint_mass, rich_mass, counters, event_i)


def _flush_state(path, *arrays) -> None:
    names = ["status", "false_stable", "uncovered", "covered", "contained", "supported_n", "transitions", "disagreements", "native_prob", "matched_prob", "n_valid", "signature_mass", "fingerprint_mass", "rich_mass"]
    np.savez_compressed(path, **{name: array for name, array in zip(names, arrays)})


def _pack(*args) -> dict:
    names = ["status", "false_stable", "uncovered", "covered", "contained", "supported_n", "transitions", "disagreements", "native_prob", "matched_prob", "n_valid", "signature_mass", "fingerprint_mass", "rich_mass", "counters", "events"]
    return {name: value for name, value in zip(names, args)}


def repair_unsupported(result: dict) -> dict:
    """The first scoring pass recorded supported statuses but skipped the unsupported accumulator."""
    status = result["status"]
    supported = result["supported_n"]
    n_valid = result["n_valid"]
    for model_i in range(3):
        for release_i in range(4):
            observed = supported[:, model_i, release_i]
            for tau_i in range(3):
                counted = status[:, model_i, release_i, tau_i, 1] + status[:, model_i, release_i, tau_i, 2]
                if not np.array_equal(counted, observed):
                    raise SystemExit("supported status does not match the supported-event counter")
                status[:, model_i, release_i, tau_i, 0] = n_valid - observed
    return result


def load_completed(n_users: int) -> dict:
    previous = np.load(OUT / "private_event_cache" / "running_state.npz")
    progress = json.loads((OUT / "execution_progress.json").read_text())
    counters = json.loads((OUT / "private_event_cache" / "counters.json").read_text())
    loaded = _pack(
        previous["status"], previous["false_stable"], previous["uncovered"], previous["covered"], previous["contained"],
        previous["supported_n"], previous["transitions"], previous["disagreements"], previous["native_prob"], previous["matched_prob"],
        previous["n_valid"], previous["signature_mass"], previous["fingerprint_mass"], previous["rich_mass"], counters, int(progress["next_event"]),
    )
    loaded["status"] = np.array(loaded["status"], copy=True)
    return loaded


def certificate_rows(result: dict) -> list[dict]:
    rows = []
    for model_i, model in enumerate(MODELS):
        for release_i, release in enumerate(RELEASES):
            for tau_i, tau in enumerate(TAUS):
                unsup = int(result["status"][:, model_i, release_i, tau_i, 0].sum())
                stable = int(result["status"][:, model_i, release_i, tau_i, 1].sum())
                sett = int(result["status"][:, model_i, release_i, tau_i, 2].sum())
                supported = stable + sett
                n_u = int(result["uncovered"][:, model_i, release_i, tau_i].sum())
                n_fs = int(result["false_stable"][:, model_i, release_i, tau_i].sum())
                cov = int(result["covered"][:, model_i, release_i].sum())
                contained = int(result["contained"][:, model_i, release_i, tau_i].sum())
                kind = case_of(stable, n_u)
                if kind == "A":
                    second = n_fs / n_u
                else:
                    second = "N/A"
                rows.append({
                    "model": model,
                    "release": release,
                    "tau": tau,
                    "valid_targets": VALID,
                    "supported": supported,
                    "unsupported": unsup,
                    "stable": stable,
                    "set_valued": sett,
                    "event_nc": unsup + sett,
                    "event_nc_rate": (unsup + sett) / VALID,
                    "set_valued_over_supported": ratio_pair(sett, supported),
                    "fingerprint_covered_supported": cov,
                    "fingerprint_coverage_rate": ratio_pair(cov, supported),
                    "stable_uncovered": n_u,
                    "false_stable": n_fs,
                    "false_stable_over_stable": 0.0 if kind == "B" else ratio_pair(n_fs, stable),
                    "false_stable_over_supported": ratio_pair(n_fs, supported),
                    "action_contained_supported": contained,
                    "learners_with_supported_event": int(np.sum(result["supported_n"][:, model_i, release_i] > 0)),
                    "rich_signatures_enumerated": int(result["rich_mass"][model_i, release_i]),
                    "development_event_multiplicity": int(result["signature_mass"][model_i, release_i]),
                    "consumer_fingerprints_scored": int(result["fingerprint_mass"][model_i, release_i]),
                    "conditional_case": kind,
                    "second_factor": second,
                    "n_S": stable,
                    "n_U": n_u,
                    "n_FS": n_fs,
                })
    return rows


def reproduction(rows: list[dict]) -> dict:
    cells = []
    ok = True
    for model, release in EXPECTED:
        expected = EXPECTED[(model, release)]
        found = next(row for row in rows if row["model"] == model and row["release"] == release and row["tau"] == PRIMARY)
        observed = (found["unsupported"], found["stable"], found["set_valued"], found["false_stable"])
        match = observed == expected
        ok = ok and match
        cells.append({"model": model, "release": release, "expected": expected, "observed": observed, "match": match})
    payload = {"pass": ok, "cells": cells}
    dump("historical_and_frozen_reproduction.json", payload)
    return payload


def effect_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    effects = []
    parts = []
    index = {(row["model"], row["release"], row["tau"]): row for row in rows}
    names = {
        "hint_only": ("R10", "R00"),
        "attempt_only": ("R01", "R00"),
        "hint_given_attempt": ("R11", "R01"),
        "attempt_given_hint": ("R11", "R10"),
        "joint": ("R11", "R00"),
    }
    for model in MODELS:
        base = {release: index[(model, release, PRIMARY)] for release in RELEASES}
        nc = {release: base[release]["event_nc"] for release in RELEASES}
        interaction = nc["R11"] - nc["R10"] - nc["R01"] + nc["R00"]
        effects.append({
            "model": model,
            "tau": PRIMARY,
            "NC00": nc["R00"],
            "NC10": nc["R10"],
            "NC01": nc["R01"],
            "NC11": nc["R11"],
            "delta_H": nc["R10"] - nc["R00"],
            "delta_A": nc["R01"] - nc["R00"],
            "delta_H_given_A": nc["R11"] - nc["R01"],
            "delta_A_given_H": nc["R11"] - nc["R10"],
            "interaction": interaction,
            "joint": nc["R11"] - nc["R00"],
            "delta_H_rate": (nc["R10"] - nc["R00"]) / VALID,
            "delta_A_rate": (nc["R01"] - nc["R00"]) / VALID,
            "delta_H_given_A_rate": (nc["R11"] - nc["R01"]) / VALID,
            "delta_A_given_H_rate": (nc["R11"] - nc["R10"]) / VALID,
            "interaction_rate": interaction / VALID,
            "joint_rate": (nc["R11"] - nc["R00"]) / VALID,
            "interaction_definition": "NC11 - NC10 - NC01 + NC00; descriptive certificate interaction, not a causal interaction",
        })
        for label, (right, left) in names.items():
            a, b = base[right], base[left]
            parts.append({
                "model": model,
                "contrast": label,
                "right": right,
                "left": left,
                "delta_event_nc": a["event_nc"] - b["event_nc"],
                "delta_unsupported": a["unsupported"] - b["unsupported"],
                "delta_set_valued": a["set_valued"] - b["set_valued"],
                "delta_stable": a["stable"] - b["stable"],
                "delta_false_stable": a["false_stable"] - b["false_stable"],
                "delta_fingerprint_covered": a["fingerprint_covered_supported"] - b["fingerprint_covered_supported"],
                "delta_event_nc_rate": (a["event_nc"] - b["event_nc"]) / VALID,
            })
        parts.append({
            "model": model,
            "contrast": "interaction",
            "right": "R11",
            "left": "R10+R01-R00",
            "delta_event_nc": interaction,
            "delta_unsupported": base["R11"]["unsupported"] - base["R10"]["unsupported"] - base["R01"]["unsupported"] + base["R00"]["unsupported"],
            "delta_set_valued": base["R11"]["set_valued"] - base["R10"]["set_valued"] - base["R01"]["set_valued"] + base["R00"]["set_valued"],
            "delta_stable": base["R11"]["stable"] - base["R10"]["stable"] - base["R01"]["stable"] + base["R00"]["stable"],
            "delta_false_stable": base["R11"]["false_stable"] - base["R10"]["false_stable"] - base["R01"]["false_stable"] + base["R00"]["false_stable"],
            "delta_fingerprint_covered": base["R11"]["fingerprint_covered_supported"] - base["R10"]["fingerprint_covered_supported"] - base["R01"]["fingerprint_covered_supported"] + base["R00"]["fingerprint_covered_supported"],
            "delta_event_nc_rate": interaction / VALID,
        })
    return effects, parts


def transition_rows(result: dict) -> list[dict]:
    rows = []
    for model_i, model in enumerate(MODELS):
        for tau_i, tau in enumerate(TAUS):
            for edge_i, (left, right) in enumerate(EDGES):
                matrix = result["transitions"][model_i, tau_i, edge_i]
                left_nc = int(matrix[0, :].sum() + matrix[2, :].sum())
                right_nc = int(matrix[:, 0].sum() + matrix[:, 2].sum())
                row = {
                    "model": model,
                    "tau": tau,
                    "left": left,
                    "right": right,
                    "matrix_sum": int(matrix.sum()),
                    "stable_to_unsupported": int(matrix[1, 0]),
                    "set_to_unsupported": int(matrix[2, 0]),
                    "set_to_stable": int(matrix[2, 1]),
                    "stable_to_set": int(matrix[1, 2]),
                    "unsupported_to_supported": int(matrix[0, 1] + matrix[0, 2]),
                    "stable_action_disagreements_primary_definition": int(result["disagreements"][model_i, tau_i, edge_i]),
                    "net_event_nc": right_nc - left_nc,
                    "gross_support_loss": int(matrix[1, 0] + matrix[2, 0]),
                    "gross_ambiguity_resolution": int(matrix[2, 1]),
                }
                for a, a_name in enumerate(STATUS_NAME):
                    for b, b_name in enumerate(STATUS_NAME):
                        row[f"{a_name}_to_{b_name}"] = int(matrix[a, b])
                rows.append(row)
    return rows


def common_rows(result: dict) -> list[dict]:
    rows = []
    # Recompute from the stored per-event cache so support loss is removed exactly.
    scores = np.memmap(OUT / "private_event_cache" / "scores.f8", dtype=np.float64, mode="r", shape=(VALID, 3, 4, 3))
    r11 = np.isfinite(scores[:, 0, 3, 0])
    n = int(r11.sum())
    for model_i, model in enumerate(MODELS):
        for release_i, release in enumerate(RELEASES):
            lo = scores[r11, model_i, release_i, 0]
            hi = scores[r11, model_i, release_i, 1]
            finite = np.isfinite(lo) & np.isfinite(hi)
            for tau in TAUS:
                stable = int(np.sum(finite & ((lo >= tau) | (hi < tau))))
                sett = int(np.sum(finite & ~((lo >= tau) | (hi < tau))))
                unsup = int(np.sum(~finite))
                rows.append({
                    "model": model,
                    "release": release,
                    "tau": tau,
                    "r11_supported_events": n,
                    "finite_intervals": int(finite.sum()),
                    "unsupported": unsup,
                    "stable": stable,
                    "set_valued": sett,
                    "event_nc": unsup + sett,
                })
    return rows


def capacity_tables(result: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    native_rows, matched_rows, risk_rows, transition_rows_out = [], [], [], []
    n_full = result["native_prob"].shape[0]
    matched_users = np.isfinite(result["matched_prob"][:, 0, 0, 0])
    for model_i, model in enumerate(MODELS):
        for release_i, release in enumerate(RELEASES):
            native = result["native_prob"][:, model_i, release_i, :]
            eligible = np.isfinite(native[:, 0]) & np.isfinite(native[:, 1])
            lo = 1.0 - native[eligible, 1]
            hi = 1.0 - native[eligible, 0]
            actual = 1.0 - native[eligible, 2]
            counts = chronorep.membership_counts(lo, hi, 0.05)
            native_rows.append(chronorep.capacity_row(model, release, 0.05, n_full, int(eligible.sum()), counts, "completion_time_native"))
            risk_rows.append(chronorep.risk_block(model, release, 0.05, lo, hi, actual, "completion_time_native"))
        matched = result["matched_prob"][:, model_i, :, :]
        common = matched_users & np.all(np.isfinite(matched[:, :, 0]), axis=1) & np.all(np.isfinite(matched[:, :, 1]), axis=1)
        intervals = []
        labels = []
        for release_i, release in enumerate(RELEASES):
            cell = matched[common, release_i, :]
            lo = 1.0 - cell[:, 1]
            hi = 1.0 - cell[:, 0]
            actual = 1.0 - cell[:, 2]
            counts = chronorep.membership_counts(lo, hi, 0.05)
            matched_rows.append(chronorep.capacity_row(model, release, 0.05, n_full, int(common.sum()), counts, "matched_r11_event"))
            risk_rows.append(chronorep.risk_block(model, release, 0.05, lo, hi, actual, "matched_r11_event"))
            intervals.append((cell[:, 0], cell[:, 1]))
            labels.append(counts["labels"])
        for edge_i, (left, right) in enumerate(EDGES):
            left_i = RELEASES.index(left)
            right_i = RELEASES.index(right)
            lo_l, hi_l = intervals[left_i]
            lo_r, hi_r = intervals[right_i]
            contraction_fail = int(np.sum((lo_r < lo_l) | (hi_r > hi_l)))
            gin_loss = int(np.sum((labels[left_i] == "guaranteed_in") & (labels[right_i] != "guaranteed_in")))
            gout_loss = int(np.sum((labels[left_i] == "guaranteed_out") & (labels[right_i] != "guaranteed_out")))
            transition_rows_out.append({
                "model": model,
                "left": left,
                "right": right,
                "n_matched": int(common.sum()),
                "interval_contraction_failures": contraction_fail,
                "guaranteed_in_not_retained": gin_loss,
                "guaranteed_out_not_retained": gout_loss,
                "max_lo_decrease": float(np.max(lo_l - lo_r)) if len(lo_l) else 0.0,
                "max_hi_increase": float(np.max(hi_r - hi_l)) if len(hi_l) else 0.0,
            })
    return native_rows, matched_rows, risk_rows, transition_rows_out


def bootstrap(result: dict) -> list[dict]:
    rng = np.random.default_rng(20312)
    n_users = len(result["n_valid"])
    draws = rng.integers(0, n_users, size=(2000, n_users))
    rows = []
    status = result["status"]
    n_valid = result["n_valid"].astype(np.float64)
    for model_i, model in enumerate(MODELS):
        samples = {name: [] for name in ("NC00", "NC10", "NC01", "NC11", "delta_H", "delta_A", "delta_H_given_A", "delta_A_given_H", "interaction", "joint")}
        point = {}
        for release_i, release in enumerate(RELEASES):
            unsup = status[:, model_i, release_i, 1, 0].astype(np.float64)
            sett = status[:, model_i, release_i, 1, 2].astype(np.float64)
            nc = unsup + sett
            point[release] = float(nc.sum() / VALID)
            rate = nc[draws].sum(axis=1) / n_valid[draws].sum(axis=1)
            samples[{"R00": "NC00", "R10": "NC10", "R01": "NC01", "R11": "NC11"}[release]] = rate
        samples["delta_H"] = samples["NC10"] - samples["NC00"]
        samples["delta_A"] = samples["NC01"] - samples["NC00"]
        samples["delta_H_given_A"] = samples["NC11"] - samples["NC01"]
        samples["delta_A_given_H"] = samples["NC11"] - samples["NC10"]
        samples["interaction"] = samples["NC11"] - samples["NC10"] - samples["NC01"] + samples["NC00"]
        samples["joint"] = samples["NC11"] - samples["NC00"]
        points = {
            "NC00": point["R00"], "NC10": point["R10"], "NC01": point["R01"], "NC11": point["R11"],
            "delta_H": point["R10"] - point["R00"], "delta_A": point["R01"] - point["R00"],
            "delta_H_given_A": point["R11"] - point["R01"], "delta_A_given_H": point["R11"] - point["R10"],
            "interaction": point["R11"] - point["R10"] - point["R01"] + point["R00"],
            "joint": point["R11"] - point["R00"],
        }
        for name, values in samples.items():
            low, high = np.percentile(values, [2.5, 97.5])
            rows.append({
                "analysis": "event_nc_learner_bootstrap",
                "model": model,
                "quantity": name,
                "point_rate": points[name],
                "ci_low": float(low),
                "ci_high": float(high),
                "replicates": 2000,
                "failures": 0,
                "seed": 20312,
                "interval": "percentile_2.5_97.5",
                "resample": "confirmatory learners with replacement; duplicate learners keep all of their events",
            })
    matched = result["matched_prob"]
    common = np.flatnonzero(np.isfinite(matched[:, 0, 0, 0]) & np.all(np.isfinite(matched[:, 0, :, 0]), axis=1))
    if len(common) == 0:
        return rows
    for model_i, model in enumerate(MODELS):
        for release_i, release in enumerate(RELEASES):
            cell = matched[common, model_i, release_i, :]
            lo0 = 1.0 - cell[:, 1]
            hi0 = 1.0 - cell[:, 0]
            point_counts = chronorep.membership_counts(lo0, hi0, 0.05)
            unresolved = np.empty(2000, dtype=np.float64)
            for replicate, draw in enumerate(rng.integers(0, len(common), size=(2000, len(common)))):
                counts = chronorep.membership_counts(lo0[draw], hi0[draw], 0.05)
                unresolved[replicate] = counts["unresolved"] / len(common)
            low, high = np.percentile(unresolved, [2.5, 97.5])
            rows.append({
                "analysis": "matched_capacity_unresolved_rate",
                "model": model,
                "quantity": f"{release}_unresolved_rate",
                "point_rate": point_counts["unresolved"] / len(common),
                "ci_low": float(low),
                "ci_high": float(high),
                "replicates": 2000,
                "failures": 0,
                "seed": 20312,
                "interval": "percentile_2.5_97.5",
                "resample": "matched R11-supported learners with replacement; duplicate positions are separate ranking positions",
            })
    return rows


def independent_qa(pilot, gru, attention, grouped, bundles) -> dict:
    source = np.memmap(OUT / "private_event_cache" / "source_row.i8", dtype=np.int64, mode="r", shape=(VALID,))
    scores = np.memmap(OUT / "private_event_cache" / "scores.f8", dtype=np.float64, mode="r", shape=(VALID, 3, 4, 3))
    index = {int(source[i]): i for i in range(VALID)}
    pool = []
    for uid, rows in grouped.items():
        by_start, by_end = chronorep.order_valid(rows)
        ends = np.asarray([item["end"] for item in by_end], dtype=np.int64)
        unfinished_available = any(not row["valid"] or True for row in rows)
        for row in by_start:
            event_i = index.get(row["source_row"])
            if event_i is None:
                continue
            cut = int(np.searchsorted(ends, row["start"], side="left"))
            prefix_n = cut
            unfinished = any(other["source_row"] != row["source_row"] and other["start"] < row["start"] and other["end"] >= row["start"] for other in rows)
            rank = hashlib.sha256(f"WSDM203-G12-QA|{row['source_row']}".encode()).hexdigest()
            pool.append((rank, uid, row, event_i, prefix_n, unfinished, by_end, cut))
    pool.sort()
    chosen = set()

    def flags_of(event_i: int, row: dict) -> dict:
        return {release: supported_event(bundles[release].get(release_of(row, release))) for release in RELEASES}

    def take(predicate, limit: int) -> list:
        out = []
        for item in pool:
            if len(out) >= limit or len(chosen) >= 250:
                break
            if item[3] in chosen or not predicate(item):
                continue
            out.append(item)
            chosen.add(item[3])
        return out

    strata = {
        "short_history": take(lambda item: item[4] <= 2, 20),
        "long_history": take(lambda item: item[4] >= 20, 20),
        "high_hint": take(lambda item: item[2]["hint_count"] >= 3, 15),
        "low_hint": take(lambda item: item[2]["hint_count"] == 0, 15),
        "high_attempt": take(lambda item: item[2]["attempt_count"] >= 3, 15),
        "low_attempt": take(lambda item: item[2]["attempt_count"] <= 1, 15),
        "supported_r00_not_r10": take(lambda item: flags_of(item[3], item[2])["R00"] and not flags_of(item[3], item[2])["R10"], 20),
        "supported_r00_not_r01": take(lambda item: flags_of(item[3], item[2])["R00"] and not flags_of(item[3], item[2])["R01"], 20),
        "supported_all_four": take(lambda item: all(flags_of(item[3], item[2]).values()), 20),
        "near_threshold": take(lambda item: any(
            np.isfinite(scores[item[3], model, release, 2]) and min(abs(float(scores[item[3], model, release, k]) - tau) for k in range(3) for tau in TAUS) <= 1e-4
            for model in range(3) for release in range(4)
        ), 20),
        "unfinished_prior": take(lambda item: item[5], 10),
    }
    fill = take(lambda item: True, 250)
    selected = [item for group in strata.values() for item in group]
    selected.extend(fill)
    max_diff = 0.0
    order_diff = 0.0
    label_mismatch = token_mismatch = history_mismatch = future = leak = nesting = key_mismatch = 0
    direct_subset_mismatch = 0
    checked = 0
    for _rank, uid, row, event_i, _prefix_n, _unfinished, by_end, cut in selected[:250]:
        checked += 1
        prefix_rows = by_end[:cut]
        if any(item["end"] >= row["start"] for item in prefix_rows):
            future += 1
        direct = [item["source_row"] for item in by_end if item["end"] < row["start"]]
        if [item["source_row"] for item in prefix_rows] != direct:
            history_mismatch += 1
        prefix = [chronorep.as_event(item) for item in prefix_rows[-MAX_EVAL_PREFIX_EVENTS:]]
        ev = chronorep.as_event(row)
        tokens = pilot._tokens(prefix, ev, False)
        reference = chronorep.independent_tokens(pilot, prefix, ev)
        if tokens != reference:
            token_mismatch += 1
        if row["action_types"] and tokens[1] and tokens[1][-1] in (pilot.model.outcome0, pilot.model.outcome1):
            leak += 1
        for release in RELEASES:
            key = release_of(row, release)
            if release == "R10" and str(row["hint_count"]) != key[3]:
                key_mismatch += 1
            if release == "R01" and str(row["attempt_count"]) != key[3]:
                key_mismatch += 1
            if release == "R11" and (str(row["hint_count"]) != key[3] or str(row["attempt_count"]) != key[4]):
                key_mismatch += 1
        flags = flags_of(event_i, row)
        for left, right in EDGES:
            if flags[right] and not flags[left]:
                nesting += 1
            if flags[left] and flags[right]:
                left_i = RELEASES.index(left)
                right_i = RELEASES.index(right)
                for model_i in range(3):
                    lo_l, hi_l = float(scores[event_i, model_i, left_i, 0]), float(scores[event_i, model_i, left_i, 1])
                    lo_r, hi_r = float(scores[event_i, model_i, right_i, 0]), float(scores[event_i, model_i, right_i, 1])
                    if lo_r < lo_l or hi_r > hi_l:
                        nesting += 1
                    for tau in TAUS:
                        code_l = classify(lo_l, hi_l, float(scores[event_i, model_i, left_i, 2]), tau)[0]
                        code_r = classify(lo_r, hi_r, float(scores[event_i, model_i, right_i, 2]), tau)[0]
                        action_l = classify(lo_l, hi_l, float(scores[event_i, model_i, left_i, 2]), tau)[1]
                        action_r = classify(lo_r, hi_r, float(scores[event_i, model_i, right_i, 2]), tau)[1]
                        if code_r == 2 and code_l == 1:
                            nesting += 1
                        if code_l == 1 and code_r == 1 and action_l != action_r:
                            nesting += 1
        serial_gru = _serial(gru, tokens)
        serial_att = _serial(attention, tokens)
        max_diff = max(max_diff, abs(serial_gru - float(scores[event_i, 1, 0, 2])) if np.isfinite(scores[event_i, 1, 0, 2]) else abs(serial_gru - float(chronorep.score_forward(gru, [tokens], pilot.device)[0])))
        stored_att = float(scores[event_i, 2, 0, 2]) if np.isfinite(scores[event_i, 2, 0, 2]) else serial_att
        max_diff = max(max_diff, abs(serial_att - stored_att))
        for model_i, serial in ((1, serial_gru), (2, serial_att)):
            for release_i in range(4):
                if not np.isfinite(scores[event_i, model_i, release_i, 2]):
                    continue
                for tau in TAUS:
                    stored_action = "progress" if float(scores[event_i, model_i, release_i, 2]) >= tau else "review"
                    serial_action = "progress" if serial >= tau else "review"
                    if stored_action != serial_action:
                        label_mismatch += 1
        # Direct rescore of one supported finer class, plus reversed order of the R00 class when it is not huge.
        target_release = next((release for release in ("R10", "R01", "R11", "R00") if flags[release]), None)
        if target_release is not None:
            slot = bundles[target_release].get(release_of(row, target_release))
            own = chronorep.gru_fp(row)
            fps = list(slot.gru)
            if 0 < len(fps) <= 2000:
                forward = [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in fps if fp != own]
                reverse = list(reversed(forward))
                if forward:
                    a = chronorep.score_forward(gru, forward, pilot.device)
                    b = chronorep.score_forward(gru, reverse, pilot.device)
                    order_diff = max(order_diff, abs(float(np.min(a)) - float(np.min(b))), abs(float(np.max(a)) - float(np.max(b))))
                    values = list(a)
                    if own in slot.gru:
                        values.append(serial_gru)
                    lo, hi = min(values), max(values)
                    release_i = RELEASES.index(target_release)
                    if abs(lo - float(scores[event_i, 1, release_i, 0])) > 0 or abs(hi - float(scores[event_i, 1, release_i, 1])) > 0:
                        # A one-ulp gap is recorded in max_diff. A label change is a mismatch.
                        max_diff = max(max_diff, abs(lo - float(scores[event_i, 1, release_i, 0])), abs(hi - float(scores[event_i, 1, release_i, 1])))
                        for tau in TAUS:
                            if classify(lo, hi, serial_gru, tau)[0] != classify(float(scores[event_i, 1, release_i, 0]), float(scores[event_i, 1, release_i, 1]), float(scores[event_i, 1, release_i, 2]), tau)[0]:
                                direct_subset_mismatch += 1
                    if order_diff > 0:
                        for tau in TAUS:
                            lo_b = min(list(b) + ([serial_gru] if own in slot.gru else []))
                            hi_b = max(list(b) + ([serial_gru] if own in slot.gru else []))
                            if classify(lo, hi, serial_gru, tau)[:2] != classify(lo_b, hi_b, serial_gru, tau)[:2]:
                                direct_subset_mismatch += 1
    payload = {
        "events": checked,
        "strata_selected": {name: len(group) for name, group in strata.items()},
        "max_abs_diff": max_diff,
        "order_minmax_max_abs": order_diff,
        "numeric_tolerance": float(np.finfo(np.float32).eps),
        "label_mismatches": label_mismatch,
        "direct_or_order_label_mismatches": direct_subset_mismatch,
        "token_mismatches": token_mismatch,
        "history_mismatches": history_mismatch,
        "future_or_unfinished_in_prefix": future,
        "current_outcome_leaks": leak,
        "nesting_mismatches": nesting,
        "source_key_mismatches": key_mismatch,
        "pass": checked >= 250 and max_diff <= float(np.finfo(np.float32).eps) and label_mismatch == 0 and direct_subset_mismatch == 0 and token_mismatch == 0 and history_mismatch == 0 and future == 0 and leak == 0 and nesting == 0 and key_mismatch == 0,
    }
    dump("independent_scoring_qa.json", payload)
    return payload


def _serial(model, tokens) -> float:
    skills, atypes = tokens
    if not skills:
        return 0.5
    with torch.inference_mode():
        pred, _ = model.forward_from_state(torch.tensor([skills], dtype=torch.long), torch.tensor([atypes], dtype=torch.long), None)
        return float(pred)


def frozen_hashes() -> dict:
    paths = {
        "gru_checkpoint": CHRONOREP_CKPT,
        "attention_checkpoint": ATT_CKPT,
        "r7_core": ROOT / "scripts/wdlf_r7/r7_core.py",
        "chronorep_frontier": ROOT / "results/assistments/chronological_replication/event_certificate_frontier.csv",
        "attncons_frontier": ROOT / "results/assistments/attention_consumer/event_certificate_frontier.csv",
        "thrfs_grid": ROOT / "results/assistments/attention_consumer/threshold_false_stable/threshold_false_stable_full_grid.csv",
    }
    return {name: chronorep.sha256_file(path) for name, path in paths.items()}


def write_checksums() -> dict:
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    payload = {"algorithm": "sha256", "excluded": ["results/assistments/release_field_ablation/output_checksums.json"], "n_files": len(files), "files": files}
    dump("output_checksums.json", payload)
    bad = [item["path"] for item in files if chronorep.sha256_file(ROOT / item["path"]) != item["sha256"]]
    return {"n_files": len(files), "mismatches": bad}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    print("preflight", flush=True)
    pre = chronorep.preflight()
    pre["gru_checkpoint_sha256"] = chronorep.sha256_file(CHRONOREP_CKPT)
    pre["attention_checkpoint_sha256"] = chronorep.sha256_file(ATT_CKPT)
    pre["gru_match"] = pre["gru_checkpoint_sha256"] == GRU_SHA
    pre["attention_match"] = pre["attention_checkpoint_sha256"] == ATT_SHA
    pre["pass"] = bool(pre["pass"] and pre["gru_match"] and pre["attention_match"])
    before = frozen_hashes()
    pre["frozen_artifact_sha256"] = before
    dump("source_and_checkpoint_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_MODEL_IDENTITY"})
        write_checksums()
        raise SystemExit(2)
    gru_state = torch.load(CHRONOREP_CKPT, map_location="cpu", weights_only=False)
    att_state = torch.load(ATT_CKPT, map_location="cpu", weights_only=False)
    skill2i = gru_state["skill2i"]
    atype2i = gru_state["atype2i"]
    n_skills = len(skill2i) - 1
    n_atypes = len(atype2i) - 1
    gru = ActionTokenGRU(n_skills, n_atypes)
    gru.load_state_dict(gru_state["state_dict"])
    attention = attncons.CausalTokenAttentionKT(n_skills, n_atypes)
    attention.load_state_dict(att_state["state_dict"])
    gru.eval()
    attention.eval()
    pilot = chronorep.NeuralPilot.__new__(chronorep.NeuralPilot)
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.model = gru
    pilot.device = torch.device("cpu")
    gru_params = int(sum(p.numel() for p in gru.parameters()))
    att_params = int(sum(p.numel() for p in attention.parameters()))
    if gru_params != 33857 or att_params != 88897:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_MODEL_IDENTITY", "gru_params": gru_params, "attention_params": att_params})
        write_checksums()
        raise SystemExit(3)
    print("load confirmatory", flush=True)
    grouped = chronorep.load_confirmatory()
    print("development geometry", flush=True)
    geometry = build_development()
    write_csv("development_key_geometry.csv", geometry["geometry_rows"])
    dump("support_lattice_checks.json", {
        "development_events": geometry["events"],
        "unique_rich_signatures": geometry["unique_rich"],
        "duplicate_rich_signatures": geometry["duplicate_rich"],
        "release_key_mismatches_versus_s0_s1": geometry["key_mismatches"],
        "fingerprint_subset_checks": geometry["subset_checks"],
        "fingerprint_subset_violations": geometry["subset_violations"],
        "event_versus_rich_predicate_disagreements_by_release": geometry["predicate_disagreements"],
        "support_predicate": "n_users >= 2 or development_event_count >= 2",
        "rich_predicate_recorded_for_comparison": "n_users >= 2 or unique_rich_signatures >= 2",
        "candidate_construction": "Observed development interactions only. No Cartesian product of hint and attempt values.",
    })
    dump("release_lattice_protocol_freeze.json", {
        "releases": {
            "R00": "(problem_id, correct)",
            "R10": "(problem_id, correct, hint_count)",
            "R01": "(problem_id, correct, attempt_count)",
            "R11": "(problem_id, correct, hint_count, attempt_count)",
        },
        "field_source": "original hint_count and attempt_count columns after the frozen integer normalization",
        "not_derived_from_action_stream": True,
        "history": "same student, valid interval, end < start, ordered by (end, source_row)",
        "models": list(MODELS),
        "no_hist_or_len": True,
        "training": False,
        "primary_tau": PRIMARY,
        "kappa": 0.05,
        "tie_rule": "adversarial",
        "bootstrap_seed": 20312,
        "parameter_counts": {"ActionTokenGRU": gru_params, "CausalTokenAttentionKT": att_params},
    })
    if geometry["events"] != 4286981 or geometry["key_mismatches"] or geometry["subset_violations"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_MODEL_IDENTITY", "geometry": {k: geometry[k] for k in ("events", "key_mismatches", "subset_violations")}})
        write_checksums()
        raise SystemExit(4)
    plans = checkpoint_plans(grouped, geometry["bundles"])
    if plans["valid"] != VALID or plans["invalid"] != 91:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_MODEL_IDENTITY", "valid": plans["valid"], "invalid": plans["invalid"]})
        write_checksums()
        raise SystemExit(4)
    progress = json.loads((OUT / "execution_progress.json").read_text()) if (OUT / "execution_progress.json").exists() else {}
    if progress.get("complete") and (OUT / "private_event_cache" / "running_state.npz").exists():
        print("reuse completed score cache", flush=True)
        result = load_completed(len(plans["users"]))
    else:
        print("score", flush=True)
        result = score_population(pilot, gru, attention, grouped, geometry["bundles"], plans)
    result = repair_unsupported(result)
    rows = certificate_rows(result)
    write_csv("event_certificate_factorial.csv", [row for row in rows if row["tau"] == PRIMARY])
    write_csv("threshold_sensitivity.csv", rows)
    write_csv("false_stable_and_fingerprint_coverage.csv", rows)
    effects, parts = effect_rows(rows)
    write_csv("factorial_nc_effects.csv", effects)
    write_csv("factorial_support_ambiguity_decomposition.csv", parts)
    transitions = transition_rows(result)
    write_csv("paired_event_transitions.csv", transitions)
    write_csv("common_supported_event_analysis.csv", common_rows(result))
    native_rows, matched_rows, risk_rows, capacity_transitions = capacity_tables(result)
    write_csv("native_completion_capacity.csv", native_rows)
    write_csv("matched_event_capacity.csv", matched_rows)
    write_csv("matched_capacity_transitions.csv", capacity_transitions)
    write_csv("realized_risk_validity.csv", risk_rows)
    reproduced = reproduction(rows)
    print("bootstrap", flush=True)
    write_csv("learner_bootstrap_intervals.csv", bootstrap(result))
    print("qa", flush=True)
    qa = independent_qa(pilot, gru, attention, grouped, geometry["bundles"])
    after = frozen_hashes()
    matrix_ok = all(row["matrix_sum"] == VALID for row in transitions)
    accounting = all(row["unsupported"] + row["stable"] + row["set_valued"] == VALID for row in rows)
    support_fixed = all(len({row["unsupported"] for row in rows if row["release"] == release}) == 1 for release in RELEASES)
    coverage_fixed = all(len({row["fingerprint_covered_supported"] for row in rows if row["model"] == model and row["release"] == release}) == 1 for model in MODELS for release in RELEASES)
    fs_bound = all(row["false_stable"] <= row["stable_uncovered"] for row in rows)
    no_pt = not any(path.suffix == ".pt" for path in OUT.rglob("*"))
    contraction_ok = all(row["interval_contraction_failures"] == 0 for row in capacity_transitions)
    status = "COMPLETE"
    if not reproduced["pass"]:
        status = "BLOCKED_BASELINE_REPRODUCTION"
    elif result["counters"]["support_violation"] or result["counters"]["set_violation"] or result["counters"]["action_violation"] or result["counters"]["future"] or result["counters"]["implication"] or not qa["pass"] or not matrix_ok or not contraction_ok or before != after:
        status = "BLOCKED_NUMERICAL_OR_NESTING_INCONSISTENCY"
    invariants = {
        "valid_targets": int(result["events"]),
        "invalid_excluded": plans["invalid"],
        "accounting": accounting,
        "support_identical_across_consumers_at_r00": support_fixed,
        "coverage_independent_of_tau": coverage_fixed,
        "false_stable_not_above_uncovered": fs_bound,
        "transition_matrices_sum_to_valid": matrix_ok,
        "baseline_reproduction": reproduced["pass"],
        "future_violations": result["counters"]["future"],
        "implication_violations": result["counters"]["implication"],
        "support_nesting_violations": result["counters"]["support_violation"],
        "set_nesting_violations": result["counters"]["set_violation"],
        "stable_action_violations": result["counters"]["action_violation"],
        "predicate_disagreements_on_confirmatory_events": result["counters"]["predicate_disagreement"],
        "empty_candidate_classes": result["counters"]["empty_class"],
        "plan_mismatches": result["counters"]["plan_mismatch"],
        "interval_contraction": contraction_ok,
        "independent_qa_pass": qa["pass"],
        "frozen_artifacts_unchanged": before == after,
        "no_new_checkpoint": no_pt,
        "training_occurred": False,
        "status": status,
    }
    dump("numerical_invariants.json", invariants)
    dump("experiment_manifest.json", {
        "experiment": "pipeline",
        "script": "scripts/assistments/release_field_ablation.py",
        "script_sha256": chronorep.sha256_file(Path(__file__).resolve()),
        "elapsed_sec": time.time() - started,
        "status": status,
        "gru_sha256": pre["gru_checkpoint_sha256"],
        "attention_sha256": pre["attention_checkpoint_sha256"],
        "training_occurred": False,
        "new_checkpoint_created": False,
    })
    checksums = write_checksums()
    print(status, "checksum_mismatches", len(checksums["mismatches"]), flush=True)
    if status != "COMPLETE" or checksums["mismatches"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
