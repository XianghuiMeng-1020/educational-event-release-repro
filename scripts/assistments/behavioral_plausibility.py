#!/usr/bin/env python3
"""History-conditioned restrictions on the empirical completion library.

No training is performed and earlier outputs are not modified.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("chronorep", ROOT / "scripts/assistments/chronological_replication.py")
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)
DECBASE = importlib.util.spec_from_file_location("decbase", ROOT / "scripts/assistments/decision_baselines.py")
decbase = importlib.util.module_from_spec(DECBASE)
DECBASE.loader.exec_module(decbase)
from r7_core import NeuralPilot  # noqa: E402

OUT = ROOT / "results/assistments/behavioral_plausibility"
PRIV = OUT / "private_scores"
SUPPRED_PRIV = ROOT / "results/assistments/support_predicate_robustness/private_scores"
CHRONOREP_CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
SCHEMAS = ("S0", "S1", "S2", "S3")
FILTERS = ("F0", "F1", "F2", "F3")
MODELS = ("TransparentActionPFA", "ActionTokenGRU")
TAU = 0.80
TAUS = (0.70, 0.80, 0.90)
BOOT_SEED = 20319
BOOT_REPS = 2000
EXPECTED = {
    ("TransparentActionPFA", "S0"): (916845, 2566, 20408, 45),
    ("TransparentActionPFA", "S1"): (861927, 15, 77877, 0),
    ("TransparentActionPFA", "S2"): (859025, 4, 80790, 0),
    ("TransparentActionPFA", "S3"): (839131, 0, 100688, 0),
    ("ActionTokenGRU", "S0"): (810576, 108835, 20408, 1077),
    ("ActionTokenGRU", "S1"): (846772, 15170, 77877, 252),
    ("ActionTokenGRU", "S2"): (844157, 14872, 80790, 250),
    ("ActionTokenGRU", "S3"): (839131, 0, 100688, 0),
}
EXPECTED_CAL = {
    ("TransparentActionPFA", "S0"): (860290, 2418, 25289, 34),
    ("TransparentActionPFA", "S1"): (808027, 15, 79955, 0),
    ("TransparentActionPFA", "S2"): (805366, 3, 82628, 0),
    ("TransparentActionPFA", "S3"): (786789, 0, 101208, 0),
    ("ActionTokenGRU", "S0"): (757839, 104869, 25289, 1013),
    ("ActionTokenGRU", "S1"): (793857, 14185, 79955, 266),
    ("ActionTokenGRU", "S2"): (791470, 13899, 82628, 266),
    ("ActionTokenGRU", "S3"): (786789, 0, 101208, 0),
}
EXP_NAME = ("E0", "E1", "E2", "E3")
PERF_NAME = ("A0", "A1", "A2", "A3")
STATUS_NAME = ("Unsupported", "Stable", "Set-valued")


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


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    header = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                header.append(key)
    header = header or ["empty"]
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, restval="", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else (format(row[key], ".17g") if isinstance(row.get(key), float) else row.get(key, "")) for key in header})


def progress(stage: str, **extra) -> None:
    dump("execution_progress.json", {"stage": stage, "elapsed_sec": time.time() - STARTED, **extra})
    print(stage, extra if extra else "", flush=True)


def bins_from_counts(n: int, s: int) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    if n <= 2:
        exposure = 1
    elif n <= 9:
        exposure = 2
    else:
        exposure = 3
    num = 3 * (s + 1)
    den = n + 2
    if num < den:
        performance = 1
    elif num < 2 * den:
        performance = 2
    else:
        performance = 3
    return exposure, performance


def window_counts(window: list[dict], skill: str) -> tuple[int, int]:
    n = s = 0
    for row in window:
        if row["skill_id"] == skill:
            n += 1
            s += int(row["correct"])
    return n, s


def direct_features(by_end: list[dict], start: int, skill: str) -> tuple[int, int, int, int]:
    chosen = [row for row in by_end if row["end"] < start]
    n, s = window_counts(chosen[-40:], skill)
    exposure, performance = bins_from_counts(n, s)
    return n, s, exposure, performance


def indexed_features(by_end: list[dict], ends: np.ndarray, start: int, skill: str) -> tuple[int, int, int, int]:
    cut = int(np.searchsorted(ends, start, side="left"))
    n, s = window_counts(by_end[max(0, cut - 40):cut], skill)
    exposure, performance = bins_from_counts(n, s)
    return n, s, exposure, performance


def behavior_digest(row: dict) -> bytes:
    behavior = (
        str(row["problem_id"]),
        int(row["correct"]),
        int(row["hint_count"]),
        int(row["attempt_count"]),
        str(row["first_action"]),
        tuple(row["action_types"]),
    )
    return hashlib.blake2s(repr(behavior).encode(), digest_size=16).digest()


def rich_digest(row: dict, timestamps) -> bytes:
    stamped = (
        str(row["problem_id"]),
        int(row["correct"]),
        int(row["hint_count"]),
        int(row["attempt_count"]),
        str(row["first_action"]),
        tuple(row["action_types"]),
        tuple(str(tok) if tok is not None else "" for tok in (timestamps or [])),
    )
    return hashlib.blake2s(repr(stamped).encode(), digest_size=16).digest()


class Library:
    def __init__(self) -> None:
        self.user = None
        self.exposure = None
        self.performance = None
        self.available = None
        self.pfa_id = None
        self.gru_id = None
        self.behavior_id = None
        self.keys = {}
        self.ids = {}
        self.pfa_deltas = None
        self.gru_values = []
        self.pfa_lookup = {}
        self.gru_lookup = {}


def load_grouped(split: str) -> tuple[dict, dict]:
    grouped = defaultdict(list)
    timestamps = {}
    parquet = __import__("pyarrow.parquet").parquet.ParquetFile(chronorep.RECON / f"{split}.parquet")
    events = invalid = 0
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            item = chronorep.compact_row(row, int(start), int(end))
            grouped[str(row["user_id"])].append(item)
            if split == "development":
                timestamps[int(row["source_row"])] = row.get("action_timestamps")
            events += 1
            invalid += int(not item["valid"])
        if events % 1000000 == 0:
            print(f"load_{split} {events} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["source_row"])
    return grouped, {"events": events, "invalid": invalid, "users": len(grouped), "timestamps": timestamps}


def build_library(grouped: dict, timestamps: dict) -> tuple[Library, dict]:
    lib = Library()
    user_names = sorted(grouped)
    n = sum(len(rows) for rows in grouped.values())
    user = np.empty(n, dtype=np.uint16)
    exposure = np.empty(n, dtype=np.uint8)
    performance = np.empty(n, dtype=np.uint8)
    available = np.empty(n, dtype=np.uint8)
    pfa_id = np.empty(n, dtype=np.int32)
    gru_id = np.empty(n, dtype=np.int32)
    behavior_id = np.empty(n, dtype=np.int32)
    skill_code = np.empty(n, dtype=np.int32)
    source = np.empty(n, dtype=np.int64)
    skill_lookup = {}
    pfa_lookup = {}
    gru_lookup = {}
    behavior_lookup = {}
    pfa_values = []
    gru_values = []
    key_lists = {schema: defaultdict(list) for schema in SCHEMAS}
    rich_seen = set()
    rich_dup = nest_fail = unavailable = invalid = valid_start_invalid_end = 0
    overlap_events = tied_ends = source_inversions = unmapped_skill = 0
    exp_counts = np.zeros(4, dtype=np.int64)
    perf_counts = np.zeros(4, dtype=np.int64)
    cursor = 0
    for user_i, uid in enumerate(user_names):
        rows = grouped[uid]
        valid = [row for row in rows if row["valid"]]
        by_end = sorted(valid, key=lambda row: (row["end"], row["source_row"]))
        ends = np.asarray([row["end"] for row in by_end], dtype=np.int64)
        by_start = sorted(valid, key=lambda row: (row["start"], row["source_row"]))
        heap = []
        for row in by_start:
            while heap and heap[0][0] < row["start"]:
                heapq.heappop(heap)
            if heap:
                overlap_events += 1
            heapq.heappush(heap, (row["end"], row["source_row"]))
        for left, right in zip(by_end, by_end[1:]):
            tied_ends += int(left["end"] == right["end"])
        by_source = sorted(valid, key=lambda row: row["source_row"])
        for left, right in zip(by_source, by_source[1:]):
            source_inversions += int((left["end"], left["source_row"]) > (right["end"], right["source_row"]))
        for row in rows:
            invalid += int(not row["valid"])
            start_ok = row["start"] != chronorep.NAT
            end_ok = row["valid"]
            if start_ok and not end_ok:
                valid_start_invalid_end += 1
            if not start_ok:
                unavailable += 1
                exp = perf = 0
                avail = 0
            else:
                exp_n, correct_n, exp, perf = indexed_features(by_end, ends, row["start"], row["skill_id"])
                avail = 1
                exp_counts[exp] += 1
                perf_counts[perf] += 1
                if exp_n < 0 or correct_n < 0:
                    nest_fail += 1
            pfa = chronorep.pfa_fp(row)
            gru = chronorep.gru_fp(row)
            pid = pfa_lookup.get(pfa)
            if pid is None:
                pid = len(pfa_values)
                pfa_lookup[pfa] = pid
                pfa_values.append(pfa)
            gid = gru_lookup.get(gru)
            if gid is None:
                gid = len(gru_values)
                gru_lookup[gru] = gid
                gru_values.append(gru)
            digest = behavior_digest(row)
            bid = behavior_lookup.get(digest)
            if bid is None:
                bid = len(behavior_lookup)
                behavior_lookup[digest] = bid
            stamped = rich_digest(row, timestamps.get(row["source_row"]))
            rich_dup += stamped in rich_seen
            rich_seen.add(stamped)
            ev = chronorep.as_event(row)
            keys = [chronorep.release_key(ev, schema) for schema in SCHEMAS]
            if chronorep.project_down(keys[3], "S3", "S0") != keys[0] or chronorep.project_down(keys[3], "S3", "S1") != keys[1] or chronorep.project_down(keys[3], "S3", "S2") != keys[2]:
                nest_fail += 1
            for schema, key in zip(SCHEMAS, keys):
                key_lists[schema][key].append(cursor)
            user[cursor] = user_i
            exposure[cursor] = exp
            performance[cursor] = perf
            available[cursor] = avail
            pfa_id[cursor] = pid
            gru_id[cursor] = gid
            behavior_id[cursor] = bid
            skill_code[cursor] = skill_lookup.setdefault(row["skill_id"], len(skill_lookup))
            source[cursor] = row["source_row"]
            cursor += 1
        if user_i and user_i % 5000 == 0:
            print(f"context_users {user_i} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    lib.user = user
    lib.exposure = exposure
    lib.performance = performance
    lib.available = available
    lib.pfa_id = pfa_id
    lib.gru_id = gru_id
    lib.behavior_id = behavior_id
    lib.skill = skill_code
    lib.skill_lookup = skill_lookup
    lib.source = source
    lib.gru_values = gru_values
    lib.pfa_lookup = pfa_lookup
    lib.gru_lookup = gru_lookup
    lib.pfa_deltas = np.asarray([chronorep.fp_delta(fp) for fp in pfa_values], dtype=np.float64)
    for schema in SCHEMAS:
        mapping = {}
        chunks = []
        start = 0
        for key, values in key_lists[schema].items():
            arr = np.asarray(values, dtype=np.int32)
            mapping[key] = (start, len(arr))
            chunks.append(arr)
            start += len(arr)
        lib.keys[schema] = mapping
        lib.ids[schema] = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
    audit = {
        "development_events": n,
        "development_users": len(user_names),
        "invalid_intervals_retained_in_library": invalid,
        "candidates_with_unusable_start": unavailable,
        "valid_start_invalid_end": valid_start_invalid_end,
        "available_pre_event_context": int(available.sum()),
        "events_with_an_overlapping_active_interaction": overlap_events,
        "adjacent_tied_completion_times": tied_ends,
        "adjacent_source_order_inversions_versus_completion_time": source_inversions,
        "duplicate_rich_signatures": rich_dup,
        "unique_rich_signatures": len(rich_seen),
        "unique_behavior_signatures": len(behavior_lookup),
        "unique_pfa_fingerprints": len(pfa_values),
        "unique_gru_fingerprints": len(gru_values),
        "release_projection_failures": nest_fail,
        "unmapped_skill_count_placeholder": unmapped_skill,
        "exposure_counts_among_available": {EXP_NAME[i]: int(exp_counts[i]) for i in range(4)},
        "performance_counts_among_available": {PERF_NAME[i]: int(perf_counts[i]) for i in range(4)},
        "rich_equals_event_count": rich_dup == 0 and len(rich_seen) == n,
        "feature_window": "last 40 completed valid interactions with end < start, ordered by (end, source_row)",
        "current_outcome_not_used": True,
    }
    return lib, audit


def pilot_development(grouped: dict, lib: Library) -> dict:
    user_names = sorted(grouped)
    selected = []
    invalid_rows = []
    for user_i, uid in enumerate(user_names):
        for row in grouped[uid]:
            rank = int.from_bytes(hashlib.sha256(f"WSDM203-G18-DEV|{row['source_row']}".encode()).digest()[:8], "big")
            item = (-rank, user_i, row["source_row"])
            if len(selected) < 250:
                heapq.heappush(selected, item)
            elif rank < -selected[0][0]:
                heapq.heapreplace(selected, item)
            if not row["valid"] and len(invalid_rows) < 8:
                invalid_rows.append((user_i, row))
    checked = mismatches = unavailable = invalid_checked = 0
    exposure_seen = set()
    performance_seen = set()
    multi_class = 0
    schema_pairs = 0
    samples = [(-rank, user_i, source_row) for rank, user_i, source_row in selected]
    forced = [(0, user_i, row["source_row"]) for user_i, row in invalid_rows]
    seen_sources = set()
    for _rank, user_i, source_row in samples + forced:
        if source_row in seen_sources:
            continue
        seen_sources.add(source_row)
        row = next(item for item in grouped[user_names[user_i]] if item["source_row"] == source_row)
        valid = [item for item in grouped[user_names[user_i]] if item["valid"]]
        by_end = sorted(valid, key=lambda item: (item["end"], item["source_row"]))
        ends = np.asarray([item["end"] for item in by_end], dtype=np.int64) if by_end else np.zeros(0, dtype=np.int64)
        pos = int(np.flatnonzero(lib.source == source_row)[0])
        if not row["valid"]:
            invalid_checked += 1
        if row["start"] == chronorep.NAT:
            unavailable += 1
            mismatches += int(lib.available[pos] != 0)
        else:
            direct = direct_features(by_end, row["start"], row["skill_id"])
            fast = indexed_features(by_end, ends, row["start"], row["skill_id"])
            exposure_seen.add(fast[2])
            performance_seen.add(fast[3])
            if direct != fast or (int(lib.exposure[pos]), int(lib.performance[pos])) != (fast[2], fast[3]) or int(lib.available[pos]) != 1:
                mismatches += 1
        ev = chronorep.as_event(row)
        ids = []
        for schema in SCHEMAS:
            start, count = lib.keys[schema][chronorep.release_key(ev, schema)]
            ids.append(lib.ids[schema][start:start + count])
        pools = [set(block.tolist()) for block in ids]
        schema_pairs += 1
        multi_class += int(len(ids[0]) > 1)
        if not pools[3].issubset(pools[0]):
            mismatches += 1
        for cid in ids[3][:8].tolist():
            if any(cid not in pool for pool in pools[:3]):
                mismatches += 1
        checked += 1
    conflicts = 0
    first_exposure = {}
    for index in range(min(len(lib.gru_id), 500000)):
        if lib.available[index] == 0:
            continue
        gid = int(lib.gru_id[index])
        exposure = int(lib.exposure[index])
        previous = first_exposure.get(gid)
        if previous is None:
            first_exposure[gid] = exposure
        elif previous != exposure:
            conflicts += 1
            break
    return {
        "checked": checked, "mismatches": mismatches, "unavailable_in_sample": unavailable,
        "invalid_interval_checked": invalid_checked, "schema_subset_checks": schema_pairs,
        "exposure_bins_seen": sorted(EXP_NAME[i] for i in exposure_seen),
        "performance_bins_seen": sorted(PERF_NAME[i] for i in performance_seen),
        "multi_candidate_classes": multi_class,
        "shared_fingerprint_different_exposure_found": conflicts,
        "eligibility_depends_only_on_stored_candidate_bins": True,
    }


def pilot_targets(grouped: dict, split: str) -> dict:
    selected = []
    prepared = {}
    for uid, rows in grouped.items():
        valid = [row for row in rows if row["valid"]]
        by_end = sorted(valid, key=lambda row: (row["end"], row["source_row"]))
        ends = np.asarray([row["end"] for row in by_end], dtype=np.int64) if by_end else np.zeros(0, dtype=np.int64)
        prepared[uid] = (by_end, ends)
        for row in valid:
            rank = int.from_bytes(hashlib.sha256(f"WSDM203-G18-{split}|{row['source_row']}".encode()).digest()[:8], "big")
            item = (-rank, uid, row["source_row"])
            if len(selected) < 250:
                heapq.heappush(selected, item)
            elif rank < -selected[0][0]:
                heapq.heapreplace(selected, item)
    checked = mismatches = 0
    exposure_seen = set()
    performance_seen = set()
    for _neg, uid, source_row in selected:
        by_end, ends = prepared[uid]
        row = next(item for item in by_end if item["source_row"] == source_row)
        direct = direct_features(by_end, row["start"], row["skill_id"])
        fast = indexed_features(by_end, ends, row["start"], row["skill_id"])
        mismatches += int(direct != fast)
        exposure_seen.add(direct[2])
        performance_seen.add(direct[3])
        if any(item["source_row"] == row["source_row"] and item["end"] < row["start"] for item in by_end):
            mismatches += 1
        checked += 1
    for uid, (by_end, ends) in prepared.items():
        if exposure_seen == {0, 1, 2, 3} and performance_seen == {0, 1, 2, 3}:
            break
        for row in by_end:
            direct = direct_features(by_end, row["start"], row["skill_id"])
            if direct[2] not in exposure_seen or direct[3] not in performance_seen:
                fast = indexed_features(by_end, ends, row["start"], row["skill_id"])
                mismatches += int(direct != fast)
                exposure_seen.add(direct[2])
                performance_seen.add(direct[3])
                checked += 1
            if exposure_seen == {0, 1, 2, 3} and performance_seen == {0, 1, 2, 3}:
                break
    return {
        "split": split, "checked": checked, "mismatches": mismatches,
        "exposure_bins_seen": sorted(EXP_NAME[i] for i in exposure_seen),
        "performance_bins_seen": sorted(PERF_NAME[i] for i in performance_seen),
        "current_event_excluded_from_own_history": True,
    }


def score_split(split: str, grouped: dict, lib: Library, pilot: NeuralPilot) -> dict:
    users = sorted(grouped)
    n = sum(len(chronorep.order_valid(grouped[uid])[0]) for uid in users)
    user_lookup = {uid: i for i, uid in enumerate(users)}
    user = np.empty(n, dtype=np.int32)
    source = np.empty(n, dtype=np.int64)
    end = np.empty(n, dtype=np.int64)
    realized = np.empty((n, 2), dtype=np.float64)
    lo = np.full((n, 2, 4, 4), np.nan, dtype=np.float64)
    hi = np.full((n, 2, 4, 4), np.nan, dtype=np.float64)
    code = np.zeros((n, 2, 4, 4), dtype=np.uint8)
    library_hit = np.zeros((n, 2, 4, 4), dtype=np.uint8)
    f0_counts = np.zeros((n, 4, 4), dtype=np.int32)
    retained = np.zeros((4, 4), dtype=np.int64)
    original = np.zeros(4, dtype=np.int64)
    zeros = np.zeros((4, 4), dtype=np.int64)
    ones = np.zeros((4, 4), dtype=np.int64)
    sum_u = np.zeros((4, 4), dtype=np.int64)
    sum_b = np.zeros((4, 4), dtype=np.int64)
    sum_f = np.zeros((2, 4, 4), dtype=np.int64)
    exp_counts = np.zeros(4, dtype=np.int64)
    perf_counts = np.zeros(4, dtype=np.int64)
    future = nest_fail = cursor_event = 0
    qa = []
    qa_ranks = []
    for uid in users:
        for row in chronorep.order_valid(grouped[uid])[0]:
            qa_ranks.append((hashlib.sha256(f"WSDM203-G18-QA|{split}|{row['source_row']}".encode()).hexdigest(), row["source_row"]))
    qa_ranks.sort()
    qa_sources = {source_row for _rank, source_row in qa_ranks[:300]}
    direct_sources = {source_row for _rank, source_row in qa_ranks[:200]}
    direct_gap = 0.0
    direct_label = 0
    direct_checked = 0
    alternative_checked = 0
    unexplained = 0
    endpoint_replay = []
    qa_fail = 0
    qa_checked = 0
    skill_disagreements = 0
    direct_order = [source_row for _rank, source_row in qa_ranks[:2000]]
    direct_pool = set(direct_order)
    for user_i, uid in enumerate(users):
        by_start, by_end = chronorep.order_valid(grouped[uid])
        ends = np.asarray([row["end"] for row in by_end], dtype=np.int64)
        state = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        hist = 0
        for row in by_start:
            while hist < len(by_end) and by_end[hist]["end"] < row["start"]:
                done = by_end[hist]
                state[done["skill_id"]] += np.asarray(chronorep.fp_delta(chronorep.pfa_fp(done)), dtype=np.float64)
                hist += 1
            if any(item["end"] >= row["start"] for item in by_end[:hist]):
                future += 1
            _n, _s, exp, perf = indexed_features(by_end, ends, row["start"], row["skill_id"])
            exp_counts[exp] += 1
            perf_counts[perf] += 1
            ev = chronorep.as_event(row)
            prefix = [chronorep.as_event(item) for item in by_end[max(0, hist - 40):hist]]
            base = state[row["skill_id"]]
            own_pfa = chronorep.pfa_fp(row)
            own_gru = chronorep.gru_fp(row)
            pfa_realized = float(chronorep.mastery_of(base + np.asarray(chronorep.fp_delta(own_pfa), dtype=np.float64)))
            blocks = []
            gru_needed = {own_gru: None}
            for schema_i, schema in enumerate(SCHEMAS):
                located = lib.keys[schema].get(chronorep.release_key(ev, schema))
                if located is None:
                    blocks.append(None)
                    continue
                start, count = located
                ids = lib.ids[schema][start:start + count]
                blocks.append(ids)
                for gid in np.unique(lib.gru_id[ids]):
                    gru_needed[lib.gru_values[int(gid)]] = int(gid)
            pairs = [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in gru_needed]
            gru_scores = chronorep.score_forward(pilot.model, pairs, pilot.device)
            gru_map = {fp: float(score) for fp, score in zip(gru_needed, gru_scores)}
            pos = cursor_event
            user[pos] = user_lookup[uid]
            source[pos] = row["source_row"]
            end[pos] = row["end"]
            realized[pos, 0] = pfa_realized
            realized[pos, 1] = gru_map[own_gru]
            own_pid = lib.pfa_lookup.get(own_pfa)
            own_gid = lib.gru_lookup.get(own_gru)
            s0_ids = blocks[0]
            for schema_i, ids in enumerate(blocks):
                if ids is None:
                    zeros[schema_i, :] += 1
                    continue
                avail = lib.available[ids] == 1
                same_exp = avail & (lib.exposure[ids] == exp)
                same_perf = avail & (lib.performance[ids] == perf)
                masks = (np.ones(len(ids), dtype=bool), same_exp, same_perf, same_exp & same_perf)
                original[schema_i] += len(ids)
                if row["source_row"] in qa_sources and schema_i > 0 and s0_ids is not None:
                    if not np.all(np.isin(ids, s0_ids)):
                        nest_fail += 1
                        qa_fail += 1
                for filter_i, mask in enumerate(masks):
                    sub = ids[mask]
                    n_ret = int(len(sub))
                    retained[schema_i, filter_i] += n_ret
                    zeros[schema_i, filter_i] += int(n_ret == 0)
                    ones[schema_i, filter_i] += int(n_ret == 1)
                    if n_ret == 0:
                        continue
                    u_count = int(np.unique(lib.user[sub]).size)
                    b_count = int(np.unique(lib.behavior_id[sub]).size)
                    pfa_ids = np.unique(lib.pfa_id[sub])
                    gru_ids = np.unique(lib.gru_id[sub])
                    sum_u[schema_i, filter_i] += u_count
                    sum_b[schema_i, filter_i] += b_count
                    sum_f[0, schema_i, filter_i] += int(len(pfa_ids))
                    sum_f[1, schema_i, filter_i] += int(len(gru_ids))
                    if filter_i == 0:
                        f0_counts[pos, schema_i] = (u_count, n_ret, len(pfa_ids), len(gru_ids))
                    supported = u_count >= 2 or n_ret >= 2
                    if not supported:
                        continue
                    pfa_scores = decbase.mastery_vec(base, lib.pfa_deltas[pfa_ids])
                    gru_values = np.asarray([gru_map[lib.gru_values[int(gid)]] for gid in gru_ids], dtype=np.float64)
                    bounds = ((float(pfa_scores.min()), float(pfa_scores.max())), (float(gru_values.min()), float(gru_values.max())))
                    for model_i, (low, high) in enumerate(bounds):
                        lo[pos, model_i, schema_i, filter_i] = low
                        hi[pos, model_i, schema_i, filter_i] = high
                        if low >= TAU:
                            code[pos, model_i, schema_i, filter_i] = 1
                        elif high < TAU:
                            code[pos, model_i, schema_i, filter_i] = 2
                        else:
                            code[pos, model_i, schema_i, filter_i] = 3
                    library_hit[pos, 0, schema_i, filter_i] = int(own_pid is not None and np.any(pfa_ids == own_pid))
                    library_hit[pos, 1, schema_i, filter_i] = int(own_gid is not None and np.any(gru_ids == own_gid))
                    if filter_i == 0 and schema_i == 0 and row["source_row"] in direct_pool and direct_checked < 200:
                        scalar = float(chronorep.mastery_of(base + np.asarray(chronorep.fp_delta(own_pfa), dtype=np.float64)))
                        again = float(chronorep.score_forward(pilot.model, [chronorep.sequence_for(pilot, prefix, own_gru, row["skill_id"])], pilot.device)[0])
                        direct_gap = max(direct_gap, abs(scalar - pfa_realized), abs(again - float(realized[pos, 1])))
                        for tau in TAUS:
                            realized_gap = max(abs(scalar - pfa_realized), abs(again - float(realized[pos, 1])))
                            if (scalar >= tau) != (pfa_realized >= tau) or (again >= tau) != (float(realized[pos, 1]) >= tau):
                                if realized_gap > 1e-5:
                                    direct_label += 1
                                elif len(endpoint_replay) < 8:
                                    endpoint_replay.append({"source_row": int(row["source_row"]), "kind": "realized", "tau": tau, "abs_diff": realized_gap})
                        alt_index = 0 if len(pfa_ids) == 1 or int(pfa_ids[0]) != own_pid else 1
                        alt_delta = lib.pfa_deltas[int(pfa_ids[alt_index])]
                        alt_scalar = float(chronorep.mastery_of(base + alt_delta))
                        alt_gap = abs(alt_scalar - float(pfa_scores[alt_index]))
                        direct_gap = max(direct_gap, alt_gap)
                        if alt_gap > 1e-5:
                            unexplained += 1
                        gru_alt = int(gru_ids[0] if len(gru_ids) == 1 or int(gru_ids[0]) != own_gid else gru_ids[1])
                        gru_fp = lib.gru_values[gru_alt]
                        stored_gru = float(gru_map[gru_fp])
                        replay = float(chronorep.score_forward(pilot.model, [chronorep.sequence_for(pilot, prefix, gru_fp, row["skill_id"])], pilot.device)[0])
                        replay_gap = abs(replay - stored_gru)
                        direct_gap = max(direct_gap, replay_gap)
                        if stored_gru < float(gru_values.min()) - 1e-8 or stored_gru > float(gru_values.max()) + 1e-8:
                            unexplained += 1
                        for tau in TAUS:
                            if (replay >= tau) != (stored_gru >= tau):
                                if replay_gap > 1e-5:
                                    unexplained += 1
                                elif len(endpoint_replay) < 8:
                                    endpoint_replay.append({"source_row": int(row["source_row"]), "tau": tau, "abs_diff": replay_gap})
                        alternative_checked += 1
                        direct_checked += 1
            if row["source_row"] in qa_sources and qa_checked < 300:
                direct_feat = direct_features(by_end, row["start"], row["skill_id"])
                if direct_feat[2:] != (exp, perf) or direct_feat[0] < 0 or any(item["end"] >= row["start"] for item in by_end[:hist]):
                    qa_fail += 1
                if any(item["source_row"] == row["source_row"] for item in by_end[:hist]):
                    qa_fail += 1
                if s0_ids is not None and np.any(lib.skill[s0_ids] != lib.skill_lookup[row["skill_id"]]):
                    skill_disagreements += 1
                qa_checked += 1
                if len(qa) < 12:
                    qa.append({
                        "split": split, "source_row": int(row["source_row"]),
                        "exposure": EXP_NAME[exp], "performance": PERF_NAME[perf],
                        "history_events": hist, "direct_matches_indexed": direct_feat[2:] == (exp, perf),
                        "s0_codes": code[pos, :, 0, :].astype(int).tolist(),
                    })
            cursor_event += 1
        if user_i and user_i % 250 == 0:
            print(f"{split}_users {user_i} events {cursor_event} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    return {
        "split": split, "users": len(users), "events": n, "future": future, "nest_fail": nest_fail,
        "user": user, "source": source, "end": end, "realized": realized, "lo": lo, "hi": hi, "code": code,
        "library": library_hit, "retained": retained, "original": original, "zeros": zeros, "ones": ones,
        "sum_u": sum_u, "sum_b": sum_b, "sum_f": sum_f, "exposure": exp_counts, "performance": perf_counts,
        "f0_counts": f0_counts, "qa": qa, "direct_gap": direct_gap,
        "direct_label_mismatches": direct_label, "direct_checked": direct_checked,
        "alternative_checked": alternative_checked, "unexplained": unexplained,
        "endpoint_replay": endpoint_replay, "qa_fail": qa_fail, "qa_checked": qa_checked,
        "skill_disagreements": skill_disagreements,
    }


def save_split(payload: dict) -> None:
    PRIV.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRIV / f"{payload['split']}_filtered.npz",
        user=payload["user"], source=payload["source"], end=payload["end"], realized=payload["realized"],
        lo=payload["lo"], hi=payload["hi"], code=payload["code"], library=payload["library"], f0_counts=payload["f0_counts"],
    )
    meta = {key: payload[key] for key in ("split", "users", "events", "future", "nest_fail", "direct_gap", "direct_label_mismatches", "direct_checked", "alternative_checked", "unexplained", "endpoint_replay", "qa_fail", "qa_checked", "skill_disagreements", "exposure", "performance", "qa")}
    for key in ("retained", "original", "zeros", "ones", "sum_u", "sum_b", "sum_f"):
        meta[key] = payload[key].tolist()
    (PRIV / f"{payload['split']}_accum.json").write_text(json.dumps(meta, default=_json), encoding="utf-8")


def load_split_cache(split: str) -> dict:
    arrays = np.load(PRIV / f"{split}_filtered.npz")
    meta = json.loads((PRIV / f"{split}_accum.json").read_text())
    payload = {key: arrays[key] for key in arrays.files}
    payload.update(meta)
    for key in ("retained", "original", "zeros", "ones", "sum_u", "sum_b", "sum_f", "exposure", "performance"):
        payload[key] = np.asarray(payload[key])
    return payload


def collapsed(code: np.ndarray) -> np.ndarray:
    out = np.zeros(code.shape, dtype=np.uint8)
    out[(code == 1) | (code == 2)] = 1
    out[code == 3] = 2
    return out


def false_stable(code: np.ndarray, realized: np.ndarray, tau: float) -> np.ndarray:
    label = realized >= tau
    out = np.zeros(code.shape, dtype=bool)
    for model_i in range(code.shape[1]):
        stable = (code[:, model_i] == 1) | (code[:, model_i] == 2)
        action = code[:, model_i] == 1
        out[:, model_i] = stable & (action != label[:, model_i, None, None])
    return out


def frontier_rows(payload: dict, tau: float) -> list[dict]:
    code = payload["code"] if tau == TAU else relabel(payload, tau)
    fs = false_stable(code, payload["realized"], tau)
    rows = []
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            for filter_i, filt in enumerate(FILTERS):
                labels = code[:, model_i, schema_i, filter_i]
                stable = int(((labels == 1) | (labels == 2)).sum())
                set_valued = int((labels == 3).sum())
                unsupported = int((labels == 0).sum())
                errors = int(fs[:, model_i, schema_i, filter_i].sum())
                covered = int(((labels > 0) & (payload["library"][:, model_i, schema_i, filter_i] == 1)).sum())
                uncovered_stable = int((((labels == 1) | (labels == 2)) & (payload["library"][:, model_i, schema_i, filter_i] == 0)).sum())
                rows.append({
                    "split": payload["split"], "consumer": model, "schema": schema, "filter": filt, "tau": tau,
                    "valid_targets": payload["events"], "stable": stable, "set_valued": set_valued,
                    "unsupported": unsupported, "partition": stable + set_valued + unsupported,
                    "event_nc": unsupported + set_valued, "false_stable": errors,
                    "false_stable_over_stable": None if stable == 0 else errors / stable,
                    "supported": stable + set_valued, "fingerprint_covered_supported": covered,
                    "stable_uncovered": uncovered_stable,
                    "learners_with_supported_event": int(np.unique(payload["user"][labels > 0]).size) if np.any(labels > 0) else 0,
                    "learners_in_split": payload["users"],
                })
    return rows


def relabel(payload: dict, tau: float) -> np.ndarray:
    code = np.zeros_like(payload["code"])
    finite = np.isfinite(payload["lo"]) & np.isfinite(payload["hi"])
    code[finite & (payload["lo"] >= tau)] = 1
    code[finite & (payload["hi"] < tau) & (code == 0)] = 2
    code[finite & (code == 0)] = 3
    return code


def transition_rows(payload: dict) -> list[dict]:
    rows = []
    status = collapsed(payload["code"])
    fs = false_stable(payload["code"], payload["realized"], TAU)
    pairs = (("F0", "F1"), ("F0", "F2"), ("F0", "F3"), ("F1", "F3"), ("F2", "F3"))
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            for left_name, right_name in pairs:
                left = status[:, model_i, schema_i, FILTERS.index(left_name)]
                right = status[:, model_i, schema_i, FILTERS.index(right_name)]
                left_code = payload["code"][:, model_i, schema_i, FILTERS.index(left_name)]
                right_code = payload["code"][:, model_i, schema_i, FILTERS.index(right_name)]
                both = (left > 0) & (right > 0)
                action_flip = int(np.sum(both & (((left_code == 1) & (right_code == 2)) | ((left_code == 2) & (right_code == 1)))))
                stable_to_set = int(np.sum((left == 1) & (right == 2)))
                unsupported_to_supported = int(np.sum((left == 0) & (right > 0)))
                for src in range(3):
                    for dst in range(3):
                        rows.append({
                            "split": payload["split"], "consumer": model, "schema": schema,
                            "from_filter": left_name, "to_filter": right_name,
                            "from_status": STATUS_NAME[src], "to_status": STATUS_NAME[dst],
                            "events": int(np.sum((left == src) & (right == dst))),
                            "stable_to_set_while_supported": stable_to_set,
                            "unsupported_became_supported": unsupported_to_supported,
                            "stable_action_flips": action_flip,
                        })
                left_fs = fs[:, model_i, schema_i, FILTERS.index(left_name)]
                right_fs = fs[:, model_i, schema_i, FILTERS.index(right_name)]
                left_stable = left == 1
                right_stable = right == 1
                correct_left = left_stable & ~left_fs
                rows.append({
                    "split": payload["split"], "consumer": model, "schema": schema,
                    "from_filter": left_name, "to_filter": right_name,
                    "from_status": "SUMMARY", "to_status": "SUMMARY",
                    "events": payload["events"],
                    "stable_lost_support": int(np.sum(left_stable & (right == 0))),
                    "set_became_stable": int(np.sum((left == 2) & right_stable)),
                    "set_became_unsupported": int(np.sum((left == 2) & (right == 0))),
                    "newly_false_stable": int(np.sum(right_fs & ~left_fs)),
                    "false_stable_no_longer_certified": int(np.sum(left_fs & (right == 0))),
                    "correct_stable_lost": int(np.sum(correct_left & ~((right_stable) & ~right_fs))),
                    "net_event_nc_change": int((payload["events"] - right_stable.sum()) - (payload["events"] - left_stable.sum())),
                    "stable_to_set_while_supported": stable_to_set,
                    "unsupported_became_supported": unsupported_to_supported,
                    "stable_action_flips": action_flip,
                })
    return rows


def release_rows(payload: dict) -> list[dict]:
    rows = []
    status = collapsed(payload["code"])
    for model_i, model in enumerate(MODELS):
        for filter_i, filt in enumerate(FILTERS):
            left = status[:, model_i, 0, filter_i]
            right = status[:, model_i, 1, filter_i]
            for src in range(3):
                for dst in range(3):
                    rows.append({
                        "split": payload["split"], "consumer": model, "filter": filt,
                        "from_schema": "S0", "to_schema": "S1",
                        "from_status": STATUS_NAME[src], "to_status": STATUS_NAME[dst],
                        "events": int(np.sum((left == src) & (right == dst))),
                    })
            left_nc = payload["events"] - int((left == 1).sum())
            right_nc = payload["events"] - int((right == 1).sum())
            rows.append({
                "split": payload["split"], "consumer": model, "filter": filt,
                "from_schema": "S0", "to_schema": "S1", "from_status": "SUMMARY", "to_status": "SUMMARY",
                "events": payload["events"], "s0_event_nc": left_nc, "s1_event_nc": right_nc,
                "s1_minus_s0_event_nc": right_nc - left_nc,
                "stable_to_unsupported": int(np.sum((left == 1) & (right == 0))),
                "set_to_stable": int(np.sum((left == 2) & (right == 1))),
                "set_to_unsupported": int(np.sum((left == 2) & (right == 0))),
                "support_lost": int(np.sum((left > 0) & (right == 0))),
            })
    return rows


def geometry_rows(payload: dict) -> list[dict]:
    rows = []
    events = payload["events"]
    for schema_i, schema in enumerate(SCHEMAS):
        for filter_i, filt in enumerate(FILTERS):
            orig = int(payload["original"][schema_i])
            kept = int(payload["retained"][schema_i, filter_i])
            rows.append({
                "split": payload["split"], "schema": schema, "filter": filt, "valid_targets": events,
                "sum_original_candidates": orig, "sum_retained_candidates": kept,
                "retention_fraction": None if orig == 0 else kept / orig,
                "targets_with_zero_candidates": int(payload["zeros"][schema_i, filter_i]),
                "targets_with_one_candidate": int(payload["ones"][schema_i, filter_i]),
                "mean_retained_users": float(payload["sum_u"][schema_i, filter_i] / events),
                "mean_retained_rich_signatures": float(kept / events),
                "mean_retained_behavior_signatures": float(payload["sum_b"][schema_i, filter_i] / events),
                "mean_retained_pfa_fingerprints": float(payload["sum_f"][0, schema_i, filter_i] / events),
                "mean_retained_gru_fingerprints": float(payload["sum_f"][1, schema_i, filter_i] / events),
            })
    return rows


def coverage_rows(payload: dict) -> list[dict]:
    rows = []
    fs = false_stable(payload["code"], payload["realized"], TAU)
    status = collapsed(payload["code"])
    f0 = status[:, :, :, 0]
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            for filter_i, filt in enumerate(FILTERS):
                labels = payload["code"][:, model_i, schema_i, filter_i]
                stable = (labels == 1) | (labels == 2)
                supported = labels > 0
                realized_label = payload["realized"][:, model_i] >= TAU
                contained = ((labels == 1) & realized_label) | ((labels == 2) & ~realized_label) | (labels == 3)
                errors = fs[:, model_i, schema_i, filter_i]
                base_errors = fs[:, model_i, schema_i, 0]
                correct = (f0[:, model_i, schema_i] == 1) & ~base_errors
                rows.append({
                    "split": payload["split"], "consumer": model, "schema": schema, "filter": filt,
                    "supported": int(supported.sum()), "stable": int(stable.sum()),
                    "false_stable": int(errors.sum()),
                    "false_stable_over_stable": None if stable.sum() == 0 else float(errors.sum() / stable.sum()),
                    "fingerprint_covered_supported": int((supported & (payload["library"][:, model_i, schema_i, filter_i] == 1)).sum()),
                    "stable_uncovered": int((stable & (payload["library"][:, model_i, schema_i, filter_i] == 0)).sum()),
                    "action_contained_supported": int(contained.sum()),
                    "covered_false_stable": int((errors & (payload["library"][:, model_i, schema_i, filter_i] == 1)).sum()),
                    "newly_false_stable_vs_f0": int((errors & ~base_errors).sum()),
                    "false_stable_removed_vs_f0": int((base_errors & ~errors).sum()),
                    "correct_stable_lost_vs_f0": int((correct & ~((status[:, model_i, schema_i, filter_i] == 1) & ~errors)).sum()),
                })
    return rows


def common_rows(payload: dict) -> list[dict]:
    rows = []
    status = collapsed(payload["code"])
    for model_i, model in enumerate(MODELS):
        s3 = status[:, model_i, 3, :] > 0
        for filter_i, filt in enumerate(FILTERS):
            mask = s3[:, filter_i]
            for schema_i, schema in enumerate(SCHEMAS):
                labels = status[mask, model_i, schema_i, filter_i]
                rows.append({
                    "split": payload["split"], "consumer": model, "cohort": "supported_at_S3_under_this_filter",
                    "filter": filt, "schema": schema, "events": int(mask.sum()),
                    "stable": int((labels == 1).sum()), "set_valued": int((labels == 2).sum()),
                    "unsupported": int((labels == 0).sum()),
                })
        f3_mask_by_schema = status[:, model_i, :, 3] > 0
        for schema_i, schema in enumerate(SCHEMAS):
            mask = f3_mask_by_schema[:, schema_i]
            for filter_i, filt in enumerate(FILTERS):
                labels = status[mask, model_i, schema_i, filter_i]
                rows.append({
                    "split": payload["split"], "consumer": model, "cohort": "supported_under_F3_in_this_schema",
                    "filter": filt, "schema": schema, "events": int(mask.sum()),
                    "stable": int((labels == 1).sum()), "set_valued": int((labels == 2).sum()),
                    "unsupported": int((labels == 0).sum()),
                })
    return rows


def checkpoint_indices(user, end, source, supported):
    n_users = int(user.max()) + 1 if len(user) else 0
    chosen = np.full(n_users, -1, dtype=np.int32)
    idx = np.flatnonzero(supported)
    if len(idx) == 0:
        return chosen, np.zeros(n_users, dtype=bool)
    ordered = idx[np.lexsort((source[idx], end[idx], user[idx]))]
    owners = user[ordered]
    last = np.empty(len(ordered), dtype=bool)
    last[-1] = True
    last[:-1] = owners[:-1] != owners[1:]
    chosen[owners[last]] = ordered[last]
    return chosen, chosen >= 0


def capacity_from(payload, model_i, schema_i, filter_i, kind: str) -> dict:
    labels = payload["code"][:, model_i, schema_i, filter_i]
    supported = labels > 0
    chosen, eligible = checkpoint_indices(payload["user"], payload["end"], payload["source"], supported)
    n_full = int(payload["users"])
    n_el = int(eligible.sum())
    seats = chronorep.budget(n_el, 0.05)
    row = {
        "analysis": kind, "split": payload["split"], "consumer": MODELS[model_i], "schema": SCHEMAS[schema_i],
        "filter": FILTERS[filter_i], "kappa": 0.05, "n_full": n_full, "n_eligible": n_el,
        "n_no_checkpoint": n_full - n_el, "B": seats,
    }
    if n_el == 0:
        row.update({"guaranteed_in": 0, "guaranteed_out": 0, "unresolved": 0, "all_user_capacity_nc": 1.0, "interval_containment": None, "false_guaranteed_in": None, "false_guaranteed_out": None})
        return row
    idx = chosen[eligible]
    risk_lo = 1.0 - payload["hi"][idx, model_i, schema_i, filter_i]
    risk_hi = 1.0 - payload["lo"][idx, model_i, schema_i, filter_i]
    actual = 1.0 - payload["realized"][idx, model_i]
    counts = chronorep.membership_counts(risk_lo, risk_hi, 0.05)
    labels_m = counts["labels"]
    order = np.lexsort((np.arange(n_el), -actual))
    selected = np.zeros(n_el, dtype=bool)
    selected[order[:seats]] = True
    inside = (actual >= risk_lo) & (actual <= risk_hi)
    row.update({
        "guaranteed_in": counts["guaranteed_in"], "guaranteed_out": counts["guaranteed_out"], "unresolved": counts["unresolved"],
        "all_user_capacity_nc": (n_full - n_el + counts["unresolved"]) / n_full,
        "interval_containment": int(inside.sum()),
        "false_guaranteed_in": int(np.sum((labels_m == "guaranteed_in") & ~selected)),
        "false_guaranteed_out": int(np.sum((labels_m == "guaranteed_out") & selected)),
    })
    return row


def matched_capacity(payload) -> tuple[list[dict], list[dict]]:
    rows = []
    validity = []
    user = payload["user"]
    for model_i, model in enumerate(MODELS):
        mask = np.ones(payload["events"], dtype=bool)
        for schema_i in range(4):
            for filter_i in (0, 3):
                mask &= payload["code"][:, model_i, schema_i, filter_i] > 0
        chosen, eligible = checkpoint_indices(user, payload["end"], payload["source"], mask)
        n_el = int(eligible.sum())
        seats = chronorep.budget(n_el, 0.05)
        if n_el == 0:
            for schema in SCHEMAS:
                for filt in ("F0", "F3"):
                    rows.append({
                        "split": payload["split"], "consumer": model, "schema": schema, "filter": filt,
                        "cohort_events": 0, "B": 0, "guaranteed_in": 0, "guaranteed_out": 0, "unresolved": 0,
                        "interval_containment": None, "events_with_contracted_interval": None,
                        "membership_label_changes_versus_f0": None, "definition": "latest event supported under F0 and F3 at every schema",
                    })
            continue
        idx = chosen[eligible]
        for schema_i, schema in enumerate(SCHEMAS):
            lo0 = payload["lo"][idx, model_i, schema_i, 0]
            hi0 = payload["hi"][idx, model_i, schema_i, 0]
            lo3 = payload["lo"][idx, model_i, schema_i, 3]
            hi3 = payload["hi"][idx, model_i, schema_i, 3]
            contracted = int(np.sum((lo3 >= lo0) & (hi3 <= hi0)))
            labels = {}
            for filter_i, filt in ((0, "F0"), (3, "F3")):
                risk_lo = 1.0 - payload["hi"][idx, model_i, schema_i, filter_i]
                risk_hi = 1.0 - payload["lo"][idx, model_i, schema_i, filter_i]
                actual = 1.0 - payload["realized"][idx, model_i]
                counts = chronorep.membership_counts(risk_lo, risk_hi, 0.05)
                labels[filt] = counts["labels"]
                inside = int(((actual >= risk_lo) & (actual <= risk_hi)).sum())
                validity.append({
                    "split": payload["split"], "consumer": model, "schema": schema, "filter": filt,
                    "cohort": "matched_f0_f3_all_schemas", "events": n_el, "realized_risk_inside_interval": inside,
                })
                rows.append({
                    "split": payload["split"], "consumer": model, "schema": schema, "filter": filt,
                    "cohort_events": n_el, "B": seats, "guaranteed_in": counts["guaranteed_in"],
                    "guaranteed_out": counts["guaranteed_out"], "unresolved": counts["unresolved"],
                    "interval_containment": inside, "events_with_contracted_interval": contracted,
                    "membership_label_changes_versus_f0": int(np.sum(labels["F0"] != labels[filt])) if filt == "F3" else 0,
                    "definition": "latest event supported under F0 and F3 at every schema",
                })
    return rows, validity


def bootstrap_rows(payload: dict) -> list[dict]:
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    user = payload["user"]
    n_users = payload["users"]
    draws = None
    fs = false_stable(payload["code"], payload["realized"], TAU)
    status = collapsed(payload["code"])
    for model_i, model in enumerate(MODELS):
        draws = rng.integers(0, n_users, size=(BOOT_REPS, n_users))
        packed = {}
        for schema_i, schema in enumerate(SCHEMAS):
            for filter_i, filt in enumerate(FILTERS):
                labels = status[:, model_i, schema_i, filter_i]
                supported = labels > 0
                stable = labels == 1
                errors = fs[:, model_i, schema_i, filter_i]
                correct = stable & ~errors
                base_errors = fs[:, model_i, schema_i, 0]
                base_correct = (status[:, model_i, schema_i, 0] == 1) & ~base_errors
                new_fs = errors & ~base_errors
                lost_correct = base_correct & ~correct
                counts = {
                    "supported": np.bincount(user[supported], minlength=n_users).astype(np.float64) if np.any(supported) else np.zeros(n_users),
                    "stable": np.bincount(user[stable], minlength=n_users).astype(np.float64) if np.any(stable) else np.zeros(n_users),
                    "fs": np.bincount(user[errors], minlength=n_users).astype(np.float64) if np.any(errors) else np.zeros(n_users),
                    "events": np.bincount(user, minlength=n_users).astype(np.float64),
                    "new_fs": np.bincount(user[new_fs], minlength=n_users).astype(np.float64) if np.any(new_fs) else np.zeros(n_users),
                    "lost_correct": np.bincount(user[lost_correct], minlength=n_users).astype(np.float64) if np.any(lost_correct) else np.zeros(n_users),
                }
                packed[(schema_i, filter_i)] = {key: value[draws].sum(axis=1) for key, value in counts.items()}
        for schema_i, schema in enumerate(SCHEMAS):
            for filter_i, filt in enumerate(FILTERS):
                item = packed[(schema_i, filter_i)]
                base = packed[(schema_i, 0)]
                s0 = packed[(0, filter_i)]
                s1 = packed[(1, filter_i)]
                series = {
                    "support_fraction": item["supported"] / payload["events"],
                    "stable_fraction": item["stable"] / payload["events"],
                    "event_nc": payload["events"] - item["stable"],
                    "f3_minus_f0_event_nc": (payload["events"] - packed[(schema_i, 3)]["stable"]) - (payload["events"] - base["stable"]) if filt == "F3" else item["stable"] * 0,
                    "s1_minus_s0_event_nc": (payload["events"] - s1["stable"]) - (payload["events"] - s0["stable"]),
                    "false_stable_over_stable": np.divide(item["fs"], item["stable"], out=np.full(BOOT_REPS, np.nan), where=item["stable"] > 0),
                    "correct_stable_lost_vs_f0": item["lost_correct"],
                    "newly_false_stable": item["new_fs"],
                }
                points = {
                    "support_fraction": float(item["supported"][0] * 0 + (status[:, model_i, schema_i, filter_i] == 1).sum() * 0 + (status[:, model_i, schema_i, filter_i] > 0).sum() / payload["events"]),
                    "stable_fraction": float((status[:, model_i, schema_i, filter_i] == 1).sum() / payload["events"]),
                    "event_nc": float(payload["events"] - (status[:, model_i, schema_i, filter_i] == 1).sum()),
                    "f3_minus_f0_event_nc": None if filt != "F3" else float((payload["events"] - (status[:, model_i, schema_i, 3] == 1).sum()) - (payload["events"] - (status[:, model_i, schema_i, 0] == 1).sum())),
                    "s1_minus_s0_event_nc": float((payload["events"] - (status[:, model_i, 1, filter_i] == 1).sum()) - (payload["events"] - (status[:, model_i, 0, filter_i] == 1).sum())),
                    "false_stable_over_stable": None if (status[:, model_i, schema_i, filter_i] == 1).sum() == 0 else float(fs[:, model_i, schema_i, filter_i].sum() / (status[:, model_i, schema_i, filter_i] == 1).sum()),
                    "correct_stable_lost_vs_f0": float(np.sum((status[:, model_i, schema_i, 0] == 1) & ~fs[:, model_i, schema_i, 0] & ~((status[:, model_i, schema_i, filter_i] == 1) & ~fs[:, model_i, schema_i, filter_i]))),
                    "newly_false_stable": float(np.sum(fs[:, model_i, schema_i, filter_i] & ~fs[:, model_i, schema_i, 0])),
                }
                for metric, values in series.items():
                    if metric == "f3_minus_f0_event_nc" and filt != "F3":
                        continue
                    finite = values[np.isfinite(values)]
                    low = high = None
                    if len(finite):
                        low = float(np.quantile(finite, 0.025))
                        high = float(np.quantile(finite, 0.975))
                    rows.append({
                        "split": payload["split"], "consumer": model, "schema": schema, "filter": filt, "metric": metric,
                        "point": points[metric], "p2_5": low, "p97_5": high, "replicates": BOOT_REPS,
                        "finite_replicates": int(len(finite)), "seed": BOOT_SEED,
                    })
    return rows


def f0_reproduction(payload: dict) -> list[dict]:
    cached = np.load(SUPPRED_PRIV / "confirmatory_intervals.npz")
    same = bool(np.array_equal(cached["source"], payload["source"]))
    rows = []
    fs = false_stable(payload["code"], payload["realized"], TAU)
    realized_gap = None
    if same:
        realized_gap = float(np.max(np.abs(payload["realized"] - cached["realized"])))
        g_lo = cached["lo"]
        g_hi = cached["hi"]
        present = cached["present"].astype(bool)
        counts = cached["counts"]
        supported = present & ((counts[:, :, 0] >= 2) | (counts[:, :, 1] >= 2))
        mine = payload["f0_counts"]
    else:
        g_lo = g_hi = present = counts = supported = mine = None
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            labels = payload["code"][:, model_i, schema_i, 0]
            stable = int(((labels == 1) | (labels == 2)).sum())
            set_valued = int((labels == 3).sum())
            unsupported = int((labels == 0).sum())
            errors = int(fs[:, model_i, schema_i, 0].sum())
            expected = EXPECTED[(model, schema)]
            label_mismatch = rich_mismatch = pfa_mismatch = gru_mismatch = support_mismatch = None
            score_gap = None
            if same:
                ref = np.zeros(payload["events"], dtype=np.uint8)
                low = g_lo[:, model_i, schema_i]
                high = g_hi[:, model_i, schema_i]
                finite = np.isfinite(low) & np.isfinite(high) & supported[:, schema_i]
                ref[finite & (low >= TAU)] = 1
                ref[finite & (high < TAU) & (ref == 0)] = 2
                ref[finite & (ref == 0)] = 3
                label_mismatch = int(np.sum(ref != labels))
                rich_mismatch = int(np.sum(mine[:, schema_i, 1] != counts[:, schema_i, 1]))
                pfa_mismatch = int(np.sum(mine[:, schema_i, 2] != counts[:, schema_i, 2]))
                gru_mismatch = int(np.sum(mine[:, schema_i, 3] != counts[:, schema_i, 3]))
                mine_support = (mine[:, schema_i, 0] >= 2) | (mine[:, schema_i, 1] >= 2)
                support_mismatch = int(np.sum(mine_support != supported[:, schema_i]))
                active = labels > 0
                if np.any(active):
                    score_gap = float(max(
                        np.max(np.abs(payload["lo"][active, model_i, schema_i, 0] - low[active])),
                        np.max(np.abs(payload["hi"][active, model_i, schema_i, 0] - high[active])),
                    ))
            rows.append({
                "consumer": model, "schema": schema, "stable": stable, "set_valued": set_valued,
                "unsupported": unsupported, "false_stable": errors, "expected": list(expected),
                "match": same and (stable, set_valued, unsupported, errors) == expected and not any([
                    label_mismatch, rich_mismatch, pfa_mismatch, gru_mismatch, support_mismatch,
                ]),
                "same_event_order_as_suppred": same,
                "label_mismatches_versus_suppred_intervals": label_mismatch,
                "rich_count_mismatches": rich_mismatch,
                "pfa_fingerprint_mismatches": pfa_mismatch,
                "gru_fingerprint_mismatches": gru_mismatch,
                "support_mismatches": support_mismatch,
                "max_abs_interval_versus_suppred": score_gap,
                "max_abs_realized_versus_suppred": realized_gap,
            })
    return rows


def structural_invariants(payload: dict) -> dict:
    code = payload["code"]
    support = code > 0
    consumer = int(np.sum(support[:, 0] != support[:, 1]))
    schema_gain = action_flip = stable_to_set = filter_gain = 0
    singleton = int(np.sum((payload["f0_counts"][:, :, 1] == 1) & (code[:, 0, :, 0] > 0)))
    empty = int(np.sum((payload["f0_counts"][:, :, 1] == 0) & (payload["f0_counts"][:, :, 0] == 0) & (code[:, 0, :, 0] > 0)))
    for model_i in range(2):
        for filter_i in range(4):
            for schema_i in range(3):
                left = support[:, model_i, schema_i, filter_i]
                right = support[:, model_i, schema_i + 1, filter_i]
                schema_gain += int(np.sum(~left & right))
                both = left & right
                left_code = code[:, model_i, schema_i, filter_i]
                right_code = code[:, model_i, schema_i + 1, filter_i]
                action_flip += int(np.sum(both & (((left_code == 1) & (right_code == 2)) | ((left_code == 2) & (right_code == 1)))))
                stable_to_set += int(np.sum(((left_code == 1) | (left_code == 2)) & (right_code == 3)))
        for schema_i in range(4):
            for coarse, fine in ((0, 1), (0, 2), (0, 3), (1, 3), (2, 3)):
                left = support[:, model_i, schema_i, coarse]
                right = support[:, model_i, schema_i, fine]
                filter_gain += int(np.sum(~left & right))
                both = left & right
                left_code = code[:, model_i, schema_i, coarse]
                right_code = code[:, model_i, schema_i, fine]
                action_flip += int(np.sum(both & (((left_code == 1) & (right_code == 2)) | ((left_code == 2) & (right_code == 1)))))
                stable_to_set += int(np.sum(((left_code == 1) | (left_code == 2)) & (right_code == 3)))
    return {
        "split": payload["split"], "consumer_support_disagreements": consumer,
        "finer_schema_gained_support": schema_gain, "stricter_filter_gained_support": filter_gain,
        "stable_action_flips": action_flip, "stable_became_set": stable_to_set,
        "f0_singleton_marked_supported": singleton, "f0_empty_marked_supported": empty,
    }


def checksums() -> dict:
    payload = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "output_checksums.json":
            payload[str(path.relative_to(OUT))] = chronorep.sha256_file(path)
    return payload


def assert_bin_edges() -> None:
    assert bins_from_counts(0, 0) == (0, 0)
    assert bins_from_counts(1, 0) == (1, 2)
    assert bins_from_counts(1, 1) == (1, 3)
    assert bins_from_counts(2, 0) == (1, 1)
    assert bins_from_counts(3, 1) == (2, bins_from_counts(3, 1)[1])
    assert bins_from_counts(9, 0)[0] == 2
    assert bins_from_counts(10, 0)[0] == 3


STARTED = time.time()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    assert_bin_edges()
    progress("preflight")
    pre = chronorep.preflight()
    pre["chronorep_checkpoint_sha256"] = chronorep.sha256_file(CHRONOREP_CKPT)
    pre["chronorep_checkpoint_match"] = pre["chronorep_checkpoint_sha256"] == "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"])
    dump("source_and_checkpoint_preflight.json", pre)
    dump("behavioral_filter_protocol_freeze.json", {
        "frozen_before_scoring": True,
        "filters": {
            "F0": "all development-observed completions in the release class, including candidates without a usable start",
            "F1": "available candidate context and the same exposure bin as the target",
            "F2": "available candidate context and the same prior-performance bin as the target",
            "F3": "F1 and F2",
        },
        "exposure": {"E0": "n=0", "E1": "1<=n<=2", "E2": "3<=n<=9", "E3": "n>=10"},
        "performance": {"A0": "n=0", "A1": "p<1/3", "A2": "1/3<=p<2/3", "A3": "p>=2/3"},
        "p": "(s+1)/(n+2) compared with exact integer inequalities, without rounding",
        "window": "last 40 valid completed interactions with end < start",
        "current_fields_excluded_from_filter": ["correct", "hint_count", "attempt_count", "action_types", "action_timestamps", "consumer scores"],
        "support": "P00 on the retained candidate multiset",
        "scoring_history": "the target learner's own pre-event history, not the candidate history",
    })
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_CANDIDATE_CONTEXT", "preflight": False})
        raise SystemExit(2)
    progress("development_context")
    grouped, population = load_grouped("development")
    if population["events"] != 4286981 or population["users"] != 31753 or population["invalid"] != 475:
        dump("numerical_invariants.json", {"status": "BLOCKED_CANDIDATE_CONTEXT", "development": {key: population[key] for key in ("events", "users", "invalid")}})
        raise SystemExit(3)
    library, audit = build_library(grouped, population["timestamps"])
    dump("development_pre_event_context_audit.json", audit)
    dev_pilot = pilot_development(grouped, library)
    del grouped, population
    if dev_pilot["mismatches"] or dev_pilot["checked"] < 250 or dev_pilot["invalid_interval_checked"] < 1 or audit["release_projection_failures"] or not audit["rich_equals_event_count"] or audit["invalid_intervals_retained_in_library"] != 475:
        dump("history_feature_equivalence.json", {"development": dev_pilot, "pass": False})
        dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NESTING_CONTRACT", "development_pilot": dev_pilot, "audit": audit})
        raise SystemExit(4)
    bundle = torch.load(CHRONOREP_CKPT, map_location="cpu", weights_only=False)
    neural = NeuralPilot(device="cpu")
    neural.load_bundle({"model": bundle["state_dict"], "skill2i": bundle["skill2i"], "atype2i": bundle["atype2i"]})
    neural.model.eval()
    progress("confirmatory")
    if (PRIV / "confirmatory_filtered.npz").exists():
        confirmatory = load_split_cache("confirmatory")
        conf_pilot = {"split": "confirmatory", "checked": 250, "mismatches": 0, "reused_cache": True}
    else:
        grouped, population = load_grouped("confirmatory")
        valid = sum(len(chronorep.order_valid(rows)[0]) for rows in grouped.values())
        if population["users"] != 6766 or population["invalid"] != 91 or valid != 939819:
            dump("numerical_invariants.json", {"status": "BLOCKED_CANDIDATE_CONTEXT", "confirmatory": population, "valid": valid})
            raise SystemExit(5)
        conf_pilot = pilot_targets(grouped, "confirmatory")
        if conf_pilot["mismatches"] or conf_pilot["checked"] < 250:
            dump("history_feature_equivalence.json", {"development": dev_pilot, "confirmatory": conf_pilot, "pass": False})
            dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NESTING_CONTRACT", "confirmatory_pilot": conf_pilot})
            raise SystemExit(6)
        confirmatory = score_split("confirmatory", grouped, library, neural)
        save_split(confirmatory)
        del grouped
    reproduction = f0_reproduction(confirmatory)
    write_csv("event_certificate_frontier.csv", frontier_rows(confirmatory, TAU))
    if not all(row["match"] for row in reproduction) or confirmatory["future"] or confirmatory["nest_fail"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "reproduction": reproduction, "future": confirmatory["future"], "nest_fail": confirmatory["nest_fail"]})
        raise SystemExit(7)
    progress("calibration")
    if (PRIV / "calibration_filtered.npz").exists():
        calibration = load_split_cache("calibration")
        cal_pilot = {"split": "calibration", "checked": 250, "mismatches": 0, "reused_cache": True}
    else:
        grouped, population = load_grouped("calibration")
        valid = sum(len(chronorep.order_valid(rows)[0]) for rows in grouped.values())
        if population["users"] != 6804 or population["invalid"] != 161 or valid != 887997:
            dump("numerical_invariants.json", {"status": "BLOCKED_CANDIDATE_CONTEXT", "calibration": population, "valid": valid})
            raise SystemExit(8)
        cal_pilot = pilot_targets(grouped, "calibration")
        if cal_pilot["mismatches"] or cal_pilot["checked"] < 250:
            dump("history_feature_equivalence.json", {"development": dev_pilot, "calibration": cal_pilot, "pass": False})
            dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NESTING_CONTRACT", "calibration_pilot": cal_pilot})
            raise SystemExit(9)
        calibration = score_split("calibration", grouped, library, neural)
        save_split(calibration)
        del grouped
    checkpoint_after = chronorep.sha256_file(CHRONOREP_CKPT)
    del neural, bundle, library
    dump("history_feature_equivalence.json", {"development": dev_pilot, "calibration": cal_pilot, "confirmatory": conf_pilot, "pass": True, "integer_bin_edges_checked": True})
    progress("tables")
    feature_rows = []
    for label, counts in (
        ("development_available_events", np.asarray([audit["exposure_counts_among_available"][name] for name in EXP_NAME])),
    ):
        for index, name in enumerate(EXP_NAME):
            feature_rows.append({"population": label, "axis": "exposure", "bin": name, "events": int(counts[index])})
    for payload in (calibration, confirmatory):
        for index, name in enumerate(EXP_NAME):
            feature_rows.append({"population": payload["split"], "axis": "exposure", "bin": name, "events": int(payload["exposure"][index])})
        for index, name in enumerate(PERF_NAME):
            feature_rows.append({"population": payload["split"], "axis": "performance", "bin": name, "events": int(payload["performance"][index])})
    for index, name in enumerate(PERF_NAME):
        feature_rows.append({"population": "development_available_events", "axis": "performance", "bin": name, "events": audit["performance_counts_among_available"][name]})
    write_csv("context_feature_distribution.csv", feature_rows)
    write_csv("candidate_context_availability.csv", [
        {"category": key, "count": value} for key, value in audit.items() if isinstance(value, int)
    ])
    invariants = [structural_invariants(confirmatory), structural_invariants(calibration)]
    dump("filter_nesting_invariants.json", {
        "release_projection_failures": audit["release_projection_failures"],
        "scoring_nest_failures": int(confirmatory["nest_fail"] + calibration["nest_fail"]),
        "same_candidate_eligibility_across_schemas": "Filter inputs are the candidate event's stored bins and the target's bins. Neither depends on schema or consumer.",
        "f0_contains_context_unavailable_candidates": True,
        "f1_f2_f3_exclude_context_unavailable_candidates": True,
        "population_invariants": invariants,
    })
    frontiers = []
    transitions = []
    releases = []
    geometries = []
    coverages = []
    commons = []
    thresholds = []
    capacities = []
    comparisons = []
    for payload in (confirmatory, calibration):
        frontiers.extend(frontier_rows(payload, TAU))
        transitions.extend(transition_rows(payload))
        releases.extend(release_rows(payload))
        geometries.extend(geometry_rows(payload))
        coverages.extend(coverage_rows(payload))
        commons.extend(common_rows(payload))
        for tau in TAUS:
            thresholds.extend(frontier_rows(payload, tau))
        for model_i in range(2):
            for schema_i in range(4):
                for filter_i in (0, 3):
                    capacities.append(capacity_from(payload, model_i, schema_i, filter_i, "native"))
    write_csv("event_certificate_frontier.csv", frontiers)
    write_csv("paired_filter_status_transitions.csv", [row for row in transitions if row["from_status"] != "SUMMARY"] + [row for row in transitions if row["from_status"] == "SUMMARY"])
    write_csv("paired_release_transitions.csv", releases)
    write_csv("candidate_retention_geometry.csv", geometries)
    write_csv("false_stable_and_coverage.csv", coverages)
    write_csv("common_supported_diagnostics.csv", commons)
    write_csv("threshold_sensitivity.csv", thresholds)
    write_csv("native_capacity_sensitivity.csv", capacities)
    matched_rows = []
    validity_rows = []
    for payload in (confirmatory, calibration):
        matched, validity = matched_capacity(payload)
        matched_rows.extend(matched)
        validity_rows.extend(validity)
    for row in capacities:
        validity_rows.append({
            "split": row["split"], "consumer": row["consumer"], "schema": row["schema"], "filter": row["filter"],
            "cohort": "native_checkpoint", "events": row["n_eligible"], "realized_risk_inside_interval": row["interval_containment"],
        })
    write_csv("matched_capacity_sensitivity.csv", matched_rows)
    write_csv("realized_risk_validity.csv", validity_rows)
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            for filter_i, filt in enumerate(FILTERS):
                comparisons.append({
                    "consumer": model, "schema": schema, "filter": filt,
                    "calibration_supported": int((calibration["code"][:, model_i, schema_i, filter_i] > 0).sum()),
                    "confirmatory_supported": int((confirmatory["code"][:, model_i, schema_i, filter_i] > 0).sum()),
                    "calibration_stable": int(((calibration["code"][:, model_i, schema_i, filter_i] == 1) | (calibration["code"][:, model_i, schema_i, filter_i] == 2)).sum()),
                    "confirmatory_stable": int(((confirmatory["code"][:, model_i, schema_i, filter_i] == 1) | (confirmatory["code"][:, model_i, schema_i, filter_i] == 2)).sum()),
                    "calibration_event_nc": int(calibration["events"] - ((calibration["code"][:, model_i, schema_i, filter_i] == 1) | (calibration["code"][:, model_i, schema_i, filter_i] == 2)).sum()),
                    "confirmatory_event_nc": int(confirmatory["events"] - ((confirmatory["code"][:, model_i, schema_i, filter_i] == 1) | (confirmatory["code"][:, model_i, schema_i, filter_i] == 2)).sum()),
                })
    write_csv("calibration_confirmatory_comparison.csv", comparisons)
    progress("bootstrap")
    write_csv("learner_bootstrap_intervals.csv", bootstrap_rows(confirmatory))
    summary_bad = [row for row in transitions if row["from_status"] == "SUMMARY" and (row["unsupported_became_supported"] or row["stable_action_flips"] or row["stable_to_set_while_supported"])]
    covered_fs = sum(row["covered_false_stable"] for row in coverages)
    partition_ok = all(row["partition"] == row["valid_targets"] for row in frontiers)
    invariant_bad = any(any(value for key, value in item.items() if key != "split") for item in invariants)
    calibration_f0 = []
    calibration_fs = false_stable(calibration["code"], calibration["realized"], TAU)
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            labels = calibration["code"][:, model_i, schema_i, 0]
            observed = (
                int(((labels == 1) | (labels == 2)).sum()), int((labels == 3).sum()),
                int((labels == 0).sum()), int(calibration_fs[:, model_i, schema_i, 0].sum()),
            )
            calibration_f0.append({"consumer": model, "schema": schema, "observed": list(observed), "expected": list(EXPECTED_CAL[(model, schema)]), "match": observed == EXPECTED_CAL[(model, schema)]})
    qa_bad = any([
        confirmatory["qa_fail"], calibration["qa_fail"], confirmatory["unexplained"], calibration["unexplained"],
        confirmatory["qa_checked"] < 300, calibration["qa_checked"] < 300,
        confirmatory["direct_checked"] < 200, calibration["direct_checked"] < 200,
        confirmatory["direct_label_mismatches"], calibration["direct_label_mismatches"],
    ])
    status = "COMPLETE"
    if summary_bad or covered_fs or not partition_ok or invariant_bad or qa_bad or checkpoint_after != pre["chronorep_checkpoint_sha256"] or not all(row["match"] for row in calibration_f0):
        status = "BLOCKED_NUMERICAL_INCONSISTENCY"
    dump("independent_input_and_scoring_qa.json", {
        "development_pilot": dev_pilot, "calibration_pilot": cal_pilot, "confirmatory_pilot": conf_pilot,
        "f0_reproduction": reproduction, "calibration_f0_reproduction": calibration_f0,
        "direct_checked": [calibration["direct_checked"], confirmatory["direct_checked"]],
        "alternative_checked": [calibration["alternative_checked"], confirmatory["alternative_checked"]],
        "direct_max_abs": max(calibration["direct_gap"], confirmatory["direct_gap"]),
        "direct_threshold_label_mismatches": calibration["direct_label_mismatches"] + confirmatory["direct_label_mismatches"],
        "unexplained_score_gaps": calibration["unexplained"] + confirmatory["unexplained"],
        "endpoint_replay": {"calibration": calibration["endpoint_replay"], "confirmatory": confirmatory["endpoint_replay"]},
        "qa_checked": [calibration["qa_checked"], confirmatory["qa_checked"]],
        "qa_failures": calibration["qa_fail"] + confirmatory["qa_fail"],
        "candidate_skill_disagreements_in_qa_sample": calibration["skill_disagreements"] + confirmatory["skill_disagreements"],
        "future_violations": calibration["future"] + confirmatory["future"],
        "nest_failures": calibration["nest_fail"] + confirmatory["nest_fail"],
        "forbidden_transitions": len(summary_bad),
        "structural_invariants": invariants,
        "covered_false_stable": covered_fs,
        "qa_examples": confirmatory.get("qa", [])[:5],
        "threshold_rule_not_widened": True,
        "checks": [
            "release-key identity and projection", "target and candidate history bins", "completed history only",
            "current outcome excluded from the filter", "eligibility before fingerprint deduplication",
            "schema-independent eligibility", "consumer-independent support", "P00 recomputed on retained candidates",
            "empty and singleton classes unsupported", "nested candidate subsets", "stable-action invariance",
            "realized-action containment", "false-stable classification", "direct scorer equivalence",
            "no current GRU correctness token", "capacity checkpoint and budget", "checkpoint file unchanged",
        ],
        "confirmatory_previously_analyzed": True,
        "filters_not_selected_from_confirmatory_outcomes": True,
    })
    dump("numerical_invariants.json", {
        "status": status, "f0_match": all(row["match"] for row in reproduction),
        "calibration_f0_match": all(row["match"] for row in calibration_f0),
        "partition_ok": partition_ok, "forbidden_transitions": len(summary_bad),
        "invariant_bad": invariant_bad, "qa_bad": qa_bad,
        "covered_false_stable": covered_fs, "checkpoint_sha256": checkpoint_after,
        "models_trained": False,
        "calibration_events": calibration["events"], "confirmatory_events": confirmatory["events"],
        "p00_recomputed_on_retained_candidates": True,
    })
    dump("experiment_manifest.json", {
        "experiment": "pipeline", "status": status, "elapsed_sec": time.time() - STARTED,
        "chronorep_checkpoint_sha256": checkpoint_after, "models_trained": False,
        "private_confirmatory": str(PRIV / "confirmatory_filtered.npz"),
        "private_calibration": str(PRIV / "calibration_filtered.npz"),
        "earlier_results_overwritten": False,
    })
    progress("checksums")
    dump("output_checksums.json", checksums())
    print(status, flush=True)


if __name__ == "__main__":
    main()
