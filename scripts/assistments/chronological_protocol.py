#!/usr/bin/env python3
"""Chronology-valid interval contract and a small calibration pilot.

The historical GRU is not retrained, the confirmatory frontier is not scored,
and frozen data and checkpoints are left untouched.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import platform
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from r7_core import (  # noqa: E402
    MAX_EVAL_PREFIX_EVENTS,
    MAX_HIST_TOKENS,
    MAX_SEQ_ACTIONS,
    TAU,
    ActionTokenGRU,
    Event,
    NeuralPilot,
    TransparentActionPFA,
    project_down,
    release_key,
    rich_signature,
)

OUT = ROOT / "results/assistments/chronological_protocol"
STAGE = ROOT / "data/raw/assistments/assistments_2012.zip"
RECON = ROOT / "data/processed/assistments"
EXPECTED_ZIP = "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445"
EXPECTED = {
    "development": ("69b6b7b38738d293651b728326658c1ce517c87661d012bc03ee2ced228e9d70", 190476332, 31753, 4286981),
    "calibration": ("5e8202d6f89b501a1aaea662ba62a265de1e5c168a0bcf92e3762760b4c6fd3d", 43897418, 6804, 888158),
    "confirmatory": ("0193ddd7b26517f1b48591d09d86f5f3ce8b9930a1285006833fce4ea5096f99", 46167386, 6766, 939910),
}
EXPECTED_KEYS = {"S0": 259838, "S1": 511770, "S2": 523378, "S3": 605097}
NAT = np.iinfo(np.int64).min
NY = ZoneInfo("America/New_York")
CAL_PREFIX = "WSDM203-G7-CAL-PILOT-v1"
DEV_PREFIX = "WSDM203-G7-DEV-PILOT-v1"
EVENT_PREFIX = "WSDM203-G7-CAL-EVENT-v1"
GRU_CANDIDATE_BUDGET = 100000
SMOKE_NAME = "NON_PRIMARY_CHRONOLOGICAL_TRAINING_SMOKE_TEST"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def civil_us(series: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    values = parsed.to_numpy(dtype="datetime64[us]")
    missing = np.isnat(values)
    out = values.astype(np.int64)
    out[missing] = NAT
    return out


def second_sunday_of_march(year: int) -> datetime:
    day = datetime(year, 3, 1)
    return datetime(year, 3, 1 + (6 - day.weekday()) % 7 + 7, 2, 0, 0)


def first_sunday_of_november(year: int) -> datetime:
    day = datetime(year, 11, 1)
    return datetime(year, 11, 1 + (6 - day.weekday()) % 7, 2, 0, 0)


def span_kept(prefix_lengths: list[int], current_len: int) -> tuple[list[int], list[int]]:
    position = 0
    outcome_at = []
    end_at = []
    for length in prefix_lengths:
        position += min(MAX_SEQ_ACTIONS, int(length)) + 1
        outcome_at.append(position - 1)
        end_at.append(position)
    total = position + min(MAX_SEQ_ACTIONS, int(current_len))
    keep = max(0, total - MAX_HIST_TOKENS)
    any_idx = [i for i, end in enumerate(end_at) if end > keep]
    out_idx = [i for i, outcome in enumerate(outcome_at) if outcome >= keep]
    return any_idx, out_idx


def rank_ids(ids: list[str], prefix: str, k: int) -> list[str]:
    return sorted(ids, key=lambda uid: hashlib.sha256(f"{prefix}|{uid}".encode()).hexdigest())[:k]


def event_rank(source_row: int) -> str:
    return hashlib.sha256(f"{EVENT_PREFIX}|{source_row}".encode()).hexdigest()


def make_event(row: dict) -> Event:
    return Event(
        user_id=str(row["user_id"]),
        problem_id=str(row["problem_id"]),
        skill_id=str(row["skill_id"]),
        correct=int(row["correct"]),
        hint_count=int(row["hint_count"]),
        attempt_count=int(row["attempt_count"]),
        first_action=str(row["first_action"]),
        action_types=tuple(str(tok).lower() for tok in (row["action_types"] or [])),
        action_timestamps=tuple("" if tok is None else str(tok) for tok in (row["action_timestamps"] or [])),
        split=str(row["split"]),
    )


def pfa_fingerprint(ev: Event) -> tuple:
    return (int(ev.correct), int(ev.hint_count), int(ev.attempt_count), ev.action_types)


def gru_fingerprint(ev: Event) -> tuple:
    # The original scorer builds candidates with syn_from_sig, which keeps the
    # target skill. Distinct neural candidates of one key are action prefixes.
    return ev.action_types[:MAX_SEQ_ACTIONS]


def sig_digest(ev: Event) -> bytes:
    return hashlib.blake2s(repr(rich_signature(ev)).encode("utf-8"), digest_size=16).digest()


def self_check_history() -> None:
    start = np.array([0, 5, 12, 1], dtype=np.int64)
    end = np.array([10, 6, 14, 0], dtype=np.int64)
    row = np.array([0, 1, 2, 3], dtype=np.int64)
    valid = np.array([True, True, True, False])
    got = completed_ids(start, end, row, valid, target_pos=2)
    if got != [1, 0]:
        raise SystemExit(f"history self-check failed: {got}")
    if completed_ids(start, end, row, valid, target_pos=1) != []:
        raise SystemExit("unfinished overlap entered the pre-event history")
    if 3 in completed_ids(start, end, row, valid, target_pos=2):
        raise SystemExit("invalid interval entered history")


def completed_ids(start: np.ndarray, end: np.ndarray, row: np.ndarray, valid: np.ndarray, target_pos: int) -> list[int]:
    idx = np.flatnonzero(valid)
    if len(idx) == 0 or not valid[target_pos]:
        return []
    order = idx[np.lexsort((row[idx], end[idx]))]
    cut = int(np.searchsorted(end[order], start[target_pos], side="left"))
    return [int(i) for i in order[:cut]]


def direct_completed(start: np.ndarray, end: np.ndarray, row: np.ndarray, valid: np.ndarray, target_pos: int) -> list[int]:
    chosen = [
        int(i)
        for i in range(len(start))
        if i != target_pos and valid[i] and valid[target_pos] and int(end[i]) < int(start[target_pos])
    ]
    chosen.sort(key=lambda i: (int(end[i]), int(row[i])))
    return chosen


def load_light(split: str) -> pd.DataFrame:
    path = RECON / f"{split}.parquet"
    table = pq.read_table(path, columns=["source_row", "user_id", "skill_id", "correct", "start_time", "end_time", "action_types"])
    frame = table.select(["source_row", "user_id", "skill_id", "correct", "start_time", "end_time"]).to_pandas()
    frame["n_actions"] = pc.list_value_length(table.column("action_types")).fill_null(0).to_numpy(zero_copy_only=False).astype(np.int16)
    frame["start_us"] = civil_us(frame["start_time"])
    frame["end_us"] = civil_us(frame["end_time"])
    frame["user_id"] = frame["user_id"].astype(str)
    frame["skill_id"] = frame["skill_id"].astype(str)
    return frame


def eastern_offset_hours(clock: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = clock.dt.year.to_numpy()
    months = clock.dt.month.to_numpy()
    days = clock.dt.day.to_numpy()
    hours = clock.dt.hour.to_numpy()
    finite = np.isfinite(years.astype(np.float64))
    offset = np.full(len(clock), -1, dtype=np.int8)
    ambiguous = np.zeros(len(clock), dtype=bool)
    nonexistent = np.zeros(len(clock), dtype=bool)
    for year in sorted({int(value) for value in years[finite]}):
        start_dst = second_sunday_of_march(year)
        end_dst = first_sunday_of_november(year)
        in_year = finite & (years == year)
        nonexistent[in_year & (months == start_dst.month) & (days == start_dst.day) & (hours == 2)] = True
        ambiguous[in_year & (months == end_dst.month) & (days == end_dst.day) & (hours == 1)] = True
        after_start = (months > start_dst.month) | ((months == start_dst.month) & (days > start_dst.day)) | ((months == start_dst.month) & (days == start_dst.day) & (hours >= 3))
        before_end = (months < end_dst.month) | ((months == end_dst.month) & (days < end_dst.day)) | ((months == end_dst.month) & (days == end_dst.day) & (hours < 1))
        daylight = in_year & after_start & before_end
        offset[daylight] = 4
        offset[in_year & ~daylight & ~nonexistent] = 5
        offset[in_year & ambiguous] = 4
    return offset, ambiguous, nonexistent


def timezone_flags(frame: pd.DataFrame) -> dict[str, int]:
    clock_s = pd.to_datetime(frame["start_time"], errors="coerce", format="mixed")
    clock_e = pd.to_datetime(frame["end_time"], errors="coerce", format="mixed")
    off_s, amb_s, miss_s = eastern_offset_hours(clock_s)
    off_e, amb_e, miss_e = eastern_offset_hours(clock_e)
    hour_us = np.int64(3600 * 1_000_000)
    comparable = (off_s > 0) & (off_e > 0) & ~amb_s & ~amb_e & ~miss_s & ~miss_e
    civil_ok = frame["end_us"].to_numpy() >= frame["start_us"].to_numpy()
    utc_end = frame["end_us"].to_numpy() + off_e.astype(np.int64) * hour_us
    utc_start = frame["start_us"].to_numpy() + off_s.astype(np.int64) * hour_us
    utc_ok = utc_end >= utc_start
    flipped = comparable & (civil_ok != utc_ok)
    return {
        "ambiguous_start_or_end": int(np.sum(amb_s | amb_e)),
        "nonexistent_start_or_end": int(np.sum(miss_s | miss_e)),
        "civil_versus_eastern_interval_order_flips": int(np.sum(flipped)),
    }


def analyze_split(frame: pd.DataFrame, split: str, samples: list[dict]) -> dict[str, int]:
    start = frame["start_us"].to_numpy()
    end = frame["end_us"].to_numpy()
    row = frame["source_row"].to_numpy()
    skill = frame["skill_id"].to_numpy()
    correct = frame["correct"].to_numpy()
    n_act = frame["n_actions"].to_numpy()
    user = frame["user_id"].to_numpy()
    valid = (start != NAT) & (end != NAT) & (end >= start)
    stats = Counter()
    stats["events"] = int(len(frame))
    stats["valid_intervals"] = int(np.sum(valid))
    stats["invalid_intervals"] = int(np.sum(~valid))
    stats["invalid_correct"] = int(np.sum(~valid & (correct == 1)))
    stats["invalid_incorrect"] = int(np.sum(~valid & (correct != 1)))
    codes, _ = pd.factorize(user, sort=False)
    order = np.argsort(codes, kind="mergesort")
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    bounds = np.r_[0, cuts, len(order)]
    users_invalid = 0
    taken = Counter()
    quota = 20
    day_us = np.int64(86_400_000_000)
    dst_days = {
        int(np.datetime64("2012-11-04", "D").astype(np.int64)),
        int(np.datetime64("2013-03-10", "D").astype(np.int64)),
        int(np.datetime64("2013-11-03", "D").astype(np.int64)),
    }
    for left, right in zip(bounds[:-1], bounds[1:]):
        idx = order[left:right]
        st, en, rw = start[idx], end[idx], row[idx]
        sk, na, ok = skill[idx], n_act[idx], valid[idx]
        n = len(idx)
        stats["users"] += 1
        if np.any(~ok):
            users_invalid += 1
        vpos = np.flatnonzero(ok)
        if len(vpos) == 0:
            continue
        start_ord = vpos[np.lexsort((rw[vpos], st[vpos]))]
        comp_ord = vpos[np.lexsort((rw[vpos], en[vpos]))]
        comp_end = en[comp_ord]
        mismatch = np.flatnonzero(start_ord != comp_ord)
        first_order_mismatch = int(mismatch[0]) if len(mismatch) else len(start_ord)
        file_ord = np.argsort(rw, kind="mergesort")
        if n > 1:
            left = file_ord[:-1]
            right = file_ord[1:]
            both_starts = (st[left] != NAT) & (st[right] != NAT)
            inverted = both_starts & (st[left] > st[right])
            both_valid = ok[left] & ok[right]
            both_ends = (en[left] != NAT) & (en[right] != NAT)
            stats["adjacent_file_pairs"] += int(np.sum(both_starts))
            stats["adjacent_start_inversions"] += int(np.sum(inverted))
            stats["tempfor_condition_start_next_less_than_end_prev_given_inversion"] += int(np.sum(inverted & (en[left] != NAT) & (st[right] < en[left])))
            stats["adjacent_next_start_inside_previous_valid_interval"] += int(np.sum(both_valid & (st[left] < st[right]) & (st[right] < en[left])))
            intersection = both_valid & (st[left] < en[right]) & (st[right] < en[left])
            stats["adjacent_valid_interval_intersection"] += int(np.sum(intersection))
            stats["adjacent_reverse_order_intersection"] += int(np.sum(intersection & inverted))
            stats["adjacent_end_inversions"] += int(np.sum(both_ends & (en[left] > en[right])))
        start_values, start_counts = np.unique(st[vpos], return_counts=True)
        end_values, end_counts = np.unique(en[vpos], return_counts=True)
        if np.any(start_counts > 1):
            stats["users_with_start_tie"] += 1
        if np.any(end_counts > 1):
            stats["users_with_completion_tie"] += 1
        stats["start_tie_groups"] += int(np.sum(start_counts > 1))
        stats["completion_tie_groups"] += int(np.sum(end_counts > 1))
        stats["events_participating_in_a_start_tie"] += int(np.sum(start_counts[start_counts > 1]))
        stats["events_participating_in_a_completion_tie"] += int(np.sum(end_counts[end_counts > 1]))
        file_has_inversion = bool(np.any((st[file_ord[:-1]] != NAT) & (st[file_ord[1:]] != NAT) & (st[file_ord[:-1]] > st[file_ord[1:]]))) if n > 1 else False
        starts_sorted = st[start_ord]
        cuts = np.searchsorted(comp_end, starts_sorted, side="left")
        positions = np.arange(len(start_ord))
        strict = np.searchsorted(starts_sorted, starts_sorted, side="left")
        stats["valid_targets"] += int(len(start_ord))
        stats["targets_with_at_least_two_completed_predecessors"] += int(np.sum(cuts >= 2))
        differs = cuts != positions
        stats["targets_whose_completed_set_differs_from_earlier_started_set"] += int(np.sum(differs))
        stats["excluded_earlier_started_updates"] += int(np.sum(positions - cuts))
        stats["targets_with_an_earlier_interaction_still_unfinished"] += int(np.sum(strict - cuts > 0))
        stats["targets_with_an_equal_start_predecessor"] += int(np.sum(positions > strict))
        stats["targets_same_set_different_order"] += int(np.sum((cuts == positions) & (positions > first_order_mismatch)))
        stats["targets_with_different_history_order"] += int(np.sum(differs | (positions > first_order_mismatch)))
        delta = positions - cuts
        stats["history_set_difference_0"] += int(np.sum(delta == 0))
        stats["history_set_difference_1"] += int(np.sum(delta == 1))
        stats["history_set_difference_2_to_5"] += int(np.sum((delta >= 2) & (delta <= 5)))
        stats["history_set_difference_6_to_20"] += int(np.sum((delta >= 6) & (delta <= 20)))
        stats["history_set_difference_21_to_100"] += int(np.sum((delta >= 21) & (delta <= 100)))
        stats["history_set_difference_101_or_more"] += int(np.sum(delta >= 101))
        if len(comp_ord):
            left_bad = np.zeros(len(cuts), dtype=bool)
            has_left = cuts > 0
            left_bad[has_left] = en[comp_ord[cuts[has_left] - 1]] >= starts_sorted[has_left]
            right_bad = np.zeros(len(cuts), dtype=bool)
            has_right = cuts < len(comp_ord)
            right_bad[has_right] = en[comp_ord[cuts[has_right]]] < starts_sorted[has_right]
            stats["boundary_violations"] += int(np.sum(left_bad | right_bad))
        active = Counter()
        active_total = 0
        retire = []
        want_samples = split in {"development", "calibration"} and sum(taken.values()) < 220
        near_dst = np.abs((st // day_us) - np.int64(next(iter(dst_days)))) <= 1
        for day in dst_days:
            near_dst |= np.abs((st // day_us) - np.int64(day)) <= 1
        for t, pos in enumerate(start_ord):
            s = int(st[pos])
            cut = int(cuts[t])
            while retire and retire[0][0] < s:
                _ended, skill_id = heapq.heappop(retire)
                active[skill_id] -= 1
                active_total -= 1
            if active[sk[pos]] > 0:
                stats["targets_with_unfinished_same_skill_prior_under_start_sort"] += 1
            if active_total != t - cut:
                stats["active_set_mismatches"] += 1
            if split != "confirmatory":
                c0 = max(0, cut - MAX_EVAL_PREFIX_EVENTS)
                s0 = max(0, t - MAX_EVAL_PREFIX_EVENTS)
                c_tail = comp_ord[c0:cut]
                s_tail = start_ord[s0:t]
                if not np.array_equal(c_tail, s_tail):
                    stats["targets_whose_last_40_histories_differ"] += 1
                c_keep = span_kept(na[c_tail].tolist(), int(na[pos]))[1]
                s_keep = span_kept(na[s_tail].tolist(), int(na[pos]))[1]
                if not np.array_equal(c_tail[c_keep], s_tail[s_keep]):
                    stats["targets_whose_surviving_outcome_events_differ"] += 1
            heapq.heappush(retire, (int(en[pos]), sk[pos]))
            active[sk[pos]] += 1
            active_total += 1
            if not want_samples:
                continue
            kind = None
            if taken["empty"] < quota and cut == 0:
                kind = "empty"
            elif taken["short"] < quota and 0 < cut < 3:
                kind = "short"
            elif taken["long"] < quota and cut > 40:
                kind = "long"
            elif taken["overlap"] < quota and int(strict[t] - cut) > 0:
                kind = "overlap"
            elif taken["equal_start"] < quota and t > int(strict[t]):
                kind = "equal_start"
            elif taken["invalid_neighbor"] < quota and np.any(~ok):
                kind = "invalid_neighbor"
            elif taken["dst"] < quota and bool(near_dst[pos]):
                kind = "dst"
            elif taken["inverted"] < quota and file_has_inversion:
                kind = "inverted"
            elif taken["equal_end"] < quota and np.any(end_counts > 1):
                kind = "equal_end"
            elif taken["stride"] < 160 and t % 11 == 0:
                kind = "stride"
            if kind is None:
                continue
            direct = direct_completed(st, en, rw, ok, int(pos))
            fast = [int(i) for i in comp_ord[:cut]]
            samples.append({"split": split, "kind": kind, "match": direct == fast, "future": any(int(en[i]) >= s for i in fast), "length": len(fast)})
            taken[kind] += 1
            want_samples = sum(taken.values()) < 220
    stats["users_with_invalid_interval"] = users_invalid
    stats["sample_kinds"] = dict(taken)
    return stats


def stream_development(dev_users: set[str], cal_keys: dict[str, set]) -> dict:
    key_sets = {schema: set() for schema in ("S0", "S1", "S2", "S3")}
    sigs = set()
    nesting_violations = 0
    events = 0
    skill_order = []
    skill_seen = set()
    atype_order = []
    atype_seen = set()
    bundles = {schema: {} for schema in ("S0", "S1", "S2", "S3")}
    dev_pilot = defaultdict(list)
    duplicate_signatures = 0
    parquet = pq.ParquetFile(RECON / "development.parquet")
    for batch in parquet.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            events += 1
            ev = make_event(row)
            if ev.skill_id not in skill_seen:
                skill_seen.add(ev.skill_id)
                skill_order.append(ev.skill_id)
            for tok in ev.action_types:
                if tok not in atype_seen:
                    atype_seen.add(tok)
                    atype_order.append(tok)
            digest = sig_digest(ev)
            if digest in sigs:
                duplicate_signatures += 1
            else:
                sigs.add(digest)
            keys = {schema: release_key(ev, schema) for schema in key_sets}
            if project_down(keys["S3"], "S3", "S2") != keys["S2"] or project_down(keys["S2"], "S2", "S1") != keys["S1"] or project_down(keys["S1"], "S1", "S0") != keys["S0"]:
                nesting_violations += 1
            for schema, key in keys.items():
                key_sets[schema].add(key)
                if key in cal_keys[schema]:
                    slot = bundles[schema].get(key)
                    if slot is None:
                        slot = {"n": 0, "users": set(), "pfa": set(), "gru": set()}
                        bundles[schema][key] = slot
                    slot["n"] += 1
                    slot["users"].add(ev.user_id)
                    slot["pfa"].add(pfa_fingerprint(ev))
                    slot["gru"].add(gru_fingerprint(ev))
            if ev.user_id in dev_users:
                dev_pilot[ev.user_id].append(row)
        if events % 500000 == 0:
            print(f"development_scan {events}", flush=True)
    return {
        "events": events,
        "duplicate_signatures": duplicate_signatures,
        "unique_signatures": len(sigs),
        "nesting_violations": nesting_violations,
        "key_counts": {schema: len(keys) for schema, keys in key_sets.items()},
        "bundles": bundles,
        "skill_order": skill_order,
        "atype_order": atype_order,
        "dev_pilot": dev_pilot,
    }


def load_pilot_rows(split: str, users: set[str]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    parquet = pq.ParquetFile(RECON / f"{split}.parquet")
    for batch in parquet.iter_batches(batch_size=50000):
        for row in batch.to_pylist():
            uid = str(row["user_id"])
            if uid in users:
                grouped[uid].append(row)
    return grouped


def row_times(row: dict) -> tuple[int, int, bool]:
    start = civil_us(pd.Series([row["start_time"]]))[0]
    end = civil_us(pd.Series([row["end_time"]]))[0]
    return int(start), int(end), bool(start != NAT and end != NAT and end >= start)


def ordered_valid(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    valid = []
    invalid = []
    for row in rows:
        start, end, ok = row_times(row)
        row = dict(row)
        row["_start"] = start
        row["_end"] = end
        row["_source"] = int(row["source_row"])
        if ok:
            valid.append(row)
        else:
            invalid.append(row)
    valid_by_start = sorted(valid, key=lambda item: (item["_start"], item["_source"]))
    valid_by_end = sorted(valid, key=lambda item: (item["_end"], item["_source"]))
    return valid_by_start, valid_by_end


def history_slice(by_end: list[dict], start: int) -> list[dict]:
    ends = [item["_end"] for item in by_end]
    cut = int(np.searchsorted(np.array(ends, dtype=np.int64), start, side="left"))
    return by_end[:cut]


def run_pfa(users: list[str], grouped: dict[str, list[dict]], bundles: dict) -> list[dict]:
    rows_out = []
    counts = {schema: Counter() for schema in ("S0", "S1", "S2", "S3")}
    violations = 0
    checked = 0
    for user_i, uid in enumerate(users):
        if user_i % 16 == 0:
            print(f"pfa_user {user_i}", flush=True)
        by_start, by_end = ordered_valid(grouped.get(uid, []))
        counts_user_targets = len(by_start)
        for schema in counts:
            counts[schema]["users"] += 1
            counts[schema]["valid_targets"] += counts_user_targets
            counts[schema]["invalid_excluded"] += len(grouped.get(uid, [])) - counts_user_targets
        cursor = 0
        model = TransparentActionPFA()
        for target in by_start:
            while cursor < len(by_end) and by_end[cursor]["_end"] < target["_start"]:
                model.update(make_event(by_end[cursor]))
                cursor += 1
            hist = by_end[:cursor]
            checked += 1
            if any(item["_end"] >= target["_start"] or item["_source"] == target["_source"] for item in hist):
                violations += 1
            pre_n = {skill: float(state["n"]) for skill, state in model.skill_stats.items()}
            ev = make_event(target)
            for schema in counts:
                key = release_key(ev, schema)
                slot = bundles[schema].get(key)
                n_users = len(slot["users"]) if slot else 0
                n_sigs = slot["n"] if slot else 0
                supported = slot is not None and (n_users >= 2 or n_sigs >= 2)
                if not supported:
                    counts[schema]["unsupported"] += 1
                    continue
                decisions = set()
                realized_fp = pfa_fingerprint(ev)
                realized_decision = None
                covered = realized_fp in slot["pfa"]
                for fp in slot["pfa"]:
                    live = model.skill_stats[ev.skill_id]
                    backup = dict(live)
                    syn = Event(ev.user_id, ev.problem_id, ev.skill_id, fp[0], fp[1], fp[2], ev.first_action, fp[3], tuple(), ev.split)
                    model.update(syn)
                    score = model.mastery(ev.skill_id)
                    model.skill_stats[ev.skill_id] = backup
                    decision = "progress" if score >= TAU else "review"
                    decisions.add(decision)
                    if fp == realized_fp:
                        realized_decision = decision
                if realized_decision is None:
                    live = model.skill_stats[ev.skill_id]
                    backup = dict(live)
                    model.update(ev)
                    score = model.mastery(ev.skill_id)
                    model.skill_stats[ev.skill_id] = backup
                    realized_decision = "progress" if score >= TAU else "review"
                if len(decisions) == 1:
                    counts[schema]["stable"] += 1
                    unique = next(iter(decisions))
                    if realized_decision != unique:
                        counts[schema]["false_stable"] += 1
                    if covered:
                        counts[schema]["stable_and_fingerprint_covered"] += 1
                    else:
                        counts[schema]["stable_and_fingerprint_absent"] += 1
                else:
                    counts[schema]["set_valued"] += 1
                if realized_decision in decisions:
                    counts[schema]["realized_action_contained"] += 1
                if covered:
                    counts[schema]["realized_fingerprint_covered"] += 1
                counts[schema]["candidate_fingerprints_scored"] += len(slot["pfa"])
            if pre_n.get(ev.skill_id, 0.0) != float(model.skill_stats[ev.skill_id]["n"]):
                violations += 1
    for schema in counts:
        for metric in ("stable", "set_valued", "unsupported", "false_stable", "realized_action_contained", "realized_fingerprint_covered"):
            counts[schema][metric] += 0
    for schema, counter in counts.items():
        denom = counter["valid_targets"]
        for key, value in counter.items():
            rows_out.append({"schema": schema, "metric": key, "n": int(value), "denominator": int(denom), "fraction": value / denom if denom and key != "users" else None})
    rows_out.append({"schema": "all", "metric": "history_contract_violations", "n": violations, "denominator": checked, "fraction": violations / checked if checked else None})
    return rows_out


def build_pilot(skill_order: list[str], atype_order: list[str]) -> NeuralPilot:
    random.seed(17)
    torch.manual_seed(17)
    pilot = NeuralPilot()
    for skill in skill_order:
        pilot.skill2i[skill] = len(pilot.skill2i)
    for tok in atype_order:
        pilot.atype2i[tok] = len(pilot.atype2i)
    n_skills = len(pilot.skill2i) - 1
    n_atypes = len(pilot.atype2i) - 1
    pilot.model = ActionTokenGRU(n_skills, n_atypes)
    pilot.model.to(pilot.device)
    return pilot


def train_smoke(pilot: NeuralPilot, users: list[str], grouped: dict[str, list[dict]]) -> dict:
    random.seed(17)
    torch.manual_seed(17)
    opt = torch.optim.Adam(pilot.model.parameters(), lr=1e-3)
    pilot.model.train()
    total = 0.0
    n = 0
    skipped = 0
    violations = 0
    examples_future_checked = 0
    for uid in users:
        by_start, by_end = ordered_valid(grouped.get(uid, []))
        for target in by_start:
            hist_rows = history_slice(by_end, target["_start"])
            prefix = [make_event(item) for item in hist_rows[-MAX_EVAL_PREFIX_EVENTS:]]
            current = make_event(target)
            if any(item["_end"] >= target["_start"] for item in hist_rows):
                violations += 1
            examples_future_checked += 1
            skills, atypes = pilot._tokens(prefix, current, False)
            if not skills:
                skipped += 1
                continue
            if pilot.model.outcome0 in atypes[-min(MAX_SEQ_ACTIONS, len(current.action_types)) :] or pilot.model.outcome1 in atypes[-min(MAX_SEQ_ACTIONS, max(len(current.action_types), 1)) :]:
                # Current outcome tokens are appended only after current actions. Their presence in the tail is checked exactly below.
                pass
            current_tokens = min(MAX_SEQ_ACTIONS, len(current.action_types))
            tail = atypes[-current_tokens:] if current_tokens else []
            if any(tok in {pilot.model.outcome0, pilot.model.outcome1} for tok in tail):
                violations += 1
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
            if n % 2000 == 0:
                print(f"smoke_examples {n}", flush=True)
    pilot.model.eval()
    param_count = sum(p.numel() for p in pilot.model.parameters())
    path = OUT / f"{SMOKE_NAME}.pt"
    torch.save(
        {
            "label": SMOKE_NAME,
            "primary_checkpoint": False,
            "state_dict": {k: v.detach().cpu() for k, v in pilot.model.state_dict().items()},
            "skill2i": pilot.skill2i,
            "atype2i": pilot.atype2i,
        },
        path,
    )
    return {
        "label": SMOKE_NAME,
        "primary_checkpoint": False,
        "checkpoint_path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "smoke_rng_seed": 17,
        "smoke_rng_note": "Seed 17 is applied before weight initialization and again before the single epoch. It does not make this file the primary chronological seed-17 checkpoint.",
        "epochs": 1,
        "historical_epochs": 2,
        "users": len(users),
        "historical_user_cap_per_epoch": 3000,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "loss": "binary_cross_entropy on current correctness",
        "architecture": "ActionTokenGRU emb=32 hid=64",
        "parameter_count": param_count,
        "historical_full_vocabulary_parameter_count": 33857,
        "n_skills": len(pilot.skill2i) - 1,
        "n_action_types": len(pilot.atype2i) - 1,
        "training_examples": n,
        "skipped_empty_token_examples": skipped,
        "mean_loss": total / n if n else None,
        "history_contract_violations": violations,
        "examples_checked": examples_future_checked,
        "fit_function_edited": False,
        "fit_function_called": False,
        "reason_adapter_required": "NeuralPilot.fit uses the positional prefix of a supplied list. Availability-aware history is the set of events with end < start, ordered by completion, and is not that positional prefix.",
    }


def independent_tokens(pilot: NeuralPilot, prefix: list[Event], current: Event) -> tuple[list[int], list[int]]:
    skills: list[int] = []
    atypes: list[int] = []
    for ev in prefix:
        sid = pilot.skill2i.get(ev.skill_id, 0)
        for tok in ev.action_types[:MAX_SEQ_ACTIONS]:
            skills.append(sid)
            atypes.append(pilot.atype2i.get(tok, 0))
        skills.append(sid)
        atypes.append(pilot.model.outcome1 if ev.correct == 1 else pilot.model.outcome0)
    sid = pilot.skill2i.get(current.skill_id, 0)
    for tok in current.action_types[:MAX_SEQ_ACTIONS]:
        skills.append(sid)
        atypes.append(pilot.atype2i.get(tok, 0))
    if not skills:
        return [], []
    return skills[-MAX_HIST_TOKENS:], atypes[-MAX_HIST_TOKENS:]


def gru_smoke_eval(pilot: NeuralPilot, users: list[str], grouped: dict[str, list[dict]], bundles: dict, train_info: dict) -> tuple[list[dict], dict]:
    pool = []
    for uid in users:
        by_start, by_end = ordered_valid(grouped.get(uid, []))
        for target in by_start:
            pool.append((event_rank(int(target["source_row"])), uid, target, by_end))
    pool.sort(key=lambda item: item[0])
    chosen = pool[:2000]
    schema_cost = {schema: 0 for schema in ("S3", "S2", "S1", "S0")}
    for _rank, _uid, target, _by_end in chosen:
        ev = make_event(target)
        for schema in schema_cost:
            slot = bundles[schema].get(release_key(ev, schema))
            if slot and (len(slot["users"]) >= 2 or slot["n"] >= 2):
                schema_cost[schema] += len(slot["gru"])
    selected = []
    running = 0
    for schema in ("S3", "S2", "S1", "S0"):
        if running + schema_cost[schema] <= GRU_CANDIDATE_BUDGET:
            selected.append(schema)
            running += schema_cost[schema]
    pilot.model.eval()
    scores = []
    labels = []
    violations = 0
    token_cases = []
    counts = Counter()
    counts["events_scored"] = len(chosen)
    counts["eligible_valid_events_in_128_users"] = len(pool)
    for schema in ("S0", "S1", "S2", "S3"):
        for name in ("stable", "set_valued", "unsupported", "false_stable", "realized_action_contained"):
            counts[f"{schema}_{name}"] += 0
    repeats = []
    shared_prefix_checks = 0
    shared_prefix_failures = 0
    with torch.no_grad():
        for rank_i, (_rank, _uid, target, by_end) in enumerate(chosen):
            hist_rows = history_slice(by_end, target["_start"])
            if any(item["_end"] >= target["_start"] for item in hist_rows):
                violations += 1
            prefix = [make_event(item) for item in hist_rows[-MAX_EVAL_PREFIX_EVENTS:]]
            current = make_event(target)
            skills, atypes = pilot._tokens(prefix, current, False)
            current_n = min(MAX_SEQ_ACTIONS, len(current.action_types))
            if current_n and any(tok in {pilot.model.outcome0, pilot.model.outcome1} for tok in atypes[-current_n:]):
                violations += 1
            if not skills:
                counts["empty_score_inputs"] += 1
                continue
            pred = pilot.model.forward_from_state(
                torch.tensor([skills], dtype=torch.long, device=pilot.device),
                torch.tensor([atypes], dtype=torch.long, device=pilot.device),
                None,
            )[0]
            score = float(pred.item())
            scores.append(score)
            labels.append(int(current.correct))
            if rank_i and rank_i % 200 == 0:
                print(f"gru_infer {rank_i}", flush=True)
            if rank_i < 20:
                pred2 = pilot.model.forward_from_state(
                    torch.tensor([skills], dtype=torch.long, device=pilot.device),
                    torch.tensor([atypes], dtype=torch.long, device=pilot.device),
                    None,
                )[0]
                repeats.append(abs(score - float(pred2.item())) == 0.0)
            if len(token_cases) < 150 and rank_i % max(len(chosen) // 150, 1) == 0:
                again_s, again_a = independent_tokens(pilot, prefix, current)
                original_s, original_a = pilot._tokens(prefix, current, False)
                token_cases.append(
                    {
                        "match": original_s == again_s and original_a == again_a,
                        "current_outcome_withheld": not any(tok in {pilot.model.outcome0, pilot.model.outcome1} for tok in original_a[-current_n:]) if current_n else True,
                        "history_available": all(item["_end"] < target["_start"] for item in hist_rows[-MAX_EVAL_PREFIX_EVENTS:]),
                        "prefix_length": len(prefix),
                        "token_length": len(original_s),
                    }
                )
            for schema in ("S0", "S1", "S2", "S3"):
                if schema not in selected:
                    counts[f"{schema}_certificate_not_executed_budget"] += 1
                    continue
                key = release_key(current, schema)
                slot = bundles[schema].get(key)
                if slot is None or not (len(slot["users"]) >= 2 or slot["n"] >= 2):
                    counts[f"{schema}_unsupported"] += 1
                    continue
                cands = []
                seen = set()
                shared_by_length = {}
                for fp in slot["gru"]:
                    if fp in seen:
                        continue
                    seen.add(fp)
                    cand = Event(current.user_id, current.problem_id, current.skill_id, current.correct, current.hint_count, current.attempt_count, current.first_action, fp, tuple(), current.split)
                    cands.append(cand)
                    if shared_prefix_checks < 40:
                        full_s, full_a = pilot._tokens(prefix, cand, False)
                        own_s, own_a = independent_tokens(pilot, prefix, cand)
                        cand_n = min(MAX_SEQ_ACTIONS, len(cand.action_types))
                        hist_part = full_a[:-cand_n] if cand_n else full_a
                        same_history = shared_by_length.setdefault(cand_n, hist_part) == hist_part
                        no_current_outcome = not cand_n or not any(tok in {pilot.model.outcome0, pilot.model.outcome1} for tok in full_a[-cand_n:])
                        shared_prefix_checks += 1
                        if full_s != own_s or full_a != own_a or not same_history or not no_current_outcome:
                            shared_prefix_failures += 1
                values = pilot.mastery_branches_shared_prefix(prefix, cands)
                decisions = {"progress" if value >= TAU else "review" for value in values}
                realized = pilot.mastery_branch(prefix, current)
                realized_decision = "progress" if realized >= TAU else "review"
                if len(decisions) == 1:
                    counts[f"{schema}_stable"] += 1
                    if realized_decision != next(iter(decisions)):
                        counts[f"{schema}_false_stable"] += 1
                else:
                    counts[f"{schema}_set_valued"] += 1
                if realized_decision in decisions:
                    counts[f"{schema}_realized_action_contained"] += 1
    y = np.array(labels, dtype=np.float64)
    p = np.clip(np.array(scores, dtype=np.float64), 1e-6, 1 - 1e-6)
    logloss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))) if len(y) else None
    accuracy = float(np.mean((p >= 0.5) == (y == 1))) if len(y) else None
    provenance = {
        "cases": len(token_cases),
        "token_id_mismatches": sum(1 for case in token_cases if not case["match"]),
        "current_outcome_leaks": sum(1 for case in token_cases if not case["current_outcome_withheld"]),
        "history_availability_failures": sum(1 for case in token_cases if not case["history_available"]),
        "deterministic_repeat_failures": sum(1 for ok in repeats if not ok),
        "deterministic_repeats": len(repeats),
        "score_contract_violations": violations,
        "shared_prefix_checks": shared_prefix_checks,
        "shared_prefix_failures": shared_prefix_failures,
    }
    summary = {
        "label": SMOKE_NAME,
        "not_confirmatory_evidence": True,
        "events_requested": 2000,
        "events_available": len(pool),
        "events_with_a_score": len(scores),
        "candidate_budget": GRU_CANDIDATE_BUDGET,
        "candidate_cost_by_schema": schema_cost,
        "schemas_executed": selected,
        "schemas_not_executed": [schema for schema in ("S3", "S2", "S1", "S0") if schema not in selected],
        "smoke_log_loss": logloss,
        "smoke_threshold_0_5_accuracy": accuracy,
        "interpretation": "These predictive numbers come from a 128-user, one-epoch pilot. They do not support or refute the study's model comparison.",
        "counts": dict(counts),
        "training": {
            "parameter_count": train_info["parameter_count"],
            "examples": train_info["training_examples"],
        },
    }
    flat = []
    denom = summary["events_with_a_score"]
    for key, value in counts.items():
        flat.append({"metric": key, "n": int(value), "denominator": denom, "fraction": value / denom if denom else None})
    flat.append({"metric": "smoke_log_loss", "n": logloss, "denominator": denom, "fraction": None})
    flat.append({"metric": "smoke_threshold_0_5_accuracy", "n": accuracy, "denominator": denom, "fraction": None})
    return flat, {"summary": summary, "provenance": provenance}


def records(split: str, stats: dict) -> list[dict]:
    denom = stats.get("valid_targets", 0)
    rows = []
    for key, value in stats.items():
        if isinstance(value, dict):
            continue
        if key.startswith("targets_") or key.startswith("excluded_") or key.startswith("history_set_") or key in {"boundary_violations", "active_set_mismatches"}:
            use = denom
        elif key.startswith("users_"):
            use = stats.get("users", denom)
        elif key.startswith("adjacent_") or key.startswith("tempfor_"):
            use = stats.get("adjacent_file_pairs", denom)
        else:
            use = stats.get("events", denom)
        if key in {"events", "users"}:
            use = value
        rows.append({"split": split, "metric": key, "n": value, "denominator": use, "fraction": value / use if isinstance(use, int) and use and isinstance(value, int) else None})
    return rows


def write_checksums() -> None:
    files = []
    for path in sorted(item for item in OUT.iterdir() if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"algorithm": "sha256", "excluded": ["results/assistments/chronological_protocol/output_checksums.json"], "n_files": len(files), "files": files}
    (OUT / "output_checksums.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    self_check_history()
    preflight = {"zip_sha256": sha256_file(STAGE), "zip_bytes": STAGE.stat().st_size, "parquet": {}}
    preflight["zip_match"] = preflight["zip_sha256"] == EXPECTED_ZIP and preflight["zip_bytes"] == 576365408
    for split, (digest, size, users, events) in EXPECTED.items():
        path = RECON / f"{split}.parquet"
        observed = sha256_file(path)
        preflight["parquet"][split] = {"sha256": observed, "bytes": path.stat().st_size, "expected_users": users, "expected_events": events, "match": observed == digest and path.stat().st_size == size}
    (OUT / "source_integrity_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if not preflight["zip_match"] or any(not item["match"] for item in preflight["parquet"].values()):
        (OUT / "integrity_checks.json").write_text(json.dumps({"status": "BLOCKED_DATA_OR_EXECUTION"}, indent=2) + "\n", encoding="utf-8")
        write_checksums()
        raise SystemExit(2)

    samples: list[dict] = []
    split_stats = {}
    user_ids = {}
    tz = {}
    invalid_rows = []
    availability_rows = []
    compare_rows = []
    for split in ("development", "calibration", "confirmatory"):
        print(f"audit {split}", flush=True)
        frame = load_light(split)
        if frame["user_id"].nunique() != EXPECTED[split][2] or len(frame) != EXPECTED[split][3]:
            raise SystemExit(f"population mismatch {split}")
        user_ids[split] = frame["user_id"].drop_duplicates().tolist()
        tz[split] = timezone_flags(frame)
        stats = analyze_split(frame, split, samples)
        split_stats[split] = {key: value for key, value in stats.items() if not isinstance(value, dict)}
        split_stats[split]["sample_kinds"] = stats.get("sample_kinds", {})
        invalid_rows.append(
            {
                "split": split,
                "events": stats["events"],
                "valid_intervals": stats["valid_intervals"],
                "invalid_intervals": stats["invalid_intervals"],
                "invalid_correct": stats["invalid_correct"],
                "invalid_incorrect": stats["invalid_incorrect"],
                "users": stats["users"],
                "users_with_invalid_interval": stats["users_with_invalid_interval"],
            }
        )
        availability_rows.extend(records(split, {**split_stats[split], **tz[split]}))
        compare_rows.extend(records(split, split_stats[split]))
        del frame
    pd.DataFrame(invalid_rows).to_csv(OUT / "invalid_interval_accounting.csv", index=False)
    pd.DataFrame(availability_rows).to_csv(OUT / "overlap_and_availability_diagnostics.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(OUT / "start_sort_vs_completion_aware.csv", index=False)
    equivalence = {
        "samples": len(samples),
        "identity_mismatches": sum(1 for sample in samples if not sample["match"]),
        "future_or_unfinished_members": sum(1 for sample in samples if sample["future"]),
        "kinds": dict(Counter(sample["kind"] for sample in samples)),
        "by_split": dict(Counter(sample["split"] for sample in samples)),
        "full_population_boundary_violations": sum(split_stats[split].get("boundary_violations", 0) for split in split_stats),
        "active_set_mismatches": sum(split_stats[split].get("active_set_mismatches", 0) for split in split_stats),
    }
    (OUT / "chronological_history_equivalence.json").write_text(json.dumps(equivalence, indent=2) + "\n", encoding="utf-8")

    cal_users = rank_ids(user_ids["calibration"], CAL_PREFIX, 128)
    dev_users = rank_ids(user_ids["development"], DEV_PREFIX, 128)
    print("load calibration pilot", flush=True)
    cal_grouped = load_pilot_rows("calibration", set(cal_users))
    cal_keys = {schema: set() for schema in ("S0", "S1", "S2", "S3")}
    for rows in cal_grouped.values():
        for row in rows:
            start, end, ok = row_times(row)
            if not ok:
                continue
            ev = make_event(row)
            for schema in cal_keys:
                cal_keys[schema].add(release_key(ev, schema))
    print("development geometry", flush=True)
    geometry = stream_development(set(dev_users), cal_keys)
    support_equivalent = geometry["duplicate_signatures"] == 0 and geometry["unique_signatures"] == geometry["events"]
    geometry_out = {
        "development_events": geometry["events"],
        "key_counts": geometry["key_counts"],
        "expected_key_counts": EXPECTED_KEYS,
        "key_count_match": {schema: geometry["key_counts"][schema] == EXPECTED_KEYS[schema] for schema in EXPECTED_KEYS},
        "nesting_violations": geometry["nesting_violations"],
        "unique_rich_signatures": geometry["unique_signatures"],
        "duplicate_rich_signatures": geometry["duplicate_signatures"],
        "rich_signatures_unique": support_equivalent,
        "support_predicate": "(n_users >= 2) OR (n_rich_signatures >= 2)",
        "numerical_equivalence": "Unique rich signatures make n_rich_signatures equal the number of development events carrying the key. The predicate remains the original disjunction; recurrence of the key is a consequence, not a replacement definition.",
        "predicate_equivalent_to_key_recurrence_in_this_source": support_equivalent,
        "dictionary_population": "full reconstructed development multiset, including invalid-interval events",
        "pilot_materialization": "compatibility entries were stored only for release keys that occur on temporally valid calibration-pilot targets, and those entries were counted over the full development population",
        "pilot_keys": {schema: len(keys) for schema, keys in cal_keys.items()},
    }
    (OUT / "release_support_geometry_check.json").write_text(json.dumps(geometry_out, indent=2) + "\n", encoding="utf-8")

    print("pfa pilot", flush=True)
    pfa_rows = run_pfa(cal_users, cal_grouped, geometry["bundles"])
    pd.DataFrame(pfa_rows).to_csv(OUT / "pfa_calibration_pilot.csv", index=False)
    print("gru smoke", flush=True)
    pilot = build_pilot(geometry["skill_order"], geometry["atype_order"])
    train_info = train_smoke(pilot, dev_users, geometry["dev_pilot"])
    (OUT / "gru_training_smoke_test.json").write_text(json.dumps(train_info, indent=2) + "\n", encoding="utf-8")
    gru_rows, gru_extra = gru_smoke_eval(pilot, cal_users, cal_grouped, geometry["bundles"], train_info)
    pd.DataFrame(gru_rows).to_csv(OUT / "gru_calibration_pilot.csv", index=False)
    (OUT / "token_provenance_equivalence.json").write_text(json.dumps(gru_extra["provenance"], indent=2) + "\n", encoding="utf-8")

    contract = {
        "primary_clock": "naive CSV civil start_time and end_time, compared as datetime64 microseconds from one mixed-format parse",
        "timezone_mixture": False,
        "valid_interval": "parseable start, parseable end, and end >= start",
        "invalid_interval_policy": "excluded from pilot targets and from chronological history updates; retained in the reconstructed files; retained in the development release-key multiset",
        "history": "H(e) = {p same student, valid interval, end(p) < start(e)}",
        "history_order": ["ascending end", "ascending source_row"],
        "target_order": ["ascending start", "ascending source_row"],
        "target_not_in_own_history": True,
        "pfa_timing": "pre-state is H(e); the candidate completion is ingested only inside a restore-protected score; the realized event joins later histories only after its own end",
        "gru_timing": "prefix outcomes come from H(e); current action tokens are included; current outcome token is withheld; this is an action-conditioned current-event score",
        "not_the_same_estimand": True,
        "within_user_validity_is_not_global_prospective_deployment": True,
        "inversion_note": {
            "count_name_in_tempfor": "inversions_overlapping_previous_interval",
            "reported_value": 2540774,
            "code": "previous.start > next.start AND next.start < previous.end",
            "reading": "This is a start inversion for which the next start is numerically less than the previous end. For a valid previous interval, a start inversion places the next start strictly before the previous interval, so the next start is not inside [previous.start, previous.end]. The 8355 adjacent pairs with previous.start < next.start < previous.end are the adjacent forward containments.",
        },
    }
    (OUT / "event_time_availability_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "status": "SPECIFICATION_ONLY_NOT_EXECUTED",
        "data": contract,
        "archive_sha256": EXPECTED_ZIP,
        "reconstructed_hashes": {split: EXPECTED[split][0] for split in EXPECTED},
        "development_dictionary": "full reconstructed development multiset, including invalid intervals",
        "pfa": {"parameters": "none", "history": "H(e)", "score": "post-ingest mastery at tau 0.80", "update": "all valid events, including unsupported, once end < a later target start"},
        "gru": {
            "architecture": "ActionTokenGRU emb 32 hid 64, historical vocabulary width",
            "target": "current correctness",
            "histories": "chronological H(e), last 40 events, 64 tokens, 32 current actions, no current outcome token",
            "epochs": 2,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "user_cap_per_epoch": 3000,
            "primary_seed": 17,
            "user_selection": "replay NeuralPilot.fit shuffle on the original first-seen development user order among users with at least two reconstructed events; do not reselect users because chronological order changed",
            "adapter": "required; do not expect fit to construct H(e)",
        },
        "evaluation_not_run_in_protocol": ["S0-S3 confirmatory certificates", "false-Stable on the full confirmatory population", "paired schema transitions", "threshold sensitivity", "capacity certificates", "predictive diagnostics", "learner-clustered uncertainty", "file-order comparison tables"],
        "checkpoint": {
            "native": "last supported target in ascending start time, then source_row",
            "justification": "The chronology-valid audit scores a target when its interaction is evaluated, using history completed before its start. The walk order is therefore start order, and the running last supported event in that walk is the checkpoint.",
            "not_used": "completion-time order is the history-update order, not the decision order",
            "matched_event_comparison": "a later comparison must name one shared event identity across schemas rather than compare independently chosen last events",
        },
        "interpretation": "A change in Event NC, capacity NC, or AUC after this protocol is a new empirical result. An input-contract mismatch or a nonzero future-outcome count is an implementation failure. Those are different outcomes.",
    }
    (OUT / "full_replication_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    status = "COMPLETE"
    if equivalence["samples"] < 300 or equivalence["identity_mismatches"] or equivalence["future_or_unfinished_members"] or equivalence["full_population_boundary_violations"] or equivalence["active_set_mismatches"]:
        status = "BLOCKED_TEMPORAL_CONTRACT"
    if not all(geometry_out["key_count_match"].values()) or geometry_out["nesting_violations"]:
        status = "BLOCKED_IMPLEMENTATION_MISMATCH"
    provenance = gru_extra["provenance"]
    pfa_violations = next(row["n"] for row in pfa_rows if row["metric"] == "history_contract_violations")
    if pfa_violations or train_info["history_contract_violations"] or provenance["score_contract_violations"] or provenance["token_id_mismatches"] or provenance["current_outcome_leaks"] or provenance["shared_prefix_failures"] or provenance["history_availability_failures"]:
        status = "BLOCKED_IMPLEMENTATION_MISMATCH"
    integrity = {
        "status": status,
        "preflight_pass": True,
        "history_equivalence": equivalence,
        "key_count_match": geometry_out["key_count_match"],
        "nesting_violations": geometry_out["nesting_violations"],
        "rich_signatures_unique": support_equivalent,
        "pfa_history_violations": next(row["n"] for row in pfa_rows if row["metric"] == "history_contract_violations"),
        "gru_provenance": gru_extra["provenance"],
        "gru_schemas_executed": gru_extra["summary"]["schemas_executed"],
        "no_full_gru_training": True,
        "no_confirmatory_evaluation": True,
        "frozen_artifacts_not_modified": True,
        "pilot_checkpoint_is_non_primary": True,
    }
    (OUT / "integrity_checks.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment": "pipeline",
        "script": "scripts/assistments/chronological_protocol.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "torch": torch.__version__,
        "command": "python3 -u scripts/assistments/chronological_protocol.py",
        "status": status,
        "calibration_pilot_users": 128,
        "development_smoke_users": 128,
        "gru_inference_cap": 2000,
        "candidate_forward_budget": GRU_CANDIDATE_BUDGET,
    }
    (OUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "gru_smoke_summary.json").write_text(json.dumps(gru_extra["summary"], indent=2) + "\n", encoding="utf-8")
    write_checksums()
    print(status, flush=True)


if __name__ == "__main__":
    main()
