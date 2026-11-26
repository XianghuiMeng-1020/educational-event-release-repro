#!/usr/bin/env python3
"""Data-only temporal analysis of ASSISTments 2012-13.

Measures source-order inversions, timestamp semantics, and the history exposure of
the effective consumer inputs. No training, no GRU forward pass, no certificate
replay, and no modification of frozen datasets or checkpoints.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.util
import io
import json
import platform
import random
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from r7_core import (  # noqa: E402
    MAX_EVAL_PREFIX_EVENTS,
    MAX_HIST_TOKENS,
    MAX_SEQ_ACTIONS,
    MAX_TRAIN_EVENTS_PER_USER,
    NEURAL_SEED,
    ActionTokenGRU,
    Event,
    NeuralPilot,
    TransparentActionPFA,
    release_key,
    rich_signature,
    user_bucket,
)

OUT = ROOT / "results/assistments/temporal_forensics"
STAGE = ROOT / "data/raw/assistments/assistments_2012.zip"
RECON = ROOT / "data/processed/assistments"
INNER = "2012-2013-data-with-predictions-4-final.csv"
EXPECTED_ZIP = "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445"
EXPECTED_ZIP_BYTES = 576365408
EXPECTED = {
    "development": ("69b6b7b38738d293651b728326658c1ce517c87661d012bc03ee2ced228e9d70", 190476332, 31753, 4286981),
    "calibration": ("5e8202d6f89b501a1aaea662ba62a265de1e5c168a0bcf92e3762760b4c6fd3d", 43897418, 6804, 888158),
    "confirmatory": ("0193ddd7b26517f1b48591d09d86f5f3ce8b9930a1285006833fce4ea5096f99", 46167386, 6766, 939910),
}
SRCRECON_PAIRS = 6069726
SRCRECON_INVERSIONS = 2540823
SRCRECON_GE_7D = 1801750
DAY_US = 86400 * 1_000_000
HOUR_US = 3600 * 1_000_000
SEVEN_US = 7 * DAY_US
THIRTY_US = 30 * DAY_US
NY = ZoneInfo("America/New_York")
NAT = np.iinfo(np.int64).min
HINT_TYPES = {"hint"}
ATTEMPT_TYPES = {"answer", "attempt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def second_sunday_of_march(year: int) -> datetime:
    day = datetime(year, 3, 1)
    offset = (6 - day.weekday()) % 7
    return datetime(year, 3, 1 + offset + 7, 2, 0, 0)


def first_sunday_of_november(year: int) -> datetime:
    day = datetime(year, 11, 1)
    offset = (6 - day.weekday()) % 7
    return datetime(year, 11, 1 + offset, 2, 0, 0)


def civil_clock_class(clock: datetime) -> str:
    start = second_sunday_of_march(clock.year)
    end = first_sunday_of_november(clock.year)
    if start <= clock < start + timedelta(hours=1):
        return "nonexistent"
    if end - timedelta(hours=1) <= clock < end:
        return "ambiguous"
    if start < clock < end:
        return "daylight"
    return "standard"


def assert_eastern_rules() -> None:
    for year, winter_off, summer_off in ((2012, -5, -4), (2013, -5, -4)):
        winter = datetime(year, 1, 15, 12, 0)
        summer = datetime(year, 7, 15, 12, 0)
        if civil_clock_class(winter) != "standard" or civil_clock_class(summer) != "daylight":
            raise SystemExit("Eastern calendar rule mismatch")
        if winter.replace(tzinfo=NY).utcoffset().total_seconds() != winter_off * 3600:
            raise SystemExit("winter offset mismatch")
        if summer.replace(tzinfo=NY).utcoffset().total_seconds() != summer_off * 3600:
            raise SystemExit("summer offset mismatch")
    ambiguous = datetime(2012, 11, 4, 1, 30)
    early = ambiguous.replace(tzinfo=NY, fold=0).utcoffset().total_seconds()
    late = ambiguous.replace(tzinfo=NY, fold=1).utcoffset().total_seconds()
    if early != -4 * 3600 or late != -5 * 3600:
        raise SystemExit("ambiguous fold offset mismatch")
    missing = datetime(2013, 3, 10, 2, 30)
    if civil_clock_class(missing) != "nonexistent":
        raise SystemExit("spring-forward gap mismatch")


def reference_annotations(prefix_lengths: list[int], current_len: int) -> list[tuple[int, int]]:
    annotated: list[tuple[int, int]] = []
    for index, length in enumerate(prefix_lengths):
        for _ in range(min(MAX_SEQ_ACTIONS, int(length))):
            annotated.append((index, 0))
        annotated.append((index, 1))
    for _ in range(min(MAX_SEQ_ACTIONS, int(current_len))):
        annotated.append((-1, 0))
    return annotated[-MAX_HIST_TOKENS:]


def span_kept(prefix_lengths: list[int], current_len: int) -> tuple[list[int], list[int]]:
    position = 0
    outcome_at: list[int] = []
    end_at: list[int] = []
    for length in prefix_lengths:
        position += min(MAX_SEQ_ACTIONS, int(length)) + 1
        outcome_at.append(position - 1)
        end_at.append(position)
    total = position + min(MAX_SEQ_ACTIONS, int(current_len))
    keep = max(0, total - MAX_HIST_TOKENS)
    any_idx = [i for i, end in enumerate(end_at) if end > keep]
    out_idx = [i for i, outcome in enumerate(outcome_at) if outcome >= keep]
    return any_idx, out_idx


def assert_span_matches_reference() -> None:
    rng = np.random.default_rng(20306)
    for _ in range(300):
        prefix = rng.integers(0, 45, size=int(rng.integers(0, 50))).tolist()
        current = int(rng.integers(0, 45))
        annotated = reference_annotations(prefix, current)
        any_idx, out_idx = span_kept(prefix, current)
        got_any = sorted({i for i, flag in annotated if i >= 0})
        got_out = sorted({i for i, flag in annotated if flag == 1})
        if got_any != any_idx or got_out != out_idx:
            raise SystemExit("token-span logic disagrees with the explicit token list")


def civil_microseconds(series: pd.Series) -> np.ndarray:
    # Fractional seconds in the CSV have 1 to 6 digits. A single inferred
    # format drops the nonmatching rows; mixed parsing keeps every civil time.
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    values = parsed.to_numpy(dtype="datetime64[us]")
    missing = np.isnat(values)
    out = values.astype(np.int64)
    out[missing] = NAT
    return out


def load_light(split: str, with_actions: bool = False) -> pd.DataFrame:
    columns = [
        "source_row",
        "user_id",
        "skill_id",
        "correct",
        "hint_count",
        "attempt_count",
        "start_time",
        "end_time",
        "split",
        "action_types",
    ]
    table = pq.read_table(RECON / f"{split}.parquet", columns=columns)
    lengths = pc.list_value_length(table.column("action_types")).fill_null(0)
    frame = table.select([c for c in columns if c != "action_types"]).to_pandas()
    frame["n_actions"] = lengths.to_numpy(zero_copy_only=False).astype(np.int16)
    if with_actions:
        frame["action_types"] = table.column("action_types").to_pylist()
    frame["start_us"] = civil_microseconds(frame["start_time"])
    frame["end_us"] = civil_microseconds(frame["end_time"])
    frame["user_id"] = frame["user_id"].astype(str)
    frame["skill_id"] = frame["skill_id"].astype(str)
    return frame


def gap_name(microseconds: int) -> str:
    if microseconds <= 0:
        return "not_positive"
    if microseconds <= HOUR_US:
        return "le_1h"
    if microseconds <= DAY_US:
        return "1h_to_1d"
    if microseconds <= SEVEN_US:
        return "1d_to_7d"
    if microseconds <= THIRTY_US:
        return "7d_to_30d"
    return "gt_30d"


def inversion_report(frame: pd.DataFrame, split: str) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]:
    file_order = frame.sort_values("source_row", kind="mergesort")
    file_monotonic = bool(np.all(np.diff(file_order["source_row"].to_numpy()) > 0)) if len(file_order) > 1 else True
    # Within-user file order, not consecutive CSV rows. Users are interleaved,
    # so the next row is usually a different student and is not a chronology pair.
    ordered = frame.sort_values(["user_id", "source_row"], kind="mergesort")
    user = ordered["user_id"].to_numpy()
    start = ordered["start_us"].to_numpy()
    end = ordered["end_us"].to_numpy()
    same = user[:-1] == user[1:]
    left_ok = start[:-1] != NAT
    right_ok = start[1:] != NAT
    valid = same & left_ok & right_ok
    delta = start[:-1] - start[1:]
    inv = valid & (delta > 0)
    tie = valid & (delta == 0)
    end_left = end[:-1] != NAT
    end_right = end[1:] != NAT
    end_valid = same & end_left & end_right
    end_delta = end[:-1] - end[1:]
    end_inv = end_valid & (end_delta > 0)
    end_tie = end_valid & (end_delta == 0)
    magnitudes = delta[inv]
    thresholds = {
        "ge_1h": int(np.sum(magnitudes >= HOUR_US)),
        "gt_1h": int(np.sum(magnitudes > HOUR_US)),
        "ge_1d": int(np.sum(magnitudes >= DAY_US)),
        "gt_1d": int(np.sum(magnitudes > DAY_US)),
        "ge_7d": int(np.sum(magnitudes >= SEVEN_US)),
        "gt_7d": int(np.sum(magnitudes > SEVEN_US)),
        "ge_30d": int(np.sum(magnitudes >= THIRTY_US)),
        "gt_30d": int(np.sum(magnitudes > THIRTY_US)),
    }
    pair_user = user[:-1][same]
    pair_inv = inv[same].astype(np.int8)
    pair_ok = valid[same].astype(np.int8)
    grouped = pd.DataFrame({"user": pair_user, "inv": pair_inv, "ok": pair_ok}).groupby("user", sort=False).sum()
    fractions = (grouped["inv"] / grouped["ok"].where(grouped["ok"] > 0)).dropna().to_numpy()
    bins = (
        ("0", fractions == 0),
        ("(0,0.25]", (fractions > 0) & (fractions <= 0.25)),
        ("(0.25,0.5]", (fractions > 0.25) & (fractions <= 0.5)),
        ("(0.5,0.75]", (fractions > 0.5) & (fractions <= 0.75)),
        ("(0.75,1]", fractions > 0.75),
    )
    dist = [{"split": split, "inversion_fraction_bin": label, "n_users": int(np.sum(mask))} for label, mask in bins]
    forward_overlap = valid & (delta < 0) & (end[:-1] != NAT) & (start[1:] < end[:-1])
    contained_inversion = inv & (end[:-1] != NAT) & (start[1:] < end[:-1])
    # Eastern offset. Unambiguous civil times differ from UTC by 4h or 5h.
    # A one-hour difference cannot reorder events whose civil gap exceeds the
    # offset gap. Ambiguous and nonexistent civil times are counted separately.
    clock = pd.to_datetime(ordered["start_time"], errors="coerce", format="mixed")
    years = clock.dt.year.to_numpy()
    months = clock.dt.month.to_numpy()
    days = clock.dt.day.to_numpy()
    hours = clock.dt.hour.to_numpy()
    transition = {}
    for year in sorted({int(y) for y in years if y == y}):
        if year < 1990 or year > 2100:
            continue
        start_dst = second_sunday_of_march(year)
        end_dst = first_sunday_of_november(year)
        transition[year] = (
            start_dst.month,
            start_dst.day,
            end_dst.month,
            end_dst.day,
        )
    offset = np.full(len(ordered), -1, dtype=np.int8)
    ambiguous = np.zeros(len(ordered), dtype=bool)
    nonexistent = np.zeros(len(ordered), dtype=bool)
    finite = np.isfinite(years.astype(np.float64))
    for year, (sm, sd, em, ed) in transition.items():
        in_year = finite & (years == year)
        nonexistent[in_year & (months == sm) & (days == sd) & (hours == 2)] = True
        ambiguous[in_year & (months == em) & (days == ed) & (hours == 1)] = True
        after_start = (months > sm) | ((months == sm) & (days > sd)) | ((months == sm) & (days == sd) & (hours >= 3))
        before_end = (months < em) | ((months == em) & (days < ed)) | ((months == em) & (days == ed) & (hours < 1))
        daylight = in_year & after_start & before_end
        offset[daylight] = 4
        offset[in_year & ~daylight & ~nonexistent] = 5
        offset[in_year & ambiguous] = 4
    utc = np.full(len(ordered), NAT, dtype=np.int64)
    known = offset > 0
    utc[known] = start[known] + offset[known].astype(np.int64) * HOUR_US
    utc_valid = valid & (utc[:-1] != NAT) & (utc[1:] != NAT) & ~ambiguous[:-1] & ~ambiguous[1:] & ~nonexistent[:-1] & ~nonexistent[1:]
    utc_delta = utc[:-1] - utc[1:]
    utc_inv = utc_valid & (utc_delta > 0)
    civil_inv_compared = inv & utc_valid
    disappeared = civil_inv_compared & ~utc_inv
    appeared = utc_inv & ~inv
    long_flip = disappeared & (delta >= HOUR_US)
    week_flip = disappeared & (delta >= SEVEN_US)
    touched_ambiguous = inv & (ambiguous[:-1] | ambiguous[1:] | nonexistent[:-1] | nonexistent[1:])
    week_touch = touched_ambiguous & (delta >= SEVEN_US)
    metrics = {
        "users": int(ordered["user_id"].nunique()),
        "events": int(len(ordered)),
        "adjacent_pairs": int(np.sum(same)),
        "valid_start_pairs": int(np.sum(valid)),
        "missing_start": int(np.sum(start == NAT)),
        "missing_end": int(np.sum(end == NAT)),
        "start_inversions": int(np.sum(inv)),
        "start_ties": int(np.sum(tie)),
        "inversion_rate": float(np.sum(inv) / np.sum(same)) if np.sum(same) else None,
        "users_with_inversion": int(np.sum(grouped["inv"] > 0)),
        "users_without_inversion": int(np.sum((grouped["ok"] > 0) & (grouped["inv"] == 0))),
        "users_with_no_adjacent_pair": int(ordered["user_id"].nunique() - len(grouped)),
        "max_backward_seconds": float(np.max(magnitudes) / 1_000_000) if len(magnitudes) else None,
        "median_backward_seconds": float(np.median(magnitudes) / 1_000_000) if len(magnitudes) else None,
        "fraction_p50": float(np.quantile(fractions, 0.50)) if len(fractions) else None,
        "fraction_p90": float(np.quantile(fractions, 0.90)) if len(fractions) else None,
        "fraction_p99": float(np.quantile(fractions, 0.99)) if len(fractions) else None,
        "end_inversions": int(np.sum(end_inv)),
        "end_ties": int(np.sum(end_tie)),
        "end_before_start": int(np.sum((start != NAT) & (end != NAT) & (end < start))),
        "forward_overlap_nondecreasing_start": int(np.sum(forward_overlap)),
        "inversions_overlapping_previous_interval": int(np.sum(contained_inversion)),
        "ambiguous_start_timestamps": int(np.sum(ambiguous)),
        "nonexistent_start_timestamps": int(np.sum(nonexistent)),
        "utc_pairs_compared": int(np.sum(utc_valid)),
        "civil_inversions_that_disappear_in_eastern_utc": int(np.sum(disappeared)),
        "utc_inversions_absent_from_civil_time": int(np.sum(appeared)),
        "civil_inversions_of_at_least_1h_that_disappear_in_utc": int(np.sum(long_flip)),
        "civil_inversions_of_at_least_7d_that_disappear_in_utc": int(np.sum(week_flip)),
        "civil_inversions_touching_ambiguous_or_nonexistent_clock": int(np.sum(touched_ambiguous)),
        "civil_inversions_ge_7d_touching_ambiguous_or_nonexistent_clock": int(np.sum(week_touch)),
        "source_row_strictly_increasing": file_monotonic,
        "split_column_mismatches": int(np.sum(ordered["split"].astype(str) != split)),
    }
    metrics.update(thresholds)
    rows = [{"split": split, "metric": key, "value": value} for key, value in metrics.items()]
    return rows, dist, magnitudes.astype(np.int64, copy=False), fractions.astype(np.float64, copy=False)


def exposure_confirmatory(frame: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], list[dict[str, object]], dict[str, int]]:
    gru = Counter()
    pfa = Counter()
    gru_gaps = Counter()
    pfa_gaps = Counter()
    month_events = Counter()
    month_exposed = Counter()
    samples: list[dict[str, object]] = []
    targets = {"short": 20, "long": 20, "long_actions": 20, "inverted": 20, "ordered": 20, "many": 20}
    taken = Counter()
    pfa_failures = 0
    pfa_compared = 0
    replay_every = 160
    for user_index, (uid, group) in enumerate(frame.groupby("user_id", sort=False)):
        group = group.sort_values("source_row", kind="mergesort")
        starts = group["start_us"].to_numpy()
        ends = group["end_us"].to_numpy()
        acts = group["n_actions"].astype(int).tolist()
        skills = group["skill_id"].tolist()
        correct = group["correct"].to_numpy()
        hints = group["hint_count"].to_numpy()
        attempts = group["attempt_count"].to_numpy()
        months = group["start_time"].astype(str).str.slice(0, 7).tolist()
        types = group["action_types"].tolist()
        rows = group["source_row"].to_numpy()
        replay = TransparentActionPFA() if user_index % replay_every == 0 else None
        skill_max: dict[str, int] = {}
        skill_correct: dict[str, int] = {}
        skill_incorrect: dict[str, int] = {}
        skill_hint: dict[str, int] = {}
        skill_n: dict[str, int] = defaultdict(int)
        run_max = NAT
        for t in range(len(group)):
            gru["events"] += 1
            pfa["events"] += 1
            cur = int(starts[t])
            month_events[months[t]] += 1
            if cur == NAT:
                gru["current_time_invalid"] += 1
                pfa["current_time_invalid"] += 1
                continue
            lo = max(0, t - MAX_EVAL_PREFIX_EVENTS)
            future_full = run_max != NAT and run_max > cur
            future_40 = False
            tie_40 = False
            overlap_40 = False
            for i in range(lo, t):
                prev = int(starts[i])
                if prev == NAT:
                    continue
                if prev > cur:
                    future_40 = True
                elif prev == cur:
                    tie_40 = True
                elif int(ends[i]) != NAT and int(ends[i]) > cur:
                    overlap_40 = True
            if future_full:
                gru["future_anywhere_in_source_prefix"] += 1
            if future_40:
                gru["future_in_last_40_source_events"] += 1
            if tie_40:
                gru["tie_in_last_40"] += 1
            if overlap_40 and not future_40:
                gru["overlap_without_future_start_in_last_40"] += 1
            any_idx, out_idx = span_kept(acts[lo:t], acts[t])
            future_token = False
            future_outcome = 0
            outcome_gap = 0
            for local in any_idx:
                prev = int(starts[lo + local])
                if prev != NAT and prev > cur:
                    future_token = True
            for local in out_idx:
                prev = int(starts[lo + local])
                gru["outcome_tokens_in_window"] += 1
                if prev != NAT and prev > cur:
                    future_outcome += 1
                    gru["future_outcome_tokens_in_window"] += 1
                    outcome_gap = max(outcome_gap, prev - cur)
            if future_token:
                gru["future_token_survives_64"] += 1
            if future_outcome:
                gru["future_outcome_survives_64"] += 1
                gru_gaps[gap_name(outcome_gap)] += 1
                month_exposed[months[t]] += 1
            if t < 10:
                gru["events_hist_lt_10"] += 1
                gru["exposed_hist_lt_10"] += int(future_outcome > 0)
            elif t <= 40:
                gru["events_hist_10_to_40"] += 1
                gru["exposed_hist_10_to_40"] += int(future_outcome > 0)
            else:
                gru["events_hist_gt_40"] += 1
                gru["exposed_hist_gt_40"] += int(future_outcome > 0)
            label = "correct" if int(correct[t]) == 1 else "incorrect"
            gru[f"events_{label}"] += 1
            gru[f"exposed_{label}"] += int(future_outcome > 0)
            skill = skills[t]
            seen = skill_max.get(skill)
            if seen is not None and seen > cur:
                pfa["future_same_skill_events"] += 1
                pfa_gaps[gap_name(seen - cur)] += 1
                if skill_correct.get(skill, NAT) > cur:
                    pfa["future_correctness_positive"] += 1
                if skill_incorrect.get(skill, NAT) > cur:
                    pfa["future_correctness_zero"] += 1
                if skill_hint.get(skill, NAT) > cur:
                    pfa["future_positive_hint_count"] += 1
                pfa["future_attempt_counter_increment"] += 1
                if t < 10:
                    pfa["exposed_user_hist_lt_10"] += 1
                elif t <= 40:
                    pfa["exposed_user_hist_10_to_40"] += 1
                else:
                    pfa["exposed_user_hist_gt_40"] += 1
            if t < 10:
                pfa["events_user_hist_lt_10"] += 1
            elif t <= 40:
                pfa["events_user_hist_10_to_40"] += 1
            else:
                pfa["events_user_hist_gt_40"] += 1
            prior_n = skill_n[skill]
            if prior_n < 5:
                pfa["events_same_skill_hist_lt_5"] += 1
                pfa["exposed_same_skill_hist_lt_5"] += int(seen is not None and seen > cur)
            else:
                pfa["events_same_skill_hist_ge_5"] += 1
                pfa["exposed_same_skill_hist_ge_5"] += int(seen is not None and seen > cur)
            if skill == "UNK":
                pfa["events_skill_unk"] += 1
                pfa["exposed_skill_unk"] += int(seen is not None and seen > cur)
            else:
                pfa["events_skill_named"] += 1
                pfa["exposed_skill_named"] += int(seen is not None and seen > cur)
            if future_full:
                pfa["future_any_skill_in_source_prefix"] += 1
            if replay is not None:
                observed_n = float(replay.skill_stats[skill]["n"])
                if observed_n != float(prior_n):
                    pfa_failures += 1
                pfa_compared += 1
                replay.update(
                    Event(
                        str(uid),
                        "p",
                        skill,
                        int(correct[t]),
                        int(hints[t]),
                        int(attempts[t]),
                        "0",
                        tuple(str(tok).lower() for tok in types[t]),
                        tuple(),
                        "confirmatory",
                    )
                )
            if seen is None or cur > seen:
                skill_max[skill] = cur
            if int(correct[t]) == 1 and cur > skill_correct.get(skill, NAT):
                skill_correct[skill] = cur
            if int(correct[t]) == 0 and cur > skill_incorrect.get(skill, NAT):
                skill_incorrect[skill] = cur
            if int(hints[t]) > 0 and cur > skill_hint.get(skill, NAT):
                skill_hint[skill] = cur
            skill_n[skill] = prior_n + 1
            if run_max == NAT or cur > run_max:
                run_max = cur
            kind = None
            if taken["short"] < targets["short"] and t < 3:
                kind = "short"
            elif taken["many"] < targets["many"] and t >= 80:
                kind = "many"
            elif taken["long"] < targets["long"] and t > 40:
                kind = "long"
            elif taken["long_actions"] < targets["long_actions"] and acts[t] > 16 and t > 0:
                kind = "long_actions"
            elif taken["inverted"] < targets["inverted"] and future_40:
                kind = "inverted"
            elif taken["ordered"] < targets["ordered"] and t >= 5 and not future_full:
                kind = "ordered"
            if kind is not None:
                taken[kind] += 1
                samples.append(
                    {
                        "kind": kind,
                        "prefix_types": [tuple(str(tok).lower() for tok in types[i]) for i in range(lo, t)],
                        "current_types": tuple(str(tok).lower() for tok in types[t]),
                        "prefix_correct": [int(correct[i]) for i in range(lo, t)],
                        "current_correct": int(correct[t]),
                        "prefix_skills": skills[lo:t],
                        "current_skill": skill,
                        "future_flags": [int(starts[i]) != NAT and int(starts[i]) > cur for i in range(lo, t)],
                        "formula_future_outcomes": future_outcome,
                        "formula_future_tokens": int(future_token),
                    }
                )
    for key, value in gru_gaps.items():
        gru[f"gap:{key}"] = value
    for key, value in pfa_gaps.items():
        pfa[f"gap:{key}"] = value
    for key, value in month_events.items():
        gru[f"month_events:{key}"] = value
        gru[f"month_exposed:{key}"] = month_exposed[key]
    pfa_check = {"pfa_state_counts_compared": pfa_compared, "pfa_state_count_failures": pfa_failures, "sample_kinds": dict(taken)}
    return dict(gru), dict(pfa), samples, pfa_check


def training_exposure(frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame.sort_values("source_row", kind="mergesort")
    counts = ordered.groupby("user_id", sort=False).size()
    eligible: list[str] = []
    seen = set()
    for uid in ordered["user_id"].tolist():
        if uid not in seen:
            seen.add(uid)
            if int(counts[uid]) >= 2:
                eligible.append(uid)
    by_user = frame.sort_values(["user_id", "source_row"], kind="mergesort").groupby("user_id", sort=False)
    slices = {}
    for uid, group in by_user:
        slices[str(uid)] = (
            group["start_us"].to_numpy(),
            group["n_actions"].astype(int).tolist(),
            group["source_row"].to_numpy(),
        )
    users = eligible.copy()
    random.seed(NEURAL_SEED)
    selected = []
    for _ in range(2):
        random.shuffle(users)
        selected.append(users[:3000].copy())
    examples = exposed = future_token_examples = outcome_tokens = future_outcomes = empty_prefix = 0
    gaps = Counter()
    unique_exposed = set()
    for epoch, cohort in enumerate(selected):
        for uid in cohort:
            starts, acts, rows = slices[uid]
            hist_start = max(0, len(starts) - MAX_TRAIN_EVENTS_PER_USER)
            for t in range(hist_start, len(starts)):
                examples += 1
                local_t = t - hist_start
                if local_t == 0:
                    empty_prefix += 1
                cur = int(starts[t])
                any_idx, out_idx = span_kept(acts[hist_start:t], acts[t])
                outcome_tokens += len(out_idx)
                hit = False
                token_hit = False
                max_gap = 0
                if cur != NAT:
                    for local in out_idx:
                        prev = int(starts[hist_start + local])
                        if prev != NAT and prev > cur:
                            future_outcomes += 1
                            hit = True
                            max_gap = max(max_gap, prev - cur)
                    for local in any_idx:
                        prev = int(starts[hist_start + local])
                        if prev != NAT and prev > cur:
                            token_hit = True
                if token_hit:
                    future_token_examples += 1
                if hit:
                    exposed += 1
                    gaps[gap_name(max_gap)] += 1
                    unique_exposed.add((uid, int(rows[t]), epoch))
    return {
        "selection_rule": "replay of NeuralPilot.fit global random.seed(17), two shuffles, first 3000 users with at least two file-order events, last 40 file-order events; no optimizer and no forward pass",
        "development_events_available_to_fit": int(len(frame)),
        "development_users": int(frame["user_id"].nunique()),
        "users_with_at_least_two_events": len(eligible),
        "epoch1_selected_users": len(selected[0]),
        "epoch2_selected_users": len(selected[1]),
        "union_selected_users": len(set(selected[0]) | set(selected[1])),
        "intersection_selected_users": len(set(selected[0]) & set(selected[1])),
        "training_examples_with_epoch_multiplicity": examples,
        "examples_with_empty_prefix": empty_prefix,
        "examples_with_future_token_in_window": future_token_examples,
        "examples_with_future_outcome_in_window": exposed,
        "outcome_tokens_in_window": outcome_tokens,
        "future_outcome_tokens_in_window": future_outcomes,
        "future_outcome_token_fraction": future_outcomes / outcome_tokens if outcome_tokens else None,
        "gap_bins_json": json.dumps(gaps),
        "checkpoint_meta_epochs": 2,
        "checkpoint_meta_user_cap": 3000,
        "checkpoint_meta_max_events": 40,
        "checkpoint_meta_train_events": 4286981,
        "not_retrained": True,
        "unique_exposed_example_epochs": len(unique_exposed),
    }


def validate_token_samples(samples: list[dict[str, object]]) -> dict[str, int]:
    pilot = NeuralPilot()
    for sample in samples:
        for skill in list(sample["prefix_skills"]) + [sample["current_skill"]]:
            if skill not in pilot.skill2i:
                pilot.skill2i[skill] = len(pilot.skill2i)
        for seq in list(sample["prefix_types"]) + [sample["current_types"]]:
            for token in seq:
                if token not in pilot.atype2i:
                    pilot.atype2i[token] = len(pilot.atype2i)
    pilot.model = ActionTokenGRU(max(len(pilot.skill2i) - 1, 1), max(len(pilot.atype2i) - 1, 1))
    failures = 0
    provenance_failures = 0
    for sample in samples:
        prefix = []
        for skill, correct, types in zip(sample["prefix_skills"], sample["prefix_correct"], sample["prefix_types"], strict=True):
            prefix.append(Event("u", "p", str(skill), int(correct), 0, 1, "0", tuple(types), tuple(), "confirmatory"))
        current = Event("u", "p", str(sample["current_skill"]), int(sample["current_correct"]), 0, 1, "0", tuple(sample["current_types"]), tuple(), "confirmatory")
        _skills, atypes = pilot._tokens(prefix, current, False)
        annotated = reference_annotations([len(ev.action_types) for ev in prefix], len(current.action_types))
        if len(atypes) != len(annotated):
            failures += 1
            continue
        for (index, flag), atype in zip(annotated, atypes, strict=True):
            is_outcome = atype in {pilot.model.outcome0, pilot.model.outcome1}
            if bool(flag) != is_outcome:
                failures += 1
                break
        formula = 0
        for index, flag in annotated:
            if flag == 1 and index >= 0 and sample["future_flags"][index]:
                formula += 1
        if formula != sample["formula_future_outcomes"]:
            provenance_failures += 1
    return {
        "gru_token_samples": len(samples),
        "gru_token_builder_failures": failures,
        "gru_outcome_provenance_failures": provenance_failures,
    }


def clock_and_fields() -> dict[str, object]:
    years = Counter()
    hour_hist = Counter()
    action_order = Counter()
    fields = Counter()
    last_civil: dict[str, int] = {}
    last_epoch: dict[str, int | None] = {}
    last_user = None
    switches = 0
    rows = 0
    monotonic_by_split = {}
    for split in ("development", "calibration", "confirmatory"):
        previous_source = -1
        monotonic = True
        parquet = pq.ParquetFile(RECON / f"{split}.parquet")
        for batch in parquet.iter_batches(
            batch_size=100000,
            columns=["source_row", "user_id", "start_time", "end_time", "hint_count", "attempt_count", "first_action", "action_types", "action_timestamps"],
        ):
            data = batch.to_pylist()
            for row in data:
                rows += 1
                source = int(row["source_row"])
                if source <= previous_source:
                    monotonic = False
                previous_source = source
                uid = str(row["user_id"])
                if last_user is not None and uid != last_user:
                    switches += 1
                last_user = uid
                start = pd.to_datetime(row["start_time"], errors="coerce")
                end = pd.to_datetime(row["end_time"], errors="coerce")
                if not pd.isna(start):
                    years[int(start.year)] += 1
                types = [str(tok).lower() for tok in row["action_types"]]
                stamps = row["action_timestamps"]
                hint_like = sum(1 for tok in types if tok in HINT_TYPES or tok.startswith("hint"))
                attempt_like = sum(1 for tok in types if tok in ATTEMPT_TYPES)
                fields["events"] += 1
                fields["hint_equal"] += int(int(row["hint_count"]) == hint_like)
                fields["hint_disagree"] += int(int(row["hint_count"]) != hint_like)
                fields["attempt_equal"] += int(int(row["attempt_count"]) == attempt_like)
                fields["attempt_disagree"] += int(int(row["attempt_count"]) != attempt_like)
                first = types[0] if types else ""
                fields["first_action_literal_equals_first_token"] += int(str(row["first_action"]).strip().lower() == first)
                numeric = []
                for stamp in stamps:
                    try:
                        numeric.append(float(stamp))
                    except (TypeError, ValueError):
                        continue
                has_epoch = any(value > 1e11 for value in numeric)
                has_small = any(0 <= value < 1e8 for value in numeric)
                if has_epoch and has_small:
                    fields["mixed_epoch_and_small_action_timestamps"] += 1
                epoch = None
                for tok, stamp in zip(types, stamps, strict=True):
                    if tok != "start":
                        continue
                    try:
                        epoch = float(stamp)
                    except (TypeError, ValueError):
                        fields["start_action_unparsed"] += 1
                    break
                else:
                    fields["start_action_absent"] += 1
                civil = None if pd.isna(start) else int(pd.Timestamp(start).value // 1000)
                if epoch is not None and epoch > 1e11 and civil is not None:
                    utc = datetime.fromtimestamp(epoch / 1000.0, timezone.utc).replace(tzinfo=None)
                    delta_seconds = (pd.Timestamp(utc) - pd.Timestamp(start)).total_seconds()
                    hour_hist[str(round(delta_seconds / 3600))] += 1
                    if min(abs(delta_seconds - 14400), abs(delta_seconds - 18000)) <= 1:
                        fields["offset_4h_or_5h_within_1s"] += 1
                    else:
                        fields["offset_other"] += 1
                    epoch_us = int(round(epoch * 1000))
                else:
                    epoch_us = None
                    if epoch is not None and epoch <= 1e11:
                        fields["start_action_not_epoch_ms"] += 1
                prev_civil = last_civil.get(uid)
                prev_epoch = last_epoch.get(uid)
                if prev_civil is not None and civil is not None and prev_epoch is not None and epoch_us is not None:
                    action_order["pairs_with_both_epoch_starts"] += 1
                    civil_back = prev_civil > civil
                    epoch_back = prev_epoch > epoch_us
                    if civil_back:
                        action_order["civil_inversions_with_epoch_pair"] += 1
                        if not epoch_back:
                            action_order["civil_inversion_not_confirmed_by_epoch"] += 1
                            if prev_civil - civil >= DAY_US:
                                action_order["unconfirmed_civil_inversion_ge_1d"] += 1
                            if prev_civil - civil >= SEVEN_US:
                                action_order["unconfirmed_civil_inversion_ge_7d"] += 1
                        elif prev_civil - civil >= SEVEN_US:
                            action_order["epoch_confirmed_civil_inversion_ge_7d"] += 1
                    if epoch_back and not civil_back:
                        action_order["epoch_inversion_without_civil_inversion"] += 1
                if civil is not None:
                    last_civil[uid] = civil
                else:
                    last_civil.pop(uid, None)
                if epoch_us is not None:
                    last_epoch[uid] = epoch_us
                else:
                    last_epoch[uid] = None
                _ = end
        monotonic_by_split[split] = monotonic
    return {
        "rows_scanned": rows,
        "source_row_strictly_increasing_within_each_split_file": monotonic_by_split,
        "user_switches_within_concatenated_split_files": switches,
        "year_counts": dict(years),
        "csv_minus_epoch_rounded_hours": dict(hour_hist),
        "field_counts": dict(fields),
        "action_epoch_order": dict(action_order),
        "note": "Adjacent action-start epochs are absolute Unix times. They are compared with adjacent naive CSV start times. A large civil inversion that the epoch also inverts is not a time-zone labeling artifact.",
    }


def user_funnel(retained_rows: set[int], retained_users: set[str]) -> list[dict[str, object]]:
    path = ROOT / "scripts/assistments/source_reconstruction.py"
    spec = importlib.util.spec_from_file_location("srcrecon_assist_source_recovery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reason_rows = Counter()
    short_events = Counter()
    profiles: dict[str, Counter] = defaultdict(Counter)
    raw_users = set()
    raw_rows = 0
    with zipfile.ZipFile(STAGE) as archive, archive.open(INNER) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        for source_row, row in enumerate(csv.DictReader(text)):
            raw_rows += 1
            uid = "" if module.blank(row.get("user_id")) else str(row.get("user_id")).strip()
            if uid:
                raw_users.add(uid)
            if source_row in retained_rows:
                continue
            reason, record, _flags, _reasons = module.classify_row(source_row, row)
            if reason is None and record is not None:
                reason_rows["accepted_then_removed_by_minimum_event_rule"] += 1
                if uid:
                    short_events[uid] += 1
            else:
                reason_rows[reason or "unclassified"] += 1
                if uid:
                    profiles[uid][reason or "unclassified"] += 1
    zero_users = [uid for uid in raw_users if uid not in retained_users and uid not in short_events]
    pure = Counter()
    for uid in zero_users:
        keys = tuple(sorted(profiles[uid]))
        pure["|".join(keys) if keys else "no_rejected_rows"] += 1
    rows = [
        {"stage": "raw_csv", "rows": raw_rows, "distinct_users": len(raw_users), "users_eliminated_entirely": 0, "users_remaining": len(raw_users), "note": "distinct non-empty user ids"},
        {"stage": "accepted_by_row_predicate", "rows": len(retained_rows) + int(reason_rows["accepted_then_removed_by_minimum_event_rule"]), "distinct_users": len(retained_users) + len(short_events), "users_eliminated_entirely": len(zero_users), "users_remaining": len(retained_users) + len(short_events), "note": "users with at least one row accepted before the three-event rule"},
        {"stage": "minimum_three_events", "rows": len(retained_rows), "distinct_users": len(retained_users), "users_eliminated_entirely": len(short_events), "users_remaining": len(retained_users), "note": "users with one or two accepted rows are removed with all of their rows"},
    ]
    for reason, count in sorted(reason_rows.items()):
        rows.append({"stage": f"nonretained_row_reason:{reason}", "rows": int(count), "distinct_users": "", "users_eliminated_entirely": "", "users_remaining": "", "note": "exclusive row reason among rows absent from the final parquet"})
    for profile, count in sorted(pure.items()):
        rows.append({"stage": f"users_with_zero_accepted_rows:{profile}", "rows": "", "distinct_users": int(count), "users_eliminated_entirely": int(count), "users_remaining": "", "note": "a user is counted once, under the combination of row reasons that removed every one of that user's rows"})
    short_sizes = Counter(short_events.values())
    for size, count in sorted(short_sizes.items()):
        rows.append({"stage": f"users_removed_for_event_count_{size}", "rows": int(size) * int(count), "distinct_users": int(count), "users_eliminated_entirely": int(count), "users_remaining": "", "note": "accepted rows before the minimum-event rule"})
    return rows


def support_checkpoints(conf_users: set[str]) -> list[dict[str, object]]:
    state: dict[str, dict[tuple, list]] = {schema: {} for schema in ("S0", "S1", "S2", "S3")}
    supported: dict[str, set[tuple]] = {schema: set() for schema in state}
    same_signature_repeats = Counter()
    parquet = pq.ParquetFile(RECON / "development.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=100000):
        for row in batch.to_pylist():
            seen += 1
            event = Event(
                user_id=str(row["user_id"]),
                problem_id=str(row["problem_id"]),
                skill_id=str(row["skill_id"]),
                correct=int(row["correct"]),
                hint_count=int(row["hint_count"]),
                attempt_count=int(row["attempt_count"]),
                first_action=str(row["first_action"]),
                action_types=tuple(str(tok).lower() for tok in row["action_types"]),
                action_timestamps=tuple("" if tok is None else str(tok) for tok in row["action_timestamps"]),
                split=str(row["split"]),
            )
            digest = hashlib.blake2s(repr(rich_signature(event)).encode("utf-8"), digest_size=16).digest()
            for schema in state:
                key = release_key(event, schema)
                if key in supported[schema]:
                    continue
                slot = state[schema].get(key)
                if slot is None:
                    state[schema][key] = [1, event.user_id, digest]
                elif event.user_id != slot[1] or digest != slot[2]:
                    supported[schema].add(key)
                    del state[schema][key]
                else:
                    slot[0] += 1
                    same_signature_repeats[schema] += 1
        if seen % 500000 == 0:
            print(f"support_scan {seen}", flush=True)
    for schema in state:
        state[schema].clear()
    gc.collect()
    source_last = {schema: {} for schema in supported}
    chrono_last = {schema: {} for schema in supported}
    supported_events = Counter()
    parquet = pq.ParquetFile(RECON / "confirmatory.parquet")
    for batch in parquet.iter_batches(batch_size=80000):
        for row in batch.to_pylist():
            event = Event(
                user_id=str(row["user_id"]),
                problem_id=str(row["problem_id"]),
                skill_id=str(row["skill_id"]),
                correct=int(row["correct"]),
                hint_count=int(row["hint_count"]),
                attempt_count=int(row["attempt_count"]),
                first_action=str(row["first_action"]),
                action_types=tuple(str(tok).lower() for tok in row["action_types"]),
                action_timestamps=tuple("" if tok is None else str(tok) for tok in row["action_timestamps"]),
                split=str(row["split"]),
            )
            start = civil_microseconds(pd.Series([row["start_time"]]))[0]
            source = int(row["source_row"])
            for schema in supported:
                if release_key(event, schema) not in supported[schema]:
                    continue
                supported_events[schema] += 1
                current = source_last[schema].get(event.user_id)
                if current is None or source > current[0]:
                    source_last[schema][event.user_id] = (source, int(start))
                current_time = chrono_last[schema].get(event.user_id)
                candidate = (int(start), source)
                if current_time is None or candidate > current_time:
                    chrono_last[schema][event.user_id] = candidate
    rows = []
    for schema in supported:
        same = earlier_1d = earlier_7d = earlier_30d = missing_time = 0
        gaps = []
        for uid in conf_users:
            source = source_last[schema].get(uid)
            chrono = chrono_last[schema].get(uid)
            if source is None or chrono is None:
                continue
            if source[1] == NAT or chrono[0] == NAT:
                missing_time += 1
                continue
            if source[0] == chrono[1]:
                same += 1
            gap = chrono[0] - source[1]
            gaps.append(gap)
            if gap >= DAY_US:
                earlier_1d += 1
            if gap >= SEVEN_US:
                earlier_7d += 1
            if gap >= THIRTY_US:
                earlier_30d += 1
        with_ckpt = len(source_last[schema])
        comparable = with_ckpt - missing_time
        rows.append(
            {
                "consumer": "shared_support_predicate_for_TransparentActionPFA_and_ActionTokenGRU",
                "schema": schema,
                "support_source": "reconstructed_development_dictionary_not_lfs_pickle",
                "supported_development_keys": len(supported[schema]),
                "same_signature_repeat_events": int(same_signature_repeats[schema]),
                "confirmatory_supported_events": int(supported_events[schema]),
                "confirmatory_users": len(conf_users),
                "users_with_supported_checkpoint": with_ckpt,
                "users_without_supported_checkpoint": len(conf_users) - with_ckpt,
                "source_checkpoint_equals_chronological_last": same,
                "users_with_different_checkpoint": comparable - same,
                "fraction_different_among_users_with_checkpoint": (comparable - same) / comparable if comparable else None,
                "source_checkpoint_earlier_by_at_least_1d": earlier_1d,
                "source_checkpoint_earlier_by_at_least_7d": earlier_7d,
                "source_checkpoint_earlier_by_at_least_30d": earlier_30d,
                "median_chrono_minus_source_seconds": float(np.median(gaps) / 1_000_000) if gaps else None,
                "missing_checkpoint_time": missing_time,
                "note": "Identity and time comparison only. No capacity certificate or risk interval was computed.",
            }
        )
    return rows


def lineage_sample() -> dict[str, object]:
    parquet = pq.ParquetFile(RECON / "confirmatory.parquet")
    batch = next(parquet.iter_batches(batch_size=200))
    mismatches = 0
    first_types = Counter()
    action_container = Counter()
    for row in batch.to_pylist():
        action_container[type(row["action_types"]).__name__] += 1
        first_types[type(row["first_action"]).__name__] += 1
        event = Event(
            user_id=str(row["user_id"]),
            problem_id=str(row["problem_id"]),
            skill_id=str(row["skill_id"]),
            correct=int(row["correct"]),
            hint_count=int(row["hint_count"]),
            attempt_count=int(row["attempt_count"]),
            first_action=str(row["first_action"]),
            action_types=tuple(str(tok).lower() for tok in row["action_types"]),
            action_timestamps=tuple("" if tok is None else str(tok) for tok in row["action_timestamps"]),
            split=str(row["split"]),
        )
        if release_key(event, "S3")[6] != tuple(str(tok).lower() for tok in row["action_types"]):
            mismatches += 1
        if user_bucket(event.user_id) != "confirmatory":
            mismatches += 1
    return {
        "confirmatory_sample_rows": batch.num_rows,
        "release_key_or_bucket_mismatches": mismatches,
        "action_types_python_types": dict(action_container),
        "first_action_python_types": dict(first_types),
        "contract": "Parquet preserves source_row order. release_key lowercases action tokens and stores them as a tuple. first_action remains the CSV field, not the first parsed token. This sample is not a claim of byte identity with the missing LFS event pickle.",
    }


def records_from_counter(prefix: str, counts: dict[str, int], denominator_key: str) -> list[dict[str, object]]:
    events = counts.get(denominator_key)
    rows = []
    for key, value in counts.items():
        if key == "future_outcome_tokens_in_window":
            denom = counts.get("outcome_tokens_in_window")
        elif key.startswith("month_exposed:"):
            denom = counts.get("month_events:" + key.split(":", 1)[1])
        elif key.startswith("exposed_"):
            denom = counts.get("events_" + key[len("exposed_") :], events)
        elif key.startswith("gap:"):
            denom = counts.get("future_outcome_survives_64", counts.get("future_same_skill_events"))
        elif key == denominator_key:
            denom = value
        else:
            denom = events
        fraction = value / denom if isinstance(value, int) and isinstance(denom, int) and denom else None
        rows.append({"population": prefix, "quantity": key, "n": value, "denominator": denom, "fraction": fraction})
    return rows


def write_checksums() -> None:
    files = []
    for path in sorted(item for item in OUT.iterdir() if item.is_file() and item.name != "output_checksums.json"):
        files.append(
            {
                "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {"algorithm": "sha256", "excluded": ["results/assistments/temporal_forensics/output_checksums.json"], "n_files": len(files), "files": files}
    (OUT / "output_checksums.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def metric_map(rows: list[dict[str, object]], split: str) -> dict[str, object]:
    return {row["metric"]: row["value"] for row in rows if row["split"] == split}


def main(reuse_slow: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_eastern_rules()
    assert_span_matches_reference()
    zip_digest = sha256_file(STAGE)
    preflight = {
        "zip_sha256": zip_digest,
        "zip_bytes": STAGE.stat().st_size,
        "zip_match": zip_digest == EXPECTED_ZIP and STAGE.stat().st_size == EXPECTED_ZIP_BYTES,
        "inner_member": INNER,
        "parquet": {},
    }
    for split, (digest, size, users, events) in EXPECTED.items():
        path = RECON / f"{split}.parquet"
        observed = sha256_file(path)
        preflight["parquet"][split] = {
            "sha256": observed,
            "bytes": path.stat().st_size,
            "expected_users": users,
            "expected_events": events,
            "match": observed == digest and path.stat().st_size == size,
        }
    (OUT / "srcrecon_source_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if not preflight["zip_match"] or any(not item["match"] for item in preflight["parquet"].values()):
        (OUT / "integrity_checks.json").write_text(json.dumps({"status": "BLOCKED_SRCRECON_SOURCE_INTEGRITY"}, indent=2) + "\n", encoding="utf-8")
        write_checksums()
        raise SystemExit(2)

    inversion_rows: list[dict[str, object]] = []
    dist_rows: list[dict[str, object]] = []
    magnitude_parts: list[np.ndarray] = []
    fraction_parts: list[np.ndarray] = []
    retained_rows: set[int] = set()
    retained_users: set[str] = set()
    bucket_mismatches = Counter()
    train = None
    conf_users: set[str] = set()
    for split in ("development", "calibration", "confirmatory"):
        print(f"load {split}", flush=True)
        frame = load_light(split, with_actions=False)
        if frame["user_id"].nunique() != EXPECTED[split][2] or len(frame) != EXPECTED[split][3]:
            raise SystemExit(f"population mismatch {split}")
        retained_rows.update(int(v) for v in frame["source_row"].tolist())
        users = set(frame["user_id"].tolist())
        retained_users.update(users)
        if split == "confirmatory":
            conf_users = users
        for uid in frame["user_id"].drop_duplicates():
            if user_bucket(str(uid)) != split:
                bucket_mismatches[split] += 1
        print(f"inversions {split}", flush=True)
        rows, dist, magnitudes, fractions = inversion_report(frame, split)
        inversion_rows.extend(rows)
        dist_rows.extend(dist)
        magnitude_parts.append(magnitudes)
        fraction_parts.append(fractions)
        if split == "development":
            print("training selection", flush=True)
            train = training_exposure(frame)
        del frame
        gc.collect()
    combined = Counter()
    numeric_metrics = [
        "users",
        "events",
        "adjacent_pairs",
        "valid_start_pairs",
        "missing_start",
        "missing_end",
        "start_inversions",
        "start_ties",
        "users_with_inversion",
        "users_without_inversion",
        "users_with_no_adjacent_pair",
        "end_inversions",
        "end_ties",
        "end_before_start",
        "forward_overlap_nondecreasing_start",
        "inversions_overlapping_previous_interval",
        "ambiguous_start_timestamps",
        "nonexistent_start_timestamps",
        "utc_pairs_compared",
        "civil_inversions_that_disappear_in_eastern_utc",
        "utc_inversions_absent_from_civil_time",
        "civil_inversions_of_at_least_1h_that_disappear_in_utc",
        "civil_inversions_of_at_least_7d_that_disappear_in_utc",
        "civil_inversions_touching_ambiguous_or_nonexistent_clock",
        "civil_inversions_ge_7d_touching_ambiguous_or_nonexistent_clock",
        "ge_1h",
        "gt_1h",
        "ge_1d",
        "gt_1d",
        "ge_7d",
        "gt_7d",
        "ge_30d",
        "gt_30d",
    ]
    for metric in numeric_metrics:
        combined[metric] = sum(int(metric_map(inversion_rows, split)[metric]) for split in EXPECTED)
    pairs = combined["adjacent_pairs"]
    combined_rows = [{"split": "combined", "metric": key, "value": int(value)} for key, value in combined.items()]
    combined_rows.append({"split": "combined", "metric": "inversion_rate", "value": combined["start_inversions"] / pairs})
    combined_rows.append({"split": "combined", "metric": "srcrecon_adjacent_pairs_match", "value": pairs == SRCRECON_PAIRS})
    combined_rows.append({"split": "combined", "metric": "srcrecon_start_inversions_match", "value": combined["start_inversions"] == SRCRECON_INVERSIONS})
    combined_rows.append({"split": "combined", "metric": "srcrecon_ge_7d_match", "value": combined["ge_7d"] == SRCRECON_GE_7D})
    combined_rows.append({"split": "combined", "metric": "adjacent_pairs_equal_events_minus_users", "value": pairs == combined["events"] - combined["users"]})
    all_magnitudes = np.concatenate(magnitude_parts) if magnitude_parts else np.array([], dtype=np.int64)
    all_fractions = np.concatenate(fraction_parts) if fraction_parts else np.array([], dtype=np.float64)
    combined_rows.append({"split": "combined", "metric": "max_backward_seconds", "value": float(np.max(all_magnitudes) / 1_000_000) if len(all_magnitudes) else None})
    combined_rows.append({"split": "combined", "metric": "median_backward_seconds", "value": float(np.median(all_magnitudes) / 1_000_000) if len(all_magnitudes) else None})
    if len(all_fractions):
        combined_rows.append({"split": "combined", "metric": "fraction_p50", "value": float(np.quantile(all_fractions, 0.50))})
        combined_rows.append({"split": "combined", "metric": "fraction_p90", "value": float(np.quantile(all_fractions, 0.90))})
        combined_rows.append({"split": "combined", "metric": "fraction_p99", "value": float(np.quantile(all_fractions, 0.99))})
    for label in ("0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1]"):
        total_users = sum(int(row["n_users"]) for row in dist_rows if row["inversion_fraction_bin"] == label)
        dist_rows.append({"split": "combined", "inversion_fraction_bin": label, "n_users": total_users})
    inversion_rows.extend(combined_rows)
    pd.DataFrame(inversion_rows).to_csv(OUT / "source_order_inversions.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUT / "inversion_by_user_distribution.csv", index=False)

    print("confirmatory exposure", flush=True)
    confirmatory = load_light("confirmatory", with_actions=True)
    gru, pfa, samples, pfa_check = exposure_confirmatory(confirmatory)
    token_check = validate_token_samples(samples)
    del confirmatory
    gc.collect()
    pd.DataFrame(records_from_counter("confirmatory", gru, "events")).to_csv(OUT / "effective_gru_prefix_exposure.csv", index=False)
    pd.DataFrame(records_from_counter("confirmatory", pfa, "events")).to_csv(OUT / "pfa_history_exposure.csv", index=False)
    pd.DataFrame([train]).to_csv(OUT / "gru_training_exposure.csv", index=False)

    if reuse_slow:
        required = (
            "timestamp_semantics.json",
            "user_count_funnel_reconciliation.csv",
            "capacity_checkpoint_time_comparison.csv",
        )
        missing = [name for name in required if not (OUT / name).exists()]
        if missing:
            raise SystemExit(f"cannot reuse missing outputs: {missing}")
        print("reuse clock, funnel, and reconstructed support outputs", flush=True)
    else:
        print("clock and fields", flush=True)
        clocks = clock_and_fields()
        (OUT / "timestamp_semantics.json").write_text(json.dumps(clocks, indent=2) + "\n", encoding="utf-8")
        print("user funnel", flush=True)
        funnel = user_funnel(retained_rows, retained_users)
        pd.DataFrame(funnel).to_csv(OUT / "user_count_funnel_reconciliation.csv", index=False)
        print("support checkpoints", flush=True)
        checkpoints = support_checkpoints(conf_users)
        pd.DataFrame(checkpoints).to_csv(OUT / "capacity_checkpoint_time_comparison.csv", index=False)
    lineage = lineage_sample()
    lineage["bucket_mismatches"] = dict(bucket_mismatches)
    lineage["retained_source_rows"] = len(retained_rows)
    lineage["retained_users"] = len(retained_users)
    (OUT / "reconstruction_lineage_audit.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")

    integrity = {
        "adjacent_pairs_identity": pairs == combined["events"] - combined["users"],
        "split_pair_sum": pairs,
        "expected_pairs": SRCRECON_PAIRS,
        "srcrecon_inversion_reproduction": {
            "pairs": pairs == SRCRECON_PAIRS,
            "start_inversions": combined["start_inversions"] == SRCRECON_INVERSIONS,
            "ge_7d": combined["ge_7d"] == SRCRECON_GE_7D,
        },
        "threshold_nesting": bool(
            combined["gt_30d"] <= combined["gt_7d"] <= combined["gt_1d"] <= combined["gt_1h"] <= combined["start_inversions"]
            and combined["ge_30d"] <= combined["ge_7d"] <= combined["ge_1d"] <= combined["ge_1h"] <= combined["start_inversions"]
        ),
        "inversions_do_not_exceed_pairs": bool(combined["start_inversions"] <= pairs),
        "token_equivalence": token_check,
        "pfa_replay": pfa_check,
        "bucket_mismatches": dict(bucket_mismatches),
        "no_training": True,
        "no_neural_forward": True,
        "no_certificate_replay": True,
        "frozen_results_not_overwritten": True,
    }
    status = "COMPLETE"
    if token_check["gru_token_builder_failures"] or token_check["gru_outcome_provenance_failures"] or pfa_check["pfa_state_count_failures"]:
        status = "TEMPFOR_PARTIAL_EFFECTIVE_INPUT_EXPOSURE"
    if not all(integrity["srcrecon_inversion_reproduction"].values()):
        status = "TEMPFOR_PARTIAL_EFFECTIVE_INPUT_EXPOSURE"
    integrity["status"] = status
    (OUT / "integrity_checks.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment": "pipeline",
        "script": "scripts/assistments/temporal_forensics.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "command": "python3 -u scripts/assistments/temporal_forensics.py" + (" --reuse-clock-funnel-support" if reuse_slow else ""),
        "reused_outputs": ["timestamp_semantics.json", "user_count_funnel_reconciliation.csv", "capacity_checkpoint_time_comparison.csv"] if reuse_slow else [],
        "reuse_reason": "The clock scan is slow and does not depend on within-user order, so its outputs can be reused." if reuse_slow else "",
        "timezone_primary": "naive CSV civil time, differences computed from datetime64 microseconds",
        "timezone_sensitivity": "America/New_York offset, fold=0 for the ambiguous hour",
        "neural_seed_replayed": NEURAL_SEED,
        "constants": {
            "MAX_EVAL_PREFIX_EVENTS": MAX_EVAL_PREFIX_EVENTS,
            "MAX_HIST_TOKENS": MAX_HIST_TOKENS,
            "MAX_SEQ_ACTIONS": MAX_SEQ_ACTIONS,
            "MAX_TRAIN_EVENTS_PER_USER": MAX_TRAIN_EVENTS_PER_USER,
        },
        "status": status,
    }
    (OUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_checksums()
    print(status, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-clock-funnel-support", action="store_true")
    main(parser.parse_args().reuse_clock_funnel_support)
