#!/usr/bin/env python3
"""Reconstruct the ASSISTments 2012-13 development, calibration, and confirmatory splits.

Imports the frozen parser, correctness rule, user split, and release-key
functions. No training or neural inference is performed, and nothing is written
into prospective/wdlf_r7/sealed or results/wdlf_r7.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from r7_core import (  # noqa: E402
    MIN_USER_EVENTS,
    SPLIT_SEED,
    Event,
    bin_correct,
    project_down,
    release_key,
    rich_signature,
    user_bucket,
)
from src.data.assistments_action_parser import parse_actions_field  # noqa: E402

STAGE = ROOT / "data/raw/assistments"
RECON = ROOT / "data/processed/assistments"
OUT = ROOT / "results/assistments/source_reconstruction"
ZIP_PATH = STAGE / "assistments_2012.zip"
INNER = "2012-2013-data-with-predictions-4-final.csv"
EXPECTED_ZIP_SHA = "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445"
EXPECTED = {
    "development": {"users": 31753, "events": 4286981},
    "calibration": {"users": 6804, "events": 888158},
    "confirmatory": {"users": 6766, "events": 939910},
}
EXPECTED_USERS = 45323
EXPECTED_EVENTS = 6115049
REPORTED_APPROX_KEYS = {"S0": 259838, "S1": 511770, "S2": 523378, "S3": 605097}
HINT_TYPES = {"hint"}
ATTEMPT_TYPES = {"answer", "attempt"}
TIME_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
SCHEMA = pa.schema(
    [
        ("source_row", pa.int64()),
        ("user_id", pa.string()),
        ("problem_id", pa.string()),
        ("skill_id", pa.string()),
        ("skill_name", pa.string()),
        ("correct", pa.int8()),
        ("hint_count", pa.int32()),
        ("attempt_count", pa.int32()),
        ("first_action", pa.string()),
        ("action_types", pa.list_(pa.string())),
        ("action_timestamps", pa.list_(pa.string())),
        ("start_time", pa.string()),
        ("end_time", pa.string()),
        ("split", pa.string()),
    ]
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_member(path: Path, member: str) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with zipfile.ZipFile(path) as zf, zf.open(member) as raw:
        for chunk in iter(lambda: raw.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def correctness_class(raw: Any) -> str:
    if blank(raw):
        return "correctness_missing"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return "correctness_unparseable"
    if value >= 1.0:
        return "correctness_binary_correct"
    if value <= 0.0:
        return "correctness_binary_incorrect"
    return "correctness_partial"


def number_class(raw: Any) -> str:
    if blank(raw):
        return "blank"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return "unparseable"
    if not np.isfinite(value):
        return "unparseable"
    if float(value).is_integer():
        return "integer"
    return "noninteger"


def classify_row(source_row: int, row: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, dict[str, int], dict[str, int]]:
    """Return exclusive reason, kept record, independent flags, parser-reason counts.

    The kept predicate is the frozen stream_all_events predicate. Exclusive
    reasons follow that function's short-circuit order and only split a
    combined branch into the reason the same tests already distinguish.
    """
    flags: dict[str, int] = Counter()
    reasons: dict[str, int] = Counter()
    uid_raw = row.get("user_id")
    pid_raw = row.get("problem_id")
    uid = "" if blank(uid_raw) else str(uid_raw).strip()
    pid = "" if blank(pid_raw) else str(pid_raw).strip()
    if not uid:
        flags["missing_user_id"] += 1
    if not pid:
        flags["missing_problem_id"] += 1
    cclass = correctness_class(row.get("correct"))
    flags[cclass] += 1
    if blank(row.get("actions")):
        flags["actions_empty"] += 1
    if blank(row.get("start_time")):
        flags["start_time_missing"] += 1
    if blank(row.get("end_time")):
        flags["end_time_missing"] += 1
    if blank(row.get("skill_id")):
        flags["skill_id_missing"] += 1
    if blank(row.get("skill")):
        flags["skill_name_missing"] += 1
    if blank(row.get("first_action")):
        flags["first_action_missing"] += 1
    for field, prefix in (("hint_count", "hint_count"), ("attempt_count", "attempt_count")):
        kind = number_class(row.get(field))
        flags[f"{prefix}_{kind}"] += 1
    if not uid:
        return "missing_user_id", None, flags, reasons
    if not pid:
        return "missing_problem_id", None, flags, reasons
    y = bin_correct(row.get("correct"))
    if y is None:
        if cclass == "correctness_partial":
            return "correctness_partial", None, flags, reasons
        if cclass == "correctness_missing":
            return "correctness_missing", None, flags, reasons
        return "correctness_unparseable", None, flags, reasons
    parsed = parse_actions_field(row.get("actions"))
    flags[f"parser_{parsed.status}"] += 1
    if parsed.reason_code:
        reasons[parsed.reason_code] += 1
    if parsed.non_monotonic:
        flags["parser_non_monotonic_flag"] += 1
    if parsed.duplicated_actions:
        flags["parser_duplicated_actions_flag"] += 1
    if parsed.status != "success" or not parsed.action_types:
        if parsed.status == "partial":
            return "action_parse_partial", None, flags, reasons
        if parsed.status == "success":
            return "empty_action_sequence", None, flags, reasons
        return "action_parse_failed", None, flags, reasons
    try:
        hint = int(float(row.get("hint_count") or 0))
        attempt = int(float(row.get("attempt_count") or 0))
    except (TypeError, ValueError):
        return "hint_or_attempt_unparseable", None, flags, reasons
    skill = str(row.get("skill_id") or row.get("skill") or "UNK").strip() or "UNK"
    first = row.get("first_action")
    first_action = "" if first is None else str(first)
    record = {
        "source_row": source_row,
        "user_id": uid,
        "problem_id": pid,
        "skill_id": skill,
        "skill_name": "" if blank(row.get("skill")) else str(row.get("skill")).strip(),
        "correct": int(y),
        "hint_count": hint,
        "attempt_count": attempt,
        "first_action": first_action,
        "action_types": [str(t).lower() for t in parsed.action_types],
        "action_timestamps": ["" if t is None else str(t) for t in parsed.timestamps],
        "start_time": "" if blank(row.get("start_time")) else str(row.get("start_time")).strip(),
        "end_time": "" if blank(row.get("end_time")) else str(row.get("end_time")).strip(),
        "split": user_bucket(uid),
    }
    return None, record, flags, reasons


def process_batch(payload: tuple[int, list[dict[str, str]]]) -> dict[str, Any]:
    start, rows = payload
    exclusive: Counter[str] = Counter()
    independent: Counter[str] = Counter()
    parser_reasons: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    users: set[str] = set()
    reason_users: dict[str, set[str]] = defaultdict(set)
    for offset, row in enumerate(rows):
        uid = "" if blank(row.get("user_id")) else str(row.get("user_id")).strip()
        if uid:
            users.add(uid)
        reason, record, flags, reasons = classify_row(start + offset, row)
        independent.update(flags)
        parser_reasons.update(reasons)
        if reason is None and record is not None:
            kept.append(record)
        else:
            exclusive[reason or "unclassified"] += 1
            if uid:
                reason_users[reason or "unclassified"].add(uid)
    return {
        "exclusive": dict(exclusive),
        "independent": dict(independent),
        "parser_reasons": dict(parser_reasons),
        "kept": kept,
        "users": users,
        "reason_users": {k: v for k, v in reason_users.items()},
    }


def parse_clock(text: str) -> datetime | None:
    if not text:
        return None
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def magnitude_bin(seconds: float) -> str:
    value = abs(seconds)
    edges = (
        (1, "lt_1s"),
        (10, "1s_to_10s"),
        (60, "10s_to_60s"),
        (600, "1min_to_10min"),
        (3600, "10min_to_60min"),
        (6 * 3600, "1h_to_6h"),
        (24 * 3600, "6h_to_24h"),
        (7 * 24 * 3600, "1d_to_7d"),
    )
    for limit, name in edges:
        if value < limit:
            return name
    return "ge_7d"


def signature_digest(sig: tuple) -> bytes:
    return hashlib.blake2s(repr(sig).encode("utf-8"), digest_size=16).digest()


def write_table(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    arrays = {name: [row[name] for row in rows] for name in SCHEMA.names}
    writer.write_table(pa.Table.from_pydict(arrays, schema=SCHEMA))


def stream_prefilter(limit: int | None, workers: int) -> tuple[Path, dict[str, Any]]:
    from concurrent.futures import ProcessPoolExecutor

    tmp = RECON / "_pre_min_filter.parquet"
    if tmp.exists():
        tmp.unlink()
    RECON.mkdir(parents=True, exist_ok=True)
    exclusive: Counter[str] = Counter()
    independent: Counter[str] = Counter()
    parser_reasons: Counter[str] = Counter()
    reason_users: dict[str, set[str]] = defaultdict(set)
    input_users: set[str] = set()
    raw_runs: Counter[str] = Counter()
    prev_uid = None
    switches = 0
    input_rows = 0
    kept_rows = 0
    batch: list[dict[str, str]] = []
    batch_start = 0
    fields = (
        "user_id",
        "problem_id",
        "skill",
        "skill_id",
        "correct",
        "hint_count",
        "attempt_count",
        "first_action",
        "actions",
        "start_time",
        "end_time",
    )

    def note_raw(row: dict[str, str]) -> None:
        nonlocal prev_uid, switches
        uid = "" if blank(row.get("user_id")) else str(row.get("user_id")).strip()
        if uid != prev_uid:
            if uid:
                raw_runs[uid] += 1
            if prev_uid is not None:
                switches += 1
            prev_uid = uid

    def consume(result: dict[str, Any], writer: pq.ParquetWriter) -> None:
        nonlocal kept_rows
        exclusive.update(result["exclusive"])
        independent.update(result["independent"])
        parser_reasons.update(result["parser_reasons"])
        input_users.update(result["users"])
        for key, users in result["reason_users"].items():
            reason_users[key].update(users)
        write_table(writer, result["kept"])
        kept_rows += len(result["kept"])

    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer, zipfile.ZipFile(ZIP_PATH) as zf, zf.open(INNER) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        if workers <= 1:
            for row in reader:
                if limit is not None and input_rows >= limit:
                    break
                note_raw(row)
                slim = {key: row.get(key) for key in fields}
                consume(process_batch((input_rows, [slim])), writer)
                input_rows += 1
                if input_rows % 200000 == 0:
                    print(f"rows={input_rows} kept={kept_rows}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending = []
                for row in reader:
                    if limit is not None and input_rows >= limit:
                        break
                    note_raw(row)
                    if not batch:
                        batch_start = input_rows
                    batch.append({key: row.get(key) for key in fields})
                    input_rows += 1
                    if len(batch) >= 2500:
                        pending.append(pool.submit(process_batch, (batch_start, batch)))
                        batch = []
                        if len(pending) >= workers:
                            consume(pending.pop(0).result(), writer)
                            if input_rows % 200000 < 2500:
                                print(f"rows={input_rows} kept={kept_rows}", flush=True)
                if batch:
                    pending.append(pool.submit(process_batch, (batch_start, batch)))
                for future in pending:
                    consume(future.result(), writer)
    meta = {
        "input_rows": input_rows,
        "pre_min_rows": kept_rows,
        "exclusive": dict(exclusive),
        "independent": dict(independent),
        "parser_reasons": dict(parser_reasons),
        "input_users": len(input_users),
        "reason_users": {k: len(v) for k, v in reason_users.items()},
        "raw_user_runs": {
            "users_with_id": len(raw_runs),
            "contiguous_users": sum(1 for n in raw_runs.values() if n == 1),
            "interleaved_users": sum(1 for n in raw_runs.values() if n > 1),
            "user_runs": int(sum(raw_runs.values())),
            "user_switches": switches,
        },
    }
    return tmp, meta


def count_splits(tmp: Path) -> dict[str, Any]:
    user_counts: Counter[str] = Counter()
    user_split: dict[str, str] = {}
    conflict = 0
    pf = pq.ParquetFile(tmp)
    for batch in pf.iter_batches(batch_size=200000, columns=["user_id", "split"]):
        users = batch.column("user_id").to_pylist()
        splits = batch.column("split").to_pylist()
        for uid, split in zip(users, splits, strict=True):
            user_counts[uid] += 1
            prev = user_split.get(uid)
            if prev is None:
                user_split[uid] = split
            elif prev != split:
                conflict += 1
    kept_users = {uid for uid, n in user_counts.items() if n >= MIN_USER_EVENTS}
    events = Counter()
    users = Counter()
    removed_events = 0
    removed_users = 0
    for uid, n in user_counts.items():
        if uid in kept_users:
            events[user_split[uid]] += n
            users[user_split[uid]] += 1
        else:
            removed_events += n
            removed_users += 1
    return {
        "pre_min_users": len(user_counts),
        "pre_min_events": int(sum(user_counts.values())),
        "removed_users": removed_users,
        "removed_events": removed_events,
        "events": dict(events),
        "users": dict(users),
        "split_conflicts": conflict,
        "kept_users": kept_users,
        "user_split": user_split,
    }


def diagnose_and_maybe_write(tmp: Path, kept_users: set[str], approved: bool) -> dict[str, Any]:
    last: dict[str, tuple[datetime | None, datetime | None]] = {}
    runs: Counter[str] = Counter()
    prev_user = None
    chrono = Counter()
    magnitudes = Counter()
    action = Counter()
    joint = Counter()
    substantive_joint = Counter()
    type_counts = Counter()
    keys = {schema: set() for schema in ("S0", "S1", "S2", "S3")}
    key_users: dict[str, dict[tuple, set[int]]] = {schema: {} for schema in ("S0", "S1", "S2", "S3")}
    multi_sig = {schema: 0 for schema in ("S0", "S1", "S2", "S3")}
    sig_seen: dict[str, dict[tuple, bytes]] = {schema: {} for schema in ("S0", "S1", "S2", "S3")}
    rich_seen: set[bytes] = set()
    rich_duplicate_events = 0
    rich_duplicate_signatures = 0
    timestamp_variants = 0
    ts_seen: dict[tuple, bytes] = {}
    nesting_checked = 0
    nesting_violations = 0
    dev_events = 0
    writers = {}
    if approved:
        for split in ("development", "calibration", "confirmatory"):
            path = RECON / f"{split}.parquet"
            if path.exists():
                path.unlink()
            writers[split] = pq.ParquetWriter(path, SCHEMA, compression="zstd")
    pf = pq.ParquetFile(tmp)
    try:
        for batch in pf.iter_batches(batch_size=100000):
            rows = batch.to_pylist()
            kept_rows = [row for row in rows if row["user_id"] in kept_users]
            for row in kept_rows:
                uid = row["user_id"]
                if uid != prev_user:
                    runs[uid] += 1
                    prev_user = uid
                start = parse_clock(row["start_time"])
                end = parse_clock(row["end_time"])
                if start is None:
                    chrono["retained_start_unparsed"] += 1
                else:
                    chrono["retained_start_parsed"] += 1
                if end is None:
                    chrono["retained_end_unparsed"] += 1
                else:
                    chrono["retained_end_parsed"] += 1
                if start is not None and end is not None and end < start:
                    chrono["within_event_end_before_start"] += 1
                prev = last.get(uid)
                if prev is not None:
                    prev_start, prev_end = prev
                    if start is not None and prev_start is not None:
                        delta = (start - prev_start).total_seconds()
                        if delta < 0:
                            chrono["adjacent_start_inversions"] += 1
                            magnitudes[magnitude_bin(delta)] += 1
                            if prev_end is not None and start < prev_end:
                                chrono["start_inversions_overlapping_previous_interval"] += 1
                        elif delta == 0:
                            chrono["adjacent_start_ties"] += 1
                        elif prev_end is not None and start < prev_end:
                            chrono["overlap_with_nondecreasing_start"] += 1
                    if end is not None and prev_end is not None:
                        if end < prev_end:
                            chrono["adjacent_end_inversions"] += 1
                        elif end == prev_end:
                            chrono["adjacent_end_ties"] += 1
                    if start is not None and prev_start is not None and end is not None and prev_end is not None:
                        ds = (start - prev_start).total_seconds()
                        de = (end - prev_end).total_seconds()
                        if ds == 0 or de == 0:
                            chrono["start_end_order_tie"] += 1
                        elif (ds > 0) != (de > 0):
                            chrono["start_end_order_disagreement"] += 1
                        else:
                            chrono["start_end_order_agreement"] += 1
                last[uid] = (start, end)
                types = [str(t).lower() for t in row["action_types"]]
                stamps = ["" if t is None else str(t) for t in row["action_timestamps"]]
                type_counts.update(types)
                hint_like = sum(1 for t in types if t in HINT_TYPES or t.startswith("hint"))
                attempt_like = sum(1 for t in types if t in ATTEMPT_TYPES)
                action["retained_events"] += 1
                action["hint_equal_hint_like"] += int(row["hint_count"] == hint_like)
                action["hint_disagree_hint_like"] += int(row["hint_count"] != hint_like)
                action["attempt_equal_answer_or_attempt"] += int(row["attempt_count"] == attempt_like)
                action["attempt_disagree_answer_or_attempt"] += int(row["attempt_count"] != attempt_like)
                first_type = types[0] if types else ""
                literal = str(row["first_action"]).strip().lower()
                action["first_action_literal_equals_first_type"] += int(literal == first_type)
                action["first_action_literal_differs"] += int(literal != first_type)
                joint[(str(row["first_action"]), first_type)] += 1
                substantive = next((t for t in types if t not in {"start", "end", "resume"}), "")
                substantive_joint[(str(row["first_action"]), substantive)] += 1
                if start is not None:
                    for atype, stamp in zip(types, stamps, strict=True):
                        if atype != "start" or not stamp:
                            continue
                        try:
                            epoch = float(stamp)
                        except ValueError:
                            action["start_action_timestamp_nonnumeric"] += 1
                            continue
                        if epoch > 1e11:
                            predicted = datetime.fromtimestamp(epoch / 1000.0, timezone.utc).replace(tzinfo=None)
                            action["start_action_epoch_ms"] += 1
                            if abs((predicted - start).total_seconds()) <= 1:
                                action["start_action_matches_start_time_1s"] += 1
                            else:
                                action["start_action_differs_from_start_time"] += 1
                        else:
                            action["start_action_small_offset"] += 1
                        break
                if row["split"] != "development":
                    continue
                dev_events += 1
                ev = Event(
                    user_id=uid,
                    problem_id=row["problem_id"],
                    skill_id=row["skill_id"],
                    correct=int(row["correct"]),
                    hint_count=int(row["hint_count"]),
                    attempt_count=int(row["attempt_count"]),
                    first_action=str(row["first_action"]),
                    action_types=tuple(types),
                    action_timestamps=tuple(stamps),
                    split=row["split"],
                )
                sig = rich_signature(ev)
                digest = signature_digest(sig)
                if digest in rich_seen:
                    rich_duplicate_events += 1
                else:
                    rich_seen.add(digest)
                s3 = release_key(ev, "S3")
                ts_digest = signature_digest(tuple(stamps))
                previous_ts = ts_seen.get(s3)
                if previous_ts is None:
                    ts_seen[s3] = ts_digest
                elif previous_ts != ts_digest and previous_ts != b"multi":
                    timestamp_variants += 1
                    ts_seen[s3] = b"multi"
                for schema in keys:
                    key = release_key(ev, schema)
                    keys[schema].add(key)
                    users = key_users[schema].get(key)
                    if users is None:
                        key_users[schema][key] = {int(uid)}
                    elif len(users) < 2:
                        users.add(int(uid))
                    seen = sig_seen[schema].get(key)
                    if seen is None:
                        sig_seen[schema][key] = digest
                    elif seen != digest and seen != b"multi":
                        multi_sig[schema] += 1
                        sig_seen[schema][key] = b"multi"
                k4 = release_key(ev, "S4")
                for hi, lo in (("S1", "S0"), ("S2", "S1"), ("S3", "S2"), ("S4", "S3")):
                    khi = k4 if hi == "S4" else release_key(ev, hi)
                    if project_down(khi, hi, lo) != release_key(ev, lo):
                        nesting_violations += 1
                    nesting_checked += 1
            if approved:
                by_split = defaultdict(list)
                for row in kept_rows:
                    by_split[row["split"]].append(row)
                for split, group in by_split.items():
                    write_table(writers[split], group)
    finally:
        for writer in writers.values():
            writer.close()
    rich_duplicate_signatures = sum(1 for value in ts_seen.values() if False)
    del rich_duplicate_signatures
    geometry = []
    for schema in ("S0", "S1", "S2", "S3"):
        geometry.append(
            {
                "schema": schema,
                "development_events": dev_events,
                "n_keys": len(keys[schema]),
                "n_keys_with_multiple_rich_signatures": multi_sig[schema],
                "n_keys_with_at_least_two_users": sum(1 for users in key_users[schema].values() if len(users) >= 2),
                "reported_approximate_keys": REPORTED_APPROX_KEYS[schema],
                "frozen_repository_key_count": None,
                "difference_from_reported_approximation": len(keys[schema]) - REPORTED_APPROX_KEYS[schema],
            }
        )
    geometry.append(
        {
            "schema": "rich_signature",
            "development_events": dev_events,
            "n_keys": len(rich_seen),
            "n_keys_with_multiple_rich_signatures": rich_duplicate_events,
            "n_keys_with_at_least_two_users": None,
            "reported_approximate_keys": None,
            "frozen_repository_key_count": None,
            "difference_from_reported_approximation": None,
            "n_s3_keys_with_multiple_timestamp_variants": timestamp_variants,
            "nesting_checked": nesting_checked,
            "nesting_violations": nesting_violations,
            "digest": "blake2s-128 of repr(rich_signature); cardinality only",
        }
    )
    return {
        "chrono": dict(chrono),
        "magnitudes": dict(magnitudes),
        "action": dict(action),
        "joint": [{"first_action": a, "first_parsed_type": b, "n": n} for (a, b), n in joint.most_common()],
        "substantive_joint": [{"first_action": a, "first_nonboundary_type": b, "n": n} for (a, b), n in substantive_joint.most_common()],
        "action_type_counts": type_counts.most_common(),
        "geometry": geometry,
        "retained_user_runs": {
            "contiguous_users": sum(1 for n in runs.values() if n == 1),
            "interleaved_users": sum(1 for n in runs.values() if n > 1),
            "users": len(runs),
        },
        "dev_events": dev_events,
        "nesting_checked": nesting_checked,
        "nesting_violations": nesting_violations,
        "timestamp_variant_s3_keys": timestamp_variants,
        "rich_signatures": len(rich_seen),
        "rich_duplicate_events": rich_duplicate_events,
    }


def funnel_rows(meta: dict[str, Any], splits: dict[str, Any]) -> list[dict[str, Any]]:
    order = (
        "missing_user_id",
        "missing_problem_id",
        "correctness_missing",
        "correctness_unparseable",
        "correctness_partial",
        "action_parse_failed",
        "action_parse_partial",
        "empty_action_sequence",
        "hint_or_attempt_unparseable",
    )
    rows = []
    cumulative = meta["input_rows"]
    rows.append(
        {
            "stage": "input",
            "exclusive_reason": "",
            "input_rows": meta["input_rows"],
            "rows_retained": meta["input_rows"],
            "rows_excluded": 0,
            "cumulative_retained_rows": meta["input_rows"],
            "user_count": meta["input_users"],
            "note": "CSV data rows in file order",
        }
    )
    for reason in order:
        excluded = int(meta["exclusive"].get(reason, 0))
        cumulative -= excluded
        rows.append(
            {
                "stage": "exclusive_filter",
                "exclusive_reason": reason,
                "input_rows": meta["input_rows"],
                "rows_retained": cumulative,
                "rows_excluded": excluded,
                "cumulative_retained_rows": cumulative,
                "user_count": meta["reason_users"].get(reason, 0),
                "note": "users with at least one row assigned this exclusive reason; a user may also have retained rows",
            }
        )
    if cumulative != meta["pre_min_rows"]:
        raise SystemExit(f"funnel mismatch {cumulative} != {meta['pre_min_rows']}")
    rows.append(
        {
            "stage": "minimum_user_events",
            "exclusive_reason": "fewer_than_3_retained_events",
            "input_rows": meta["input_rows"],
            "rows_retained": meta["pre_min_rows"] - splits["removed_events"],
            "rows_excluded": splits["removed_events"],
            "cumulative_retained_rows": meta["pre_min_rows"] - splits["removed_events"],
            "user_count": splits["removed_users"],
            "note": f"entire user removed when retained events < {MIN_USER_EVENTS}; applied after row filters and preserves order",
        }
    )
    for name, count in sorted(meta["independent"].items()):
        rows.append(
            {
                "stage": "independent_flag",
                "exclusive_reason": name,
                "input_rows": meta["input_rows"],
                "rows_retained": "",
                "rows_excluded": count,
                "cumulative_retained_rows": "",
                "user_count": "",
                "note": "not an exclusive reason; flags can overlap",
            }
        )
    for name, count in sorted(meta["parser_reasons"].items()):
        rows.append(
            {
                "stage": "parser_reason_code",
                "exclusive_reason": name,
                "input_rows": meta["input_rows"],
                "rows_retained": "",
                "rows_excluded": count,
                "cumulative_retained_rows": "",
                "user_count": "",
                "note": "reason_code from parse_actions_field on rows that reached the parser",
            }
        )
    return rows


def write_checksums() -> dict[str, Any]:
    rows = []
    for path in sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "output_checksums.json"):
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"algorithm": "sha256", "excluded": ["results/assistments/source_reconstruction/output_checksums.json"], "n_files": len(rows), "files": rows}
    (OUT / "output_checksums.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    limit = None
    workers = 6
    args = sys.argv[1:]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])
    OUT.mkdir(parents=True, exist_ok=True)
    RECON.mkdir(parents=True, exist_ok=True)
    print("hash zip", flush=True)
    zip_sha = sha256_file(ZIP_PATH)
    identity = "EXACT_RAW_SOURCE_MATCH" if zip_sha == EXPECTED_ZIP_SHA else "OFFICIAL_SOURCE_NONIDENTICAL"
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [
            {"name": info.filename, "bytes": info.file_size, "compressed_bytes": info.compress_size, "date_time": list(info.date_time)}
            for info in zf.infolist()
        ]
    print("hash csv member", flush=True)
    csv_sha, csv_bytes = sha256_member(ZIP_PATH, INNER)
    provenance = {
        "official_page": "https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect",
        "documented_drive_page": "https://drive.google.com/file/d/1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY/view?usp=sharing",
        "retrieval": "Public Google Drive download after the virus-scan confirmation form. No account and no credential.",
        "download_date": "2026-11-24",
        "file_name_on_official_page": "2012-2013-data-with-predictions-4-final.zip",
        "staged_path": ZIP_PATH.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": ZIP_PATH.stat().st_size,
        "sha256": zip_sha,
        "container": "zip",
        "authentication_required": False,
        "redistribution": "Raw archive is retained locally for reproduction and is not copied into the report or the results directory.",
        "citation_note": "Official page requests citation of Feng, Heffernan, and Koedinger (2009) when affect columns are not used. Problem text is not in this archive.",
        "repository_contract_path": "data/raw/assistments/assistments_2012.zip",
        "repository_contract_sha256": EXPECTED_ZIP_SHA,
    }
    (OUT / "source_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (OUT / "archive_contents.json").write_text(
        json.dumps({"members": members, "expected_member": INNER, "expected_member_present": any(m["name"] == INNER for m in members), "csv_uncompressed_bytes": csv_bytes, "csv_sha256": csv_sha}, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT / "source_identity_checks.json").write_text(
        json.dumps(
            {
                "classification": identity,
                "expected_zip_sha256": EXPECTED_ZIP_SHA,
                "observed_zip_sha256": zip_sha,
                "zip_sha256_match": zip_sha == EXPECTED_ZIP_SHA,
                "csv_sha256_is_a_different_object": True,
                "csv_sha256": csv_sha,
                "expected_csv_member_present": any(m["name"] == INNER for m in members),
                "filename_differs_from_repository_path": True,
                "byte_identity_rather_than_filename": zip_sha == EXPECTED_ZIP_SHA,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    contract = {
        "functions": {
            "reader": "scripts/wdlf_r7/run_release_frontier_confirmatory.py:stream_all_events",
            "parser": "src/data/assistments_action_parser.py:parse_actions_field",
            "correctness": "scripts/wdlf_r7/r7_core.py:bin_correct",
            "split": "scripts/wdlf_r7/r7_core.py:user_bucket",
            "release_key": "scripts/wdlf_r7/r7_core.py:release_key",
            "rich_signature": "scripts/wdlf_r7/r7_core.py:rich_signature",
        },
        "split_seed": SPLIT_SEED,
        "fractions": {"development": 0.70, "calibration": 0.15, "confirmatory": 0.15},
        "minimum_user_events": MIN_USER_EVENTS,
        "correctness": "float(raw) >= 1 -> 1; float(raw) <= 0 -> 0; otherwise drop, including partial credit",
        "parser_acceptance": "status == success and action_types nonempty. Partial and failed parses are dropped.",
        "action_type_normalization": "str(type).lower() with no further synonym map",
        "timestamp_handling": "parser timestamps are stored as strings; None becomes empty string. Event start_time and end_time are not read by the frozen consumer.",
        "release_keys": {
            "S0": "(S0, problem_id, correct)",
            "S1": "(S1, problem_id, correct, hint_count, attempt_count)",
            "S2": "(S2, problem_id, correct, hint_count, attempt_count, first_action)",
            "S3": "(S3, problem_id, correct, hint_count, attempt_count, first_action, action_types)",
            "rich_signature": "(problem_id, correct, hint_count, attempt_count, first_action, action_types, action_timestamps)",
        },
        "order_dependencies": [
            "DictReader yields CSV row order.",
            "stream_all_events appends kept rows in that order.",
            "The minimum-event filter uses a list comprehension and preserves relative order.",
            "No per-user temporal sort exists in the parser or the event builder.",
            "user_bucket and release_key do not read row position.",
            "Later certificate code may sort user ids for ranking; it does not reorder a user's events.",
        ],
        "counts_not_rederived": "hint_count, attempt_count, and first_action are copied from CSV columns. They are not replaced by parsed action counts.",
        "historical_script_not_executed": True,
    }
    (OUT / "preprocessing_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    if identity != "EXACT_RAW_SOURCE_MATCH":
        blocked = {"status": "NOT_EXECUTED", "blocker": identity}
        for name in (
            "preprocessing_funnel.csv",
            "split_reconstruction.json",
            "chronology_diagnostics.csv",
            "action_column_consistency.csv",
            "release_geometry_reconciliation.csv",
            "reconstruction_manifest.json",
            "experiment_readiness_matrix.csv",
        ):
            path = OUT / name
            if name.endswith(".csv"):
                path.write_text("status\nNOT_EXECUTED\n", encoding="utf-8")
            else:
                path.write_text(json.dumps(blocked, indent=2) + "\n", encoding="utf-8")
        (OUT / "integrity_checks.json").write_text(json.dumps({"classification": identity, "reconstruction": "NOT_EXECUTED"}, indent=2) + "\n", encoding="utf-8")
        write_checksums()
        raise SystemExit(2)

    print("stream", flush=True)
    tmp, meta = stream_prefilter(limit, workers)
    print("count splits", flush=True)
    split_info = count_splits(tmp)
    kept_users = split_info.pop("kept_users")
    user_split = split_info.pop("user_split")
    observed_users = sum(split_info["users"].values())
    observed_events = sum(split_info["events"].values())
    split_match = (
        observed_users == EXPECTED_USERS
        and observed_events == EXPECTED_EVENTS
        and all(split_info["users"].get(name) == spec["users"] and split_info["events"].get(name) == spec["events"] for name, spec in EXPECTED.items())
        and split_info["split_conflicts"] == 0
    )
    approved = bool(split_match and limit is None)
    print("diagnose", approved, flush=True)
    diagnostics = diagnose_and_maybe_write(tmp, kept_users, approved)
    import pandas as pd

    pd.DataFrame(funnel_rows(meta, split_info)).to_csv(OUT / "preprocessing_funnel.csv", index=False)
    split_payload = {
        "status": "EXACT_MATCH" if approved else "MISMATCH",
        "approved_reconstruction": approved,
        "seed": SPLIT_SEED,
        "minimum_user_events": MIN_USER_EVENTS,
        "expected": EXPECTED,
        "expected_users": EXPECTED_USERS,
        "expected_events": EXPECTED_EVENTS,
        "observed_users": {k: int(v) for k, v in split_info["users"].items()},
        "observed_events": {k: int(v) for k, v in split_info["events"].items()},
        "observed_user_total": int(observed_users),
        "observed_event_total": int(observed_events),
        "pre_min_users": split_info["pre_min_users"],
        "pre_min_events": split_info["pre_min_events"],
        "removed_users": split_info["removed_users"],
        "removed_events": split_info["removed_events"],
        "split_conflicts": split_info["split_conflicts"],
        "user_assignment_is_a_function": split_info["split_conflicts"] == 0,
        "event_order": "CSV order preserved; no temporal sort",
    }
    (OUT / "split_reconstruction.json").write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")
    chrono_rows = [{"metric": key, "value": value} for key, value in diagnostics["chrono"].items()]
    chrono_rows.extend({"metric": f"inversion_magnitude_{key}", "value": value} for key, value in diagnostics["magnitudes"].items())
    chrono_rows.extend({"metric": f"retained_{key}", "value": value} for key, value in diagnostics["retained_user_runs"].items())
    chrono_rows.extend({"metric": f"raw_{key}", "value": value} for key, value in meta["raw_user_runs"].items())
    pd.DataFrame(chrono_rows).to_csv(OUT / "chronology_diagnostics.csv", index=False)
    action_rows = [{"comparison": key, "n": value, "scope": "retained_events"} for key, value in diagnostics["action"].items()]
    action_rows.extend(
        {"comparison": f"joint_first_action[{item['first_action']}]_first_type[{item['first_parsed_type']}]", "n": item["n"], "scope": "retained_events"}
        for item in diagnostics["joint"]
    )
    action_rows.extend(
        {
            "comparison": f"candidate_first_action[{item['first_action']}]_first_nonboundary[{item['first_nonboundary_type']}]",
            "n": item["n"],
            "scope": "candidate_mapping_not_used_by_pipeline",
        }
        for item in diagnostics["substantive_joint"]
    )
    action_rows.extend({"comparison": f"action_type[{name}]", "n": count, "scope": "retained_action_tokens"} for name, count in diagnostics["action_type_counts"])
    pd.DataFrame(action_rows).to_csv(OUT / "action_column_consistency.csv", index=False)
    pd.DataFrame(diagnostics["geometry"]).to_csv(OUT / "release_geometry_reconciliation.csv", index=False)
    recon_files = []
    if approved:
        for split in ("development", "calibration", "confirmatory"):
            path = RECON / f"{split}.parquet"
            recon_files.append(
                {
                    "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "rows": int(split_info["events"][split]),
                    "users": int(split_info["users"][split]),
                    "label": "reconstructed_from_exact_source",
                    "not_an_original_lfs_payload": True,
                }
            )
        if tmp.exists():
            tmp.unlink()
    reconstruction = {
        "status": "WRITTEN" if approved else "NOT_APPROVED",
        "format": "parquet",
        "schema": SCHEMA.names,
        "order": "source_row ascending within each file; source_row is the CSV data-row index",
        "files": recon_files,
        "pre_min_cache_removed": approved,
        "does_not_replace": [
            "prospective/wdlf_r7/sealed/dicts_dev_compact.pkl",
            "prospective/wdlf_r7/sealed/splits/confirmatory.pkl",
        ],
    }
    (OUT / "reconstruction_manifest.json").write_text(json.dumps(reconstruction, indent=2) + "\n", encoding="utf-8")
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__, "platform": platform.platform()}
    integrity = {
        "classification": identity,
        "zip_sha256_match": True,
        "split_exact_match": approved,
        "funnel_reconciles": True,
        "split_conflicts": split_info["split_conflicts"],
        "nesting_checked": diagnostics["nesting_checked"],
        "nesting_violations": diagnostics["nesting_violations"],
        "workers": workers,
        "limit": limit,
        "versions": versions,
        "command": "python3 -u scripts/assistments/source_reconstruction.py --workers 6",
        "historical_script_not_executed": True,
        "training": False,
        "neural_inference": False,
    }
    (OUT / "integrity_checks.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    checksums = write_checksums()
    bad = [row["path"] for row in checksums["files"] if sha256_file(ROOT / row["path"]) != row["sha256"]]
    print(json.dumps({"classification": identity, "approved": approved, "users": split_payload["observed_users"], "events": split_payload["observed_events"], "checksums": checksums["n_files"], "bad": bad}, indent=2))
    if bad or not approved:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
