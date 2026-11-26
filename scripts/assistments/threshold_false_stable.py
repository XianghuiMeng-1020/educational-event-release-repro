#!/usr/bin/env python3
"""Complete false-Stable counts at tau 0.70, 0.80, and 0.90 for the attention consumer.

Uses the frozen CausalTokenAttentionKT seed-17 checkpoint. No training, no new
checkpoint, and no change to the attention-consumer tables.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "attncons_attention_consumer", ROOT / "scripts/assistments/attention_consumer.py"
)
attncons = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attncons)
chronorep = attncons.chronorep
matchedinfo = attncons.matchedinfo

from r7_core import Event, NeuralPilot  # noqa: E402

OUT = ROOT / "results/assistments/attention_consumer/threshold_false_stable"
ATTNCONS = ROOT / "results/assistments/attention_consumer"
CKPT = ATTNCONS / "causal_token_attention_seed17.pt"
CKPT_SHA = "8728fb05b33399df24daaaa48a76b3c6d6870aa2cee8419909cdeda5f88c0aaa"
VALID = 939819
INVALID = 91
SOURCE_EVENTS = 939910
TAUS = (0.70, 0.80, 0.90)
SCHEMAS = ("S0", "S1", "S2", "S3")
EXPECTED_080 = {
    "S0": {"unsupported": 20408, "stable": 763342, "set_valued": 156069, "n_U": 33420, "false_stable": 1579},
    "S1": {"unsupported": 77877, "stable": 808049, "set_valued": 53893, "n_U": 13918, "false_stable": 697},
    "S2": {"unsupported": 80790, "stable": 805452, "set_valued": 53577, "n_U": 12242, "false_stable": 694},
    "S3": {"unsupported": 100688, "stable": 839131, "set_valued": 0, "n_U": 0, "false_stable": 0},
}
EXPECTED_NC = {
    0.70: {"S0": 172864, "S1": 122263, "delta": -50601},
    0.80: {"S0": 176477, "S1": 131770, "delta": -44707},
    0.90: {"S0": 190079, "S1": 154946, "delta": -35133},
}


def dump(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=chronorep._json_default) + "\n", encoding="utf-8")


def ratio(num: int, den: int):
    if den == 0:
        return "N/A"
    return num / den


def case_of(n_s: int, n_u: int) -> str:
    if n_s == 0:
        return "C"
    if n_u == 0:
        return "B"
    return "A"


def classify(lo: float, hi: float, realized: float, tau: float) -> tuple[int, bool, bool]:
    """Return status code, false-stable, action-contained.

    Status codes match the chronological replication: 1 stable, 2 set-valued. Unsupported is handled
    by the caller. Comparisons are the original exact threshold tests.
    """
    if lo >= tau or hi < tau:
        action_progress = lo >= tau
        realized_progress = realized >= tau
        false_stable = action_progress != realized_progress
        return 1, false_stable, not false_stable
    return 2, False, True


def cache_readiness() -> dict:
    frame = pq.read_schema(ATTNCONS / "private_score_caches/attention_event_scores.parquet")
    status = np.load(ATTNCONS / "private_score_caches/attention_user_status.npz")
    columns = list(frame.names)
    required = [
        "event_identity",
        "release_schema",
        "empirical_support",
        "score_min",
        "score_max",
        "score_actual",
        "fingerprint_coverage",
        "fixed_history_identity",
    ]
    present = {
        "event_identity": "event_key" in columns,
        "release_schema": False,
        "empirical_support": False,
        "score_min": False,
        "score_max": False,
        "score_actual": "score" in columns,
        "fingerprint_coverage": False,
        "fixed_history_identity": False,
    }
    payload = {
        "event_score_columns": columns,
        "user_status_keys": list(status.files),
        "user_status_shape": list(status["status"].shape),
        "user_status_meaning": "per-user counts of unsupported, stable, set-valued at three thresholds; not per-event intervals",
        "field_present": present,
        "required_fields": required,
        "sufficient_for_false_stable_at_arbitrary_tau": all(present[name] for name in required),
        "decision": "frozen-model inference required",
        "reason": "The attention-consumer cache stores one realized score per event and aggregated status counts. Certificate minima, maxima, support, and fingerprint coverage are absent, so false-Stable at 0.70 and 0.90 cannot be derived without rescoring the frozen checkpoint.",
    }
    dump("cache_readiness.json", payload)
    return payload


def load_model() -> torch.nn.Module:
    digest = chronorep.sha256_file(CKPT)
    if digest != CKPT_SHA:
        raise SystemExit(f"checkpoint hash mismatch {digest}")
    frozen = torch.load(ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt", map_location="cpu", weights_only=False)
    skill2i = frozen["skill2i"]
    atype2i = frozen["atype2i"]
    del frozen
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = attncons.CausalTokenAttentionKT(len(skill2i) - 1, len(atype2i) - 1)
    model.load_state_dict(state["state_dict"])
    model.eval()
    if model.training:
        raise SystemExit("model left train mode")
    pilot = NeuralPilot()
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.model = model
    pilot.device = torch.device("cpu")
    return pilot, int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def open_cache(n_events: int, resume: bool):
    cache = OUT / "private_event_cache"
    cache.mkdir(parents=True, exist_ok=True)
    mode = "r+" if resume else "w+"
    source = np.memmap(cache / "source_row.i8", dtype=np.int64, mode=mode, shape=(n_events,))
    scores = np.memmap(cache / "scores.f8", dtype=np.float64, mode=mode, shape=(n_events, 4, 3))
    flags = np.memmap(cache / "flags.u8", dtype=np.uint8, mode=mode, shape=(n_events, 4, 2))
    return source, scores, flags


def score_population(pilot: NeuralPilot, grouped: dict, bundles: dict, cached_keys: np.ndarray, cached_scores: np.ndarray) -> dict:
    users = sorted(grouped)
    progress_path = OUT / "score_progress.json"
    resume_from = 0
    event_base = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        resume_from = int(progress["next_user"])
        event_base = int(progress["next_event"])
    source, scores, flags = open_cache(VALID, resume_from > 0)
    counters_path = OUT / "running_counters.npz"
    if resume_from > 0 and counters_path.exists():
        previous = np.load(counters_path)
        status = previous["status"]
        false_stable = previous["false_stable"]
        uncovered = previous["uncovered_stable"]
        contained = previous["contained"]
        covered = previous["covered"]
        supported = previous["supported"]
        max_abs = float(previous["max_abs_vs_cache"][0])
        side_flips = int(previous["side_flips"][0])
        future_violations = int(previous["future_violations"][0])
        implication_violations = int(previous["implication_violations"][0])
    else:
        status = np.zeros((len(users), 4, 3, 3), dtype=np.int32)
        false_stable = np.zeros((len(users), 4, 3), dtype=np.int32)
        uncovered = np.zeros((len(users), 4, 3), dtype=np.int32)
        contained = np.zeros((len(users), 4, 3), dtype=np.int32)
        covered = np.zeros((len(users), 4), dtype=np.int32)
        supported = np.zeros((len(users), 4), dtype=np.int32)
        max_abs = 0.0
        side_flips = 0
        future_violations = 0
        implication_violations = 0
        resume_from = 0
        event_base = 0
    device = torch.device("cpu")
    seen = event_base
    for user_i, uid in enumerate(users):
        by_start, by_end = chronorep.order_valid(grouped[uid])
        if user_i < resume_from:
            continue
        cursor = 0
        realized_pairs = []
        event_rows = []
        for row in by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                cursor += 1
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                future_violations += 1
            prefix = [chronorep.as_event(item) for item in by_end[max(0, cursor - chronorep.MAX_EVAL_PREFIX_EVENTS) : cursor]]
            ev = chronorep.as_event(row)
            keys = {schema: chronorep.release_key(ev, schema) for schema in SCHEMAS}
            slots = {schema: bundles[schema].get(keys[schema]) for schema in SCHEMAS}
            support_flags = {schema: chronorep.supported(slots[schema]) for schema in SCHEMAS}
            realized_pairs.append(pilot._tokens(prefix, ev, False))
            event_rows.append((row, prefix, slots, support_flags))
        realized = chronorep.score_forward(pilot.model, realized_pairs, device)
        extra_pairs = []
        extra_meta = []
        for event_i, (row, prefix, slots, support_flags) in enumerate(event_rows):
            for schema in SCHEMAS:
                if not support_flags[schema]:
                    continue
                own = chronorep.gru_fp(row)
                for fp in slots[schema].gru:
                    if fp == own:
                        continue
                    extra_pairs.append(chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]))
                    extra_meta.append((event_i, schema))
        extra = chronorep.score_forward(pilot.model, extra_pairs, device) if extra_pairs else np.zeros(0)
        grouped_extra: dict[tuple[int, str], list[float]] = {}
        for (event_i, schema), score in zip(extra_meta, extra):
            grouped_extra.setdefault((event_i, schema), []).append(float(score))
        for event_i, (row, _prefix, slots, support_flags) in enumerate(event_rows):
            if seen >= VALID:
                raise SystemExit("event overflow")
            key = attncons.event_key(row["source_row"])
            if cached_keys[seen] != key:
                raise SystemExit(f"cached event order diverged at {seen}")
            realized_score = float(realized[event_i])
            diff = abs(realized_score - float(cached_scores[seen]))
            max_abs = max(max_abs, diff)
            for tau in TAUS:
                if (realized_score >= tau) != (float(cached_scores[seen]) >= tau):
                    side_flips += 1
            source[seen] = row["source_row"]
            own = chronorep.gru_fp(row)
            for schema_i, schema in enumerate(SCHEMAS):
                if not support_flags[schema]:
                    status[user_i, schema_i, :, 0] += 1
                    flags[seen, schema_i, 0] = 0
                    flags[seen, schema_i, 1] = 0
                    scores[seen, schema_i, :] = np.nan
                    continue
                slot = slots[schema]
                values = list(grouped_extra.get((event_i, schema), []))
                is_covered = own in slot.gru
                if is_covered:
                    values.append(realized_score)
                if not values:
                    values = [0.5]
                lo = float(min(values))
                hi = float(max(values))
                supported[user_i, schema_i] += 1
                if is_covered:
                    covered[user_i, schema_i] += 1
                flags[seen, schema_i, 0] = 1
                flags[seen, schema_i, 1] = int(is_covered)
                scores[seen, schema_i, 0] = lo
                scores[seen, schema_i, 1] = hi
                scores[seen, schema_i, 2] = realized_score
                for tau_i, tau in enumerate(TAUS):
                    code, is_false, is_contained = classify(lo, hi, realized_score, tau)
                    status[user_i, schema_i, tau_i, code] += 1
                    if code == 1 and not is_covered:
                        uncovered[user_i, schema_i, tau_i] += 1
                    if is_false:
                        false_stable[user_i, schema_i, tau_i] += 1
                        if is_covered:
                            implication_violations += 1
                    if is_contained:
                        contained[user_i, schema_i, tau_i] += 1
            seen += 1
        if user_i and user_i % 200 == 0:
            source.flush()
            scores.flush()
            flags.flush()
            np.savez(
                counters_path, status=status, false_stable=false_stable, uncovered_stable=uncovered,
                contained=contained, covered=covered, supported=supported,
                max_abs_vs_cache=np.asarray([max_abs]), side_flips=np.asarray([side_flips]),
                future_violations=np.asarray([future_violations]),
                implication_violations=np.asarray([implication_violations]),
            )
            progress_path.write_text(json.dumps({"next_user": user_i + 1, "next_event": seen, "complete": False}) + "\n")
            print(f"users {user_i} events {seen} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    source.flush()
    scores.flush()
    flags.flush()
    progress_path.write_text(json.dumps({"next_user": len(users), "next_event": seen, "complete": True}) + "\n")
    np.savez(
        counters_path, status=status, false_stable=false_stable, uncovered_stable=uncovered,
        contained=contained, covered=covered, supported=supported,
        max_abs_vs_cache=np.asarray([max_abs]), side_flips=np.asarray([side_flips]),
        future_violations=np.asarray([future_violations]),
        implication_violations=np.asarray([implication_violations]),
    )
    return {
        "events": seen,
        "users": len(users),
        "status": status,
        "false_stable": false_stable,
        "uncovered_stable": uncovered,
        "contained": contained,
        "covered": covered,
        "supported": supported,
        "max_abs_vs_cache": max_abs,
        "side_flips": side_flips,
        "future_violations": future_violations,
        "implication_violations": implication_violations,
    }


def rows_from_counters(result: dict) -> list[dict]:
    rows = []
    status = result["status"]
    for schema_i, schema in enumerate(SCHEMAS):
        covered = int(result["covered"][:, schema_i].sum())
        supported = int(result["supported"][:, schema_i].sum())
        for tau_i, tau in enumerate(TAUS):
            unsup = int(status[:, schema_i, tau_i, 0].sum())
            stable = int(status[:, schema_i, tau_i, 1].sum())
            sett = int(status[:, schema_i, tau_i, 2].sum())
            n_u = int(result["uncovered_stable"][:, schema_i, tau_i].sum())
            n_fs = int(result["false_stable"][:, schema_i, tau_i].sum())
            contained = int(result["contained"][:, schema_i, tau_i].sum())
            kind = case_of(stable, n_u)
            if kind == "A":
                second = n_fs / n_u
            else:
                second = "N/A"
            rows.append({
                "schema": schema,
                "tau": tau,
                "valid_targets": VALID,
                "supported": supported,
                "unsupported": unsup,
                "stable": stable,
                "set_valued": sett,
                "event_nc": unsup + sett,
                "event_nc_rate": (unsup + sett) / VALID,
                "fingerprint_covered_supported": covered,
                "stable_uncovered": n_u,
                "false_stable": n_fs,
                "false_stable_over_stable": ratio(n_fs, stable),
                "false_stable_over_supported": ratio(n_fs, supported),
                "action_contained_supported": contained,
                "conditional_case": kind,
                "second_factor": second,
            })
    return rows


def write_table(rows: list[dict]) -> None:
    header = [
        "schema", "tau", "valid_targets", "supported", "unsupported", "stable", "set_valued",
        "event_nc", "event_nc_rate", "fingerprint_covered_supported", "stable_uncovered",
        "false_stable", "false_stable_over_stable", "false_stable_over_supported",
        "action_contained_supported", "conditional_case", "second_factor",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[name]) for name in header))
    (OUT / "threshold_false_stable_full_grid.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def replay_memmap(result: dict) -> dict:
    source, scores, flags = open_cache(VALID, True)
    status = np.zeros((4, 3, 3), dtype=np.int64)
    false_stable = np.zeros((4, 3), dtype=np.int64)
    uncovered = np.zeros((4, 3), dtype=np.int64)
    contained = np.zeros((4, 3), dtype=np.int64)
    covered = np.zeros(4, dtype=np.int64)
    for event_i in range(VALID):
        for schema_i in range(4):
            if flags[event_i, schema_i, 0] == 0:
                status[schema_i, :, 0] += 1
                continue
            if flags[event_i, schema_i, 1]:
                covered[schema_i] += 1
            lo, hi, realized = (float(scores[event_i, schema_i, k]) for k in range(3))
            for tau_i, tau in enumerate(TAUS):
                code, is_false, is_contained = classify(lo, hi, realized, tau)
                status[schema_i, tau_i, code] += 1
                if code == 1 and flags[event_i, schema_i, 1] == 0:
                    uncovered[schema_i, tau_i] += 1
                if is_false:
                    false_stable[schema_i, tau_i] += 1
                if is_contained:
                    contained[schema_i, tau_i] += 1
    mismatches = []
    live = result["status"].sum(axis=0)
    for schema_i, schema in enumerate(SCHEMAS):
        for tau_i, tau in enumerate(TAUS):
            for code in range(3):
                if int(live[schema_i, tau_i, code]) != int(status[schema_i, tau_i, code]):
                    mismatches.append({"schema": schema, "tau": tau, "code": code})
            if int(result["false_stable"][:, schema_i, tau_i].sum()) != int(false_stable[schema_i, tau_i]):
                mismatches.append({"schema": schema, "tau": tau, "field": "false_stable"})
            if int(result["uncovered_stable"][:, schema_i, tau_i].sum()) != int(uncovered[schema_i, tau_i]):
                mismatches.append({"schema": schema, "tau": tau, "field": "uncovered"})
    del source, scores, flags
    return {"memmap_replay_mismatches": mismatches}


def reproduction(rows: list[dict], stored_status: np.ndarray) -> dict:
    by_key = {(row["schema"], float(row["tau"])): row for row in rows}
    primary = []
    primary_pass = True
    for schema, expected in EXPECTED_080.items():
        row = by_key[(schema, 0.80)]
        observed = {
            "unsupported": row["unsupported"], "stable": row["stable"], "set_valued": row["set_valued"],
            "n_U": row["stable_uncovered"], "false_stable": row["false_stable"],
        }
        match = observed == expected
        primary_pass = primary_pass and match
        primary.append({"schema": schema, "expected": expected, "observed": observed, "match": match})
    nc = []
    nc_pass = True
    for tau, expected in EXPECTED_NC.items():
        s0 = by_key[("S0", tau)]["event_nc"]
        s1 = by_key[("S1", tau)]["event_nc"]
        delta = s1 - s0
        match = s0 == expected["S0"] and s1 == expected["S1"] and delta == expected["delta"]
        nc_pass = nc_pass and match
        nc.append({"tau": tau, "s0": s0, "s1": s1, "delta": delta, "expected": expected, "match": match})
    stored = []
    stored_pass = True
    for schema_i, schema in enumerate(SCHEMAS):
        for tau_i, tau in enumerate(TAUS):
            observed = [int(stored_status[:, schema_i, tau_i, code].sum()) for code in range(3)]
            recomputed = [by_key[(schema, tau)][name] for name in ("unsupported", "stable", "set_valued")]
            match = observed == recomputed
            stored_pass = stored_pass and match
            stored.append({"schema": schema, "tau": tau, "stored_user_status": observed, "recomputed": recomputed, "match": match})
    payload = {
        "tau_0_80_exact": primary_pass,
        "tau_0_80_cells": primary,
        "s0_s1_event_nc": nc_pass,
        "s0_s1_cells": nc,
        "stored_status_counts": stored_pass,
        "stored_status_cells": stored,
    }
    dump("threshold_reproduction_checks.json", payload)
    return payload


def decomposition(rows: list[dict], implication_violations: int) -> dict:
    cells = []
    identity_ok = True
    bounds_ok = True
    for row in rows:
        n_s = row["stable"]
        n_u = row["stable_uncovered"]
        n_fs = row["false_stable"]
        kind = row["conditional_case"]
        if kind == "A":
            product = (n_u / n_s) * (n_fs / n_u)
            identity = abs(n_fs / n_s - product) <= 1e-12
        elif kind == "B":
            identity = n_fs == 0 and row["false_stable_over_stable"] == 0 and row["second_factor"] == "N/A"
        else:
            identity = row["false_stable_over_stable"] == "N/A"
        if n_fs > n_u or n_fs < 0 or n_u < 0:
            bounds_ok = False
        identity_ok = identity_ok and identity
        contained_gap = row["supported"] - row["action_contained_supported"]
        cells.append({
            "schema": row["schema"], "tau": row["tau"], "n_S": n_s, "n_U": n_u, "n_FS": n_fs,
            "case": kind, "identity_holds": identity,
            "false_stable_equals_action_noncontainment": contained_gap == n_fs,
            "covered_but_false_stable_would_break_implication": implication_violations,
        })
    payload = {
        "implication": "FalseStable implies Stable and not fingerprint-covered. The converse is not assumed.",
        "implication_violations": implication_violations,
        "bounds_ok": bounds_ok,
        "identity_ok": identity_ok,
        "cells": cells,
        "note": "A noncovered realized fingerprint can still induce the certificate action. That event is n_U and not n_FS.",
    }
    dump("three_case_decomposition.json", payload)
    return payload


def independent_qa(pilot: NeuralPilot, grouped: dict, bundles: dict) -> dict:
    _source, scores, flags = open_cache(VALID, True)
    source = np.memmap(OUT / "private_event_cache/source_row.i8", dtype=np.int64, mode="r")
    index = {int(source[i]): i for i in range(VALID)}
    pool = []
    located = {}
    for uid, rows in grouped.items():
        for row in rows:
            if not row["valid"]:
                continue
            event_i = index.get(row["source_row"])
            if event_i is None:
                continue
            rank = hashlib.sha256(f"WSDM203-G11R1-QA|{row['source_row']}".encode()).hexdigest()
            pool.append((rank, uid, row, event_i))
            located[row["source_row"]] = (uid, row)
    pool.sort()

    def take(predicate, limit: int, chosen: set[int]) -> list[tuple]:
        out = []
        for item in pool:
            if len(out) >= limit:
                break
            if item[3] in chosen or not predicate(item[3]):
                continue
            out.append(item)
        return out

    chosen_ids: set[int] = set()
    strata = []
    near = take(lambda i: any(
        flags[i, s, 0] and min(abs(float(scores[i, s, 2]) - tau), abs(float(scores[i, s, 0]) - tau), abs(float(scores[i, s, 1]) - tau)) <= 1e-4
        for s in range(4) for tau in TAUS if np.isfinite(scores[i, s, 0])
    ), 40, chosen_ids)
    uncovered_contained = take(lambda i: any(
        flags[i, s, 0] and flags[i, s, 1] == 0 and classify(float(scores[i, s, 0]), float(scores[i, s, 1]), float(scores[i, s, 2]), 0.80)[2]
        and classify(float(scores[i, s, 0]), float(scores[i, s, 1]), float(scores[i, s, 2]), 0.80)[0] == 1
        for s in range(4)
    ), 40, chosen_ids)
    false_cells = take(lambda i: any(
        flags[i, s, 0] and classify(float(scores[i, s, 0]), float(scores[i, s, 1]), float(scores[i, s, 2]), 0.80)[1]
        for s in range(4)
    ), 40, chosen_ids)
    unsupported = take(lambda i: flags[i, 0, 0] == 0, 40, chosen_ids)
    for name, group in (
        ("near_threshold", near), ("uncovered_action_contained", uncovered_contained),
        ("false_stable_0_80", false_cells), ("unsupported_s0", unsupported),
    ):
        for item in group:
            chosen_ids.add(item[3])
        strata.append({"stratum": name, "available_selected": len(group)})
    filler = []
    for item in pool:
        if len(chosen_ids) >= 200:
            break
        if item[3] in chosen_ids:
            continue
        chosen_ids.add(item[3])
        filler.append(item)
    strata.append({"stratum": "hash_fill", "available_selected": len(filler)})
    selected = [item for item in pool if item[3] in chosen_ids][:200]
    max_diff = 0.0
    order_abs = 0.0
    token_mismatch = 0
    history_mismatch = 0
    future = 0
    current_leak = 0
    order_mismatch = 0
    interval_label_mismatch = 0
    serial_label_mismatch = 0
    false_stable_mismatch = 0
    checked = 0
    supported_intervals = 0
    model = pilot.model
    model.eval()
    for _rank, uid, row, event_i in selected:
        by_start, by_end = chronorep.order_valid(grouped[uid])
        prefix = chronorep.prefix_events(by_end, row["start"])
        direct = chronorep.direct_history(grouped[uid], row)
        prefix_sources = [item["source_row"] for item in by_end[max(0, chronorep.history_cut(by_end, row["start"]) - chronorep.MAX_EVAL_PREFIX_EVENTS) : chronorep.history_cut(by_end, row["start"])]]
        if prefix_sources != direct[-len(prefix_sources):]:
            history_mismatch += 1
        current = chronorep.as_event(row)
        left = pilot._tokens(prefix, current, False)
        full_s, full_a, _cut = matchedinfo.history_tokens(by_end, row["start"], pilot.skill2i, pilot.atype2i, model.outcome0, model.outcome1)
        actions = [pilot.atype2i.get(tok, 0) for tok in row["action_types"][:32]]
        sid = pilot.skill2i.get(row["skill_id"], 0)
        built = ((full_s + [sid] * len(actions))[-64:], (full_a + actions)[-64:])
        if built != left:
            token_mismatch += 1
        cut = chronorep.history_cut(by_end, row["start"])
        if any(item["end"] >= row["start"] for item in by_end[:cut]):
            future += 1
        length = min(32, len(row["action_types"]))
        if length and any(tok in (model.outcome0, model.outcome1) for tok in left[1][-length:]):
            current_leak += 1
        with torch.inference_mode():
            serial = float(model.forward_from_state(torch.tensor([left[0]]), torch.tensor([left[1]]), None)[0]) if left[0] else 0.5
            batched = float(chronorep.score_forward(model, [left], torch.device("cpu"))[0])
        max_diff = max(max_diff, abs(serial - batched))
        realized_ref = None
        for schema_i in range(4):
            if flags[event_i, schema_i, 0]:
                realized_ref = float(scores[event_i, schema_i, 2])
                break
        if realized_ref is not None:
            max_diff = max(max_diff, abs(serial - realized_ref), abs(batched - realized_ref))
            for tau in TAUS:
                if (serial >= tau) != (batched >= tau) or (serial >= tau) != (realized_ref >= tau):
                    serial_label_mismatch += 1
        ev = chronorep.as_event(row)
        own = chronorep.gru_fp(row)
        for schema_i, schema in enumerate(SCHEMAS):
            if not flags[event_i, schema_i, 0]:
                continue
            supported_intervals += 1
            slot = bundles[schema].get(chronorep.release_key(ev, schema))
            fps = list(slot.gru)
            forward_scores = chronorep.score_forward(model, [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in fps], torch.device("cpu")) if fps else np.zeros(0)
            reverse_scores = chronorep.score_forward(model, [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in reversed(fps)], torch.device("cpu")) if fps else np.zeros(0)
            if fps:
                fwd_lo, fwd_hi = float(np.min(forward_scores)), float(np.max(forward_scores))
                rev_lo, rev_hi = float(np.min(reverse_scores)), float(np.max(reverse_scores))
                order_abs = max(order_abs, abs(fwd_lo - rev_lo), abs(fwd_hi - rev_hi))
                for tau in TAUS:
                    if classify(fwd_lo, fwd_hi, serial, tau)[:2] != classify(rev_lo, rev_hi, serial, tau)[:2]:
                        order_mismatch += 1
            values = [float(score) for fp, score in zip(fps, forward_scores) if fp != own]
            if own in slot.gru:
                values.append(serial)
            if not values:
                values = [0.5]
            lo = float(min(values))
            hi = float(max(values))
            stored_lo = float(scores[event_i, schema_i, 0])
            stored_hi = float(scores[event_i, schema_i, 1])
            max_diff = max(max_diff, abs(lo - stored_lo), abs(hi - stored_hi))
            for tau in TAUS:
                code, is_false, _contained = classify(lo, hi, serial, tau)
                stored_code, stored_false, _stored_contained = classify(stored_lo, stored_hi, float(scores[event_i, schema_i, 2]), tau)
                if code != stored_code or is_false != stored_false:
                    interval_label_mismatch += 1
                if is_false != stored_false:
                    false_stable_mismatch += 1
        checked += 1
    payload = {
        "events": checked,
        "strata": strata,
        "supported_intervals_rescored": supported_intervals,
        "max_abs_diff": max_diff,
        "order_minmax_max_abs": order_abs,
        "numeric_tolerance": float(np.finfo(np.float32).eps),
        "tolerance_note": "Float32 evaluation can differ by one ulp across batch orders. A difference counts as a failure only when it changes a threshold label.",
        "token_mismatches": token_mismatch,
        "history_mismatches": history_mismatch,
        "future_or_unfinished": future,
        "current_outcome_leaks": current_leak,
        "order_interval_mismatches": order_mismatch,
        "interval_label_mismatches": interval_label_mismatch,
        "serial_batch_label_mismatches": serial_label_mismatch,
        "false_stable_mismatches": false_stable_mismatch,
        "checkpoint_sha256": CKPT_SHA,
        "evaluation_mode": not model.training,
        "pass": checked >= 200 and max_diff <= float(np.finfo(np.float32).eps) and token_mismatch == 0 and history_mismatch == 0 and future == 0 and current_leak == 0 and order_mismatch == 0 and interval_label_mismatch == 0 and serial_label_mismatch == 0 and false_stable_mismatch == 0,
    }
    dump("independent_scoring_qa.json", payload)
    del source, scores, flags
    return payload


def invariants(rows: list[dict], result: dict, repro: dict, qa: dict, ready: dict) -> dict:
    support_fixed = len({(row["schema"], row["supported"], row["unsupported"]) for row in rows}) == 12 or all(
        len({row["supported"] for row in rows if row["schema"] == schema}) == 1 for schema in SCHEMAS
    )
    accounting = all(row["unsupported"] + row["stable"] + row["set_valued"] == VALID and row["event_nc"] == row["unsupported"] + row["set_valued"] for row in rows)
    coverage_fixed = all(len({row["fingerprint_covered_supported"] for row in rows if row["schema"] == schema}) == 1 for schema in SCHEMAS)
    fs_bound = all(row["false_stable"] <= row["stable_uncovered"] for row in rows)
    s3 = [row for row in rows if row["schema"] == "S3"]
    s3_uncovered = all(row["stable_uncovered"] == 0 and row["false_stable"] == 0 for row in s3)
    no_missing = all(row[name] is not None and row[name] != "" for row in rows for name in row)
    payload = {
        "valid_targets_once": result["events"] == VALID,
        "invalid_excluded": INVALID,
        "source_events": SOURCE_EVENTS,
        "support_identical_across_thresholds": support_fixed,
        "status_accounting": accounting,
        "coverage_independent_of_tau": coverage_fixed,
        "false_stable_not_above_uncovered": fs_bound,
        "s3_uncovered_stable": s3_uncovered,
        "tau_0_80_reproduced": repro["tau_0_80_exact"],
        "s0_s1_reproduced": repro["s0_s1_event_nc"],
        "stored_status_reproduced": repro["stored_status_counts"],
        "future_violations": result["future_violations"],
        "implication_violations": result["implication_violations"],
        "realized_score_max_abs_vs_attncons_cache": result["max_abs_vs_cache"],
        "realized_score_threshold_side_flips": result["side_flips"],
        "independent_qa_pass": qa["pass"],
        "no_new_checkpoint": not any(path.suffix == ".pt" for path in OUT.rglob("*")),
        "training_occurred": False,
        "cache_was_sufficient": ready["sufficient_for_false_stable_at_arbitrary_tau"],
        "no_missing_cells": no_missing,
        "memmap_replay_mismatches": result.get("memmap_replay_mismatches", []),
    }
    ok = all([
        payload["valid_targets_once"], payload["status_accounting"], payload["support_identical_across_thresholds"],
        payload["coverage_independent_of_tau"], payload["false_stable_not_above_uncovered"], payload["s3_uncovered_stable"],
        payload["tau_0_80_reproduced"], payload["s0_s1_reproduced"], payload["stored_status_reproduced"],
        payload["future_violations"] == 0, payload["implication_violations"] == 0,
        payload["realized_score_threshold_side_flips"] == 0, payload["independent_qa_pass"],
        payload["no_new_checkpoint"], not payload["training_occurred"], payload["no_missing_cells"],
        payload["memmap_replay_mismatches"] == [],
    ])
    payload["status"] = "COMPLETE" if ok else "BLOCKED_SCIENTIFIC_INCONSISTENCY"
    dump("numerical_invariants.json", payload)
    return payload


def write_checksums() -> dict:
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({
            "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": chronorep.sha256_file(path),
        })
    payload = {
        "algorithm": "sha256",
        "excluded": ["results/assistments/attention_consumer/threshold_false_stable/output_checksums.json"],
        "n_files": len(files),
        "files": files,
    }
    dump("output_checksums.json", payload)
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    print("preflight", flush=True)
    pre = chronorep.preflight()
    ckpt_hash = chronorep.sha256_file(CKPT)
    pre["attention_checkpoint_sha256"] = ckpt_hash
    pre["attention_checkpoint_match"] = ckpt_hash == CKPT_SHA
    pre["thrfs_pass"] = bool(pre["pass"] and pre["attention_checkpoint_match"] and pre["parquet"]["confirmatory"]["sha256"] == "0193ddd7b26517f1b48591d09d86f5f3ce8b9930a1285006833fce4ea5096f99")
    dump("preflight.json", {
        "zip_sha256": pre["zip_sha256"], "zip_match": pre["zip_match"],
        "confirmatory": pre["parquet"]["confirmatory"],
        "development": pre["parquet"]["development"],
        "r7_core_sha256": pre["frozen"]["r7_core_sha256"], "r7_core_match": pre["frozen"]["r7_core_match"],
        "attention_checkpoint_sha256": ckpt_hash, "attention_checkpoint_match": pre["attention_checkpoint_match"],
        "pass": pre["thrfs_pass"],
    })
    if not pre["thrfs_pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_CACHE_OR_SCORING", "reason": "source or checkpoint identity failed"})
        write_checksums()
        raise SystemExit(2)
    ready = cache_readiness()
    print(ready["decision"], flush=True)
    pilot, n_params = load_model()
    if n_params != 88897:
        dump("numerical_invariants.json", {"status": "BLOCKED_SCIENTIFIC_INCONSISTENCY", "parameter_count": n_params})
        write_checksums()
        raise SystemExit(3)
    print("load confirmatory", flush=True)
    grouped = chronorep.load_confirmatory()
    invalid = sum(1 for rows in grouped.values() for row in rows if not row["valid"])
    valid = sum(1 for rows in grouped.values() for row in rows if row["valid"])
    if invalid != INVALID or valid != VALID or sum(len(rows) for rows in grouped.values()) != SOURCE_EVENTS:
        dump("numerical_invariants.json", {"status": "BLOCKED_SCIENTIFIC_INCONSISTENCY", "valid": valid, "invalid": invalid})
        write_checksums()
        raise SystemExit(4)
    print("development geometry", flush=True)
    geometry = chronorep.build_development(set())
    for bundle in geometry["bundles"].values():
        for slot in bundle.values():
            slot.deltas = None
            slot.pfa_list = []
    cached = pq.read_table(ATTNCONS / "private_score_caches/attention_event_scores.parquet", columns=["event_key", "score"])
    cached_keys = cached.column("event_key").to_numpy()
    cached_scores = cached.column("score").to_numpy()
    stored_status = np.load(ATTNCONS / "private_score_caches/attention_user_status.npz")["status"]
    progress_path = OUT / "score_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if progress.get("complete") and (OUT / "running_counters.npz").exists():
        print("reuse completed score cache", flush=True)
        previous = np.load(OUT / "running_counters.npz")
        result = {
            "events": int(progress["next_event"]),
            "status": previous["status"],
            "false_stable": previous["false_stable"],
            "uncovered_stable": previous["uncovered_stable"],
            "contained": previous["contained"],
            "covered": previous["covered"],
            "supported": previous["supported"],
            "max_abs_vs_cache": float(previous["max_abs_vs_cache"][0]),
            "side_flips": int(previous["side_flips"][0]),
            "future_violations": int(previous["future_violations"][0]),
            "implication_violations": int(previous["implication_violations"][0]),
            "memmap_replay_mismatches": [],
        }
    else:
        print("score", flush=True)
        result = score_population(pilot, grouped, geometry["bundles"], cached_keys, cached_scores)
        print("replay", flush=True)
        replay = replay_memmap(result)
        result["memmap_replay_mismatches"] = replay["memmap_replay_mismatches"]
    rows = rows_from_counters(result)
    write_table(rows)
    repro = reproduction(rows, stored_status)
    decomp = decomposition(rows, result["implication_violations"])
    print("qa", flush=True)
    qa = independent_qa(pilot, grouped, geometry["bundles"])
    inv = invariants(rows, result, repro, qa, ready)
    dump("execution_manifest.json", {
        "experiment": "pipeline",
        "script": "scripts/assistments/threshold_false_stable.py",
        "script_sha256": chronorep.sha256_file(Path(__file__).resolve()),
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "checkpoint_sha256": CKPT_SHA,
        "parameter_count": n_params,
        "training_occurred": False,
        "new_checkpoint_created": False,
        "elapsed_sec": time.time() - started,
        "status": inv["status"],
        "decomposition_identity": decomp["identity_ok"],
    })
    checksums = write_checksums()
    bad = []
    for item in checksums["files"]:
        path = ROOT / item["path"]
        if chronorep.sha256_file(path) != item["sha256"]:
            bad.append(item["path"])
    print(inv["status"], "checksum_mismatches", len(bad), flush=True)
    if inv["status"] != "COMPLETE" or bad:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
