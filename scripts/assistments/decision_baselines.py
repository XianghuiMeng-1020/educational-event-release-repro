#!/usr/bin/env python3
"""Decision-level baselines on the frozen post-ingest policy action.

Previously trained consumers and earlier results are not modified.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("chronorep", ROOT / "scripts/assistments/chronological_replication.py")
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)
from r7_core import NeuralPilot  # noqa: E402

OUT = ROOT / "results/assistments/decision_baselines"
CACHE = OUT / "private_scores"
CHRONOREP_CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
SCHEMAS = ("S0", "S1", "S2", "S3")
MODELS = ("TransparentActionPFA", "ActionTokenGRU")
TAU = 0.80
GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 1.00)
CHRONOREP_CERT = {
    ("TransparentActionPFA", "S0"): (20408, 916845, 2566, 45),
    ("TransparentActionPFA", "S1"): (77877, 861927, 15, 0),
    ("TransparentActionPFA", "S2"): (80790, 859025, 4, 0),
    ("TransparentActionPFA", "S3"): (100688, 839131, 0, 0),
    ("ActionTokenGRU", "S0"): (20408, 810576, 108835, 1077),
    ("ActionTokenGRU", "S1"): (77877, 846772, 15170, 252),
    ("ActionTokenGRU", "S2"): (80790, 844157, 14872, 250),
    ("ActionTokenGRU", "S3"): (100688, 839131, 0, 0),
}


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
    if not rows:
        (OUT / name).write_text("", encoding="utf-8")
        return
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value for key, value in row.items()})


class Slot:
    __slots__ = ("n", "n_users", "first", "pfa", "gru", "pfa_index", "pfa_counts", "pfa_deltas", "gru_index", "gru_counts", "modal_pfa", "modal_gru", "pfa_tie", "gru_tie")

    def __init__(self) -> None:
        self.n = 0
        self.n_users = 0
        self.first = None
        self.pfa = {}
        self.gru = {}


def supported_slot(slot: Slot | None) -> bool:
    return slot is not None and (slot.n_users >= 2 or slot.n >= 2)


def modal_index(freq: dict) -> tuple[int, bool]:
    items = list(freq)
    counts = [freq[fp] for fp in items]
    best = max(counts)
    winners = [i for i, count in enumerate(counts) if count == best]
    winners.sort(key=lambda i: repr(items[i]))
    return winners[0], len(winners) > 1


def finalize(slot: Slot) -> None:
    pfa_items = list(slot.pfa)
    slot.pfa_index = {fp: i for i, fp in enumerate(pfa_items)}
    slot.pfa_counts = np.asarray([slot.pfa[fp] for fp in pfa_items], dtype=np.int32)
    slot.pfa_deltas = np.asarray([chronorep.fp_delta(fp) for fp in pfa_items], dtype=np.float64) if pfa_items else np.zeros((0, 7))
    slot.modal_pfa, slot.pfa_tie = modal_index(slot.pfa)
    gru_items = list(slot.gru)
    slot.gru_index = {fp: i for i, fp in enumerate(gru_items)}
    slot.gru_counts = np.asarray([slot.gru[fp] for fp in gru_items], dtype=np.int32)
    slot.modal_gru, slot.gru_tie = modal_index(slot.gru)
    slot.pfa = None
    slot.gru = None


def mastery_vec(base: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    state = deltas + base.reshape(1, 7)
    n = state[:, 0]
    out = np.empty(len(state), dtype=np.float64)
    cold = n <= 0
    out[cold] = 0.5
    live = ~cold
    if np.any(live):
        s = state[live]
        nn = s[:, 0]
        score = (s[:, 1] + 1.0) / (nn + 2.0)
        score -= 0.05 * s[:, 2] / nn
        score -= 0.04 * np.maximum(s[:, 3] / nn - 1.0, 0.0)
        score -= 0.03 * np.maximum(s[:, 4] / nn - 1.0, 0.0)
        score -= 0.06 * s[:, 5] / nn
        score -= 0.05 * s[:, 6] / nn
        out[live] = np.clip(score, 0.01, 0.99)
    return out


def build_development() -> tuple[dict, dict]:
    bundles = {schema: {} for schema in SCHEMAS}
    signatures = set()
    duplicate = 0
    events = 0
    parquet = __import__("pyarrow.parquet").parquet.ParquetFile(chronorep.RECON / "development.parquet")
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            events += 1
            item = chronorep.compact_row(row, int(start), int(end))
            ev = chronorep.as_event(item)
            digest = hashlib.blake2s(repr((ev.problem_id, ev.correct, ev.hint_count, ev.attempt_count, ev.first_action, ev.action_types, tuple(str(tok) if tok is not None else "" for tok in (row["action_timestamps"] or [])))).encode(), digest_size=16).digest()
            duplicate += digest in signatures
            signatures.add(digest)
            pfa = chronorep.pfa_fp(item)
            gru = chronorep.gru_fp(item)
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                slot = bundles[schema].get(key)
                if slot is None:
                    slot = Slot()
                    bundles[schema][key] = slot
                slot.n += 1
                if slot.n_users < 2:
                    if slot.first is None:
                        slot.first = item["user_id"]
                        slot.n_users = 1
                    elif slot.first != item["user_id"]:
                        slot.n_users = 2
                        slot.first = None
                slot.pfa[pfa] = slot.pfa.get(pfa, 0) + 1
                slot.gru[gru] = slot.gru.get(gru, 0) + 1
        if events % 1000000 == 0:
            print(f"development {events} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    geometry = []
    for schema, bundle in bundles.items():
        pfa_counts = []
        gru_counts = []
        pfa_ties = gru_ties = single_pfa = single_gru = 0
        max_pfa = max_gru = 0
        for slot in bundle.values():
            finalize(slot)
            if int(slot.pfa_counts.sum()) != slot.n or int(slot.gru_counts.sum()) != slot.n:
                raise RuntimeError("fingerprint frequency does not conserve event mass")
            pfa_counts.append(len(slot.pfa_counts))
            gru_counts.append(len(slot.gru_counts))
            pfa_ties += int(slot.pfa_tie)
            gru_ties += int(slot.gru_tie)
            single_pfa += int(len(slot.pfa_counts) == 1)
            single_gru += int(len(slot.gru_counts) == 1)
            max_pfa = max(max_pfa, int(slot.pfa_counts.max()))
            max_gru = max(max_gru, int(slot.gru_counts.max()))
        geometry.append({
            "schema": schema, "keys": len(bundle), "development_events": events,
            "duplicate_rich_signatures": duplicate, "unique_rich_signatures": len(signatures),
            "mean_pfa_fingerprints": float(np.mean(pfa_counts)), "mean_gru_fingerprints": float(np.mean(gru_counts)),
            "keys_with_pfa_frequency_tie": pfa_ties, "keys_with_gru_frequency_tie": gru_ties,
            "keys_with_single_pfa_fingerprint": single_pfa, "keys_with_single_gru_fingerprint": single_gru,
            "max_pfa_fingerprint_frequency": max_pfa, "max_gru_fingerprint_frequency": max_gru,
            "rich_signature_frequency_uniformly_one": duplicate == 0 and len(signatures) == events,
        })
    summary = {"events": events, "duplicate_rich_signatures": duplicate, "unique_rich_signatures": len(signatures), "key_counts": {schema: len(bundle) for schema, bundle in bundles.items()}}
    return bundles, {"summary": summary, "rows": geometry}


def load_split(split: str) -> dict:
    grouped = defaultdict(list)
    parquet = __import__("pyarrow.parquet").parquet.ParquetFile(chronorep.RECON / f"{split}.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            grouped[str(row["user_id"])].append(chronorep.compact_row(row, int(start), int(end)))
        seen += len(rows)
        if seen % 400000 == 0:
            print(f"load_{split} {seen}", flush=True)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["source_row"])
    return grouped


def score_split(split: str, grouped: dict, bundles: dict, pilot: NeuralPilot) -> dict:
    users = sorted(grouped)
    valid_rows = []
    for uid in users:
        by_start, _by_end = chronorep.order_valid(grouped[uid])
        for row in by_start:
            valid_rows.append((uid, row["source_row"]))
    n = len(valid_rows)
    user_lookup = {uid: i for i, uid in enumerate(users)}
    user_index = np.empty(n, dtype=np.int32)
    source_row = np.empty(n, dtype=np.int64)
    y = np.empty((n, 2), dtype=np.uint8)
    realized_score = np.empty((n, 2), dtype=np.float64)
    p_progress = np.full((n, 2, 4), 0.5, dtype=np.float32)
    n_dev = np.zeros((n, 2, 4), dtype=np.int32)
    n_prog = np.zeros((n, 2, 4), dtype=np.int32)
    class_status = np.zeros((n, 2, 4), dtype=np.uint8)
    in_lib = np.zeros((n, 2, 4), dtype=np.uint8)
    modal = np.zeros((n, 2, 4), dtype=np.uint8)
    modal_diff = np.zeros((n, 2, 4), dtype=np.uint8)
    cert = np.zeros((n, 2, 4), dtype=np.uint8)
    qa_ids = sorted(range(n), key=lambda i: hashlib.sha256(f"WSDM203-G15-QA|{split}|{valid_rows[i][1]}".encode()).hexdigest())[:300]
    qa_set = set(qa_ids)
    qa = []
    future = schema_label_changes = 0
    scored_pfa = scored_gru = 0
    cursor_event = 0
    direct_gap = 0.0
    direct_action_mismatch = 0
    direct_checked = 0
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
            slots = []
            gru_needed = {own_gru: None}
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                slot = bundles[schema].get(key)
                slots.append(slot)
                if slot is not None:
                    gru_needed.update(slot.gru_index)
            pairs = [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in gru_needed]
            gru_scores = chronorep.score_forward(pilot.model, pairs, pilot.device)
            gru_map = {fp: float(score) for fp, score in zip(gru_needed, gru_scores)}
            gru_realized = gru_map[own_gru]
            y_models = (1 if pfa_realized >= TAU else 0, 1 if gru_realized >= TAU else 0)
            pos = cursor_event
            user_index[pos] = user_lookup[uid]
            source_row[pos] = row["source_row"]
            y[pos] = y_models
            realized_score[pos] = (pfa_realized, gru_realized)
            for schema_i, slot in enumerate(slots):
                for model_i, (own, realized) in enumerate(((own_pfa, pfa_realized), (own_gru, gru_realized))):
                    if slot is None:
                        p_progress[pos, model_i, schema_i] = 0.5
                        continue
                    if model_i == 0:
                        scores = mastery_vec(base, slot.pfa_deltas)
                        counts = slot.pfa_counts
                        inside = own in slot.pfa_index
                        modal_i = slot.modal_pfa
                        tie = slot.pfa_tie
                        scored_pfa += len(scores)
                    else:
                        scores = np.asarray([gru_map[fp] for fp in slot.gru_index], dtype=np.float64)
                        counts = slot.gru_counts
                        inside = own in slot.gru_index
                        modal_i = slot.modal_gru
                        tie = slot.gru_tie
                        scored_gru += len(scores)
                    progress = scores >= TAU
                    votes = int(counts[progress].sum()) if len(counts) else 0
                    total = int(slot.n)
                    p = (votes + 1) / (total + 2)
                    majority = 1 if votes * 2 > total else 0
                    modal_action = 1 if scores[modal_i] >= TAU else 0
                    p_progress[pos, model_i, schema_i] = p
                    n_dev[pos, model_i, schema_i] = total
                    n_prog[pos, model_i, schema_i] = votes
                    class_status[pos, model_i, schema_i] = 2 if supported_slot(slot) else 1
                    in_lib[pos, model_i, schema_i] = int(inside)
                    modal[pos, model_i, schema_i] = modal_action + 1
                    modal_diff[pos, model_i, schema_i] = int(modal_action != majority)
                    if class_status[pos, model_i, schema_i] == 2:
                        lo = float(scores.min())
                        hi = float(scores.max())
                        if lo >= TAU:
                            cert[pos, model_i, schema_i] = 1
                        elif hi < TAU:
                            cert[pos, model_i, schema_i] = 2
                        else:
                            cert[pos, model_i, schema_i] = 3
                    if inside and abs(float(scores[slot.pfa_index[own] if model_i == 0 else slot.gru_index[own]]) - realized) > 1e-8:
                        schema_label_changes += 1
            if y[pos, 0] != y_models[0] or y[pos, 1] != y_models[1]:
                schema_label_changes += 1
            if pos in qa_set:
                qa.append({
                    "split": split, "source_row": int(row["source_row"]), "history": cursor,
                    "future_in_prefix": int(any(item["end"] >= row["start"] for item in by_end[:cursor])),
                    "pfa_label": int(y[pos, 0]), "gru_label": int(y[pos, 1]),
                    "class_status": class_status[pos].tolist(), "in_library": in_lib[pos].tolist(),
                    "probability_gap": float(np.max(np.abs(p_progress[pos] + (1.0 - p_progress[pos]) - 1.0))),
                    "modal_tie_ignored_labels": True,
                })
            if direct_checked < 200 and pos in qa_set:
                scalar = float(chronorep.mastery_of(base + np.asarray(chronorep.fp_delta(own_pfa), dtype=np.float64)))
                branch = float(pilot.mastery_branch(prefix, ev))
                direct_gap = max(direct_gap, abs(scalar - pfa_realized), abs(branch - gru_realized))
                if (scalar >= TAU) != (pfa_realized >= TAU) or (branch >= TAU) != (gru_realized >= TAU):
                    direct_action_mismatch += 1
                direct_checked += 1
            cursor_event += 1
        if user_i and user_i % 250 == 0:
            print(f"{split}_users {user_i} events {cursor_event} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    return {
        "users": len(users), "events": n, "user_index": user_index, "source_row": source_row, "y": y,
        "realized_score": realized_score, "p_progress": p_progress, "n_dev": n_dev, "n_progress": n_prog,
        "class_status": class_status, "in_library": in_lib, "modal": modal, "modal_majority_disagreement": modal_diff,
        "certificate": cert, "future": future, "schema_label_changes": schema_label_changes,
        "scored_pfa": scored_pfa, "scored_gru": scored_gru, "qa": qa, "direct_checked": direct_checked,
        "direct_gap": direct_gap, "direct_action_mismatch": direct_action_mismatch,
    }


def save_cache(split: str, payload: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = {key: payload[key] for key in ("user_index", "source_row", "y", "realized_score", "p_progress", "n_dev", "n_progress", "class_status", "in_library", "modal", "modal_majority_disagreement", "certificate")}
    np.savez_compressed(CACHE / f"{split}_decisions.npz", **arrays)
    meta = {key: payload[key] for key in ("users", "events", "future", "schema_label_changes", "scored_pfa", "scored_gru", "qa", "direct_checked", "direct_gap", "direct_action_mismatch")}
    (CACHE / f"{split}_meta.json").write_text(json.dumps(meta, default=_json), encoding="utf-8")


def load_cache(split: str) -> dict:
    arrays = np.load(CACHE / f"{split}_decisions.npz")
    meta = json.loads((CACHE / f"{split}_meta.json").read_text(encoding="utf-8"))
    payload = {key: arrays[key] for key in arrays.files}
    payload.update(meta)
    return payload


def cert_counts(payload: dict) -> dict:
    out = {}
    cert = payload["certificate"]
    y = payload["y"]
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            code = cert[:, model_i, schema_i]
            label = y[:, model_i]
            stable = (code == 1) | (code == 2)
            stable_action = np.where(code == 1, 1, 0)
            false = int(np.sum(stable & (stable_action != label)))
            out[(model, schema)] = (int(np.sum(code == 0)), int(np.sum(stable)), int(np.sum(code == 3)), false)
    return out


def selective_tables(payload: dict, population: str) -> tuple[list[dict], dict]:
    rows = []
    chosen = {}
    y = payload["y"]
    p = payload["p_progress"]
    user = payload["user_index"]
    n_users = int(payload["users"])
    rng = np.random.default_rng(20315)
    draws = rng.integers(0, n_users, size=(1000, n_users))
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            prob = p[:, model_i, schema_i].astype(np.float64)
            label = y[:, model_i]
            confidence = np.maximum(prob, 1.0 - prob)
            base = np.where(prob > 0.5, 1, 0).astype(np.uint8)
            issued_by = []
            err_by = []
            point = []
            for threshold in GRID:
                issued = confidence >= threshold
                correct = issued & (base == label)
                wrong = issued & (base != label)
                issued_user = np.bincount(user, weights=issued.astype(np.float64), minlength=n_users)
                wrong_user = np.bincount(user, weights=wrong.astype(np.float64), minlength=n_users)
                issued_by.append(issued_user)
                err_by.append(wrong_user)
                point.append((int(issued.sum()), int(correct.sum()), int(wrong.sum())))
            uppers = []
            nonempty_reps = []
            for issued_user, wrong_user in zip(issued_by, err_by):
                rates = []
                empty = 0
                for draw in draws:
                    weight = np.bincount(draw, minlength=n_users)
                    denom = float(weight @ issued_user)
                    if denom <= 0:
                        empty += 1
                        continue
                    rates.append(float(weight @ wrong_user) / denom)
                nonempty_reps.append(1000 - empty)
                uppers.append(float(np.percentile(rates, 95)) if rates else None)
            qualifying = [i for i, (upper, counts) in enumerate(zip(uppers, point)) if upper is not None and upper <= 0.05 and counts[0] > 0]
            # Secondary alphas are recorded, but the primary selector is alpha 0.05.
            selected = None
            if qualifying:
                selected = max(qualifying, key=lambda i: (point[i][0], GRID[i]))
            chosen[(model, schema, 0.05)] = None if selected is None else GRID[selected]
            for alpha in (0.01, 0.10):
                pool = [i for i, (upper, counts) in enumerate(zip(uppers, point)) if upper is not None and upper <= alpha and counts[0] > 0]
                chosen[(model, schema, alpha)] = None if not pool else GRID[max(pool, key=lambda i: (point[i][0], GRID[i]))]
            n_valid = len(label)
            for i, threshold in enumerate(GRID):
                issued_n, correct_n, wrong_n = point[i]
                rows.append({
                    "population": population, "consumer": model, "schema": schema, "threshold": threshold,
                    "alpha_reference": 0.05, "issued": issued_n, "correct": correct_n, "incorrect": wrong_n,
                    "coverage": issued_n / n_valid, "selective_risk": (wrong_n / issued_n) if issued_n else None,
                    "upper_error_p95": uppers[i], "nonempty_bootstrap_replicates": nonempty_reps[i],
                    "qualifies_alpha_0.05": int(i in qualifying), "selected_alpha_0.05": int(selected == i),
                    "selected_alpha_0.01": int(chosen[(model, schema, 0.01)] == threshold),
                    "selected_alpha_0.10": int(chosen[(model, schema, 0.10)] == threshold),
                    "bootstrap_seed": 20315, "bootstrap_replicates": 1000, "valid_targets": n_valid,
                })
    return rows, chosen


def conformal_fit(payload: dict) -> dict:
    y = payload["y"]
    p = payload["p_progress"].astype(np.float64)
    user = payload["user_index"]
    n_users = int(payload["users"])
    present = np.zeros(n_users, dtype=bool)
    present[user] = True
    result = {"calibration_users": n_users, "users_with_valid_events": int(present.sum()), "users_without_valid_events": int((~present).sum()), "levels": {}}
    for alpha in (0.05, 0.10):
        level = {}
        n = int(present.sum())
        k = int(math.ceil((n + 1) * (1.0 - alpha)))
        for model_i, model in enumerate(MODELS):
            for schema_i, schema in enumerate(SCHEMAS):
                prob = p[:, model_i, schema_i]
                label = y[:, model_i]
                realized_p = np.where(label == 1, prob, 1.0 - prob)
                score = 1.0 - realized_p
                maxima = np.full(n_users, -np.inf)
                np.maximum.at(maxima, user, score)
                observed = np.sort(maxima[present])
                q = float("inf") if k > n else float(observed[k - 1])
                level[f"{model}|{schema}"] = {"n": n, "k": k, "quantile": q, "k_exceeds_n": k > n}
        result["levels"][str(alpha)] = level
    return result


def apply_selective(payload: dict, thresholds: dict, alpha: float) -> np.ndarray:
    pred = np.zeros(payload["y"].shape[:1] + (2, 4), dtype=np.uint8)
    p = payload["p_progress"]
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            threshold = thresholds[(model, schema, alpha)]
            if threshold is None:
                continue
            prob = p[:, model_i, schema_i]
            confidence = np.maximum(prob, 1.0 - prob)
            base = np.where(prob > 0.5, 1, 0)
            issued = confidence >= threshold
            pred[:, model_i, schema_i] = np.where(issued, base + 1, 0)
    return pred


def apply_conformal(payload: dict, fit: dict, alpha: float) -> np.ndarray:
    sets = np.zeros(payload["y"].shape[:1] + (2, 4, 2), dtype=np.uint8)
    p = payload["p_progress"].astype(np.float64)
    level = fit["levels"][str(alpha)]
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            q = level[f"{model}|{schema}"]["quantile"]
            prob = p[:, model_i, schema_i]
            if math.isinf(q):
                sets[:, model_i, schema_i, :] = 1
            else:
                sets[:, model_i, schema_i, 1] = (1.0 - prob) <= q
                sets[:, model_i, schema_i, 0] = prob <= q
    return sets


def ratio(num: int, den: int):
    return None if den == 0 else num / den


def decision_rows(split: str, payload: dict, selective: dict[float, np.ndarray], conformal: dict[float, np.ndarray]) -> tuple[list, list, list, list, list]:
    y = payload["y"]
    modal = payload["modal"]
    status = payload["class_status"]
    library = payload["in_library"]
    cert = payload["certificate"]
    modal_rows, selective_rows, conformal_rows, reference_rows, strata = [], [], [], [], []
    n = len(y)
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            label = y[:, model_i]
            issued = modal[:, model_i, schema_i] > 0
            pred = modal[:, model_i, schema_i] - 1
            correct = issued & (pred == label)
            support = status[:, model_i, schema_i] == 2
            observed = status[:, model_i, schema_i] == 1
            missing = status[:, model_i, schema_i] == 0
            inside = library[:, model_i, schema_i] == 1
            modal_rows.append({
                "population": split, "consumer": model, "schema": schema, "valid_targets": n,
                "issued": int(issued.sum()), "no_development_class": int(missing.sum()),
                "correct": int(correct.sum()), "incorrect": int((issued & ~correct).sum()),
                "accuracy_among_issued": ratio(int(correct.sum()), int(issued.sum())),
                "full_population_correct_fraction": int(correct.sum()) / n,
                "supported_events": int(support.sum()), "supported_issued": int((support & issued).sum()),
                "supported_correct": int((support & correct).sum()),
                "supported_accuracy": ratio(int((support & correct).sum()), int((support & issued).sum())),
                "unsupported_events": int(observed.sum()), "unsupported_issued": int((observed & issued).sum()),
                "unsupported_correct": int((observed & correct).sum()),
                "unsupported_accuracy": ratio(int((observed & correct).sum()), int((observed & issued).sum())),
                "out_of_library_events": int((~inside & ~missing).sum()),
                "out_of_library_errors": int((~inside & issued & ~correct).sum()),
                "modal_majority_disagreements": int(payload["modal_majority_disagreement"][:, model_i, schema_i].sum()),
            })
            code = cert[:, model_i, schema_i]
            stable = (code == 1) | (code == 2)
            stable_pred = np.where(code == 1, 1, 0)
            stable_correct = stable & (stable_pred == label)
            reference_rows.append({
                "population": split, "consumer": model, "schema": schema, "tau": TAU, "valid_targets": n,
                "unsupported": int((code == 0).sum()), "stable": int(stable.sum()), "set_valued": int((code == 3).sum()),
                "false_stable": int((stable & ~stable_correct).sum()),
                "singleton_correct": int(stable_correct.sum()),
                "set_contains_realized": int((stable_correct | (code == 3)).sum()),
            })
            for alpha, pred_s in selective.items():
                choice = pred_s[:, model_i, schema_i]
                issued_s = choice > 0
                correct_s = issued_s & ((choice - 1) == label)
                selective_rows.append({
                    "population": split, "consumer": model, "schema": schema, "alpha": alpha, "valid_targets": n,
                    "issued": int(issued_s.sum()), "abstained": int((~issued_s).sum()),
                    "correct": int(correct_s.sum()), "incorrect": int((issued_s & ~correct_s).sum()),
                    "selective_error": ratio(int((issued_s & ~correct_s).sum()), int(issued_s.sum())),
                    "correct_action_coverage": int(correct_s.sum()) / n, "issuance_rate": int(issued_s.sum()) / n,
                })
            for alpha, sets in conformal.items():
                member = sets[:, model_i, schema_i]
                size = member.sum(axis=1)
                contained = member[np.arange(n), label] == 1
                singleton = size == 1
                singleton_action = np.where(member[:, 1] == 1, 1, 0)
                singleton_correct = singleton & (singleton_action == label)
                user_all = []
                conformal_rows.append({
                    "population": split, "consumer": model, "schema": schema, "alpha": alpha, "valid_targets": n,
                    "event_coverage": int(contained.sum()) / n, "event_covered": int(contained.sum()),
                    "singleton_events": int(singleton.sum()), "correct_singletons": int(singleton_correct.sum()),
                    "false_singletons": int((singleton & ~singleton_correct).sum()),
                    "false_singleton_fraction_among_singletons": ratio(int((singleton & ~singleton_correct).sum()), int(singleton.sum())),
                    "correct_singleton_coverage": int(singleton_correct.sum()) / n,
                    "mean_set_size": float(size.mean()), "empty_sets": int((size == 0).sum()),
                    "two_action_sets": int((size == 2).sum()), "empty_rate": int((size == 0).sum()) / n,
                    "two_action_rate": int((size == 2).sum()) / n, "singleton_rate": int(singleton.sum()) / n,
                })
                user_all.append((contained, payload["user_index"]))
                # learner coverage is filled by a dedicated table
                _ = user_all
            for method, issued_m, correct_m, contained_m in (
                ("ModalFingerprintImputation", issued, correct, correct),
                ("EmpiricalCertificate", stable, stable_correct, stable_correct | (code == 3)),
            ):
                for stratum, mask in (
                    ("all", np.ones(n, dtype=bool)),
                    ("supported", support),
                    ("observed_but_unsupported", observed),
                    ("no_development_class", missing),
                    ("in_library", inside),
                    ("out_of_library", ~inside & ~missing),
                ):
                    strata.append({
                        "population": split, "consumer": model, "schema": schema, "method": method, "stratum": stratum,
                        "events": int(mask.sum()), "singleton_issued": int((mask & issued_m).sum()),
                        "singleton_correct": int((mask & correct_m).sum()),
                        "set_contains_realized": int((mask & contained_m).sum()),
                    })
    return modal_rows, selective_rows, conformal_rows, reference_rows, strata


def learner_coverage(payload: dict, sets: np.ndarray, alpha: float) -> list[dict]:
    rows = []
    user = payload["user_index"]
    y = payload["y"]
    n_users = int(payload["users"])
    present = np.zeros(n_users, dtype=bool)
    present[user] = True
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            contained = sets[:, model_i, schema_i][np.arange(len(user)), y[:, model_i]] == 1
            ok = np.ones(n_users, dtype=bool)
            np.logical_and.at(ok, user, contained)
            rows.append({
                "population": "confirmatory", "consumer": model, "schema": schema, "alpha": alpha,
                "learners_in_split": n_users, "learners_with_valid_events": int(present.sum()),
                "learners_without_valid_events": int((~present).sum()),
                "learners_all_events_covered": int(np.sum(ok & present)),
                "learner_all_event_coverage": ratio(int(np.sum(ok & present)), int(present.sum())),
            })
    return rows


def bootstrap(payload: dict, selective_pred: np.ndarray, conformal_sets: np.ndarray) -> list[dict]:
    y = payload["y"]
    user = payload["user_index"]
    n_users = int(payload["users"])
    n = len(user)
    rng = np.random.default_rng(20316)
    draws = rng.integers(0, n_users, size=(2000, n_users))
    rows = []
    for model_i, model in enumerate(MODELS):
        label = y[:, model_i]
        for schema_i, schema in enumerate(SCHEMAS):
            methods = {}
            modal = payload["modal"][:, model_i, schema_i]
            methods["ModalFingerprintImputation"] = (modal > 0, (modal - 1) == label, (modal - 1) == label)
            sel = selective_pred[:, model_i, schema_i]
            methods["CalibratedSelectiveAction"] = (sel > 0, (sel - 1) == label, (sel - 1) == label)
            member = conformal_sets[:, model_i, schema_i]
            size = member.sum(axis=1)
            singleton = size == 1
            singleton_action = np.where(member[:, 1] == 1, 1, 0)
            contained = member[np.arange(n), label] == 1
            methods["UserClusterSplitConformalActionSet"] = (singleton, singleton & (singleton_action == label), contained)
            code = payload["certificate"][:, model_i, schema_i]
            stable = (code == 1) | (code == 2)
            stable_action = np.where(code == 1, 1, 0)
            methods["EmpiricalCertificate"] = (stable, stable & (stable_action == label), stable & (stable_action == label) | (code == 3))
            user_ok = np.ones(n_users, dtype=np.float64)
            present = np.zeros(n_users, dtype=bool)
            present[user] = True
            np.logical_and.at(user_ok, user, contained)
            stored = {}
            for name, (issued, correct, cover) in methods.items():
                issued_u = np.bincount(user, weights=issued.astype(np.float64), minlength=n_users)
                correct_u = np.bincount(user, weights=correct.astype(np.float64), minlength=n_users)
                cover_u = np.bincount(user, weights=cover.astype(np.float64), minlength=n_users)
                events_u = np.bincount(user, minlength=n_users).astype(np.float64)
                wrong_u = issued_u - correct_u
                bag = {"issuance": [], "error": [], "correct_coverage": [], "set_coverage": []}
                for draw in draws:
                    weight = np.bincount(draw, minlength=n_users).astype(np.float64)
                    ev = float(weight @ events_u)
                    iss = float(weight @ issued_u)
                    cor = float(weight @ correct_u)
                    cov = float(weight @ cover_u)
                    bag["issuance"].append(iss / ev)
                    bag["correct_coverage"].append(cor / ev)
                    bag["set_coverage"].append(cov / ev)
                    bag["error"].append(None if iss <= 0 else float(weight @ wrong_u) / iss)
                if name == "UserClusterSplitConformalActionSet":
                    bag["user_coverage"] = []
                    denom_users = float(present.sum())
                    for draw in draws:
                        weight = np.bincount(draw, minlength=n_users).astype(np.float64)
                        bag["user_coverage"].append(float(weight[present] @ user_ok[present]) / float(weight[present].sum()) if denom_users else None)
                stored[name] = (bag, int(issued.sum()), int(correct.sum()), int(cover.sum()), int(np.sum(issued & correct)))
            points = {}
            for name, (bag, issued_n, correct_n, cover_n, _same) in stored.items():
                points[name] = {
                    "singleton_issuance": (issued_n / n, issued_n, n, bag["issuance"]),
                    "correct_action_coverage": (correct_n / n, correct_n, n, bag["correct_coverage"]),
                    "action_set_coverage": (cover_n / n, cover_n, n, bag["set_coverage"]),
                }
                if issued_n:
                    wrong_n = issued_n - correct_n
                    points[name]["issued_action_error"] = (wrong_n / issued_n, wrong_n, issued_n, [value for value in bag["error"] if value is not None])
                if "user_coverage" in bag:
                    num = int(np.sum((user_ok == 1) & present))
                    den = int(present.sum())
                    points[name]["conformal_user_all_event_coverage"] = (num / den, num, den, bag["user_coverage"])
            for name, quantities in points.items():
                for quantity, (point, num, den, values) in quantities.items():
                    arr = np.asarray(values, dtype=np.float64)
                    low, high = np.percentile(arr, [2.5, 97.5])
                    rows.append({
                        "consumer": model, "schema": schema, "method": name, "quantity": quantity, "point": point,
                        "numerator": num, "denominator": den, "ci_low": float(low), "ci_high": float(high),
                        "replicates_used": int(len(arr)), "replicates_requested": 2000, "seed": 20316, "reference": "self",
                    })
            base = points["EmpiricalCertificate"]["correct_action_coverage"][3]
            for name in ("ModalFingerprintImputation", "CalibratedSelectiveAction", "UserClusterSplitConformalActionSet"):
                diff = np.asarray(points[name]["correct_action_coverage"][3]) - np.asarray(base)
                low, high = np.percentile(diff, [2.5, 97.5])
                point = points[name]["correct_action_coverage"][0] - points["EmpiricalCertificate"]["correct_action_coverage"][0]
                rows.append({
                    "consumer": model, "schema": schema, "method": name, "quantity": "correct_action_coverage_minus_certificate",
                    "point": point, "numerator": points[name]["correct_action_coverage"][1], "denominator": n,
                    "ci_low": float(low), "ci_high": float(high), "replicates_used": 2000, "replicates_requested": 2000,
                    "seed": 20316, "reference": "EmpiricalCertificate",
                })
            if "issued_action_error" in points["EmpiricalCertificate"]:
                cert_err = points["EmpiricalCertificate"]["issued_action_error"][3]
                for name in ("ModalFingerprintImputation", "CalibratedSelectiveAction", "UserClusterSplitConformalActionSet"):
                    if "issued_action_error" not in points[name]:
                        continue
                    # Replicates were filtered independently. Pair by recomputing is not available here,
                    # so the error-difference interval is omitted when the nonempty replicate sets differ.
                    if len(points[name]["issued_action_error"][3]) == len(cert_err) == 2000:
                        diff = np.asarray(points[name]["issued_action_error"][3]) - np.asarray(cert_err)
                        low, high = np.percentile(diff, [2.5, 97.5])
                        point = points[name]["issued_action_error"][0] - points["EmpiricalCertificate"]["issued_action_error"][0]
                        rows.append({
                            "consumer": model, "schema": schema, "method": name, "quantity": "issued_action_error_minus_certificate",
                            "point": point, "numerator": points[name]["issued_action_error"][1], "denominator": points[name]["issued_action_error"][2],
                            "ci_low": float(low), "ci_high": float(high), "replicates_used": 2000, "replicates_requested": 2000,
                            "seed": 20316, "reference": "EmpiricalCertificate",
                        })
    return rows


def matched_rows(modal_rows, selective_rows, conformal_rows, reference_rows) -> list[dict]:
    out = []
    for row in modal_rows:
        if row["population"] != "confirmatory":
            continue
        out.append({
            "consumer": row["consumer"], "schema": row["schema"], "method": "ModalFingerprintImputation",
            "valid_targets": row["valid_targets"], "singleton_issued": row["issued"], "singleton_correct": row["correct"],
            "singleton_incorrect": row["incorrect"], "issued_action_error": row["accuracy_among_issued"] and (1 - row["accuracy_among_issued"]),
            "correct_action_coverage": row["full_population_correct_fraction"], "set_containment": row["correct"],
            "abstentions": 0, "two_action": 0, "empty": 0, "no_prediction": row["no_development_class"], "unsupported_status": "",
            "guarantee": "point imputation; no fiber or coverage guarantee",
        })
    for row in selective_rows:
        if row["population"] != "confirmatory" or row["alpha"] != 0.05:
            continue
        out.append({
            "consumer": row["consumer"], "schema": row["schema"], "method": "CalibratedSelectiveAction",
            "valid_targets": row["valid_targets"], "singleton_issued": row["issued"], "singleton_correct": row["correct"],
            "singleton_incorrect": row["incorrect"], "issued_action_error": row["selective_error"],
            "correct_action_coverage": row["correct_action_coverage"], "set_containment": row["correct"],
            "abstentions": row["abstained"], "two_action": 0, "empty": 0, "no_prediction": 0, "unsupported_status": "",
            "guarantee": "calibration-selected selective risk; not a finite-sample Geifman-El-Yaniv guarantee",
        })
    for row in conformal_rows:
        if row["population"] != "confirmatory" or row["alpha"] != 0.05:
            continue
        out.append({
            "consumer": row["consumer"], "schema": row["schema"], "method": "UserClusterSplitConformalActionSet",
            "valid_targets": row["valid_targets"], "singleton_issued": row["singleton_events"], "singleton_correct": row["correct_singletons"],
            "singleton_incorrect": row["false_singletons"], "issued_action_error": row["false_singleton_fraction_among_singletons"],
            "correct_action_coverage": row["correct_singleton_coverage"], "set_containment": row["event_covered"],
            "abstentions": 0, "two_action": row["two_action_sets"], "empty": row["empty_sets"], "no_prediction": 0, "unsupported_status": "",
            "guarantee": "learner-trajectory exchangeability claim; not event-IID and not fiber identifiability",
        })
    for row in reference_rows:
        if row["population"] != "confirmatory":
            continue
        out.append({
            "consumer": row["consumer"], "schema": row["schema"], "method": "EmpiricalCertificate",
            "valid_targets": row["valid_targets"], "singleton_issued": row["stable"], "singleton_correct": row["singleton_correct"],
            "singleton_incorrect": row["false_stable"], "issued_action_error": ratio(row["false_stable"], row["stable"]),
            "correct_action_coverage": row["singleton_correct"] / row["valid_targets"], "set_containment": row["set_contains_realized"],
            "abstentions": 0, "two_action": row["set_valued"], "empty": 0, "no_prediction": 0, "unsupported_status": row["unsupported"],
            "guarantee": "finite development-library agreement conditional on support; no statistical coverage guarantee",
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    print("preflight", flush=True)
    pre = chronorep.preflight()
    pre["chronorep_checkpoint_sha256"] = chronorep.sha256_file(CHRONOREP_CKPT)
    pre["chronorep_checkpoint_match"] = pre["chronorep_checkpoint_sha256"] == "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"])
    dump("source_and_checkpoint_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_CALIBRATION_OR_DATA"})
        raise SystemExit(2)
    base = np.array([3.0, 2.0, 1.0, 4.0, 5.0, 1.0, 0.0])
    delta = np.asarray(chronorep.fp_delta((1, 2, 3, ("hint", "answer"))), dtype=np.float64)
    if abs(float(mastery_vec(base, delta.reshape(1, 7))[0]) - chronorep.mastery_of(base + delta)) > 1e-12:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "where": "mastery_vec"})
        raise SystemExit(3)
    dump("baseline_method_fidelity_and_assumptions.json", {
        "frozen_before_confirmatory_scoring": True,
        "policy": "progress if consumer score >= 0.80 else review",
        "modal": "largest development-event fingerprint frequency; ties broken by repr order; no labels used",
        "selective": {
            "lineage": "Geifman and El-Yaniv, NeurIPS 2017, implemented as calibration-selected confidence rejection",
            "exact_original_risk_control_claimed": False,
            "grid": GRID, "primary_alpha": 0.05, "secondary_alpha": [0.01, 0.10],
            "bootstrap_seed": 20315, "replicates": 1000, "upper_percentile": 95,
            "empty_replicate_rule": "excluded from the percentile; never treated as zero error",
            "tie_break": "among risk-qualified thresholds, maximum calibration coverage, then largest threshold",
            "base_tie": "review when p_progress == p_review",
        },
        "conformal": {
            "lineage": "Angelopoulos and Bates, split conformal prediction",
            "name": "UserClusterSplitConformalActionSet",
            "unit": "one maximum nonconformity per calibration learner",
            "event_exchangeability_claimed": False,
            "primary_alpha": 0.05, "secondary_alpha": 0.10,
            "order_statistic": "k = ceil((n+1)*(1-alpha)); q is the kth ascending learner maximum; infinity if k > n",
            "set_rule": "include action a when 1 - p(a) <= q; empty sets are retained",
        },
        "smoothing": "(count + 1) / (n + 2), fixed before outcomes are examined",
        "absent_key": "p=0.5 and NO_DEVELOPMENT_CLASS; modal returns no prediction",
    })
    write_csv("decision_information_access_matrix.csv", [
        {"information": "pre-event history", "available_to_baseline": "yes", "role": "frozen steward-side state at interaction start", "external_release_user": "not established"},
        {"information": "release key S0-S3", "available_to_baseline": "yes", "role": "compatibility-class lookup", "external_release_user": "schema-dependent"},
        {"information": "development compatibility frequencies", "available_to_baseline": "yes", "role": "action votes and modal fingerprint", "external_release_user": "audit library"},
        {"information": "frozen consumer weights", "available_to_baseline": "yes", "role": "score a fingerprint at the fixed state", "external_release_user": "model access assumed for the audit"},
        {"information": "hidden current completion outside the release key", "available_to_baseline": "no", "role": "excluded from prediction", "external_release_user": "absent"},
        {"information": "realized policy action", "available_to_baseline": "evaluation and calibration only", "role": "label, not a predictive feature", "external_release_user": "not an input"},
    ])
    print("development library", flush=True)
    bundles, geometry = build_development()
    if geometry["summary"]["key_counts"] != {"S0": 259838, "S1": 511770, "S2": 523378, "S3": 605097} or geometry["summary"]["duplicate_rich_signatures"] != 0:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "geometry": geometry["summary"]})
        raise SystemExit(4)
    write_csv("development_fingerprint_frequency_geometry.csv", geometry["rows"])
    dump("candidate_action_distribution_qa.json", {"frequency_mass_conserved": True, "rich_signatures_unique": True, "modal_uses_fingerprint_frequency": True, **geometry["summary"]})
    bundle = torch.load(CHRONOREP_CKPT, map_location="cpu", weights_only=False)
    pilot = NeuralPilot(device="cpu")
    pilot.load_bundle({"model": bundle["state_dict"], "skill2i": bundle["skill2i"], "atype2i": bundle["atype2i"]})
    pilot.model.eval()
    print("score calibration", flush=True)
    if (CACHE / "calibration_decisions.npz").exists() and (CACHE / "calibration_meta.json").exists():
        calibration = load_cache("calibration")
        calibration_rows = None
        print("reuse calibration cache", flush=True)
    else:
        calibration_rows = load_split("calibration")
        calibration = score_split("calibration", calibration_rows, bundles, pilot)
    if calibration["events"] != 887997 or calibration["future"] or calibration["schema_label_changes"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "calibration_events": calibration["events"], "future": calibration["future"], "label_changes": calibration["schema_label_changes"]})
        raise SystemExit(5)
    save_cache("calibration", calibration)
    threshold_rows, thresholds = selective_tables(calibration, "calibration")
    write_csv("selective_calibration_thresholds.csv", threshold_rows)
    conformal = conformal_fit(calibration)
    dump("conformal_cluster_calibration.json", conformal)
    dump("calibration_population_integrity.json", {
        "calibration_users": calibration["users"], "calibration_valid_events": calibration["events"],
        "expected_users": 6804, "expected_valid_events": 887997,
        "thresholds_selected_before_confirmatory": True, "confirmatory_events_used_for_selection": 0,
        "users_with_valid_events": conformal["users_with_valid_events"],
        "users_without_valid_events": conformal["users_without_valid_events"],
    })
    del calibration_rows
    print("score confirmatory", flush=True)
    if (CACHE / "confirmatory_decisions.npz").exists() and (CACHE / "confirmatory_meta.json").exists():
        confirmatory = load_cache("confirmatory")
        print("reuse confirmatory cache", flush=True)
    else:
        confirmatory_rows = load_split("confirmatory")
        confirmatory = score_split("confirmatory", confirmatory_rows, bundles, pilot)
    if confirmatory["events"] != 939819 or confirmatory["future"] or confirmatory["schema_label_changes"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "events": confirmatory["events"], "future": confirmatory["future"], "label_changes": confirmatory["schema_label_changes"]})
        raise SystemExit(6)
    del bundles
    found = cert_counts(confirmatory)
    cert_match = found == CHRONOREP_CERT
    dump("empirical_certificate_reference_check.json", {"match": cert_match, "found": {f"{a}|{b}": value for (a, b), value in found.items()}})
    if not cert_match:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "certificate_match": False})
        save_cache("confirmatory", confirmatory)
        raise SystemExit(7)
    save_cache("confirmatory", confirmatory)
    selective = {alpha: apply_selective(confirmatory, thresholds, alpha) for alpha in (0.01, 0.05, 0.10)}
    selective_cal = {alpha: apply_selective(calibration, thresholds, alpha) for alpha in (0.01, 0.05, 0.10)}
    conf_sets = {alpha: apply_conformal(confirmatory, conformal, alpha) for alpha in (0.05, 0.10)}
    cal_sets = {alpha: apply_conformal(calibration, conformal, alpha) for alpha in (0.05, 0.10)}
    m1, s1, c1, r1, t1 = decision_rows("calibration", calibration, selective_cal, cal_sets)
    m2, s2, c2, r2, t2 = decision_rows("confirmatory", confirmatory, selective, conf_sets)
    write_csv("modal_imputation_results.csv", m1 + m2)
    write_csv("selective_confirmatory_results.csv", [row for row in s1 + s2 if row["population"] == "confirmatory"])
    write_csv("selective_risk_coverage_curves.csv", threshold_rows)
    write_csv("conformal_action_set_results.csv", c2)
    write_csv("conformal_user_coverage.csv", learner_coverage(confirmatory, conf_sets[0.05], 0.05) + learner_coverage(confirmatory, conf_sets[0.10], 0.10))
    write_csv("empirical_certificate_reference.csv", r2)
    comparison = matched_rows(m2, s2, c2, r2)
    write_csv("matched_decision_comparison.csv", comparison)
    write_csv("support_and_fingerprint_strata.csv", t2)
    print("bootstrap", flush=True)
    intervals = bootstrap(confirmatory, selective[0.05], conf_sets[0.05])
    write_csv("learner_bootstrap_intervals.csv", intervals)
    dump("independent_scoring_qa.json", {
        "calibration_qa_events": len(calibration["qa"]), "confirmatory_qa_events": len(confirmatory["qa"]),
        "direct_checked_per_split": [calibration["direct_checked"], confirmatory["direct_checked"]],
        "direct_max_abs": max(calibration["direct_gap"], confirmatory["direct_gap"]),
        "direct_action_mismatches": calibration["direct_action_mismatch"] + confirmatory["direct_action_mismatch"],
        "future_violations": calibration["future"] + confirmatory["future"],
        "schema_label_changes": 0, "certificate_reproduction": cert_match,
        "calibration_qa_head": calibration["qa"][:5], "confirmatory_qa_head": confirmatory["qa"][:5],
    })
    dump("statistical_guarantee_scope.json", {
        "modal": "A single development-observed fingerprint. It is not a certificate over the compatibility class.",
        "selective": "A threshold chosen from clustered calibration resampling. It is not the original Geifman-El-Yaniv finite-sample guarantee.",
        "conformal": "A learner-maximum split-conformal set under exchangeable learner trajectories and a calibration-independent score. It is not event-level exchangeability and not conditional support-stratum coverage.",
        "empirical_certificate": "Agreement over represented development completions when the support rule holds. It is not a statistical coverage guarantee.",
        "capacity": "These baselines do not produce guaranteed-in or guaranteed-out capacity intervals.",
    })
    after = chronorep.sha256_file(CHRONOREP_CKPT)
    status = "COMPLETE" if after == pre["chronorep_checkpoint_sha256"] else "BLOCKED_NUMERICAL_INCONSISTENCY"
    dump("numerical_invariants.json", {
        "status": status, "certificate_match": cert_match, "calibration_events": 887997, "confirmatory_events": 939819,
        "future": 0, "schema_label_changes": 0, "chronorep_checkpoint_unchanged": after == pre["chronorep_checkpoint_sha256"],
        "direct_max_abs": max(calibration["direct_gap"], confirmatory["direct_gap"]),
    })
    dump("execution_progress.json", {
        "complete": True, "elapsed_sec": time.time() - started, "rss_mb": chronorep.rss_mb(),
        "calibration_candidate_pfa": calibration["scored_pfa"], "calibration_candidate_gru": calibration["scored_gru"],
        "confirmatory_candidate_pfa": confirmatory["scored_pfa"], "confirmatory_candidate_gru": confirmatory["scored_gru"],
    })
    dump("experiment_manifest.json", {
        "experiment": "pipeline", "status": status, "script_sha256": chronorep.sha256_file(Path(__file__).resolve()),
        "elapsed_sec": time.time() - started, "chronorep_checkpoint_sha256": after, "models_retrained": False,
    })
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/decision_baselines/output_checksums.json"], "n_files": len(files), "files": files})
    print(status, "cert", cert_match, flush=True)


if __name__ == "__main__":
    main()
