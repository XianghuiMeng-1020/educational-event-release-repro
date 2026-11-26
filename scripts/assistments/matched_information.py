#!/usr/bin/env python3
"""Matched HIST / LEN / ACT ActionTokenGRU input contrast.

The frozen core, the primary checkpoint, and frozen results are not modified.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "chronorep", ROOT / "scripts/assistments/chronological_replication.py"
)
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)

from r7_core import MAX_EVAL_PREFIX_EVENTS, MAX_HIST_TOKENS, MAX_SEQ_ACTIONS, NEURAL_SEED, ActionTokenGRU  # noqa: E402

OUT = ROOT / "results/assistments/matched_information"
CHRONOREP_CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
CHRONOREP_CKPT_SHA = "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
HIST_BUDGET = 32
PATTERN_MIN = 20
ARMS = ("HIST", "LEN", "ACT")


def dump(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, default=chronorep._json_default) + "\n", encoding="utf-8")


def make_model(skill2i: dict, atype2i: dict) -> ActionTokenGRU:
    random.seed(NEURAL_SEED)
    np.random.seed(NEURAL_SEED)
    torch.manual_seed(NEURAL_SEED)
    model = ActionTokenGRU(len(skill2i) - 1, len(atype2i) - 1)
    model.to(torch.device("cpu"))
    model.train()
    return model


def history_tokens(row_end: list[dict], start: int, skill2i: dict, atype2i: dict, outcome0: int, outcome1: int):
    ends = np.asarray([item["end"] for item in row_end], dtype=np.int64)
    cut = int(np.searchsorted(ends, start, side="left"))
    window = row_end[max(0, cut - MAX_EVAL_PREFIX_EVENTS) : cut]
    skills: list[int] = []
    atypes: list[int] = []
    for item in window:
        sid = skill2i.get(item["skill_id"], 0)
        for tok in item["action_types"][:MAX_SEQ_ACTIONS]:
            skills.append(sid)
            atypes.append(atype2i.get(tok, 0))
        skills.append(sid)
        atypes.append(outcome1 if item["correct"] == 1 else outcome0)
    return skills, atypes, cut


def arm_sequences(row: dict, by_end: list[dict], skill2i: dict, atype2i: dict, marker: int, outcome0: int, outcome1: int):
    full_s, full_a, cut = history_tokens(by_end, row["start"], skill2i, atype2i, outcome0, outcome1)
    hist_s = full_s[-HIST_BUDGET:]
    hist_a = full_a[-HIST_BUDGET:]
    actions = row["action_types"][:MAX_SEQ_ACTIONS]
    length = len(actions)
    sid = skill2i.get(row["skill_id"], 0)

    def pack(current_ids: list[int]) -> tuple[list[int], list[int]]:
        return hist_s + [sid] * len(current_ids), hist_a + list(current_ids)

    hist = pack([marker])
    leng = pack([marker] * length)
    act = pack([atype2i.get(tok, 0) for tok in actions])
    return {
        "hist_prefix": (hist_s, hist_a),
        "full_history_tokens": len(full_s),
        "cut": cut,
        "length": length,
        "HIST": hist,
        "LEN": leng,
        "ACT": act,
        "chronorep_history_tokens_retained": min(len(full_s), max(0, MAX_HIST_TOKENS - length)),
    }


def load_selected(selected: set[str], skill2i: dict, atype2i: dict) -> tuple[dict, dict, dict]:
    rows = defaultdict(list)
    stats = {0: defaultdict(float), 1: defaultdict(float)}
    patterns = defaultdict(lambda: [0, 0])
    lengths = defaultdict(lambda: [0, 0])
    sample = {schema: {} for schema in ("S0", "S1", "S2", "S3")}
    events = 0
    invalid = 0
    parquet = pq.ParquetFile(chronorep.RECON / "development.parquet")
    for batch in parquet.iter_batches(batch_size=50000):
        raw = batch.to_pylist()
        starts = chronorep.civil_us([item["start_time"] for item in raw])
        ends = chronorep.civil_us([item["end_time"] for item in raw])
        for item, start, end in zip(raw, starts, ends):
            events += 1
            row = chronorep.compact_row(item, int(start), int(end))
            if not row["valid"]:
                invalid += 1
            label = row["correct"]
            actions = row["action_types"]
            retained = actions[:MAX_SEQ_ACTIONS]
            bucket = stats[label]
            bucket["n"] += 1
            bucket["action_count"] += len(actions)
            bucket["hint_count"] += sum(tok == "hint" for tok in actions)
            bucket["answer_count"] += sum(tok == "answer" for tok in actions)
            bucket["has_hint"] += float(any(tok == "hint" for tok in actions))
            bucket["has_answer"] += float(any(tok == "answer" for tok in actions))
            bucket["has_start"] += float(any(tok == "start" for tok in actions))
            bucket["has_end"] += float(any(tok == "end" for tok in actions))
            bucket["distinct"] += len(set(actions))
            bucket["truncated"] += float(len(actions) > MAX_SEQ_ACTIONS)
            key = retained
            patterns[key][0] += 1
            patterns[key][1] += label
            lengths[min(len(retained), MAX_SEQ_ACTIONS)][0] += 1
            lengths[min(len(retained), MAX_SEQ_ACTIONS)][1] += label
            if row["user_id"] in selected:
                rows[row["user_id"]].append(row)
            if row["valid"]:
                ev = chronorep.as_event(row)
                for schema in sample:
                    release = chronorep.release_key(ev, schema)
                    slot = sample[schema].get(release)
                    fingerprint = retained
                    if slot is None and len(sample[schema]) < 40:
                        sample[schema][release] = [fingerprint]
                    elif slot is not None and len(slot) < 2 and fingerprint not in slot:
                        slot.append(fingerprint)
        if events % 1000000 == 0:
            print(f"development_load {events}", flush=True)
    dev_stats = {"events": events, "invalid": invalid, "by_label": stats, "patterns": patterns, "lengths": lengths, "class_sample": sample}
    return rows, dev_stats, {"events": events, "invalid": invalid}


def confirmatory_rows() -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    parquet = pq.ParquetFile(chronorep.RECON / "confirmatory.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=50000):
        raw = batch.to_pylist()
        starts = chronorep.civil_us([item["start_time"] for item in raw])
        ends = chronorep.civil_us([item["end_time"] for item in raw])
        for item, start, end in zip(raw, starts, ends):
            grouped[str(item["user_id"])].append(chronorep.compact_row(item, int(start), int(end)))
            seen += 1
        if seen % 200000 == 0:
            print(f"confirmatory_load {seen}", flush=True)
    return grouped


def epoch_targets(rows: dict, cohorts: list[list[str]]) -> list[list[dict]]:
    epochs = []
    for cohort in cohorts:
        targets = []
        seen = set()
        duplicates = 0
        for uid in cohort:
            ordered = sorted(rows[uid], key=lambda item: item["source_row"])
            window = ordered[-chronorep.MAX_TRAIN_EVENTS_PER_USER :]
            _by_start, by_end = chronorep.order_valid(ordered)
            for row in window:
                if not row["valid"]:
                    continue
                key = row["source_row"]
                if key in seen:
                    duplicates += 1
                seen.add(key)
                targets.append((row, by_end))
        epochs.append({"targets": targets, "duplicates": duplicates})
    return epochs


def audit_targets(targets: list, skill2i, atype2i, marker, outcome0, outcome1) -> dict:
    ranked = sorted(targets, key=lambda item: hashlib.sha256(f"WSDM203-G9-TRAIN|{item[0]['source_row']}".encode()).hexdigest())
    long_actions = [item for item in targets if len(item[0]["action_types"][:MAX_SEQ_ACTIONS]) >= 16]
    short_actions = [item for item in targets if len(item[0]["action_types"][:MAX_SEQ_ACTIONS]) <= 2]
    chosen = ranked[:160]
    chosen.extend(long_actions[:20])
    chosen.extend(short_actions[:20])
    report = {
        "cases": 0,
        "prefix_mismatches": 0,
        "label_mismatches": 0,
        "history_future_members": 0,
        "current_outcome_leaks": 0,
        "hist_marker_count_errors": 0,
        "length_errors": 0,
        "act_len_length_mismatches": 0,
        "long_current_examples": 0,
        "short_history_examples": 0,
        "chronorep_retains_fewer_history_tokens_than_matched_budget": 0,
        "chronorep_history_retained_min": None,
        "chronorep_history_retained_max": None,
        "empty_len_act_inputs": 0,
    }
    retained = []
    for row, by_end in chosen:
        built = arm_sequences(row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
        report["cases"] += 1
        hist_s, hist_a = built["hist_prefix"]
        for name in ARMS:
            skills, atypes = built[name]
            if skills[: len(hist_s)] != hist_s or atypes[: len(hist_a)] != hist_a:
                report["prefix_mismatches"] += 1
            if any(tok in (outcome0, outcome1) for tok in atypes[len(hist_a) :]):
                report["current_outcome_leaks"] += 1
        if built["cut"] and any(item["end"] >= row["start"] for item in by_end[: built["cut"]]):
            report["history_future_members"] += 1
        if len(built["HIST"][0]) != len(hist_s) + 1 or built["HIST"][1][-1] != marker:
            report["hist_marker_count_errors"] += 1
        if len(built["LEN"][0]) != len(hist_s) + built["length"]:
            report["length_errors"] += 1
        if len(built["LEN"][0]) != len(built["ACT"][0]):
            report["act_len_length_mismatches"] += 1
        if built["length"] >= 16:
            report["long_current_examples"] += 1
        if built["full_history_tokens"] <= 8:
            report["short_history_examples"] += 1
        if not built["LEN"][0]:
            report["empty_len_act_inputs"] += 1
        chronorep_kept = built["chronorep_history_tokens_retained"]
        retained.append(chronorep_kept)
        if chronorep_kept < min(built["full_history_tokens"], HIST_BUDGET):
            report["chronorep_retains_fewer_history_tokens_than_matched_budget"] += 1
    report["chronorep_history_retained_min"] = int(min(retained)) if retained else None
    report["chronorep_history_retained_max"] = int(max(retained)) if retained else None
    report["note"] = (
        "With the original 32-action cap, the chronological contract retains min(H, 64-L) history tokens, which is at least "
        "the matched 32-token budget whenever H is long. A long current sequence reduces chronological's surviving "
        "history relative to a short current sequence, but it does not fall below 32."
    )
    return report


def train_arm(name: str, path: Path, epochs: list[dict], skill2i, atype2i, marker, outcome0, outcome1) -> dict:
    model = make_model(skill2i, atype2i)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    used = []
    skipped = []
    for epoch_i, epoch in enumerate(epochs, start=1):
        total = 0.0
        n = 0
        empty = 0
        for row, by_end in epoch["targets"]:
            built = arm_sequences(row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
            skills, atypes = built[name]
            if not skills:
                empty += 1
                continue
            pred, _ = model.forward_from_state(
                torch.tensor([skills], dtype=torch.long),
                torch.tensor([atypes], dtype=torch.long),
                None,
            )
            y = torch.tensor([float(row["correct"])])
            loss = torch.nn.functional.binary_cross_entropy(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
            if n % 20000 == 0:
                print(f"train {name} epoch {epoch_i} examples {n}", flush=True)
        losses.append(total / n if n else None)
        used.append(n)
        skipped.append(empty)
        print(f"train_done {name} epoch {epoch_i} loss {losses[-1]} examples {n} empty {empty}", flush=True)
    model.eval()
    torch.save(
        {
            "label": f"MATCHEDINFO_MATCHED_{name}_SEED17",
            "arm": name,
            "seed": NEURAL_SEED,
            "epochs": 2,
            "warm_started_from_chronorep": False,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "skill2i": skill2i,
            "atype2i": atype2i,
            "marker_action": "start",
            "marker_id": marker,
            "history_token_budget": HIST_BUDGET,
        },
        path,
    )
    return {
        "arm": name,
        "checkpoint": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "checkpoint_sha256": chronorep.sha256_file(path),
        "checkpoint_bytes": path.stat().st_size,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "loss_by_epoch": losses,
        "examples_by_epoch": used,
        "empty_inputs_skipped_by_epoch": skipped,
        "warm_started_from_chronorep": False,
    }


def score_model(model: ActionTokenGRU, pairs: list[tuple[list[int], list[int]]]) -> np.ndarray:
    return chronorep.score_forward(model, pairs, torch.device("cpu"))


def evaluate(models: dict[str, ActionTokenGRU], grouped: dict, skill2i, atype2i, marker, outcome0, outcome1):
    users = sorted(grouped)
    user_index = {uid: i for i, uid in enumerate(users)}
    scores = {name: [] for name in ARMS}
    labels = []
    users_i = []
    keys = []
    pfa_s0 = []
    pfa_y = []
    pfa_user = []
    pfa_hist = []
    lengths = []
    violations = defaultdict(int)
    serial_diff = {name: 0.0 for name in ARMS}
    serial_n = 0
    sample = []
    for uid, rows in grouped.items():
        for row in rows:
            if row["valid"]:
                sample.append((hashlib.sha256(f"WSDM203-G9-SCORE|{row['source_row']}".encode()).hexdigest(), uid, row["source_row"]))
    sample = {(uid, source) for _rank, uid, source in sorted(sample)[:40]}
    for uid in users:
        _by_start, by_end = chronorep.order_valid(grouped[uid])
        pred0 = defaultdict(lambda: np.zeros(2, dtype=np.float64))
        cursor = 0
        built_rows = []
        pairs = {name: [] for name in ARMS}
        for row in _by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                done = by_end[cursor]
                pred0[done["skill_id"]][0] += 1.0
                pred0[done["skill_id"]][1] += float(done["correct"])
                cursor += 1
            built = arm_sequences(row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
            if built["cut"] != cursor:
                violations["history_cut_mismatch"] += 1
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                violations["future_history"] += 1
            hist_s, hist_a = built["hist_prefix"]
            for name in ARMS:
                skills, atypes = built[name]
                if skills[: len(hist_s)] != hist_s or atypes[: len(hist_a)] != hist_a:
                    violations["prefix_mismatch"] += 1
                if any(tok in (outcome0, outcome1) for tok in atypes[len(hist_a) :]):
                    violations["current_outcome"] += 1
                pairs[name].append((skills, atypes))
            if len(built["LEN"][0]) != len(built["ACT"][0]):
                violations["len_act_length"] += 1
            if len(built["HIST"][0]) != len(hist_s) + 1:
                violations["hist_marker"] += 1
            state = pred0[row["skill_id"]]
            pfa_score = None if state[0] <= 0 else chronorep.mastery_of(np.array([state[0], state[1], 0, 0, 0, 0, 0], dtype=np.float64))
            built_rows.append((row, pfa_score, built["length"]))
        pred = {name: score_model(models[name], pairs[name]) for name in ARMS}
        for event_i, (row, pfa_score, length) in enumerate(built_rows):
            labels.append(row["correct"])
            users_i.append(user_index[uid])
            keys.append(hashlib.blake2s(f"WSDM203-G9|{row['source_row']}".encode(), digest_size=16).hexdigest())
            lengths.append(length)
            for name in ARMS:
                scores[name].append(float(pred[name][event_i]))
            if pfa_score is not None:
                pfa_s0.append(pfa_score)
                pfa_y.append(row["correct"])
                pfa_user.append(user_index[uid])
                pfa_hist.append(float(pred["HIST"][event_i]))
            if (uid, row["source_row"]) in sample:
                for name in ARMS:
                    skills, atypes = pairs[name][event_i]
                    if not skills:
                        continue
                    serial = float(models[name].forward_from_state(
                        torch.tensor([skills], dtype=torch.long),
                        torch.tensor([atypes], dtype=torch.long),
                        None,
                    )[0].item())
                    serial_diff[name] = max(serial_diff[name], abs(serial - float(pred[name][event_i])))
                    serial_n += 1
    return {
        "users": users,
        "y": np.asarray(labels, dtype=np.int8),
        "user_index": np.asarray(users_i, dtype=np.int32),
        "scores": {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()},
        "keys": keys,
        "lengths": np.asarray(lengths, dtype=np.int16),
        "pfa_s0": np.asarray(pfa_s0, dtype=np.float64),
        "pfa_y": np.asarray(pfa_y, dtype=np.int8),
        "pfa_user": np.asarray(pfa_user, dtype=np.int32),
        "pfa_hist": np.asarray(pfa_hist, dtype=np.float64),
        "violations": dict(violations),
        "serial_max_abs": serial_diff,
        "serial_comparisons": serial_n,
    }


def metrics_row(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    base = chronorep.binary_metrics(y, p)
    finite = np.isfinite(p)
    pred = p[finite]
    accuracy = float(np.mean((pred >= 0.5) == y[finite])) if len(pred) else None
    quantiles = np.quantile(pred, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(pred) else [None] * 5
    return {
        "model": name,
        "n": int(len(y)),
        "n_learners": None,
        "base_rate": base["positive_rate"],
        "auroc": base["roc_auc"],
        "log_loss": base["log_loss"],
        "brier": base["brier"],
        "accuracy_at_0.5": accuracy,
        "mean_prediction": float(pred.mean()) if len(pred) else None,
        "q10": None if quantiles[0] is None else float(quantiles[0]),
        "q25": None if quantiles[1] is None else float(quantiles[1]),
        "q50": None if quantiles[2] is None else float(quantiles[2]),
        "q75": None if quantiles[3] is None else float(quantiles[3]),
        "q90": None if quantiles[4] is None else float(quantiles[4]),
        "nonfinite": int((~finite).sum()),
        "outside_unit_interval": int(np.sum((p < 0) | (p > 1))),
    }


def calibration_rows(name: str, y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    for bin_id in range(10):
        mask = bins == bin_id
        rows.append({
            "model": name,
            "bin": bin_id,
            "lo": bin_id / 10,
            "hi": (bin_id + 1) / 10,
            "n": int(mask.sum()),
            "mean_prediction": float(p[mask].mean()) if mask.any() else None,
            "observed_frequency": float(y[mask].mean()) if mask.any() else None,
        })
    return rows


def clustered_bootstrap(y, user_index, score_map, contrasts, seed: int = 20309, n_users: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_users = int(user_index.max()) + 1 if n_users is None else int(n_users)
    draws = rng.integers(0, n_users, size=(2000, n_users))
    buckets = [[] for _ in range(n_users)]
    for index, user in enumerate(user_index.tolist()):
        buckets[user].append(index)
    indexes = [np.asarray(bucket, dtype=np.int32) for bucket in buckets]
    per_n = np.asarray([len(bucket) for bucket in indexes], dtype=np.int32)
    per_log = {name: np.zeros(n_users) for name in score_map}
    per_brier = {name: np.zeros(n_users) for name in score_map}
    pieces = {name: [] for name in score_map}
    label_pieces = []
    for user_i, idx in enumerate(indexes):
        yy = y[idx].astype(float) if len(idx) else np.zeros(0)
        label_pieces.append(yy)
        for name, values in score_map.items():
            pp = np.clip(values[idx], 1e-12, 1 - 1e-12) if len(idx) else np.zeros(0)
            pieces[name].append(pp)
            if len(idx):
                per_log[name][user_i] = -np.sum(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
                per_brier[name][user_i] = np.sum((pp - yy) ** 2)
    point = {name: chronorep.binary_metrics(y, score_map[name]) for name in score_map}
    aucs = {name: np.empty(2000) for name in score_map}
    failures = 0
    y_buffer = np.empty(int(per_n.sum()), dtype=np.float64)
    p_buffer = np.empty(int(per_n.sum()), dtype=np.float64)
    for rep, draw in enumerate(draws):
        needed = int(per_n[draw].sum())
        if needed > len(y_buffer):
            y_buffer = np.empty(needed, dtype=np.float64)
            p_buffer = np.empty(needed, dtype=np.float64)
        pos = 0
        for user_i in draw:
            block = label_pieces[user_i]
            n = len(block)
            if n:
                y_buffer[pos : pos + n] = block
                pos += n
        if pos == 0:
            failures += 1
            for name in score_map:
                aucs[name][rep] = np.nan
            continue
        yy = y_buffer[:pos]
        for name in score_map:
            cursor = 0
            for user_i in draw:
                block = pieces[name][user_i]
                n = len(block)
                if n:
                    p_buffer[cursor : cursor + n] = block
                    cursor += n
            aucs[name][rep] = chronorep.binary_metrics(yy, p_buffer[:cursor])["roc_auc"]
        if rep and rep % 100 == 0:
            print(f"bootstrap {rep}", flush=True)
    rows = []
    n_draw = per_n[draws].sum(axis=1)
    for name in score_map:
        finite = aucs[name][np.isfinite(aucs[name])]
        rows.append({"analysis": "auroc", "contrast": name, "point": point[name]["roc_auc"], "bootstrap_mean": float(np.mean(finite)), "ci95_low": float(np.percentile(finite, 2.5)), "ci95_high": float(np.percentile(finite, 97.5)), "replicates": 2000, "replicate_failures": int(np.sum(~np.isfinite(aucs[name]))), "seed": seed})
        for analysis, store, observed in (
            ("log_loss", per_log, point[name]["log_loss"]),
            ("brier", per_brier, point[name]["brier"]),
        ):
            values = store[name][draws].sum(axis=1) / np.maximum(n_draw, 1)
            rows.append({"analysis": analysis, "contrast": name, "point": observed, "bootstrap_mean": float(np.mean(values)), "ci95_low": float(np.percentile(values, 2.5)), "ci95_high": float(np.percentile(values, 97.5)), "replicates": 2000, "replicate_failures": failures, "seed": seed})
    for left, right, label in contrasts:
        delta = aucs[left] - aucs[right]
        finite = delta[np.isfinite(delta)]
        rows.append({"analysis": "auroc_difference", "contrast": label, "point": point[left]["roc_auc"] - point[right]["roc_auc"], "bootstrap_mean": float(np.mean(finite)), "ci95_low": float(np.percentile(finite, 2.5)), "ci95_high": float(np.percentile(finite, 97.5)), "replicates": 2000, "replicate_failures": int(np.sum(~np.isfinite(delta))), "seed": seed})
        for analysis, store, key in (("log_loss_difference", per_log, "log_loss"), ("brier_difference", per_brier, "brier")):
            values = (store[left][draws].sum(axis=1) - store[right][draws].sum(axis=1)) / np.maximum(n_draw, 1)
            rows.append({"analysis": analysis, "contrast": label, "point": point[left][key] - point[right][key], "bootstrap_mean": float(np.mean(values)), "ci95_low": float(np.percentile(values, 2.5)), "ci95_high": float(np.percentile(values, 97.5)), "replicates": 2000, "replicate_failures": failures, "seed": seed})
    return pd.DataFrame(rows)


def pattern_probability(pattern: tuple, length: int, patterns, lengths, global_p: float) -> float:
    count, correct = patterns.get(pattern, (0, 0))
    if count >= PATTERN_MIN:
        return (correct + 1) / (count + 2)
    count, correct = lengths.get(length, (0, 0))
    if count >= PATTERN_MIN:
        return (correct + 1) / (count + 2)
    return global_p


def action_diagnostics(dev_stats: dict, grouped: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    for label, bucket in dev_stats["by_label"].items():
        n = bucket["n"]
        rows.append({
            "split": "development", "correct": int(label), "n": int(n),
            "mean_action_count": bucket["action_count"] / n,
            "mean_hint_count": bucket["hint_count"] / n,
            "mean_answer_count": bucket["answer_count"] / n,
            "fraction_with_hint": bucket["has_hint"] / n,
            "fraction_with_answer": bucket["has_answer"] / n,
            "fraction_with_start": bucket["has_start"] / n,
            "fraction_with_end": bucket["has_end"] / n,
            "mean_distinct_types": bucket["distinct"] / n,
            "fraction_truncated_at_32": bucket["truncated"] / n,
        })
    conf = {0: defaultdict(float), 1: defaultdict(float)}
    total_dev = sum(bucket["n"] for bucket in dev_stats["by_label"].values())
    correct_dev = dev_stats["by_label"][1]["n"]
    global_p = (correct_dev + 1) / (total_dev + 2)
    patterns = {key: (value[0], value[1]) for key, value in dev_stats["patterns"].items()}
    lengths = {key: (value[0], value[1]) for key, value in dev_stats["lengths"].items()}
    near = 0
    covered = 0
    valid = 0
    preds = []
    ys = []
    deterministic_patterns = 0
    deterministic_events = 0
    for _count, correct in patterns.values():
        if _count >= PATTERN_MIN and ((correct + 1) / (_count + 2) >= 0.95 or (correct + 1) / (_count + 2) <= 0.05):
            deterministic_patterns += 1
    for rows_user in grouped.values():
        for row in rows_user:
            if not row["valid"]:
                continue
            valid += 1
            actions = row["action_types"]
            label = row["correct"]
            bucket = conf[label]
            bucket["n"] += 1
            bucket["action_count"] += len(actions)
            bucket["hint_count"] += sum(tok == "hint" for tok in actions)
            bucket["answer_count"] += sum(tok == "answer" for tok in actions)
            bucket["has_hint"] += float(any(tok == "hint" for tok in actions))
            bucket["has_answer"] += float(any(tok == "answer" for tok in actions))
            bucket["has_start"] += float(any(tok == "start" for tok in actions))
            bucket["has_end"] += float(any(tok == "end" for tok in actions))
            bucket["distinct"] += len(set(actions))
            bucket["truncated"] += float(len(actions) > MAX_SEQ_ACTIONS)
            retained = actions[:MAX_SEQ_ACTIONS]
            count = patterns.get(retained, (0, 0))[0]
            probability = pattern_probability(retained, len(retained), patterns, lengths, global_p)
            preds.append(probability)
            ys.append(label)
            if count >= PATTERN_MIN:
                covered += 1
                if probability >= 0.95 or probability <= 0.05:
                    near += 1
                    deterministic_events += 1
    for label, bucket in conf.items():
        n = bucket["n"]
        rows.append({
            "split": "confirmatory_valid", "correct": int(label), "n": int(n),
            "mean_action_count": bucket["action_count"] / n,
            "mean_hint_count": bucket["hint_count"] / n,
            "mean_answer_count": bucket["answer_count"] / n,
            "fraction_with_hint": bucket["has_hint"] / n,
            "fraction_with_answer": bucket["has_answer"] / n,
            "fraction_with_start": bucket["has_start"] / n,
            "fraction_with_end": bucket["has_end"] / n,
            "mean_distinct_types": bucket["distinct"] / n,
            "fraction_truncated_at_32": bucket["truncated"] / n,
        })
    diagnostic = chronorep.binary_metrics(np.asarray(ys, dtype=float), np.asarray(preds, dtype=float))
    summary = {
        "smoothing": "(count+1)/(n+2); pattern if development n>=20, else retained-length bucket if n>=20, else global development rate",
        "pattern_min_count": PATTERN_MIN,
        "frozen_before_confirmatory_scoring": True,
        "development_patterns": len(patterns),
        "development_patterns_n_ge_20_and_smoothed_p_outside_0.05_0.95": deterministic_patterns,
        "confirmatory_valid_events": valid,
        "confirmatory_events_with_development_pattern_n_ge_20": covered,
        "confirmatory_events_with_nearly_deterministic_development_pattern": deterministic_events,
        "action_pattern_diagnostic_auroc": diagnostic["roc_auc"],
        "action_pattern_diagnostic_log_loss": diagnostic["log_loss"],
        "action_pattern_diagnostic_brier": diagnostic["brier"],
        "label": "development-estimated action-pattern probability; not a matched neural model",
    }
    return pd.DataFrame(rows), summary


def degeneracy(sample: dict, skill2i, atype2i, marker) -> dict:
    checked = {schema: 0 for schema in sample}
    identical = {schema: 0 for schema in sample}
    action_differences = {schema: 0 for schema in sample}
    for schema, keys in sample.items():
        for fingerprints in keys.values():
            if len(fingerprints) < 2:
                continue
            checked[schema] += 1
            left, right = fingerprints[0], fingerprints[1]
            sid = 3
            hist = ([sid, sid], [1, 9])
            def encode(actions):
                ids = [atype2i.get(tok, 0) for tok in actions]
                return hist[0] + [sid] * 1, hist[1] + [marker]
            a = encode(left)
            b = encode(right)
            act_a = (hist[0] + [sid] * len(left), hist[1] + [atype2i.get(tok, 0) for tok in left])
            act_b = (hist[0] + [sid] * len(right), hist[1] + [atype2i.get(tok, 0) for tok in right])
            if a == b:
                identical[schema] += 1
            if act_a != act_b:
                action_differences[schema] += 1
    return {
        "claim": "HIST current input is one constant start marker on the target skill. It does not depend on candidate action identities or length.",
        "structural_implication": "For a fixed history and skill, every candidate completion receives the same HIST score. The score interval is a singleton, so a supported HIST certificate cannot be Set-valued. That stability is mechanical and is not evidence of stronger decision identifiability.",
        "classes_with_two_distinct_action_fingerprints": checked,
        "hist_token_sequences_identical": identical,
        "act_token_sequences_differ": action_differences,
        "marker": "start",
        "marker_id": marker,
    }


def write_checksums() -> None:
    files = []
    for path in sorted(item for item in OUT.iterdir() if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/matched_information/output_checksums.json"], "n_files": len(files), "files": files})


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    pre = chronorep.preflight()
    chronorep_hash = chronorep.sha256_file(CHRONOREP_CKPT)
    pre["chronorep_checkpoint_sha256"] = chronorep_hash
    pre["chronorep_checkpoint_match"] = chronorep_hash == CHRONOREP_CKPT_SHA
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"])
    dump("source_and_checkpoint_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_CONTRACT", "reason": "source or chronological checkpoint mismatch"})
        write_checksums()
        raise SystemExit(2)
    checkpoint = torch.load(CHRONOREP_CKPT, map_location="cpu", weights_only=False)
    skill2i = checkpoint["skill2i"]
    atype2i = checkpoint["atype2i"]
    probe = {"source_row": 1, "skill_id": next(iter(skill2i)), "action_types": ("hint", "answer"), "start": 10, "end": 5, "correct": 1, "valid": True}
    prior = {"source_row": 0, "skill_id": probe["skill_id"], "action_types": ("start", "answer", "end"), "start": 0, "end": 4, "correct": 0, "valid": True}
    built = arm_sequences(probe, [prior], skill2i, atype2i, atype2i["start"], len(atype2i), len(atype2i) + 1)
    other = dict(probe)
    other["action_types"] = ("start", "hint", "end")
    built_other = arm_sequences(other, [prior], skill2i, atype2i, atype2i["start"], len(atype2i), len(atype2i) + 1)
    if built["HIST"] != built_other["HIST"] or len(built["LEN"][0]) != len(built["ACT"][0]) or built["LEN"][1][-2:] != [atype2i["start"], atype2i["start"]] or built["ACT"] == built["LEN"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_CONTRACT", "reason": "arm self-check failed"})
        write_checksums()
        raise SystemExit(3)
    if "start" not in atype2i or len(skill2i) - 1 != 264 or len(atype2i) - 1 != 8:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_CONTRACT", "reason": "start marker or vocabulary mismatch"})
        write_checksums()
        raise SystemExit(3)
    marker = atype2i["start"]
    outcome0 = len(atype2i)
    outcome1 = len(atype2i) + 1
    print("preflight_pass", flush=True)
    _frame, first_seen = chronorep.load_order("development")
    if _frame["user_id"].nunique() != 31753 or len(_frame) != 4286981 or int((~_frame["valid"]).sum()) != 475:
        raise SystemExit("development population mismatch")
    del _frame
    selection = chronorep.replay_selection(first_seen)
    if not selection["matches_tempfor"] or selection["selection_sha256"] != "cbe4fd9e0075c78ae5997892153d37920497d14c64a21af3ffbf3594e5c907e2":
        dump("training_user_selection.json", {"matches_tempfor": selection["matches_tempfor"], "union": selection["union"], "intersection": selection["intersection"]})
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_CONTRACT", "reason": "user selection mismatch"})
        write_checksums()
        raise SystemExit(4)
    print("selection_pass", selection["union"], selection["intersection"], flush=True)
    training_rows, dev_stats, population = load_selected(selection["selected_users"], skill2i, atype2i)
    epochs = epoch_targets(training_rows, selection["cohorts"])
    counts = [len(item["targets"]) for item in epochs]
    duplicates = [item["duplicates"] for item in epochs]
    if counts != [90950, 90942] or any(duplicates):
        dump("training_user_selection.json", {"counts": counts, "duplicates": duplicates, "expected": [90950, 90942]})
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_CONTRACT", "reason": "training target count or duplicate mismatch"})
        write_checksums()
        raise SystemExit(5)
    audit = audit_targets(epochs[0]["targets"], skill2i, atype2i, marker, outcome0, outcome1)
    dump("training_input_equivalence.json", audit)
    if audit["prefix_mismatches"] or audit["current_outcome_leaks"] or audit["hist_marker_count_errors"] or audit["length_errors"] or audit["act_len_length_mismatches"] or audit["history_future_members"] or audit["empty_len_act_inputs"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_CONTRACT", "audit": audit})
        write_checksums()
        raise SystemExit(6)
    contract = {
        "history": "H(e) = {same student, valid interval, end(p) < start(e)}",
        "history_event_cap": MAX_EVAL_PREFIX_EVENTS,
        "shared_history_tokens": HIST_BUDGET,
        "current_action_cap": MAX_SEQ_ACTIONS,
        "total_capacity": MAX_HIST_TOKENS,
        "marker": "start",
        "marker_id": marker,
        "outcome_token_ids_excluded_from_current_input": [outcome0, outcome1],
        "arms": {
            "HIST": "shared history, current skill, one constant start marker",
            "LEN": "shared history, current skill, L copies of the start marker",
            "ACT": "shared history, current skill, actual current action ids",
        },
        "chronorep_checkpoint_is_not_an_arm": True,
        "pattern_rule_frozen_before_confirmatory_evaluation": {
            "min_count": PATTERN_MIN,
            "smoothing": "(c+1)/(n+2)",
            "backoff": ["action tuple capped at 32", "retained length", "global development rate"],
        },
    }
    dump("matched_input_contract.json", contract)
    dump("training_user_selection.json", {
        "epoch_sizes": selection["epoch_sizes"], "union": selection["union"], "intersection": selection["intersection"],
        "selection_sha256": selection["selection_sha256"], "matches_chronorep": True,
        "valid_targets_by_epoch": counts, "duplicate_targets_by_epoch": duplicates,
    })
    print("contract_frozen", flush=True)
    models = {}
    arm_files = (("HIST", "gru_hist_seed17.pt"), ("LEN", "gru_length_seed17.pt"), ("ACT", "gru_action_seed17_matched.pt"))
    existing = OUT / "training_logs.json"
    if existing.exists() and all((OUT / filename).exists() for _name, filename in arm_files):
        logs = json.loads(existing.read_text())["models"]
        print("training_resumed_from_checkpoints", flush=True)
    else:
        logs = []
        for name, filename in arm_files:
            info = train_arm(name, OUT / filename, epochs, skill2i, atype2i, marker, outcome0, outcome1)
            logs.append(info)
            dump("training_logs.json", {"models": logs})
    for name, filename in arm_files:
        model = ActionTokenGRU(len(skill2i) - 1, len(atype2i) - 1)
        state = torch.load(OUT / filename, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        model.eval()
        models[name] = model
    if any(item["parameter_count"] != 33857 or item["examples_by_epoch"] != [90950, 90942] for item in logs):
        dump("numerical_invariants.json", {"status": "BLOCKED_TRAINING_OR_EXECUTION", "logs": logs})
        write_checksums()
        raise SystemExit(7)
    print("training_complete", flush=True)
    del training_rows, epochs
    gc.collect()
    grouped = confirmatory_rows()
    valid = sum(row["valid"] for rows in grouped.values() for row in rows)
    invalid = sum((not row["valid"]) for rows in grouped.values() for row in rows)
    if valid != 939819 or invalid != 91 or len(grouped) != 6766:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_CONTRACT", "valid": valid, "invalid": invalid, "users": len(grouped)})
        write_checksums()
        raise SystemExit(8)
    cal_users = set(pd.read_parquet(chronorep.RECON / "calibration.parquet", columns=["user_id"])["user_id"].astype(str))
    conf_users = set(grouped)
    overlap_train = conf_users & selection["selected_users"]
    overlap_cal = cal_users & selection["selected_users"]
    evaluated = evaluate(models, grouped, skill2i, atype2i, marker, outcome0, outcome1)
    n_learners = int(len(np.unique(evaluated["user_index"])))
    metric_rows = []
    cal_rows = []
    for name in ARMS:
        row = metrics_row(name, evaluated["y"], evaluated["scores"][name])
        row["n_learners"] = n_learners
        metric_rows.append(row)
        cal_rows.extend(calibration_rows(name, evaluated["y"], evaluated["scores"][name]))
    contrasts = []
    by_name = {row["model"]: row for row in metric_rows}
    for left, right, label in (("LEN", "HIST", "LEN_minus_HIST"), ("ACT", "LEN", "ACT_minus_LEN"), ("ACT", "HIST", "ACT_minus_HIST")):
        contrasts.append({
            "contrast": label, "n": by_name[left]["n"],
            "auroc_difference": by_name[left]["auroc"] - by_name[right]["auroc"],
            "log_loss_difference": by_name[left]["log_loss"] - by_name[right]["log_loss"],
            "brier_difference": by_name[left]["brier"] - by_name[right]["brier"],
            "auroc_sign_convention": "positive favors the first named condition",
            "loss_sign_convention": "negative favors the first named condition",
        })
    pd.DataFrame(metric_rows).to_csv(OUT / "confirmatory_predictive_metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "paired_predictive_contrasts.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(OUT / "predictive_calibration_bins.csv", index=False)
    print("metrics_written", flush=True)
    boot = clustered_bootstrap(
        evaluated["y"], evaluated["user_index"], evaluated["scores"],
        [("LEN", "HIST", "LEN_minus_HIST"), ("ACT", "LEN", "ACT_minus_LEN"), ("ACT", "HIST", "ACT_minus_HIST")],
        n_users=len(evaluated["users"]),
    )
    boot.to_csv(OUT / "paired_bootstrap_intervals.csv", index=False)
    print("bootstrap_written", flush=True)
    feature_table, pattern_summary = action_diagnostics(dev_stats, grouped)
    feature_table.to_csv(OUT / "action_pattern_label_diagnostics.csv", index=False)
    dump("history_only_certificate_degeneracy.json", degeneracy(dev_stats["class_sample"], skill2i, atype2i, marker))
    pfa_metrics = metrics_row("PFA_preupdate_S0", evaluated["pfa_y"], evaluated["pfa_s0"])
    hist_metrics = metrics_row("HIST_on_nonempty_history", evaluated["pfa_y"], evaluated["pfa_hist"])
    pfa_boot = clustered_bootstrap(
        evaluated["pfa_y"], evaluated["pfa_user"],
        {"HIST": evaluated["pfa_hist"], "PFA": evaluated["pfa_s0"]},
        [("HIST", "PFA", "HIST_minus_PFA")],
        n_users=len(evaluated["users"]),
    )
    optional_rows = [
        {**pfa_metrics, "population": "valid targets with a nonempty completed history", "role": "optional_comparator"},
        {**hist_metrics, "population": "valid targets with a nonempty completed history", "role": "optional_comparator"},
        {
            "model": "HIST_minus_PFA",
            "n": int(len(evaluated["pfa_y"])),
            "auroc_difference": hist_metrics["auroc"] - pfa_metrics["auroc"],
            "log_loss_difference": hist_metrics["log_loss"] - pfa_metrics["log_loss"],
            "brier_difference": hist_metrics["brier"] - pfa_metrics["brier"],
            "population": "valid targets with a nonempty completed history",
            "role": "optional_comparator",
            "note": "Different representations and inductive biases. Not an architecture-only contrast.",
        },
    ]
    pd.DataFrame(optional_rows).to_csv(OUT / "optional_pfa_hist_comparator.csv", index=False)
    pfa_boot.to_csv(OUT / "optional_pfa_hist_bootstrap.csv", index=False)
    table = pa.table({
        "event_key": evaluated["keys"],
        "user_index": evaluated["user_index"],
        "correct": evaluated["y"],
        "p_hist": evaluated["scores"]["HIST"],
        "p_len": evaluated["scores"]["LEN"],
        "p_act": evaluated["scores"]["ACT"],
        "current_action_length": evaluated["lengths"],
    })
    pq.write_table(table, OUT / "restricted_event_score_cache.parquet")
    cache = OUT / "restricted_event_score_cache.parquet"
    provenance = {
        "training_audit": audit,
        "evaluation_violations": evaluated["violations"],
        "serial_versus_batch_max_abs": evaluated["serial_max_abs"],
        "serial_comparisons": evaluated["serial_comparisons"],
        "training_confirmatory_user_overlap": len(overlap_train),
        "training_calibration_user_overlap": len(overlap_cal),
        "development_events": population["events"],
        "development_invalid": population["invalid"],
        "action_pattern_summary": pattern_summary,
    }
    dump("input_provenance_checks.json", provenance)
    status = "COMPLETE"
    if evaluated["violations"] or any(row["nonfinite"] or row["outside_unit_interval"] or row["n"] != 939819 for row in metric_rows) or overlap_train or overlap_cal:
        status = "BLOCKED_NUMERICAL_INCONSISTENCY"
    invariants = {
        "status": status,
        "valid_events_each_arm": [row["n"] for row in metric_rows],
        "parameter_counts": [item["parameter_count"] for item in logs],
        "examples_by_epoch": [item["examples_by_epoch"] for item in logs],
        "violations": evaluated["violations"],
        "duration_seconds": time.time() - t0,
        "max_rss_mb": chronorep.rss_mb(),
        "chronorep_checkpoint_unchanged": chronorep.sha256_file(CHRONOREP_CKPT) == CHRONOREP_CKPT_SHA,
    }
    dump("numerical_invariants.json", invariants)
    dump("execution_progress.json", {"status": status, "duration_seconds": invariants["duration_seconds"], "max_rss_mb": invariants["max_rss_mb"]})
    dump("experiment_manifest.json", {
        "experiment": "pipeline",
        "script": "scripts/assistments/matched_information.py",
        "script_sha256": chronorep.sha256_file(Path(__file__).resolve()),
        "status": status,
        "chronorep_checkpoint_sha256": chronorep_hash,
        "command": "python3 -u scripts/assistments/matched_information.py",
    })
    write_checksums()
    print(status, flush=True)


if __name__ == "__main__":
    main()
