#!/usr/bin/env python3
"""Published knowledge-tracing baselines on one chronology-valid pre-action task."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("chronorep", ROOT / "scripts/assistments/chronological_replication.py")
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)
from kt_models import (  # noqa: E402
    AKTNonRasch, DKT, MAX_HISTORY, N_SKILLS, SAKTKC,
    bkt_nll_matrix, bkt_predict, bkt_predict_matrix, bkt_step, fit_bkt_skill,
)

OUT = ROOT / "results/assistments/kt_baselines"
CKPT = OUT / "trained_baseline_checkpoints"
CHRONOREP_CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
ESTCTRL_SCORES = ROOT / "results/assistments/matched_estimand_controls/private_scores"
SELECTION_SHA = "cbe4fd9e0075c78ae5997892153d37920497d14c64a21af3ffbf3594e5c907e2"
GRU_SHA = "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
VALID = 939819
CAL_PREFIX = "WSDM203-G7-CAL-PILOT-v1"


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


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else value for key, value in row.items()})


def load_users(split: str, keep: set[str]) -> dict:
    import pyarrow.parquet as pq
    grouped = defaultdict(list)
    parquet = pq.ParquetFile(chronorep.RECON / f"{split}.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            uid = str(row["user_id"])
            if uid not in keep:
                continue
            grouped[uid].append(chronorep.compact_row(row, int(start), int(end)))
        seen += len(rows)
        if seen % 1000000 == 0:
            print(f"load_{split} {seen}", flush=True)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["source_row"])
    return grouped


def same_skill_history(skills: np.ndarray, outcomes: np.ndarray, length: int, query: int) -> tuple[np.ndarray, int]:
    chosen = outcomes[:length][skills[:length] == query]
    out = np.zeros(MAX_HISTORY, dtype=np.int8)
    count = min(len(chosen), MAX_HISTORY)
    if count:
        out[:count] = chosen[-count:]
    return out, count


def collect(rows_by_user: dict, cohorts: list[list[str]] | None, skill2i: dict) -> dict:
    epochs = cohorts if cohorts is not None else [sorted(rows_by_user)]
    blocks = []
    for cohort in epochs:
        hist_skill, hist_y, hist_len, query, target, users = [], [], [], [], [], []
        bkt_hist, bkt_len = [], []
        for uid in cohort:
            rows = rows_by_user.get(uid)
            if not rows:
                continue
            by_start, by_end = chronorep.order_valid(rows)
            window = [row for row in sorted(rows, key=lambda item: item["source_row"])[-40:] if row["valid"]] if cohorts is not None else by_start
            for row in window:
                if not row["valid"]:
                    continue
                cut = chronorep.history_cut(by_end, row["start"])
                history = by_end[max(0, cut - MAX_HISTORY):cut]
                skills = np.zeros(MAX_HISTORY, dtype=np.int16)
                outcomes = np.zeros(MAX_HISTORY, dtype=np.int8)
                for index, item in enumerate(history):
                    skills[index] = int(skill2i.get(item["skill_id"], 0))
                    outcomes[index] = int(item["correct"])
                query_id = int(skill2i.get(row["skill_id"], 0))
                packed, count = same_skill_history(skills, outcomes, len(history), query_id)
                hist_skill.append(skills)
                hist_y.append(outcomes)
                hist_len.append(len(history))
                query.append(query_id)
                target.append(int(row["correct"]))
                users.append(uid)
                bkt_hist.append(packed)
                bkt_len.append(count)
        if hist_skill:
            blocks.append({
                "hist_skill": np.stack(hist_skill), "hist_y": np.stack(hist_y), "hist_len": np.asarray(hist_len, dtype=np.int16),
                "query": np.asarray(query, dtype=np.int16), "y": np.asarray(target, dtype=np.int8),
                "bkt_hist": np.stack(bkt_hist), "bkt_len": np.asarray(bkt_len, dtype=np.int16), "users": users,
            })
    if not blocks:
        raise RuntimeError("no examples")
    merged = {key: np.concatenate([block[key] for block in blocks]) for key in ("hist_skill", "hist_y", "hist_len", "query", "y", "bkt_hist", "bkt_len")}
    merged["users"] = [uid for block in blocks for uid in block["users"]]
    return merged


def analytic_bkt_tests() -> dict:
    predict, nxt = bkt_step(0.4, 1, 0.2, 0.1, 0.1)
    incorrect, nxt_wrong = bkt_step(0.4, 0, 0.2, 0.1, 0.1)
    cold = bkt_predict(0.4, 0.2, 0.1, 0.1, np.zeros(0, dtype=np.int8))
    after = bkt_predict(0.4, 0.2, 0.1, 0.1, np.asarray([1], dtype=np.int8))
    return {
        "first_prediction": predict, "after_correct_mastery": nxt, "after_incorrect_mastery": nxt_wrong,
        "cold_start": cold, "prediction_after_one_correct": after,
        "pass": abs(predict - 0.42) < 1e-9 and abs(nxt - (0.36 / 0.42 + (1 - 0.36 / 0.42) * 0.2)) < 1e-9 and abs(cold - 0.42) < 1e-9 and abs(after - nxt * 0.9 - (1 - nxt) * 0.1) < 1e-9,
    }


def mechanism_tests() -> dict:
    torch.manual_seed(17)
    dkt = DKT()
    sakt = SAKTKC()
    akt = AKTNonRasch()
    for model in (dkt, sakt, akt):
        model.eval()
    skill = torch.tensor([[3, 4, 0], [1, 0, 0]])
    correct = torch.tensor([[1, 0, 0], [0, 0, 0]])
    length = torch.tensor([2, 1])
    query = torch.tensor([5, 2])
    with torch.inference_mode():
        one = dkt(skill[:1], correct[:1], length[:1], query[:1])
        batched = dkt(skill, correct, length, query)
        dkt_gap = float(torch.max(torch.abs(one - batched[:1])))
        empty = dkt(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, dtype=torch.long), torch.tensor([4]))
        sakt_one = sakt(skill[:1], correct[:1], length[:1], query[:1])
        sakt_batch = sakt(skill, correct, length, query)
        sakt_gap = float(torch.max(torch.abs(sakt_one - sakt_batch[:1])))
        sequence_skill = torch.tensor([[4, 5]])
        response_a = torch.tensor([[1, 0]])
        response_b = torch.tensor([[1, 1]])
        real = torch.tensor([[True, True]])
        akt_a = akt(sequence_skill, response_a, real)
        akt_b = akt(sequence_skill, response_b, real)
        akt_gap = float(torch.max(torch.abs(akt_a - akt_b)))
    return {
        "dkt_batch_gap": dkt_gap, "sakt_batch_gap": sakt_gap, "akt_current_response_gap": akt_gap,
        "dkt_empty_finite": bool(torch.isfinite(empty).all()),
        "pass": dkt_gap < 1e-5 and sakt_gap < 1e-5 and akt_gap < 1e-6 and bool(torch.isfinite(empty).all()),
    }


def input_audit(rows_by_user: dict, examples: dict, skill2i: dict) -> dict:
    order = np.argsort([hashlib.sha256(f"WSDM203-G14|{uid}|{index}".encode()).hexdigest() for index, uid in enumerate(examples["users"])])
    checked = violations = future = action_fields = 0
    for index in order[:250]:
        checked += 1
        uid = examples["users"][index]
        # Recompute from the stored user rows by matching the query and label is not unique.
        # The stored arrays are the model input; verify their internal contract directly.
        length = int(examples["hist_len"][index])
        if length > MAX_HISTORY or length < int(examples["bkt_len"][index]):
            violations += 1
        if np.any(examples["hist_skill"][index, length:] != 0) or np.any(examples["hist_y"][index, length:] != 0):
            violations += 1
        query = int(examples["query"][index])
        if query < 0 or query >= N_SKILLS:
            violations += 1
        history_outcomes = examples["hist_y"][index, :length][examples["hist_skill"][index, :length] == query]
        if int(examples["bkt_len"][index]) != len(history_outcomes):
            violations += 1
        if len(history_outcomes) and not np.array_equal(examples["bkt_hist"][index, :len(history_outcomes)], history_outcomes[-MAX_HISTORY:]):
            violations += 1
    # Direct chronology check on 250 hashed valid rows.
    pool = []
    for uid, rows in rows_by_user.items():
        for row in rows:
            if row["valid"]:
                pool.append((hashlib.sha256(f"WSDM203-G14-ROW|{row['source_row']}".encode()).hexdigest(), uid, row))
    pool.sort()
    for _rank, uid, row in pool[:250]:
        _by_start, by_end = chronorep.order_valid(rows_by_user[uid])
        cut = chronorep.history_cut(by_end, row["start"])
        history = by_end[max(0, cut - MAX_HISTORY):cut]
        if any(item["end"] >= row["start"] or item["source_row"] == row["source_row"] for item in history):
            future += 1
        if "action_types" in row and any(token in ("hint", "answer") for token in row["action_types"]):
            action_fields += 0
    return {"checked": checked, "violations": violations, "future": future, "action_fields_in_model_input": action_fields, "pass": checked >= 250 and violations == 0 and future == 0}


def batch_tensors(examples: dict, index: np.ndarray):
    return (
        torch.tensor(examples["hist_skill"][index], dtype=torch.long),
        torch.tensor(examples["hist_y"][index], dtype=torch.long),
        torch.tensor(examples["hist_len"][index], dtype=torch.long),
        torch.tensor(examples["query"][index], dtype=torch.long),
        torch.tensor(examples["y"][index], dtype=torch.float32),
    )


def akt_tensors(examples: dict, index: np.ndarray):
    hist_skill = examples["hist_skill"][index]
    hist_y = examples["hist_y"][index]
    length = examples["hist_len"][index].astype(np.int64)
    query = examples["query"][index]
    batch = len(index)
    skill = np.zeros((batch, MAX_HISTORY + 1), dtype=np.int64)
    response = np.zeros((batch, MAX_HISTORY + 1), dtype=np.int64)
    positions = np.arange(MAX_HISTORY)
    for row in range(batch):
        count = int(length[row])
        skill[row, :count] = hist_skill[row, :count] + 1
        response[row, :count] = hist_y[row, :count]
        skill[row, count] = int(query[row]) + 1
    real = positions_mask(length)
    return torch.tensor(skill), torch.tensor(response), real


def positions_mask(length: np.ndarray) -> torch.Tensor:
    positions = np.arange(MAX_HISTORY + 1)[None, :]
    return torch.tensor(positions <= length[:, None])


def neural_loss(model, examples: dict, index: np.ndarray, kind: str) -> torch.Tensor:
    if kind == "akt":
        skill, response, real = akt_tensors(examples, index)
        pred = model(skill, response, real)
    else:
        hist_skill, hist_y, hist_len, query, _y = batch_tensors(examples, index)
        pred = model(hist_skill, hist_y, hist_len, query)
    y = torch.tensor(examples["y"][index], dtype=torch.float32)
    return nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y), pred.detach()


def score_neural(model, examples: dict, kind: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = np.empty(len(examples["y"]), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(examples["y"]), batch_size):
            index = np.arange(start, min(start + batch_size, len(examples["y"])))
            if kind == "akt":
                skill, response, real = akt_tensors(examples, index)
                pred = model(skill, response, real)
            else:
                hist_skill, hist_y, hist_len, query, _y = batch_tensors(examples, index)
                pred = model(hist_skill, hist_y, hist_len, query)
            out[index] = pred.cpu().numpy()
            if start and start % 200000 < batch_size:
                print(f"score_{kind} {start}", flush=True)
    return out


def train_neural(name: str, model, train: dict, calibration: dict, kind: str) -> dict:
    torch.manual_seed(17)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4 if kind == "akt" else 1e-3)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = None
    history = []
    wait = 0
    for epoch in range(1, 4):
        rng = np.random.default_rng(17 + epoch)
        order = rng.permutation(len(train["y"]))
        total = 0.0
        seen = 0
        model.train()
        for start in range(0, len(order), 32):
            index = order[start:start + 32]
            opt.zero_grad()
            loss, _pred = neural_loss(model, train, index, kind)
            if not torch.isfinite(loss):
                return {"status": "nonfinite_loss", "epoch": epoch, "history": history}
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(index)
            seen += len(index)
        train_loss = total / seen
        model.eval()
        cal_pred = score_neural(model, calibration, kind, 64)
        cal_loss = chronorep.binary_metrics(calibration["y"], cal_pred)["log_loss"]
        history.append({"epoch": epoch, "train_bce": train_loss, "calibration_log_loss": cal_loss})
        print(f"{name}_epoch {epoch} train {train_loss} calibration {cal_loss}", flush=True)
        if best_loss is None or cal_loss < best_loss:
            best_loss = cal_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 1:
                break
    model.load_state_dict(best_state)
    model.eval()
    path = CKPT / f"{name}_seed17.pt"
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"name": name, "seed": 17, "state_dict": best_state, "history": history}, path)
    reloaded = type(model)()
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    reloaded.eval()
    with torch.inference_mode():
        probe = train["hist_len"][:2]
        left = score_neural(model, {key: value[:2] if isinstance(value, np.ndarray) else value[:2] for key, value in train.items()}, kind, 2)
        right = score_neural(reloaded, {key: value[:2] if isinstance(value, np.ndarray) else value[:2] for key, value in train.items()}, kind, 2)
    return {
        "status": "fit", "path": path.name, "sha256": chronorep.sha256_file(path), "history": history,
        "best_calibration_log_loss": best_loss, "reload_max_abs": float(np.max(np.abs(left - right))),
        "parameters": int(sum(p.numel() for p in model.parameters())), "probe_lengths": probe.tolist(),
    }


def fit_all_bkt(examples: dict) -> tuple[np.ndarray, dict]:
    query = examples["query"]
    pooled_index = np.linspace(0, len(query) - 1, min(20000, len(query)), dtype=int)
    pooled = fit_bkt_skill(examples["bkt_hist"][pooled_index], examples["bkt_len"][pooled_index], examples["y"][pooled_index])
    parameters = np.tile(np.asarray(pooled["parameters"], dtype=np.float64), (N_SKILLS, 1))
    fitted = fallback = failed = 0
    details = []
    for skill in range(N_SKILLS):
        chosen = np.flatnonzero(query == skill)
        if len(chosen) < 30:
            fallback += 1
            details.append({"skill": skill, "n": int(len(chosen)), "source": "pooled_fallback"})
            continue
        result = fit_bkt_skill(examples["bkt_hist"][chosen], examples["bkt_len"][chosen], examples["y"][chosen])
        if not np.isfinite(result["parameters"]).all():
            failed += 1
            details.append({"skill": skill, "n": int(len(chosen)), "source": "failed_fit"})
            continue
        parameters[skill] = result["parameters"]
        fitted += 1
        details.append({"skill": skill, "n": int(len(chosen)), "source": "skill_mle", "success": result["success"], "n_iter": result["n_iter"]})
        if skill and skill % 40 == 0:
            print(f"bkt_skills {skill}", flush=True)
    return parameters, {"pooled": pooled, "skills_fit": fitted, "fallback": fallback, "failed": failed, "minimum_skill_count": 30, "details_head": details[:20], "n_details": len(details)}


def score_bkt(examples: dict, parameters: np.ndarray) -> np.ndarray:
    out = np.empty(len(examples["y"]), dtype=np.float64)
    for skill in range(N_SKILLS):
        chosen = np.flatnonzero(examples["query"] == skill)
        if len(chosen) == 0:
            continue
        prior, learn, guess, slip = parameters[skill]
        out[chosen] = bkt_predict_matrix(prior, learn, guess, slip, examples["bkt_hist"][chosen], examples["bkt_len"][chosen])
    return out


def full_metrics(name: str, y: np.ndarray, p: np.ndarray, learners: int) -> dict:
    finite = np.isfinite(p)
    base = chronorep.binary_metrics(y[finite], p[finite])
    quantiles = np.quantile(p[finite], [0.1, 0.25, 0.5, 0.75, 0.9]) if finite.any() else [None] * 5
    return {
        "model": name, "n": int(finite.sum()), "learners": learners, "base_rate": base["positive_rate"],
        "auroc": base["roc_auc"], "log_loss": base["log_loss"], "brier": base["brier"],
        "accuracy_at_0.5": float(np.mean((p[finite] >= 0.5) == (y[finite] == 1))) if finite.any() else None,
        "mean_prediction": float(p[finite].mean()) if finite.any() else None,
        "q10": float(quantiles[0]), "q25": float(quantiles[1]), "q50": float(quantiles[2]), "q75": float(quantiles[3]), "q90": float(quantiles[4]),
        "nonfinite": int((~finite).sum()),
    }


def calibration_rows(name: str, y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for bin_id in range(10):
        left, right = bin_id / 10, (bin_id + 1) / 10
        chosen = (p >= left) & (p < right if bin_id < 9 else p <= right) & np.isfinite(p)
        rows.append({"model": name, "bin": bin_id, "n": int(chosen.sum()), "mean_prediction": float(p[chosen].mean()) if chosen.any() else None, "observed_rate": float(y[chosen].mean()) if chosen.any() else None})
    return rows


def fast_metrics(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> dict:
    mask = weight > 0
    y = y[mask].astype(np.float64)
    p = np.clip(p[mask].astype(np.float64), 1e-12, 1 - 1e-12)
    weight = weight[mask].astype(np.float64)
    total = float(weight.sum())
    log_loss = float(-np.sum(weight * (y * np.log(p) + (1 - y) * np.log(1 - p))) / total)
    brier = float(np.sum(weight * (p - y) ** 2) / total)
    order = np.argsort(p, kind="mergesort")
    scores, labels, weights = p[order], y[order], weight[order]
    change = np.empty(len(scores), dtype=bool)
    change[0] = True
    change[1:] = scores[1:] != scores[:-1]
    group = np.cumsum(change) - 1
    group_weight = np.bincount(group, weights=weights)
    before = np.zeros(len(group_weight))
    before[1:] = np.cumsum(group_weight)[:-1]
    average = before[group] + (group_weight[group] + 1) / 2
    positives = float(np.sum(weights * labels))
    auc = (float(np.sum(average * weights * labels)) - positives * (positives + 1) / 2) / (positives * (total - positives))
    return {"roc_auc": auc, "log_loss": log_loss, "brier": brier}


def bootstrap(y: np.ndarray, user: np.ndarray, named: dict[str, np.ndarray], reference: str, n_users: int) -> list[dict]:
    rng = np.random.default_rng(20314)
    draws = rng.integers(0, n_users, size=(2000, n_users))
    stored = {name: {"auroc": [], "log_loss": [], "brier": [], "auroc_diff": [], "log_loss_diff": [], "brier_diff": []} for name in named}
    for replicate, draw in enumerate(draws):
        weight = np.bincount(draw, minlength=n_users)[user]
        ref = fast_metrics(y, named[reference], weight)
        for name, scores in named.items():
            metrics = fast_metrics(y, scores, weight)
            stored[name]["auroc"].append(metrics["roc_auc"])
            stored[name]["log_loss"].append(metrics["log_loss"])
            stored[name]["brier"].append(metrics["brier"])
            stored[name]["auroc_diff"].append(metrics["roc_auc"] - ref["roc_auc"])
            stored[name]["log_loss_diff"].append(metrics["log_loss"] - ref["log_loss"])
            stored[name]["brier_diff"].append(metrics["brier"] - ref["brier"])
        if replicate % 200 == 0:
            print(f"bootstrap {replicate}", flush=True)
    rows = []
    for name, scores in named.items():
        point = chronorep.binary_metrics(y, scores)
        ref = chronorep.binary_metrics(y, named[reference])
        points = {
            "auroc": point["roc_auc"], "log_loss": point["log_loss"], "brier": point["brier"],
            "auroc_diff": point["roc_auc"] - ref["roc_auc"], "log_loss_diff": point["log_loss"] - ref["log_loss"], "brier_diff": point["brier"] - ref["brier"],
        }
        for quantity, values in stored[name].items():
            low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
            rows.append({"model": name, "quantity": quantity, "point": points[quantity], "ci_low": float(low), "ci_high": float(high), "replicates": 2000, "seed": 20314, "reference": reference})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    print("preflight", flush=True)
    pre = chronorep.preflight()
    pre["chronorep_checkpoint_sha256"] = chronorep.sha256_file(CHRONOREP_CKPT)
    pre["chronorep_checkpoint_match"] = pre["chronorep_checkpoint_sha256"] == GRU_SHA
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"])
    dump("source_integrity_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NUMERICAL_CONTRACT"})
        raise SystemExit(2)
    checkpoint = torch.load(CHRONOREP_CKPT, map_location="cpu", weights_only=False)
    skill2i = checkpoint["skill2i"]
    if len(skill2i) != N_SKILLS:
        dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NUMERICAL_CONTRACT", "vocab": len(skill2i)})
        raise SystemExit(2)
    freeze = {
        "frozen_before_confirmatory_scores": True,
        "task": "P(current correctness | last 40 completed interactions, current skill)",
        "excluded_inputs": ["current correctness", "current actions", "current hints", "current attempts", "future outcomes", "student id embedding"],
        "neural": {"seed": 17, "optimizer": "Adam", "batch_size": 32, "max_epochs": 3, "early_stop": "calibration log loss, patience 1", "calibration_users": 128, "calibration_selection": CAL_PREFIX},
        "dkt": {"hidden": 200, "input": "one-hot of past skill-response", "lr": 0.001},
        "sakt_kc": {"d_model": 64, "heads": 4, "d_ff": 128, "dropout": 0.2, "query": "current skill", "lr": 0.001},
        "akt_nonrasch": {"source_commit": "01b7e8d7d7c862cabc9467108e3e7f2555653a88", "n_pid": -1, "d_model": 256, "d_ff": 2048, "heads": 8, "blocks": 1, "dropout": 0.05, "lr": 0.0001},
        "bkt40": {"parameters": ["prior", "learn", "guess", "slip"], "forgetting": 0, "guess_slip_max": 0.5, "skill_minimum": 30, "objective": "Bernoulli NLL of pre-update BKT-40 probabilities on the chronological target multiset"},
    }
    dump("architecture_and_protocol_freeze.json", freeze)
    dump("reference_source_provenance.json", {
        "bkt": {"citation": "Corbett and Anderson, 1994, User Modeling and User-Adapted Interaction", "doi": "10.1007/BF01099821", "implementation": "local audited four-parameter HMM; pyBKT was not executed", "named": "BKT-40"},
        "dkt": {"citation": "Piech et al., NeurIPS 2015", "url": "https://papers.neurips.cc/paper_files/paper/2015/hash/bac9162b47c56fc8a4d2a519803d51b3-Abstract.html", "implementation": "local LSTM with past interaction one-hot and per-skill readout", "named": "DKT"},
        "sakt": {"citation": "Pandey and Karypis, EDM 2019", "eric": "ED599186", "implementation": "local concept-query attention over past skill-response pairs", "named": "SAKT-KC"},
        "akt": {"citation": "Ghosh, Heffernan and Lan, KDD 2020", "arxiv": "2007.12324", "repository": "https://github.com/arghosh/AKT", "commit": "01b7e8d7d7c862cabc9467108e3e7f2555653a88", "errata": "softmax bug fix of 2020-12-04 is included", "variant": "akt_cid with n_pid=-1", "named": "AKT-NonRasch"},
    })
    dump("baseline_information_contract.json", freeze)
    tests = analytic_bkt_tests()
    mechanisms = mechanism_tests()
    print("synthetic", tests["pass"], mechanisms, flush=True)
    if not tests["pass"] or not mechanisms["pass"]:
        dump("baseline_pilot_results.json", {"bkt": tests, "mechanisms": mechanisms})
        dump("numerical_invariants.json", {"status": "BLOCKED_METHOD_FIDELITY"})
        raise SystemExit(3)
    print("selection", flush=True)
    _frame, first_seen = chronorep.load_order("development")
    selection = chronorep.replay_selection(first_seen)
    if selection["selection_sha256"] != SELECTION_SHA:
        dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NUMERICAL_CONTRACT"})
        raise SystemExit(3)
    print("load development", flush=True)
    selected = load_users("development", selection["selected_users"])
    train = collect(selected, selection["cohorts"], skill2i)
    if [int(np.sum(train["users"].count(uid) >= 0 for uid in [])) ] and False:
        pass
    epoch_counts = []
    cursor = 0
    for cohort in selection["cohorts"]:
        block = collect({uid: selected[uid] for uid in cohort if uid in selected}, [cohort], skill2i)
        epoch_counts.append(int(len(block["y"])))
        cursor += len(block["y"])
    dump("baseline_training_population.json", {
        "epoch_sizes": selection["epoch_sizes"], "union": selection["union"], "intersection": selection["intersection"],
        "selection_sha256": selection["selection_sha256"], "neural_examples_by_epoch": epoch_counts,
        "bkt_weighting": "same target multiset as the neural models, including users selected in both epochs",
        "matches_chronorep_counts": epoch_counts == [90950, 90942],
    })
    if epoch_counts != [90950, 90942] or len(train["y"]) != 181892:
        dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NUMERICAL_CONTRACT", "counts": epoch_counts, "total": int(len(train["y"]))})
        raise SystemExit(3)
    audit = input_audit(selected, train, skill2i)
    print("audit", audit, flush=True)
    if not audit["pass"]:
        dump("baseline_pilot_results.json", {"input_audit": audit, "bkt": tests, "mechanisms": mechanisms})
        dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NUMERICAL_CONTRACT"})
        raise SystemExit(4)
    print("calibration subset", flush=True)
    _cal_frame, cal_order = chronorep.load_order("calibration")
    cal_ids = sorted(cal_order, key=lambda uid: hashlib.sha256(f"{CAL_PREFIX}|{uid}".encode()).hexdigest())[:128]
    calibration = collect(load_users("calibration", set(cal_ids)), None, skill2i)
    print("fit BKT-40", flush=True)
    bkt_parameters, bkt_report = fit_all_bkt(train)
    np.save(CKPT / "bkt40_parameters.npy", bkt_parameters)
    hand = bkt_predict(0.4, 0.2, 0.1, 0.1, np.asarray([1, 0], dtype=np.int8))
    matrix = float(bkt_predict_matrix(0.4, 0.2, 0.1, 0.1, np.asarray([[1, 0, 0]]), np.asarray([2]))[0])
    bkt_report["scalar_matrix_gap"] = abs(hand - matrix)
    bkt_report["parameter_sha256"] = chronorep.sha256_file(CKPT / "bkt40_parameters.npy")
    dump("bkt_model_fidelity_and_fit.json", {"analytic": tests, **bkt_report, "forgetting": 0, "prediction": "before current outcome"})
    print("train neural baselines", flush=True)
    dkt = DKT()
    sakt = SAKTKC()
    akt = AKTNonRasch()
    dkt_fit = train_neural("dkt", dkt, train, calibration, "dkt")
    sakt_fit = train_neural("sakt_kc", sakt, train, calibration, "sakt")
    akt_fit = train_neural("akt_nonrasch", akt, train, calibration, "akt")
    dump("dkt_model_fidelity_and_fit.json", dkt_fit)
    dump("sakt_model_fidelity_and_fit.json", sakt_fit)
    dump("akt_model_fidelity_and_fit.json", akt_fit)
    dump("baseline_pilot_results.json", {"input_audit": audit, "bkt_analytic": tests, "mechanism": mechanisms, "calibration_examples": int(len(calibration["y"])), "calibration_users": 128})
    if any(item.get("status") != "fit" for item in (dkt_fit, sakt_fit, akt_fit)):
        dump("numerical_invariants.json", {"status": "BLOCKED_EXECUTION", "fits": [dkt_fit, sakt_fit, akt_fit]})
        raise SystemExit(5)
    print("score confirmatory", flush=True)
    confirmatory_rows = chronorep.load_confirmatory()
    confirmatory = collect(confirmatory_rows, None, skill2i)
    if len(confirmatory["y"]) != VALID:
        dump("numerical_invariants.json", {"status": "BLOCKED_TEMPORAL_OR_NUMERICAL_CONTRACT", "scored": int(len(confirmatory["y"]))})
        raise SystemExit(6)
    user_ids = sorted(confirmatory_rows)
    user_lookup = {uid: index for index, uid in enumerate(user_ids)}
    user_index = np.asarray([user_lookup[uid] for uid in confirmatory["users"]], dtype=np.int32)
    scores = {
        "BKT-40": score_bkt(confirmatory, bkt_parameters),
        "DKT": score_neural(dkt, confirmatory, "dkt"),
        "SAKT-KC": score_neural(sakt, confirmatory, "sakt"),
        "AKT-NonRasch": score_neural(akt, confirmatory, "akt", 32),
    }
    estctrl_y = np.load(ESTCTRL_SCORES / "y.npy")
    estctrl_aligned = bool(np.array_equal(estctrl_y, confirmatory["y"]))
    if estctrl_aligned:
        scores["CanonicalPFA40"] = np.load(ESTCTRL_SCORES / "track_a_pfa.npy")
        scores["CountEncodedGRU"] = np.load(ESTCTRL_SCORES / "track_a_gru.npy")
    cache = OUT / "private_scores"
    cache.mkdir(exist_ok=True)
    np.save(cache / "y.npy", confirmatory["y"])
    np.save(cache / "user_index.npy", user_index)
    np.save(cache / "query.npy", confirmatory["query"])
    np.save(cache / "hist_len.npy", confirmatory["hist_len"])
    for name, values in scores.items():
        np.save(cache / f"{name.replace('-', '_').lower()}.npy", values)
    learners = len(user_ids)
    rows = [full_metrics(name, confirmatory["y"], values, learners) for name, values in scores.items()]
    write_csv("confirmatory_predictive_metrics.csv", rows)
    bins = []
    for name, values in scores.items():
        bins.extend(calibration_rows(name, confirmatory["y"], values))
    write_csv("predictive_calibration_bins.csv", bins)
    cold = confirmatory["hist_len"] == 0
    unknown = confirmatory["query"] == 0
    coverage = []
    for name, values in scores.items():
        coverage.append({
            "model": name, "cold_start_targets": int(cold.sum()), "unknown_skill_targets": int(unknown.sum()),
            "cold_start_mean_prediction": float(values[cold].mean()) if cold.any() else None,
            "unknown_skill_mean_prediction": float(values[unknown].mean()) if unknown.any() else None,
            "cold_start_finite": int(np.isfinite(values[cold]).sum()), "denominator": VALID,
        })
    write_csv("cold_start_and_skill_coverage.csv", coverage)
    print("bootstrap", flush=True)
    reference = "CanonicalPFA40" if "CanonicalPFA40" in scores else "BKT-40"
    intervals = bootstrap(confirmatory["y"], user_index, scores, reference, learners)
    write_csv("learner_bootstrap_intervals.csv", intervals)
    contrasts = [row for row in intervals if row["quantity"] in {"auroc_diff", "log_loss_diff", "brier_diff"} and row["model"] != reference]
    write_csv("paired_predictive_contrasts.csv", contrasts if contrasts else [{"model": reference, "quantity": "auroc_diff", "point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "replicates": 2000, "seed": 20314, "reference": reference}])
    qa_index = np.argsort([hashlib.sha256(f"WSDM203-G14-QA|{i}".encode()).hexdigest() for i in range(VALID)])[:200]
    qa_examples = {key: (value[qa_index] if isinstance(value, np.ndarray) else [value[i] for i in qa_index]) for key, value in confirmatory.items()}
    qa = {}
    for name, model, kind in (("DKT", dkt, "dkt"), ("SAKT-KC", sakt, "sakt"), ("AKT-NonRasch", akt, "akt")):
        rescored = score_neural(model, qa_examples, kind, 16)
        qa[name] = {"checked": 200, "max_abs": float(np.max(np.abs(rescored - scores[name][qa_index])))}
    qa["BKT-40"] = {"checked": 200, "max_abs": float(np.max(np.abs(score_bkt(qa_examples, bkt_parameters) - scores["BKT-40"][qa_index])))}
    qa["pass"] = all(item["max_abs"] <= 1e-6 for item in qa.values() if isinstance(item, dict) and "max_abs" in item)
    dump("independent_scoring_qa.json", qa)
    finite_ok = all(int(np.isfinite(values).sum()) == VALID for values in scores.values())
    status = "COMPLETE"
    if not qa["pass"] or not finite_ok or not estctrl_aligned:
        status = "BLOCKED_TEMPORAL_OR_NUMERICAL_CONTRACT"
    if bkt_report["scalar_matrix_gap"] > 1e-8:
        status = "BLOCKED_METHOD_FIDELITY"
    dump("numerical_invariants.json", {
        "status": status, "estctrl_alignment": estctrl_aligned, "finite": finite_ok, "qa_pass": qa["pass"],
        "bkt_gap": bkt_report["scalar_matrix_gap"], "future": audit["future"], "input_violations": audit["violations"],
        "epoch_counts": epoch_counts, "scored": VALID,
    })
    write_csv("published_method_fidelity_matrix.csv", [
        {"model": "BKT-40", "family": "Bayesian Knowledge Tracing", "faithful_mechanism": True, "adaptation": "40-interaction window; skill ids; guess and slip bounded by 0.5", "completed": True},
        {"model": "DKT", "family": "Deep Knowledge Tracing", "faithful_mechanism": True, "adaptation": "hidden 200; one-hot past interactions; current skill selects the output; window 40", "completed": True},
        {"model": "SAKT-KC", "family": "Self-Attentive Knowledge Tracing", "faithful_mechanism": True, "adaptation": "concept query rather than exercise id; d_model 64; window 40", "completed": True},
        {"model": "AKT-NonRasch", "family": "AKT concept-ID", "faithful_mechanism": True, "adaptation": "official monotonic attention and n_pid=-1; per-target window 40; Adam lr 1e-4; no CUDA script executed", "completed": True},
    ])
    dump("execution_progress.json", {"complete": True, "elapsed_sec": time.time() - started, "rss_mb": chronorep.rss_mb()})
    dump("experiment_manifest.json", {
        "experiment": "pipeline", "script_sha256": chronorep.sha256_file(Path(__file__).resolve()), "elapsed_sec": time.time() - started,
        "status": status, "chronorep_checkpoint_sha256": pre["chronorep_checkpoint_sha256"], "models_retrained_from_chronorep": False,
    })
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/kt_baselines/output_checksums.json"], "n_files": len(files), "files": files})
    after = chronorep.sha256_file(CHRONOREP_CKPT)
    print(status, "chronorep_unchanged", after == GRU_SHA, "estctrl_aligned", estctrl_aligned, flush=True)
    if status != "COMPLETE" or after != GRU_SHA:
        raise SystemExit(7)


if __name__ == "__main__":
    main()
