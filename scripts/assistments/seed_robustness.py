#!/usr/bin/env python3
"""Fixed-cohort initialization robustness for the primary and matched GRU contracts.

Trains seeds 29 and 41. Seed 17 is not retrained.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "matchedinfo", ROOT / "scripts/assistments/matched_information.py"
)
matchedinfo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matchedinfo)
chronorep = matchedinfo.chronorep

from r7_core import MAX_HIST_TOKENS, MAX_SEQ_ACTIONS, ActionTokenGRU, NeuralPilot  # noqa: E402

OUT = ROOT / "results/assistments/seed_robustness"
CACHE = OUT / "private_score_caches"
CHRONOREP_DIR = ROOT / "results/assistments/chronological_replication"
MATCHEDINFO_DIR = ROOT / "results/assistments/matched_information"
SEEDS = (29, 41)
MATCHED = ("HIST", "LEN", "ACT_MATCHED")
CONTRACTS = ("HIST", "LEN", "ACT_MATCHED", "ACT_PRIMARY")
SELECTION_SHA = "cbe4fd9e0075c78ae5997892153d37920497d14c64a21af3ffbf3594e5c907e2"
FROZEN = {
    "zip": "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445",
    "development": "69b6b7b38738d293651b728326658c1ce517c87661d012bc03ee2ced228e9d70",
    "confirmatory": "0193ddd7b26517f1b48591d09d86f5f3ce8b9930a1285006833fce4ea5096f99",
    "chronorep_act": "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196",
    "hist": "7815232ef52b7aca7214a3eb6a9dc32f68b81575731fe91a5ac40af36b1ce6bc",
    "len": "88d620d16ab22d8a6f61f8f09547b35ab8d3d9ca78cb7071f5bca9fe3bdf46ec",
    "act_matched": "8cc717595eb03df5561cd016f73eade0f548c4707cd7e59e8f0ec593fbca4ed2",
}


def dump(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, default=chronorep._json_default) + "\n", encoding="utf-8")


def checkpoint_name(contract: str, seed: int) -> str:
    return f"{contract.lower()}_seed{seed}.pt"


def state_sha(model: ActionTokenGRU) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def init_model(seed: int, skill2i: dict, atype2i: dict) -> ActionTokenGRU:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ActionTokenGRU(len(skill2i) - 1, len(atype2i) - 1)
    model.to(torch.device("cpu"))
    return model


def chronorep_tokens(row, by_end, skill2i, atype2i, outcome0, outcome1):
    full_s, full_a, cut = matchedinfo.history_tokens(by_end, row["start"], skill2i, atype2i, outcome0, outcome1)
    actions = row["action_types"][:MAX_SEQ_ACTIONS]
    sid = skill2i.get(row["skill_id"], 0)
    current = [atype2i.get(tok, 0) for tok in actions]
    skills = full_s + [sid] * len(current)
    atypes = full_a + current
    return skills[-MAX_HIST_TOKENS:], atypes[-MAX_HIST_TOKENS:], cut, len(actions), full_s, full_a


def sequence_for(contract, row, by_end, skill2i, atype2i, marker, outcome0, outcome1):
    if contract == "ACT_PRIMARY":
        skills, atypes, cut, length, full_s, full_a = chronorep_tokens(row, by_end, skill2i, atype2i, outcome0, outcome1)
        kept = min(len(full_s), max(0, MAX_HIST_TOKENS - length))
        return skills, atypes, cut, length, full_s[-32:], full_a[-32:], kept
    built = matchedinfo.arm_sequences(row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
    key = "ACT" if contract == "ACT_MATCHED" else contract
    skills, atypes = built[key]
    hist_s, hist_a = built["hist_prefix"]
    return skills, atypes, built["cut"], built["length"], hist_s, hist_a, built["chronorep_history_tokens_retained"]


def load_training_rows(selected: set[str]) -> dict:
    rows = defaultdict(list)
    parquet = pq.ParquetFile(chronorep.RECON / "development.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=50000):
        raw = batch.to_pylist()
        starts = chronorep.civil_us([item["start_time"] for item in raw])
        ends = chronorep.civil_us([item["end_time"] for item in raw])
        for item, start, end in zip(raw, starts, ends):
            if str(item["user_id"]) in selected:
                rows[str(item["user_id"])].append(chronorep.compact_row(item, int(start), int(end)))
        seen += len(raw)
        if seen % 1000000 == 0:
            print(f"training_rows {seen}", flush=True)
    return rows


def run_qa(targets, skill2i, atype2i, marker, outcome0, outcome1) -> dict:
    pilot = NeuralPilot()
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.model = ActionTokenGRU(len(skill2i) - 1, len(atype2i) - 1)
    indexed = []
    length_available = defaultdict(int)
    for row, by_end in targets:
        built = matchedinfo.arm_sequences(row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
        length_available[built["length"]] += 1
        unfinished = any(item["source_row"] != row["source_row"] and item["start"] < row["start"] and item["end"] >= row["start"] for item in by_end)
        indexed.append((row, by_end, built, unfinished))
    picked = []
    seen = set()
    stratum_counts = {}

    def take(name, predicate, limit):
        available = 0
        got = 0
        for item in indexed:
            if not predicate(item):
                continue
            available += 1
            if got < limit and item[0]["source_row"] not in seen:
                picked.append((name, item))
                seen.add(item[0]["source_row"])
                got += 1
        stratum_counts[name] = {"available": available, "selected": got}

    for length in (1, 2, 3, 4, 8, 16, 32):
        take(f"length_{length}", lambda item, length=length: item[2]["length"] == length, 20)
    take("short_history", lambda item: item[2]["full_history_tokens"] <= 8, 20)
    take("long_history", lambda item: item[2]["full_history_tokens"] >= 64, 20)
    take("recent_predecessor", lambda item: item[2]["cut"] >= 1, 20)
    take("start_order_would_include_unfinished", lambda item: item[3], 20)
    take("chronorep_budget_differs_from_32", lambda item: item[2]["chronorep_history_tokens_retained"] != min(item[2]["full_history_tokens"], 32), 30)
    ranked = sorted(indexed, key=lambda item: hashlib.sha256(f"WSDM203-G10-QA|{item[0]['source_row']}".encode()).hexdigest())
    for item in ranked:
        if len(picked) >= 250:
            break
        if item[0]["source_row"] not in seen:
            picked.append(("hash_fill", item))
            seen.add(item[0]["source_row"])
    violations = defaultdict(int)
    examples = defaultdict(list)
    arms_seen = defaultdict(int)
    for _name, (row, by_end, built, _unfinished) in picked:
        hist_s, hist_a = built["hist_prefix"]
        for contract in CONTRACTS:
            skills, atypes, cut, length, prefix_s, prefix_a, chronorep_kept = sequence_for(
                contract, row, by_end, skill2i, atype2i, marker, outcome0, outcome1
            )
            arms_seen[contract] += 1
            if contract != "ACT_PRIMARY" and (prefix_s != hist_s or prefix_a != hist_a):
                violations["matched_prefix"] += 1
            if any(tok in (outcome0, outcome1) for tok in atypes[len(prefix_a) if contract != "ACT_PRIMARY" else max(0, len(atypes) - length) :]):
                violations["current_outcome"] += 1
            if cut and any(item["end"] >= row["start"] for item in by_end[:cut]):
                violations["future_history"] += 1
            if contract == "HIST" and (len(skills) != len(hist_s) + 1 or atypes[-1] != marker):
                violations["hist_marker"] += 1
            if contract == "LEN" and len(skills) != len(hist_s) + length:
                violations["len_count"] += 1
            if contract == "LEN" and any(tok != marker for tok in atypes[len(hist_a) :]):
                violations["len_identity"] += 1
            if contract == "ACT_MATCHED" and len(skills) != len(hist_s) + length:
                violations["act_length"] += 1
            if contract == "ACT_PRIMARY" and len(skills) > MAX_HIST_TOKENS:
                violations["chronorep_window"] += 1
        hist_seq = sequence_for("HIST", row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
        len_seq = sequence_for("LEN", row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
        act_seq = sequence_for("ACT_MATCHED", row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
        if len(len_seq[0]) != len(act_seq[0]):
            violations["len_act_equal_length"] += 1
        prefix = chronorep.prefix_events(by_end, row["start"])
        reference = pilot._tokens(prefix, chronorep.as_event(row), False)
        chronorep_seq = chronorep_tokens(row, by_end, skill2i, atype2i, outcome0, outcome1)
        if reference != (chronorep_seq[0], chronorep_seq[1]):
            violations["chronorep_token_reference"] += 1
            if len(examples["chronorep_token_reference"]) < 3:
                examples["chronorep_token_reference"].append({"source_row": int(row["source_row"]), "reference_length": len(reference[0]), "constructed_length": len(chronorep_seq[0])})
        if hist_seq[4] != len_seq[4] or hist_seq[4] != act_seq[4]:
            violations["shared_history"] += 1
    return {
        "design": "FIXED_COHORT_INITIALIZATION_AND_OPTIMIZATION_SEED_ROBUSTNESS",
        "cases": len(picked),
        "target_cases": 250,
        "stratum_counts": stratum_counts,
        "length_available_in_training_targets": {str(key): int(value) for key, value in sorted(length_available.items())},
        "arms_checked": dict(arms_seen),
        "violations": dict(violations),
        "violation_examples": dict(examples),
        "vocabulary_action_ids": atype2i,
        "epochs_represented": "unique valid targets from both fixed epochs",
    }


def train_model(seed, contract, epochs, skill2i, atype2i, marker, outcome0, outcome1, init_hash: str) -> dict:
    path = OUT / checkpoint_name(contract, seed)
    progress = OUT / f"{checkpoint_name(contract, seed)}.progress.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        print(f"resume_checkpoint {contract} seed {seed}", flush=True)
        return state["log"]
    model = init_model(seed, skill2i, atype2i)
    observed = state_sha(model)
    if observed != init_hash:
        raise SystemExit(f"initial hash changed for {contract} seed {seed}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses, used = [], []
    next_epoch, next_index = 1, 0
    epoch_loss_sum, epoch_n = 0.0, 0
    resumed_flag = False
    if progress.exists():
        resumed_flag = True
        prog = torch.load(progress, map_location="cpu", weights_only=False)
        if prog["seed"] != seed or prog["contract"] != contract or prog["initial_parameter_sha256"] != init_hash:
            raise SystemExit(f"progress identity mismatch for {contract} seed {seed}")
        model.load_state_dict(prog["state_dict"])
        opt.load_state_dict(prog["optimizer"])
        losses = list(prog["losses"])
        used = list(prog["used"])
        next_epoch = int(prog["next_epoch"])
        next_index = int(prog["next_index"])
        epoch_loss_sum = float(prog["epoch_loss_sum"])
        epoch_n = int(prog["epoch_n"])
        print(f"resume_progress {contract} seed {seed} epoch {next_epoch} index {next_index}", flush=True)
    model.train()
    last_skills, last_atypes = [marker], [marker]
    for epoch_i, epoch in enumerate(epochs, start=1):
        if epoch_i < next_epoch:
            continue
        begin = next_index if epoch_i == next_epoch else 0
        targets = epoch["targets"]
        for offset in range(begin, len(targets)):
            row, by_end = targets[offset]
            skills, atypes, *_rest = sequence_for(contract, row, by_end, skill2i, atype2i, marker, outcome0, outcome1)
            if not skills:
                raise SystemExit(f"empty input {contract} seed {seed}")
            last_skills, last_atypes = skills, atypes
            pred, _ = model.forward_from_state(
                torch.tensor([skills], dtype=torch.long),
                torch.tensor([atypes], dtype=torch.long),
                None,
            )
            loss = torch.nn.functional.binary_cross_entropy(pred, torch.tensor([float(row["correct"])]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.item())
            epoch_n += 1
            if epoch_n % 20000 == 0:
                print(f"train {contract} seed {seed} epoch {epoch_i} {epoch_n}", flush=True)
                torch.save({
                    "seed": seed, "contract": contract, "initial_parameter_sha256": init_hash,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "optimizer": opt.state_dict(), "losses": losses, "used": used,
                    "next_epoch": epoch_i, "next_index": offset + 1,
                    "epoch_loss_sum": epoch_loss_sum, "epoch_n": epoch_n,
                }, progress)
        losses.append(epoch_loss_sum / epoch_n)
        used.append(epoch_n)
        print(f"train_done {contract} seed {seed} epoch {epoch_i} loss {losses[-1]} n {epoch_n}", flush=True)
        epoch_loss_sum, epoch_n = 0.0, 0
        next_index = 0
        torch.save({
            "seed": seed, "contract": contract, "initial_parameter_sha256": init_hash,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": opt.state_dict(), "losses": losses, "used": used,
            "next_epoch": epoch_i + 1, "next_index": 0, "epoch_loss_sum": 0.0, "epoch_n": 0,
        }, progress)
    model.eval()
    with torch.inference_mode():
        repeat = model.forward_from_state(torch.tensor([last_skills], dtype=torch.long), torch.tensor([last_atypes], dtype=torch.long), None)[0].item()
        again = model.forward_from_state(torch.tensor([last_skills], dtype=torch.long), torch.tensor([last_atypes], dtype=torch.long), None)[0].item()
    log = {
        "seed": seed, "contract": contract, "initial_parameter_sha256": init_hash,
        "training_selection_sha256": SELECTION_SHA, "examples_by_epoch": used, "loss_by_epoch": losses,
        "optimizer": "Adam", "learning_rate": 0.001, "epochs": 2, "loss_name": "binary_cross_entropy",
        "warm_start": False, "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "deterministic_repeat_abs_diff": abs(float(repeat) - float(again)),
        "checkpoint": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "interruptions": "resumed_from_progress" if resumed_flag else "none",
    }
    torch.save({"log": log, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "seed": seed, "contract": contract}, path)
    log["checkpoint_sha256"] = chronorep.sha256_file(path)
    log["checkpoint_bytes"] = path.stat().st_size
    torch.save({"log": log, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "seed": seed, "contract": contract}, path)
    log["checkpoint_sha256"] = chronorep.sha256_file(path)
    log["checkpoint_bytes"] = path.stat().st_size
    if progress.exists():
        progress.unlink()
    return log


def score_matched(models, grouped, skill2i, atype2i, marker, outcome0, outcome1):
    users = sorted(grouped)
    index = {uid: i for i, uid in enumerate(users)}
    out = {name: [] for name in models}
    labels, users_i, keys = [], [], []
    violations = defaultdict(int)
    for uid in users:
        by_start, by_end = chronorep.order_valid(grouped[uid])
        pairs = {name: [] for name in models}
        rows = []
        for row in by_start:
            rows.append(row)
            for name, model in models.items():
                skills, atypes, cut, length, prefix_s, prefix_a, _kept = sequence_for(
                    name, row, by_end, skill2i, atype2i, marker, outcome0, outcome1
                )
                if name != "ACT_PRIMARY" and (skills[: len(prefix_s)] != prefix_s or atypes[: len(prefix_a)] != prefix_a):
                    violations["prefix"] += 1
                current = atypes[len(prefix_a) :] if name != "ACT_PRIMARY" else atypes[-length:] if length else []
                if any(tok in (outcome0, outcome1) for tok in current):
                    violations["current_outcome"] += 1
                if cut and any(item["end"] >= row["start"] for item in by_end[:cut]):
                    violations["future"] += 1
                pairs[name].append((skills, atypes))
        pred = {name: chronorep.score_forward(model, pairs[name], torch.device("cpu")) for name, model in models.items()}
        for event_i, row in enumerate(rows):
            labels.append(row["correct"])
            users_i.append(index[uid])
            keys.append(hashlib.blake2s(f"WSDM203-G9|{row['source_row']}".encode(), digest_size=16).hexdigest())
            for name in models:
                out[name].append(float(pred[name][event_i]))
    return {
        "y": np.asarray(labels, dtype=np.int8),
        "user_index": np.asarray(users_i, dtype=np.int32),
        "keys": keys,
        "scores": {name: np.asarray(values, dtype=np.float64) for name, values in out.items()},
        "violations": dict(violations),
    }


def save_cache(name: str, payload: dict, score_name: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "event_key": payload["keys"],
        "user_index": payload["user_index"],
        "correct": payload["y"],
        "score": payload["scores"][score_name],
    })
    pq.write_table(table, CACHE / f"{name}.parquet")


def metric_row(seed, contract, y, p) -> dict:
    base = chronorep.binary_metrics(y, p)
    finite = np.isfinite(p)
    return {
        "seed": seed, "contract": contract, "n": int(len(y)), "n_learners": 6766,
        "base_rate": base["positive_rate"], "auroc": base["roc_auc"], "log_loss": base["log_loss"], "brier": base["brier"],
        "accuracy_at_0.5": float(np.mean((p >= 0.5) == y)) if len(y) else None,
        "mean_prediction": float(p.mean()) if len(p) else None,
        "nonfinite": int((~finite).sum()),
        "outside_unit_interval": int(np.sum((p < 0) | (p > 1))),
    }


def calibration_rows(seed, contract, y, p) -> list[dict]:
    bins = np.minimum((np.clip(p, 0.0, 1.0) * 10).astype(int), 9)
    rows = []
    for bin_id in range(10):
        mask = bins == bin_id
        rows.append({
            "seed": seed, "contract": contract, "bin": bin_id, "lo": bin_id / 10, "hi": (bin_id + 1) / 10,
            "n": int(mask.sum()),
            "mean_prediction": float(p[mask].mean()) if mask.any() else None,
            "observed_frequency": float(y[mask].mean()) if mask.any() else None,
        })
    return rows


def fast_auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    order = np.argsort(p, kind="mergesort")
    y_sorted = y[order]
    p_sorted = p[order]
    change = np.empty(len(p), dtype=bool)
    change[0] = True
    change[1:] = p_sorted[1:] != p_sorted[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(p)
    ranks = np.repeat(0.5 * (starts + 1 + ends), ends - starts)
    n_pos = float(y_sorted.sum())
    n_neg = float(len(y) - n_pos)
    return float((np.sum(ranks * y_sorted) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_seed17_metrics() -> list[dict]:
    frame = pd.read_csv(MATCHEDINFO_DIR / "confirmatory_predictive_metrics.csv")
    rows = []
    for record in frame.to_dict(orient="records"):
        contract = "ACT_MATCHED" if record["model"] == "ACT" else record["model"]
        rows.append({
            "seed": 17, "contract": contract, "n": int(record["n"]), "n_learners": int(record["n_learners"]),
            "base_rate": record["base_rate"], "auroc": record["auroc"], "log_loss": record["log_loss"], "brier": record["brier"],
            "accuracy_at_0.5": record["accuracy_at_0.5"], "mean_prediction": record["mean_prediction"],
            "nonfinite": int(record["nonfinite"]), "outside_unit_interval": int(record["outside_unit_interval"]),
            "source": "results/assistments/matched_information/confirmatory_predictive_metrics.csv",
        })
    return rows


def contrasts_from(metrics: list[dict]) -> list[dict]:
    rows = []
    by_seed = defaultdict(dict)
    for row in metrics:
        if row["contract"] in MATCHED:
            by_seed[int(row["seed"])][row["contract"]] = row
    for seed, arms in sorted(by_seed.items()):
        for left, right, label in (("LEN", "HIST", "LEN_minus_HIST"), ("ACT_MATCHED", "LEN", "ACT_MATCHED_minus_LEN"), ("ACT_MATCHED", "HIST", "ACT_MATCHED_minus_HIST")):
            rows.append({
                "seed": seed, "contrast": label, "n": arms[left]["n"],
                "auroc_difference": arms[left]["auroc"] - arms[right]["auroc"],
                "log_loss_difference": arms[left]["log_loss"] - arms[right]["log_loss"],
                "brier_difference": arms[left]["brier"] - arms[right]["brier"],
                "len_exceeds_hist": bool(arms["LEN"]["auroc"] > arms["HIST"]["auroc"]),
                "act_exceeds_len": bool(arms["ACT_MATCHED"]["auroc"] > arms["LEN"]["auroc"]),
            })
    return rows


def seed_summary(contrasts: list[dict]) -> list[dict]:
    frame = pd.DataFrame(contrasts)
    rows = []
    for contrast, group in frame.groupby("contrast"):
        for column in ("auroc_difference", "log_loss_difference", "brier_difference"):
            values = group[column].to_numpy(dtype=float)
            rows.append({
                "contrast": contrast, "metric": column, "n_seeds": int(len(values)),
                "mean": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max()),
                "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
                "interval_type": "three-seed descriptive range, not a 95 percent confidence interval",
            })
    return rows


def certificate_tables(eval_out, seed: int):
    status = eval_out["status"]
    model_i = 1
    frontier, false_rows, transition_rows, threshold_rows = [], [], [], []
    valid = 939819
    for schema_i, schema in enumerate(chronorep.SCHEMAS):
        for tau_i, tau in enumerate(chronorep.TAUS):
            unsup = int(status[:, model_i, schema_i, tau_i, 0].sum())
            stable = int(status[:, model_i, schema_i, tau_i, 1].sum())
            sett = int(status[:, model_i, schema_i, tau_i, 2].sum())
            if unsup + stable + sett != valid:
                raise SystemExit(f"certificate accounting {schema} {tau} {unsup + stable + sett}")
            threshold_rows.append({
                "seed": seed, "schema": schema, "tau": tau, "unsupported": unsup, "stable": stable,
                "set_valued": sett, "event_nc": unsup + sett, "event_nc_rate": (unsup + sett) / valid, "valid_targets": valid,
            })
            if abs(tau - 0.8) < 1e-9:
                fs = int(eval_out["false_stable"][:, model_i, schema_i].sum())
                nu = int(eval_out["uncovered_stable"][:, model_i, schema_i].sum())
                frontier.append({
                    "seed": seed, "schema": schema, "tau": 0.8, "valid_targets": valid,
                    "unsupported": unsup, "stable": stable, "set_valued": sett, "supported": stable + sett,
                    "event_nc": unsup + sett, "event_nc_rate": (unsup + sett) / valid,
                    "fingerprint_covered": int(eval_out["covered"][:, model_i, schema_i].sum()),
                    "learners_with_supported_event": int(np.sum(eval_out["supported_n"][:, model_i, schema_i] > 0)),
                })
                false_rows.append({
                    "seed": seed, "schema": schema, "n_S": stable, "n_U": nu, "n_FS": fs,
                    "n_FS_over_n_S": fs / stable if stable else None,
                    "n_U_over_n_S": nu / stable if stable else None,
                    "n_FS_over_n_U": fs / nu if nu else None,
                })
    labels = ("unsupported", "stable", "set_valued")
    for tau in chronorep.TAUS:
        matrix = eval_out["transitions"][f"ActionTokenGRU|{tau}|S0|S1"]
        if int(matrix.sum()) != valid:
            raise SystemExit(f"transition accounting {tau} {int(matrix.sum())}")
        for i, src in enumerate(labels):
            for j, dst in enumerate(labels):
                transition_rows.append({"seed": seed, "tau": tau, "from_schema": "S0", "to_schema": "S1", "from_label": src, "to_label": dst, "n": int(matrix[i, j]), "denominator": valid})
        loss = int(matrix[1, 0] + matrix[2, 0])
        resolved = int(matrix[2, 1])
        nc0 = int(matrix[0, :].sum() + matrix[2, :].sum())
        nc1 = int(matrix[:, 0].sum() + matrix[:, 2].sum())
        transition_rows.append({"seed": seed, "tau": tau, "from_schema": "S0", "to_schema": "S1", "from_label": "SUMMARY", "to_label": "support_loss", "n": loss, "denominator": valid})
        transition_rows.append({"seed": seed, "tau": tau, "from_schema": "S0", "to_schema": "S1", "from_label": "SUMMARY", "to_label": "set_valued_to_stable", "n": resolved, "denominator": valid})
        transition_rows.append({"seed": seed, "tau": tau, "from_schema": "S0", "to_schema": "S1", "from_label": "SUMMARY", "to_label": "event_nc_s1_minus_s0", "n": nc1 - nc0, "denominator": valid})
    native, matched = [], []
    for kind, store_name, bucket in (("native_completion_time", "completion", native), ("matched_event", "matched", matched)):
        masks = []
        for schema in chronorep.SCHEMAS:
            cell = eval_out[store_name]["ActionTokenGRU"][schema]
            mask = np.isfinite(cell["lo"]) & np.isfinite(cell["hi"])
            masks.append(mask)
            counts = chronorep.membership_counts(cell["lo"][mask], cell["hi"][mask], 0.05)
            if counts["guaranteed_in"] + counts["guaranteed_out"] + counts["unresolved"] != int(mask.sum()):
                raise SystemExit(f"capacity partition {kind} {schema}")
            risk = chronorep.risk_block("ActionTokenGRU", schema, 0.05, cell["lo"][mask], cell["hi"][mask], cell["actual"][mask], kind)
            if risk["inside"] + risk["outside"] != int(mask.sum()):
                raise SystemExit(f"risk partition {kind} {schema}")
            bucket.append({
                "seed": seed, "schema": schema, "kappa": 0.05, "n_full": 6766, "n_eligible": int(mask.sum()),
                "n_no_checkpoint": 6766 - int(mask.sum()), "B": counts["B"],
                "guaranteed_in": counts["guaranteed_in"], "guaranteed_out": counts["guaranteed_out"], "unresolved": counts["unresolved"],
                "all_user_capacity_nc": (6766 - int(mask.sum()) + counts["unresolved"]) / 6766,
                "inside_interval": risk["inside"], "outside_interval": risk["outside"],
                "false_guaranteed_in": risk["false_guaranteed_in"], "false_guaranteed_out": risk["false_guaranteed_out"],
                "model": "ActionTokenGRU", "tie_rule": "adversarial",
            })
        if store_name == "matched" and any(not np.array_equal(masks[0], item) for item in masks[1:]):
            raise SystemExit("matched-event checkpoint masks differ across schemas")
    return frontier, false_rows, transition_rows, threshold_rows, native, matched


def seed17_certificates():
    frontier = pd.read_csv(CHRONOREP_DIR / "event_certificate_frontier.csv")
    frontier = frontier[frontier.model == "ActionTokenGRU"].copy()
    frontier["seed"] = 17
    frontier["source"] = "chronorep_event_certificate_frontier.csv"
    false_rows = frontier[["seed", "schema", "n_S", "n_U", "n_FS", "n_U_over_n_S", "n_FS_over_n_U"]].copy()
    false_rows["n_FS_over_n_S"] = false_rows["n_FS"] / false_rows["n_S"]
    transitions = pd.read_csv(CHRONOREP_DIR / "paired_event_transitions.csv")
    transitions = transitions[(transitions.model == "ActionTokenGRU") & (transitions.tau == 0.8) & (transitions.from_schema == "S0") & (transitions.to_schema == "S1")].copy()
    transitions["seed"] = 17
    native = pd.read_csv(CHRONOREP_DIR / "native_completion_time_capacity.csv")
    native = native[(native.model == "ActionTokenGRU") & (native.kappa == 0.05)].copy()
    native["seed"] = 17
    matched = pd.read_csv(CHRONOREP_DIR / "matched_event_capacity.csv")
    matched = matched[(matched.model == "ActionTokenGRU") & (matched.kappa == 0.05)].copy()
    matched["seed"] = 17
    risk = pd.read_csv(CHRONOREP_DIR / "realized_risk_validity.csv")
    risk = risk[(risk.model == "ActionTokenGRU") & (risk.kappa == 0.05)]
    return frontier, false_rows, transitions, native, matched, risk


def equivalence(pilot, grouped) -> dict:
    pool = []
    for uid, rows in grouped.items():
        for row in rows:
            if row["valid"]:
                pool.append((hashlib.sha256(f"WSDM203-G10-EQ|{row['source_row']}".encode()).hexdigest(), uid, row))
    pool.sort()
    mismatches = defaultdict(int)
    max_diff = 0.0
    checked = 0
    for _rank, uid, row in pool[:200]:
        _by_start, by_end = chronorep.order_valid(grouped[uid])
        prefix = chronorep.prefix_events(by_end, row["start"])
        current = chronorep.as_event(row)
        reference = pilot.mastery_branch(prefix, current)
        batched = float(chronorep.score_forward(pilot.model, [pilot._tokens(prefix, current, False)], torch.device("cpu"))[0])
        max_diff = max(max_diff, abs(reference - batched))
        alt_actions = tuple(list(current.action_types[:32])[::-1]) or ("hint",)
        alt = chronorep.Event("", "", current.skill_id, 0, 0, 0, "", alt_actions, tuple(), "")
        ref_alt = pilot.mastery_branch(prefix, alt)
        bat_alt = float(chronorep.score_forward(pilot.model, [pilot._tokens(prefix, alt, False)], torch.device("cpu"))[0])
        max_diff = max(max_diff, abs(ref_alt - bat_alt))
        for tau in (0.70, 0.80, 0.90):
            if (reference >= tau) != (batched >= tau) or (ref_alt >= tau) != (bat_alt >= tau):
                mismatches[str(tau)] += 1
        checked += 1
    return {"events": checked, "max_abs_diff": max_diff, "action_label_mismatches": dict(mismatches)}


def load_pilot(path, skill2i, atype2i) -> NeuralPilot:
    pilot = NeuralPilot()
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.model = ActionTokenGRU(len(skill2i) - 1, len(atype2i) - 1)
    state = torch.load(path, map_location="cpu", weights_only=False)
    pilot.model.load_state_dict(state["state_dict"])
    pilot.model.eval()
    return pilot


def write_checksums() -> None:
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/seed_robustness/output_checksums.json"], "n_files": len(files), "files": files})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    pre = chronorep.preflight()
    paths = {
        "chronorep_act": CHRONOREP_DIR / "gru_chronological_seed17.pt",
        "hist": MATCHEDINFO_DIR / "gru_hist_seed17.pt",
        "len": MATCHEDINFO_DIR / "gru_length_seed17.pt",
        "act_matched": MATCHEDINFO_DIR / "gru_action_seed17_matched.pt",
    }
    hashes = {name: chronorep.sha256_file(path) for name, path in paths.items()}
    pre["frozen_checkpoints"] = hashes
    pre["checkpoint_match"] = hashes == {key: FROZEN[key] for key in hashes}
    pre["development_hash_match"] = pre["parquet"]["development"]["sha256"] == FROZEN["development"]
    pre["confirmatory_hash_match"] = pre["parquet"]["confirmatory"]["sha256"] == FROZEN["confirmatory"]
    pre["pass"] = bool(pre["pass"] and pre["checkpoint_match"] and pre["development_hash_match"] and pre["confirmatory_hash_match"])
    dump("source_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_SOURCE"})
        write_checksums()
        raise SystemExit(2)
    checkpoint = torch.load(paths["chronorep_act"], map_location="cpu", weights_only=False)
    skill2i = checkpoint["skill2i"]
    atype2i = checkpoint["atype2i"]
    if len(skill2i) - 1 != 264 or len(atype2i) - 1 != 8 or "start" not in atype2i:
        raise SystemExit("vocabulary mismatch")
    marker = atype2i["start"]
    outcome0 = len(atype2i)
    outcome1 = len(atype2i) + 1
    del checkpoint
    print("preflight_pass", flush=True)
    seed17_cache = pd.read_parquet(MATCHEDINFO_DIR / "restricted_event_score_cache.parquet")
    seed17_metrics = pd.read_csv(MATCHEDINFO_DIR / "confirmatory_predictive_metrics.csv")
    reproduction = {}
    for model, column in (("HIST", "p_hist"), ("LEN", "p_len"), ("ACT", "p_act")):
        scored = chronorep.binary_metrics(seed17_cache.correct.to_numpy(), seed17_cache[column].to_numpy())
        stored = float(seed17_metrics.loc[seed17_metrics.model == model, "auroc"].iloc[0])
        reproduction[model] = {"cache_auroc": scored["roc_auc"], "stored_auroc": stored, "abs_diff": abs(scored["roc_auc"] - stored)}
    stored_frontier_all = pd.read_csv(CHRONOREP_DIR / "event_certificate_frontier.csv")
    gru17 = stored_frontier_all[stored_frontier_all.model == "ActionTokenGRU"]
    status_sums = (gru17.unsupported + gru17.stable + gru17.set_valued).astype(int).tolist()
    pfa_s0_stable_stored = int(stored_frontier_all[(stored_frontier_all.model == "TransparentActionPFA") & (stored_frontier_all.schema == "S0") & (stored_frontier_all.tau == 0.8)].stable.iloc[0])
    fast_check = abs(fast_auroc(seed17_cache.correct.to_numpy(), seed17_cache.p_hist.to_numpy()) - reproduction["HIST"]["cache_auroc"])
    reproduction_pass = all(item["abs_diff"] < 1e-12 for item in reproduction.values()) and status_sums == [939819, 939819, 939819, 939819] and fast_check < 1e-10
    dump("execution_progress.json", {"stage": "seed17_reproduction", "reproduction": reproduction, "chronorep_status_sums": status_sums, "fast_auroc_abs_diff": fast_check, "pass": reproduction_pass})
    if not reproduction_pass:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "reproduction": reproduction, "chronorep_status_sums": status_sums, "fast_auroc_abs_diff": fast_check})
        write_checksums()
        raise SystemExit(12)
    _frame, first_seen = chronorep.load_order("development")
    if _frame["user_id"].nunique() != 31753 or len(_frame) != 4286981:
        raise SystemExit("development population mismatch")
    del _frame
    selection = chronorep.replay_selection(first_seen)
    if selection["selection_sha256"] != SELECTION_SHA or not selection["matches_tempfor"]:
        dump("training_cohort_and_selection.json", {"status": "MISMATCH", "sha256": selection["selection_sha256"]})
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_SOURCE"})
        write_checksums()
        raise SystemExit(3)
    training_rows = load_training_rows(selection["selected_users"])
    epochs = matchedinfo.epoch_targets(training_rows, selection["cohorts"])
    counts = [len(item["targets"]) for item in epochs]
    if counts != [90950, 90942] or any(item["duplicates"] for item in epochs):
        dump("training_cohort_and_selection.json", {"counts": counts, "duplicates": [item["duplicates"] for item in epochs]})
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_SOURCE"})
        write_checksums()
        raise SystemExit(4)
    dump("training_cohort_and_selection.json", {
        "design": "FIXED_COHORT_INITIALIZATION_AND_OPTIMIZATION_SEED_ROBUSTNESS",
        "selection_sha256": selection["selection_sha256"],
        "epoch_sizes": selection["epoch_sizes"], "union": selection["union"], "intersection": selection["intersection"],
        "valid_examples_by_epoch": counts, "duplicate_targets_by_epoch": [item["duplicates"] for item in epochs],
        "limitation": "Seeds change initialization and optimization only. They do not resample the training users.",
    })
    unique_targets = []
    seen_targets = set()
    for epoch in epochs:
        for item in epoch["targets"]:
            if item[0]["source_row"] not in seen_targets:
                seen_targets.add(item[0]["source_row"])
                unique_targets.append(item)
    qa = run_qa(unique_targets, skill2i, atype2i, marker, outcome0, outcome1)
    dump("input_contract_validation.json", qa)
    if qa["cases"] < 250 or qa["violations"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_SOURCE", "qa_cases": qa["cases"], "violations": qa["violations"]})
        write_checksums()
        raise SystemExit(5)
    print("qa_pass", qa["cases"], flush=True)
    init_hashes = {}
    for seed in SEEDS:
        hashes_now = []
        for _contract in CONTRACTS:
            hashes_now.append(state_sha(init_model(seed, skill2i, atype2i)))
        if len(set(hashes_now)) != 1:
            dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "seed": seed, "initial_hashes": hashes_now})
            write_checksums()
            raise SystemExit(6)
        init_hashes[str(seed)] = hashes_now[0]
        print(f"init_match seed {seed} {hashes_now[0]}", flush=True)
    logs = []
    for seed in SEEDS:
        for contract in CONTRACTS:
            logs.append(train_model(seed, contract, epochs, skill2i, atype2i, marker, outcome0, outcome1, init_hashes[str(seed)]))
            dump("training_logs.json", {"models": logs, "initial_hashes": init_hashes})
            dump("execution_progress.json", {"stage": "training", "completed": len(logs)})
    if any(item["parameter_count"] != 33857 or item["examples_by_epoch"] != [90950, 90942] for item in logs):
        dump("numerical_invariants.json", {"status": "BLOCKED_TRAINING_OR_EXECUTION", "logs": logs})
        write_checksums()
        raise SystemExit(7)
    print("training_complete", flush=True)
    del training_rows, epochs
    grouped = matchedinfo.confirmatory_rows()
    valid = sum(row["valid"] for rows in grouped.values() for row in rows)
    if valid != 939819 or len(grouped) != 6766:
        raise SystemExit(f"confirmatory population {valid} {len(grouped)}")
    matched_rows = load_seed17_metrics()
    calibration = pd.read_csv(MATCHEDINFO_DIR / "predictive_calibration_bins.csv")
    calibration["seed"] = 17
    calibration["contract"] = calibration["model"].replace({"ACT": "ACT_MATCHED"})
    calibration_out = calibration.rename(columns={"model": "source_model"}).to_dict(orient="records")
    score_violations = {}
    for seed in SEEDS:
        models = {}
        for contract in MATCHED:
            model = ActionTokenGRU(len(skill2i) - 1, len(atype2i) - 1)
            state = torch.load(OUT / checkpoint_name(contract, seed), map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"])
            model.eval()
            models[contract] = model
        scored = score_matched(models, grouped, skill2i, atype2i, marker, outcome0, outcome1)
        score_violations[str(seed)] = scored["violations"]
        if scored["violations"] or len(scored["y"]) != 939819:
            dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "score_violations": score_violations})
            write_checksums()
            raise SystemExit(8)
        for contract in MATCHED:
            row = metric_row(seed, contract, scored["y"], scored["scores"][contract])
            row["n_learners"] = 6766
            row["source"] = "seed_robustness_evaluation"
            matched_rows.append(row)
            calibration_out.extend(calibration_rows(seed, contract, scored["y"], scored["scores"][contract]))
            save_cache(f"{contract.lower()}_seed{seed}", scored, contract)
        np.savez_compressed(CACHE / f"matched_seed{seed}_bootstrap.npz", y=scored["y"], user_index=scored["user_index"], **{contract: scored["scores"][contract] for contract in MATCHED})
        print(f"matched_scored seed {seed}", flush=True)
    pd.DataFrame(matched_rows).to_csv(OUT / "matched_predictive_metrics_by_seed.csv", index=False)
    contrast_rows = contrasts_from(matched_rows)
    pd.DataFrame(contrast_rows).to_csv(OUT / "matched_predictive_contrasts_by_seed.csv", index=False)
    pd.DataFrame(seed_summary(contrast_rows)).to_csv(OUT / "seed_variability_summary.csv", index=False)
    pd.DataFrame(calibration_out).to_csv(OUT / "matched_calibration_bins_by_seed.csv", index=False)
    print("matched_tables_written", flush=True)
    geometry = chronorep.build_development(selection["selected_users"])
    plans = chronorep.plan_checkpoints(grouped, geometry["bundles"])
    eq = None
    frontiers, falses, transitions, thresholds, natives, matched_caps = [], [], [], [], [], []
    chronorep_predictive = []
    chronorep_pred = pd.read_csv(CHRONOREP_DIR / "predictive_diagnostics.csv").iloc[0].to_dict()
    chronorep_predictive.append({"seed": 17, "contract": "ACT_PRIMARY", "n": int(chronorep_pred["n"]), "auroc": chronorep_pred["roc_auc"], "log_loss": chronorep_pred["log_loss"], "brier": chronorep_pred["brier"], "source": "chronorep_predictive_diagnostics.csv"})
    pfa_reference = None
    for seed in SEEDS:
        pilot = load_pilot(OUT / checkpoint_name("ACT_PRIMARY", seed), skill2i, atype2i)
        if eq is None:
            eq = equivalence(pilot, grouped)
            dump("scoring_equivalence_checks.json", eq)
            if eq["action_label_mismatches"] or eq["max_abs_diff"] > 1e-5:
                dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "equivalence": eq})
                write_checksums()
                raise SystemExit(9)
        print(f"certificate_eval seed {seed}", flush=True)
        evaluated = chronorep.evaluate(pilot, grouped, geometry["bundles"], plans["plans"])
        if evaluated["events_seen"] != 939819 or evaluated["future_violations"] or evaluated["nested_violations"]:
            dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "seed": seed, "events": evaluated["events_seen"], "future": evaluated["future_violations"], "nested": evaluated["nested_violations"]})
            write_checksums()
            raise SystemExit(10)
        pfa_stable = int(evaluated["status"][:, 0, 0, chronorep.TAUS.index(0.8), 1].sum())
        if pfa_stable != pfa_s0_stable_stored or (pfa_reference is not None and pfa_stable != pfa_reference):
            dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "reason": "PFA S0 stable count diverged from the stored chronological aggregate", "stored": pfa_s0_stable_stored, "observed": pfa_stable, "previous": pfa_reference})
            write_checksums()
            raise SystemExit(11)
        pfa_reference = pfa_stable
        frontier, false_rows, transition_rows, threshold_rows, native, matched = certificate_tables(evaluated, seed)
        frontiers.extend(frontier)
        falses.extend(false_rows)
        transitions.extend(transition_rows)
        thresholds.extend(threshold_rows)
        natives.extend(native)
        matched_caps.extend(matched)
        y = np.asarray(evaluated["pred_y"])
        p = np.asarray(evaluated["pred_gru"])
        base = chronorep.binary_metrics(y, p)
        chronorep_predictive.append({"seed": seed, "contract": "ACT_PRIMARY", "n": int(len(y)), "auroc": base["roc_auc"], "log_loss": base["log_loss"], "brier": base["brier"], "accuracy_at_0.5": float(np.mean((p >= 0.5) == y)), "mean_prediction": float(p.mean()), "nonfinite": int((~np.isfinite(p)).sum()), "source": "seed_robustness_evaluation"})
        chronorep_keys = []
        for uid in sorted(grouped):
            by_start, _by_end = chronorep.order_valid(grouped[uid])
            for row in by_start:
                chronorep_keys.append(hashlib.blake2s(f"WSDM203-G9|{row['source_row']}".encode(), digest_size=16).hexdigest())
        if len(chronorep_keys) != len(y):
            raise SystemExit("ACT_PRIMARY score order length mismatch")
        save_cache(f"act_primary_seed{seed}", {"keys": chronorep_keys, "user_index": evaluated["pred_user"], "y": y, "scores": {"ACT_PRIMARY": p}}, "ACT_PRIMARY")
        np.savez_compressed(CACHE / f"act_primary_seed{seed}_user_status.npz", status=evaluated["status"][:, 1], user_count=len(evaluated["users"]))
        print(f"certificate_done seed {seed} auroc {base['roc_auc']}", flush=True)
    stored_frontier, stored_false, stored_trans, stored_native, stored_matched, _stored_risk = seed17_certificates()
    pd.DataFrame(chronorep_predictive).to_csv(OUT / "act_primary_predictive_metrics_by_seed.csv", index=False)
    pd.concat([stored_frontier, pd.DataFrame(frontiers)], ignore_index=True).to_csv(OUT / "act_primary_event_certificate_by_seed.csv", index=False)
    pd.concat([stored_false, pd.DataFrame(falses)], ignore_index=True).to_csv(OUT / "act_primary_false_stable_by_seed.csv", index=False)
    pd.concat([stored_trans, pd.DataFrame(transitions)], ignore_index=True).to_csv(OUT / "act_primary_s0_s1_transitions_by_seed.csv", index=False)
    stored_thr = pd.read_csv(CHRONOREP_DIR / "threshold_sensitivity.csv")
    stored_thr = stored_thr[stored_thr.model == "ActionTokenGRU"].copy()
    stored_thr["seed"] = 17
    stored_thr["source"] = "chronorep_threshold_sensitivity.csv"
    pd.concat([stored_thr, pd.DataFrame(thresholds)], ignore_index=True).to_csv(OUT / "act_primary_threshold_sensitivity_by_seed.csv", index=False)
    pd.concat([stored_native, pd.DataFrame(natives)], ignore_index=True).to_csv(OUT / "act_primary_native_capacity_by_seed.csv", index=False)
    pd.concat([stored_matched, pd.DataFrame(matched_caps)], ignore_index=True).to_csv(OUT / "act_primary_matched_capacity_by_seed.csv", index=False)
    print("certificate_tables_written", flush=True)
    for row in falses:
        if row["n_FS"] > row["n_U"] or row["n_U"] > row["n_S"]:
            dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "false_stable": row})
            write_checksums()
            raise SystemExit(15)
        if row["n_U"] > 0 and abs((row["n_FS"] / row["n_S"]) - (row["n_U"] / row["n_S"]) * (row["n_FS"] / row["n_U"])) > 1e-12:
            dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "false_stable_identity": row})
            write_checksums()
            raise SystemExit(16)
    variability = pd.read_csv(OUT / "seed_variability_summary.csv")
    nc0_rate = float(stored_frontier.loc[stored_frontier.schema == "S0", "event_nc_over_valid"].iloc[0])
    nc1_rate = float(stored_frontier.loc[stored_frontier.schema == "S1", "event_nc_over_valid"].iloc[0])
    new_nc_rates = [float(item["n"]) / 939819 for item in transitions if item["to_label"] == "event_nc_s1_minus_s0" and abs(float(item["tau"]) - 0.8) < 1e-9]
    extra_values = {
        "ACT_PRIMARY_auroc": [float(item["auroc"]) for item in chronorep_predictive],
        "ACT_PRIMARY_event_nc_s1_minus_s0": [nc1_rate - nc0_rate, *new_nc_rates],
    }
    extra_rows = []
    for contrast, values in extra_values.items():
        values = np.asarray(values, dtype=float)
        extra_rows.append({
            "contrast": contrast, "metric": "value", "n_seeds": int(len(values)),
            "mean": float(values.mean()), "minimum": float(values.min()), "maximum": float(values.max()),
            "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
            "interval_type": "three-seed descriptive range, not a 95 percent confidence interval",
        })
    pd.concat([variability, pd.DataFrame(extra_rows)], ignore_index=True).to_csv(OUT / "seed_variability_summary.csv", index=False)
    seed17_cache = pd.read_parquet(MATCHEDINFO_DIR / "restricted_event_score_cache.parquet")
    packs = {17: {"y": seed17_cache.correct.to_numpy(), "user_index": seed17_cache.user_index.to_numpy(), "HIST": seed17_cache.p_hist.to_numpy(), "LEN": seed17_cache.p_len.to_numpy(), "ACT_MATCHED": seed17_cache.p_act.to_numpy()}}
    for seed in SEEDS:
        stored = np.load(CACHE / f"matched_seed{seed}_bootstrap.npz")
        packs[seed] = {contract: stored[contract] for contract in MATCHED}
        packs[seed]["y"] = stored["y"]
        packs[seed]["user_index"] = stored["user_index"]
        if not np.array_equal(packs[17]["y"], packs[seed]["y"]) or not np.array_equal(packs[17]["user_index"], packs[seed]["user_index"]):
            dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "reason": "confirmatory event alignment differs across seeds", "seed": seed})
            write_checksums()
            raise SystemExit(13)
    if list(seed17_cache.event_key) != list(pd.read_parquet(CACHE / "hist_seed29.parquet").event_key):
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "reason": "matched-information and seed-robustness event-key order differ"})
        write_checksums()
        raise SystemExit(14)
    boot_rows = []
    nc0 = float(stored_frontier.loc[stored_frontier.schema == "S0", "event_nc_over_valid"].iloc[0])
    nc1 = float(stored_frontier.loc[stored_frontier.schema == "S1", "event_nc_over_valid"].iloc[0])
    boot_rows.append({
        "analysis": "act_primary_event_nc_s1_minus_s0", "seed": 17, "contrast": "S1_minus_S0",
        "point": nc1 - nc0, "bootstrap_mean": None, "ci95_low": None, "ci95_high": None,
        "replicates": 0, "replicate_failures": None, "seed_rng": 20310,
        "estimand": "point read from stored chronological aggregates; learner interval unavailable because the chronological replication did not retain per-user certificate counts and a second seed-17 evaluation was not run",
    })
    rng = np.random.default_rng(20310)
    user_index = packs[17]["user_index"]
    n_users = 6766
    draws = rng.integers(0, n_users, size=(2000, n_users))
    # Per-seed AUROC differences and the mean across the three fixed checkpoints.
    auc = {seed: {contrast: np.empty(2000) for contrast in ("LEN_minus_HIST", "ACT_MATCHED_minus_LEN", "ACT_MATCHED_minus_HIST")} for seed in (17, 29, 41)}
    pieces = {}
    labels = None
    per_n = None
    for seed, pack in packs.items():
        order_users = pack["user_index"]
        buckets = [[] for _ in range(n_users)]
        for index, user in enumerate(order_users.tolist()):
            buckets[user].append(index)
        indexes = [np.asarray(bucket, dtype=np.int32) for bucket in buckets]
        if per_n is None:
            per_n = np.asarray([len(bucket) for bucket in indexes], dtype=np.int32)
            labels = [pack["y"][idx].astype(float) if len(idx) else np.zeros(0) for idx in indexes]
        pieces[seed] = {contract: [np.clip(pack[contract][idx], 1e-12, 1 - 1e-12) if len(idx) else np.zeros(0) for idx in indexes] for contract in MATCHED}
    y_buffer = np.empty(int(per_n.max()) * n_users, dtype=np.float64)
    p_buffer = np.empty_like(y_buffer)
    for rep, draw in enumerate(draws):
        pos = 0
        for user_i in draw:
            block = labels[user_i]
            n = len(block)
            if n:
                y_buffer[pos:pos + n] = block
                pos += n
        yy = y_buffer[:pos]
        for seed in (17, 29, 41):
            arm_auc = {}
            for contract in MATCHED:
                cursor = 0
                for user_i in draw:
                    block = pieces[seed][contract][user_i]
                    n = len(block)
                    if n:
                        p_buffer[cursor:cursor + n] = block
                        cursor += n
                arm_auc[contract] = fast_auroc(yy, p_buffer[:cursor])
            auc[seed]["LEN_minus_HIST"][rep] = arm_auc["LEN"] - arm_auc["HIST"]
            auc[seed]["ACT_MATCHED_minus_LEN"][rep] = arm_auc["ACT_MATCHED"] - arm_auc["LEN"]
            auc[seed]["ACT_MATCHED_minus_HIST"][rep] = arm_auc["ACT_MATCHED"] - arm_auc["HIST"]
        if rep and rep % 100 == 0:
            print(f"bootstrap {rep}", flush=True)
    point = {(row["seed"], row["contrast"]): row["auroc_difference"] for row in contrast_rows}
    for seed in (17, 29, 41):
        for contrast, values in auc[seed].items():
            boot_rows.append({"analysis": "per_seed_auroc_difference", "seed": seed, "contrast": contrast, "point": point[(seed, contrast)], "bootstrap_mean": float(np.mean(values)), "ci95_low": float(np.percentile(values, 2.5)), "ci95_high": float(np.percentile(values, 97.5)), "replicates": 2000, "replicate_failures": 0, "seed_rng": 20310, "estimand": "learner resampling conditional on this trained checkpoint"})
    for contrast in ("LEN_minus_HIST", "ACT_MATCHED_minus_LEN", "ACT_MATCHED_minus_HIST"):
        values = (auc[17][contrast] + auc[29][contrast] + auc[41][contrast]) / 3
        boot_rows.append({"analysis": "three_seed_mean_auroc_difference", "seed": "17_29_41", "contrast": contrast, "point": float(np.mean([point[(seed, contrast)] for seed in (17, 29, 41)])), "bootstrap_mean": float(np.mean(values)), "ci95_low": float(np.percentile(values, 2.5)), "ci95_high": float(np.percentile(values, 97.5)), "replicates": 2000, "replicate_failures": 0, "seed_rng": 20310, "estimand": "learner resampling conditional on the three trained checkpoints; not uncertainty over new initializations"})
    # ACT_PRIMARY event NC for seeds 29 and 41. Seed 17 has stored aggregates only.
    for seed in SEEDS:
        status = np.load(CACHE / f"act_primary_seed{seed}_user_status.npz")["status"]
        primary = chronorep.TAUS.index(0.8)
        unsup0 = status[:, 0, primary, 0]
        set0 = status[:, 0, primary, 2]
        den0 = status[:, 0, primary, :].sum(axis=1)
        unsup1 = status[:, 1, primary, 0]
        set1 = status[:, 1, primary, 2]
        den1 = status[:, 1, primary, :].sum(axis=1)
        point_nc = ((unsup1 + set1).sum() / den1.sum()) - ((unsup0 + set0).sum() / den0.sum())
        reps = []
        for draw in draws:
            d0 = den0[draw].sum()
            d1 = den1[draw].sum()
            reps.append(((unsup1[draw] + set1[draw]).sum() / d1) - ((unsup0[draw] + set0[draw]).sum() / d0))
        values = np.asarray(reps)
        boot_rows.append({"analysis": "act_primary_event_nc_s1_minus_s0", "seed": seed, "contrast": "S1_minus_S0", "point": float(point_nc), "bootstrap_mean": float(np.mean(values)), "ci95_low": float(np.percentile(values, 2.5)), "ci95_high": float(np.percentile(values, 97.5)), "replicates": 2000, "replicate_failures": 0, "seed_rng": 20310, "estimand": "learner resampling conditional on this ACT_PRIMARY checkpoint"})
    pd.DataFrame(boot_rows).to_csv(OUT / "learner_bootstrap_intervals.csv", index=False)
    end_hashes = {name: chronorep.sha256_file(path) for name, path in paths.items()}
    status = "COMPLETE"
    invariants = {
        "status": status,
        "qa_cases": qa["cases"],
        "qa_violations": qa["violations"],
        "score_violations": score_violations,
        "equivalence": eq,
        "pfa_s0_stable_reference_across_gru_seeds": pfa_reference,
        "frozen_checkpoints_unchanged": end_hashes == hashes,
        "models_trained": [f"{item['contract']}_seed{item['seed']}" for item in logs],
        "design": "FIXED_COHORT_INITIALIZATION_AND_OPTIMIZATION_SEED_ROBUSTNESS",
    }
    dump("numerical_invariants.json", invariants)
    dump("execution_progress.json", {"status": status, "models": len(logs)})
    dump("experiment_manifest.json", {"experiment": "pipeline", "script": "scripts/assistments/seed_robustness.py", "script_sha256": chronorep.sha256_file(Path(__file__).resolve()), "status": status, "design": invariants["design"]})
    write_checksums()
    print(status, flush=True)


if __name__ == "__main__":
    main()
