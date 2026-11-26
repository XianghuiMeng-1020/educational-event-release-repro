#!/usr/bin/env python3
"""Matched-estimand transparent and neural controls.

Track A compares CanonicalPFA40 with CountEncodedGRU on (skill, S, F) only.
Track B compares ActionTokenLogit with the frozen chronological GRU on the
original token interface. The chronological GRU is not retrained.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
import importlib.util
SPEC = importlib.util.spec_from_file_location(
    "chronorep", ROOT / "scripts/assistments/chronological_replication.py"
)
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)
from r7_core import MAX_HIST_TOKENS, MAX_TRAIN_EVENTS_PER_USER, NEURAL_SEED, ActionTokenGRU  # noqa: E402

OUT = ROOT / "results/assistments/matched_estimand_controls"
CHRONOREP_CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
SELECTION_SHA = "cbe4fd9e0075c78ae5997892153d37920497d14c64a21af3ffbf3594e5c907e2"
GRU_SHA = "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
VALID = 939819
SCHEMAS = ("S0", "S1", "S2", "S3")
TAUS = (0.70, 0.80, 0.90)
PRIMARY = 0.80
EDGES = (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3"))
N_SKILL = 265
N_TOK = 11


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def counts_of(prefix: list, skill: str) -> tuple[int, int]:
    success = failure = 0
    for event in prefix:
        if event.skill_id != skill:
            continue
        if int(event.correct) == 1:
            success += 1
        else:
            failure += 1
    return success, failure


def count_tokens(skill_i: int, success: int, failure: int, outcome0: int, outcome1: int, start_id: int) -> tuple[list[int], list[int]]:
    skills = [skill_i] * (failure + success + 1)
    atypes = [outcome0] * failure + [outcome1] * success + [start_id]
    return skills, atypes


def load_split(split: str, keep: set[str] | None) -> dict:
    import pyarrow.parquet as pq
    grouped = defaultdict(list)
    seen = 0
    parquet = pq.ParquetFile(chronorep.RECON / f"{split}.parquet")
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            uid = str(row["user_id"])
            if keep is not None and uid not in keep:
                continue
            grouped[uid].append(chronorep.compact_row(row, int(start), int(end)))
        seen += len(rows)
        if seen % 500000 == 0:
            print(f"load_{split} {seen}", flush=True)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["source_row"])
    return grouped


def collect_examples(rows_by_user: dict, cohorts: list[list[str]], pilot) -> list[list[dict]]:
    epochs = []
    for cohort in cohorts:
        examples = []
        for uid in cohort:
            rows = rows_by_user[uid]
            window = sorted(rows, key=lambda item: item["source_row"])[-MAX_TRAIN_EVENTS_PER_USER:]
            _by_start, by_end = chronorep.order_valid(rows)
            for row in window:
                if not row["valid"]:
                    continue
                prefix = chronorep.prefix_events(by_end, row["start"])
                current = chronorep.as_event(row)
                tokens = pilot._tokens(prefix, current, False)
                if not tokens[0]:
                    continue
                success, failure = counts_of(prefix, current.skill_id)
                examples.append({
                    "user": uid,
                    "source_row": row["source_row"],
                    "skill": current.skill_id,
                    "skill_i": int(pilot.skill2i.get(current.skill_id, 0)),
                    "S": success,
                    "F": failure,
                    "y": int(current.correct),
                    "tokens": tokens,
                    "prefix_n": len(prefix),
                })
        epochs.append(examples)
    return epochs


def qa_examples(epochs: list[list[dict]], rows_by_user: dict, outcome0: int, outcome1: int, start_id: int) -> dict:
    pool = []
    for epoch_i, examples in enumerate(epochs):
        for example in examples:
            rank = hashlib.sha256(f"WSDM203-G13-A|{epoch_i}|{example['source_row']}".encode()).hexdigest()
            pool.append((rank, epoch_i, example))
    pool.sort()
    violations = 0
    checked = 0
    for _rank, epoch_i, example in pool[:250]:
        checked += 1
        uid = example["user"]
        target = next(row for row in rows_by_user[uid] if row["source_row"] == example["source_row"])
        _by_start, by_end = chronorep.order_valid(rows_by_user[uid])
        cut = chronorep.history_cut(by_end, target["start"])
        history = by_end[max(0, cut - 40):cut]
        prefix = [chronorep.as_event(item) for item in history]
        success, failure = counts_of(prefix, target["skill_id"])
        if any(item["end"] >= target["start"] for item in history):
            violations += 1
        if (success, failure, target["skill_id"], int(target["correct"])) != (example["S"], example["F"], example["skill"], example["y"]):
            violations += 1
        encoded = count_tokens(example["skill_i"], success, failure, outcome0, outcome1, start_id)
        if len(encoded[0]) != success + failure + 1 or encoded[1][-1] != start_id:
            violations += 1
        if any(tok not in {outcome0, outcome1, start_id} for tok in encoded[1]):
            violations += 1
        if success + failure > 40 or len(encoded[0]) > MAX_HIST_TOKENS:
            violations += 1
        shuffled = prefix.copy()
        random.Random(example["source_row"]).shuffle(shuffled)
        other = counts_of(shuffled, target["skill_id"])
        if other != (success, failure) or count_tokens(example["skill_i"], other[0], other[1], outcome0, outcome1, start_id) != encoded:
            violations += 1
        if epoch_i not in (0, 1):
            violations += 1
    return {"checked": checked, "violations": violations}


def design_matrix(examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = len(examples)
    x = np.zeros((n, N_SKILL * 3), dtype=np.float64)
    y = np.empty(n, dtype=np.int8)
    skill = np.fromiter((item["skill_i"] for item in examples), dtype=np.int32, count=n)
    success = np.fromiter((item["S"] for item in examples), dtype=np.float64, count=n)
    failure = np.fromiter((item["F"] for item in examples), dtype=np.float64, count=n)
    rows = np.arange(n)
    x[rows, skill * 3] = 1.0
    x[rows, skill * 3 + 1] = success
    x[rows, skill * 3 + 2] = failure
    y[:] = [item["y"] for item in examples]
    return x, y


def fit_pfa(examples: list[dict]) -> LogisticRegression:
    x, y = design_matrix(examples)
    model = LogisticRegression(
        C=1.0, l1_ratio=0.0, solver="liblinear", tol=1e-6, max_iter=500,
        fit_intercept=False, random_state=17,
    )
    model.fit(x, y)
    return model


def pfa_probability(skill_i: np.ndarray, success: np.ndarray, failure: np.ndarray, coef: np.ndarray) -> np.ndarray:
    beta = coef[skill_i * 3]
    gamma_s = coef[skill_i * 3 + 1]
    gamma_f = coef[skill_i * 3 + 2]
    return sigmoid(beta + gamma_s * success + gamma_f * failure)


def train_count_gru(epochs: list[list[dict]], outcome0: int, outcome1: int, start_id: int) -> tuple[ActionTokenGRU, dict]:
    random.seed(NEURAL_SEED)
    np.random.seed(NEURAL_SEED)
    torch.manual_seed(NEURAL_SEED)
    model = ActionTokenGRU(N_SKILL - 1, N_TOK - 3)
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    steps = []
    nonfinite = 0
    model.train()
    for epoch_i, examples in enumerate(epochs, start=1):
        total = 0.0
        seen = 0
        for item in examples:
            skills, atypes = count_tokens(item["skill_i"], item["S"], item["F"], outcome0, outcome1, start_id)
            pred, _ = model.forward_from_state(
                torch.tensor([skills], dtype=torch.long, device=device),
                torch.tensor([atypes], dtype=torch.long, device=device),
                None,
            )
            if not torch.isfinite(pred).all():
                nonfinite += 1
                continue
            y = torch.tensor([float(item["y"])], dtype=torch.float32, device=device)
            loss = torch.nn.functional.binary_cross_entropy(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            seen += 1
            if seen % 20000 == 0:
                print(f"count_gru_epoch {epoch_i} examples {seen}", flush=True)
        losses.append(total / seen if seen else None)
        steps.append(seen)
        print(f"count_gru_epoch {epoch_i} loss {losses[-1]} examples {seen}", flush=True)
    model.eval()
    path = OUT / "track_a_count_gru_seed17.pt"
    torch.save({
        "label": "COUNT_ENCODED_GRU_SEED17",
        "warm_start": False,
        "seed": 17,
        "epochs": 2,
        "batch_size": 64,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }, path)
    reloaded = ActionTokenGRU(N_SKILL - 1, N_TOK - 3)
    reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
    reloaded.eval()
    sample = [count_tokens(3, 2, 1, outcome0, outcome1, start_id)]
    with torch.inference_mode():
        left = float(chronorep.score_forward(model, sample, device)[0])
        right = float(chronorep.score_forward(reloaded, sample, device)[0])
    return model, {
        "loss_by_epoch": losses,
        "optimizer_steps_by_epoch": steps,
        "nonfinite_examples_skipped": nonfinite,
        "reload_abs_diff": abs(left - right),
        "checkpoint": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "checkpoint_sha256": chronorep.sha256_file(path),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "updates": "one Adam step per occurrence",
        "order": "epoch cohort order, no shuffle",
    }


def sparse_from_tokens(sequences: list[tuple[list[int], list[int]]]) -> sparse.csr_matrix:
    indptr = [0]
    indices = []
    stride_s = N_TOK
    stride_p = N_SKILL * N_TOK
    for skills, atypes in sequences:
        for pos, (skill_i, tok) in enumerate(zip(skills, atypes)):
            indices.append(pos * stride_p + skill_i * stride_s + tok)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float64)
    return sparse.csr_matrix((data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)), shape=(len(sequences), 64 * stride_p))


def logit_probabilities(sequences: list[tuple[list[int], list[int]]], weight: np.ndarray, intercept: float) -> np.ndarray:
    table = weight.reshape(64, N_SKILL, N_TOK)
    out = np.empty(len(sequences), dtype=np.float64)
    for i, (skills, atypes) in enumerate(sequences):
        total = intercept
        for pos, (skill_i, tok) in enumerate(zip(skills, atypes)):
            total += table[pos, skill_i, tok]
        out[i] = total
    return sigmoid(out)


def full_metrics(name: str, y: np.ndarray, p: np.ndarray, learners: int) -> dict:
    finite = np.isfinite(p)
    base = chronorep.binary_metrics(y[finite], p[finite])
    accuracy = float(np.mean((p[finite] >= 0.5) == (y[finite] == 1))) if finite.any() else None
    qs = np.quantile(p[finite], [0.1, 0.25, 0.5, 0.75, 0.9]) if finite.any() else [None] * 5
    return {
        "model": name, "n": int(finite.sum()), "learners": learners, "base_rate": base["positive_rate"],
        "auroc": base["roc_auc"], "log_loss": base["log_loss"], "brier": base["brier"],
        "accuracy_at_0.5": accuracy, "mean_prediction": float(p[finite].mean()) if finite.any() else None,
        "q10": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]), "q75": float(qs[3]), "q90": float(qs[4]),
        "nonfinite": int((~finite).sum()),
    }


def calibration_rows(model: str, y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for bin_id in range(10):
        left, right = bin_id / 10, (bin_id + 1) / 10
        chosen = (p >= left) & (p < right if bin_id < 9 else p <= right) & np.isfinite(p)
        rows.append({
            "model": model, "bin": bin_id, "n": int(chosen.sum()),
            "mean_prediction": float(p[chosen].mean()) if chosen.any() else None,
            "observed_rate": float(y[chosen].mean()) if chosen.any() else None,
        })
    return rows


def classify(lo: float, hi: float, realized: float, tau: float) -> tuple[int, str | None, bool]:
    if lo >= tau:
        action = "progress"
    elif hi < tau:
        action = "review"
    else:
        return 2, None, True
    realized_action = "progress" if realized >= tau else "review"
    return 1, action, realized_action == action


def weighted_metrics(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> dict:
    mask = weight > 0
    if not np.any(mask):
        return {"roc_auc": None, "log_loss": None, "brier": None}
    y = y[mask].astype(np.float64, copy=False)
    p = p[mask].astype(np.float64, copy=False)
    weight = weight[mask].astype(np.float64, copy=False)
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    total = float(weight.sum())
    log_loss = float(-np.sum(weight * (y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))) / total)
    brier = float(np.sum(weight * (clipped - y) ** 2) / total)
    order = np.argsort(p, kind="mergesort")
    scores = p[order]
    labels = y[order]
    weights = weight[order]
    change = np.empty(len(scores), dtype=bool)
    change[0] = True
    change[1:] = scores[1:] != scores[:-1]
    group = np.cumsum(change) - 1
    group_weight = np.bincount(group, weights=weights)
    before = np.zeros(len(group_weight), dtype=np.float64)
    before[1:] = np.cumsum(group_weight)[:-1]
    average_rank = before[group] + (group_weight[group] + 1.0) / 2.0
    positives = float(np.sum(weights * labels))
    negatives = total - positives
    if positives == 0 or negatives == 0:
        auc = None
    else:
        auc = (float(np.sum(average_rank * weights * labels)) - positives * (positives + 1.0) / 2.0) / (positives * negatives)
    return {"roc_auc": auc, "log_loss": log_loss, "brier": brier}


def bootstrap_paired(y: np.ndarray, user: np.ndarray, named: dict[str, np.ndarray], n_users: int) -> list[dict]:
    rng = np.random.default_rng(20313)
    rows = []
    draws = rng.integers(0, n_users, size=(2000, n_users))
    names = list(named)
    stored = {name: [] for name in ("A_PFA_auc", "A_GRU_auc", "A_auc_diff", "A_log_loss_diff", "A_brier_diff", "B_LOGIT_auc", "B_GRU_auc", "B_auc_diff", "B_log_loss_diff", "B_brier_diff")}
    for replicate, draw in enumerate(draws):
        weight = np.bincount(draw, minlength=n_users)[user]
        a_pfa = weighted_metrics(y, named["CanonicalPFA40"], weight)
        a_gru = weighted_metrics(y, named["CountEncodedGRU"], weight)
        b_logit = weighted_metrics(y, named["ActionTokenLogit"], weight)
        b_gru = weighted_metrics(y, named["ActionTokenGRU_CHRONOREP"], weight)
        stored["A_PFA_auc"].append(a_pfa["roc_auc"])
        stored["A_GRU_auc"].append(a_gru["roc_auc"])
        stored["A_auc_diff"].append(a_gru["roc_auc"] - a_pfa["roc_auc"])
        stored["A_log_loss_diff"].append(a_gru["log_loss"] - a_pfa["log_loss"])
        stored["A_brier_diff"].append(a_gru["brier"] - a_pfa["brier"])
        stored["B_LOGIT_auc"].append(b_logit["roc_auc"])
        stored["B_GRU_auc"].append(b_gru["roc_auc"])
        stored["B_auc_diff"].append(b_logit["roc_auc"] - b_gru["roc_auc"])
        stored["B_log_loss_diff"].append(b_logit["log_loss"] - b_gru["log_loss"])
        stored["B_brier_diff"].append(b_logit["brier"] - b_gru["brier"])
        if replicate % 200 == 0:
            print(f"bootstrap {replicate}", flush=True)
    points = {
        "A_PFA_auc": chronorep.binary_metrics(y, named["CanonicalPFA40"])["roc_auc"],
        "A_GRU_auc": chronorep.binary_metrics(y, named["CountEncodedGRU"])["roc_auc"],
        "B_LOGIT_auc": chronorep.binary_metrics(y, named["ActionTokenLogit"])["roc_auc"],
        "B_GRU_auc": chronorep.binary_metrics(y, named["ActionTokenGRU_CHRONOREP"])["roc_auc"],
    }
    points["A_auc_diff"] = points["A_GRU_auc"] - points["A_PFA_auc"]
    points["A_log_loss_diff"] = chronorep.binary_metrics(y, named["CountEncodedGRU"])["log_loss"] - chronorep.binary_metrics(y, named["CanonicalPFA40"])["log_loss"]
    points["A_brier_diff"] = chronorep.binary_metrics(y, named["CountEncodedGRU"])["brier"] - chronorep.binary_metrics(y, named["CanonicalPFA40"])["brier"]
    points["B_auc_diff"] = points["B_LOGIT_auc"] - points["B_GRU_auc"]
    points["B_log_loss_diff"] = chronorep.binary_metrics(y, named["ActionTokenLogit"])["log_loss"] - chronorep.binary_metrics(y, named["ActionTokenGRU_CHRONOREP"])["log_loss"]
    points["B_brier_diff"] = chronorep.binary_metrics(y, named["ActionTokenLogit"])["brier"] - chronorep.binary_metrics(y, named["ActionTokenGRU_CHRONOREP"])["brier"]
    for name, values in stored.items():
        low, high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
        rows.append({
            "track": "A" if name.startswith("A_") else "B", "quantity": name, "point": points[name],
            "ci_low": float(low), "ci_high": float(high), "replicates": 2000, "seed": 20313,
            "resample": "paired confirmatory learners with replacement",
        })
    return rows


def try_reuse():
    if os.environ.get("ESTCTRL_REUSE_FIT") != "1":
        return None
    needed = [
        OUT / "track_a_canonical_pfa_parameters.json",
        OUT / "track_a_count_gru_seed17.pt",
        OUT / "track_a_training_log.json",
        OUT / "track_b_action_token_logit_parameters.npz",
        OUT / "track_b_training_convergence.json",
    ]
    if not all(path.exists() for path in needed):
        return None
    pfa_saved = json.loads(needed[0].read_text(encoding="utf-8"))
    conv = json.loads(needed[4].read_text(encoding="utf-8"))
    if not pfa_saved.get("converged") or not conv.get("converged"):
        return None
    coef = np.asarray(pfa_saved["coefficients"]["values"], dtype=np.float64)
    packed = np.load(needed[3])
    weight = np.asarray(packed["coef"], dtype=np.float64)
    intercept = float(np.asarray(packed["intercept"]).reshape(-1)[0])
    model = ActionTokenGRU(N_SKILL - 1, N_TOK - 3)
    model.load_state_dict(torch.load(needed[1], map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    gru_log = json.loads(needed[2].read_text(encoding="utf-8"))
    return coef, model, gru_log, weight, intercept, float(conv["fit_gap"]), int(pfa_saved["n_iter"]), int(conv["n_iter"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "private_scores").mkdir(exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    print("preflight", flush=True)
    pre = chronorep.preflight()
    pre["chronorep_checkpoint_sha256"] = chronorep.sha256_file(CHRONOREP_CKPT)
    pre["chronorep_checkpoint_match"] = pre["chronorep_checkpoint_sha256"] == GRU_SHA
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"])
    dump("source_and_checkpoint_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_ESTIMAND_CONTRACT"})
        raise SystemExit(2)
    checkpoint = torch.load(CHRONOREP_CKPT, map_location="cpu", weights_only=False)
    skill2i = checkpoint["skill2i"]
    atype2i = checkpoint["atype2i"]
    if len(skill2i) != N_SKILL or len(atype2i) != 9:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_ESTIMAND_CONTRACT", "vocab": [len(skill2i), len(atype2i)]})
        raise SystemExit(2)
    pilot = chronorep.NeuralPilot.__new__(chronorep.NeuralPilot)
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.device = torch.device("cpu")
    frozen = ActionTokenGRU(N_SKILL - 1, N_TOK - 3)
    frozen.load_state_dict(checkpoint["state_dict"])
    frozen.eval()
    pilot.model = frozen
    outcome0, outcome1 = frozen.outcome0, frozen.outcome1
    start_id = int(atype2i["start"])
    dump("two_track_estimand_contract.json", {
        "track_a_information": "(skill, successes on that skill among the last 40 completed events, failures on that skill)",
        "track_a_excludes": ["current actions", "current outcome", "historical action streams", "other skills", "original outcome order"],
        "track_a_pfa": {"name": "CanonicalPFA40", "formula": "logit = beta_skill + gamma_success_skill * S + gamma_failure_skill * F", "C": 1.0, "l2": True, "tol": 1e-6, "max_iter": 500, "solver": "liblinear", "fit_intercept": False, "solver_note": "Same sklearn L2 logistic objective. liblinear is the frozen solver because lbfgs overflowed and reached the 500-iteration limit on the raw counts before any confirmatory score was examined."},
        "track_a_gru": {"name": "CountEncodedGRU", "encoding": "F failure tokens, then S success tokens, then one start token, all carrying the target skill", "optimizer": "Adam", "lr": 0.001, "epochs": 2, "updates": "one Adam step per training occurrence, in cohort order, no shuffle", "seed": 17, "warm_start": False},
        "track_b_information": "exact chronological _tokens(prefix, current, include_current_outcome=False)",
        "track_b_logit": {"name": "ActionTokenLogit", "feature": "(token_position, skill_id, action_or_outcome_id)", "C": 1.0, "tol": 1e-4, "max_iter": 500, "solver": "liblinear", "seed": 17},
        "not_the_post_ingest_pfa": True,
        "frozen_before_confirmatory_scoring": True,
    })
    print("selection", flush=True)
    _frame, first_seen = chronorep.load_order("development")
    selection = chronorep.replay_selection(first_seen)
    if selection["selection_sha256"] != SELECTION_SHA or selection["union"] != 5715 or selection["intersection"] != 285:
        dump("training_population_audit.json", {"status": "MISMATCH", "selection_sha256": selection["selection_sha256"]})
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_ESTIMAND_CONTRACT"})
        raise SystemExit(3)
    print("load development targets", flush=True)
    selected_rows = load_split("development", selection["selected_users"])
    epochs = collect_examples(selected_rows, selection["cohorts"], pilot)
    counts = [len(epoch) for epoch in epochs]
    dump("training_population_audit.json", {
        "epoch_sizes": selection["epoch_sizes"], "union": selection["union"], "intersection": selection["intersection"],
        "selection_sha256": selection["selection_sha256"], "examples_by_epoch": counts,
        "matches_expected_examples": counts == [90950, 90942],
    })
    if counts != [90950, 90942]:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_ESTIMAND_CONTRACT", "examples": counts})
        raise SystemExit(3)
    example_qa = qa_examples(epochs, selected_rows, outcome0, outcome1, start_id)
    # Explicit permutation invariance on 50 sampled prefixes.
    perm_bad = 0
    rng = np.random.default_rng(13)
    checked_perm = 0
    for uid in list(selected_rows)[:80]:
        _by_start, by_end = chronorep.order_valid(selected_rows[uid])
        if len(by_end) < 5:
            continue
        target = by_end[len(by_end) // 2]
        prefix = chronorep.prefix_events(by_end, target["start"])
        if len(prefix) < 2:
            continue
        shuffled = prefix.copy()
        rng.shuffle(shuffled)
        left = counts_of(prefix, target["skill_id"])
        right = counts_of(shuffled, target["skill_id"])
        skill_i = int(skill2i.get(target["skill_id"], 0))
        if left != right or count_tokens(skill_i, left[0], left[1], outcome0, outcome1, start_id) != count_tokens(skill_i, right[0], right[1], outcome0, outcome1, start_id):
            perm_bad += 1
        checked_perm += 1
        if checked_perm >= 50:
            break
    dump("track_a_sufficient_statistics_qa.json", {**example_qa, "permutation_checks": checked_perm, "permutation_violations": perm_bad})
    if example_qa["violations"] or perm_bad or example_qa["checked"] < 250:
        dump("numerical_invariants.json", {"status": "BLOCKED_INPUT_OR_ESTIMAND_CONTRACT", "qa": example_qa, "perm_bad": perm_bad})
        raise SystemExit(4)
    stacked = epochs[0] + epochs[1]
    saved = try_reuse()
    if saved:
        coef, count_model, gru_log, weight, intercept, fit_gap, pfa_iterations, logit_iterations = saved
        print("reuse fitted models", flush=True)
    else:
        print("fit CanonicalPFA40", flush=True)
        pfa = fit_pfa(stacked)
        coef = pfa.coef_.ravel()
        dump("track_a_canonical_pfa_parameters.json", {
            "name": "CanonicalPFA40",
            "classical_unbounded_pfa": False,
            "adapted_fixed_window": True,
            "horizon_events": 40,
            "n_examples": len(stacked),
            "n_iter": int(np.max(pfa.n_iter_)),
            "converged": bool(np.max(pfa.n_iter_) < 500),
            "C": 1.0,
            "tol": 1e-6,
            "parameters": int(coef.size),
            "coefficients": {"layout": "[beta, gamma_success, gamma_failure] per skill id 0..264", "values": coef.tolist()},
        })
        print(f"pfa_iter {int(np.max(pfa.n_iter_))} finite {bool(np.isfinite(coef).all())}", flush=True)
        if int(np.max(pfa.n_iter_)) >= 500 or not np.isfinite(coef).all():
            dump("numerical_invariants.json", {"status": "BLOCKED_TRAINING_OR_EXECUTION", "pfa_iterations": int(np.max(pfa.n_iter_)), "pfa_finite": bool(np.isfinite(coef).all())})
            raise SystemExit(5)
        print("train CountEncodedGRU", flush=True)
        count_model, gru_log = train_count_gru(epochs, outcome0, outcome1, start_id)
        dump("track_a_training_log.json", gru_log)
        print("fit ActionTokenLogit", flush=True)
        token_seqs = [item["tokens"] for item in stacked]
        labels = np.asarray([item["y"] for item in stacked], dtype=np.int8)
        features = sparse_from_tokens(token_seqs)
        logit = LogisticRegression(C=1.0, l1_ratio=0.0, solver="liblinear", tol=1e-4, max_iter=500, fit_intercept=True, random_state=17)
        logit.fit(features, labels)
        weight = logit.coef_.ravel().astype(np.float64)
        intercept = float(logit.intercept_[0])
        sample_idx = np.linspace(0, len(token_seqs) - 1, 200, dtype=int)
        manual = logit_probabilities([token_seqs[i] for i in sample_idx], weight, intercept)
        library = logit.predict_proba(features[sample_idx])[:, 1]
        fit_gap = float(np.max(np.abs(manual - library)))
        np.savez_compressed(
            OUT / "track_b_action_token_logit_parameters.npz",
            coef=weight, intercept=np.asarray([intercept]), n_position=np.asarray([64]), n_skill=np.asarray([N_SKILL]), n_token=np.asarray([N_TOK]),
        )
        dump("track_b_token_feature_contract.json", {
            "feature": "(position, skill_id, token_id)",
            "positions": "left-aligned index in the retained chronological sequence",
            "n_features": int(weight.size),
            "intercept": True,
            "excluded": ["current correctness", "raw hint and attempt columns", "student id", "future events", "source row", "clock time"],
            "manual_versus_library_max_abs": fit_gap,
        })
        dump("track_b_training_convergence.json", {
            "solver": "liblinear", "C": 1.0, "tol": 1e-4, "max_iter": 500,
            "n_iter": int(np.max(logit.n_iter_)), "converged": bool(np.max(logit.n_iter_) < 500),
            "n_examples": int(len(stacked)), "fit_gap": fit_gap,
            "objective_status": "converged" if int(np.max(logit.n_iter_)) < 500 else "iteration_limit",
        })
        if fit_gap > 1e-6:
            dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "fit_gap": fit_gap})
            raise SystemExit(5)
    if saved:
        pass
    else:
        pfa_iterations = int(np.max(pfa.n_iter_))
        logit_iterations = int(np.max(logit.n_iter_))
    print("score confirmatory", flush=True)
    confirmatory = chronorep.load_confirmatory()
    dev_users = set(selection["selected_users"])
    overlap = len(dev_users & set(confirmatory))
    users = sorted(confirmatory)
    cache = OUT / "private_scores"
    score_names = ("y.npy", "user_index.npy", "track_a_pfa.npy", "track_a_gru.npy", "track_b_logit.npy", "track_b_primary_gru.npy", "nonempty_history.npy")
    if os.environ.get("ESTCTRL_REUSE_SCORES") == "1" and all((cache / name).exists() for name in score_names):
        y = np.load(cache / "y.npy")
        user_index = np.load(cache / "user_index.npy")
        pfa_p = np.load(cache / "track_a_pfa.npy")
        gru_a = np.load(cache / "track_a_gru.npy")
        logit_p = np.load(cache / "track_b_logit.npy")
        gru_b = np.load(cache / "track_b_primary_gru.npy")
        nonempty = np.load(cache / "nonempty_history.npy")
        cursor = int(y.shape[0])
        print(f"reuse confirmatory scores {cursor}", flush=True)
    else:
        y = np.empty(VALID, dtype=np.int8)
        user_index = np.empty(VALID, dtype=np.int32)
        pfa_p = np.empty(VALID, dtype=np.float64)
        gru_a = np.empty(VALID, dtype=np.float64)
        logit_p = np.empty(VALID, dtype=np.float64)
        gru_b = np.empty(VALID, dtype=np.float64)
        nonempty = np.zeros(VALID, dtype=bool)
        cursor = 0
        pending_a: list = []
        pending_b: list = []
        pending_meta: list = []

        def flush_predictions() -> None:
            nonlocal cursor
            if not pending_a:
                return
            pred_a = chronorep.score_forward(count_model, pending_a, pilot.device)
            pred_b = chronorep.score_forward(frozen, pending_b, pilot.device)
            pred_l = logit_probabilities(pending_b, weight, intercept)
            skill_ids = np.asarray([item[3] for item in pending_meta], dtype=np.int32)
            successes = np.asarray([item[1] for item in pending_meta], dtype=np.float64)
            failures = np.asarray([item[2] for item in pending_meta], dtype=np.float64)
            pred_p = pfa_probability(skill_ids, successes, failures, coef)
            for local, meta in enumerate(pending_meta):
                y[cursor] = meta[0]
                user_index[cursor] = meta[5]
                pfa_p[cursor] = pred_p[local]
                gru_a[cursor] = pred_a[local]
                logit_p[cursor] = pred_l[local]
                gru_b[cursor] = pred_b[local]
                nonempty[cursor] = meta[4]
                cursor += 1
            pending_a.clear()
            pending_b.clear()
            pending_meta.clear()

        for user_i, uid in enumerate(users):
            by_start, by_end = chronorep.order_valid(confirmatory[uid])
            for row in by_start:
                prefix = chronorep.prefix_events(by_end, row["start"])
                current = chronorep.as_event(row)
                success, failure = counts_of(prefix, current.skill_id)
                skill_i = int(skill2i.get(current.skill_id, 0))
                pending_a.append(count_tokens(skill_i, success, failure, outcome0, outcome1, start_id))
                pending_b.append(pilot._tokens(prefix, current, False))
                pending_meta.append((int(row["correct"]), success, failure, skill_i, len(prefix) > 0, user_i))
                if len(pending_a) >= 8192:
                    flush_predictions()
            if user_i and user_i % 500 == 0:
                print(f"predict_users {user_i}", flush=True)
        flush_predictions()
    if cursor != VALID:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "scored": cursor})
        raise SystemExit(6)
    cache = OUT / "private_scores"
    np.save(cache / "y.npy", y)
    np.save(cache / "user_index.npy", user_index)
    np.save(cache / "track_a_pfa.npy", pfa_p)
    np.save(cache / "track_a_gru.npy", gru_a)
    np.save(cache / "track_b_logit.npy", logit_p)
    np.save(cache / "track_b_primary_gru.npy", gru_b)
    np.save(cache / "nonempty_history.npy", nonempty)
    learners = len(users)
    rows = [
        full_metrics("CanonicalPFA40", y, pfa_p, learners),
        full_metrics("CountEncodedGRU", y, gru_a, learners),
        full_metrics("ActionTokenLogit", y, logit_p, learners),
        full_metrics("ActionTokenGRU_primary_frozen", y, gru_b, learners),
        full_metrics("CanonicalPFA40_nonempty_history_context", y[nonempty], pfa_p[nonempty], int(np.unique(user_index[nonempty]).size)),
        full_metrics("CountEncodedGRU_nonempty_history_context", y[nonempty], gru_a[nonempty], int(np.unique(user_index[nonempty]).size)),
    ]
    write_csv("track_a_predictive_metrics.csv", rows[:2] + rows[4:])
    write_csv("track_b_predictive_metrics.csv", rows[2:4])
    bins = []
    bins.extend(calibration_rows("CanonicalPFA40", y, pfa_p))
    bins.extend(calibration_rows("CountEncodedGRU", y, gru_a))
    bins.extend(calibration_rows("ActionTokenLogit", y, logit_p))
    bins.extend(calibration_rows("ActionTokenGRU_primary_frozen", y, gru_b))
    write_csv("track_a_calibration_bins.csv", [row for row in bins if row["model"] in {"CanonicalPFA40", "CountEncodedGRU"}])
    write_csv("track_b_calibration_bins.csv", [row for row in bins if row["model"] in {"ActionTokenLogit", "ActionTokenGRU_primary_frozen"}])
    chronorep_reference = 0.9790957170706067
    chronorep_gap = abs(rows[3]["auroc"] - chronorep_reference)
    print("bootstrap predictive", flush=True)
    paired = bootstrap_paired(y, user_index, {
        "CanonicalPFA40": pfa_p, "CountEncodedGRU": gru_a,
        "ActionTokenLogit": logit_p, "ActionTokenGRU_CHRONOREP": gru_b,
    }, learners)
    write_csv("track_a_paired_contrasts.csv", [row for row in paired if row["track"] == "A"])
    print("development bundles", flush=True)
    geometry = chronorep.build_development(set())
    for bundle in geometry["bundles"].values():
        for slot in bundle.values():
            slot.pfa = set()
            slot.pfa_list = []
            slot.deltas = None
            if not chronorep.supported(slot):
                slot.gru = set()
    print("certificates", flush=True)
    status, false_stable, uncovered, covered, contained, supported_n, transitions, native, matched, counters = score_certificates(
        confirmatory, users, geometry["bundles"], pilot, weight, intercept, logit_p
    )
    write_certificate_tables(status, false_stable, uncovered, covered, contained, supported_n, transitions, native, matched, learners)
    print("certificate bootstrap", flush=True)
    cert_rows = certificate_bootstrap(status, y, user_index, logit_p, gru_b, learners)
    write_csv("learner_bootstrap_intervals.csv", paired + cert_rows)
    qa = independent_certificate_qa(confirmatory, geometry["bundles"], pilot, weight, intercept, outcome0, outcome1, start_id, coef)
    dump("independent_input_and_scoring_qa.json", {"track_a_development": example_qa, "permutation_violations": perm_bad, "development_confirmatory_user_overlap": overlap, "track_b": qa, "chronorep_auroc_abs_gap": chronorep_gap})
    invariants = {
        "examples": counts,
        "scored": cursor,
        "overlap_users": overlap,
        "chronorep_auroc_abs_gap": chronorep_gap,
        "fit_gap": fit_gap,
        "pfa_iterations": pfa_iterations,
        "logit_iterations": logit_iterations,
        "certificate_counters": counters,
        "future_violations": counters["future"],
        "nesting_violations": counters["support"] + counters["set"] + counters["action"],
        "qa_pass": example_qa["violations"] == 0 and perm_bad == 0 and qa["pass"],
        "status": "COMPLETE",
    }
    totals_ok = counters["status_sums"] == [VALID] * 4 and counters["transition_sums"] == [VALID] * 4 and counters["event_cursor"] == VALID and counters["cache_abs"] <= 1e-9 and counters["implication"] == 0
    numerical_failure = chronorep_gap > 1e-4 or fit_gap > 1e-6 or counters["future"] or counters["support"] or counters["set"] or counters["action"] or not qa["pass"] or overlap or not totals_ok
    if numerical_failure:
        invariants["status"] = "BLOCKED_NUMERICAL_INCONSISTENCY"
    elif logit_iterations >= 500 or pfa_iterations >= 500:
        invariants["status"] = "BLOCKED_TRAINING_OR_EXECUTION"
    dump("numerical_invariants.json", invariants)
    dump("execution_progress.json", {"complete": True, "elapsed_sec": time.time() - started})
    dump("experiment_manifest.json", {
        "experiment": "pipeline",
        "script_sha256": chronorep.sha256_file(Path(__file__).resolve()),
        "elapsed_sec": time.time() - started,
        "status": invariants["status"],
        "chronorep_checkpoint_sha256": pre["chronorep_checkpoint_sha256"],
        "count_gru_sha256": gru_log["checkpoint_sha256"],
        "training_of_frozen_chronorep": False,
        "new_chronorep_checkpoint": False,
    })
    checksums = write_checksums()
    after = chronorep.sha256_file(CHRONOREP_CKPT)
    print(invariants["status"], "checksum_mismatches", len(checksums["mismatches"]), "chronorep_unchanged", after == GRU_SHA, flush=True)
    if invariants["status"] != "COMPLETE" or checksums["mismatches"] or after != GRU_SHA:
        raise SystemExit(7)


def score_certificates(grouped, users, bundles, pilot, weight, intercept, realized_cache: np.ndarray):
    n_users = len(users)
    status = np.zeros((n_users, 4, 3, 3), dtype=np.int32)
    false_stable = np.zeros((n_users, 4, 3), dtype=np.int32)
    uncovered = np.zeros((n_users, 4, 3), dtype=np.int32)
    covered = np.zeros((n_users, 4), dtype=np.int32)
    contained = np.zeros((n_users, 4, 3), dtype=np.int32)
    supported_n = np.zeros((n_users, 4), dtype=np.int32)
    transitions = np.zeros((3, 4, 3, 3), dtype=np.int64)
    native = np.full((n_users, 4, 3), np.nan)
    matched = np.full((n_users, 4, 3), np.nan)
    counters = {"future": 0, "support": 0, "set": 0, "action": 0, "implication": 0, "cache_abs": 0.0}
    event_cursor = 0
    for user_i, uid in enumerate(users):
        by_start, by_end = chronorep.order_valid(grouped[uid])
        supported_rows = {schema: [] for schema in SCHEMAS}
        cursor = 0
        records = []
        for row in by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                cursor += 1
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                counters["future"] += 1
            prefix = chronorep.prefix_events(by_end, row["start"])
            ev = chronorep.as_event(row)
            flags = {}
            slots = {}
            for schema in SCHEMAS:
                slot = bundles[schema].get(chronorep.release_key(ev, schema))
                slots[schema] = slot
                flags[schema] = chronorep.supported(slot)
            for left, right in EDGES:
                if flags[right] and not flags[left]:
                    counters["support"] += 1
            records.append((row, prefix, flags, slots))
            for schema in SCHEMAS:
                if flags[schema]:
                    supported_rows[schema].append(row)
        native_rows = {}
        for schema in SCHEMAS:
            native_rows[schema] = max(supported_rows[schema], key=lambda item: (item["end"], item["source_row"]))["source_row"] if supported_rows[schema] else None
        common = [row for row, _prefix, flags, _slots in records if all(flags[schema] for schema in SCHEMAS)]
        matched_row = max(common, key=lambda item: (item["end"], item["source_row"]))["source_row"] if common else None
        for row, prefix, flags, slots in records:
            own = chronorep.gru_fp(row)
            realized = float(realized_cache[event_cursor])
            event_cursor += 1
            if event_cursor % 25 == 0:
                fresh = float(logit_probabilities([pilot._tokens(prefix, chronorep.as_event(row), False)], weight, intercept)[0])
                counters["cache_abs"] = max(counters["cache_abs"], abs(fresh - realized))
            score_map = {}
            if flags["S0"]:
                others = [fp for fp in slots["S0"].gru if fp != own]
                if others:
                    scored = logit_probabilities([chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in others], weight, intercept)
                    score_map = {fp: float(value) for fp, value in zip(others, scored)}
                if own in slots["S0"].gru:
                    score_map[own] = realized
            codes = np.zeros((4, 3), dtype=np.int8)
            actions = [None] * 4
            for schema_i, schema in enumerate(SCHEMAS):
                if not flags[schema]:
                    for tau_i in range(3):
                        status[user_i, schema_i, tau_i, 0] += 1
                    continue
                values = [score_map[fp] for fp in slots[schema].gru]
                if not values:
                    values = [0.5]
                lo, hi = min(values), max(values)
                is_covered = own in slots[schema].gru
                supported_n[user_i, schema_i] += 1
                if is_covered:
                    covered[user_i, schema_i] += 1
                for tau_i, tau in enumerate(TAUS):
                    code, action, contained_flag = classify(lo, hi, realized, tau)
                    codes[schema_i, tau_i] = code
                    if tau == PRIMARY:
                        actions[schema_i] = action
                    status[user_i, schema_i, tau_i, code] += 1
                    if code == 1 and contained_flag:
                        contained[user_i, schema_i, tau_i] += 1
                    elif code == 1:
                        false_stable[user_i, schema_i, tau_i] += 1
                        if is_covered:
                            counters["implication"] += 1
                    else:
                        contained[user_i, schema_i, tau_i] += 1
                    if code == 1 and not is_covered:
                        uncovered[user_i, schema_i, tau_i] += 1
                if native_rows[schema] == row["source_row"]:
                    native[user_i, schema_i] = (1 - hi, 1 - lo, 1 - realized)
                if matched_row == row["source_row"]:
                    matched[user_i, schema_i] = (1 - hi, 1 - lo, 1 - realized)
            for tau_i, tau in enumerate(TAUS):
                for edge_i, (left, right) in enumerate(EDGES):
                    left_i, right_i = SCHEMAS.index(left), SCHEMAS.index(right)
                    transitions[tau_i, edge_i, codes[left_i, tau_i], codes[right_i, tau_i]] += 1
                    if codes[left_i, tau_i] == 1 and codes[right_i, tau_i] == 2:
                        counters["set"] += 1
                    if tau == PRIMARY and codes[left_i, tau_i] == 1 and codes[right_i, tau_i] == 1 and actions[left_i] != actions[right_i]:
                        counters["action"] += 1
        if user_i and user_i % 200 == 0:
            print(f"certificate_users {user_i}", flush=True)
    counters["event_cursor"] = event_cursor
    counters["status_sums"] = [int(status[:, i, 1].sum()) for i in range(4)]
    counters["transition_sums"] = [int(transitions[1, i].sum()) for i in range(4)]
    return status, false_stable, uncovered, covered, contained, supported_n, transitions, native, matched, counters


def write_certificate_tables(status, false_stable, uncovered, covered, contained, supported_n, transitions, native, matched, learners):
    rows = []
    false_rows = []
    tau_rows = []
    for schema_i, schema in enumerate(SCHEMAS):
        for tau_i, tau in enumerate(TAUS):
            unsup = int(status[:, schema_i, tau_i, 0].sum())
            stable = int(status[:, schema_i, tau_i, 1].sum())
            sett = int(status[:, schema_i, tau_i, 2].sum())
            n_u = int(uncovered[:, schema_i, tau_i].sum())
            n_fs = int(false_stable[:, schema_i, tau_i].sum())
            row = {
                "model": "ActionTokenLogit", "schema": schema, "tau": tau, "valid_targets": VALID,
                "unsupported": unsup, "stable": stable, "set_valued": sett, "supported": stable + sett,
                "event_nc": unsup + sett, "event_nc_rate": (unsup + sett) / VALID,
                "fingerprint_covered_supported": int(covered[:, schema_i].sum()),
                "stable_uncovered": n_u, "false_stable": n_fs,
                "false_stable_over_stable": (n_fs / stable) if stable else "N/A",
                "action_contained_supported": int(contained[:, schema_i, tau_i].sum()),
                "learners": int(np.sum(supported_n[:, schema_i] > 0)),
            }
            tau_rows.append(row)
            if tau == PRIMARY:
                rows.append(row)
                false_rows.append(row)
    write_csv("track_b_event_certificates.csv", rows)
    write_csv("track_b_false_stable_and_fingerprint_coverage.csv", false_rows)
    write_csv("track_b_threshold_sensitivity.csv", tau_rows)
    trans_rows = []
    for tau_i, tau in enumerate(TAUS):
        for edge_i, (left, right) in enumerate(EDGES):
            matrix = transitions[tau_i, edge_i]
            trans_rows.append({
                "model": "ActionTokenLogit", "tau": tau, "left": left, "right": right, "matrix_sum": int(matrix.sum()),
                "stable_to_unsupported": int(matrix[1, 0]), "set_to_unsupported": int(matrix[2, 0]),
                "set_to_stable": int(matrix[2, 1]), "stable_to_set": int(matrix[1, 2]),
                "unsupported_to_supported": int(matrix[0, 1] + matrix[0, 2]),
                "gross_support_loss": int(matrix[1, 0] + matrix[2, 0]),
                "gross_ambiguity_resolution": int(matrix[2, 1]),
                "net_event_nc": int((matrix[:, 0].sum() + matrix[:, 2].sum()) - (matrix[0, :].sum() + matrix[2, :].sum())),
            })
    write_csv("track_b_paired_release_transitions.csv", trans_rows)
    native_rows, matched_rows, risk_rows = [], [], []
    common = np.isfinite(matched[:, 0, 0])
    for schema_i, schema in enumerate(SCHEMAS):
        eligible = np.isfinite(native[:, schema_i, 0])
        counts = chronorep.membership_counts(native[eligible, schema_i, 0], native[eligible, schema_i, 1], 0.05)
        native_rows.append(chronorep.capacity_row("ActionTokenLogit", schema, 0.05, learners, int(eligible.sum()), counts, "completion_time_native"))
        risk_rows.append(chronorep.risk_block("ActionTokenLogit", schema, 0.05, native[eligible, schema_i, 0], native[eligible, schema_i, 1], native[eligible, schema_i, 2], "completion_time_native"))
        cell = matched[common, schema_i]
        mcounts = chronorep.membership_counts(cell[:, 0], cell[:, 1], 0.05)
        matched_rows.append(chronorep.capacity_row("ActionTokenLogit", schema, 0.05, learners, int(common.sum()), mcounts, "matched_event"))
        risk_rows.append(chronorep.risk_block("ActionTokenLogit", schema, 0.05, cell[:, 0], cell[:, 1], cell[:, 2], "matched_event"))
    write_csv("track_b_native_capacity.csv", native_rows)
    write_csv("track_b_matched_capacity.csv", matched_rows)
    write_csv("track_b_realized_risk_validity.csv", risk_rows)


def certificate_bootstrap(status, y, user, logit_p, gru_b, learners: int) -> list[dict]:
    rng = np.random.default_rng(20313)
    draws = rng.integers(0, learners, size=(2000, learners))
    rows = []
    point_nc = []
    for schema_i, schema in enumerate(SCHEMAS):
        nc = (status[:, schema_i, 1, 0] + status[:, schema_i, 1, 2]).astype(np.float64)
        valid = status[:, schema_i, 1].sum(axis=1).astype(np.float64)
        point = float(nc.sum() / VALID)
        rates = nc[draws].sum(axis=1) / valid[draws].sum(axis=1)
        low, high = np.percentile(rates, [2.5, 97.5])
        rows.append({"track": "B_certificate", "quantity": f"{schema}_event_nc_rate", "point": point, "ci_low": float(low), "ci_high": float(high), "replicates": 2000, "seed": 20313, "resample": "paired confirmatory learners with replacement"})
        point_nc.append(nc)
    diff = (point_nc[1] - point_nc[0])
    rates = diff[draws].sum(axis=1) / VALID
    # The valid denominator is the full population size of each replicate's events, not 939819.
    valid_draw = status[:, 0, 1].sum(axis=1).astype(np.float64)
    rates = diff[draws].sum(axis=1) / valid_draw[draws].sum(axis=1)
    low, high = np.percentile(rates, [2.5, 97.5])
    rows.append({"track": "B_certificate", "quantity": "S1_minus_S0_event_nc_rate", "point": float(diff.sum() / VALID), "ci_low": float(low), "ci_high": float(high), "replicates": 2000, "seed": 20313, "resample": "paired confirmatory learners with replacement"})
    return rows


def independent_certificate_qa(grouped, bundles, pilot, weight, intercept, outcome0, outcome1, start_id, coef) -> dict:
    pool = []
    for uid, rows in grouped.items():
        for row in rows:
            if row["valid"]:
                pool.append((hashlib.sha256(f"WSDM203-G13-B|{row['source_row']}".encode()).hexdigest(), uid, row))
    pool.sort()
    table = weight.reshape(64, N_SKILL, N_TOK)
    max_diff = 0.0
    label_mismatch = leak = future = order_mismatch = 0
    alternatives = 0
    checked = 0

    def manual(skills: list[int], atypes: list[int]) -> float:
        total = intercept
        for pos, (skill_i, tok) in enumerate(zip(skills, atypes)):
            total += float(table[pos, skill_i, tok])
        return float(sigmoid(np.asarray([total]))[0])

    for _rank, uid, row in pool[:200]:
        checked += 1
        _by_start, by_end = chronorep.order_valid(grouped[uid])
        cut = chronorep.history_cut(by_end, row["start"])
        history = by_end[max(0, cut - 40):cut]
        prefix = [chronorep.as_event(item) for item in history]
        if any(item["end"] >= row["start"] for item in history):
            future += 1
        tokens = pilot._tokens(prefix, chronorep.as_event(row), False)
        leaked = pilot._tokens(prefix, chronorep.as_event(row), True)
        if row["action_types"] and tokens[1] and tokens[1][-1] in {outcome0, outcome1}:
            leak += 1
        if len(tokens[0]) < MAX_HIST_TOKENS and len(leaked[0]) != len(tokens[0]) + 1:
            leak += 1
        direct = float(logit_probabilities([tokens], weight, intercept)[0])
        manual_direct = manual(tokens[0], tokens[1])
        max_diff = max(max_diff, abs(direct - manual_direct))
        ev = chronorep.as_event(row)
        slot = bundles["S0"].get(chronorep.release_key(ev, "S0"))
        if chronorep.supported(slot) and slot.gru:
            fps = list(slot.gru)
            chosen = fps if len(fps) <= 48 else fps[:24]
            seqs = [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in chosen]
            forward = logit_probabilities(seqs, weight, intercept)
            manual_scores = np.asarray([manual(skills, atypes) for skills, atypes in seqs], dtype=np.float64)
            alternatives += len(seqs)
            max_diff = max(max_diff, float(np.max(np.abs(forward - manual_scores))))
            lo, hi = float(np.min(forward)), float(np.max(forward))
            lo2, hi2 = float(np.min(manual_scores)), float(np.max(manual_scores))
            for tau in TAUS:
                if classify(lo, hi, direct, tau)[0] != classify(lo2, hi2, manual_direct, tau)[0]:
                    label_mismatch += 1
            reversed_scores = logit_probabilities(list(reversed(seqs)), weight, intercept)
            if float(np.min(forward)) != float(np.min(reversed_scores)) or float(np.max(forward)) != float(np.max(reversed_scores)):
                order_mismatch += 1
        success, failure = counts_of(prefix, row["skill_id"])
        encoded = count_tokens(int(pilot.skill2i.get(row["skill_id"], 0)), success, failure, outcome0, outcome1, start_id)
        repeated = pfa_probability(np.asarray([encoded[0][0]]), np.asarray([float(success)]), np.asarray([float(failure)]), coef)[0]
        repeated_again = pfa_probability(np.asarray([encoded[0][-1]]), np.asarray([float(success)]), np.asarray([float(failure)]), coef)[0]
        max_diff = max(max_diff, abs(float(repeated) - float(repeated_again)))
        shuffled = prefix.copy()
        random.Random(int(row["source_row"])).shuffle(shuffled)
        if counts_of(shuffled, row["skill_id"]) != (success, failure):
            leak += 1
        if any(tok not in {outcome0, outcome1, start_id} for tok in encoded[1]):
            leak += 1
    return {
        "checked": checked, "alternative_completions": alternatives, "max_abs_diff": max_diff,
        "label_mismatches": label_mismatch, "order_mismatches": order_mismatch, "leaks": leak, "future": future,
        "pass": checked >= 200 and alternatives >= 200 and label_mismatch == 0 and order_mismatch == 0 and leak == 0 and future == 0 and max_diff <= 1e-9,
    }


def write_checksums() -> dict:
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/matched_estimand_controls/output_checksums.json"], "n_files": len(files), "files": files})
    bad = [item["path"] for item in files if chronorep.sha256_file(ROOT / item["path"]) != item["sha256"]]
    return {"mismatches": bad}


if __name__ == "__main__":
    main()
