#!/usr/bin/env python3
"""Chronology-valid causal token-attention consumer.

The model is an action-conditioned adaptation. It is not SAKT and it is not AKT.
Earlier artifacts are not modified.
"""
from __future__ import annotations

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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "matchedinfo", ROOT / "scripts/assistments/matched_information.py"
)
matchedinfo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matchedinfo)
chronorep = matchedinfo.chronorep

from r7_core import MAX_HIST_TOKENS, MAX_SEQ_ACTIONS, Event, NeuralPilot  # noqa: E402

OUT = ROOT / "results/assistments/attention_consumer"
CACHE = OUT / "private_score_caches"
CHRONOREP_DIR = ROOT / "results/assistments/chronological_replication"
MATCHEDINFO_DIR = ROOT / "results/assistments/matched_information"
SEEDROB_DIR = ROOT / "results/assistments/seed_robustness"
SELECTION_SHA = "cbe4fd9e0075c78ae5997892153d37920497d14c64a21af3ffbf3594e5c907e2"
CHRONOREP_CKPT_SHA = "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
ARCH = {
    "name": "CausalTokenAttentionKT",
    "consumer_name": "Causal Token-Attention Consumer",
    "token_embedding_dim": 64,
    "layers": 2,
    "heads": 4,
    "feedforward_dim": 128,
    "activation": "GELU",
    "dropout": 0.10,
    "norm": "LayerNorm",
    "norm_eps": 1e-5,
    "max_positions": 64,
    "positional_convention": "left_aligned_index_in_retained_64_token_window",
    "readout": "last_real_token",
    "combination": "sum of skill embedding and action-or-outcome embedding",
    "padding_mask": "positional, not token-id based; unseen skills may use id 0 and remain real tokens",
    "causal_mask": "standard upper-triangular block; not AKT monotonic attention",
    "batch_size": 64,
    "epochs": 2,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "binary_cross_entropy",
    "seed": 17,
}


def dump(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=chronorep._json_default) + "\n", encoding="utf-8")


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None, causal: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class CausalTokenAttentionKT(nn.Module):
    def __init__(self, n_skills: int, n_atypes: int) -> None:
        super().__init__()
        dim = ARCH["token_embedding_dim"]
        self.skill_emb = nn.Embedding(n_skills + 1, dim, padding_idx=0)
        self.atype_emb = nn.Embedding(n_atypes + 3, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(ARCH["max_positions"], dim)
        self.blocks = nn.ModuleList(
            AttentionBlock(dim, ARCH["heads"], ARCH["feedforward_dim"], ARCH["dropout"]) for _ in range(ARCH["layers"])
        )
        self.final_ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 1)
        self.outcome0 = n_atypes + 1
        self.outcome1 = n_atypes + 2
        self.max_len = ARCH["max_positions"]

    def _causal(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def encode(self, skill_ids: torch.Tensor, atype_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        batch, length = skill_ids.shape
        pos = torch.arange(length, device=skill_ids.device).unsqueeze(0).expand(batch, length)
        x = self.skill_emb(skill_ids) + self.atype_emb(atype_ids) + self.pos_emb(pos.clamp(max=self.max_len - 1))
        causal = self._causal(length, skill_ids.device)
        pad = None
        if lengths is not None:
            pad = torch.arange(length, device=skill_ids.device).unsqueeze(0) >= lengths.unsqueeze(1)
        for block in self.blocks:
            x = block(x, pad, causal)
        return self.final_ln(x)

    def forward_from_state(self, skill_ids: torch.Tensor, atype_ids: torch.Tensor, h0=None):
        hidden = self.encode(skill_ids, atype_ids, None)
        score = torch.sigmoid(self.out(hidden[:, -1, :]).squeeze(-1))
        return score, None

    def forward_padded(self, skill_ids: torch.Tensor, atype_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(skill_ids, atype_ids, lengths)
        index = (lengths - 1).clamp(min=0)
        chosen = hidden[torch.arange(skill_ids.shape[0], device=skill_ids.device), index]
        return torch.sigmoid(self.out(chosen).squeeze(-1))


def make_model(n_skills: int, n_atypes: int, seed: int) -> CausalTokenAttentionKT:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CausalTokenAttentionKT(n_skills, n_atypes)
    model.to(torch.device("cpu"))
    return model


def mask_tests(n_skills: int, n_atypes: int) -> dict:
    model = make_model(n_skills, n_atypes, 17)
    model.eval()
    torch.manual_seed(0)
    skills = torch.randint(1, n_skills + 1, (4, 17))
    atypes = torch.randint(1, n_atypes + 1, (4, 17))
    with torch.inference_mode():
        hidden = model.encode(skills, atypes, None)
        changed = atypes.clone()
        changed[:, 12:] = torch.randint(1, n_atypes + 1, (4, 5))
        hidden_changed = model.encode(skills, changed, None)
        causal_diff = float((hidden[:, :12] - hidden_changed[:, :12]).abs().max())
        score = model.forward_from_state(skills, atypes, None)[0]
        again = model.forward_from_state(skills, atypes, None)[0]
        repeat = float((score - again).abs().max())
        lengths = torch.full((4,), 17)
        padded_s = torch.zeros(4, 24, dtype=torch.long)
        padded_a = torch.zeros(4, 24, dtype=torch.long)
        padded_s[:, :17] = skills
        padded_a[:, :17] = atypes
        padded_a[:, 17:] = 4
        padded_s[:, 17:] = 3
        padded_score = model.forward_padded(padded_s, padded_a, lengths)
        pad_diff = float((score - padded_score).abs().max())
        empty_ok = hidden.shape == (4, 17, ARCH["token_embedding_dim"]) and score.shape == (4,)
    result = {
        "batch": 4,
        "length": 17,
        "hidden_shape": list(hidden.shape),
        "score_shape": list(score.shape),
        "shapes_ok": bool(empty_ok),
        "causal_max_abs_diff_before_edit": causal_diff,
        "padding_invariance_max_abs_diff": pad_diff,
        "dropout_disabled_repeat_max_abs_diff": repeat,
        "tolerance": 1e-5,
        "pass": bool(empty_ok and causal_diff <= 1e-5 and pad_diff <= 1e-5 and repeat <= 1e-5 and torch.isfinite(score).all()),
    }
    del model
    return result


def load_rows(split: str, selected: set[str] | None = None) -> dict:
    rows = defaultdict(list)
    parquet = pq.ParquetFile(chronorep.RECON / f"{split}.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=50000):
        raw = batch.to_pylist()
        starts = chronorep.civil_us([item["start_time"] for item in raw])
        ends = chronorep.civil_us([item["end_time"] for item in raw])
        for item, start, end in zip(raw, starts, ends):
            uid = str(item["user_id"])
            if selected is None or uid in selected:
                rows[uid].append(chronorep.compact_row(item, int(start), int(end)))
        seen += len(raw)
        if seen % 1000000 == 0:
            print(f"load {split} {seen}", flush=True)
    return rows


def tokens_of(model, skill2i, atype2i, row, by_end):
    prefix = chronorep.prefix_events(by_end, row["start"])
    pilot = NeuralPilot()
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.model = model
    return pilot._tokens(prefix, chronorep.as_event(row), False), prefix, pilot


def run_token_qa(model, skill2i, atype2i, targets, raw_rows) -> dict:
    unique = []
    seen_rows = set()
    for item in targets:
        if item[0]["source_row"] not in seen_rows:
            seen_rows.add(item[0]["source_row"])
            unique.append(item)
    ranked = sorted(unique, key=lambda item: hashlib.sha256(f"WSDM203-G11-QA|{item[0]['source_row']}".encode()).hexdigest())
    indexed = []
    for row, by_end in ranked:
        full = matchedinfo.history_tokens(by_end, row["start"], skill2i, atype2i, model.outcome0, model.outcome1)
        length = len(row["action_types"][:MAX_SEQ_ACTIONS])
        unfinished = any(item["source_row"] != row["source_row"] and item["start"] < row["start"] and item["end"] >= row["start"] for item in by_end)
        invalid_neighbor = any((not item["valid"]) and item["start"] < row["end"] and item["end"] >= row["start"] for item in raw_rows.get(row["user_id"], ()))
        indexed.append((row, by_end, len(full[0]), length, unfinished, skill2i.get(row["skill_id"], 0) == 0, invalid_neighbor))
    # Invalid neighbors live outside by_end. Scan the raw user rows only when attached.
    picked = []
    seen = set()
    strata = {}

    def take(name, predicate, limit):
        available = got = 0
        for item in indexed:
            if predicate(item):
                available += 1
                if got < limit and item[0]["source_row"] not in seen:
                    picked.append((name, item))
                    seen.add(item[0]["source_row"])
                    got += 1
        strata[name] = {"available": available, "selected": got}

    take("empty_history", lambda item: item[2] == 0, 20)
    take("short_history", lambda item: 0 < item[2] <= 8, 20)
    take("long_history", lambda item: item[2] >= 64, 30)
    take("current_near_cap", lambda item: item[3] >= 16, 30)
    take("unfinished_if_start_sorted", lambda item: item[4], 20)
    take("invalid_neighbor", lambda item: item[6], 15)
    take("completed_predecessor", lambda item: item[2] > 0, 20)
    take("unseen_skill_id_zero", lambda item: item[5], 15)
    for item in indexed:
        if len(picked) >= 250:
            break
        if item[0]["source_row"] not in seen:
            picked.append(("hash_fill", item))
            seen.add(item[0]["source_row"])
    violations = defaultdict(int)
    last_token_checks = 0
    model.eval()
    for _name, (row, by_end, _n, length, _unfinished, _unk, _invalid) in picked:
        (skills, atypes), prefix, pilot = tokens_of(model, skill2i, atype2i, row, by_end)
        reference = pilot._tokens(prefix, chronorep.as_event(row), False)
        if reference != (skills, atypes):
            violations["chronorep_token_identity"] += 1
        if len(skills) > MAX_HIST_TOKENS or length > MAX_SEQ_ACTIONS:
            violations["caps"] += 1
        if length and any(tok in (model.outcome0, model.outcome1) for tok in atypes[-length:]):
            violations["current_outcome"] += 1
        if length and atypes[-1] != atype2i.get(row["action_types"][:MAX_SEQ_ACTIONS][-1], 0):
            violations["last_token_not_current_action"] += 1
        cut = chronorep.history_cut(by_end, row["start"])
        if any(item["end"] >= row["start"] for item in by_end[:cut]):
            violations["future_history"] += 1
        if len(prefix) > 40:
            violations["history_event_cap"] += 1
        same = pilot._tokens(prefix, Event("", "", row["skill_id"], 1 - row["correct"], 0, 0, "", row["action_types"], tuple(), ""), False)
        if same != (skills, atypes):
            violations["fingerprint_not_sufficient"] += 1
        if skills:
            last_token_checks += 1
            with torch.inference_mode():
                hidden = model.encode(torch.tensor([skills]), torch.tensor([atypes]), None)
                if hidden.shape[-2] != len(skills):
                    violations["readout_shape"] += 1
    return {
        "cases": len(picked),
        "target_cases": 250,
        "strata": strata,
        "violations": dict(violations),
        "sequences_with_tokens": last_token_checks,
        "pass": len(picked) >= 250 and not violations,
    }


def batch_step(model, opt, batch, skill2i, atype2i):
    seqs = []
    labels = []
    pilot = NeuralPilot()
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.model = model
    for row, by_end in batch:
        skills, atypes = pilot._tokens(chronorep.prefix_events(by_end, row["start"]), chronorep.as_event(row), False)
        if not skills:
            raise SystemExit("empty chronological token sequence in a training example")
        seqs.append((skills, atypes))
        labels.append(float(row["correct"]))
    width = max(len(item[0]) for item in seqs)
    skills = torch.zeros(len(seqs), width, dtype=torch.long)
    atypes = torch.zeros(len(seqs), width, dtype=torch.long)
    lengths = torch.zeros(len(seqs), dtype=torch.long)
    for i, (sk, at) in enumerate(seqs):
        skills[i, : len(sk)] = torch.tensor(sk)
        atypes[i, : len(at)] = torch.tensor(at)
        lengths[i] = len(sk)
    pred = model.forward_padded(skills, atypes, lengths)
    y = torch.tensor(labels)
    loss = nn.functional.binary_cross_entropy(pred, y)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item()), int(torch.isfinite(loss).all()), len(seqs), pred.detach()


def train_epoch(model, opt, targets, skill2i, atype2i, seed_label: str, progress: Path | None, start_index: int, running: dict):
    model.train()
    nonfinite = 0
    total = running.get("loss_sum", 0.0)
    seen = running.get("seen", 0)
    steps = running.get("steps", 0)
    begin = start_index
    for offset in range(begin, len(targets), ARCH["batch_size"]):
        batch = targets[offset : offset + ARCH["batch_size"]]
        loss, finite, n, _pred = batch_step(model, opt, batch, skill2i, atype2i)
        if not finite:
            nonfinite += 1
        total += loss * n
        seen += n
        steps += 1
        if steps % 50 == 0:
            print(f"train {seed_label} step {steps} examples {seen} loss {total / seen}", flush=True)
        if progress is not None and steps % 100 == 0:
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(),
                "next_index": offset + len(batch),
                "loss_sum": total,
                "seen": seen,
                "steps": steps,
                "nonfinite": nonfinite,
                "label": seed_label,
            }, progress)
    return {"loss": total / seen if seen else None, "examples": seen, "steps": steps, "nonfinite": nonfinite}


def smoke(model, skill2i, atype2i, dev_rows, cal_rows) -> dict:
    users = sorted(dev_rows, key=lambda uid: hashlib.sha256(f"WSDM203-G7-DEV-PILOT-v1|{uid}".encode()).hexdigest())[:128]
    targets = []
    for uid in users:
        ordered = sorted(dev_rows[uid], key=lambda item: item["source_row"])[-chronorep.MAX_TRAIN_EVENTS_PER_USER :]
        _by_start, by_end = chronorep.order_valid(dev_rows[uid])
        for row in ordered:
            if row["valid"]:
                targets.append((row, by_end))
    opt = torch.optim.Adam(model.parameters(), lr=ARCH["learning_rate"])
    started = time.time()
    result = train_epoch(model, opt, targets, skill2i, atype2i, "smoke", None, 0, {})
    path = OUT / "NON_PRIMARY_ATTENTION_SMOKE_TEST.pt"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "label": "NON_PRIMARY_ATTENTION_SMOKE_TEST"}, path)
    reloaded = CausalTokenAttentionKT(len(skill2i) - 1, len(atype2i) - 1)
    state = torch.load(path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(state["state_dict"])
    reloaded.eval()
    model.eval()
    sample = targets[:32]
    with torch.inference_mode():
        _loss, finite_a, _n, pred_a = batch_step(model, None, sample, skill2i, atype2i)
        _loss, finite_b, _n, pred_b = batch_step(reloaded, None, sample, skill2i, atype2i)
    serial = []
    batched = []
    with torch.inference_mode():
        for row, by_end in sample[:8]:
            (skills, atypes), _prefix, _pilot = tokens_of(reloaded, skill2i, atype2i, row, by_end)
            one = float(reloaded.forward_from_state(torch.tensor([skills]), torch.tensor([atypes]), None)[0])
            serial.append(one)
        _loss, _finite, _n, pred = batch_step(reloaded, None, sample[:8], skill2i, atype2i)
        batched = [float(x) for x in pred]
    cal_users = sorted(cal_rows, key=lambda uid: hashlib.sha256(f"WSDM203-G7-CAL-PILOT-v1|{uid}".encode()).hexdigest())[:128]
    finite_scores = 0
    scored = 0
    for uid in cal_users[:32]:
        _by_start, by_end = chronorep.order_valid(cal_rows[uid])
        pairs = []
        for row in _by_start[:20]:
            (skills, atypes), _prefix, _pilot = tokens_of(reloaded, skill2i, atype2i, row, by_end)
            if skills:
                pairs.append((skills, atypes))
        if pairs:
            values = chronorep.score_forward(reloaded, pairs, torch.device("cpu"))
            finite_scores += int(np.isfinite(values).sum())
            scored += len(values)
    payload = {
        "label": "NON_PRIMARY_ATTENTION_SMOKE_TEST",
        "development_users": len(users),
        "training_examples": result["examples"],
        "epoch_loss": result["loss"],
        "nonfinite_losses": result["nonfinite"],
        "optimizer_steps": result["steps"],
        "reload_max_abs_diff": float((pred_a - pred_b).abs().max()) if finite_a and finite_b else None,
        "serial_batch_max_abs_diff": float(max(abs(a - b) for a, b in zip(serial, batched))) if serial else None,
        "calibration_scores": scored,
        "calibration_finite": finite_scores,
        "elapsed_sec": time.time() - started,
        "used_for_architecture_selection": False,
    }
    payload["pass"] = bool(
        result["loss"] is not None and np.isfinite(result["loss"]) and result["nonfinite"] == 0
        and payload["reload_max_abs_diff"] is not None and payload["reload_max_abs_diff"] <= 1e-5
        and payload["serial_batch_max_abs_diff"] is not None and payload["serial_batch_max_abs_diff"] <= 1e-5
        and scored > 0 and finite_scores == scored
    )
    dump("NON_PRIMARY_ATTENTION_SMOKE_TEST.json", payload)
    return payload


def fast_auroc(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    order = np.argsort(p, kind="mergesort")
    ys = y[order]
    ps = p[order]
    change = np.empty(len(p), dtype=bool)
    change[0] = True
    change[1:] = ps[1:] != ps[:-1]
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(p)
    ranks = np.repeat(0.5 * (starts + 1 + ends), ends - starts)
    n_pos = float(ys.sum())
    n_neg = float(len(y) - n_pos)
    return float((np.sum(ranks * ys) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def event_key(source_row: int) -> str:
    return hashlib.blake2s(f"WSDM203-G9|{source_row}".encode(), digest_size=16).hexdigest()


def independent_check(model, grouped, bundles, skill2i, atype2i) -> dict:
    pool = []
    for uid, rows in grouped.items():
        for row in rows:
            if row["valid"]:
                pool.append((hashlib.sha256(f"WSDM203-G11-EQ|{row['source_row']}".encode()).hexdigest(), uid, row))
    pool.sort()
    model.eval()
    max_diff = 0.0
    label_mismatch = defaultdict(int)
    order_mismatch = 0
    future = 0
    token_mismatch = 0
    current_leak = 0
    checked = 0
    alternatives = 0
    pilot = NeuralPilot()
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.model = model
    for _rank, uid, row in pool:
        if checked >= 200:
            break
        _by_start, by_end = chronorep.order_valid(grouped[uid])
        prefix = chronorep.prefix_events(by_end, row["start"])
        current = chronorep.as_event(row)
        left = pilot._tokens(prefix, current, False)
        cut = chronorep.history_cut(by_end, row["start"])
        full_s, full_a, _cut = matchedinfo.history_tokens(by_end, row["start"], skill2i, atype2i, model.outcome0, model.outcome1)
        actions = [atype2i.get(tok, 0) for tok in row["action_types"][:32]]
        sid = skill2i.get(row["skill_id"], 0)
        built = ((full_s + [sid] * len(actions))[-64:], (full_a + actions)[-64:])
        if built != left:
            token_mismatch += 1
        if any(item["end"] >= row["start"] for item in by_end[:cut]):
            future += 1
        length = min(32, len(row["action_types"]))
        if length and any(tok in (model.outcome0, model.outcome1) for tok in left[1][-length:]):
            current_leak += 1
        if not left[0]:
            continue
        with torch.inference_mode():
            serial = float(model.forward_from_state(torch.tensor([left[0]]), torch.tensor([left[1]]), None)[0])
            batched = float(chronorep.score_forward(model, [left], torch.device("cpu"))[0])
        max_diff = max(max_diff, abs(serial - batched))
        ev = chronorep.as_event(row)
        slot = bundles["S0"].get(chronorep.release_key(ev, "S0"))
        cands = []
        if slot is not None:
            own = row["action_types"][:32]
            others = [fp for fp in slot.gru if fp != own]
            if others:
                cands.append(others[0])
                alternatives += 1
        alt_actions = tuple(list(row["action_types"][:32])[::-1]) or ("hint",)
        cands.append(alt_actions)
        scores = []
        for actions in cands:
            alt = Event("", "", row["skill_id"], 0, 0, 0, "", actions, tuple(), "")
            seq = pilot._tokens(prefix, alt, False)
            with torch.inference_mode():
                a = float(model.forward_from_state(torch.tensor([seq[0]]), torch.tensor([seq[1]]), None)[0])
                b = float(chronorep.score_forward(model, [seq], torch.device("cpu"))[0])
            max_diff = max(max_diff, abs(a - b))
            scores.append(a)
        if len(scores) >= 2:
            rev = list(reversed(scores))
            if abs(min(scores) - min(rev)) > 1e-8 or abs(max(scores) - max(rev)) > 1e-8:
                order_mismatch += 1
        for tau in (0.70, 0.80, 0.90):
            if (serial >= tau) != (batched >= tau):
                label_mismatch[str(tau)] += 1
        checked += 1
    return {
        "events": checked,
        "alternatives_from_development_class": alternatives,
        "max_abs_diff": max_diff,
        "token_mismatches": token_mismatch,
        "current_outcome_leaks": current_leak,
        "future_or_unfinished": future,
        "order_interval_mismatches": order_mismatch,
        "action_label_mismatches": dict(label_mismatch),
        "pass": checked >= 200 and max_diff <= 1e-5 and not label_mismatch and future == 0 and current_leak == 0 and token_mismatch == 0,
    }


def certificate_rows(eval_out):
    status = eval_out["status"]
    model_i = 1
    valid = 939819
    frontier, false_rows, transitions, thresholds = [], [], [], []
    labels = ("unsupported", "stable", "set_valued")
    for schema_i, schema in enumerate(chronorep.SCHEMAS):
        for tau_i, tau in enumerate(chronorep.TAUS):
            unsup = int(status[:, model_i, schema_i, tau_i, 0].sum())
            stable = int(status[:, model_i, schema_i, tau_i, 1].sum())
            sett = int(status[:, model_i, schema_i, tau_i, 2].sum())
            if unsup + stable + sett != valid:
                raise SystemExit(f"accounting {schema} {tau}")
            fs = int(eval_out["false_stable"][:, model_i, schema_i].sum()) if abs(tau - 0.8) < 1e-9 else None
            nu = int(eval_out["uncovered_stable"][:, model_i, schema_i].sum()) if abs(tau - 0.8) < 1e-9 else None
            thresholds.append({
                "model": "CausalTokenAttentionKT", "schema": schema, "tau": tau, "valid_targets": valid,
                "unsupported": unsup, "stable": stable, "set_valued": sett, "event_nc": unsup + sett,
                "event_nc_rate": (unsup + sett) / valid, "false_stable": fs, "n_U": nu,
            })
            if abs(tau - 0.8) < 1e-9:
                frontier.append({
                    "model": "CausalTokenAttentionKT", "schema": schema, "tau": 0.8, "valid_targets": valid,
                    "invalid_excluded": 91, "unsupported": unsup, "stable": stable, "set_valued": sett,
                    "supported": stable + sett, "event_nc": unsup + sett, "event_nc_over_valid": (unsup + sett) / valid,
                    "fingerprint_covered_supported": int(eval_out["covered"][:, model_i, schema_i].sum()),
                    "n_S": stable, "n_U": nu, "n_FS": fs,
                    "learners_with_supported_event": int(np.sum(eval_out["supported_n"][:, model_i, schema_i] > 0)),
                    "consumer_fingerprints_summed_over_events": int(eval_out["fingerprints"][model_i, schema_i]),
                    "rich_signatures_summed_over_events": int(eval_out["signatures"][model_i, schema_i]),
                })
                false_rows.append({
                    "model": "CausalTokenAttentionKT", "schema": schema, "tau": 0.8,
                    "n_S": stable, "n_U": nu, "n_FS": fs,
                    "fingerprint_covered_supported": int(eval_out["covered"][:, model_i, schema_i].sum()),
                    "n_FS_over_n_S": fs / stable if stable else None,
                    "n_U_over_n_S": nu / stable if stable else None,
                    "n_FS_over_n_U": (fs / nu) if nu else None,
                    "second_factor": "N/A" if stable and nu == 0 else (fs / nu if nu else None),
                })
    for pair in (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3")):
        for tau in chronorep.TAUS:
            matrix = eval_out["transitions"][f"ActionTokenGRU|{tau}|{pair[0]}|{pair[1]}"]
            if int(matrix.sum()) != valid:
                raise SystemExit(f"transition {pair} {tau}")
            for i, src in enumerate(labels):
                for j, dst in enumerate(labels):
                    transitions.append({
                        "model": "CausalTokenAttentionKT", "tau": tau, "from_schema": pair[0], "to_schema": pair[1],
                        "from_label": src, "to_label": dst, "n": int(matrix[i, j]), "denominator": valid,
                    })
    return frontier, false_rows, transitions, thresholds


def capacity_tables(eval_out):
    native, matched, risk_rows, tie_rows, member_transitions = [], [], [], [], []
    users = np.asarray(eval_out["users"], dtype=object)
    for kind, store_name in (("completion_time_native", "completion"), ("matched_event", "matched")):
        masks = []
        label_map = {}
        for schema in chronorep.SCHEMAS:
            cell = eval_out[store_name]["ActionTokenGRU"][schema]
            mask = np.isfinite(cell["lo"]) & np.isfinite(cell["hi"])
            masks.append(mask)
            counts = chronorep.membership_counts(cell["lo"][mask], cell["hi"][mask], 0.05)
            risk = chronorep.risk_block("CausalTokenAttentionKT", schema, 0.05, cell["lo"][mask], cell["hi"][mask], cell["actual"][mask], kind)
            row = {
                "analysis": kind, "model": "CausalTokenAttentionKT", "schema": schema, "kappa": 0.05,
                "tie_rule": "adversarial", "n_full": 6766, "n_eligible": int(mask.sum()),
                "n_no_checkpoint": 6766 - int(mask.sum()), "B": counts["B"],
                "guaranteed_in": counts["guaranteed_in"], "guaranteed_out": counts["guaranteed_out"],
                "unresolved": counts["unresolved"],
                "all_user_capacity_nc": (6766 - int(mask.sum()) + counts["unresolved"]) / 6766,
            }
            (native if kind.startswith("completion") else matched).append(row)
            risk_rows.append({
                "analysis": kind, "model": "CausalTokenAttentionKT", "schema": schema, "kappa": 0.05,
                "n": int(mask.sum()), "inside": risk["inside"], "outside": risk["outside"],
                "false_guaranteed_in": risk["false_guaranteed_in"], "false_guaranteed_out": risk["false_guaranteed_out"],
            })
            if kind == "matched_event":
                label_map[schema] = counts["labels"]
            ids = users[mask]
            adv = counts["labels"]
            uid_labels = chronorep.labels_from_ranks(*chronorep.userid_ranks(ids, cell["lo"][mask], cell["hi"][mask]), counts["B"])
            tie_rows.append({
                "analysis": kind, "schema": schema, "kappa": 0.05, "n_eligible": int(mask.sum()),
                "membership_changes_vs_userid": int(np.sum(adv != uid_labels)),
            })
        if kind == "matched_event":
            if any(not np.array_equal(masks[0], item) for item in masks[1:]):
                raise SystemExit("matched masks differ")
            seats = label_map["S0"]
            for left, right in (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3")):
                for src in ("guaranteed_in", "guaranteed_out", "unresolved"):
                    for dst in ("guaranteed_in", "guaranteed_out", "unresolved"):
                        member_transitions.append({
                            "model": "CausalTokenAttentionKT", "kappa": 0.05, "from_schema": left, "to_schema": right,
                            "from_label": src, "to_label": dst,
                            "n": int(np.sum((label_map[left] == src) & (label_map[right] == dst))),
                            "denominator": int(len(label_map["S0"])),
                        })
            del seats
    return native, matched, risk_rows, tie_rows, member_transitions


def write_checksums() -> None:
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/attention_consumer/output_checksums.json"], "n_files": len(files), "files": files})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    pre = chronorep.preflight()
    chronorep_path = CHRONOREP_DIR / "gru_chronological_seed17.pt"
    pre["chronorep_checkpoint_sha256"] = chronorep.sha256_file(chronorep_path)
    pre["chronorep_checkpoint_match"] = pre["chronorep_checkpoint_sha256"] == CHRONOREP_CKPT_SHA
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"] and pre["parquet"]["development"]["sha256"] == "69b6b7b38738d293651b728326658c1ce517c87661d012bc03ee2ced228e9d70" and pre["parquet"]["confirmatory"]["sha256"] == "0193ddd7b26517f1b48591d09d86f5f3ce8b9930a1285006833fce4ea5096f99")
    dump("source_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_INPUT_CONTRACT"})
        write_checksums()
        raise SystemExit(2)
    checkpoint = torch.load(chronorep_path, map_location="cpu", weights_only=False)
    skill2i = checkpoint["skill2i"]
    atype2i = checkpoint["atype2i"]
    del checkpoint
    n_skills = len(skill2i) - 1
    n_atypes = len(atype2i) - 1
    if n_skills != 264 or n_atypes != 8:
        raise SystemExit("vocabulary mismatch")
    print("preflight_pass", flush=True)
    tests = mask_tests(n_skills, n_atypes)
    dump("attention_mask_and_token_qa.json", {"synthetic_mask_tests": tests})
    if not tests["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_ARCHITECTURE_OR_MASKING", "tests": tests})
        write_checksums()
        raise SystemExit(3)
    probe = make_model(n_skills, n_atypes, 17)
    param_count = int(sum(p.numel() for p in probe.parameters()))
    literature = {
        "proposed_name": "CausalTokenAttentionKT",
        "also_called": "Causal Token-Attention Consumer",
        "not_claimed": ["SAKT", "AKT", "SAINT"],
        "rows": [
            {"model": "SAKT (Pandey and Karypis, 2019)", "task": "predict the response to the next exercise from past interactions", "current_actions": False, "attention": "self-attention over past interactions with the current exercise as query", "embeddings": "exercise and interaction embeddings"},
            {"model": "AKT (Ghosh, Heffernan, and Lan, KDD 2020)", "task": "predict a response from past context and the current question", "current_actions": False, "attention": "context-aware monotonic attention with distance decay", "embeddings": "Rasch-based question and response embeddings"},
            {"model": "CausalTokenAttentionKT", "task": "P(current correctness | completed history, current skill, current action tokens)", "current_actions": True, "attention": "two-layer causal Transformer encoder with a standard triangular mask", "embeddings": "sum of frozen-vocabulary skill and action embeddings plus learned positions"},
        ],
    }
    dump("literature_and_architecture_fidelity.json", literature)
    freeze = {
        "frozen_before_confirmatory_scores": True,
        "architecture": ARCH,
        "n_skills": n_skills,
        "n_action_types": n_atypes,
        "atype_vocabulary": atype2i,
        "outcome_token_ids": [probe.outcome0, probe.outcome1],
        "parameter_count": param_count,
        "parameter_match_to_gru": False,
        "optimizer_match_to_gru": False,
        "reason": "Different parameter count and minibatch updates. The shared contract is the chronological input and the current-event correctness task.",
        "mask_tests_pass": True,
        "script_sha256": chronorep.sha256_file(Path(__file__).resolve()),
    }
    dump("architecture_protocol_freeze.json", freeze)
    del probe
    print("architecture_frozen", param_count, flush=True)
    frame, first_seen = chronorep.load_order("development")
    if frame["user_id"].nunique() != 31753 or len(frame) != 4286981:
        raise SystemExit("development population")
    del frame
    selection = chronorep.replay_selection(first_seen)
    if selection["selection_sha256"] != SELECTION_SHA or selection["union"] != 5715 or selection["intersection"] != 285:
        dump("training_selection_audit.json", {"status": "MISMATCH", "selection": {k: selection[k] for k in ("epoch_sizes", "union", "intersection", "selection_sha256")}})
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_INPUT_CONTRACT"})
        write_checksums()
        raise SystemExit(4)
    training_rows = load_rows("development", selection["selected_users"])
    epochs = matchedinfo.epoch_targets(training_rows, selection["cohorts"])
    counts = [len(item["targets"]) for item in epochs]
    if counts != [90950, 90942]:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_INPUT_CONTRACT", "counts": counts})
        write_checksums()
        raise SystemExit(5)
    dump("training_selection_audit.json", {
        "selection_sha256": selection["selection_sha256"], "epoch_sizes": selection["epoch_sizes"],
        "union": selection["union"], "intersection": selection["intersection"], "valid_examples_by_epoch": counts,
    })
    qa_model = make_model(n_skills, n_atypes, 17)
    qa = run_token_qa(qa_model, skill2i, atype2i, epochs[0]["targets"] + epochs[1]["targets"], training_rows)
    prior = json.loads((OUT / "attention_mask_and_token_qa.json").read_text())
    prior["development_token_qa"] = qa
    prior["pass"] = bool(prior["synthetic_mask_tests"]["pass"] and qa["pass"])
    dump("attention_mask_and_token_qa.json", prior)
    del qa_model
    if not qa["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_ARCHITECTURE_OR_MASKING", "qa": qa})
        write_checksums()
        raise SystemExit(6)
    print("token_qa_pass", qa["cases"], flush=True)
    smoke_dev_ids = sorted(first_seen, key=lambda uid: hashlib.sha256(f"WSDM203-G7-DEV-PILOT-v1|{uid}".encode()).hexdigest())[:128]
    missing = [uid for uid in smoke_dev_ids if uid not in training_rows]
    if missing:
        training_rows.update(load_rows("development", set(missing)))
    cal_ids_frame = pd.read_parquet(chronorep.RECON / "calibration.parquet", columns=["user_id"])
    cal_ids = sorted(set(cal_ids_frame["user_id"].astype(str)))
    del cal_ids_frame
    smoke_cal_ids = sorted(cal_ids, key=lambda uid: hashlib.sha256(f"WSDM203-G7-CAL-PILOT-v1|{uid}".encode()).hexdigest())[:128]
    cal_rows = load_rows("calibration", set(smoke_cal_ids))
    smoke_model = make_model(n_skills, n_atypes, 17)
    smoke_result = smoke(smoke_model, skill2i, atype2i, {uid: training_rows[uid] for uid in smoke_dev_ids}, cal_rows)
    del smoke_model, cal_rows
    if not smoke_result["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_COMPUTATIONAL_EXECUTION", "smoke": smoke_result})
        write_checksums()
        raise SystemExit(7)
    print("smoke_pass", flush=True)
    primary = OUT / "causal_token_attention_seed17.pt"
    progress = OUT / "causal_token_attention_seed17.progress.pt"
    if primary.exists():
        state = torch.load(primary, map_location="cpu", weights_only=False)
        log = state["log"]
        model = CausalTokenAttentionKT(n_skills, n_atypes)
        model.load_state_dict(state["state_dict"])
        print("resume_primary", flush=True)
    else:
        model = make_model(n_skills, n_atypes, 17)
        if int(sum(p.numel() for p in model.parameters())) != param_count:
            raise SystemExit("parameter count changed after freeze")
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        logs = []
        train_started = time.time()
        for epoch_i, epoch in enumerate(epochs, start=1):
            stats = train_epoch(model, opt, epoch["targets"], skill2i, atype2i, f"primary-epoch-{epoch_i}", progress, 0, {})
            logs.append(stats)
            print(f"epoch_done {epoch_i} {stats}", flush=True)
            if progress.exists():
                progress.unlink()
        model.eval()
        sample = epochs[0]["targets"][-1]
        (skills, atypes), _prefix, _pilot = tokens_of(model, skill2i, atype2i, sample[0], sample[1])
        with torch.inference_mode():
            a = float(model.forward_from_state(torch.tensor([skills]), torch.tensor([atypes]), None)[0])
            b = float(model.forward_from_state(torch.tensor([skills]), torch.tensor([atypes]), None)[0])
        log = {
            "seed": 17, "examples_by_epoch": [item["examples"] for item in logs],
            "loss_by_epoch": [item["loss"] for item in logs], "optimizer_steps_by_epoch": [item["steps"] for item in logs],
            "nonfinite_losses_by_epoch": [item["nonfinite"] for item in logs], "batch_size": 64,
            "parameter_count": param_count, "architecture": ARCH, "max_sequence_length": 64,
            "deterministic_repeat_abs_diff": abs(a - b), "elapsed_sec": time.time() - train_started,
            "rss_mb": chronorep.rss_mb(), "warm_start": False,
        }
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "log": log, "config": ARCH}, primary)
        log["checkpoint_sha256"] = chronorep.sha256_file(primary)
        log["checkpoint_bytes"] = primary.stat().st_size
        reloaded = CausalTokenAttentionKT(n_skills, n_atypes)
        reloaded.load_state_dict(torch.load(primary, map_location="cpu", weights_only=False)["state_dict"])
        reloaded.eval()
        with torch.inference_mode():
            c = float(reloaded.forward_from_state(torch.tensor([skills]), torch.tensor([atypes]), None)[0])
        log["reload_abs_diff"] = abs(a - c)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "log": log, "config": ARCH}, primary)
        log["checkpoint_sha256"] = chronorep.sha256_file(primary)
        log["checkpoint_bytes"] = primary.stat().st_size
        if not np.isfinite(log["loss_by_epoch"]).all() or log["deterministic_repeat_abs_diff"] > 1e-5 or log["reload_abs_diff"] > 1e-5:
            dump("attention_training_log.json", log)
            dump("numerical_invariants.json", {"status": "BLOCKED_COMPUTATIONAL_EXECUTION", "log": log})
            write_checksums()
            raise SystemExit(8)
    dump("attention_training_log.json", log)
    print("training_complete", log["loss_by_epoch"], flush=True)
    del training_rows, epochs
    model.eval()
    grouped = matchedinfo.confirmatory_rows()
    valid = sum(row["valid"] for rows in grouped.values() for row in rows)
    if valid != 939819 or len(grouped) != 6766:
        raise SystemExit(f"confirmatory {valid} {len(grouped)}")
    geometry = chronorep.build_development(selection["selected_users"])
    plans = chronorep.plan_checkpoints(grouped, geometry["bundles"])
    check = independent_check(model, grouped, geometry["bundles"], skill2i, atype2i)
    dump("independent_scoring_equivalence.json", check)
    if not check["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "check": check})
        write_checksums()
        raise SystemExit(9)
    print("independent_check_pass", flush=True)
    pilot = NeuralPilot()
    pilot.skill2i = skill2i
    pilot.atype2i = atype2i
    pilot.model = model
    pilot.device = torch.device("cpu")
    model.eval()
    print("certificate_eval", flush=True)
    evaluated = chronorep.evaluate(pilot, grouped, geometry["bundles"], plans["plans"])
    if evaluated["events_seen"] != 939819 or evaluated["future_violations"] or evaluated["nested_violations"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "events": evaluated["events_seen"], "future": evaluated["future_violations"], "nested": evaluated["nested_violations"]})
        write_checksums()
        raise SystemExit(10)
    pfa_stable = int(evaluated["status"][:, 0, 0, chronorep.TAUS.index(0.8), 1].sum())
    if pfa_stable != 916845:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "pfa_s0_stable": pfa_stable})
        write_checksums()
        raise SystemExit(11)
    y = np.asarray(evaluated["pred_y"])
    p = np.asarray(evaluated["pred_gru"])
    metrics = chronorep.binary_metrics(y, p)
    predictive = {
        "model": "CausalTokenAttentionKT", "seed": 17, "n": int(len(y)), "n_learners": 6766,
        "base_rate": metrics["positive_rate"], "auroc": metrics["roc_auc"], "log_loss": metrics["log_loss"], "brier": metrics["brier"],
        "accuracy_at_0.5": float(np.mean((p >= 0.5) == y)), "mean_prediction": float(np.mean(p)),
        "nonfinite": int(np.sum(~np.isfinite(p))), "estimand": "action-conditioned current-event correctness under the chronological token contract",
    }
    dump("confirmatory_predictive_metrics.json", predictive)
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    pd.DataFrame([
        {"bin": b, "lo": b / 10, "hi": (b + 1) / 10, "n": int((bins == b).sum()),
         "mean_prediction": float(p[bins == b].mean()) if np.any(bins == b) else None,
         "observed_frequency": float(y[bins == b].mean()) if np.any(bins == b) else None}
        for b in range(10)
    ]).to_csv(OUT / "predictive_calibration_bins.csv", index=False)
    users = sorted(grouped)
    keys = []
    for uid in users:
        by_start, _by_end = chronorep.order_valid(grouped[uid])
        for row in by_start:
            keys.append(event_key(row["source_row"]))
    if len(keys) != len(y):
        raise SystemExit("score order")
    table = pa.table({"event_key": keys, "user_index": evaluated["pred_user"], "correct": y, "score": p})
    pq.write_table(table, CACHE / "attention_event_scores.parquet")
    np.savez_compressed(CACHE / "attention_user_status.npz", status=evaluated["status"][:, 1])
    frontier, false_rows, transitions, thresholds = certificate_rows(evaluated)
    for row in false_rows:
        if row["n_U"] and row["n_S"] and abs(row["n_FS"] / row["n_S"] - (row["n_U"] / row["n_S"]) * (row["n_FS"] / row["n_U"])) > 1e-9:
            raise SystemExit("false-stable identity")
    pd.DataFrame(frontier).to_csv(OUT / "event_certificate_frontier.csv", index=False)
    pd.DataFrame(false_rows).to_csv(OUT / "fingerprint_coverage_and_false_stable.csv", index=False)
    pd.DataFrame(transitions).to_csv(OUT / "paired_event_transitions.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "threshold_sensitivity.csv", index=False)
    native, matched, risk_rows, tie_rows, member_transitions = capacity_tables(evaluated)
    pd.DataFrame(native).to_csv(OUT / "native_completion_time_capacity.csv", index=False)
    pd.DataFrame(matched).to_csv(OUT / "matched_event_capacity.csv", index=False)
    pd.DataFrame(member_transitions).to_csv(OUT / "matched_capacity_transitions.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(OUT / "realized_risk_validity.csv", index=False)
    pd.DataFrame(tie_rows).to_csv(OUT / "userid_tie_sensitivity.csv", index=False)
    chronorep_pred = pd.read_csv(CHRONOREP_DIR / "predictive_diagnostics.csv").iloc[0]
    seedrob = pd.read_csv(SEEDROB_DIR / "act_primary_predictive_metrics_by_seed.csv")
    matchedinfom = pd.read_csv(MATCHEDINFO_DIR / "confirmatory_predictive_metrics.csv")
    matchedinfo_act = matchedinfom[matchedinfom.model == "ACT"].iloc[0]
    comparison = [
        {"model": "CausalTokenAttentionKT", "seed": 17, "interface": "ACT_PRIMARY", "auroc": predictive["auroc"], "log_loss": predictive["log_loss"], "brier": predictive["brier"], "comparable_as_same_input": True},
        {"model": "ActionTokenGRU", "seed": 17, "interface": "ACT_PRIMARY", "auroc": float(chronorep_pred["roc_auc"]), "log_loss": float(chronorep_pred["log_loss"]), "brier": float(chronorep_pred["brier"]), "comparable_as_same_input": True},
    ]
    for _i, row in seedrob.iterrows():
        comparison.append({"model": "ActionTokenGRU", "seed": int(row["seed"]), "interface": "ACT_PRIMARY", "auroc": float(row["auroc"]), "log_loss": float(row["log_loss"]), "brier": float(row["brier"]), "comparable_as_same_input": True})
    comparison.append({"model": "ActionTokenGRU", "seed": 17, "interface": "ACT_MATCHED", "auroc": float(matchedinfo_act["auroc"]), "log_loss": float(matchedinfo_act["log_loss"]), "brier": float(matchedinfo_act["brier"]), "comparable_as_same_input": False, "note": "Fixed 32-token history. Descriptive only."})
    pd.DataFrame(comparison).to_csv(OUT / "historical_gru_comparison.csv", index=False)
    print("tables_written", predictive["auroc"], flush=True)
    status_u = evaluated["status"][:, 1]
    primary_tau = chronorep.TAUS.index(0.8)
    rng = np.random.default_rng(20311)
    draws = rng.integers(0, 6766, size=(2000, 6766))
    boot = []
    for schema_i, schema in enumerate(chronorep.SCHEMAS):
        unsup = status_u[:, schema_i, primary_tau, 0]
        sett = status_u[:, schema_i, primary_tau, 2]
        den = status_u[:, schema_i, primary_tau, :].sum(axis=1)
        point = float((unsup + sett).sum() / den.sum())
        reps = np.empty(2000)
        for i, draw in enumerate(draws):
            reps[i] = (unsup[draw] + sett[draw]).sum() / den[draw].sum()
        boot.append({"analysis": "event_nc_rate", "schema": schema, "point": point, "ci95_low": float(np.percentile(reps, 2.5)), "ci95_high": float(np.percentile(reps, 97.5)), "replicates": 2000, "seed_rng": 20311, "estimand": "learner resampling conditional on the single attention checkpoint"})
    unsup0 = status_u[:, 0, primary_tau, 0] + status_u[:, 0, primary_tau, 2]
    den0 = status_u[:, 0, primary_tau, :].sum(axis=1)
    unsup1 = status_u[:, 1, primary_tau, 0] + status_u[:, 1, primary_tau, 2]
    den1 = status_u[:, 1, primary_tau, :].sum(axis=1)
    point_delta = float(unsup1.sum() / den1.sum() - unsup0.sum() / den0.sum())
    reps = np.empty(2000)
    for i, draw in enumerate(draws):
        reps[i] = unsup1[draw].sum() / den1[draw].sum() - unsup0[draw].sum() / den0[draw].sum()
    boot.append({"analysis": "event_nc_s1_minus_s0", "schema": "S1-S0", "point": point_delta, "ci95_low": float(np.percentile(reps, 2.5)), "ci95_high": float(np.percentile(reps, 97.5)), "replicates": 2000, "seed_rng": 20311, "estimand": "learner resampling conditional on the single attention checkpoint"})
    user_index = evaluated["pred_user"]
    buckets = [[] for _ in range(6766)]
    for index, user in enumerate(user_index.tolist()):
        buckets[user].append(index)
    indexes = [np.asarray(bucket, dtype=np.int32) for bucket in buckets]
    y_blocks = [y[idx].astype(float) for idx in indexes]
    p_blocks = [np.clip(p[idx], 1e-12, 1 - 1e-12) for idx in indexes]
    gru_blocks = {}
    for seed in (29, 41):
        frame_g = pd.read_parquet(SEEDROB_DIR / "private_score_caches" / f"act_primary_seed{seed}.parquet")
        if list(frame_g.event_key) != keys:
            print("gru_cache_key_mismatch", seed, flush=True)
            continue
        gp = frame_g.score.to_numpy()
        gru_blocks[seed] = [np.clip(gp[idx], 1e-12, 1 - 1e-12) for idx in indexes]
    max_n = max(len(block) for block in y_blocks)
    y_buffer = np.empty(max_n * 6766)
    p_buffer = np.empty_like(y_buffer)
    g_buffer = np.empty_like(y_buffer)
    auc_reps = np.empty(2000)
    pair_reps = {seed: np.empty(2000) for seed in gru_blocks}
    for rep, draw in enumerate(draws):
        pos = 0
        for user_i in draw:
            block = y_blocks[user_i]
            n = len(block)
            if n:
                y_buffer[pos:pos + n] = block
                p_buffer[pos:pos + n] = p_blocks[user_i]
                pos += n
        auc_reps[rep] = fast_auroc(y_buffer[:pos], p_buffer[:pos])
        for seed, blocks_g in gru_blocks.items():
            cursor = 0
            for user_i in draw:
                block = blocks_g[user_i]
                n = len(block)
                if n:
                    g_buffer[cursor:cursor + n] = block
                    cursor += n
            pair_reps[seed][rep] = auc_reps[rep] - fast_auroc(y_buffer[:pos], g_buffer[:cursor])
        if rep and rep % 100 == 0:
            print(f"bootstrap {rep}", flush=True)
    boot.append({"analysis": "auroc", "schema": "predictive", "point": predictive["auroc"], "ci95_low": float(np.percentile(auc_reps, 2.5)), "ci95_high": float(np.percentile(auc_reps, 97.5)), "replicates": 2000, "seed_rng": 20311, "estimand": "learner resampling conditional on the single attention checkpoint"})
    for seed, values in pair_reps.items():
        point_pair = predictive["auroc"] - float(seedrob.loc[seedrob.seed == seed, "auroc"].iloc[0])
        boot.append({"analysis": "auroc_attention_minus_act_primary", "schema": f"seed_{seed}", "point": point_pair, "ci95_low": float(np.percentile(values, 2.5)), "ci95_high": float(np.percentile(values, 97.5)), "replicates": 2000, "seed_rng": 20311, "estimand": "learner resampling conditional on these two checkpoints; seed 17 GRU event scores were not retained"})
    pd.DataFrame(boot).to_csv(OUT / "learner_bootstrap_intervals.csv", index=False)
    end_chronorep = chronorep.sha256_file(chronorep_path)
    invariants = {
        "status": "COMPLETE",
        "parameter_count": param_count,
        "mask_tests": tests,
        "token_qa_cases": qa["cases"],
        "token_qa_violations": qa["violations"],
        "smoke_pass": smoke_result["pass"],
        "independent_check": check,
        "pfa_s0_stable": pfa_stable,
        "nested_violations": evaluated["nested_violations"],
        "future_violations": evaluated["future_violations"],
        "chronorep_checkpoint_unchanged": end_chronorep == CHRONOREP_CKPT_SHA,
        "events": int(evaluated["events_seen"]),
        "elapsed_sec": time.time() - started,
    }
    dump("numerical_invariants.json", invariants)
    dump("execution_progress.json", {"status": invariants["status"]})
    dump("experiment_manifest.json", {
        "experiment": "pipeline", "script": "scripts/assistments/attention_consumer.py",
        "script_sha256": chronorep.sha256_file(Path(__file__).resolve()), "status": invariants["status"],
        "model": "CausalTokenAttentionKT", "seed": 17,
    })
    write_checksums()
    print(invariants["status"], flush=True)


if __name__ == "__main__":
    main()
