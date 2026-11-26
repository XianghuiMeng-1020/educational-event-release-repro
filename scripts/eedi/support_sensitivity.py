#!/usr/bin/env python3
"""Eedi support-predicate sensitivity under fixed baseline consumers.

Scores each held-out event once on the full development-observed option set.
Support rules only mask that set. They do not change histories, weights, or options.

Writes only under results/eedi/support_sensitivity/.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r3"))

from run_r3_local_conditional import (  # noqa: E402
    DATA_PATH,
    N_QUESTIONS,
    build_compatible_dict,
    load_frame,
    load_models,
)
from src.identification.compatible_trace.capacity_identified_set import (
    certify_capacity_membership,
)
from src.identification.compatible_trace.dkt_propagation import _next_step_proba_from_state
from src.identification.compatible_trace.pfa_propagation import (
    PFAState,
    pfa_ingest_event,
    pfa_next_step_proba,
)
from src.models.categorical_dkt import interaction_token
from src.policies.capacity_support_v2 import select_capacity_list

OUT = ROOT / "results" / "eedi" / "support_sensitivity"
SEED = 203
N_BOOT = 2000
TAUS = (0.70, 0.80, 0.90)
KAPPAS = (0.01, 0.05, 0.10)
N_FRAME_EXPECTED = 1021
EXPECTED = {
    "supported": 138039,
    "unsupported": 156387,
    "pfa_stable": 138028,
    "pfa_set": 11,
    "dkt_stable": 132736,
    "dkt_set": 5303,
    "actual_in": 138036,
    "pfa_fs": 0,
    "dkt_fs": 0,
    "eligible": 1018,
    "B": 51,
    "pfa_gin": 49,
    "pfa_gout": 966,
    "pfa_amb": 3,
    "dkt_gin": 1,
    "dkt_gout": 729,
    "dkt_amb": 288,
}

# Collision-preserving grid, original cell first.
COLLISION = []
for min_users in (2, 5, 10, 20):
    for min_per in (1, 2, 5):
        COLLISION.append(
            {
                "family": "collision",
                "min_options": 2,
                "min_users": min_users,
                "min_users_per_option": min_per,
                "rule_id": f"C_u{min_users}_p{min_per}",
            }
        )
SINGLETON = [
    {
        "family": "singleton_permissive",
        "min_options": 1,
        "min_users": u,
        "min_users_per_option": 1,
        "rule_id": f"S_u{u}_p1",
    }
    for u in (2, 5, 10, 20)
]
RULES = COLLISION + SINGLETON


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_output_checksums(out_dir: Path | None = None) -> dict[str, Any]:
    """Hash completed outputs. The checksum file is excluded, so it has no self-entry."""
    out_dir = out_dir or OUT
    excluded_name = "output_checksums.json"
    rows = []
    for path in sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != excluded_name):
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "algorithm": "sha256",
        "excluded": [f"{out_dir.resolve().relative_to(ROOT.resolve()).as_posix()}/{excluded_name}"],
        "n_files": len(rows),
        "files": rows,
    }
    (out_dir / excluded_name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def build_dev_stats(dev: pd.DataFrame) -> dict[tuple, dict[str, Any]]:
    out: dict[tuple, dict[str, Any]] = {}
    groups = dev.groupby(["QuestionId", "SubjectId", "IsCorrect"], sort=False)
    for (q, s, y), g in groups:
        key = ("EEDI-R0", str(q), str(s), int(y))
        opts = sorted({int(v) for v in g["AnswerValue"].dropna().astype(int).tolist()})
        users = {str(u) for u in g["UserId"].tolist()}
        users_by_opt = {
            o: {str(u) for u in g.loc[g["AnswerValue"].astype(int) == o, "UserId"]}
            for o in opts
        }
        counts = [len(users_by_opt[o]) for o in opts]
        out[key] = {
            "options": tuple(opts),
            "n_events": int(len(g)),
            "n_users": int(len(users)),
            "n_options": int(len(opts)),
            "min_per": int(min(counts)) if counts else 0,
            "max_per": int(max(counts)) if counts else 0,
            "y": int(y),
            "q": int(q),
            "subject": str(s),
        }
    return out


def original_mask(st: dict[str, Any]) -> bool:
    return st["n_options"] >= 2 and st["n_users"] >= 2 and st["min_per"] >= 1


def rule_mask_for(st: dict[str, Any]) -> int:
    bits = 0
    for i, rule in enumerate(RULES):
        ok = (
            st["n_options"] >= rule["min_options"]
            and st["n_users"] >= rule["min_users"]
            and st["min_per"] >= rule["min_users_per_option"]
        )
        if ok:
            bits |= 1 << i
    return bits


def answer_key_map(dev: pd.DataFrame) -> dict[int, int]:
    """0 unique, 1 conflicting, 2 unavailable. Development rows only."""
    cats: dict[int, int] = {}
    for q, g in dev.groupby("QuestionId", sort=False):
        vals = {int(v) for v in g["CorrectAnswer"].dropna().tolist()}
        if len(vals) == 1:
            cats[int(q)] = 0
        elif len(vals) > 1:
            cats[int(q)] = 1
        else:
            cats[int(q)] = 2
    return cats


def dkt_option_scores(model, h, c, q: int, options: tuple[int, ...]) -> np.ndarray:
    device = h.device
    toks = [
        interaction_token(q, representation="EEDI-R1", answer_value=int(a), n_questions=N_QUESTIONS)
        for a in options
    ]
    tok_t = torch.tensor(toks, dtype=torch.long, device=device)
    # LSTM is batch_first: (batch, seq, dim) = (n, 1, dim). Hidden is (layers, batch, hidden).
    emb = model.embedding(tok_t).unsqueeze(1)
    n = len(options)
    hb = h.expand(1, n, -1).contiguous()
    cb = c.expand(1, n, -1).contiguous()
    _, (h2, _c2) = model.lstm(emb, (hb, cb))
    logit = model.output(model.dropout(h2.squeeze(0))).squeeze(-1)
    return torch.sigmoid(logit).reshape(-1).detach().cpu().numpy().astype(np.float64)


def dkt_one_score(model, h, c, q: int, av: int) -> float:
    device = h.device
    tok = interaction_token(q, representation="EEDI-R1", answer_value=int(av), n_questions=N_QUESTIONS)
    emb = model.embedding(torch.tensor([[tok]], dtype=torch.long, device=device))
    _, (h2, _c2) = model.lstm(emb, (h, c))
    return float(_next_step_proba_from_state(model, h2, _c2))


def status_from_minmax(lo: float, hi: float, tau: float, n_opt: int) -> str:
    if n_opt <= 0 or not np.isfinite(lo):
        return "unsupported"
    if lo >= tau or hi < tau:
        return "stable"
    return "set-valued"


def false_stable(lo: float, hi: float, true_s: float, tau: float, n_opt: int) -> int:
    if status_from_minmax(lo, hi, tau, n_opt) != "stable":
        return 0
    cert_progress = lo >= tau
    true_progress = true_s >= tau
    return int(cert_progress != true_progress)


def contained(lo: float, hi: float, true_s: float, tau: float, n_opt: int) -> int:
    st = status_from_minmax(lo, hi, tau, n_opt)
    if st == "set-valued":
        return 1
    if st != "stable":
        return 0
    return int(false_stable(lo, hi, true_s, tau, n_opt) == 0)


def user_band(n_users: int) -> str:
    if n_users < 2:
        return "lt2"
    if n_users <= 4:
        return "2-4"
    if n_users <= 9:
        return "5-9"
    if n_users <= 19:
        return "10-19"
    return "20+"


def per_band(min_per: int) -> str:
    if min_per <= 0:
        return "0"
    if min_per == 1:
        return "1"
    if min_per <= 4:
        return "2-4"
    return "5+"


def capacity_one(user_ids: list[str], los: list[float], his: list[float], actual: list[float], kappa: float, n_frame: int) -> dict[str, Any]:
    n_el = len(user_ids)
    no_ckpt = n_frame - n_el
    if n_el == 0:
        return {
            "n_frame": n_frame,
            "n_eligible": 0,
            "n_no_checkpoint": no_ckpt,
            "B": 0,
            "guaranteed_in": 0,
            "guaranteed_out": 0,
            "unresolved": 0,
            "all_user_nc": no_ckpt / n_frame,
            "false_gin": 0,
            "false_gout": 0,
        }
    mem = certify_capacity_membership(user_ids, los, his, capacity=kappa)
    k = mem[0].k
    counts = {"guaranteed_in": 0, "guaranteed_out": 0, "ambiguous": 0}
    statuses = {}
    for m in mem:
        counts[m.status] += 1
        statuses[m.user_id] = m.status
    arr_u = np.array(user_ids, dtype=object)
    arr_r = np.array(actual, dtype=np.float64)
    selected = set(select_capacity_list(arr_u, arr_r, capacity=kappa))
    false_gin = sum(1 for u in user_ids if statuses[u] == "guaranteed_in" and u not in selected)
    false_gout = sum(1 for u in user_ids if statuses[u] == "guaranteed_out" and u in selected)
    unresolved = counts["ambiguous"]
    return {
        "n_frame": n_frame,
        "n_eligible": n_el,
        "n_no_checkpoint": no_ckpt,
        "B": int(k),
        "guaranteed_in": counts["guaranteed_in"],
        "guaranteed_out": counts["guaranteed_out"],
        "unresolved": unresolved,
        "all_user_nc_num": no_ckpt + unresolved,
        "all_user_nc": (no_ckpt + unresolved) / n_frame,
        "false_gin": false_gin,
        "false_gout": false_gout,
        "tie_rule": "ascending_user_id_string",
        "membership_method": "interval_robust_rank_conservative",
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("load", flush=True)
    df = load_frame()
    splits = {}
    for name, g in df.groupby("split", sort=False):
        splits[str(name)] = {"users": int(g["UserId"].nunique()), "events": int(len(g))}
    overlap = 0
    seen: set[str] = set()
    for uid, g in df.groupby("UserId", sort=False):
        labs = set(g["split"].astype(str))
        if len(labs) != 1:
            overlap += 1
        if str(uid) in seen:
            overlap += 1
        seen.add(str(uid))
    dev = df.loc[df["split"] == "development"]
    conf = df.loc[df["split"] == "confirmatory"]
    print("dev stats", flush=True)
    stats = build_dev_stats(dev)
    orig = build_compatible_dict(dev)
    mismatch_keys = []
    for key, entry in orig.items():
        st = stats.get(key)
        if st is None or tuple(entry["options"]) != st["options"] or not original_mask(st):
            mismatch_keys.append(str(key))
    extra = [k for k, st in stats.items() if original_mask(st) and k not in orig]
    ak = answer_key_map(dev)
    dev_correct_q = set(int(q) for q in dev.loc[dev["IsCorrect"].astype(int) == 1, "QuestionId"].unique())

    print("models", flush=True)
    pfa_r1, _pfa_r0, dkt_r1, _dkt_r0, hashes = load_models()
    dkt_r1.eval()
    # Batch-versus-loop check on a dummy state.
    h = torch.zeros(1, 1, dkt_r1.cfg.hidden)
    c = torch.zeros(1, 1, dkt_r1.cfg.hidden)
    batch_s = dkt_option_scores(dkt_r1, h, c, 3, (1, 2, 4))
    loop_s = np.array([dkt_one_score(dkt_r1, h, c, 3, a) for a in (1, 2, 4)])
    batch_ok = bool(np.max(np.abs(batch_s - loop_s)) < 1e-6)

    users = [(str(uid), g) for uid, g in conf.groupby("UserId", sort=False)]
    # Preserve load_frame order: groupby sort=False follows first appearance, and frame is user-sorted.
    n_events = int(len(conf))
    user_index = {uid: i for i, (uid, _g) in enumerate(users)}
    n_users = len(users)

    pfa_lo = np.full(n_events, np.nan)
    pfa_hi = np.full(n_events, np.nan)
    pfa_true = np.full(n_events, np.nan)
    dkt_lo = np.full(n_events, np.nan)
    dkt_hi = np.full(n_events, np.nan)
    dkt_true = np.full(n_events, np.nan)
    n_opt = np.zeros(n_events, dtype=np.int16)
    n_users_k = np.zeros(n_events, dtype=np.int32)
    min_per = np.zeros(n_events, dtype=np.int16)
    rule_bits = np.zeros(n_events, dtype=np.uint32)
    actual_in = np.zeros(n_events, dtype=np.int8)
    y_arr = np.zeros(n_events, dtype=np.int8)
    q_arr = np.zeros(n_events, dtype=np.int16)
    ak_arr = np.full(n_events, 2, dtype=np.int8)
    dev_correct_flag = np.zeros(n_events, dtype=np.int8)
    uid_arr = np.zeros(n_events, dtype=np.int32)
    orig_bit = 0
    for i, rule in enumerate(RULES):
        if rule["rule_id"] == "C_u2_p1":
            orig_bit = 1 << i

    print(f"walk users={n_users} events={n_events}", flush=True)
    cursor = 0
    spot_checked = False
    spot_ok = True
    for ui, (uid, g) in enumerate(users):
        succ: dict[Any, float] = {}
        fail: dict[Any, float] = {}
        opt: dict[Any, list[float]] = {}
        device = next(dkt_r1.parameters()).device
        h = torch.zeros(1, 1, dkt_r1.cfg.hidden, device=device)
        cstate = torch.zeros(1, 1, dkt_r1.cfg.hidden, device=device)
        qs = g["QuestionId"].to_numpy()
        ys = g["IsCorrect"].to_numpy()
        avs = g["AnswerValue"].to_numpy()
        subjs = g["SubjectId"].to_numpy()
        for i in range(len(g)):
            subj = subjs[i]
            y = int(ys[i])
            av = int(avs[i])
            q = int(qs[i])
            if subj not in succ:
                succ[subj] = 0.0
                fail[subj] = 0.0
                opt[subj] = [0.0, 0.0, 0.0, 0.0]
            o = opt[subj]
            key = ("EEDI-R0", str(q), str(subj), int(y))
            st = stats.get(key)
            row = cursor
            y_arr[row] = y
            q_arr[row] = q
            uid_arr[row] = user_index[uid]
            ak_arr[row] = ak.get(q, 2)
            dev_correct_flag[row] = int(q in dev_correct_q)
            if st is not None and st["n_options"] >= 1:
                options = st["options"]
                pre = PFAState(succ[subj], fail[subj], o[0], o[1], o[2], o[3])
                ps = np.array(
                    [
                        pfa_next_step_proba(
                            pfa_ingest_event(pre, is_correct=y, answer_value=a, use_option_covariates=True),
                            pfa_r1,
                        )
                        for a in options
                    ],
                    dtype=np.float64,
                )
                pt = pfa_next_step_proba(
                    pfa_ingest_event(pre, is_correct=y, answer_value=av, use_option_covariates=True),
                    pfa_r1,
                )
                ds = dkt_option_scores(dkt_r1, h, cstate, q, options)
                if not spot_checked:
                    loop = np.array([dkt_one_score(dkt_r1, h, cstate, q, a) for a in options])
                    spot_ok = bool(np.max(np.abs(ds - loop)) < 1e-6)
                    spot_checked = True
                if av in options:
                    dt = float(ds[options.index(av)])
                else:
                    dt = dkt_one_score(dkt_r1, h, cstate, q, av)
                pfa_lo[row], pfa_hi[row], pfa_true[row] = float(ps.min()), float(ps.max()), float(pt)
                dkt_lo[row], dkt_hi[row], dkt_true[row] = float(ds.min()), float(ds.max()), float(dt)
                n_opt[row] = st["n_options"]
                n_users_k[row] = st["n_users"]
                min_per[row] = st["min_per"]
                rule_bits[row] = rule_mask_for(st)
                actual_in[row] = int(av in options)
            if y == 1:
                succ[subj] += 1.0
            else:
                fail[subj] += 1.0
            if 1 <= av <= 4:
                opt[subj][av - 1] += 1.0
            tok = interaction_token(q, representation="EEDI-R1", answer_value=av, n_questions=N_QUESTIONS)
            emb = dkt_r1.embedding(torch.tensor([[tok]], dtype=torch.long, device=device))
            _, (h, cstate) = dkt_r1.lstm(emb, (h, cstate))
            cursor += 1
        if ui % 100 == 0:
            print(f"user {ui}/{n_users} t={time.time()-t0:.1f}", flush=True)
    if cursor != n_events:
        raise SystemExit(f"cursor {cursor} != {n_events}")

    def labels_for(model: str, tau: float, bit: int):
        lo = pfa_lo if model == "PFA-Option" else dkt_lo
        hi = pfa_hi if model == "PFA-Option" else dkt_hi
        tr = pfa_true if model == "PFA-Option" else dkt_true
        supported = (rule_bits & bit) != 0
        stable = np.zeros(n_events, dtype=bool)
        setv = np.zeros(n_events, dtype=bool)
        fs = np.zeros(n_events, dtype=bool)
        cont = np.zeros(n_events, dtype=bool)
        ain = np.zeros(n_events, dtype=bool)
        # Vectorized status on supported rows.
        sup_idx = np.flatnonzero(supported)
        lo_s = lo[sup_idx]
        hi_s = hi[sup_idx]
        tr_s = tr[sup_idx]
        all_prog = lo_s >= tau
        all_rev = hi_s < tau
        is_stable = all_prog | all_rev
        is_set = ~is_stable
        cert_prog = all_prog  # only meaningful if stable; set-valued ignored below
        true_prog = tr_s >= tau
        is_fs = is_stable & (cert_prog != true_prog)
        is_cont = is_set | (is_stable & ~is_fs)
        stable[sup_idx] = is_stable
        setv[sup_idx] = is_set
        fs[sup_idx] = is_fs
        cont[sup_idx] = is_cont
        ain[sup_idx] = actual_in[sup_idx] == 1
        return supported, stable, setv, fs, cont, ain

    def event_metrics(supported, stable, setv, fs, cont, ain, mask=None):
        if mask is None:
            mask = np.ones(n_events, dtype=bool)
        sup = supported & mask
        st = stable & mask
        sv = setv & mask
        n = int(mask.sum())
        n_sup = int(sup.sum())
        n_st = int(st.sum())
        n_sv = int(sv.sum())
        n_un = n - n_sup
        n_fs = int((fs & mask).sum())
        n_in = int((ain & mask).sum())
        n_cont = int((cont & mask).sum())
        users_sup = int(np.unique(uid_arr[sup]).size) if n_sup else 0
        return {
            "n_events": n,
            "supported": n_sup,
            "unsupported": n_un,
            "stable": n_st,
            "set_valued": n_sv,
            "event_nc": n_un + n_sv,
            "realized_in": n_in,
            "contained": n_cont,
            "false_stable": n_fs,
            "users_with_support": users_sup,
        }

    def pack(m, model, tau, rule):
        n = m["n_events"]
        ns = m["supported"]
        nst = m["stable"]
        row = {
            "family": rule["family"],
            "rule_id": rule["rule_id"],
            "min_options": rule["min_options"],
            "min_users": rule["min_users"],
            "min_users_per_option": rule["min_users_per_option"],
            "model": model,
            "tau": tau,
            **m,
        }
        row["support_over_all"] = m["supported"] / n if n else None
        row["stable_over_all"] = m["stable"] / n if n else None
        row["set_over_supported"] = m["set_valued"] / ns if ns else None
        row["event_nc_over_all"] = m["event_nc"] / n if n else None
        row["realized_over_supported"] = m["realized_in"] / ns if ns else None
        row["contained_over_supported"] = m["contained"] / ns if ns else None
        row["false_stable_over_stable"] = (m["false_stable"] / nst) if nst else None
        row["false_stable_over_all"] = m["false_stable"] / n if n else None
        row["false_stable_over_stable_status"] = "value" if nst else "N/A"
        return row

    # Baseline at tau 0.80 before the rest of the grid is trusted.
    base_rows = {}
    base_labels = {}
    for model in ("PFA-Option", "CategoricalDKT"):
        labs = labels_for(model, 0.80, orig_bit)
        base_labels[model] = labs
        base_rows[model] = event_metrics(*labs)
    # Capacity baseline.
    cap_base = {}
    for model in ("PFA-Option", "CategoricalDKT"):
        supported = base_labels[model][0]
        lo = pfa_lo if model == "PFA-Option" else dkt_lo
        hi = pfa_hi if model == "PFA-Option" else dkt_hi
        tr = pfa_true if model == "PFA-Option" else dkt_true
        last = {}
        for i in np.flatnonzero(supported):
            last[int(uid_arr[i])] = i
        uids, los, his, acts = [], [], [], []
        for ui in sorted(last):
            i = last[ui]
            uids.append(users[ui][0])
            los.append(1.0 - float(hi[i]))
            his.append(1.0 - float(lo[i]))
            acts.append(1.0 - float(tr[i]))
        cap_base[model] = capacity_one(uids, los, his, acts, 0.05, N_FRAME_EXPECTED)

    b_pfa = base_rows["PFA-Option"]
    b_dkt = base_rows["CategoricalDKT"]
    checks = {
        "split_development": splits.get("development"),
        "split_calibration": splits.get("calibration"),
        "split_confirmatory": splits.get("confirmatory"),
        "user_split_overlap": overlap,
        "orig_dict_keys": len(orig),
        "orig_dict_mismatches": len(mismatch_keys),
        "orig_dict_missing_from_builder": len(extra),
        "dkt_batch_matches_loop_dummy": batch_ok,
        "dkt_batch_matches_loop_first_event": spot_ok,
        "supported": b_pfa["supported"] == EXPECTED["supported"] == b_dkt["supported"],
        "unsupported": b_pfa["unsupported"] == EXPECTED["unsupported"],
        "pfa_stable": b_pfa["stable"] == EXPECTED["pfa_stable"],
        "pfa_set": b_pfa["set_valued"] == EXPECTED["pfa_set"],
        "dkt_stable": b_dkt["stable"] == EXPECTED["dkt_stable"],
        "dkt_set": b_dkt["set_valued"] == EXPECTED["dkt_set"],
        "actual_in": b_pfa["realized_in"] == EXPECTED["actual_in"] == b_dkt["realized_in"],
        "pfa_fs": b_pfa["false_stable"] == 0,
        "dkt_fs": b_dkt["false_stable"] == 0,
        "pfa_eligible": cap_base["PFA-Option"]["n_eligible"] == EXPECTED["eligible"],
        "dkt_eligible": cap_base["CategoricalDKT"]["n_eligible"] == EXPECTED["eligible"],
        "pfa_B": cap_base["PFA-Option"]["B"] == EXPECTED["B"],
        "pfa_gin": cap_base["PFA-Option"]["guaranteed_in"] == EXPECTED["pfa_gin"],
        "pfa_gout": cap_base["PFA-Option"]["guaranteed_out"] == EXPECTED["pfa_gout"],
        "pfa_amb": cap_base["PFA-Option"]["unresolved"] == EXPECTED["pfa_amb"],
        "dkt_gin": cap_base["CategoricalDKT"]["guaranteed_in"] == EXPECTED["dkt_gin"],
        "dkt_gout": cap_base["CategoricalDKT"]["guaranteed_out"] == EXPECTED["dkt_gout"],
        "dkt_amb": cap_base["CategoricalDKT"]["unresolved"] == EXPECTED["dkt_amb"],
    }
    baseline = {
        "checks": checks,
        "pass": all(bool(v) for k, v in checks.items() if k not in {"split_development", "split_calibration", "split_confirmatory"}),
        "reproduced": {"PFA-Option": b_pfa, "CategoricalDKT": b_dkt},
        "capacity": cap_base,
        "expected_splits": {"development": [2865, 799590], "calibration": [1032, 288711], "confirmatory": [1021, 294426]},
    }
    # Split equality is part of pass.
    split_ok = (
        splits.get("development") == {"users": 2865, "events": 799590}
        and splits.get("calibration") == {"users": 1032, "events": 288711}
        and splits.get("confirmatory") == {"users": 1021, "events": 294426}
        and overlap == 0
    )
    bool_checks = [v for k, v in checks.items() if isinstance(v, bool)]
    baseline["pass"] = bool(all(bool_checks) and split_ok and batch_ok and spot_ok and not mismatch_keys and not extra)
    (OUT / "baseline_reproduction.json").write_text(json.dumps(baseline, indent=2, default=str), encoding="utf-8")
    print("BASELINE", baseline["pass"], flush=True)
    if not baseline["pass"]:
        print(json.dumps(checks, indent=2, default=str))
        raise SystemExit(2)

    grid_rows = []
    label_cache = {}
    for rule_i, rule in enumerate(RULES):
        bit = 1 << rule_i
        for model in ("PFA-Option", "CategoricalDKT"):
            for tau in TAUS:
                labs = labels_for(model, tau, bit)
                label_cache[(rule["rule_id"], model, tau)] = labs
                grid_rows.append(pack(event_metrics(*labs), model, tau, rule))
    fs_cases = []
    for (rule_id, model, tau), labs in label_cache.items():
        fs = labs[3]
        for i in np.flatnonzero(fs):
            fs_cases.append(
                {
                    "rule_id": rule_id,
                    "model": model,
                    "tau": tau,
                    "question_id": int(q_arr[i]),
                    "is_correct": int(y_arr[i]),
                    "n_options": int(n_opt[i]),
                    "n_dev_users": int(n_users_k[i]),
                    "min_users_per_option": int(min_per[i]),
                    "answer_key": int(ak_arr[i]),
                    "realized_in_class": int(actual_in[i]),
                }
            )
    (OUT / "false_stable_cases.json").write_text(json.dumps(fs_cases, indent=2), encoding="utf-8")
    pd.DataFrame(grid_rows).to_csv(OUT / "support_rule_grid.csv", index=False)
    pd.DataFrame([r for r in grid_rows if r["family"] == "singleton_permissive"]).to_csv(
        OUT / "threshold_sensitivity_singleton.csv", index=False
    )
    pd.DataFrame([r for r in grid_rows if r["family"] == "collision"]).to_csv(
        OUT / "threshold_sensitivity.csv", index=False
    )

    # Monotonicity inside the collision family at each tau and model.
    violations = []
    by_id = {(r["rule_id"], r["model"], r["tau"]): r for r in grid_rows}
    for model in ("PFA-Option", "CategoricalDKT"):
        for tau in TAUS:
            for a in COLLISION:
                for b in COLLISION:
                    stricter = (
                        a["min_users"] >= b["min_users"]
                        and a["min_users_per_option"] >= b["min_users_per_option"]
                        and (a["min_users"] > b["min_users"] or a["min_users_per_option"] > b["min_users_per_option"])
                    )
                    if not stricter:
                        continue
                    ra = by_id[(a["rule_id"], model, tau)]
                    rb = by_id[(b["rule_id"], model, tau)]
                    if ra["supported"] > rb["supported"]:
                        violations.append(("supported", a["rule_id"], b["rule_id"], model, tau))
                    if ra["stable"] > rb["stable"]:
                        violations.append(("stable", a["rule_id"], b["rule_id"], model, tau))
                    if ra["event_nc"] < rb["event_nc"]:
                        violations.append(("event_nc", a["rule_id"], b["rule_id"], model, tau))
                    # Same scores: a supported event under the stricter rule is supported under the looser rule,
                    # and status matches. Check on the bit masks.
                    bit_a = 1 << RULES.index(a)
                    bit_b = 1 << RULES.index(b)
                    both = ((rule_bits & bit_a) != 0) & ((rule_bits & bit_b) != 0)
                    only_strict = ((rule_bits & bit_a) != 0) & ((rule_bits & bit_b) == 0)
                    if int(only_strict.sum()) != 0:
                        violations.append(("mask_not_nested", a["rule_id"], b["rule_id"], model, tau, int(only_strict.sum())))
                    la = label_cache[(a["rule_id"], model, tau)]
                    lb = label_cache[(b["rule_id"], model, tau)]
                    if not np.array_equal(la[1][both], lb[1][both]) or not np.array_equal(la[2][both], lb[2][both]):
                        violations.append(("status_changed", a["rule_id"], b["rule_id"], model, tau))

    band_arr = np.array([user_band(int(v)) for v in n_users_k], dtype=object)
    per_arr = np.array([per_band(int(v)) for v in min_per], dtype=object)
    # Correctness / abundance / answer key / q0 for every rule at tau 0.80.
    corr_rows = []
    abund_rows = []
    single_rows = []
    rel_rows = []
    q0_rows = []
    ak_name = {0: "unique", 1: "conflicting", 2: "unavailable"}
    for rule_i, rule in enumerate(RULES):
        bit = 1 << rule_i
        orig_supported = (rule_bits & orig_bit) != 0
        this_supported = (rule_bits & bit) != 0
        newly = this_supported & ~orig_supported
        for model in ("PFA-Option", "CategoricalDKT"):
            labs = label_cache[(rule["rule_id"], model, 0.80)]
            supported, stable, setv, fs, cont, ain = labs
            for yv, yname in ((1, "correct"), (0, "incorrect")):
                mask = y_arr == yv
                met = event_metrics(supported, stable, setv, fs, cont, ain, mask)
                corr_rows.append(pack(met, model, 0.80, rule) | {"correctness": yname})
            for band in ("lt2", "2-4", "5-9", "10-19", "20+"):
                mask = band_arr == band
                # Events with no dev key have n_users 0 -> lt2.
                met = event_metrics(supported, stable, setv, fs, cont, ain, mask)
                abund_rows.append(pack(met, model, 0.80, rule) | {"user_band": band})
            for cat, cname in ak_name.items():
                mask = ak_arr == cat
                met = event_metrics(supported, stable, setv, fs, cont, ain, mask)
                corr_rows.append(pack(met, model, 0.80, rule) | {"answer_key": cname, "correctness": "all"})
            for qmask, qname in ((q_arr == 0, "question0"), (q_arr != 0, "question_nonzero_posthoc")):
                met = event_metrics(supported, stable, setv, fs, cont, ain, qmask)
                q0_rows.append(pack(met, model, 0.80, rule) | {"subset": qname})
            # Reliability strata on supported stable events.
            for band in ("2-4", "5-9", "10-19", "20+"):
                mask = stable & (band_arr == band)
                n_st = int(mask.sum())
                n_fs = int((fs & mask).sum())
                n_in = int((ain & mask).sum())
                n_out = n_st - n_in
                learners = int(np.unique(uid_arr[mask]).size) if n_st else 0
                rel_rows.append(
                    {
                        "family": rule["family"],
                        "rule_id": rule["rule_id"],
                        "model": model,
                        "tau": 0.80,
                        "stratum": f"users_{band}",
                        "stable": n_st,
                        "realized_in": n_in,
                        "realized_out": n_out,
                        "false_stable": n_fs,
                        "false_stable_over_stable": (n_fs / n_st) if n_st else None,
                        "false_stable_status": "value" if n_st else "N/A",
                        "learners": learners,
                    }
                )
            for nopt_name, nmask in (("singleton_option", n_opt == 1), ("multi_option", n_opt >= 2)):
                mask = stable & nmask
                n_st = int(mask.sum())
                rel_rows.append(
                    {
                        "family": rule["family"],
                        "rule_id": rule["rule_id"],
                        "model": model,
                        "tau": 0.80,
                        "stratum": nopt_name,
                        "stable": n_st,
                        "realized_in": int((ain & mask).sum()),
                        "realized_out": n_st - int((ain & mask).sum()),
                        "false_stable": int((fs & mask).sum()),
                        "false_stable_over_stable": (int((fs & mask).sum()) / n_st) if n_st else None,
                        "false_stable_status": "value" if n_st else "N/A",
                        "learners": int(np.unique(uid_arr[mask]).size) if n_st else 0,
                    }
                )
            for pb in ("1", "2-4", "5+"):
                mask = stable & (per_arr == pb)
                n_st = int(mask.sum())
                rel_rows.append(
                    {
                        "family": rule["family"],
                        "rule_id": rule["rule_id"],
                        "model": model,
                        "tau": 0.80,
                        "stratum": f"min_per_{pb}",
                        "stable": n_st,
                        "realized_in": int((ain & mask).sum()),
                        "realized_out": n_st - int((ain & mask).sum()),
                        "false_stable": int((fs & mask).sum()),
                        "false_stable_over_stable": (int((fs & mask).sum()) / n_st) if n_st else None,
                        "false_stable_status": "value" if n_st else "N/A",
                        "learners": int(np.unique(uid_arr[mask]).size) if n_st else 0,
                    }
                )
        if rule["family"] == "singleton_permissive":
            # Newly supported relative to the original collision cell. Model-free support mask.
            n_new = int(newly.sum())
            n_new_correct = int((newly & (y_arr == 1)).sum())
            # Singleton keys among this rule's supported events.
            sing_sup = this_supported & (n_opt == 1)
            # User-count distribution of those events' development keys, event-weighted and key-weighted.
            # Key-weighted: unique keys. We don't store key ids; use (q, y, n_users, n_opt) is not a key.
            # Store event-weighted band counts. Also count distinct (q, subject not stored). Use n_users and q and y and n_opt as a coarse key.
            # Better: during walk we didn't keep subject. Event-weighted bands are still the requested distribution of development-user counts for singleton supported events.
            bands = {}
            for band in ("2-4", "5-9", "10-19", "20+", "lt2"):
                bands[band] = int((sing_sup & (band_arr == band)).sum())
            for model in ("PFA-Option", "CategoricalDKT"):
                labs = label_cache[(rule["rule_id"], model, 0.80)]
                supported, stable, setv, fs, cont, ain = labs
                sing_stable = stable & (n_opt == 1)
                n_ss = int(sing_stable.sum())
                single_rows.append(
                    {
                        "rule_id": rule["rule_id"],
                        "min_users": rule["min_users"],
                        "model": model,
                        "tau": 0.80,
                        "supported": int(supported.sum()),
                        "newly_supported_events": n_new,
                        "newly_supported_correct": n_new_correct,
                        "newly_supported_correct_share": (n_new_correct / n_new) if n_new else None,
                        "singleton_supported_events": int(sing_sup.sum()),
                        "singleton_stable": n_ss,
                        "singleton_realized_in": int((ain & sing_stable).sum()),
                        "singleton_false_stable": int((fs & sing_stable).sum()),
                        "singleton_false_stable_over_stable": (int((fs & sing_stable).sum()) / n_ss) if n_ss else None,
                        "singleton_event_user_bands": json.dumps(bands),
                    }
                )

    pd.DataFrame(corr_rows).to_csv(OUT / "support_by_correctness.csv", index=False)
    pd.DataFrame(abund_rows).to_csv(OUT / "support_by_development_abundance.csv", index=False)
    pd.DataFrame(single_rows).to_csv(OUT / "singleton_permissive_diagnostics.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(OUT / "certificate_reliability_strata.csv", index=False)
    pd.DataFrame(q0_rows).to_csv(OUT / "question0_stratification.csv", index=False)

    # Development geometry, no user ids.
    geo_rows = []
    for st in stats.values():
        geo_rows.append(
            {
                "y": st["y"],
                "q": st["q"],
                "n_events": st["n_events"],
                "n_users": st["n_users"],
                "n_options": st["n_options"],
                "min_per": st["min_per"],
                "max_per": st["max_per"],
                "original_supported_key": int(original_mask(st)),
                "user_band": user_band(st["n_users"]),
                "answer_key": ak_name.get(ak.get(st["q"], 2), "unavailable"),
            }
        )
    geo = pd.DataFrame(geo_rows)
    geo_summary = {
        "n_development_keys": int(len(geo)),
        "keys_original_support": int(geo["original_supported_key"].sum()),
        "keys_by_n_options": {str(k): int(v) for k, v in geo["n_options"].value_counts().sort_index().items()},
        "keys_by_user_band": {str(k): int(v) for k, v in geo["user_band"].value_counts().items()},
        "events_by_n_options": {str(k): int(v) for k, v in geo.groupby("n_options")["n_events"].sum().items()},
        "questions_answer_key": {
            "unique": int(sum(1 for v in ak.values() if v == 0)),
            "conflicting": int(sum(1 for v in ak.values() if v == 1)),
            "unavailable": int(sum(1 for v in ak.values() if v == 2)),
        },
        "development_questions_with_a_correct_response": int(len(dev_correct_q)),
    }
    (OUT / "development_geometry.json").write_text(json.dumps(geo_summary, indent=2), encoding="utf-8")
    geo.drop(columns=[]).to_csv(OUT / "development_key_stats.csv", index=False)

    # Capacity grid.
    cap_rows = []
    for rule_i, rule in enumerate(RULES):
        bit = 1 << rule_i
        supported_mask = (rule_bits & bit) != 0
        for model in ("PFA-Option", "CategoricalDKT"):
            lo = pfa_lo if model == "PFA-Option" else dkt_lo
            hi = pfa_hi if model == "PFA-Option" else dkt_hi
            tr = pfa_true if model == "PFA-Option" else dkt_true
            last = {}
            for i in np.flatnonzero(supported_mask):
                last[int(uid_arr[i])] = i
            uids, los, his, acts = [], [], [], []
            order = sorted(last)
            for ui in order:
                i = last[ui]
                uids.append(users[ui][0])
                los.append(1.0 - float(hi[i]))
                his.append(1.0 - float(lo[i]))
                acts.append(1.0 - float(tr[i]))
            for kappa in KAPPAS:
                rec = capacity_one(uids, los, his, acts, kappa, N_FRAME_EXPECTED)
                rec.update(
                    {
                        "family": rule["family"],
                        "rule_id": rule["rule_id"],
                        "min_users": rule["min_users"],
                        "min_users_per_option": rule["min_users_per_option"],
                        "min_options": rule["min_options"],
                        "model": model,
                        "kappa": kappa,
                        "cohort_note": "eligible set and last checkpoint change with the support rule; not a fixed-cohort comparison",
                    }
                )
                cap_rows.append(rec)
    pd.DataFrame(cap_rows).to_csv(OUT / "capacity_sensitivity.csv", index=False)

    # Learner-clustered bootstrap at tau 0.80, paired to the original collision cell.
    print("bootstrap", flush=True)
    rng = np.random.default_rng(SEED)
    # Per-user counts for each collision rule and both models: events, supported, stable, set, fs.
    boot_rows = []
    draws = rng.integers(0, n_users, size=(N_BOOT, n_users))
    user_n = np.bincount(uid_arr, minlength=n_users)
    for model in ("PFA-Option", "CategoricalDKT"):
        base_labs = label_cache[("C_u2_p1", model, 0.80)]
        base_sup_u = np.bincount(uid_arr[base_labs[0]], minlength=n_users)
        base_st_u = np.bincount(uid_arr[base_labs[1]], minlength=n_users)
        base_sv_u = np.bincount(uid_arr[base_labs[2]], minlength=n_users)
        base_fs_u = np.bincount(uid_arr[base_labs[3]], minlength=n_users)
        for rule in COLLISION + SINGLETON:
            labs = label_cache[(rule["rule_id"], model, 0.80)]
            sup_u = np.bincount(uid_arr[labs[0]], minlength=n_users)
            st_u = np.bincount(uid_arr[labs[1]], minlength=n_users)
            sv_u = np.bincount(uid_arr[labs[2]], minlength=n_users)
            fs_u = np.bincount(uid_arr[labs[3]], minlength=n_users)
            # Resample sums.
            n_sum = user_n[draws].sum(axis=1).astype(np.float64)
            sup_sum = sup_u[draws].sum(axis=1).astype(np.float64)
            st_sum = st_u[draws].sum(axis=1).astype(np.float64)
            sv_sum = sv_u[draws].sum(axis=1).astype(np.float64)
            fs_sum = fs_u[draws].sum(axis=1).astype(np.float64)
            b_n = user_n[draws].sum(axis=1).astype(np.float64)
            b_sup = base_sup_u[draws].sum(axis=1).astype(np.float64)
            b_st = base_st_u[draws].sum(axis=1).astype(np.float64)
            b_nc = (b_n - b_st)
            sup_rate = sup_sum / n_sum
            st_rate = st_sum / n_sum
            nc_rate = (n_sum - st_sum) / n_sum
            b_sup_rate = b_sup / b_n
            b_st_rate = b_st / b_n
            b_nc_rate = b_nc / b_n
            point = by_id[(rule["rule_id"], model, 0.80)]
            bpoint = by_id[("C_u2_p1", model, 0.80)]

            def pct(arr):
                return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))

            metrics = {
                "support_over_all": (sup_rate, point["support_over_all"], sup_rate - b_sup_rate, point["support_over_all"] - bpoint["support_over_all"]),
                "stable_over_all": (st_rate, point["stable_over_all"], st_rate - b_st_rate, point["stable_over_all"] - bpoint["stable_over_all"]),
                "event_nc_over_all": (nc_rate, point["event_nc_over_all"], nc_rate - b_nc_rate, point["event_nc_over_all"] - bpoint["event_nc_over_all"]),
            }
            for name, (arr, point_v, contrast, contrast_point) in metrics.items():
                lo, hi = pct(arr)
                clo, chi = pct(contrast)
                boot_rows.append(
                    {
                        "family": rule["family"],
                        "rule_id": rule["rule_id"],
                        "model": model,
                        "tau": 0.80,
                        "metric": name,
                        "point": point_v,
                        "boot_p2_5": lo,
                        "boot_p97_5": hi,
                        "contrast_vs_C_u2_p1": contrast_point,
                        "contrast_p2_5": clo,
                        "contrast_p97_5": chi,
                        "n_resamples": N_BOOT,
                        "seed": SEED,
                        "unit": "learner",
                        "note": "percentile interval of the learner-cluster resample; not a future-error guarantee",
                    }
                )
            # False-stable rate among stable. Degenerate if every resample has the same zeros.
            rate = np.full(N_BOOT, np.nan)
            ok = st_sum > 0
            rate[ok] = fs_sum[ok] / st_sum[ok]
            if int(ok.sum()) == 0:
                flo = fhi = None
                status = "N/A"
            else:
                flo, fhi = pct(rate[ok])
                status = "value"
            boot_rows.append(
                {
                    "family": rule["family"],
                    "rule_id": rule["rule_id"],
                    "model": model,
                    "tau": 0.80,
                    "metric": "false_stable_over_stable",
                    "point": point["false_stable_over_stable"],
                    "boot_p2_5": flo,
                    "boot_p97_5": fhi,
                    "contrast_vs_C_u2_p1": None,
                    "contrast_p2_5": None,
                    "contrast_p97_5": None,
                    "n_resamples": N_BOOT,
                    "n_resamples_defined": int(ok.sum()),
                    "seed": SEED,
                    "unit": "learner",
                    "status": status,
                    "note": "If the observed false-Stable count is 0 for every learner, this interval is degenerate and is not evidence that future error risk is 0.",
                }
            )
    pd.DataFrame(boot_rows).to_csv(OUT / "paired_bootstrap_intervals.csv", index=False)

    # Original unsupported composition, development side information only.
    orig_unsup = (rule_bits & orig_bit) == 0
    composition = {
        "unsupported": int(orig_unsup.sum()),
        "unsupported_incorrect": int((orig_unsup & (y_arr == 0)).sum()),
        "unsupported_correct": int((orig_unsup & (y_arr == 1)).sum()),
        "unsupported_correct_unique_key_with_dev_correct": int(
            (orig_unsup & (y_arr == 1) & (ak_arr == 0) & (dev_correct_flag == 1) & (n_opt == 1)).sum()
        ),
        "note": "Answer-key categories are development side information. They are not part of the release key. Option count here is the development-observed option count, which is 1 for a unique correct option only when a single AnswerValue was seen; that is not the same object as a unique CorrectAnswer column.",
    }
    integrity = {
        "baseline_pass": True,
        "monotonicity_violations": violations,
        "n_monotonicity_violations": len(violations),
        "dkt_batch_matches_loop_dummy": batch_ok,
        "dkt_batch_matches_loop_first_scored_event": spot_ok,
        "dictionary_key_mismatches": len(mismatch_keys) + len(extra),
        "user_split_overlap": overlap,
        "unsupported_composition_diagnostic": composition,
        "elapsed_s": time.time() - t0,
        "versions": versions(),
        "hashes": hashes,
        "seed": SEED,
        "n_boot": N_BOOT,
        "command": "python scripts/eedi/support_sensitivity.py",
    }
    (OUT / "integrity_checks.json").write_text(json.dumps(integrity, indent=2, default=str), encoding="utf-8")
    manifest = {
        "experiment": "pipeline",
        "rules": RULES,
        "taus": TAUS,
        "kappas": KAPPAS,
        "seed": SEED,
        "n_boot": N_BOOT,
        "n_events": n_events,
        "n_users": n_users,
        "versions": versions(),
        "hashes": hashes,
        "split_function": "src/data/eedi_split.py:assign_eedi_split",
        "split_seed": "WDLF-G6P-SPLIT-SEED-v1",
        "release_key": "(EEDI-R0, QuestionId, SubjectId, IsCorrect)",
        "consumers": "PFA-Option R1 and CategoricalDKT R1, fixed checkpoints",
        "capacity": "last supported event, product intervals, ascending user-id ties, B=round(kappa*|U_cert|) with floor 1",
        "command": "python scripts/eedi/support_sensitivity.py",
    }
    (OUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_output_checksums(OUT)
    print("DONE", time.time() - t0, "violations", len(violations), flush=True)


if __name__ == "__main__":
    main()
