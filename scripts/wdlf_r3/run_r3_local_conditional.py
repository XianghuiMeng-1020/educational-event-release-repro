#!/usr/bin/env python3
"""WDLF R3 local-conditional confirmatory extension.

LEVEL A: fix richer prehistory; vary current option over C_dev; post-ingest mastery proxy.

Phases:
  structural-eligibility — support-only metadata (no decision effects)
  dry-run-cal            — calibration students only (engineering)
  confirmatory           — sealed confirmatory execution (no effect prints)
  unseal                 — controlled unseal after integrity audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.eedi_split import assign_eedi_split
from src.identification.compatible_trace.capacity_identified_set import (
    brute_force_membership,
    certify_capacity_membership,
)
from src.identification.compatible_trace.certificate_v2 import certify_from_propagated_scores
from src.identification.compatible_trace.dkt_propagation import _next_step_proba_from_state
from src.identification.compatible_trace.pfa_propagation import (
    PFAState,
    pfa_ingest_event,
    pfa_next_step_proba,
)
from src.models.categorical_dkt import CategoricalDKT, CategoricalDKTConfig, interaction_token
from src.models.pfa_option import PFAOption, PFAOptionConfig, PFAOptionParams
from src.policies.capacity_support_v2 import select_capacity_list

OUT = ROOT / "prospective" / "wdlf_r3"
SEALED = OUT / "sealed"
RESULTS = ROOT / "results" / "wdlf_r3"
CACHE = ROOT / "prospective" / "g6p" / "data" / "cache"
CKPT = ROOT / "prospective" / "g6p" / "checkpoints"
MANIFEST = ROOT / "experiments" / "wdlf_r3_confirmatory_extension_manifest_v3.yaml"
DATA_PATH = CACHE / "tasks_3_4_joined_min.parquet"
N_QUESTIONS = 947
MAX_HISTORY = 200
PRIMARY_TAU = 0.80
TAU_SENS = [0.70, 0.90]
PRIMARY_KAPPA = 0.05
KAPPA_SENS = [0.01, 0.10]
PRODUCT_BRUTE_LIMIT = 200_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def log_phase(msg: str) -> None:
    print(f"[r3] {msg}", flush=True)


def build_compatible_dict(dev: pd.DataFrame) -> dict[tuple, dict[str, Any]]:
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
        if len(opts) < 2 or len(users) < 2 or any(len(users_by_opt[o]) < 1 for o in opts):
            continue
        out[key] = {"options": opts, "n_students": len(users)}
    return out


def load_models() -> tuple[PFAOption, PFAOption, CategoricalDKT, CategoricalDKT, dict[str, str]]:
    pfa_blob = json.loads((CKPT / "pfa_option_params.json").read_text(encoding="utf-8"))
    pfa_r1 = PFAOption(PFAOptionConfig(**pfa_blob["r1"]["config"]), PFAOptionParams(**pfa_blob["r1"]["params"]))
    pfa_r0 = PFAOption(PFAOptionConfig(**pfa_blob["r0"]["config"]), PFAOptionParams(**pfa_blob["r0"]["params"]))
    # Sequential 1-token forks are CPU-bound; CUDA launch overhead dominates and is less stable.
    device = torch.device("cpu")
    cfg1 = CategoricalDKTConfig("EEDI-R1", n_questions=N_QUESTIONS, max_history=MAX_HISTORY)
    cfg0 = CategoricalDKTConfig("EEDI-R0", n_questions=N_QUESTIONS, max_history=MAX_HISTORY)
    dkt1 = CategoricalDKT(cfg1).to(device)
    dkt0 = CategoricalDKT(cfg0).to(device)
    dkt1.load_state_dict(torch.load(CKPT / "categorical_dkt_r1.pt", map_location=device))
    dkt0.load_state_dict(torch.load(CKPT / "categorical_dkt_r0.pt", map_location=device))
    dkt1.eval()
    dkt0.eval()
    hashes = {
        "pfa_option_params.json": sha256_file(CKPT / "pfa_option_params.json"),
        "categorical_dkt_r1.pt": sha256_file(CKPT / "categorical_dkt_r1.pt"),
        "categorical_dkt_r0.pt": sha256_file(CKPT / "categorical_dkt_r0.pt"),
        "data": sha256_file(DATA_PATH),
        "pfa_r1_config_hash": pfa_blob["r1"]["config_hash"],
        "dkt_r1_config_hash": cfg1.config_hash(),
        "dkt_r0_config_hash": cfg0.config_hash(),
    }
    return pfa_r1, pfa_r0, dkt1, dkt0, hashes


def load_frame() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["UserId"] = df["UserId"].astype(str)
    df["split"] = df["UserId"].map(assign_eedi_split)
    df = df.sort_values(["UserId", "DateAnswered_parsed", "AnswerId"], kind="mergesort")
    return df


@dataclass
class EventCert:
    user_id: str
    split: str
    event_index: int
    question_id: int
    subject_id: Any
    is_correct: int
    observed_answer: int
    supported: bool
    status: str
    actions: tuple[str, ...]
    scores: tuple[float, ...]
    n_candidates: int
    threshold: float
    model: str


def pfa_walk_user(
    g: pd.DataFrame,
    *,
    pfa_r1: PFAOption,
    pfa_r0: PFAOption,
    compat: dict,
    target_splits: set[str],
    thresholds: list[float],
) -> tuple[list[EventCert], dict[str, Any] | None, float | None, float | None]:
    uid = str(g["UserId"].iloc[0])
    split = str(g["split"].iloc[0])
    succ: dict[Any, float] = defaultdict(float)
    fail: dict[Any, float] = defaultdict(float)
    opt: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    certs: list[EventCert] = []
    pipe_r0: list[float] = []
    pipe_r1: list[float] = []
    last_supported: dict[str, Any] | None = None

    qs = g["QuestionId"].to_numpy()
    ys = g["IsCorrect"].to_numpy()
    avs = g["AnswerValue"].to_numpy()
    subjs = g["SubjectId"].to_numpy()

    for i in range(len(g)):
        subj = subjs[i]
        y = int(ys[i])
        av = int(avs[i])
        q = int(qs[i])
        o = opt[subj]
        feats0 = {
            "successes": succ[subj],
            "failures": fail[subj],
            "opt1": 0.0,
            "opt2": 0.0,
            "opt3": 0.0,
            "opt4": 0.0,
        }
        feats1 = {
            "successes": succ[subj],
            "failures": fail[subj],
            "opt1": o[0],
            "opt2": o[1],
            "opt3": o[2],
            "opt4": o[3],
        }
        pipe_r0.append(float(pfa_r0.predict_proba_from_feats(feats0)))
        pipe_r1.append(float(pfa_r1.predict_proba_from_feats(feats1)))

        key = ("EEDI-R0", str(q), str(subj), int(y))
        entry = compat.get(key)
        if split in target_splits:
            if entry is None:
                for thr in thresholds:
                    certs.append(
                        EventCert(
                            uid, split, i, q, subj, y, av, False, "unsupported", (), (), 0, thr, "PFA-Option"
                        )
                    )
            else:
                pre = PFAState(succ[subj], fail[subj], o[0], o[1], o[2], o[3])
                scores = [
                    pfa_next_step_proba(
                        pfa_ingest_event(pre, is_correct=y, answer_value=a, use_option_covariates=True),
                        pfa_r1,
                    )
                    for a in entry["options"]
                ]
                for thr in thresholds:
                    c = certify_from_propagated_scores(str(key), scores, threshold=thr, search_mode="exact")
                    certs.append(
                        EventCert(
                            uid,
                            split,
                            i,
                            q,
                            subj,
                            y,
                            av,
                            True,
                            c.status,
                            c.actions,
                            tuple(scores),
                            c.n_candidates,
                            thr,
                            "PFA-Option",
                        )
                    )
                last_supported = {
                    "user_id": uid,
                    "risks": [1.0 - s for s in scores],
                    "scores": scores,
                    "event_index": i,
                }
        if y == 1:
            succ[subj] += 1.0
        else:
            fail[subj] += 1.0
        if 1 <= av <= 4:
            opt[subj][av - 1] += 1.0

    mean_r0 = float(np.mean(pipe_r0)) if pipe_r0 else None
    mean_r1 = float(np.mean(pipe_r1)) if pipe_r1 else None
    return certs, (last_supported if split in target_splits else None), mean_r0, mean_r1


@torch.no_grad()
def dkt_walk_user(
    g: pd.DataFrame,
    *,
    dkt_r1: CategoricalDKT,
    dkt_r0: CategoricalDKT,
    compat: dict,
    target_splits: set[str],
    thresholds: list[float],
) -> tuple[list[EventCert], dict[str, Any] | None, float | None, float | None]:
    uid = str(g["UserId"].iloc[0])
    split = str(g["split"].iloc[0])
    device = next(dkt_r1.parameters()).device
    qs = g["QuestionId"].astype(int).tolist()
    ys = g["IsCorrect"].astype(int).tolist()
    avs = g["AnswerValue"].astype(int).tolist()
    subjs = g["SubjectId"].tolist()
    certs: list[EventCert] = []
    last_supported: dict[str, Any] | None = None
    pipe_r1: list[float] = []
    pipe_r0: list[float] = []

    h = torch.zeros(1, 1, dkt_r1.cfg.hidden, device=device)
    c = torch.zeros(1, 1, dkt_r1.cfg.hidden, device=device)
    h0 = torch.zeros(1, 1, dkt_r0.cfg.hidden, device=device)
    c0 = torch.zeros(1, 1, dkt_r0.cfg.hidden, device=device)

    for i in range(len(g)):
        q, y, av, subj = qs[i], ys[i], avs[i], subjs[i]
        pipe_r1.append(_next_step_proba_from_state(dkt_r1, h, c))
        pipe_r0.append(_next_step_proba_from_state(dkt_r0, h0, c0))

        key = ("EEDI-R0", str(q), str(subj), int(y))
        entry = compat.get(key)
        if split in target_splits:
            if entry is None:
                for thr in thresholds:
                    certs.append(
                        EventCert(
                            uid,
                            split,
                            i,
                            q,
                            subj,
                            y,
                            av,
                            False,
                            "unsupported",
                            (),
                            (),
                            0,
                            thr,
                            "CategoricalDKT",
                        )
                    )
            else:
                scores = []
                for a in entry["options"]:
                    tok = interaction_token(
                        q, representation="EEDI-R1", answer_value=int(a), n_questions=N_QUESTIONS
                    )
                    emb = dkt_r1.embedding(torch.tensor([[tok]], dtype=torch.long, device=device))
                    _, (h2, c2) = dkt_r1.lstm(emb, (h, c))
                    scores.append(_next_step_proba_from_state(dkt_r1, h2, c2))
                for thr in thresholds:
                    cert = certify_from_propagated_scores(str(key), scores, threshold=thr, search_mode="exact")
                    certs.append(
                        EventCert(
                            uid,
                            split,
                            i,
                            q,
                            subj,
                            y,
                            av,
                            True,
                            cert.status,
                            cert.actions,
                            tuple(scores),
                            cert.n_candidates,
                            thr,
                            "CategoricalDKT",
                        )
                    )
                last_supported = {
                    "user_id": uid,
                    "risks": [1.0 - s for s in scores],
                    "scores": scores,
                    "event_index": i,
                }

        tok_obs = interaction_token(
            q, representation="EEDI-R1", answer_value=av, n_questions=N_QUESTIONS
        )
        emb = dkt_r1.embedding(torch.tensor([[tok_obs]], dtype=torch.long, device=device))
        _, (h, c) = dkt_r1.lstm(emb, (h, c))

        tok0 = interaction_token(
            q, representation="EEDI-R0", is_correct=y, n_questions=N_QUESTIONS
        )
        emb0 = dkt_r0.embedding(torch.tensor([[tok0]], dtype=torch.long, device=device))
        _, (h0, c0) = dkt_r0.lstm(emb0, (h0, c0))

    mean_r0 = float(np.mean(pipe_r0)) if pipe_r0 else None
    mean_r1 = float(np.mean(pipe_r1)) if pipe_r1 else None
    return certs, (last_supported if split in target_splits else None), mean_r0, mean_r1


def summarize_p1(certs: list[EventCert], model: str, threshold: float) -> dict[str, Any]:
    rows = [c for c in certs if c.model == model and abs(c.threshold - threshold) < 1e-12]
    n = len(rows)
    if n == 0:
        return {"n_events": 0, "model": model, "threshold": threshold}
    counts: dict[str, int] = defaultdict(int)
    for c in rows:
        counts[c.status] += 1
    supported = n - counts["unsupported"]
    return {
        "model": model,
        "threshold": threshold,
        "n_events": n,
        "unsupported": counts["unsupported"],
        "unsupported_rate": counts["unsupported"] / n,
        "supported": supported,
        "supported_rate": supported / n,
        "stable": counts["stable"],
        "stable_rate": counts["stable"] / n,
        "set_valued": counts["set-valued"],
        "set_valued_rate": counts["set-valued"] / n,
        "uncertified": counts["uncertified"],
        "uncertified_rate": counts["uncertified"] / n,
        "abstain": counts["abstain"],
        "stable_given_supported": (counts["stable"] / supported) if supported else None,
        "set_valued_given_supported": (counts["set-valued"] / supported) if supported else None,
    }


def run_p2(
    user_payloads: list[dict[str, Any]],
    *,
    kappa: float,
    n_frame: int,
) -> dict[str, Any]:
    eligible = [u for u in user_payloads if u is not None]
    u_cert = len(eligible)
    coverage = u_cert / n_frame if n_frame else None
    if u_cert == 0:
        return {"n_eligible": 0, "coverage": coverage, "kappa": kappa, "n_frame": n_frame}
    user_ids = [u["user_id"] for u in eligible]
    risk_sets = [u["risks"] for u in eligible]
    mins = [min(r) for r in risk_sets]
    maxs = [max(r) for r in risk_sets]
    mem = certify_capacity_membership(user_ids, mins, maxs, capacity=kappa)
    product = 1
    for r in risk_sets:
        product *= max(len(set(r)), 1)
        if product > PRODUCT_BRUTE_LIMIT:
            product = -1
            break
    brute = None
    if 0 < product <= PRODUCT_BRUTE_LIMIT:
        brute = brute_force_membership(user_ids, risk_sets, capacity=kappa)
    by = {m.user_id: m for m in mem}
    status_of = {}
    for uid in user_ids:
        status_of[uid] = brute[uid] if brute is not None else by[uid].status
    counts: dict[str, int] = defaultdict(int)
    for s in status_of.values():
        counts[s] += 1
    k = int(round(u_cert * kappa))
    if kappa > 0 and k == 0:
        k = 1
    assert counts["guaranteed_in"] + counts["guaranteed_out"] + counts["ambiguous"] == u_cert
    return {
        "kappa": kappa,
        "n_frame": n_frame,
        "n_eligible": u_cert,
        "certificate_coverage": coverage,
        "seat_count": k,
        "guaranteed_in": counts["guaranteed_in"],
        "guaranteed_out": counts["guaranteed_out"],
        "ambiguous": counts["ambiguous"],
        "guaranteed_in_rate": counts["guaranteed_in"] / u_cert,
        "guaranteed_out_rate": counts["guaranteed_out"] / u_cert,
        "ambiguous_rate": counts["ambiguous"] / u_cert,
        "membership_method": "discrete_brute_force" if brute is not None else "interval_robust_rank_conservative",
        "statuses": status_of,
    }


def replacement_diagnostic(
    user_r0: dict[str, float], user_r1: dict[str, float], kappa: float
) -> dict[str, Any]:
    users = sorted(set(user_r0) & set(user_r1))
    if not users:
        return {"n_users": 0, "label": "REFERENCE_RELATIVE_PIPELINE_DIAGNOSTIC"}
    users_arr = np.array(users, dtype=object)
    r0 = np.array([1.0 - user_r0[u] for u in users], dtype=np.float64)
    r1 = np.array([1.0 - user_r1[u] for u in users], dtype=np.float64)
    sel = select_capacity_list(users_arr, r0, capacity=kappa)
    ref = select_capacity_list(users_arr, r1, capacity=kappa)
    inter = len(set(sel) & set(ref))
    repl = 1.0 - inter / max(len(ref), 1)
    return {
        "kappa": kappa,
        "n_users": len(users),
        "seats": len(ref),
        "changed_seats": len(ref) - inter,
        "replacement": repl,
        "label": "REFERENCE_RELATIVE_PIPELINE_DIAGNOSTIC",
    }


def seal_payload(payload: dict[str, Any], sealed_dir: Path) -> dict[str, Any]:
    sealed_dir.mkdir(parents=True, exist_ok=True)
    key_path = sealed_dir / ".r3_seal_key"
    if key_path.exists():
        key = key_path.read_bytes().strip()
    else:
        key = Fernet.generate_key()
        key_path.write_bytes(key + b"\n")
    f = Fernet(key)
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    blob = f.encrypt(raw)
    bin_path = sealed_dir / "wdlf_r3_effects.bin"
    bin_path.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    (sealed_dir / "wdlf_r3_effects.sha256").write_text(digest + "\n", encoding="utf-8")
    meta = {
        "created_utc": utc_now(),
        "sha256": digest,
        "bytes": len(blob),
        "plaintext_effects_remain": False,
        "schema": "wdlf_r3_local_conditional_v3",
    }
    write_json(sealed_dir / "pre_unseal_manifest.json", meta)
    return meta


def unseal_payload(sealed_dir: Path) -> dict[str, Any]:
    key = (sealed_dir / ".r3_seal_key").read_bytes().strip()
    f = Fernet(key)
    blob = (sealed_dir / "wdlf_r3_effects.bin").read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    expected = (sealed_dir / "wdlf_r3_effects.sha256").read_text(encoding="utf-8").strip()
    if digest != expected:
        raise RuntimeError("sealed checksum mismatch")
    return json.loads(f.decrypt(blob).decode("utf-8"))


def structural_eligibility(df: pd.DataFrame, compat: dict) -> dict[str, Any]:
    n_conf = int(df.loc[df["split"] == "confirmatory", "UserId"].nunique())
    n_el = 0
    for _, g in df.loc[df["split"] == "confirmatory"].groupby("UserId", sort=False):
        for _, row in g.iterrows():
            key = ("EEDI-R0", str(int(row["QuestionId"])), str(row["SubjectId"]), int(row["IsCorrect"]))
            if key in compat:
                n_el += 1
                break
    return {
        "n_conf_frame": n_conf,
        "n_eligible": n_el,
        "coverage": n_el / n_conf if n_conf else None,
        "compat_keys": len(compat),
        "note": "support-only metadata; no decision effects",
    }


def run_campaign(target_splits: set[str], *, phase: str) -> dict[str, Any]:
    log_phase(f"phase={phase} loading data")
    t0 = time.time()
    df = load_frame()
    n_frame = int(df.loc[df["split"].isin(list(target_splits)), "UserId"].nunique())
    n_conf_frame = int(df.loc[df["split"] == "confirmatory", "UserId"].nunique())
    dev = df.loc[df["split"] == "development"]
    log_phase("building development dictionary")
    compat = build_compatible_dict(dev)
    dict_hash = hashlib.sha256(
        json.dumps({str(k): v["options"] for k, v in sorted(compat.items(), key=lambda kv: str(kv[0]))}, sort_keys=True).encode()
    ).hexdigest()
    log_phase(f"compat_keys={len(compat)} dict_hash={dict_hash[:16]}")
    log_phase("loading frozen models")
    pfa_r1, pfa_r0, dkt_r1, dkt_r0, hashes = load_models()
    hashes["dictionary"] = dict_hash

    struct = structural_eligibility(df, compat)
    log_phase(f"structural_eligible_users={struct['n_eligible']}/{struct['n_conf_frame']}")

    target_users = [
        (uid, g) for uid, g in df.groupby("UserId", sort=False) if str(g["split"].iloc[0]) in target_splits
    ]
    log_phase(f"users_to_process={len(target_users)}")

    all_certs: list[EventCert] = []
    pfa_p2: list[dict[str, Any]] = []
    dkt_p2: list[dict[str, Any]] = []
    pfa_r0_means: dict[str, float] = {}
    pfa_r1_means: dict[str, float] = {}
    dkt_r0_means: dict[str, float] = {}
    dkt_r1_means: dict[str, float] = {}

    for bi, (uid, g) in enumerate(target_users):
        try:
            certs, last_p, m0, m1 = pfa_walk_user(
                g,
                pfa_r1=pfa_r1,
                pfa_r0=pfa_r0,
                compat=compat,
                target_splits=target_splits,
                thresholds=[PRIMARY_TAU] + TAU_SENS,
            )
            all_certs.extend(certs)
            if last_p is not None:
                pfa_p2.append(last_p)
            if m0 is not None:
                pfa_r0_means[str(uid)] = m0
            if m1 is not None:
                pfa_r1_means[str(uid)] = m1

            certs_d, last_d, md0, md1 = dkt_walk_user(
                g,
                dkt_r1=dkt_r1,
                dkt_r0=dkt_r0,
                compat=compat,
                target_splits=target_splits,
                thresholds=[PRIMARY_TAU] + TAU_SENS,
            )
            all_certs.extend(certs_d)
            if last_d is not None:
                dkt_p2.append(last_d)
            if md0 is not None:
                dkt_r0_means[str(uid)] = md0
            if md1 is not None:
                dkt_r1_means[str(uid)] = md1
        except Exception as exc:  # noqa: BLE001 — surface failing user then abort
            log_phase(f"ERROR user={uid} batch_index={bi} err={type(exc).__name__}: {exc}")
            raise

        if bi % 50 == 0:
            log_phase(f"batch_index={bi} users={len(target_users)} elapsed_s={time.time() - t0:.1f}")

    # P2 frame for confirmatory is always 1021; for cal dry-run use calibration |U|
    p2_frame = n_conf_frame if "confirmatory" in target_splits else n_frame
    p1 = {
        "PFA-Option": {str(t): summarize_p1(all_certs, "PFA-Option", t) for t in [PRIMARY_TAU] + TAU_SENS},
        "CategoricalDKT": {
            str(t): summarize_p1(all_certs, "CategoricalDKT", t) for t in [PRIMARY_TAU] + TAU_SENS
        },
    }
    p2 = {
        "PFA-Option": {str(k): run_p2(pfa_p2, kappa=k, n_frame=p2_frame) for k in [PRIMARY_KAPPA] + KAPPA_SENS},
        "CategoricalDKT": {
            str(k): run_p2(dkt_p2, kappa=k, n_frame=p2_frame) for k in [PRIMARY_KAPPA] + KAPPA_SENS
        },
    }
    diag = {
        "PFA-Option": {
            str(k): replacement_diagnostic(pfa_r0_means, pfa_r1_means, k) for k in [PRIMARY_KAPPA] + KAPPA_SENS
        },
        "CategoricalDKT": {
            str(k): replacement_diagnostic(dkt_r0_means, dkt_r1_means, k)
            for k in [PRIMARY_KAPPA] + KAPPA_SENS
        },
    }

    compat_sizes = [
        c.n_candidates
        for c in all_certs
        if c.model == "PFA-Option" and abs(c.threshold - PRIMARY_TAU) < 1e-12 and c.supported
    ]
    # P1 label exhaustiveness check (engineering; not an effect print)
    for model in ("PFA-Option", "CategoricalDKT"):
        cell = p1[model][str(PRIMARY_TAU)]
        accounted = (
            cell.get("unsupported", 0)
            + cell.get("stable", 0)
            + cell.get("set_valued", 0)
            + cell.get("uncertified", 0)
            + cell.get("abstain", 0)
        )
        if cell.get("n_events", 0) and accounted != cell["n_events"]:
            raise RuntimeError(f"P1 accounting incomplete for {model}")

    return {
        "phase": phase,
        "created_utc": utc_now(),
        "level": "A",
        "estimand": "local_conditional_release_adequacy",
        "population_contract": "OPTION_A_CERTIFICATE_ELIGIBLE",
        "score_contract": "post_ingest_mastery_proxy_no_future_item",
        "hashes": hashes,
        "structural": struct,
        "p1": p1,
        "p2": p2,
        "secondary_reference_diagnostic": diag,
        "compat_size_summary": {
            "n": len(compat_sizes),
            "mean": float(np.mean(compat_sizes)) if compat_sizes else None,
            "min": int(min(compat_sizes)) if compat_sizes else None,
            "max": int(max(compat_sizes)) if compat_sizes else None,
        },
        "runtime_s": time.time() - t0,
        "device": str(next(dkt_r1.parameters()).device),
        "n_certs": len(all_certs),
        "n_users_processed": len(target_users),
        "n_frame_processed": n_frame,
    }


def assert_protocol_frozen() -> None:
    tags = subprocess.check_output(["git", "tag", "-l", "wdlf-auth-r3-protocol-frozen"], text=True).strip()
    if not tags:
        raise RuntimeError("Refuse confirmatory: tag wdlf-auth-r3-protocol-frozen missing")
    if not MANIFEST.exists():
        raise RuntimeError("manifest v3 missing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "phase",
        choices=["dry-run-cal", "confirmatory", "unseal", "structural-eligibility"],
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.phase == "structural-eligibility":
        df = load_frame()
        compat = build_compatible_dict(df.loc[df["split"] == "development"])
        payload = structural_eligibility(df, compat)
        write_json(OUT / "structural_eligibility.json", payload)
        log_phase(f"eligibility {payload['n_eligible']}/{payload['n_conf_frame']}")
        return

    if args.phase == "dry-run-cal":
        payload = run_campaign({"calibration"}, phase="dry-run-cal")
        write_json(OUT / "dry_run_cal_results.json", payload)
        log_phase(f"dry-run-cal complete runtime_s={payload['runtime_s']:.1f}")
        log_phase(f"n_certs={payload['n_certs']} device={payload['device']}")
        return

    if args.phase == "confirmatory":
        assert_protocol_frozen()
        payload = run_campaign({"confirmatory"}, phase="confirmatory")
        meta = seal_payload(payload, SEALED)
        write_json(
            OUT / "execution_record.json",
            {
                "phase": "confirmatory",
                "created_utc": utc_now(),
                "runtime_s": payload["runtime_s"],
                "device": payload["device"],
                "n_certs": payload["n_certs"],
                "n_users_processed": payload["n_users_processed"],
                "hashes": payload["hashes"],
                "structural": payload["structural"],
                "seal": meta,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "freeze_tag": "wdlf-auth-r3-protocol-frozen",
            },
        )
        log_phase(f"confirmatory sealed sha={meta['sha256'][:16]} runtime_s={payload['runtime_s']:.1f}")
        return

    if args.phase == "unseal":
        payload = unseal_payload(SEALED)
        RESULTS.mkdir(parents=True, exist_ok=True)
        write_json(RESULTS / "primary_results.json", payload)
        rows_p1 = []
        for model, by_t in payload["p1"].items():
            for thr, cell in by_t.items():
                rows_p1.append({"model": model, "threshold": thr, **cell})
        pd.DataFrame(rows_p1).to_csv(RESULTS / "p1_results.csv", index=False)
        rows_p2 = []
        for model, by_k in payload["p2"].items():
            for kap, cell in by_k.items():
                rows_p2.append(
                    {
                        "model": model,
                        "kappa": kap,
                        "n_eligible": cell.get("n_eligible"),
                        "certificate_coverage": cell.get("certificate_coverage"),
                        "seat_count": cell.get("seat_count"),
                        "guaranteed_in": cell.get("guaranteed_in"),
                        "guaranteed_out": cell.get("guaranteed_out"),
                        "ambiguous": cell.get("ambiguous"),
                        "guaranteed_in_rate": cell.get("guaranteed_in_rate"),
                        "guaranteed_out_rate": cell.get("guaranteed_out_rate"),
                        "ambiguous_rate": cell.get("ambiguous_rate"),
                        "membership_method": cell.get("membership_method"),
                    }
                )
        pd.DataFrame(rows_p2).to_csv(RESULTS / "p2_results.csv", index=False)
        rows_d = []
        for model, by_k in payload["secondary_reference_diagnostic"].items():
            for kap, cell in by_k.items():
                rows_d.append({"model": model, **cell})
        pd.DataFrame(rows_d).to_csv(RESULTS / "secondary_reference_diagnostic.csv", index=False)
        # sensitivity alias
        pd.DataFrame(rows_p1).to_csv(RESULTS / "sensitivity_results.csv", index=False)
        write_json(RESULTS / "sensitivity_results.json", {"p1": payload["p1"], "p2": payload["p2"]})
        write_json(
            RESULTS / "inference_results.json",
            {
                "mode": "finite_population_primary",
                "bootstrap": "not_run_primary",
                "note": "Primary reporting is exact finite-population counts/rates per frozen plan.",
            },
        )
        log_phase("unseal complete -> results/wdlf_r3")
        return


if __name__ == "__main__":
    main()
