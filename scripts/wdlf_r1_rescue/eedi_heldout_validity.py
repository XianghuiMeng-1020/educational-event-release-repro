#!/usr/bin/env python3
"""Eedi held-out empirical validity audit (Authentic WDLF R1 rescue).

CRITICAL: confirmatory rich traces are used ONLY to evaluate certificates
after C_dev construction. Confirmatory AnswerValue/options never enter C_dev.
Primary tau=0.80, kappa=0.05 remain frozen.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r3"))

from run_r3_local_conditional import (  # noqa: E402
    PRIMARY_KAPPA,
    PRIMARY_TAU,
    N_QUESTIONS,
    build_compatible_dict,
    load_frame,
    load_models,
    run_p2,
    sha256_file,
)
from src.identification.compatible_trace.certificate_v2 import certify_from_propagated_scores
from src.identification.compatible_trace.dkt_propagation import _next_step_proba_from_state
from src.identification.compatible_trace.pfa_propagation import (
    PFAState,
    pfa_ingest_event,
    pfa_next_step_proba,
)
from src.models.categorical_dkt import interaction_token
from src.policies.capacity_support_v2 import select_capacity_list
from src.policies.mastery_policy import mastery_decision

OUT = ROOT / "results" / "wdlf_r1_rescue"
TAU = PRIMARY_TAU
KAPPA = PRIMARY_KAPPA


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _p2_false_cert(
    statuses: dict[str, str],
    actual_risks: dict[str, float],
    kappa: float,
) -> dict[str, Any]:
    """Compare interval certificates to selection under actual rich-trace risks."""
    users = sorted(actual_risks)
    if not users:
        return {
            "n_eligible": 0,
            "n_guaranteed_in": 0,
            "n_guaranteed_out": 0,
            "n_ambiguous": 0,
            "false_gin": 0,
            "false_gout": 0,
            "false_gin_rate": None,
            "false_gout_rate": None,
            "false_cert_rate_among_resolved": None,
        }
    arr_u = np.array(users, dtype=object)
    arr_r = np.array([actual_risks[u] for u in users], dtype=np.float64)
    selected = set(select_capacity_list(arr_u, arr_r, capacity=kappa))
    gin = [u for u in users if statuses.get(u) == "guaranteed_in"]
    gout = [u for u in users if statuses.get(u) in {"guaranteed_out"}]
    amb = [u for u in users if statuses.get(u) == "ambiguous"]
    false_gin = sum(1 for u in gin if u not in selected)
    false_gout = sum(1 for u in gout if u in selected)
    n_resolved = len(gin) + len(gout)
    return {
        "n_eligible": len(users),
        "n_guaranteed_in": len(gin),
        "n_guaranteed_out": len(gout),
        "n_ambiguous": len(amb),
        "false_gin": false_gin,
        "false_gout": false_gout,
        "false_gin_rate": (false_gin / len(gin)) if gin else None,
        "false_gout_rate": (false_gout / len(gout)) if gout else None,
        "false_cert_rate_among_resolved": ((false_gin + false_gout) / n_resolved) if n_resolved else None,
        "note": "unresolved/ambiguous excluded from false-cert denominator; truth=top-k on actual risks",
    }


def pfa_validity_walk(
    g: pd.DataFrame,
    *,
    pfa_r1,
    compat: dict,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    uid = str(g["UserId"].iloc[0])
    succ: dict[Any, float] = defaultdict(float)
    fail: dict[Any, float] = defaultdict(float)
    opt: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    rows: list[dict[str, Any]] = []
    last_ckpt: dict[str, Any] | None = None

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
        key = ("EEDI-R0", str(q), str(subj), int(y))
        entry = compat.get(key)
        if entry is None:
            rows.append(
                {
                    "user_id": uid,
                    "event_index": i,
                    "supported": 0,
                    "status": "unsupported",
                    "actual_in_cdev": None,
                    "true_action": None,
                    "actions": (),
                    "contained": None,
                    "false_stable": None,
                    "n_cdev": 0,
                }
            )
        else:
            options = list(entry["options"])
            # INVARIANT: certificate uses C_dev options only.
            pre = PFAState(succ[subj], fail[subj], o[0], o[1], o[2], o[3])
            scores = [
                pfa_next_step_proba(
                    pfa_ingest_event(pre, is_correct=y, answer_value=a, use_option_covariates=True),
                    pfa_r1,
                )
                for a in options
            ]
            cert = certify_from_propagated_scores(str(key), scores, threshold=TAU, search_mode="exact")
            # Evaluate with actual confirmatory completion (not injected into C_dev).
            true_score = pfa_next_step_proba(
                pfa_ingest_event(pre, is_correct=y, answer_value=av, use_option_covariates=True),
                pfa_r1,
            )
            true_action = mastery_decision(true_score, TAU)
            actual_in = int(av in options)
            contained = int(true_action in cert.actions) if cert.actions else 0
            false_stable = int(cert.status == "stable" and true_action not in cert.actions)
            rows.append(
                {
                    "user_id": uid,
                    "event_index": i,
                    "supported": 1,
                    "status": cert.status,
                    "actual_in_cdev": actual_in,
                    "true_action": true_action,
                    "actions": cert.actions,
                    "contained": contained,
                    "false_stable": false_stable,
                    "n_cdev": len(options),
                    "risk_min": float(min(1.0 - s for s in scores)),
                    "risk_max": float(max(1.0 - s for s in scores)),
                    "actual_risk": float(1.0 - true_score),
                }
            )
            last_ckpt = {
                "user_id": uid,
                "risks": [1.0 - s for s in scores],
                "scores": scores,
                "actual_risk": float(1.0 - true_score),
                "actual_in_cdev": actual_in,
                "event_index": i,
            }

        if y == 1:
            succ[subj] += 1.0
        else:
            fail[subj] += 1.0
        if 1 <= av <= 4:
            opt[subj][av - 1] += 1.0

    return rows, last_ckpt


@torch.no_grad()
def dkt_validity_walk(
    g: pd.DataFrame,
    *,
    dkt_r1,
    compat: dict,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    uid = str(g["UserId"].iloc[0])
    device = next(dkt_r1.parameters()).device
    qs = g["QuestionId"].astype(int).tolist()
    ys = g["IsCorrect"].astype(int).tolist()
    avs = g["AnswerValue"].astype(int).tolist()
    subjs = g["SubjectId"].tolist()
    rows: list[dict[str, Any]] = []
    last_ckpt: dict[str, Any] | None = None

    h = torch.zeros(1, 1, dkt_r1.cfg.hidden, device=device)
    c = torch.zeros(1, 1, dkt_r1.cfg.hidden, device=device)

    for i in range(len(g)):
        q, y, av, subj = qs[i], ys[i], avs[i], subjs[i]
        key = ("EEDI-R0", str(q), str(subj), int(y))
        entry = compat.get(key)
        if entry is None:
            rows.append(
                {
                    "user_id": uid,
                    "event_index": i,
                    "supported": 0,
                    "status": "unsupported",
                    "actual_in_cdev": None,
                    "true_action": None,
                    "actions": (),
                    "contained": None,
                    "false_stable": None,
                    "n_cdev": 0,
                }
            )
        else:
            options = list(entry["options"])
            scores = []
            for a in options:
                tok = interaction_token(
                    q, representation="EEDI-R1", answer_value=int(a), n_questions=N_QUESTIONS
                )
                emb = dkt_r1.embedding(torch.tensor([[tok]], dtype=torch.long, device=device))
                _, (h2, c2) = dkt_r1.lstm(emb, (h, c))
                scores.append(_next_step_proba_from_state(dkt_r1, h2, c2))
            cert = certify_from_propagated_scores(str(key), scores, threshold=TAU, search_mode="exact")
            tok_true = interaction_token(
                q, representation="EEDI-R1", answer_value=int(av), n_questions=N_QUESTIONS
            )
            emb_t = dkt_r1.embedding(torch.tensor([[tok_true]], dtype=torch.long, device=device))
            _, (ht, ct) = dkt_r1.lstm(emb_t, (h, c))
            true_score = _next_step_proba_from_state(dkt_r1, ht, ct)
            true_action = mastery_decision(true_score, TAU)
            actual_in = int(av in options)
            contained = int(true_action in cert.actions) if cert.actions else 0
            false_stable = int(cert.status == "stable" and true_action not in cert.actions)
            rows.append(
                {
                    "user_id": uid,
                    "event_index": i,
                    "supported": 1,
                    "status": cert.status,
                    "actual_in_cdev": actual_in,
                    "true_action": true_action,
                    "actions": cert.actions,
                    "contained": contained,
                    "false_stable": false_stable,
                    "n_cdev": len(options),
                    "risk_min": float(min(1.0 - s for s in scores)),
                    "risk_max": float(max(1.0 - s for s in scores)),
                    "actual_risk": float(1.0 - true_score),
                }
            )
            last_ckpt = {
                "user_id": uid,
                "risks": [1.0 - s for s in scores],
                "scores": scores,
                "actual_risk": float(1.0 - true_score),
                "actual_in_cdev": actual_in,
                "event_index": i,
            }

        # Advance with observed confirmatory event (history reconstruction only).
        tok_obs = interaction_token(
            q, representation="EEDI-R1", answer_value=av, n_questions=N_QUESTIONS
        )
        emb = dkt_r1.embedding(torch.tensor([[tok_obs]], dtype=torch.long, device=device))
        _, (h, c) = dkt_r1.lstm(emb, (h, c))

    return rows, last_ckpt


def summarize_p1(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    n = len(rows)
    unsupported = sum(1 for r in rows if not r["supported"])
    supported = n - unsupported
    stable = sum(1 for r in rows if r["status"] == "stable")
    setv = sum(1 for r in rows if r["status"] == "set-valued")
    actual_in = [r for r in rows if r["supported"] and r["actual_in_cdev"] is not None]
    contained = [r for r in rows if r["supported"] and r["contained"] is not None]
    false_st = [r for r in rows if r["status"] == "stable"]
    n_actual_in = sum(int(r["actual_in_cdev"]) for r in actual_in)
    n_contained = sum(int(r["contained"]) for r in contained)
    n_false_stable = sum(int(r["false_stable"] or 0) for r in false_st)
    return {
        "model": model,
        "tau": TAU,
        "n_events": n,
        "unsupported": unsupported,
        "supported": supported,
        "stable": stable,
        "set_valued": setv,
        "unsupported_rate": unsupported / n if n else None,
        "supported_rate": supported / n if n else None,
        "stable_rate": stable / n if n else None,
        "set_valued_rate": setv / n if n else None,
        "stable_given_supported": (stable / supported) if supported else None,
        "actual_answer_in_cdev": n_actual_in,
        "actual_answer_in_cdev_rate_supported": (n_actual_in / supported) if supported else None,
        "action_contained": n_contained,
        "action_containment_rate_supported": (n_contained / supported) if supported else None,
        "false_stable": n_false_stable,
        "false_stable_rate_among_stable": (n_false_stable / stable) if stable else None,
    }


def per_user_aggregates(rows: list[dict[str, Any]], ckpts: list[dict[str, Any]], p2: dict[str, Any]) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        uid = r["user_id"]
        u = by.setdefault(
            uid,
            {
                "n_events": 0,
                "n_supported": 0,
                "n_unsupported": 0,
                "n_stable": 0,
                "n_set_valued": 0,
                "n_actual_in_cdev": 0,
                "n_action_contained": 0,
                "n_false_stable": 0,
            },
        )
        u["n_events"] += 1
        if r["supported"]:
            u["n_supported"] += 1
            u["n_actual_in_cdev"] += int(r["actual_in_cdev"] or 0)
            u["n_action_contained"] += int(r["contained"] or 0)
            if r["status"] == "stable":
                u["n_stable"] += 1
                u["n_false_stable"] += int(r["false_stable"] or 0)
            elif r["status"] == "set-valued":
                u["n_set_valued"] += 1
        else:
            u["n_unsupported"] += 1
    statuses = p2.get("statuses") or {}
    ck_by = {c["user_id"]: c for c in ckpts}
    for uid, u in by.items():
        ck = ck_by.get(uid)
        u["p2_eligible"] = uid in statuses
        u["p2_status"] = statuses.get(uid)
        if ck is not None:
            u["p2_risk_min"] = float(min(ck["risks"]))
            u["p2_risk_max"] = float(max(ck["risks"]))
            u["p2_actual_risk"] = float(ck["actual_risk"])
        else:
            u["p2_risk_min"] = None
            u["p2_risk_max"] = None
            u["p2_actual_risk"] = None
    return by


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[eedi_heldout] loading frame", flush=True)
    df = load_frame()
    conf = df.loc[df["split"] == "confirmatory"]
    n_users = int(conf["UserId"].nunique())
    n_events = int(len(conf))
    print(f"[eedi_heldout] confirmatory users={n_users} events={n_events}", flush=True)

    print("[eedi_heldout] building development-only C_dev", flush=True)
    compat = build_compatible_dict(df.loc[df["split"] == "development"])
    dict_hash = hashlib.sha256(
        json.dumps(
            {str(k): v["options"] for k, v in sorted(compat.items(), key=lambda kv: str(kv[0]))},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    print(f"[eedi_heldout] C_dev keys={len(compat)} hash={dict_hash[:16]}", flush=True)

    pfa_r1, _pfa_r0, dkt_r1, _dkt_r0, hashes = load_models()
    hashes["dictionary"] = dict_hash

    target_users = [
        (uid, g) for uid, g in conf.groupby("UserId", sort=False)
    ]

    model_rows: dict[str, list[dict[str, Any]]] = {"PFA-Option": [], "CategoricalDKT": []}
    model_ckpts: dict[str, list[dict[str, Any]]] = {"PFA-Option": [], "CategoricalDKT": []}

    for bi, (uid, g) in enumerate(target_users):
        if bi and bi % 100 == 0:
            print(f"[eedi_heldout] users={bi}/{n_users} elapsed_s={time.time()-t0:.1f}", flush=True)
        rows_p, ck_p = pfa_validity_walk(g, pfa_r1=pfa_r1, compat=compat)
        model_rows["PFA-Option"].extend(rows_p)
        if ck_p is not None:
            model_ckpts["PFA-Option"].append(ck_p)
        rows_d, ck_d = dkt_validity_walk(g, dkt_r1=dkt_r1, compat=compat)
        model_rows["CategoricalDKT"].extend(rows_d)
        if ck_d is not None:
            model_ckpts["CategoricalDKT"].append(ck_d)

    summaries = {}
    p2_block = {}
    user_aggs = {}
    for model in ("PFA-Option", "CategoricalDKT"):
        p1 = summarize_p1(model_rows[model], model)
        p2 = run_p2(model_ckpts[model], kappa=KAPPA, n_frame=n_users)
        # strip bulky statuses from printed p2 later; keep for false-cert
        actual_risks = {c["user_id"]: float(c["actual_risk"]) for c in model_ckpts[model]}
        false_p2 = _p2_false_cert(p2.get("statuses") or {}, actual_risks, KAPPA)
        # actual risk outside certified interval
        outside = 0
        for c in model_ckpts[model]:
            lo, hi = min(c["risks"]), max(c["risks"])
            ar = float(c["actual_risk"])
            if ar < lo - 1e-12 or ar > hi + 1e-12:
                outside += 1
        false_p2["actual_risk_outside_interval"] = outside
        false_p2["actual_risk_outside_interval_rate"] = (
            outside / len(model_ckpts[model]) if model_ckpts[model] else None
        )
        no_ckpt = n_users - p2["n_eligible"]
        p2_summary = {
            "kappa": KAPPA,
            "n_frame": n_users,
            "n_eligible": p2["n_eligible"],
            "certificate_coverage": p2["certificate_coverage"],
            "seat_count": p2["seat_count"],
            "guaranteed_in": p2["guaranteed_in"],
            "guaranteed_out": p2["guaranteed_out"],
            "ambiguous": p2["ambiguous"],
            "ambiguous_rate_eligible": p2["ambiguous_rate"],
            "unresolved": p2["ambiguous"],
            "unresolved_rate_eligible": p2["ambiguous_rate"],
            "p2_no_checkpoint_users": no_ckpt,
            "p2_all_user_noncertification_rate": (p2["ambiguous"] + no_ckpt) / n_users if n_users else None,
            "membership_method": p2["membership_method"],
            "external_validity": false_p2,
        }
        summaries[model] = {"p1": p1, "p2": p2_summary}
        p2_block[model] = p2
        user_aggs[model] = per_user_aggregates(model_rows[model], model_ckpts[model], p2)

    runtime_s = time.time() - t0
    payload = {
        "created_utc": utc_now(),
        "protocol": {
            "tau": TAU,
            "kappa": KAPPA,
            "c_dev_source": "development_split_only",
            "confirmatory_role": "evaluate_certificates_only",
            "invariant": "confirmatory AnswerValue never enters C_dev construction",
        },
        "population": {"n_users": n_users, "n_events": n_events, "compat_keys": len(compat)},
        "hashes": hashes,
        "models": summaries,
        "runtime_s": runtime_s,
    }
    out_json = OUT / "eedi_heldout_validity.json"
    write_json(out_json, payload)

    # CSV summaries
    csv_path = OUT / "eedi_heldout_validity_p1.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "tau",
                "n_events",
                "supported",
                "unsupported",
                "stable",
                "set_valued",
                "actual_answer_in_cdev_rate_supported",
                "action_containment_rate_supported",
                "false_stable_rate_among_stable",
            ],
        )
        w.writeheader()
        for model, blk in summaries.items():
            p1 = blk["p1"]
            w.writerow({k: p1.get(k) for k in w.fieldnames})

    csv_p2 = OUT / "eedi_heldout_validity_p2.csv"
    with csv_p2.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "kappa",
                "n_eligible",
                "guaranteed_in",
                "guaranteed_out",
                "unresolved",
                "p2_all_user_noncertification_rate",
                "false_gin_rate",
                "false_gout_rate",
                "false_cert_rate_among_resolved",
                "actual_risk_outside_interval_rate",
            ],
        )
        w.writeheader()
        for model, blk in summaries.items():
            p2 = blk["p2"]
            ev = p2["external_validity"]
            w.writerow(
                {
                    "model": model,
                    "kappa": KAPPA,
                    "n_eligible": p2["n_eligible"],
                    "guaranteed_in": p2["guaranteed_in"],
                    "guaranteed_out": p2["guaranteed_out"],
                    "unresolved": p2["unresolved"],
                    "p2_all_user_noncertification_rate": p2["p2_all_user_noncertification_rate"],
                    "false_gin_rate": ev["false_gin_rate"],
                    "false_gout_rate": ev["false_gout_rate"],
                    "false_cert_rate_among_resolved": ev["false_cert_rate_among_resolved"],
                    "actual_risk_outside_interval_rate": ev["actual_risk_outside_interval_rate"],
                }
            )

    # User aggregates for bootstrap (include statuses/risks; drop event-level bulk)
    boot_path = OUT / "eedi_heldout_user_aggregates.json"
    write_json(
        boot_path,
        {
            "tau": TAU,
            "kappa": KAPPA,
            "n_frame_users": n_users,
            "models": user_aggs,
        },
    )

    print(json.dumps({"wrote": str(out_json), "runtime_s": runtime_s, "summaries": summaries}, indent=2, default=str))
    print(f"[eedi_heldout] DONE runtime_s={runtime_s:.1f}", flush=True)


if __name__ == "__main__":
    main()
