#!/usr/bin/env python3
"""Predictive ablation comparator for Authentic WDLF rescue (protocol-frozen)."""
from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from r7_core import Event, TransparentActionPFA  # noqa: E402
from src.data.eedi_split import assign_eedi_split  # noqa: E402
from src.models.pfa_option import PFAOption, PFAOptionConfig, PFAOptionParams  # noqa: E402

OUT = ROOT / "results" / "wdlf_r1_rescue"
PROTOCOL = OUT / "predictive_ablation_protocol.json"
REPORT = ROOT / "reports" / "WDLF_R1_PREDICTIVE_ABLATION.md"
SEALED = ROOT / "prospective" / "wdlf_r7" / "sealed"
EEDI_DATA = ROOT / "prospective" / "g6p" / "data" / "cache" / "tasks_3_4_joined_min.parquet"
PFA_CKPT = ROOT / "prospective" / "g6p" / "checkpoints" / "pfa_option_params.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[pred-ablation] {msg}", flush=True)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(float)
    p = np.clip(p.astype(float), 1e-12, 1.0 - 1e-12)
    # AUC (Mann–Whitney)
    pos = p[y == 1.0]
    neg = p[y == 0.0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        # exact rank AUC
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(p) + 1, dtype=float)
        # average ties
        sorted_p = p[order]
        i = 0
        while i < len(p):
            j = i
            while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
                j += 1
            if j > i:
                avg = 0.5 * (i + 1 + j + 1)
                ranks[order[i : j + 1]] = avg
            i = j + 1
        auc = float((ranks[y == 1.0].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))
    nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    brier = float(np.mean((p - y) ** 2))
    # 10-bin ECE
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.any():
            ece += float(m.mean() * abs(y[m].mean() - p[m].mean()))
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": auc,
        "log_loss": nll,
        "brier": brier,
        "ece_10bin": float(ece),
    }


def update_s0(model: TransparentActionPFA, ev: Event) -> None:
    st = model.skill_stats[ev.skill_id]
    st["n"] += 1.0
    st["c"] += float(ev.correct)


def update_s1(model: TransparentActionPFA, ev: Event) -> None:
    st = model.skill_stats[ev.skill_id]
    st["n"] += 1.0
    st["c"] += float(ev.correct)
    st["hints"] += float(ev.hint_count)
    st["attempts"] += float(max(ev.attempt_count, 1))


def assistments_predictive() -> dict[str, Any]:
    log("ASSISTments: load confirmatory split")
    t0 = time.time()
    with (SEALED / "splits" / "confirmatory.pkl").open("rb") as f:
        events: list[Event] = pickle.load(f)
    by_user: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    users = sorted(by_user)
    log(f"ASSISTments: users={len(users)} events={len(events)} load_s={time.time()-t0:.1f}")

    preds = {"S0": [], "S1": []}
    labels: list[int] = []
    # paired walk: same labels for both variants
    for ui, uid in enumerate(users):
        if ui and ui % 1000 == 0:
            log(f"ASSISTments progress users={ui}/{len(users)}")
        hist = by_user[uid]
        m0 = TransparentActionPFA()
        m1 = TransparentActionPFA()
        for t, ev in enumerate(hist):
            if t == 0:
                update_s0(m0, ev)
                update_s1(m1, ev)
                continue
            p0 = m0.mastery(ev.skill_id)
            p1 = m1.mastery(ev.skill_id)
            preds["S0"].append(p0)
            preds["S1"].append(p1)
            labels.append(ev.correct)
            update_s0(m0, ev)
            update_s1(m1, ev)

    y = np.asarray(labels, dtype=float)
    out = {}
    for schema in ("S0", "S1"):
        p = np.asarray(preds[schema], dtype=float)
        out[schema] = binary_metrics(y, p)
    out["delta_S1_minus_S0"] = {
        k: (out["S1"][k] - out["S0"][k])
        for k in ("roc_auc", "log_loss", "brier", "ece_10bin")
        if not (isinstance(out["S0"][k], float) and math.isnan(out["S0"][k]))
    }
    out["runtime_s"] = time.time() - t0
    out["n_users"] = len(users)
    out["n_pred_events"] = int(len(labels))
    return out


def load_frozen_assistments_certs() -> dict[str, Any]:
    p1_path = ROOT / "results" / "wdlf_r7" / "p1_by_schema_model.csv"
    p2_path = ROOT / "results" / "wdlf_r7" / "p2_by_schema_model.csv"
    rows_p1 = {}
    with p1_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == "TransparentActionPFA" and r["schema"] in {"S0", "S1", "S2", "S3"}:
                rows_p1[r["schema"]] = {
                    "support_rate": float(r["support_rate"]),
                    "event_noncertification_rate": float(r["event_noncertification_rate"]),
                    "event_certified_stable_coverage": float(r["event_certified_stable_coverage"]),
                }
    rows_p2 = {}
    with p2_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == "TransparentActionPFA" and r["schema"] in {"S0", "S1", "S2", "S3"}:
                rows_p2[r["schema"]] = {
                    "p2_all_user_noncertification_rate": float(r["p2_all_user_noncertification_rate"]),
                    "p2_unresolved": int(float(r["p2_unresolved"])),
                    "p2_eligible_users": int(float(r["p2_eligible_users"])),
                }
    return {
        "source_p1": str(p1_path.relative_to(ROOT)),
        "source_p2": str(p2_path.relative_to(ROOT)),
        "by_schema": {
            s: {**rows_p1[s], **rows_p2[s]} for s in ("S0", "S1", "S2", "S3") if s in rows_p1 and s in rows_p2
        },
        "delta_S1_minus_S0": {
            "event_noncertification_rate": rows_p1["S1"]["event_noncertification_rate"]
            - rows_p1["S0"]["event_noncertification_rate"],
            "p2_all_user_noncertification_rate": rows_p2["S1"]["p2_all_user_noncertification_rate"]
            - rows_p2["S0"]["p2_all_user_noncertification_rate"],
            "support_rate": rows_p1["S1"]["support_rate"] - rows_p1["S0"]["support_rate"],
        },
    }


def eedi_predictive() -> dict[str, Any]:
    import pandas as pd

    log("Eedi: load frame + frozen PFA-Option")
    t0 = time.time()
    blob = json.loads(PFA_CKPT.read_text(encoding="utf-8"))
    m0 = PFAOption(PFAOptionConfig(**blob["r0"]["config"]), PFAOptionParams(**blob["r0"]["params"]))
    m1 = PFAOption(PFAOptionConfig(**blob["r1"]["config"]), PFAOptionParams(**blob["r1"]["params"]))
    df = pd.read_parquet(EEDI_DATA)
    df["UserId"] = df["UserId"].astype(str)
    df["split"] = df["UserId"].map(assign_eedi_split)
    df = df.sort_values(["UserId", "DateAnswered_parsed", "AnswerId"], kind="mergesort")
    conf = df[df["split"] == "confirmatory"]
    log(f"Eedi confirmatory rows={len(conf)} users={conf['UserId'].nunique()}")

    y_list: list[int] = []
    p0_list: list[float] = []
    p1_list: list[float] = []
    for uid, g in conf.groupby("UserId", sort=False):
        succ: dict[Any, float] = defaultdict(float)
        fail: dict[Any, float] = defaultdict(float)
        opt: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        qs = g["QuestionId"].to_numpy()
        ys = g["IsCorrect"].to_numpy()
        avs = g["AnswerValue"].to_numpy()
        subjs = g["SubjectId"].to_numpy()
        for i in range(len(g)):
            subj = subjs[i]
            y = int(ys[i])
            av = int(avs[i])
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
            p0_list.append(m0.predict_proba_from_feats(feats0))
            p1_list.append(m1.predict_proba_from_feats(feats1))
            y_list.append(y)
            if y == 1:
                succ[subj] += 1.0
            else:
                fail[subj] += 1.0
            if 1 <= av <= 4:
                o[av - 1] += 1.0

    y = np.asarray(y_list, dtype=float)
    out = {
        "R0_omit_option": binary_metrics(y, np.asarray(p0_list)),
        "R1_option_aware": binary_metrics(y, np.asarray(p1_list)),
        "checkpoint": str(PFA_CKPT.relative_to(ROOT)),
        "runtime_s": time.time() - t0,
        "n_users": int(conf["UserId"].nunique()),
    }
    out["delta_R1_minus_R0"] = {
        k: out["R1_option_aware"][k] - out["R0_omit_option"][k]
        for k in ("roc_auc", "log_loss", "brier", "ece_10bin")
    }
    return out


def load_frozen_eedi_certs() -> dict[str, Any]:
    p1_path = ROOT / "results" / "wdlf_r3" / "p1_results.csv"
    p2_path = ROOT / "results" / "wdlf_r3" / "p2_results.csv"
    p1 = None
    with p1_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == "PFA-Option" and float(r["threshold"]) == 0.8:
                p1 = {
                    "supported_rate": float(r["supported_rate"]),
                    "unsupported_rate": float(r["unsupported_rate"]),
                    "set_valued_given_supported": float(r["set_valued_given_supported"]),
                    "stable": int(float(r["stable"])),
                    "set_valued": int(float(r["set_valued"])),
                    "unsupported": int(float(r["unsupported"])),
                    "n_events": int(float(r["n_events"])),
                    "event_noncertification_proxy": (
                        float(r["unsupported"]) + float(r["set_valued"])
                    )
                    / float(r["n_events"]),
                }
    p2 = None
    with p2_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == "PFA-Option" and float(r["kappa"]) == 0.05:
                n_eligible = float(r["n_eligible"])
                # R3 CSV does not store full-user denominator; use eligibility coverage field.
                p2 = {
                    "n_eligible": int(n_eligible),
                    "ambiguous": int(float(r["ambiguous"])),
                    "ambiguous_rate": float(r["ambiguous_rate"]),
                    "certificate_coverage": float(r["certificate_coverage"]),
                    "guaranteed_in": int(float(r["guaranteed_in"])),
                    "guaranteed_out": int(float(r["guaranteed_out"])),
                    "note": "P2 ambiguous_rate on eligible users; not ASSISTments-style nc_all",
                }
    return {
        "source_p1": str(p1_path.relative_to(ROOT)),
        "source_p2": str(p2_path.relative_to(ROOT)),
        "PFA-Option@0.8": p1,
        "PFA-Option@kappa0.05": p2,
        "interpretation": (
            "Eedi frozen certificates measure option-ambiguity under R0 keys; "
            "they are not an R0-vs-R1 predictive-input certificate ladder."
        ),
    }


def write_report(payload: dict[str, Any]) -> None:
    a = payload["assistments"]
    e = payload["eedi"]
    ap = a["predictive"]
    ac = a["certificates_frozen"]
    ep = e["predictive"]
    lines = [
        "# WDLF R1 — Predictive Ablation Comparator",
        "",
        f"- Protocol: `{PROTOCOL.relative_to(ROOT).as_posix()}` (frozen before run)",
        f"- Generated: {payload['generated_utc']}",
        f"- Runtime: {payload['total_runtime_s']:.1f}s",
        "",
        "## Key question",
        "",
        "Can predictive metrics move little while certification Event NC / P2 NC move a lot (esp. S0→S1)?",
        "",
        "## ASSISTments — TransparentActionPFA (S0 vs S1 observable inputs)",
        "",
        "Next-event correctness on confirmatory users; online mastery with schema-restricted updates.",
        "",
        "| Variant | n | AUC | NLL | Brier | ECE(10) |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| S0 observables | {ap['S0']['n']} | {ap['S0']['roc_auc']:.4f} | "
            f"{ap['S0']['log_loss']:.4f} | {ap['S0']['brier']:.4f} | {ap['S0']['ece_10bin']:.4f} |"
        ),
        (
            f"| S1 observables | {ap['S1']['n']} | {ap['S1']['roc_auc']:.4f} | "
            f"{ap['S1']['log_loss']:.4f} | {ap['S1']['brier']:.4f} | {ap['S1']['ece_10bin']:.4f} |"
        ),
        (
            f"| Δ (S1−S0) | — | {ap['delta_S1_minus_S0']['roc_auc']:+.4f} | "
            f"{ap['delta_S1_minus_S0']['log_loss']:+.4f} | {ap['delta_S1_minus_S0']['brier']:+.4f} | "
            f"{ap['delta_S1_minus_S0']['ece_10bin']:+.4f} |"
        ),
        "",
        "### Frozen R7 certificates (TransparentActionPFA; not re-tuned)",
        "",
        "| Schema | Support | Event NC | P2 NC (all-user) |",
        "|---|---:|---:|---:|",
    ]
    for s in ("S0", "S1", "S2", "S3"):
        c = ac["by_schema"][s]
        lines.append(
            f"| {s} | {c['support_rate']:.4f} | {c['event_noncertification_rate']:.4f} | "
            f"{c['p2_all_user_noncertification_rate']:.4f} |"
        )
    d = ac["delta_S1_minus_S0"]
    lines += [
        "",
        f"- Δ Event NC (S1−S0): **{d['event_noncertification_rate']:+.4f}**",
        f"- Δ P2 NC (S1−S0): **{d['p2_all_user_noncertification_rate']:+.4f}**",
        "",
        "### Reading",
        "",
        payload["headline"]["assistments_reading"],
        "",
        "## Eedi — PFA-Option (R0 omit-option vs R1 option-aware)",
        "",
        "Frozen g6p checkpoints; held-out confirmatory next-step scoring.",
        "",
        "| Variant | n | AUC | NLL | Brier | ECE(10) |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| R0 omit option | {ep['R0_omit_option']['n']} | {ep['R0_omit_option']['roc_auc']:.4f} | "
            f"{ep['R0_omit_option']['log_loss']:.4f} | {ep['R0_omit_option']['brier']:.4f} | "
            f"{ep['R0_omit_option']['ece_10bin']:.4f} |"
        ),
        (
            f"| R1 option-aware | {ep['R1_option_aware']['n']} | {ep['R1_option_aware']['roc_auc']:.4f} | "
            f"{ep['R1_option_aware']['log_loss']:.4f} | {ep['R1_option_aware']['brier']:.4f} | "
            f"{ep['R1_option_aware']['ece_10bin']:.4f} |"
        ),
        (
            f"| Δ (R1−R0) | — | {ep['delta_R1_minus_R0']['roc_auc']:+.4f} | "
            f"{ep['delta_R1_minus_R0']['log_loss']:+.4f} | {ep['delta_R1_minus_R0']['brier']:+.4f} | "
            f"{ep['delta_R1_minus_R0']['ece_10bin']:+.4f} |"
        ),
        "",
        "### Frozen R3 certificates (context; different estimand)",
        "",
        f"- Event NC proxy (unsupported+set-valued)/n: "
        f"**{e['certificates_frozen']['PFA-Option@0.8']['event_noncertification_proxy']:.4f}**",
        f"- P2 ambiguous_rate @κ=0.05 (eligible): "
        f"**{e['certificates_frozen']['PFA-Option@kappa0.05']['ambiguous_rate']:.4f}**",
        "",
        e["certificates_frozen"]["interpretation"],
        "",
        "## Limitations",
        "",
        "- Neural ActionTokenGRU / CategoricalDKT predictive ablation deferred (compute; TransparentActionPFA + PFA-Option preferred).",
        "- ASSISTments predictive variants restrict *online update fields*; they do not retrain a global likelihood.",
        "- Eedi certificate numbers are option-ambiguity certificates, not an S0/S1 schema ladder.",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    assert PROTOCOL.exists(), "Protocol must be frozen before run"
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    t0 = time.time()
    assist_pred = assistments_predictive()
    assist_cert = load_frozen_assistments_certs()
    eedi_pred = eedi_predictive()
    eedi_cert = load_frozen_eedi_certs()

    # Headline judgment (pre-specified qualitative rule; not tuned)
    d_auc = abs(assist_pred["delta_S1_minus_S0"]["roc_auc"])
    d_nll = abs(assist_pred["delta_S1_minus_S0"]["log_loss"])
    d_enc = abs(assist_cert["delta_S1_minus_S0"]["event_noncertification_rate"])
    d_p2 = abs(assist_cert["delta_S1_minus_S0"]["p2_all_user_noncertification_rate"])
    pred_small = d_auc < 0.02 and d_nll < 0.02
    cert_large = d_enc >= 0.03 or d_p2 >= 0.03
    reading = (
        f"Predictive |ΔAUC|={d_auc:.4f}, |ΔNLL|={d_nll:.4f}; "
        f"certificate |ΔEventNC|={d_enc:.4f}, |ΔP2NC|={d_p2:.4f}. "
        + (
            "YES — predictive move is small relative to certification move on S0→S1."
            if pred_small and cert_large
            else (
                "MIXED — see magnitudes; predictive and certification shifts are not cleanly separated."
                if not (pred_small and cert_large)
                else "See magnitudes."
            )
        )
    )

    payload = {
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "generated_utc": utc_now(),
        "total_runtime_s": time.time() - t0,
        "assistments": {
            "model": "TransparentActionPFA",
            "predictive": assist_pred,
            "certificates_frozen": assist_cert,
        },
        "eedi": {
            "model": "PFA-Option",
            "predictive": eedi_pred,
            "certificates_frozen": eedi_cert,
            "neural_deferred": True,
        },
        "headline": {
            "assistments_pred_small": pred_small,
            "assistments_cert_large": cert_large,
            "assistments_dissociation": bool(pred_small and cert_large),
            "assistments_reading": reading,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "predictive_ablation.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload)
    log(f"wrote {out_path}")
    log(f"wrote {REPORT}")
    log(reading)


if __name__ == "__main__":
    main()
