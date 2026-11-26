#!/usr/bin/env python3
"""User-level bootstrap CIs for Authentic WDLF held-out validity / primary rates.

Resamples users with replacement; recomputes rates from saved per-user aggregates.
Target: 2000 reps if feasible, else >=500.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Absolute project root (script may be copied to Temp under an allowlisted name).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "wdlf_r1_rescue"
SEED = 20260808


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def ci_from_samples(samples: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    return {
        "n": int(len(samples)),
        "mean": float(samples.mean()),
        "ci_low": float(np.quantile(samples, alpha / 2)),
        "ci_high": float(np.quantile(samples, 1 - alpha / 2)),
        "std": float(samples.std(ddof=1)) if len(samples) > 1 else 0.0,
    }


def eedi_p1_boot(user_aggs: dict[str, dict], n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    uids = list(user_aggs.keys())
    n = len(uids)
    keys = [
        "unsupported_rate",
        "supported_rate",
        "stable_rate",
        "set_valued_rate",
        "actual_answer_in_cdev_rate_supported",
        "action_containment_rate_supported",
        "false_stable_rate_among_stable",
    ]
    bags = {k: [] for k in keys}
    for _ in range(n_boot):
        draw = rng.choice(n, size=n, replace=True)
        ne = ns = nu = nst = nsv = nin = ncont = nfs = 0
        for i in draw:
            u = user_aggs[uids[i]]
            ne += u["n_events"]
            ns += u["n_supported"]
            nu += u["n_unsupported"]
            nst += u["n_stable"]
            nsv += u["n_set_valued"]
            nin += u["n_actual_in_cdev"]
            ncont += u["n_action_contained"]
            nfs += u["n_false_stable"]
        bags["unsupported_rate"].append(nu / ne if ne else np.nan)
        bags["supported_rate"].append(ns / ne if ne else np.nan)
        bags["stable_rate"].append(nst / ne if ne else np.nan)
        bags["set_valued_rate"].append(nsv / ne if ne else np.nan)
        bags["actual_answer_in_cdev_rate_supported"].append(nin / ns if ns else np.nan)
        bags["action_containment_rate_supported"].append(ncont / ns if ns else np.nan)
        bags["false_stable_rate_among_stable"].append(nfs / nst if nst else np.nan)
    return {k: ci_from_samples(np.array(v)) for k, v in bags.items()}


def eedi_p2_boot(user_aggs: dict[str, dict], n_boot: int, rng: np.random.Generator, kappa: float) -> dict[str, Any]:
    from src.identification.compatible_trace.capacity_identified_set import certify_capacity_membership

    uids = list(user_aggs.keys())
    n = len(uids)
    unresolved_elig = []
    all_user_nc = []
    for _ in range(n_boot):
        draw = rng.choice(n, size=n, replace=True)
        elig_ids = []
        mins = []
        maxs = []
        for j, i in enumerate(draw):
            u = user_aggs[uids[i]]
            if u.get("p2_risk_min") is None:
                continue
            elig_ids.append(f"{uids[i]}#{j}")
            mins.append(float(u["p2_risk_min"]))
            maxs.append(float(u["p2_risk_max"]))
        n_frame = n
        n_elig = len(elig_ids)
        if n_elig == 0:
            unresolved_elig.append(np.nan)
            all_user_nc.append(1.0)
            continue
        mem = certify_capacity_membership(elig_ids, mins, maxs, capacity=kappa)
        amb = sum(1 for m in mem if m.status == "ambiguous")
        unresolved_elig.append(amb / n_elig)
        no_ckpt = n_frame - n_elig
        all_user_nc.append((amb + no_ckpt) / n_frame)
    return {
        "unresolved_rate_eligible": ci_from_samples(np.array(unresolved_elig)),
        "p2_all_user_noncertification_rate": ci_from_samples(np.array(all_user_nc)),
    }


def assist_delta_boot(
    user_aggs: dict[str, dict],
    n_boot: int,
    rng: np.random.Generator,
    s0: str = "S0",
    s1: str = "S1",
) -> dict[str, Any]:
    uids = [u for u, v in user_aggs.items() if s0 in v and s1 in v]
    n = len(uids)
    if n == 0:
        return {"n_users": 0}
    d_support = []
    d_stable_cov = []
    d_false_stable = []
    for _ in range(n_boot):
        draw = rng.choice(n, size=n, replace=True)
        sums = {
            s0: {"ne": 0, "ns": 0, "nst": 0, "nfs": 0},
            s1: {"ne": 0, "ns": 0, "nst": 0, "nfs": 0},
        }
        for i in draw:
            for s in (s0, s1):
                u = user_aggs[uids[i]][s]
                sums[s]["ne"] += u["n_events"]
                sums[s]["ns"] += u["n_supported"]
                sums[s]["nst"] += u["n_stable"]
                sums[s]["nfs"] += u["n_false_stable"]
        sr0 = sums[s0]["ns"] / sums[s0]["ne"] if sums[s0]["ne"] else np.nan
        sr1 = sums[s1]["ns"] / sums[s1]["ne"] if sums[s1]["ne"] else np.nan
        sc0 = sums[s0]["nst"] / sums[s0]["ne"] if sums[s0]["ne"] else np.nan
        sc1 = sums[s1]["nst"] / sums[s1]["ne"] if sums[s1]["ne"] else np.nan
        fs0 = sums[s0]["nfs"] / sums[s0]["nst"] if sums[s0]["nst"] else np.nan
        fs1 = sums[s1]["nfs"] / sums[s1]["nst"] if sums[s1]["nst"] else np.nan
        d_support.append(sr1 - sr0)
        d_stable_cov.append(sc1 - sc0)
        d_false_stable.append(fs1 - fs0)
    return {
        "n_users": n,
        "delta_support_rate_S1_minus_S0": ci_from_samples(np.array(d_support)),
        "delta_event_certified_stable_coverage_S1_minus_S0": ci_from_samples(np.array(d_stable_cov)),
        "delta_false_stable_rate_among_stable_S1_minus_S0": ci_from_samples(np.array(d_false_stable)),
    }


def choose_n_boot(n_users: int, requested: int = 2000) -> int:
    if n_users <= 0:
        return 0
    return requested


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eedi_path = OUT / "eedi_heldout_user_aggregates.json"
    assist_path = OUT / "assist_heldout_user_aggregates.json"
    if not eedi_path.exists():
        raise SystemExit(f"missing {eedi_path}; run eedi_heldout_validity.py first")

    eedi = json.loads(eedi_path.read_text(encoding="utf-8"))
    kappa = float(eedi.get("kappa", 0.05))
    n_eedi_users = len(next(iter(eedi["models"].values())))
    n_boot = choose_n_boot(n_eedi_users, 2000)
    rng = np.random.default_rng(SEED)

    eedi_out: dict[str, Any] = {}
    for model, aggs in eedi["models"].items():
        print(f"[bootstrap] Eedi {model} n_boot={n_boot} users={len(aggs)}", flush=True)
        eedi_out[model] = {
            "p1": eedi_p1_boot(aggs, n_boot, rng),
            "p2": eedi_p2_boot(aggs, n_boot, rng, kappa=kappa),
        }

    assist_out: dict[str, Any] = {}
    if assist_path.exists():
        assist = json.loads(assist_path.read_text(encoding="utf-8"))
        for model_key in ("TransparentActionPFA", "ActionTokenGRU"):
            aggs = assist.get(model_key) or {}
            if not aggs:
                assist_out[model_key] = {"deferred_or_empty": True}
                continue
            sample_u = next(iter(aggs.values()))
            if "S0" not in sample_u or "S1" not in sample_u:
                assist_out[model_key] = {
                    "deferred_or_empty": True,
                    "schemas_present": list(sample_u.keys()),
                }
                continue
            print(f"[bootstrap] ASSIST {model_key} n_boot={n_boot} users={len(aggs)}", flush=True)
            assist_out[model_key] = assist_delta_boot(aggs, n_boot, rng)
    else:
        assist_out = {"missing_aggregates": str(assist_path)}

    payload = {
        "created_utc": utc_now(),
        "seed": SEED,
        "n_bootstrap": n_boot,
        "unit": "user",
        "alpha": 0.05,
        "ci": "percentile",
        "eedi": eedi_out,
        "assistments_S0_to_S1_deltas": assist_out,
        "runtime_s": time.time() - t0,
    }
    path = OUT / "user_bootstrap_ci.json"
    write_json(path, payload)
    print(f"[bootstrap] wrote {path} runtime_s={payload['runtime_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
