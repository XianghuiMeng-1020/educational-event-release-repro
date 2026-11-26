#!/usr/bin/env python3
"""Fast ASSIST S0→S1 user-bootstrap deltas; merge into user_bootstrap_ci.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "wdlf_r1_rescue"
SEED = 20260808
N_BOOT = 2000


def ci(samples: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(np.mean(samples)), "ci_low": float(lo), "ci_high": float(hi), "n_boot": int(len(samples))}


def deltas(aggs: dict, rng: np.random.Generator) -> dict:
    uids = [u for u, v in aggs.items() if "S0" in v and "S1" in v]
    n = len(uids)
    d_support, d_stable, d_fs = [], [], []
    for _ in range(N_BOOT):
        draw = rng.choice(n, size=n, replace=True)
        sums = {s: {"ne": 0, "ns": 0, "nst": 0, "nfs": 0} for s in ("S0", "S1")}
        for i in draw:
            for s in ("S0", "S1"):
                u = aggs[uids[i]][s]
                sums[s]["ne"] += u["n_events"]
                sums[s]["ns"] += u["n_supported"]
                sums[s]["nst"] += u["n_stable"]
                sums[s]["nfs"] += u["n_false_stable"]
        d_support.append(sums["S1"]["ns"] / sums["S1"]["ne"] - sums["S0"]["ns"] / sums["S0"]["ne"])
        d_stable.append(sums["S1"]["nst"] / sums["S1"]["ne"] - sums["S0"]["nst"] / sums["S0"]["ne"])
        fs0 = sums["S0"]["nfs"] / sums["S0"]["nst"] if sums["S0"]["nst"] else np.nan
        fs1 = sums["S1"]["nfs"] / sums["S1"]["nst"] if sums["S1"]["nst"] else np.nan
        d_fs.append(fs1 - fs0)
    return {
        "n_users": n,
        "delta_support_rate_S1_minus_S0": ci(np.array(d_support)),
        "delta_event_certified_stable_coverage_S1_minus_S0": ci(np.array(d_stable)),
        "delta_false_stable_rate_among_stable_S1_minus_S0": ci(np.array(d_fs)),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    path = OUT / "user_bootstrap_ci.json"
    blob = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    assist = json.loads((OUT / "assist_heldout_user_aggregates.json").read_text(encoding="utf-8"))
    out = {}
    for key in ("TransparentActionPFA", "ActionTokenGRU"):
        aggs = assist.get(key) or {}
        if not aggs:
            out[key] = {"deferred_or_empty": True}
            continue
        sample = next(iter(aggs.values()))
        if "S0" not in sample or "S1" not in sample:
            out[key] = {"deferred_or_empty": True, "schemas_present": list(sample.keys())}
            continue
        print(f"boot {key} users={len(aggs)}", flush=True)
        out[key] = deltas(aggs, rng)
    blob["assistments_S0_to_S1_deltas"] = out
    blob["assist_delta_boot"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "unit": "user",
    }
    path.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")
    print("WROTE", path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
