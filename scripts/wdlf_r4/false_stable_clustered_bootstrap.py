#!/usr/bin/env python3
"""Learner-clustered bootstrap CI for false-Stable rates (no new science)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "wdlf_r1_rescue" / "assist_heldout_user_aggregates.json"
OUT = ROOT / "results" / "wdlf_r4" / "false_stable_clustered_bootstrap.json"


def boot(model_users: dict, schema: str, rng: np.random.Generator, B: int = 2000) -> dict:
    users = list(model_users.keys())
    fs = np.array([model_users[u][schema]["n_false_stable"] for u in users], dtype=float)
    st = np.array([model_users[u][schema]["n_stable"] for u in users], dtype=float)
    ev = np.array([model_users[u][schema]["n_events"] for u in users], dtype=float)
    n = len(users)
    point_fs_st = float(fs.sum() / st.sum()) if st.sum() else float("nan")
    point_fs_all = float(fs.sum() / ev.sum()) if ev.sum() else float("nan")
    rates_st, rates_all = [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        den_st = st[idx].sum()
        den_ev = ev[idx].sum()
        if den_st > 0:
            rates_st.append(fs[idx].sum() / den_st)
        if den_ev > 0:
            rates_all.append(fs[idx].sum() / den_ev)
    rates_st = np.asarray(rates_st)
    rates_all = np.asarray(rates_all)
    return {
        "n_users": n,
        "n_false_stable": int(fs.sum()),
        "n_stable": int(st.sum()),
        "n_events": int(ev.sum()),
        "rate_among_stable": point_fs_st,
        "rate_among_all": point_fs_all,
        "ci95_among_stable": [
            float(np.quantile(rates_st, 0.025)),
            float(np.quantile(rates_st, 0.975)),
        ],
        "ci95_among_all": [
            float(np.quantile(rates_all, 0.025)),
            float(np.quantile(rates_all, 0.975)),
        ],
    }


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20270808)
    B = 2000
    rates = {}
    for model in ("ActionTokenGRU", "TransparentActionPFA"):
        rates[model] = {}
        for schema in ("S0", "S1", "S2", "S3"):
            rates[model][schema] = boot(data[model], schema, rng, B=B)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": 20270808,
        "B": B,
        "method": "learner_cluster_resample_with_replacement",
        "source": str(SRC.relative_to(ROOT)).replace("\\", "/"),
        "rates": rates,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for model, schemas in rates.items():
        for schema, r in schemas.items():
            if r["n_false_stable"] > 0:
                print(
                    model,
                    schema,
                    f"FS={r['n_false_stable']}/{r['n_stable']}={r['rate_among_stable']:.6%}",
                    "CI_stable",
                    r["ci95_among_stable"],
                    "CI_all",
                    r["ci95_among_all"],
                )


if __name__ == "__main__":
    main()
