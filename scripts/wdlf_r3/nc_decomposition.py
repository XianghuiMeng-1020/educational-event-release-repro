#!/usr/bin/env python3
"""Support-resolution NC decomposition from frozen Round-2 validity artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "results" / "wdlf_r1_rescue" / "assist_heldout_validity.json"
OUT = ROOT / "results" / "wdlf_r3"


def row(m: dict, schema: str) -> dict:
    r = m[schema]
    n = int(r["n_events"])
    unsup = int(r["unsupported"])
    setv = int(r["set_valued"])
    stable = int(r["stable"])
    supported = int(r["supported"])
    assert unsup + setv + stable == n
    assert supported == setv + stable
    u = unsup / n
    a = setv / n  # Ambiguous ∩ Supported / all
    amb_given_sup = setv / supported if supported else None
    nc = (unsup + setv) / n
    return {
        "n_events": n,
        "unsupported": unsup,
        "set_valued": setv,
        "stable": stable,
        "supported": supported,
        "u_unsupported_rate": u,
        "a_ambiguous_and_supported_rate": a,
        "ambiguous_given_supported": amb_given_sup,
        "nc_rate": nc,
        "nc_equals_u_plus_a": abs(nc - (u + a)) < 1e-15,
        "support_rate": supported / n,
    }


def delta(coarse: dict, fine: dict) -> dict:
    du = fine["u_unsupported_rate"] - coarse["u_unsupported_rate"]
    da = fine["a_ambiguous_and_supported_rate"] - coarse["a_ambiguous_and_supported_rate"]
    dnc = fine["nc_rate"] - coarse["nc_rate"]
    # Improvement iff u_f - u_c < a_c - a_f  i.e. du < -da
    amb_removed = coarse["a_ambiguous_and_supported_rate"] - fine["a_ambiguous_and_supported_rate"]
    return {
        "delta_u_added_unsupported": du,
        "delta_a": da,
        "ambiguity_removed": amb_removed,
        "delta_nc": dnc,
        "delta_nc_equals_du_plus_da": abs(dnc - (du + da)) < 1e-15,
        "refinement_improves_nc": du < amb_removed,  # du < -da
        "count_delta_unsupported": fine["unsupported"] - coarse["unsupported"],
        "count_delta_set_valued": fine["set_valued"] - coarse["set_valued"],
        "count_delta_nc_mass": (fine["unsupported"] + fine["set_valued"])
        - (coarse["unsupported"] + coarse["set_valued"]),
    }


def main() -> None:
    blob = json.loads(VAL.read_text(encoding="utf-8"))
    out: dict = {"source": str(VAL), "models": {}, "transitions": [], "corollary_checks": []}
    schemas = ["S0", "S1", "S2", "S3"]
    for model_name in ("TransparentActionPFA", "ActionTokenGRU"):
        m = blob[model_name]
        out["models"][model_name] = {s: row(m, s) for s in schemas}
        for c, f in zip(schemas, schemas[1:]):
            dlt = delta(out["models"][model_name][c], out["models"][model_name][f])
            rec = {"model": model_name, "transition": f"{c}->{f}", **dlt}
            # exact count identity
            c_row = out["models"][model_name][c]
            f_row = out["models"][model_name][f]
            assert rec["count_delta_nc_mass"] == rec["count_delta_unsupported"] + rec[
                "count_delta_set_valued"
            ]
            out["transitions"].append(rec)
            out["corollary_checks"].append(
                {
                    "model": model_name,
                    "transition": f"{c}->{f}",
                    "added_unsupported_mass": rec["count_delta_unsupported"],
                    "ambiguity_removed_count": -rec["count_delta_set_valued"],
                    "net_nc_count_change": rec["count_delta_nc_mass"],
                    "identity_ok": True,
                    "refinement_improves_unconditional_nc": rec["refinement_improves_nc"],
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "support_resolution_decomposition.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = OUT / "support_resolution_transitions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "transition",
                "added_unsupported_mass",
                "ambiguity_removed_count",
                "net_nc_count_change",
                "refinement_improves_unconditional_nc",
            ],
        )
        w.writeheader()
        for r in out["corollary_checks"]:
            w.writerow(
                {
                    "model": r["model"],
                    "transition": r["transition"],
                    "added_unsupported_mass": r["added_unsupported_mass"],
                    "ambiguity_removed_count": r["ambiguity_removed_count"],
                    "net_nc_count_change": r["net_nc_count_change"],
                    "refinement_improves_unconditional_nc": r[
                        "refinement_improves_unconditional_nc"
                    ],
                }
            )
    print(json.dumps({"wrote": str(path), "csv": str(csv_path), "checks": out["corollary_checks"]}, indent=2))


if __name__ == "__main__":
    main()
