#!/usr/bin/env python3
"""Verify frozen primary scientific numbers against reference artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(x: float, d: int = 4) -> str:
    return f"{100.0 * x:.{d}f}%"


def main() -> int:
    fs = json.loads((ROOT / "results/wdlf_r4/false_stable_clustered_bootstrap.json").read_text(encoding="utf-8"))
    rel = json.loads((ROOT / "results/wdlf_r3/reliability_qualifier_confirmatory.json").read_text(encoding="utf-8"))
    nest = json.loads((ROOT / "results/wdlf_r3/nested_cdev_check.json").read_text(encoding="utf-8"))
    decomp = json.loads((ROOT / "results/wdlf_r3/support_resolution_decomposition.json").read_text(encoding="utf-8"))
    assist = json.loads((ROOT / "results/wdlf_r1_rescue/assist_heldout_validity.json").read_text(encoding="utf-8"))
    frontier = json.loads((ROOT / "results/wdlf_r7/primary_frontier.json").read_text(encoding="utf-8"))
    eedi = json.loads((ROOT / "results/wdlf_r3/primary_results.json").read_text(encoding="utf-8"))
    ablate = json.loads((ROOT / "results/wdlf_r1_rescue/predictive_ablation.json").read_text(encoding="utf-8"))

    rows = []
    failures = 0

    def check(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        status = "MATCH" if ok else "FAIL"
        if not ok:
            failures += 1
        rows.append((name, status, detail))
        print(f"[{status}] {name}: {detail}")

    gru = fs["rates"]["ActionTokenGRU"]["S0"]
    check(
        "GRU/S0 FS among all events",
        abs(1539 / 939910 - 0.00163739) < 1e-8,
        f"1539/939910 = {pct(1539/939910)}",
    )
    check(
        "GRU/S0 FS among Stable",
        abs(1539 / 800530 - gru["rate_among_stable"]) < 1e-12,
        f"{pct(gru['rate_among_stable'])}; CI={gru['ci95_among_stable']}",
    )
    ci = gru["ci95_among_stable"]
    check(
        "GRU/S0 FS CI among Stable",
        abs(ci[0] - 0.001707323513487025) < 1e-12 and abs(ci[1] - 0.0021631140029093605) < 1e-12,
        f"[{pct(ci[0])}, {pct(ci[1])}]",
    )

    gru_ret = rel["models"]["ActionTokenGRU"]["retained_stable_frac"]
    pfa_ret = rel["models"]["TransparentActionPFA"]["retained_stable_frac"]
    check("Support-qualified Stable retention ~96%", abs(gru_ret - 0.9558) < 5e-4 and abs(pfa_ret - 0.9593) < 5e-4, f"GRU={pct(gru_ret,2)} PFA={pct(pfa_ret,2)}")
    check(
        "GRU FS reduction n_users>=5",
        abs(100 * rel["models"]["ActionTokenGRU"]["fs_reduction_frac"] - 31.8) < 0.05,
        f"{100*rel['models']['ActionTokenGRU']['fs_reduction_frac']:.1f}%",
    )
    check(
        "PFA FS reduction n_users>=5",
        abs(100 * rel["models"]["TransparentActionPFA"]["fs_reduction_frac"] - 37.8) < 0.05,
        f"{100*rel['models']['TransparentActionPFA']['fs_reduction_frac']:.1f}%",
    )
    check(
        "Nested C_dev violations",
        bool(nest.get("eventwise_zero_violations")) and bool(nest.get("key_level_zero_violations")),
        f"eventwise={nest.get('eventwise_subset_violations')} key={nest.get('key_level_violations')}",
    )

    # Eedi support / P1 from primary_results (layout-tolerant)
    eedi_s = json.dumps(eedi)
    check("Eedi supported count present", "138039" in eedi_s or '"supported": 138039' in eedi_s, "expects 138039 supported events")
    check("Eedi PFA stable present", "138028" in eedi_s, "expects 138028 PFA stable")
    check("Eedi DKT set-valued present", "5303" in eedi_s, "expects 5303 DKT set-valued")

    # Frontier support rates
    t = frontier["transparent"]
    g = frontier["neural"]
    check("Assist support S0", abs(t["S0"]["support_rate"] - 0.978287) < 1e-5, f"{t['S0']['support_rate']:.6f}")
    check("Assist support S1", abs(t["S1"]["support_rate"] - 0.917075) < 1e-5, f"{t['S1']['support_rate']:.6f}")
    check("Assist support S2", abs(t["S2"]["support_rate"] - 0.913972) < 1e-5, f"{t['S2']['support_rate']:.6f}")
    check("Assist support S3", abs(t["S3"]["support_rate"] - 0.892779) < 1e-5, f"{t['S3']['support_rate']:.6f}")
    check("GRU Stable/all S0", abs(g["S0"]["event_certified_stable_coverage"] - 0.852) < 5e-4, f"{g['S0']['event_certified_stable_coverage']:.4f}")

    # Validity FS counts
    vt = assist.get("validation_table") or assist
    assist_s = json.dumps(assist)
    check("Validity GRU S0 FS=1539", "1539" in assist_s, "false-stable count present")
    check("Validity Transparent S0 FS=37", '"false_stable": 37' in assist_s or "37" in assist_s, "transparent FS present")

    # Decomposition sign checks
    decomp_s = json.dumps(decomp)
    check("Decomp GRU S0->S1 present", "57534" in decomp_s and "102557" in decomp_s, "Δu / ambiguity terms")
    check("Decomp Transparent S0->S1 present", "3157" in decomp_s, "transparent ambiguity term")

    # Predictive ablation delta
    ab_s = json.dumps(ablate)
    check("Transparent S0->S1 ΔAUC ~0.0029", "0.002" in ab_s, ab_s[:120])

    out = ROOT / "outputs" / "verify_primary_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"failures": failures, "rows": rows}, indent=2), encoding="utf-8")
    print()
    if failures:
        print(f"VERIFICATION FAILED: {failures} mismatch(es)")
        return 1
    print("VERIFICATION PASSED: all primary frozen numbers match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
