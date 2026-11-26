#!/usr/bin/env python3
"""Measure matched-event interval nesting gaps for the chronology-valid GRU.

No retraining and no rescoring of the full population.
"""
from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "chronorep", ROOT / "scripts/assistments/chronological_replication.py"
)
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)


def main() -> None:
    geometry = chronorep.build_development(set())
    grouped = chronorep.load_confirmatory()
    checkpoint = chronorep.plan_checkpoints(grouped, geometry["bundles"])
    pilot = chronorep.make_pilot(geometry["skill_order"], geometry["atype_order"])
    state = torch.load(chronorep.OUT / "gru_chronological_seed17.pt", map_location=pilot.device, weights_only=False)
    pilot.model.load_state_dict(state["state_dict"])
    pilot.model.eval()
    gaps = []
    near_threshold = 0
    checked = 0
    for uid, rows in grouped.items():
        matched = checkpoint["plans"][uid]["matched"]
        if matched is None:
            continue
        by_start, by_end = chronorep.order_valid(rows)
        row = next(item for item in by_start if item["source_row"] == matched)
        prefix = chronorep.prefix_events(by_end, row["start"])
        ev = chronorep.as_event(row)
        keys = {schema: chronorep.release_key(ev, schema) for schema in chronorep.SCHEMAS}
        slots = {schema: geometry["bundles"][schema][keys[schema]] for schema in chronorep.SCHEMAS}
        cut = chronorep.history_cut(by_end, row["start"])
        base = np.zeros(7, dtype=np.float64)
        for item in by_end[:cut]:
            if item["skill_id"] == row["skill_id"]:
                base += np.asarray(chronorep.fp_delta(chronorep.pfa_fp(item)), dtype=np.float64)
        pfa = {}
        gru_unique = {}
        pairs = []
        owners = []
        for schema in chronorep.SCHEMAS:
            values = [chronorep.mastery_of(base + delta) for delta in slots[schema].deltas]
            pfa[schema] = (min(values), max(values))
            seen = {}
            for fp in slots[schema].gru:
                seq = chronorep.sequence_for(pilot, prefix, fp, row["skill_id"])
                key = (tuple(seq[0]), tuple(seq[1]))
                if key not in seen:
                    seen[key] = len(pairs)
                    pairs.append(seq)
                    owners.append((schema, key))
            gru_unique[schema] = seen
        scores = chronorep.score_forward(pilot.model, pairs, pilot.device)
        for name, table in (("TransparentActionPFA", pfa),):
            for finer, coarser in (("S3", "S2"), ("S2", "S1"), ("S1", "S0")):
                lo_f, hi_f = 1.0 - table[finer][1], 1.0 - table[finer][0]
                lo_c, hi_c = 1.0 - table[coarser][1], 1.0 - table[coarser][0]
                gap = max(0.0, lo_c - lo_f, hi_f - hi_c)
                if gap > 0:
                    gaps.append((name, finer, coarser, gap))
        schema_scores = {}
        for schema in chronorep.SCHEMAS:
            vals = [float(scores[index]) for index in gru_unique[schema].values()]
            schema_scores[schema] = (min(vals), max(vals))
            for endpoint in schema_scores[schema]:
                if min(abs(endpoint - tau) for tau in chronorep.TAUS) <= 1e-5:
                    near_threshold += 1
        for finer, coarser in (("S3", "S2"), ("S2", "S1"), ("S1", "S0")):
            lo_f, hi_f = 1.0 - schema_scores[finer][1], 1.0 - schema_scores[finer][0]
            lo_c, hi_c = 1.0 - schema_scores[coarser][1], 1.0 - schema_scores[coarser][0]
            gap = max(0.0, lo_c - lo_f, hi_f - hi_c)
            if gap > 0:
                gaps.append(("ActionTokenGRU_unique_sequence", finer, coarser, gap))
        checked += 1
        if checked % 1000 == 0:
            print(f"nesting_audit {checked} gaps {len(gaps)}", flush=True)
    summary = {
        "matched_events_checked": checked,
        "positive_gap_count": len(gaps),
        "max_gap": max((item[3] for item in gaps), default=0.0),
        "gaps_above_1e-5": sum(item[3] > 1e-5 for item in gaps),
        "gaps_above_1e-6": sum(item[3] > 1e-6 for item in gaps),
        "by_model": dict(defaultdict(int, {name: sum(item[0] == name for item in gaps) for name in {item[0] for item in gaps}})),
        "endpoints_within_1e-5_of_a_threshold": near_threshold,
        "interpretation": "Positive gaps are measured after scoring each distinct token sequence once. A zero PFA count means the closed-form intervals nest exactly.",
    }
    chronorep.dump("interval_nesting.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
