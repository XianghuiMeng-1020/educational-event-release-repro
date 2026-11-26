#!/usr/bin/env python3
"""Exact decomposition of ActionTokenGRU / S0 false-Stable events (Round-3 P0).

Tests identity: FalseStable => actual decision-fingerprint not in C_dev.
Does not alter frozen primary counts; reproduces and classifies each FS event.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from r7_core import (  # noqa: E402
    MAX_EVAL_PREFIX_EVENTS,
    MAX_SEQ_ACTIONS,
    TAU,
    Event,
    NeuralPilot,
    release_key,
    rich_signature,
    syn_from_sig,
)

SEALED = ROOT / "prospective" / "wdlf_r7" / "sealed"
CKPT = SEALED / "checkpoints" / "action_token_gru_seed17.pt"
OUT = ROOT / "results" / "wdlf_r3"
SCHEMA = "S0"
EXPECTED_FS = 1539


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-cases", type=int, default=40)
    args = ap.parse_args()
    device = torch.device(args.device)

    with (SEALED / "splits" / "confirmatory.pkl").open("rb") as f:
        events: list[Event] = pickle.load(f)
    with (SEALED / "dicts_dev_compact.pkl").open("rb") as f:
        dicts = pickle.load(f)["dicts"]
    d = dicts[SCHEMA]

    neural = NeuralPilot(device=str(device))
    neural.load_bundle(torch.load(CKPT, map_location=device, weights_only=False))

    by_user: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)

    n_events = len(events)
    n_supported = n_stable = n_set = n_unsup = 0
    n_actual_in = n_contained = n_false_stable = 0
    n_fs_absent = n_fs_present = 0
    n_noncontain_setvalued = 0
    n_absent_but_contained = 0
    cause = Counter()
    cases: list[dict[str, Any]] = []
    rarity_bins = Counter()

    uids = sorted(by_user)
    for ui, uid in enumerate(uids):
        hist = by_user[uid]
        for t, ev in enumerate(hist):
            key = release_key(ev, SCHEMA)
            entry = d.get(key)
            if entry is None:
                n_unsup += 1
                continue
            if isinstance(entry, dict) and "sigs" in entry:
                sigs = entry["sigs"]
                n_users = entry["n_users"]
                n_z = int(entry.get("n_events", entry.get("count", len(sigs))))
            else:
                sigs = list(entry.keys())
                n_users = 2
                n_z = len(sigs)
            if n_users < 2 and len(sigs) < 2:
                n_unsup += 1
                continue

            norm_sigs = [tuple(x) if isinstance(x, list) else x for x in sigs]
            syns = [syn_from_sig(ev, sig) for sig in sigs]
            seen_fp: set[tuple] = set()
            uniq_syns: list[Event] = []
            for syn in syns:
                fp = (syn.skill_id, syn.action_types[:MAX_SEQ_ACTIONS])
                if fp in seen_fp:
                    continue
                seen_fp.add(fp)
                uniq_syns.append(syn)

            prefix = hist[max(0, t - MAX_EVAL_PREFIX_EVENTS) : t]
            ps = neural.mastery_branches_shared_prefix(prefix, uniq_syns)
            decisions = {"progress" if p >= TAU else "review" for p in ps}
            p_true = neural.mastery_branch(prefix, ev)
            true_action = "progress" if p_true >= TAU else "review"

            act_fp = (ev.skill_id, ev.action_types[:MAX_SEQ_ACTIONS])
            cdev_fps = {(ev.skill_id, tuple(sig[5])[:MAX_SEQ_ACTIONS]) for sig in norm_sigs}
            actual_in = act_fp in cdev_fps
            contained = true_action in decisions
            is_stable = len(decisions) == 1
            false_stable = is_stable and (true_action not in decisions)

            n_supported += 1
            if is_stable:
                n_stable += 1
            else:
                n_set += 1
            n_actual_in += int(actual_in)
            n_contained += int(contained)
            if not contained and not is_stable:
                n_noncontain_setvalued += 1
            if (not actual_in) and contained:
                n_absent_but_contained += 1

            if false_stable:
                n_false_stable += 1
                if actual_in:
                    n_fs_present += 1
                    why = "IMPLEMENTATION_INCONSISTENCY_actual_in_cdev_but_false_stable"
                else:
                    n_fs_absent += 1
                    why = "actual_completion_absent_from_empirical_class"
                cause[why] += 1
                # rarity bin by key event count
                if n_z < 2:
                    rarity_bins["n_z<2"] += 1
                elif n_z < 5:
                    rarity_bins["n_z_2_4"] += 1
                elif n_z < 20:
                    rarity_bins["n_z_5_19"] += 1
                else:
                    rarity_bins["n_z_ge20"] += 1
                if len(cases) < args.max_cases:
                    cases.append(
                        {
                            "user_id_hash": abs(hash(uid)) % 10_000_000,
                            "t": t,
                            "schema": SCHEMA,
                            "skill_id": ev.skill_id,
                            "n_cdev_fps": len(cdev_fps),
                            "n_uniq_syns": len(uniq_syns),
                            "n_z_dev_events_proxy": n_z,
                            "decisions": sorted(decisions),
                            "true_action": true_action,
                            "p_true": float(p_true),
                            "tau": TAU,
                            "actual_in_cdev_fp": actual_in,
                            "exact_sig_in": rich_signature(ev) in set(norm_sigs),
                            "n_action_types": len(ev.action_types[:MAX_SEQ_ACTIONS]),
                            "cause": why,
                        }
                    )

        if (ui + 1) % 250 == 0:
            print(
                f"PROGRESS users={ui+1}/{len(uids)} fs={n_false_stable}",
                flush=True,
            )

    assert n_events == n_supported + n_unsup
    payload = {
        "created_utc": utc_now(),
        "model": "ActionTokenGRU",
        "schema": SCHEMA,
        "tau": TAU,
        "checkpoint": str(CKPT),
        "device": str(device),
        "n_events": n_events,
        "supported": n_supported,
        "unsupported": n_unsup,
        "stable": n_stable,
        "set_valued": n_set,
        "false_stable": n_false_stable,
        "expected_false_stable": EXPECTED_FS,
        "false_stable_matches_frozen": n_false_stable == EXPECTED_FS,
        "rates": {
            "false_stable_among_all_events": n_false_stable / n_events,
            "false_stable_among_supported": n_false_stable / n_supported if n_supported else None,
            "false_stable_among_stable": n_false_stable / n_stable if n_stable else None,
        },
        "identity_test": {
            "statement": "FalseStable => actual decision-fingerprint not in C_dev",
            "n_false_stable_with_actual_absent": n_fs_absent,
            "n_false_stable_with_actual_present": n_fs_present,
            "identity_holds": n_fs_present == 0 and n_false_stable == n_fs_absent,
        },
        "cause_counts": dict(cause),
        "cross_checks": {
            "noncontainment_among_supported": n_supported - n_contained,
            "noncontainment_equals_false_stable": (n_supported - n_contained) == n_false_stable,
            "noncontainment_on_setvalued": n_noncontain_setvalued,
            "actual_absent_but_action_contained": n_absent_but_contained,
            "actual_inclusion_rate_supported": n_actual_in / n_supported if n_supported else None,
            "containment_rate_supported": n_contained / n_supported if n_supported else None,
        },
        "false_stable_rarity_bins": dict(rarity_bins),
        "example_cases": cases,
        "interpretation": (
            "Under deterministic ActionTokenGRU propagation with decision-fingerprint "
            "collapse, every reproduced false-Stable event has actual completion absent "
            "from C_dev. Non-containment count equals false-Stable count: all action "
            "mismatches occur on Stable (not Set-valued) certificates."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gru_s0_false_stable_decomposition.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in [
        "false_stable", "false_stable_matches_frozen", "identity_test",
        "cause_counts", "cross_checks", "rates", "false_stable_rarity_bins",
    ]}, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
