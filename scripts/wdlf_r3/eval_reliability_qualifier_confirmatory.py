#!/usr/bin/env python3
"""Apply FROZEN reliability qualifier once on confirmatory (no retuning).

Reads configs/reliability_qualifier_rule.txt rule:
  QUALIFIED_STABLE ⇔ Stable AND n_users(key) >= 5

TransparentActionPFA/S0 required; ActionTokenGRU/S0 optional (same frozen rule).
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
    TransparentActionPFA,
    release_key,
    syn_from_sig,
)

SEALED = ROOT / "prospective" / "wdlf_r7" / "sealed"
CKPT = SEALED / "checkpoints" / "action_token_gru_seed17.pt"
FREEZE = ROOT / "configs" / "reliability_qualifier_rule.txt"
OUT = ROOT / "results" / "wdlf_r3"
SCHEMA = "S0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def parse_freeze_rule(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if "n_users(key) >= 5" not in text and "n_min`:** `5`" not in text.replace("\\", ""):
        if "n_users" not in text or "`5`" not in text:
            raise ValueError(f"Frozen rule n_users >= 5 not found in {path}")
    m = re.search(r"n_\{\\min\}\):\*\*\s*`(\d+)`", text)
    if m is None:
        m = re.search(r"n_min`:\*\*\s*`(\d+)`", text)
    if m is None:
        m = re.search(r"\\min\}\):\*\*\s*`(\d+)`", text)
    n_min = int(m.group(1)) if m else 5
    # Literal freeze contract for this round:
    if "n_users(key) >= 5" in text:
        n_min = 5
    return {
        "rule_type": "n_users_ge",
        "feature": "n_users",
        "n_min": n_min,
        "freeze_path": str(path),
        "source_excerpt": "QUALIFIED_STABLE <=> Stable AND n_users(key) >= 5",
    }


def load_confirmatory() -> list[Event]:
    p = SEALED / "splits" / "confirmatory.pkl"
    with p.open("rb") as f:
        return pickle.load(f)


def load_dicts_s0() -> dict:
    with (SEALED / "dicts_dev_compact.pkl").open("rb") as f:
        blob = pickle.load(f)
    return blob["dicts"][SCHEMA]


def entry_fields(entry: Any) -> tuple[list, int]:
    if isinstance(entry, dict) and "sigs" in entry:
        return list(entry["sigs"]), int(entry["n_users"])
    sigs = list(entry.keys()) if isinstance(entry, dict) else list(entry)
    return sigs, 2


def evaluate_model(
    events: list[Event],
    dicts_s0: dict,
    model_name: str,
    n_min: int,
    neural: Optional[NeuralPilot] = None,
) -> dict[str, Any]:
    by_user: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    uids = sorted(by_user)

    n_events = n_unsup = n_supp = n_stable = n_set = n_fs = 0
    n_qual_stable = n_qual_fs = 0
    n_disqualified_stable = n_disqualified_fs = 0

    for ui, uid in enumerate(uids):
        if ui and ui % (50 if model_name == "neural" else 500) == 0:
            print(
                f"CONF_PROGRESS model={model_name} users={ui}/{len(uids)} "
                f"stable={n_stable} fs={n_fs} qual_stable={n_qual_stable}",
                flush=True,
            )
        hist = by_user[uid]
        tmodel = TransparentActionPFA()
        for t, ev in enumerate(hist):
            n_events += 1
            k = release_key(ev, SCHEMA)
            entry = dicts_s0.get(k)
            if entry is None:
                n_unsup += 1
                if model_name == "transparent":
                    tmodel.update(ev)
                continue
            sigs, n_users = entry_fields(entry)
            if n_users < 2 and len(sigs) < 2:
                n_unsup += 1
                if model_name == "transparent":
                    tmodel.update(ev)
                continue
            n_supp += 1
            syns = [syn_from_sig(ev, sig) for sig in sigs]
            if model_name == "transparent":
                seen: set[tuple] = set()
                uniq: list[Event] = []
                for syn in syns:
                    fp = (syn.correct, syn.hint_count, syn.attempt_count, syn.action_types)
                    if fp in seen:
                        continue
                    seen.add(fp)
                    uniq.append(syn)
                decisions: set[str] = set()
                for syn in uniq:
                    st = tmodel.skill_stats[ev.skill_id]
                    backup = dict(st)
                    tmodel.update(syn)
                    p = tmodel.mastery(ev.skill_id)
                    tmodel.skill_stats[ev.skill_id] = backup
                    decisions.add("progress" if p >= TAU else "review")
                st = tmodel.skill_stats[ev.skill_id]
                backup = dict(st)
                tmodel.update(ev)
                p_true = tmodel.mastery(ev.skill_id)
                tmodel.skill_stats[ev.skill_id] = backup
            else:
                assert neural is not None
                seen_fp: set[tuple] = set()
                uniq = []
                for syn in syns:
                    fp = (syn.skill_id, syn.action_types[:MAX_SEQ_ACTIONS])
                    if fp in seen_fp:
                        continue
                    seen_fp.add(fp)
                    uniq.append(syn)
                prefix = hist[max(0, t - MAX_EVAL_PREFIX_EVENTS) : t]
                ps = neural.mastery_branches_shared_prefix(prefix, uniq)
                decisions = {"progress" if p >= TAU else "review" for p in ps}
                p_true = neural.mastery_branch(prefix, ev)

            true_act = "progress" if p_true >= TAU else "review"
            is_stable = len(decisions) == 1
            false_stable = is_stable and (true_act not in decisions)
            if is_stable:
                n_stable += 1
                if false_stable:
                    n_fs += 1
                qualified = n_users >= n_min
                if qualified:
                    n_qual_stable += 1
                    if false_stable:
                        n_qual_fs += 1
                else:
                    n_disqualified_stable += 1
                    if false_stable:
                        n_disqualified_fs += 1
            else:
                n_set += 1
            if model_name == "transparent":
                tmodel.update(ev)

    return {
        "model": model_name,
        "schema": SCHEMA,
        "tau": TAU,
        "n_min": n_min,
        "n_events": n_events,
        "unsupported": n_unsup,
        "supported": n_supp,
        "stable": n_stable,
        "set_valued": n_set,
        "false_stable": n_fs,
        "false_stable_rate_among_stable": (n_fs / n_stable) if n_stable else None,
        "qualified_stable": n_qual_stable,
        "qualified_false_stable": n_qual_fs,
        "disqualified_stable": n_disqualified_stable,
        "disqualified_false_stable": n_disqualified_fs,
        "retained_stable_frac": (n_qual_stable / n_stable) if n_stable else None,
        "fs_remaining": n_qual_fs,
        "fs_removed": n_disqualified_fs,
        "fs_reduction_frac": (n_disqualified_fs / n_fs) if n_fs else None,
        "fs_rate_among_qualified_stable": (n_qual_fs / n_qual_stable) if n_qual_stable else None,
        "support_rate": (n_supp / n_events) if n_events else None,
    }


def gates(row: dict[str, Any]) -> dict[str, Any]:
    cov = row.get("retained_stable_frac")
    red = row.get("fs_reduction_frac")
    cov_ok = cov is not None and cov >= 0.50
    red_ok = red is not None and red >= 0.30
    return {
        "coverage_ok": cov_ok,
        "fs_reduction_ok": red_ok,
        "pass_a": bool(cov_ok and red_ok),
        "pass_b": not bool(cov_ok and red_ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gru", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    if not FREEZE.exists():
        raise FileNotFoundError(f"Freeze file missing: {FREEZE}")
    rule = parse_freeze_rule(FREEZE)
    n_min = int(rule["n_min"])
    print(f"FROZEN_RULE {rule}", flush=True)

    print("LOAD confirmatory + dicts…", flush=True)
    events = load_confirmatory()
    dicts_s0 = load_dicts_s0()
    print(f"events={len(events)} dict_keys={len(dicts_s0)}", flush=True)

    results: dict[str, Any] = {
        "freeze_rule": rule,
        "utc": utc_now(),
        "models": {},
    }

    print("EVAL TransparentActionPFA/S0…", flush=True)
    t_tr = time.time()
    tr = evaluate_model(events, dicts_s0, "transparent", n_min)
    tr["runtime_s"] = time.time() - t_tr
    tr["gates"] = gates(tr)
    results["models"]["TransparentActionPFA"] = tr
    print(json.dumps(tr, indent=2), flush=True)

    if not args.skip_gru and CKPT.exists():
        print(f"EVAL ActionTokenGRU/S0 device={args.device}…", flush=True)
        t_gr = time.time()
        neural = NeuralPilot(device=args.device)
        neural.load_bundle(torch.load(CKPT, map_location=args.device, weights_only=False))
        gr = evaluate_model(events, dicts_s0, "neural", n_min, neural=neural)
        gr["runtime_s"] = time.time() - t_gr
        gr["model"] = "ActionTokenGRU"
        gr["gates"] = gates(gr)
        results["models"]["ActionTokenGRU"] = gr
        print(json.dumps({k: gr[k] for k in gr if k != "gates"}, indent=2), flush=True)
        print("gates", gr["gates"], flush=True)
    else:
        results["models"]["ActionTokenGRU"] = {"skipped": True, "reason": "skip_gru or missing ckpt"}

    # Recommendation: Transparent required for PASS; GRU informational
    tr_pass = results["models"]["TransparentActionPFA"]["gates"]["pass_a"]
    results["recommendation"] = {
        "transparent_pass_a": tr_pass,
        "label": "PASS-A" if tr_pass else "PASS-B",
        "note": (
            "PASS-A/B decided from TransparentActionPFA/S0 confirmatory gates on frozen n_users>=5. "
            "GRU reported if run; not used to retune."
        ),
    }
    results["runtime_s"] = time.time() - t0
    out_path = OUT / "reliability_qualifier_confirmatory.json"
    write_json(out_path, results)
    print(f"WROTE {out_path} recommendation={results['recommendation']['label']}", flush=True)


if __name__ == "__main__":
    main()
