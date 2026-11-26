#!/usr/bin/env python3
"""ASSISTments held-out empirical validity audit (Authentic WDLF R1 rescue).

Uses prospective/wdlf_r7 sealed splits + development dicts + ActionTokenGRU ckpt.
Prefer exact match to r7_core.evaluate_split decision logic; confirmatory rich
traces evaluate certificates only (never enter C_dev / dict construction).

If neural is slow: TransparentActionPFA × S0–S3 full; ActionTokenGRU × S0/S1;
document deferrals for GRU S2/S3.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# Absolute project root (script may be dispatched via Temp allowlisted launcher).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from r7_core import (  # noqa: E402
    KAPPA,
    MAX_EVAL_PREFIX_EVENTS,
    MAX_SEQ_ACTIONS,
    SCHEMAS,
    TAU,
    Event,
    NeuralPilot,
    TransparentActionPFA,
    p2_certificate,
    release_key,
    rich_signature,
    syn_from_sig,
)
from src.policies.capacity_support_v2 import select_capacity_list

SEALED = ROOT / "prospective" / "wdlf_r7" / "sealed"
CKPT = SEALED / "checkpoints" / "action_token_gru_seed17.pt"
OUT = ROOT / "results" / "wdlf_r1_rescue"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_events() -> list[Event]:
    # Prefer split cache (lower RAM) else full events_cache.
    conf_path = SEALED / "splits" / "confirmatory.pkl"
    if conf_path.exists():
        with conf_path.open("rb") as f:
            return pickle.load(f)
    with (SEALED / "events_cache.pkl").open("rb") as f:
        blob = pickle.load(f)
    return [e for e in blob["events"] if e.split == "confirmatory"]


def load_dicts() -> dict[str, dict]:
    path = SEALED / "dicts_dev_compact.pkl"
    with path.open("rb") as f:
        blob = pickle.load(f)
    return blob["dicts"]


def _p2_false_cert(intervals: dict[str, tuple[float, float]], actual: dict[str, float], kappa: float) -> dict[str, Any]:
    p2 = p2_certificate(intervals, kappa=kappa)
    # Rebuild statuses via same interval rule as p2_certificate
    eligible = sorted(intervals.items(), key=lambda kv: kv[0])
    n_el = len(eligible)
    B = int(round(kappa * n_el)) if n_el else 0
    if n_el and B < 1:
        B = 1
    statuses: dict[str, str] = {}
    if n_el and B:
        iv = [(uid, lo, hi) for uid, (lo, hi) in eligible]
        for uid, lo, hi in iv:
            best = 1 + sum(1 for u2, lo2, hi2 in iv if u2 != uid and lo2 > hi)
            worst = 1 + sum(1 for u2, lo2, hi2 in iv if u2 != uid and hi2 >= lo)
            if worst <= B:
                statuses[uid] = "guaranteed_in"
            elif best > B:
                statuses[uid] = "guaranteed_out"
            else:
                statuses[uid] = "unresolved"
    users = sorted(actual)
    if not users:
        return {"false_gin": 0, "false_gout": 0, "false_gin_rate": None, "false_gout_rate": None,
                "false_cert_rate_among_resolved": None, **{f"p2_{k}": p2[k] for k in p2}}
    arr_u = np.array(users, dtype=object)
    arr_r = np.array([actual[u] for u in users], dtype=np.float64)
    selected = set(select_capacity_list(arr_u, arr_r, capacity=kappa))
    gin = [u for u in users if statuses.get(u) == "guaranteed_in"]
    gout = [u for u in users if statuses.get(u) == "guaranteed_out"]
    false_gin = sum(1 for u in gin if u not in selected)
    false_gout = sum(1 for u in gout if u in selected)
    n_resolved = len(gin) + len(gout)
    return {
        "n_guaranteed_in": len(gin),
        "n_guaranteed_out": len(gout),
        "n_unresolved": sum(1 for u in users if statuses.get(u) == "unresolved"),
        "false_gin": false_gin,
        "false_gout": false_gout,
        "false_gin_rate": (false_gin / len(gin)) if gin else None,
        "false_gout_rate": (false_gout / len(gout)) if gout else None,
        "false_cert_rate_among_resolved": ((false_gin + false_gout) / n_resolved) if n_resolved else None,
        "p2_certificate": p2,
    }


def evaluate_validity(
    events: list[Event],
    dicts: dict[str, dict],
    model_name: str,
    schemas: list[str],
    neural: Optional[NeuralPilot] = None,
    tau: float = TAU,
    kappa: float = KAPPA,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mirror r7_core.evaluate_split, plus held-out validity counters."""
    eval_users = sorted({e.user_id for e in events})
    eval_user_set = set(eval_users)
    user_hist: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        if e.user_id in eval_user_set:
            user_hist[e.user_id].append(e)
    n_all = len(eval_users)
    out: dict[str, Any] = {}
    user_aggs: dict[str, dict[str, Any]] = {}
    ckpt_dir = OUT / "assist_validity_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for s in schemas:
        ckpt_path = ckpt_dir / f"{model_name}_{s}.json"
        user_ckpt_path = ckpt_dir / f"{model_name}_{s}_users.json"
        if ckpt_path.exists() and user_ckpt_path.exists():
            print(f"VALID_RESUME model={model_name} schema={s} from {ckpt_path.name}", flush=True)
            out[s] = json.loads(ckpt_path.read_text(encoding="utf-8"))
            users_s = json.loads(user_ckpt_path.read_text(encoding="utf-8"))
            for uid, urow in users_s.items():
                user_aggs.setdefault(uid, {})[s] = urow
            continue

        stable = set_valued = unsupported = 0
        n_actual_in = n_exact_in = n_contained = n_false_stable = 0
        n_supported_eval = 0
        intervals: dict[str, tuple[float, float]] = {}
        actual_risks: dict[str, float] = {}
        schema_user: dict[str, dict[str, int]] = {}

        progress_path = ckpt_dir / f"{model_name}_{s}.progress.json"
        start_ui = 0
        if progress_path.exists() and model_name == "neural":
            try:
                prog = json.loads(progress_path.read_text(encoding="utf-8"))
                start_ui = int(prog.get("users_done", 0))
                stable = int(prog.get("stable", 0))
                set_valued = int(prog.get("set_valued", 0))
                unsupported = int(prog.get("unsupported", 0))
                n_actual_in = int(prog.get("n_actual_in", 0))
                n_exact_in = int(prog.get("n_exact_in", 0))
                n_contained = int(prog.get("n_contained", 0))
                n_false_stable = int(prog.get("n_false_stable", 0))
                n_supported_eval = int(prog.get("n_supported_eval", 0))
                intervals = {k: tuple(v) for k, v in (prog.get("intervals") or {}).items()}
                actual_risks = dict(prog.get("actual_risks") or {})
                schema_user = prog.get("schema_user") or {}
                print(
                    f"VALID_RESUME_PROGRESS model={model_name} schema={s} from_user={start_ui}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"VALID_RESUME_PROGRESS_FAIL {exc}", flush=True)
                start_ui = 0

        for ui, uid in enumerate(eval_users):
            if ui < start_ui:
                continue
            step = 50 if model_name == "neural" else 500
            if ui and ui % step == 0:
                print(
                    f"VALID_PROGRESS model={model_name} schema={s} users={ui}/{n_all}",
                    flush=True,
                )
                if model_name == "neural":
                    write_json(
                        progress_path,
                        {
                            "model": model_name,
                            "schema": s,
                            "users_done": ui,
                            "n_all": n_all,
                            "stable": stable,
                            "set_valued": set_valued,
                            "unsupported": unsupported,
                            "n_actual_in": n_actual_in,
                            "n_exact_in": n_exact_in,
                            "n_contained": n_contained,
                            "n_false_stable": n_false_stable,
                            "n_supported_eval": n_supported_eval,
                            "intervals": {k: list(v) for k, v in intervals.items()},
                            "actual_risks": actual_risks,
                            "schema_user": schema_user,
                            "utc": utc_now(),
                        },
                    )
            elif model_name == "neural" and ui < 5:
                print(
                    f"VALID_PROGRESS model={model_name} schema={s} users={ui}/{n_all} start_user",
                    flush=True,
                )
            hist = user_hist[uid]
            tmodel = TransparentActionPFA()
            last_interval = None
            last_actual_risk = None
            urow = schema_user.setdefault(
                uid,
                {
                    "n_events": 0,
                    "n_supported": 0,
                    "n_unsupported": 0,
                    "n_stable": 0,
                    "n_set_valued": 0,
                    "n_actual_in": 0,
                    "n_exact_sig_in": 0,
                    "n_contained": 0,
                    "n_false_stable": 0,
                },
            )
            for t, ev in enumerate(hist):
                urow["n_events"] += 1
                k = release_key(ev, s)
                entry = dicts[s].get(k)
                if entry is None:
                    unsupported += 1
                    urow["n_unsupported"] += 1
                    if model_name == "transparent":
                        tmodel.update(ev)
                    continue
                if isinstance(entry, dict) and "sigs" in entry:
                    sigs = entry["sigs"]
                    n_users = entry["n_users"]
                else:
                    sigs = list(entry.keys())
                    n_users = 2
                if n_users < 2 and len(sigs) < 2:
                    unsupported += 1
                    urow["n_unsupported"] += 1
                    if model_name == "transparent":
                        tmodel.update(ev)
                    continue

                decisions = set()
                risks = []
                syns = [syn_from_sig(ev, sig) for sig in sigs]
                if model_name == "transparent":
                    seen_fp: set[tuple] = set()
                    uniq_syns: list[Event] = []
                    for syn in syns:
                        fp = (
                            syn.correct,
                            syn.hint_count,
                            syn.attempt_count,
                            syn.action_types,
                        )
                        if fp in seen_fp:
                            continue
                        seen_fp.add(fp)
                        uniq_syns.append(syn)
                    for syn in uniq_syns:
                        st = tmodel.skill_stats[ev.skill_id]
                        backup = dict(st)
                        tmodel.update(syn)
                        p = tmodel.mastery(ev.skill_id)
                        tmodel.skill_stats[ev.skill_id] = backup
                        decisions.add("progress" if p >= tau else "review")
                        risks.append(1.0 - p)
                    # True action from actual completion
                    st = tmodel.skill_stats[ev.skill_id]
                    backup = dict(st)
                    tmodel.update(ev)
                    p_true = tmodel.mastery(ev.skill_id)
                    tmodel.skill_stats[ev.skill_id] = backup
                else:
                    assert neural is not None
                    seen_fp = set()
                    uniq_syns = []
                    for syn in syns:
                        fp = (syn.skill_id, syn.action_types[:MAX_SEQ_ACTIONS])
                        if fp in seen_fp:
                            continue
                        seen_fp.add(fp)
                        uniq_syns.append(syn)
                    prefix = hist[max(0, t - MAX_EVAL_PREFIX_EVENTS) : t]
                    ps = neural.mastery_branches_shared_prefix(prefix, uniq_syns)
                    for p in ps:
                        decisions.add("progress" if p >= tau else "review")
                        risks.append(1.0 - p)
                    p_true = neural.mastery_branch(prefix, ev)

                true_action = "progress" if p_true >= tau else "review"
                actual_sig = rich_signature(ev)
                # sigs may be list of tuples or Counter keys
                norm_sigs = [tuple(x) if isinstance(x, list) else x for x in sigs]
                sig_set = set(norm_sigs)
                exact_in = int(actual_sig in sig_set)
                # Decision-relevant fingerprint (matches r7_core collapse; ignores timestamps).
                if model_name == "transparent":
                    act_fp = (
                        ev.correct,
                        ev.hint_count,
                        ev.attempt_count,
                        ev.action_types,
                    )
                    cdev_fps = {
                        (
                            int(sig[1]),
                            int(sig[2]),
                            int(sig[3]),
                            tuple(sig[5]),
                        )
                        for sig in norm_sigs
                    }
                else:
                    act_fp = (ev.skill_id, ev.action_types[:MAX_SEQ_ACTIONS])
                    cdev_fps = {
                        (ev.skill_id, tuple(sig[5])[:MAX_SEQ_ACTIONS]) for sig in norm_sigs
                    }
                actual_in = int(act_fp in cdev_fps)
                contained = int(true_action in decisions)
                is_stable = len(decisions) == 1
                false_stable = int(is_stable and true_action not in decisions)

                if is_stable:
                    stable += 1
                    urow["n_stable"] += 1
                else:
                    set_valued += 1
                    urow["n_set_valued"] += 1
                n_supported_eval += 1
                urow["n_supported"] += 1
                n_actual_in += actual_in
                n_exact_in += exact_in
                n_contained += contained
                n_false_stable += false_stable
                urow["n_actual_in"] += actual_in
                urow["n_exact_sig_in"] += exact_in
                urow["n_contained"] += contained
                urow["n_false_stable"] += false_stable

                last_interval = (min(risks), max(risks))
                last_actual_risk = 1.0 - float(p_true)
                if model_name == "transparent":
                    tmodel.update(ev)

            if last_interval is not None:
                intervals[uid] = last_interval
                actual_risks[uid] = float(last_actual_risk)

        n_events = stable + set_valued + unsupported
        p2 = p2_certificate(intervals, kappa=kappa)
        no_ckpt = n_all - p2["eligible"]
        ext = _p2_false_cert(intervals, actual_risks, kappa)
        outside = 0
        for uid, (lo, hi) in intervals.items():
            ar = actual_risks[uid]
            if ar < lo - 1e-12 or ar > hi + 1e-12:
                outside += 1
        out[s] = {
            "model": model_name,
            "schema": s,
            "tau": tau,
            "kappa": kappa,
            "n_events": n_events,
            "supported": stable + set_valued,
            "unsupported": unsupported,
            "support_rate": (stable + set_valued) / n_events if n_events else None,
            "stable": stable,
            "set_valued": set_valued,
            "stable_given_supported": stable / (stable + set_valued) if (stable + set_valued) else None,
            "event_certified_stable_coverage": stable / n_events if n_events else None,
            "actual_completion_inclusion_rate_supported": (n_actual_in / n_supported_eval) if n_supported_eval else None,
            "actual_completion_inclusion_definition": "decision_fingerprint_in_C_dev (timestamps ignored; matches evaluate_split collapse)",
            "exact_rich_signature_inclusion_rate_supported": (n_exact_in / n_supported_eval) if n_supported_eval else None,
            "actual_action_containment_rate_supported": (n_contained / n_supported_eval) if n_supported_eval else None,
            "false_stable": n_false_stable,
            "false_stable_rate_among_stable": (n_false_stable / stable) if stable else None,
            "p2_full_users": n_all,
            "p2_eligible_users": p2["eligible"],
            "p2_guaranteed_in": p2["guaranteed_in"],
            "p2_guaranteed_out": p2["guaranteed_out"],
            "p2_unresolved": p2["unresolved"],
            "p2_unresolved_rate_eligible": p2["unresolved"] / p2["eligible"] if p2["eligible"] else None,
            "p2_all_user_noncertification_rate": (p2["unresolved"] + no_ckpt) / n_all if n_all else None,
            "p2_external_validity": {
                "false_gin": ext["false_gin"],
                "false_gout": ext["false_gout"],
                "false_gin_rate": ext["false_gin_rate"],
                "false_gout_rate": ext["false_gout_rate"],
                "false_cert_rate_among_resolved": ext["false_cert_rate_among_resolved"],
                "actual_risk_outside_interval": outside,
                "actual_risk_outside_interval_rate": (outside / len(intervals)) if intervals else None,
            },
        }
        # store per-user for this schema under model key
        users_s: dict[str, Any] = {}
        for uid, urow in schema_user.items():
            bag = user_aggs.setdefault(uid, {})
            bag[s] = dict(urow)
            if uid in intervals:
                bag[s]["p2_risk_min"] = intervals[uid][0]
                bag[s]["p2_risk_max"] = intervals[uid][1]
                bag[s]["p2_actual_risk"] = actual_risks[uid]
            else:
                bag[s]["p2_risk_min"] = None
                bag[s]["p2_risk_max"] = None
                bag[s]["p2_actual_risk"] = None
            users_s[uid] = bag[s]
        write_json(ckpt_path, out[s])
        write_json(user_ckpt_path, users_s)
        print(f"VALID_CKPT model={model_name} schema={s} -> {ckpt_path.name}", flush=True)

    return out, user_aggs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--neural-schemas",
        default="S0,S1",
        help="Comma schemas for ActionTokenGRU (default S0,S1; S2/S3 deferred if slow)",
    )
    ap.add_argument("--skip-neural", action="store_true")
    ap.add_argument("--skip-pfa", action="store_true")
    ap.add_argument("--pfa-schemas", default="S0,S1,S2,S3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[assist_heldout] loading confirmatory events + dev dicts", flush=True)
    events = load_events()
    dicts = load_dicts()
    n_users = len({e.user_id for e in events})
    print(f"[assist_heldout] conf events={len(events)} users={n_users}", flush=True)

    pfa_schemas = [s.strip() for s in args.pfa_schemas.split(",") if s.strip()]
    neural_schemas = [s.strip() for s in args.neural_schemas.split(",") if s.strip()]
    deferred = []

    pfa_metrics: dict[str, Any] = {}
    pfa_users: dict[str, Any] = {}
    pfa_runtime = None
    if not args.skip_pfa:
        print(f"[assist_heldout] TransparentActionPFA schemas={pfa_schemas}", flush=True)
        t_pfa = time.time()
        pfa_metrics, pfa_users = evaluate_validity(events, dicts, "transparent", pfa_schemas)
        pfa_runtime = time.time() - t_pfa
        # Checkpoint PFA early so a neural-only rerun can merge.
        write_json(OUT / "assist_heldout_validity_pfa_partial.json", {"TransparentActionPFA": pfa_metrics, "runtime_s": pfa_runtime})
        write_json(OUT / "assist_heldout_user_aggregates_pfa_partial.json", {"TransparentActionPFA": pfa_users})
        print(f"[assist_heldout] PFA done runtime_s={pfa_runtime:.1f}", flush=True)
    else:
        partial = OUT / "assist_heldout_validity_pfa_partial.json"
        if partial.exists():
            pfa_metrics = json.loads(partial.read_text(encoding="utf-8")).get("TransparentActionPFA", {})
            pfa_users = json.loads((OUT / "assist_heldout_user_aggregates_pfa_partial.json").read_text(encoding="utf-8")).get(
                "TransparentActionPFA", {}
            )
            print(f"[assist_heldout] loaded PFA partial from {partial}", flush=True)
        else:
            deferred.append("TransparentActionPFA/* (skipped, no partial)")

    gru_metrics: dict[str, Any] = {}
    gru_users: dict[str, Any] = {}
    gru_runtime = None
    if not args.skip_neural:
        print(f"[assist_heldout] loading ActionTokenGRU {CKPT} device={args.device}", flush=True)
        neural = NeuralPilot(device=args.device)
        neural.load_bundle(torch.load(CKPT, map_location=args.device, weights_only=False))
        # Defer S2/S3 if not requested
        for s in SCHEMAS:
            if s not in neural_schemas:
                deferred.append(f"ActionTokenGRU/{s}")
        print(f"[assist_heldout] ActionTokenGRU schemas={neural_schemas} deferred={deferred}", flush=True)
        t_g = time.time()
        gru_metrics, gru_users = evaluate_validity(events, dicts, "neural", neural_schemas, neural=neural)
        gru_runtime = time.time() - t_g
        print(f"[assist_heldout] GRU done runtime_s={gru_runtime:.1f}", flush=True)
    else:
        deferred.extend([f"ActionTokenGRU/{s}" for s in SCHEMAS])

    runtime_s = time.time() - t0
    payload = {
        "created_utc": utc_now(),
        "protocol": {
            "tau": TAU,
            "kappa": KAPPA,
            "dict_source": str(SEALED / "dicts_dev_compact.pkl"),
            "events_source": "prospective/wdlf_r7/sealed confirmatory",
            "invariant": "development-only dicts; confirmatory used only for evaluation",
            "decision_logic": "exact match to r7_core.evaluate_split forks + mastery>=tau",
        },
        "population": {"n_users": n_users, "n_events": len(events)},
        "TransparentActionPFA": pfa_metrics,
        "ActionTokenGRU": gru_metrics,
        "deferred": deferred,
        "runtime_s": runtime_s,
        "runtime_breakdown_s": {"transparent": pfa_runtime, "neural": gru_runtime},
    }
    out_json = OUT / "assist_heldout_validity.json"
    write_json(out_json, payload)
    write_json(
        OUT / "assist_heldout_user_aggregates.json",
        {
            "tau": TAU,
            "kappa": KAPPA,
            "n_frame_users": n_users,
            "TransparentActionPFA": pfa_users,
            "ActionTokenGRU": gru_users,
            "schemas_pfa": pfa_schemas,
            "schemas_gru": neural_schemas if not args.skip_neural else [],
        },
    )
    print(json.dumps({"wrote": str(out_json), "runtime_s": runtime_s, "deferred": deferred}, indent=2))
    print(f"[assist_heldout] DONE runtime_s={runtime_s:.1f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
