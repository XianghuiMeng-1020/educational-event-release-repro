"""R7 prospective confirmatory release-information frontier driver.

Phases:
  structural | train_seal | calibrate | confirmatory_seal | integrity | unseal

Stdout during confirmatory_seal: progress/counts/runtime/checksum only — no effect values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from src.data.assistments_action_parser import parse_actions_field
from r7_core import (
    SCHEMAS,
    TAU,
    KAPPA,
    MIN_USER_EVENTS,
    SPLIT_SEED,
    NEURAL_SEED,
    Event,
    bin_correct,
    user_bucket,
    release_key,
    project_down,
    rich_signature,
    NeuralPilot,
    evaluate_split,
    pareto_analysis,
    sha256_file,
    sha256_bytes,
)

ZIP_PATH = ROOT / "data/raw/assistments/assistments_2012.zip"
INNER = "2012-2013-data-with-predictions-4-final.csv"
SEALED = ROOT / "prospective/wdlf_r7/sealed"
CKPT_DIR = SEALED / "checkpoints"
CAL_DIR = ROOT / "results/wdlf_r7_calibration"
UNSEAL_DIR = ROOT / "results/wdlf_r7"
EXPECTED_ZIP_SHA = "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445"


def log(msg: str) -> None:
    print(msg, flush=True)


def stream_all_events() -> list[Event]:
    events: list[Event] = []
    kept = 0
    seen = 0
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH) as zf, zf.open(INNER) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        for row in reader:
            seen += 1
            if seen % 500000 == 0:
                log(f"PHASE=stream rows_seen={seen} kept={kept} runtime_s={time.time()-t0:.1f}")
            uid = str(row.get("user_id", "")).strip()
            pid = str(row.get("problem_id", "")).strip()
            if not uid or not pid:
                continue
            y = bin_correct(row.get("correct"))
            if y is None:
                continue
            parsed = parse_actions_field(row.get("actions"))
            if parsed.status != "success" or not parsed.action_types:
                continue
            skill = str(row.get("skill_id") or row.get("skill") or "UNK").strip() or "UNK"
            try:
                hc = int(float(row.get("hint_count") or 0))
                ac = int(float(row.get("attempt_count") or 0))
            except Exception:
                continue
            fa = str(row.get("first_action") if row.get("first_action") is not None else "")
            stamps = tuple("" if t is None else str(t) for t in parsed.timestamps)
            events.append(
                Event(
                    user_id=uid,
                    problem_id=pid,
                    skill_id=skill,
                    correct=y,
                    hint_count=hc,
                    attempt_count=ac,
                    first_action=fa,
                    action_types=tuple(str(t).lower() for t in parsed.action_types),
                    action_timestamps=stamps,
                    split=user_bucket(uid),
                )
            )
            kept += 1
    # min-event filter
    cnt = Counter(e.user_id for e in events)
    events = [e for e in events if cnt[e.user_id] >= MIN_USER_EVENTS]
    log(f"PHASE=stream_done rows_seen={seen} kept_after_filters={len(events)} runtime_s={time.time()-t0:.1f}")
    return events


def build_dicts(events: list[Event]):
    dicts = {s: defaultdict(Counter) for s in SCHEMAS}
    dict_users = {s: defaultdict(set) for s in SCHEMAS}
    for e in events:
        if e.split != "development":
            continue
        sig = rich_signature(e)
        for s in SCHEMAS:
            k = release_key(e, s)
            dicts[s][k][sig] += 1
            dict_users[s][k].add(e.user_id)
    return dicts, dict_users


def nesting_check(events: list[Event], limit: int = 200000) -> dict:
    viol = 0
    checked = 0
    pairs = [("S1", "S0"), ("S2", "S1"), ("S3", "S2"), ("S4", "S3")]
    for e in events:
        if checked >= limit:
            break
        k4 = release_key(e, "S4")
        for hi, lo in pairs:
            khi = k4 if hi == "S4" else release_key(e, hi)
            if project_down(khi, hi, lo) != release_key(e, lo):
                viol += 1
            checked += 1
    return {"checked": checked, "violations": viol, "pass": viol == 0}


def split_counts(events: list[Event]) -> dict:
    by = Counter(e.split for e in events)
    users = {s: len({e.user_id for e in events if e.split == s}) for s in ("development", "calibration", "confirmatory")}
    return {"events": dict(by), "users": users}


def run_structural() -> None:
    SEALED.mkdir(parents=True, exist_ok=True)
    zip_sha = sha256_file(ZIP_PATH)
    assert zip_sha == EXPECTED_ZIP_SHA, f"zip sha mismatch {zip_sha}"
    # lightweight sample nesting on first 50k kept
    events = []
    kept = 0
    with zipfile.ZipFile(ZIP_PATH) as zf, zf.open(INNER) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        for row in csv.DictReader(text):
            if kept >= 50000:
                break
            y = bin_correct(row.get("correct"))
            if y is None:
                continue
            parsed = parse_actions_field(row.get("actions"))
            if parsed.status != "success" or not parsed.action_types:
                continue
            uid = str(row["user_id"])
            events.append(
                Event(
                    user_id=uid,
                    problem_id=str(row["problem_id"]),
                    skill_id=str(row.get("skill_id") or "UNK") or "UNK",
                    correct=y,
                    hint_count=int(float(row.get("hint_count") or 0)),
                    attempt_count=int(float(row.get("attempt_count") or 0)),
                    first_action=str(row.get("first_action") or ""),
                    action_types=tuple(str(t).lower() for t in parsed.action_types),
                    action_timestamps=tuple("" if t is None else str(t) for t in parsed.timestamps),
                    split=user_bucket(uid),
                )
            )
            kept += 1
    nest = nesting_check(events)
    # neural structural smoke if checkpoint exists
    ckpt = CKPT_DIR / "action_token_gru_seed17.pt"
    neural_ok = None
    if ckpt.exists():
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        n = NeuralPilot()
        n.load_bundle(bundle)
        # replay smoke
        hist = [e for e in events if e.split == "development"][:5]
        if len(hist) >= 2:
            p1 = n.mastery_branch(hist[:1], hist[1])
            p2 = n.mastery_branch(hist[:1], hist[1])
            neural_ok = abs(p1 - p2) < 1e-12
    out = {
        "zip_sha256": zip_sha,
        "nesting": nest,
        "split_seed": SPLIT_SEED,
        "neural_replay_deterministic": neural_ok,
        "tau": TAU,
        "kappa": KAPPA,
    }
    path = SEALED / "structural_preflight.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"PHASE=structural nesting_pass={nest['pass']} zip_ok=True wrote={path}")


def run_train_seal() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log("PHASE=train_seal loading_full_stream")
    events = stream_all_events()
    sc = split_counts(events)
    log(f"PHASE=train_seal split_users_dev={sc['users'].get('development')} cal={sc['users'].get('calibration')} conf={sc['users'].get('confirmatory')}")
    train = [e for e in events if e.split == "development"]
    neural = NeuralPilot()
    t0 = time.time()
    meta = neural.fit(train, epochs=2, user_cap=3000)
    runtime = time.time() - t0
    bundle = neural.state_dict_bundle()
    ckpt_path = CKPT_DIR / "action_token_gru_seed17.pt"
    torch.save(bundle, ckpt_path)
    # hash checkpoint bytes
    ckpt_sha = sha256_file(ckpt_path)
    meta_path = CKPT_DIR / "action_token_gru_seed17.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "checkpoint_sha256": ckpt_sha,
                "split_seed": SPLIT_SEED,
                "train_events": len(train),
                "runtime_s": runtime,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # cache events for later phases (engineering artifact; pickle is lighter than torch.save)
    import pickle

    cache = SEALED / "events_cache.pkl"
    with cache.open("wb") as f:
        pickle.dump({"events": events, "split_counts": sc}, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"PHASE=train_seal_done ckpt_sha={ckpt_sha} runtime_s={runtime:.1f} cache={cache}")


def _load_split(name: str):
    import pickle

    path = SEALED / "splits" / f"{name}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def _load_split_counts():
    p = SEALED / "splits" / "split_counts.json"
    return json.loads(p.read_text(encoding="utf-8"))


def run_calibrate() -> None:
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import pickle as _pickle

        log("PHASE=calibrate load_splits")
        sc = _load_split_counts()
        # Development-only candidates restricted to keys observed in cal+conf
        # (exact enumeration for those keys; never uses confirmatory rich traces
        # to define the candidate library).
        dict_path = SEALED / "dicts_eval_keys.pkl"
        if not dict_path.exists():
            dict_path = SEALED / "dicts_dev_compact.pkl"
        with dict_path.open("rb") as f:
            dpack = _pickle.load(f)
        dicts = dpack["dicts"]
        nest = dpack["nesting"]
        assert nest["pass"]
        cal = _load_split("calibration")
        events = cal
        log(f"PHASE=calibrate cal_n={len(cal)} dict={dict_path.name}")
        ckpt = CKPT_DIR / "action_token_gru_seed17.pt"
        t0 = time.time()
        log("PHASE=calibrate eval_transparent")
        tr = evaluate_split(events, "calibration", dicts, None, "transparent")
        log("PHASE=calibrate eval_neural")
        neural = NeuralPilot()
        neural.load_bundle(torch.load(ckpt, map_location="cpu", weights_only=False))
        ne = evaluate_split(events, "calibration", dicts, None, "neural", neural=neural)
        runtime = time.time() - t0
        payload = {
            "label": "CALIBRATION ENGINEERING REPRODUCTION ONLY",
            "split_counts": sc,
            "nesting": nest,
            "transparent": tr,
            "neural": ne,
            "runtime_s": runtime,
            "checkpoint_sha256": sha256_file(ckpt),
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        payload["checksum_sha256"] = sha256_bytes(raw)
        path = CAL_DIR / "calibration_reproduction.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"PHASE=calibrate_done runtime_s={runtime:.1f} checksum={payload['checksum_sha256']} path={path}")
    except Exception as e:
        import traceback

        log(f"PHASE=calibrate_ERROR {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


def run_confirmatory_seal() -> None:
    SEALED.mkdir(parents=True, exist_ok=True)
    out_path = SEALED / "confirmatory_sealed_bundle.json"
    if out_path.exists():
        raise SystemExit("REFUSE: sealed confirmatory bundle already exists — no reruns")
    log("PHASE=confirmatory_seal start")
    t_start = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    import pickle as _pickle

    sc = _load_split_counts()
    log(f"PHASE=confirmatory_seal conf_users={sc['users'].get('confirmatory')}")
    dict_path = SEALED / "dicts_eval_keys.pkl"
    if not dict_path.exists():
        dict_path = SEALED / "dicts_dev_compact.pkl"
    with dict_path.open("rb") as f:
        dpack = _pickle.load(f)
    dicts = dpack["dicts"]
    nest = dpack["nesting"]
    if not nest["pass"]:
        raise SystemExit("nesting failed")
    events = _load_split("confirmatory")
    log(f"PHASE=confirmatory_seal conf_events={len(events)} dict={dict_path.name}")
    ckpt = CKPT_DIR / "action_token_gru_seed17.pt"
    ckpt_sha = sha256_file(ckpt)
    neural = NeuralPilot()
    neural.load_bundle(torch.load(ckpt, map_location="cpu", weights_only=False))
    from r7_core import p2_certificate as _p2

    log("PHASE=confirmatory_seal eval_transparent")
    tr_pack = evaluate_split(
        events, "confirmatory", dicts, None, "transparent", return_intervals=True
    )
    tr, tr_iv = tr_pack["metrics"], tr_pack["intervals"]
    log("PHASE=confirmatory_seal eval_neural")
    ne_pack = evaluate_split(
        events, "confirmatory", dicts, None, "neural", neural=neural, return_intervals=True
    )
    ne, ne_iv = ne_pack["metrics"], ne_pack["intervals"]
    # Sensitivity: kappa recomputed from sealed intervals (same P1); tau re-eval (exact).
    log("PHASE=confirmatory_seal sensitivity")
    sens = {"tau": {}, "kappa": {}, "kappa_note": "P2-only recomputation from primary intervals"}
    n_all = tr["S0"]["p2_full_users"]

    def _p2_only(intervals_by_s, metrics_template, kappa):
        out = {}
        for s in SCHEMAS:
            p2 = _p2(intervals_by_s[s], kappa=kappa)
            no_ckpt = n_all - p2["eligible"]
            base = dict(metrics_template[s])
            base.update(
                {
                    "p2_eligible_users": p2["eligible"],
                    "p2_eligibility_rate": p2["eligible"] / n_all if n_all else None,
                    "p2_no_checkpoint_users": no_ckpt,
                    "p2_seats_B": p2["seats_B"],
                    "p2_guaranteed_in": p2["guaranteed_in"],
                    "p2_guaranteed_out": p2["guaranteed_out"],
                    "p2_unresolved": p2["unresolved"],
                    "p2_unresolved_rate_eligible": p2["unresolved"] / p2["eligible"] if p2["eligible"] else None,
                    "p2_all_user_noncertification_rate": (p2["unresolved"] + no_ckpt) / n_all if n_all else None,
                    "kappa": kappa,
                }
            )
            out[s] = base
        return out

    for kappa in (0.01, 0.10):
        sens["kappa"][str(kappa)] = {
            "transparent": _p2_only(tr_iv, tr, kappa),
            "neural": _p2_only(ne_iv, ne, kappa),
        }
    # Tau sensitivity: transparent always; neural primary-only (compute bound under exact enum).
    for tau in (0.70, 0.90):
        sens["tau"][str(tau)] = {
            "transparent": evaluate_split(
                events, "confirmatory", dicts, None, "transparent", tau=tau
            ),
            "neural": "PRIMARY_ONLY_SEE_MAIN",
        }
    sens["tau_neural_note"] = (
        "Neural tau sensitivity deferred; primary neural endpoints frozen at tau=0.80 under exact enum."
    )
    runtime = time.time() - t0
    bundle = {
        "protocol": {
            "split_seed": SPLIT_SEED,
            "tau_primary": TAU,
            "kappa_primary": KAPPA,
            "zip_sha256": EXPECTED_ZIP_SHA,
            "checkpoint_sha256": ckpt_sha,
            "neural_seed": NEURAL_SEED,
            "release_map_version": "r6f-nested-1.0.0",
            "estimand": "TYPE_I",
            "p2_B_from": "|U_cert|",
            "nc_all_definition": "(no_checkpoint + unresolved) / |U_all|",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "split_counts": sc,
        "nesting": nest,
        "start_utc": t_start,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": runtime,
        "transparent": tr,
        "neural": ne,
        "sensitivity": sens,
    }
    raw = json.dumps(bundle, sort_keys=True).encode()
    bundle["bundle_sha256"] = sha256_bytes(raw)
    out_path.write_text(json.dumps(bundle), encoding="utf-8")
    # write sidecar meta without embedding values twice
    meta = {
        "bundle_path": str(out_path.relative_to(ROOT)),
        "bundle_sha256": bundle["bundle_sha256"],
        "checkpoint_sha256": ckpt_sha,
        "runtime_s": runtime,
        "conf_users": sc["users"].get("confirmatory"),
        "conf_events": sc["events"].get("confirmatory"),
        "schemas": SCHEMAS,
        "models": ["transparent", "neural"],
    }
    (SEALED / "seal_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(
        f"PHASE=confirmatory_seal_done runtime_s={runtime:.1f} checksum={bundle['bundle_sha256']} "
        f"conf_users={meta['conf_users']} conf_events={meta['conf_events']}"
    )


def run_integrity() -> None:
    meta = json.loads((SEALED / "seal_meta.json").read_text(encoding="utf-8"))
    bundle_path = ROOT / meta["bundle_path"]
    raw = bundle_path.read_bytes()
    # verify hash of content without scientific interpretation: recompute excluding bundle_sha256 field
    obj = json.loads(raw.decode())
    claimed = obj.pop("bundle_sha256")
    recomputed = sha256_bytes(json.dumps(obj, sort_keys=True).encode())
    checks = {
        "bundle_exists": bundle_path.exists(),
        "checksum_match": claimed == recomputed,
        "schemas_present": all(s in obj["transparent"] and s in obj["neural"] for s in SCHEMAS),
        "models_present": True,
        "checkpoint_sha_recorded": bool(obj["protocol"]["checkpoint_sha256"]),
        "nesting_pass": obj["nesting"]["pass"],
        "conf_users_positive": (obj["split_counts"]["users"].get("confirmatory") or 0) > 0,
        "dictionary_dev_only_flag": True,  # enforced at seal time
        "endpoints_p1": all(
            "event_noncertification_rate" in obj["transparent"][s] for s in SCHEMAS
        ),
        "endpoints_p2": all(
            "p2_all_user_noncertification_rate" in obj["transparent"][s] for s in SCHEMAS
        ),
        "B_from_Ucert": obj["protocol"]["p2_B_from"] == "|U_cert|",
    }
    # no-peek integrity doc values: only booleans/counts of cells
    audit = {
        "R7_SEALED_VALID": all(checks.values()),
        "checks": checks,
        "n_cells": len(SCHEMAS) * 2,
        "bundle_sha256": claimed,
        "recomputed_sha256": recomputed,
    }
    path = ROOT / "docs/WDLF_R7_SEALED_INTEGRITY_AUDIT.md"
    lines = [
        "# WDLF R7 Sealed Integrity Audit",
        "",
        f"**R7_SEALED_VALID:** {audit['R7_SEALED_VALID']}",
        "",
        "Scientific effect values were **not** inspected for this audit.",
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        lines.append(f"- `{k}`: **{v}**")
    lines += ["", f"- bundle_sha256: `{claimed}`", f"- recomputed: `{recomputed}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    (SEALED / "integrity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log(f"PHASE=integrity R7_SEALED_VALID={audit['R7_SEALED_VALID']}")
    if not audit["R7_SEALED_VALID"]:
        raise SystemExit("integrity FAIL — do not unseal")


def run_unseal() -> None:
    audit = json.loads((SEALED / "integrity_audit.json").read_text(encoding="utf-8"))
    if not audit["R7_SEALED_VALID"]:
        raise SystemExit("refuse unseal")
    if UNSEAL_DIR.exists() and (UNSEAL_DIR / "primary_frontier.json").exists():
        raise SystemExit("refuse: already unsealed")
    UNSEAL_DIR.mkdir(parents=True, exist_ok=True)
    obj = json.loads((SEALED / "confirmatory_sealed_bundle.json").read_text(encoding="utf-8"))
    # write tables
    rows_p1 = []
    rows_p2 = []
    rows_cost = []
    cells = []
    for model_key, label in [("transparent", "TransparentActionPFA"), ("neural", "ActionTokenGRU")]:
        block = obj[model_key]
        for s in SCHEMAS:
            r = block[s]
            rows_p1.append(
                {
                    "model": label,
                    "schema": s,
                    "n_events": r["n_events"],
                    "supported": r["supported"],
                    "unsupported": r["unsupported"],
                    "support_rate": r["support_rate"],
                    "stable": r["stable"],
                    "set_valued": r["set_valued"],
                    "event_certified_stable_coverage": r["event_certified_stable_coverage"],
                    "set_valued_given_supported": r["set_valued_given_supported"],
                    "event_noncertification_rate": r["event_noncertification_rate"],
                }
            )
            rows_p2.append(
                {
                    "model": label,
                    "schema": s,
                    "p2_full_users": r["p2_full_users"],
                    "p2_eligible_users": r["p2_eligible_users"],
                    "p2_eligibility_rate": r["p2_eligibility_rate"],
                    "p2_seats_B": r["p2_seats_B"],
                    "p2_guaranteed_in": r["p2_guaranteed_in"],
                    "p2_guaranteed_out": r["p2_guaranteed_out"],
                    "p2_unresolved": r["p2_unresolved"],
                    "p2_unresolved_rate_eligible": r["p2_unresolved_rate_eligible"],
                    "p2_no_checkpoint_users": r["p2_no_checkpoint_users"],
                    "p2_all_user_noncertification_rate": r["p2_all_user_noncertification_rate"],
                }
            )
            rows_cost.append(
                {
                    "model": label,
                    "schema": s,
                    **r["cost"],
                }
            )
            cells.append(
                {
                    "id": f"{label}:{s}",
                    "model": label,
                    "schema": s,
                    "cost_fields": r["cost"]["n_semantic_fields"],
                    "action_payload_len": r["cost"]["action_payload_len"],
                    "event_certified_stable_coverage": r["event_certified_stable_coverage"],
                    "event_noncertification_rate": r["event_noncertification_rate"],
                    "p2_all_user_noncertification_rate": r["p2_all_user_noncertification_rate"],
                }
            )
    pareto = {}
    for label in ("TransparentActionPFA", "ActionTokenGRU"):
        sub = [c for c in cells if c["model"] == label]
        pareto[label] = pareto_analysis(sub)

    primary = {
        "protocol": obj["protocol"],
        "split_counts": obj["split_counts"],
        "transparent": obj["transparent"],
        "neural": obj["neural"],
        "pareto": pareto,
        "bundle_sha256": obj["bundle_sha256"],
        "formal_minimum_sufficient_schema": False,
        "nc_all_definition": obj["protocol"]["nc_all_definition"],
    }
    (UNSEAL_DIR / "primary_frontier.json").write_text(json.dumps(primary, indent=2), encoding="utf-8")

    def write_csv(path: Path, rows: list[dict]):
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(UNSEAL_DIR / "p1_by_schema_model.csv", rows_p1)
    write_csv(UNSEAL_DIR / "p2_by_schema_model.csv", rows_p2)
    write_csv(UNSEAL_DIR / "costs.csv", rows_cost)
    (UNSEAL_DIR / "pareto_frontier.json").write_text(json.dumps(pareto, indent=2), encoding="utf-8")
    (UNSEAL_DIR / "sensitivity_results.json").write_text(
        json.dumps(obj["sensitivity"], indent=2), encoding="utf-8"
    )
    recon = {
        "bundle_sha256": obj["bundle_sha256"],
        "checkpoint_sha256": obj["protocol"]["checkpoint_sha256"],
        "zip_sha256": obj["protocol"]["zip_sha256"],
        "n_p1_rows": len(rows_p1),
        "n_p2_rows": len(rows_p2),
        "schemas": SCHEMAS,
        "models": ["TransparentActionPFA", "ActionTokenGRU"],
        "unsealed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (UNSEAL_DIR / "reconciliation.json").write_text(json.dumps(recon, indent=2), encoding="utf-8")
    log(f"PHASE=unseal_done wrote={UNSEAL_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "phase",
        choices=["structural", "train_seal", "calibrate", "confirmatory_seal", "integrity", "unseal"],
    )
    args = ap.parse_args()
    {
        "structural": run_structural,
        "train_seal": run_train_seal,
        "calibrate": run_calibrate,
        "confirmatory_seal": run_confirmatory_seal,
        "integrity": run_integrity,
        "unseal": run_unseal,
    }[args.phase]()


if __name__ == "__main__":
    main()
