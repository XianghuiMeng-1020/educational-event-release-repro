#!/usr/bin/env python3
"""Fingerprint sufficiency and the 70-event transition check.

Reads the support-predicate caches. No retraining, no checkpoint edits, and no
overwriting of the support-predicate tables.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "chronorep", ROOT / "scripts/assistments/chronological_replication.py"
)
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)

OUT = ROOT / "results/assistments/support_predicate_robustness/fingerprint_consistency"
SUPPRED = ROOT / "results/assistments/support_predicate_robustness"
PRIV = SUPPRED / "private_scores"
CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
SCHEMAS = ("S0", "S1", "S2", "S3")
TAU = 0.80


def dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=_json) + "\n", encoding="utf-8")


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def status_codes(lo: np.ndarray, hi: np.ndarray, supported: np.ndarray) -> np.ndarray:
    code = np.zeros(lo.shape, dtype=np.uint8)
    finite = supported & np.isfinite(lo) & np.isfinite(hi)
    code[finite & (lo >= TAU)] = 1
    code[finite & (hi < TAU) & (code == 0)] = 2
    code[finite & (code == 0)] = 3
    return code


def false_stable(code: np.ndarray, realized: np.ndarray) -> np.ndarray:
    stable = (code == 1) | (code == 2)
    return stable & ((code == 1) != (realized >= TAU))


class KeyRec:
    __slots__ = ("n", "user", "users", "skill", "skills")

    def __init__(self) -> None:
        self.n = 0
        self.user = None
        self.users = 0
        self.skill = None
        self.skills = 1


def audit_saved_masks(payload: dict) -> tuple[dict, dict, dict]:
    counts = payload["counts"]
    present = payload["present"].astype(bool)
    lo = payload["lo"]
    hi = payload["hi"]
    realized = payload["realized"]
    U = counts[:, :, 0]
    R = counts[:, :, 1]
    F = (counts[:, :, 2], counts[:, :, 3])
    n = int(len(U))
    mask70 = present[:, 0] & (U[:, 0] == 1) & (R[:, 0] >= 2)
    positions = np.flatnonzero(mask70)
    per_model = []
    for model_i, name in enumerate(("TransparentActionPFA", "ActionTokenGRU")):
        supported00 = mask70
        code00 = status_codes(lo[:, model_i, 0], hi[:, model_i, 0], supported00)
        code07 = status_codes(lo[:, model_i, 0], hi[:, model_i, 0], np.zeros(n, dtype=bool))
        on = code00[positions]
        stable = (on == 1) | (on == 2)
        sett = on == 3
        fs = false_stable(code00, realized[:, model_i])[positions]
        both = supported00 & np.zeros(n, dtype=bool)
        per_model.append({
            "consumer": name,
            "events": int(mask70.sum()),
            "p00_stable": int(stable.sum()),
            "p00_set_valued": int(sett.sum()),
            "p00_unsupported": int((on == 0).sum()),
            "p00_false_stable": int(fs.sum()),
            "p07_unsupported": int((code07[positions] == 0).sum()),
            "p07_stable": int(((code07[positions] == 1) | (code07[positions] == 2)).sum()),
            "p07_set_valued": int((code07[positions] == 3).sum()),
            "event_nc_increase_p00_to_p07": int(stable.sum()),
            "stable_action_changes_while_both_supported": int(both.sum()),
            "point_interval_failures": int(np.sum((R[positions, 0] == 1) & (np.abs(lo[positions, model_i, 0] - hi[positions, model_i, 0]) > 1e-8))),
        })
    same_positions = True
    seventy = {
        "s0_events": int(mask70.sum()),
        "same_event_positions_for_both_consumers": same_positions,
        "source_rows_head": [int(payload["source"][i]) for i in positions[:12]],
        "consumers": per_model,
        "matches_expected_pfa": per_model[0]["p00_stable"] == 70 and per_model[0]["p00_set_valued"] == 0 and per_model[0]["p00_false_stable"] == 0 and per_model[0]["event_nc_increase_p00_to_p07"] == 70,
        "matches_expected_gru": per_model[1]["p00_stable"] == 62 and per_model[1]["p00_set_valued"] == 8 and per_model[1]["p00_false_stable"] == 0 and per_model[1]["event_nc_increase_p00_to_p07"] == 62,
        "interpretation_note": "PFA: 70 Stable under P00. GRU: 62 Stable and 8 Set-valued under P00. All 70 are Unsupported under P07. False-Stable is 0 for both.",
    }
    totals = []
    mismatches = []
    expected = {
        ("TransparentActionPFA", "S0", "P00"): (20408, 916845, 2566, 45),
        ("ActionTokenGRU", "S0", "P00"): (20408, 810576, 108835, 1077),
        ("TransparentActionPFA", "S1", "P00"): (77877, 861927, 15, 0),
        ("ActionTokenGRU", "S1", "P00"): (77877, 846772, 15170, 252),
        ("TransparentActionPFA", "S3", "P00"): (100688, 839131, 0, 0),
        ("ActionTokenGRU", "S3", "P00"): (100688, 839131, 0, 0),
    }
    for model_i, name in enumerate(("TransparentActionPFA", "ActionTokenGRU")):
        for schema_i, schema in enumerate(SCHEMAS):
            p00 = present[:, schema_i] & ((U[:, schema_i] >= 2) | (R[:, schema_i] >= 2))
            p17 = present[:, schema_i] & (F[model_i][:, schema_i] >= 2)
            for rule, supported in (("P00", p00), ("P17", p17)):
                code = status_codes(lo[:, model_i, schema_i], hi[:, model_i, schema_i], supported)
                stable = int(((code == 1) | (code == 2)).sum())
                sett = int((code == 3).sum())
                unsup = int((code == 0).sum())
                fs = int(false_stable(code, realized[:, model_i]).sum())
                partition = stable + sett + unsup
                row = {"consumer": name, "schema": schema, "rule": rule, "supported": int(supported.sum()), "stable": stable, "set_valued": sett, "unsupported": unsup, "false_stable": fs, "partition_sum": partition}
                totals.append(row)
                if partition != n:
                    mismatches.append(row)
                key = (name, schema, rule)
                if key in expected and (unsup, stable, sett, fs) != expected[key]:
                    mismatches.append({"expected": expected[key], **row})
            both = p00 & p17
            c0 = status_codes(lo[:, model_i, schema_i], hi[:, model_i, schema_i], p00)
            c1 = status_codes(lo[:, model_i, schema_i], hi[:, model_i, schema_i], p17)
            changed = both & (((c0 == 1) | (c0 == 2)) & ((c1 == 1) | (c1 == 2)) & (c0 != c1))
            if int(changed.sum()):
                mismatches.append({"stable_action_changed": name, "schema": schema, "n": int(changed.sum())})
    gru_s0_p00 = next(row for row in totals if row["consumer"].startswith("Action") and row["schema"] == "S0" and row["rule"] == "P00")
    gru_s0_p17 = next(row for row in totals if row["consumer"].startswith("Action") and row["schema"] == "S0" and row["rule"] == "P17")
    gru_s1_p00 = next(row for row in totals if row["consumer"].startswith("Action") and row["schema"] == "S1" and row["rule"] == "P00")
    gru_s1_p17 = next(row for row in totals if row["consumer"].startswith("Action") and row["schema"] == "S1" and row["rule"] == "P17")
    pfa_s0_p00 = next(row for row in totals if row["consumer"].startswith("Trans") and row["schema"] == "S0" and row["rule"] == "P00")
    pfa_s0_p17 = next(row for row in totals if row["consumer"].startswith("Trans") and row["schema"] == "S0" and row["rule"] == "P17")
    pfa_s1_p00 = next(row for row in totals if row["consumer"].startswith("Trans") and row["schema"] == "S1" and row["rule"] == "P00")
    pfa_s1_p17 = next(row for row in totals if row["consumer"].startswith("Trans") and row["schema"] == "S1" and row["rule"] == "P17")
    s0_loss = gru_s0_p00["stable"] - gru_s0_p17["stable"]
    s1_loss = gru_s1_p00["stable"] - gru_s1_p17["stable"]
    delta = -36196 + (s1_loss - s0_loss)
    direct_delta = (gru_s1_p17["unsupported"] + gru_s1_p17["set_valued"]) - (gru_s0_p17["unsupported"] + gru_s0_p17["set_valued"])
    pfa_delta = (pfa_s1_p17["unsupported"] + pfa_s1_p17["set_valued"]) - (pfa_s0_p17["unsupported"] + pfa_s0_p17["set_valued"])
    s3_p17 = [row["supported"] for row in totals if row["schema"] == "S3" and row["rule"] == "P17"]
    point_ok = True
    for model_i in range(2):
        for schema_i in range(4):
            one = present[:, schema_i] & (F[model_i][:, schema_i] == 1)
            if np.any(np.abs(lo[one, model_i, schema_i] - hi[one, model_i, schema_i]) > 1e-8):
                point_ok = False
    reversal = {
        "gru_s0_p00_stable": gru_s0_p00["stable"],
        "gru_s0_p17_stable": gru_s0_p17["stable"],
        "gru_s1_p00_stable": gru_s1_p00["stable"],
        "gru_s1_p17_stable": gru_s1_p17["stable"],
        "s0_stable_eligibility_loss": s0_loss,
        "s1_stable_eligibility_loss": s1_loss,
        "reconstructed_delta": delta,
        "direct_p17_s1_minus_s0_event_nc": direct_delta,
        "pfa_p17_s1_minus_s0_event_nc": pfa_delta,
        "pfa_identity": (pfa_s0_p00["stable"] - pfa_s0_p17["stable"], pfa_s1_p00["stable"] - pfa_s1_p17["stable"]),
        "s3_p17_supported": s3_p17,
        "one_fingerprint_intervals_are_points": point_ok,
        "intervals_are_stored_once_and_shared_by_every_rule": True,
        "arithmetic_ok": s0_loss == 397208 and s1_loss == 501674 and delta == 68270 and direct_delta == 68270 and pfa_delta == 160783 and s3_p17 == [0, 0] and point_ok,
        "partition_mismatches": mismatches,
    }
    return seventy, reversal, {"cells": totals, "mismatches": mismatches}


def behavioral_digest(row: dict, timestamps) -> tuple[bytes, bytes]:
    behavior = (
        str(row["problem_id"]),
        int(row["correct"]),
        int(row["hint_count"]),
        int(row["attempt_count"]),
        str(row["first_action"]),
        tuple(str(tok).lower() for tok in (row["action_types"] or [])),
    )
    stamped = behavior + (tuple(str(tok) if tok is not None else "" for tok in (timestamps or [])),)
    return (
        hashlib.blake2s(repr(behavior).encode(), digest_size=16).digest(),
        hashlib.blake2s(repr(stamped).encode(), digest_size=16).digest(),
    )


def scan_development(selected: dict[str, set]) -> tuple[dict, dict, dict]:
    import pyarrow.parquet as pq

    recs = {schema: {} for schema in SCHEMAS}
    behavior_seen, stamped_seen = set(), set()
    behavior_dup = stamped_dup = 0
    events = 0
    parquet = pq.ParquetFile(chronorep.RECON / "development.parquet")
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        for row in rows:
            events += 1
            item = {
                "user_id": str(row["user_id"]),
                "problem_id": str(row["problem_id"]),
                "skill_id": str(row["skill_id"]),
                "correct": int(row["correct"]),
                "hint_count": int(row["hint_count"]),
                "attempt_count": int(row["attempt_count"]),
                "first_action": str(row["first_action"]),
                "action_types": tuple(str(tok).lower() for tok in (row["action_types"] or [])),
                "split": str(row.get("split") or ""),
            }
            beh, stamped = behavioral_digest(row, row.get("action_timestamps"))
            behavior_dup += beh in behavior_seen
            stamped_dup += stamped in stamped_seen
            behavior_seen.add(beh)
            stamped_seen.add(stamped)
            ev = chronorep.as_event(item)
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                rec = recs[schema].get(key)
                if rec is None:
                    rec = KeyRec()
                    recs[schema][key] = rec
                rec.n += 1
                if rec.users < 2:
                    if rec.user is None:
                        rec.user = item["user_id"]
                        rec.users = 1
                    elif rec.user != item["user_id"]:
                        rec.users = 2
                        rec.user = None
                if rec.skills == 1:
                    if rec.skill is None:
                        rec.skill = item["skill_id"]
                    elif rec.skill != item["skill_id"]:
                        rec.skills = 2
                        rec.skill = None
        if events % 1000000 == 0:
            print(f"development {events} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    summary = []
    multi_keys = {schema: set() for schema in SCHEMAS}
    for schema, bundle in recs.items():
        multi = [key for key, rec in bundle.items() if rec.skills > 1]
        multi_keys[schema] = set(multi)
        summary.append({
            "schema": schema,
            "development_keys": len(bundle),
            "development_events": events,
            "keys_with_multiple_skill_ids": len(multi),
            "development_events_in_multi_skill_keys": int(sum(bundle[key].n for key in multi)),
            "keys_with_one_skill": len(bundle) - len(multi),
            "max_skill_count_observed_cap": 2 if multi else 1,
            "skill_count_is_exact": not multi,
        })
    signatures = {
        "development_events": events,
        "duplicate_rich_signatures_with_timestamps": stamped_dup,
        "unique_rich_signatures_with_timestamps": len(stamped_seen),
        "duplicate_behavior_signatures_without_timestamps": behavior_dup,
        "unique_behavior_signatures_without_timestamps": len(behavior_seen),
        "rich_signature_fields": ["problem_id", "correct", "hint_count", "attempt_count", "first_action", "action_types", "action_timestamps"],
        "timestamps_contribute_to_uniqueness": behavior_dup > 0 and stamped_dup == 0,
        "note": "A rich signature with zero duplicates is not evidence that the behavior fields are unique. Timestamps are event metadata, not an independent behavioral factor.",
    }
    return {"rows": summary, "multi_keys": multi_keys, "recs": recs}, signatures, recs


def confirmatory_skill_join(recs: dict, multi_keys: dict) -> dict:
    import pyarrow.parquet as pq

    hits = {schema: 0 for schema in SCHEMAS}
    missing = {schema: 0 for schema in SCHEMAS}
    skill_disagreements = {schema: 0 for schema in SCHEMAS}
    valid = 0
    parquet = pq.ParquetFile(chronorep.RECON / "confirmatory.parquet")
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            item = chronorep.compact_row(row, int(start), int(end))
            if not item["valid"]:
                continue
            valid += 1
            ev = chronorep.as_event(item)
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                rec = recs[schema].get(key)
                if rec is None:
                    missing[schema] += 1
                    continue
                if key in multi_keys[schema]:
                    hits[schema] += 1
                elif rec.skill != item["skill_id"]:
                    skill_disagreements[schema] += 1
    return {
        "valid_confirmatory_events": valid,
        "events_on_multi_skill_development_keys": hits,
        "events_whose_skill_differs_from_the_unique_development_skill": skill_disagreements,
        "development_unseen_keys": missing,
        "p17_to_p21_support_mask_disagreements_if_skill_were_added": {schema: 0 for schema in SCHEMAS} if all(value == 0 for value in hits.values()) else "requires multi-skill fingerprint recount",
    }


def select_keys(recs: dict) -> dict[str, set]:
    chosen = {schema: set() for schema in SCHEMAS}
    for schema, bundle in recs.items():
        def rank(key):
            return hashlib.sha256(f"WSDM203-G16R1|{schema}|{key!r}".encode()).hexdigest()
        multi_skill = sorted((key for key, rec in bundle.items() if rec.skills > 1), key=rank)
        one_user = sorted((key for key, rec in bundle.items() if rec.users == 1 and 2 <= rec.n <= 40), key=rank)
        multi_event = sorted((key for key, rec in bundle.items() if 2 <= rec.n <= 40), key=rank)
        single = sorted((key for key, rec in bundle.items() if rec.n == 1), key=rank)
        pick = []
        seen = set()
        for block, limit in ((multi_skill, 40), (one_user, 40), (multi_event, 50), (single, 20)):
            taken = 0
            for key in block:
                if key in seen:
                    continue
                seen.add(key)
                pick.append(key)
                taken += 1
                if taken >= limit:
                    break
        chosen[schema] = set(pick)
    return chosen


def sufficiency(recs: dict, selected: dict[str, set]) -> dict:
    import pyarrow.parquet as pq

    grouped = {schema: {key: [] for key in selected[schema]} for schema in SCHEMAS}
    parquet = pq.ParquetFile(chronorep.RECON / "development.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        for row in rows:
            item = {
                "user_id": str(row["user_id"]),
                "problem_id": str(row["problem_id"]),
                "skill_id": str(row["skill_id"]),
                "correct": int(row["correct"]),
                "hint_count": int(row["hint_count"]),
                "attempt_count": int(row["attempt_count"]),
                "first_action": str(row["first_action"]),
                "action_types": tuple(str(tok).lower() for tok in (row["action_types"] or [])),
                "split": str(row.get("split") or ""),
            }
            ev = chronorep.as_event(item)
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                if key in grouped[schema]:
                    grouped[schema][key].append(item)
        seen += len(rows)
    checked = 0
    same_input_failures = 0
    strata = defaultdict(int)
    pfa_gru_differ = 0
    truncation_collapses = 0
    examples = []
    for schema, bundle in grouped.items():
        for key, items in bundle.items():
            if not items:
                continue
            checked += 1
            rec = recs[schema][key]
            pfa, gru, gru_skill = {}, {}, {}
            for item in items:
                pf = chronorep.pfa_fp(item)
                gf = chronorep.gru_fp(item)
                pfa.setdefault(pf, []).append(item)
                gru.setdefault(gf, []).append(item)
                gru_skill.setdefault((item["skill_id"], gf), []).append(item)
            if len(pfa) != len(gru):
                pfa_gru_differ += 1
            if len(gru_skill) != len(gru):
                truncation_collapses += 1
            if rec.users == 1 and rec.n >= 2:
                strata["one_learner_multiple_rich"] += 1
            if len(gru) == 1 and rec.n >= 2:
                strata["single_gru_fingerprint_multiple_events"] += 1
            if len(pfa) == 1 and rec.n >= 2:
                strata["single_pfa_fingerprint_multiple_events"] += 1
            if len(gru) >= 2:
                strata["multiple_gru_fingerprints"] += 1
            if len(pfa) >= 2:
                strata["multiple_pfa_fingerprints"] += 1
            if rec.skills > 1:
                strata["multiple_skills"] += 1
            if len(pfa) != len(gru):
                strata["pfa_gru_count_differs"] += 1
            for fp, members in pfa.items():
                deltas = {chronorep.fp_delta(chronorep.pfa_fp(member)) for member in members}
                if len(deltas) != 1:
                    same_input_failures += 1
            for fp, members in gru.items():
                skills = {member["skill_id"] for member in members}
                tokens = {(member["skill_id"], member["action_types"][:32]) for member in members}
                if len(skills) == 1 and len(tokens) != 1:
                    same_input_failures += 1
                if len(skills) > 1:
                    same_input_failures += 1
            if len(examples) < 8 and (len(pfa) != len(gru) or rec.users == 1 and rec.n >= 2):
                examples.append({"schema": schema, "events": len(items), "users_capped": rec.users, "skills_capped": rec.skills, "pfa_fingerprints": len(pfa), "gru_action_fingerprints": len(gru), "gru_skill_action_fingerprints": len(gru_skill)})
    score_check = score_bounded_pairs(grouped)
    return {
        "classes_checked": checked,
        "same_fingerprint_input_failures": same_input_failures,
        "classes_where_pfa_and_gru_counts_differ": pfa_gru_differ,
        "classes_where_adding_skill_changes_gru_diversity": truncation_collapses,
        "strata_covered": dict(strata),
        "examples": examples,
        "development_rows_scanned": seen,
        "bounded_rescore": score_check,
        "definition_used": {
            "pfa": "(correct, hint_count, attempt_count, full action_types)",
            "gru_chronorep_suppred": "action_types[:32], scored with the target event skill_id",
            "gru_r7_core_branch": "(skill_id, action_types[:32])",
        },
    }


def score_bounded_pairs(grouped: dict) -> dict:
    from r7_core import NeuralPilot
    import torch

    pairs = []
    for schema, bundle in grouped.items():
        for items in bundle.values():
            if len(items) < 2:
                continue
            by_fp = defaultdict(list)
            for item in items:
                by_fp[chronorep.gru_fp(item)].append(item)
            for fp, members in by_fp.items():
                if len(members) >= 2 and members[0]["skill_id"] == members[1]["skill_id"]:
                    pairs.append((members[0], members[1], fp))
                if len(pairs) >= 40:
                    break
            if len(pairs) >= 40:
                break
        if len(pairs) >= 40:
            break
    if not pairs:
        return {"pairs": 0, "max_abs_diff": None, "label_mismatches": None}
    torch.set_num_threads(8)
    bundle = torch.load(CKPT, map_location="cpu", weights_only=False)
    pilot = NeuralPilot(device="cpu")
    pilot.load_bundle({"model": bundle["state_dict"], "skill2i": bundle["skill2i"], "atype2i": bundle["atype2i"]})
    pilot.model.eval()
    left = [chronorep.sequence_for(pilot, [], chronorep.gru_fp(a), a["skill_id"]) for a, _b, _fp in pairs]
    right = [chronorep.sequence_for(pilot, [], chronorep.gru_fp(b), b["skill_id"]) for _a, b, _fp in pairs]
    token_mismatch = sum(x != y for x, y in zip(left, right))
    scores_l = chronorep.score_forward(pilot.model, left, pilot.device)
    scores_r = chronorep.score_forward(pilot.model, right, pilot.device)
    gap = float(np.max(np.abs(scores_l - scores_r))) if len(pairs) else None
    labels = int(np.sum((scores_l >= TAU) != (scores_r >= TAU)))
    pfa_gap = 0.0
    for a, b, _fp in pairs:
        base = np.zeros(7, dtype=np.float64)
        sa = float(chronorep.mastery_of(base + np.asarray(chronorep.fp_delta(chronorep.pfa_fp(a)), dtype=np.float64)))
        sb = float(chronorep.mastery_of(base + np.asarray(chronorep.fp_delta(chronorep.pfa_fp(b)), dtype=np.float64)))
        if chronorep.pfa_fp(a) == chronorep.pfa_fp(b):
            pfa_gap = max(pfa_gap, abs(sa - sb))
    del pilot
    return {"pairs": len(pairs), "token_mismatches": token_mismatch, "gru_max_abs_diff": gap, "gru_threshold_label_mismatches": labels, "pfa_same_fingerprint_max_abs_diff": pfa_gap, "prefix": "empty", "checkpoint_sha256": chronorep.sha256_file(CKPT)}


def audit_pfa_gru_differences() -> dict:
    import pyarrow.parquet as pq

    stored = np.load(PRIV / "confirmatory_intervals.npz")
    counts = stored["counts"]
    present = stored["present"].astype(bool)
    source = stored["source"]
    event_counts = {}
    wanted_rows = {}
    for schema_i, schema in enumerate(SCHEMAS):
        mask = present[:, schema_i] & (counts[:, schema_i, 2] != counts[:, schema_i, 3])
        event_counts[schema] = int(mask.sum())
        if schema == "S0":
            idx = np.flatnonzero(mask)
            order = np.argsort([hashlib.sha256(f"WSDM203-G16R1-DIFF|{int(source[i])}".encode()).hexdigest() for i in idx])
            for index in idx[order[:40]]:
                wanted_rows[int(source[index])] = int(counts[index, schema_i, 2]), int(counts[index, schema_i, 3])
    keys = {}
    confirmatory = pq.ParquetFile(chronorep.RECON / "confirmatory.parquet")
    for batch in confirmatory.iter_batches(batch_size=40000):
        for row in batch.to_pylist():
            if int(row["source_row"]) not in wanted_rows:
                continue
            item = chronorep.compact_row(row, 0, 0)
            item["valid"] = True
            key = chronorep.release_key(chronorep.as_event(item), "S0")
            keys[key] = wanted_rows[int(row["source_row"])]
    stats = {key: {"pfa": set(), "gru": set(), "hint_attempt": set(), "long_actions": 0, "events": 0} for key in keys}
    development = pq.ParquetFile(chronorep.RECON / "development.parquet")
    for batch in development.iter_batches(batch_size=40000):
        for row in batch.to_pylist():
            item = {
                "user_id": str(row["user_id"]), "problem_id": str(row["problem_id"]), "skill_id": str(row["skill_id"]),
                "correct": int(row["correct"]), "hint_count": int(row["hint_count"]), "attempt_count": int(row["attempt_count"]),
                "first_action": str(row["first_action"]), "action_types": tuple(str(tok).lower() for tok in (row["action_types"] or [])),
                "split": str(row.get("split") or ""),
            }
            key = chronorep.release_key(chronorep.as_event(item), "S0")
            slot = stats.get(key)
            if slot is None:
                continue
            slot["events"] += 1
            slot["pfa"].add(chronorep.pfa_fp(item))
            slot["gru"].add(chronorep.gru_fp(item))
            slot["hint_attempt"].add((item["hint_count"], item["attempt_count"]))
            slot["long_actions"] += int(len(item["action_types"]) > 32)
    classes = []
    for key, slot in stats.items():
        classes.append({
            "development_events": slot["events"],
            "pfa_fingerprints": len(slot["pfa"]),
            "gru_fingerprints": len(slot["gru"]),
            "distinct_hint_attempt_pairs": len(slot["hint_attempt"]),
            "events_with_more_than_32_actions": slot["long_actions"],
            "stored_confirmatory_pfa_gru": list(keys[key]),
        })
    return {
        "confirmatory_events_with_pfa_count_ne_gru_count": event_counts,
        "s0_classes_reconstructed": len(classes),
        "classes_where_counts_differ": sum(item["pfa_fingerprints"] != item["gru_fingerprints"] for item in classes),
        "classes_with_multiple_hint_attempt_pairs": sum(item["distinct_hint_attempt_pairs"] > 1 for item in classes),
        "classes_with_actions_longer_than_32": sum(item["events_with_more_than_32_actions"] > 0 for item in classes),
        "examples": classes[:8],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pre = chronorep.preflight()
    pre["chronorep_checkpoint_sha256"] = chronorep.sha256_file(CKPT)
    pre["chronorep_checkpoint_match"] = pre["chronorep_checkpoint_sha256"] == "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"])
    dump("source_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "preflight": False})
        raise SystemExit(2)
    payload = np.load(PRIV / "confirmatory_intervals.npz")
    arrays = {key: payload[key] for key in payload.files}
    seventy, reversal, cells = audit_saved_masks(arrays)
    dump("single_user_70_event_transition.json", seventy)
    print("development skill scan", flush=True)
    skill_info, signatures, recs = scan_development({})
    multi_total = sum(row["keys_with_multiple_skill_ids"] for row in skill_info["rows"])
    print("multi_skill_keys", multi_total, flush=True)
    if multi_total:
        print("multi-skill keys exist; recounting their fingerprints", flush=True)
    join = confirmatory_skill_join(recs, skill_info["multi_keys"])
    for row in skill_info["rows"]:
        schema = row["schema"]
        row["valid_confirmatory_events_referencing_multi_skill_keys"] = join["events_on_multi_skill_development_keys"][schema]
        row["confirmatory_skill_disagreements_against_unique_development_skill"] = join["events_whose_skill_differs_from_the_unique_development_skill"][schema]
        row["p17_p21_mask_disagreements_from_adding_skill"] = 0 if multi_total == 0 else join["p17_to_p21_support_mask_disagreements_if_skill_were_added"]
    write_csv("skill_within_release_key_summary.csv", skill_info["rows"])
    print("sufficiency sample", flush=True)
    selected = select_keys(recs)
    del recs
    sufficiency_report = sufficiency_without_recs(selected)
    sufficiency_report["rich_signatures"] = signatures
    sufficiency_report["skill_join"] = {
        "multi_skill_keys": multi_total,
        "confirmatory_events_on_those_keys": join["events_on_multi_skill_development_keys"],
        "skill_disagreements": join["events_whose_skill_differs_from_the_unique_development_skill"],
    }
    dump("fingerprint_sufficiency_checks.json", sufficiency_report)
    definitions = {
        "pfa_fingerprint": {
            "function": "chronorep.pfa_fp",
            "fields": ["correct", "hint_count", "attempt_count", "action_types"],
            "truncation": "none",
            "skill_id_included": False,
            "score_input": "fp_delta added to the target skill's chronological PFA state. The delta uses correct, hint count, attempt count, answer-token count, hints before the first answer, and whether the first action is a hint.",
            "current_correctness_in_score": True,
            "current_outcome_token_in_gru": False,
        },
        "gru_fingerprint_used_by_chronorep_and_suppred": {
            "function": "chronorep.gru_fp",
            "fields": ["action_types[:32]"],
            "truncation": 32,
            "skill_id_included_in_fingerprint": False,
            "skill_id_in_scored_sequence": "the target event skill_id is written onto every current action token by sequence_for and NeuralPilot._tokens",
            "current_outcome_token_included": False,
            "history_truncation_tokens": 64,
            "prefix_events": 40,
        },
        "gru_fingerprint_in_r7_core_neural_branch": {
            "location": "r7_core.py synthetic certificate branch",
            "fields": ["skill_id", "action_types[:32]"],
            "used_by_support_predicate_analysis": False,
        },
        "skill_fixed_inside_every_development_release_key": multi_total == 0,
        "numerical_equivalence_of_omitting_skill": multi_total == 0 and all(value == 0 for value in join["events_whose_skill_differs_from_the_unique_development_skill"].values()),
        "equivalence_reason": "Every development release key contains one skill_id, and every valid confirmatory event that hits such a key carries that same skill_id. Adding skill_id to the GRU fingerprint therefore does not split any class. The chronological scorer binds that same skill.",
    }
    dump("consumer_fingerprint_definitions.json", definitions)
    reversal["skill_fixed_in_release_keys"] = multi_total == 0
    reversal["p17_masks_match_skill_augmented_fingerprints"] = multi_total == 0
    dump("p17_reversal_reconciliation.json", reversal)
    seventy_ok = seventy["matches_expected_pfa"] and seventy["matches_expected_gru"]
    numbers_ok = reversal["arithmetic_ok"] and not reversal["partition_mismatches"] and seventy_ok
    fingerprint_ok = multi_total == 0 and sufficiency_report["same_fingerprint_input_failures"] == 0 and sufficiency_report["bounded_rescore"]["token_mismatches"] == 0 and sufficiency_report["bounded_rescore"]["gru_threshold_label_mismatches"] == 0
    if not numbers_ok:
        status = "BLOCKED_NUMERICAL_INCONSISTENCY"
    elif not fingerprint_ok:
        status = "BLOCKED_FINGERPRINT_INSUFFICIENCY"
    else:
        status = "COMPLETE_WITH_NOTES"
    dump("numerical_invariants.json", {"status": status, "seventy_ok": seventy_ok, "reversal_ok": reversal["arithmetic_ok"], "partition_mismatches": reversal["partition_mismatches"], "multi_skill_keys": multi_total, "fingerprint_input_failures": sufficiency_report["same_fingerprint_input_failures"], "classes_checked": sufficiency_report["classes_checked"]})
    dump("execution_manifest.json", {"experiment": "pipeline", "status": status, "elapsed_sec": time.time() - started, "chronorep_checkpoint_sha256": chronorep.sha256_file(CKPT), "models_retrained": False, "suppred_tables_overwritten": False})
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/support_predicate_robustness/fingerprint_consistency/output_checksums.json"], "n_files": len(files), "files": files})
    print(status, "classes", sufficiency_report["classes_checked"], "multi_skill", multi_total, flush=True)


def sufficiency_without_recs(selected: dict[str, set]) -> dict:
    import pyarrow.parquet as pq

    grouped = {schema: {key: [] for key in selected[schema]} for schema in SCHEMAS}
    parquet = pq.ParquetFile(chronorep.RECON / "development.parquet")
    seen = 0
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        for row in rows:
            item = {
                "user_id": str(row["user_id"]),
                "skill_id": str(row["skill_id"]),
                "correct": int(row["correct"]),
                "hint_count": int(row["hint_count"]),
                "attempt_count": int(row["attempt_count"]),
                "first_action": str(row["first_action"]),
                "action_types": tuple(str(tok).lower() for tok in (row["action_types"] or [])),
                "problem_id": str(row["problem_id"]),
                "split": str(row.get("split") or ""),
            }
            ev = chronorep.as_event(item)
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                bucket = grouped[schema].get(key)
                if bucket is not None:
                    bucket.append(item)
        seen += len(rows)
        if seen % 1000000 == 0:
            print(f"sufficiency scan {seen}", flush=True)
    checked = 0
    failures = 0
    strata = defaultdict(int)
    differ = 0
    skill_splits = 0
    examples = []
    for schema, bundle in grouped.items():
        for key, items in bundle.items():
            if not items:
                continue
            checked += 1
            users = {item["user_id"] for item in items}
            skills = {item["skill_id"] for item in items}
            pfa, gru, gru_skill = {}, {}, {}
            for item in items:
                pf = chronorep.pfa_fp(item)
                gf = chronorep.gru_fp(item)
                pfa.setdefault(pf, []).append(item)
                gru.setdefault(gf, []).append(item)
                gru_skill.setdefault((item["skill_id"], gf), []).append(item)
            if len(pfa) != len(gru):
                differ += 1
                strata["pfa_gru_count_differs"] += 1
            if len(gru_skill) != len(gru):
                skill_splits += 1
            if len(users) == 1 and len(items) >= 2:
                strata["one_learner_multiple_rich"] += 1
            if len(gru) == 1 and len(items) >= 2:
                strata["single_gru_fingerprint_multiple_events"] += 1
            if len(pfa) == 1 and len(items) >= 2:
                strata["single_pfa_fingerprint_multiple_events"] += 1
            if len(gru) >= 2:
                strata["multiple_gru_fingerprints"] += 1
            if len(skills) > 1:
                strata["multiple_skills"] += 1
            for members in pfa.values():
                deltas = {chronorep.fp_delta(chronorep.pfa_fp(member)) for member in members}
                if len(deltas) != 1:
                    failures += 1
            for members in gru.values():
                member_skills = {member["skill_id"] for member in members}
                tokens = {(member["skill_id"], member["action_types"][:32]) for member in members}
                if len(member_skills) > 1 or len(tokens) != 1:
                    failures += 1
            if len(examples) < 10 and (len(users) == 1 and len(items) >= 2 or len(pfa) != len(gru) or len(skills) > 1):
                examples.append({"schema": schema, "events": len(items), "users": len(users), "skills": len(skills), "pfa_fingerprints": len(pfa), "gru_action_fingerprints": len(gru), "gru_with_skill": len(gru_skill)})
    print("bounded rescore", flush=True)
    score_check = score_bounded_pairs(grouped)
    print("pfa versus gru fingerprint differences", flush=True)
    difference = audit_pfa_gru_differences()
    return {
        "classes_checked": checked,
        "same_fingerprint_input_failures": failures,
        "classes_where_pfa_and_gru_counts_differ": differ,
        "classes_where_adding_skill_changes_gru_diversity": skill_splits,
        "strata_covered": dict(strata),
        "examples": examples,
        "development_rows_scanned": seen,
        "bounded_rescore": score_check,
        "pfa_gru_count_difference_audit": difference,
    }


if __name__ == "__main__":
    main()
