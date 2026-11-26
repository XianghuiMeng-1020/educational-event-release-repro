#!/usr/bin/env python3
"""ASSISTments common-cohort capacity certification.

Recomputes membership from stored risk intervals. No training and no GRU or PFA
inference are involved, and frozen results are never rewritten.

Primary ranking is the original adversarial-tie rule:
  best = 1 + count(other: lower > own upper)
  worst = 1 + count(other: upper >= own lower)
  guaranteed-in iff worst <= B; guaranteed-out iff best > B.
B = max(1, round(kappa * n)) with Python round-half-to-even, for n > 0.

Ascending user-id ties are a separate sensitivity. On an exact endpoint tie,
the lexicographically smaller string id ranks ahead.
"""
from __future__ import annotations

import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "assistments" / "capacity_certification"
AGG_PATH = ROOT / "results" / "wdlf_r1_rescue" / "assist_heldout_user_aggregates.json"
P2_PATH = ROOT / "results" / "wdlf_r7" / "p2_by_schema_model.csv"
BUNDLE_PATH = ROOT / "prospective" / "wdlf_r7" / "sealed" / "confirmatory_sealed_bundle.json"
VALIDITY_PATH = ROOT / "results" / "wdlf_r1_rescue" / "assist_heldout_validity.json"
FRONTIER_PATH = ROOT / "results" / "wdlf_r7" / "primary_frontier.json"

MODELS = ("TransparentActionPFA", "ActionTokenGRU")
BUNDLE_MODEL = {"TransparentActionPFA": "transparent", "ActionTokenGRU": "neural"}
SCHEMAS = ("S0", "S1", "S2", "S3")
KAPPAS = (0.01, 0.05, 0.1)
PRIMARY_KAPPA = 0.05
N_FULL_EXPECTED = 6766
SEED = 20304
N_BOOT = 2000
LABELS = ("guaranteed_in", "guaranteed_out", "unresolved")
NATIVE_005 = {
    ("TransparentActionPFA", "S0"): (6754, 338, 173, 6229, 352),
    ("TransparentActionPFA", "S1"): (6749, 337, 334, 6409, 6),
    ("TransparentActionPFA", "S2"): (6746, 337, 337, 6409, 0),
    ("TransparentActionPFA", "S3"): (6744, 337, 336, 6405, 3),
    ("ActionTokenGRU", "S0"): (6754, 338, 38, 5161, 1555),
    ("ActionTokenGRU", "S1"): (6749, 337, 250, 6260, 239),
    ("ActionTokenGRU", "S2"): (6746, 337, 264, 6305, 177),
    ("ActionTokenGRU", "S3"): (6744, 337, 337, 6407, 0),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def budget(n: int, kappa: float) -> int:
    if n <= 0:
        return 0
    b = int(round(float(kappa) * int(n)))
    if kappa > 0 and b < 1:
        b = 1
    return min(b, n)


def adversarial_ranks(los: np.ndarray, his: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact adversarial ranks. Self is excluded. No tolerance is added."""
    n = int(len(los))
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    order_lo = np.sort(los)
    order_hi = np.sort(his)
    n_lo_gt = n - np.searchsorted(order_lo, his, side="right")
    self_best = (los > his).astype(np.int32)
    best = (1 + n_lo_gt - self_best).astype(np.int32)
    n_hi_ge = n - np.searchsorted(order_hi, los, side="left")
    self_worst = (his >= los).astype(np.int32)
    worst = np.where(self_worst == 1, n_hi_ge, 1 + n_hi_ge).astype(np.int32)
    return best, worst


def _tie_counts(endpoints: np.ndarray, query: np.ndarray, ids: np.ndarray) -> np.ndarray:
    groups: dict[float, list[str]] = defaultdict(list)
    for value, uid in zip(endpoints.tolist(), ids.tolist(), strict=True):
        groups[float(value)].append(str(uid))
    for rows in groups.values():
        rows.sort()
    out = np.zeros(len(query), dtype=np.int32)
    for i, value in enumerate(query.tolist()):
        rows = groups.get(float(value))
        if not rows:
            continue
        uid = str(ids[i])
        lo, hi = 0, len(rows)
        while lo < hi:
            mid = (lo + hi) // 2
            if rows[mid] < uid:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo
    return out


def userid_ranks(ids: np.ndarray, los: np.ndarray, his: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ascending string-id tie break. Equal ids do not rank ahead of each other."""
    n = int(len(ids))
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    order_lo = np.sort(los)
    order_hi = np.sort(his)
    n_lo_gt = n - np.searchsorted(order_lo, his, side="right")
    self_best = (los > his).astype(np.int32)
    best_ties = _tie_counts(los, his, ids)
    best = (1 + n_lo_gt - self_best + best_ties).astype(np.int32)
    n_hi_gt = n - np.searchsorted(order_hi, los, side="right")
    self_strict = (his > los).astype(np.int32)
    worst_ties = _tie_counts(his, los, ids)
    worst = (1 + n_hi_gt - self_strict + worst_ties).astype(np.int32)
    return best, worst


def labels_from_ranks(best: np.ndarray, worst: np.ndarray, b_seats: int) -> np.ndarray:
    out = np.full(len(best), "unresolved", dtype=object)
    if b_seats <= 0 or len(best) == 0:
        return out
    out[worst <= b_seats] = "guaranteed_in"
    out[(best > b_seats) & (worst > b_seats)] = "guaranteed_out"
    return out


def loop_adversarial(ids: list[str], los: list[float], his: list[float], b_seats: int) -> list[str]:
    labels = []
    for i, _uid in enumerate(ids):
        best = 1 + sum(1 for j in range(len(ids)) if j != i and los[j] > his[i])
        worst = 1 + sum(1 for j in range(len(ids)) if j != i and his[j] >= los[i])
        if b_seats > 0 and worst <= b_seats:
            labels.append("guaranteed_in")
        elif b_seats > 0 and best > b_seats:
            labels.append("guaranteed_out")
        else:
            labels.append("unresolved")
    return labels


def loop_userid(ids: list[str], los: list[float], his: list[float], b_seats: int) -> list[str]:
    labels = []
    for i, uid in enumerate(ids):
        best = 1
        worst = 1
        for j, other in enumerate(ids):
            if j == i:
                continue
            if los[j] > his[i] or (los[j] == his[i] and other < uid):
                best += 1
            if his[j] > los[i] or (his[j] == los[i] and other < uid):
                worst += 1
        if b_seats > 0 and worst <= b_seats:
            labels.append("guaranteed_in")
        elif b_seats > 0 and best > b_seats:
            labels.append("guaranteed_out")
        else:
            labels.append("unresolved")
    return labels


def counts(labels: np.ndarray) -> dict[str, int]:
    return {name: int(np.sum(labels == name)) for name in LABELS}


def assert_rank_edges() -> None:
    cases: list[tuple[str, list[str], list[float], list[float], int]] = [
        ("equal_points", ["1", "2", "3", "4"], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], 2),
        ("overlap", ["a", "b"], [0.0, 0.5], [1.0, 0.6], 1),
        ("disjoint", ["a", "b"], [0.9, 0.0], [1.0, 0.1], 1),
        ("rank_equals_B", ["a", "b", "c"], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1], 2),
        ("boundary_interval", ["a", "b", "c"], [0.2, 0.5, 0.8], [0.7, 0.55, 0.9], 1),
        ("id_resolves_adversarial_unresolved", ["b", "a"], [0.4, 0.4], [0.4, 0.4], 1),
        ("zero_width", ["m", "z"], [0.2, 0.8], [0.2, 0.8], 1),
        ("shared_endpoints", ["c", "a", "b"], [0.5, 0.5, 0.2], [0.5, 0.5, 0.2], 1),
        ("duplicate_positions", ["u", "u", "v"], [0.2, 0.2, 0.9], [0.2, 0.2, 0.9], 1),
    ]
    for name, ids, los, his, b_seats in cases:
        arr_id = np.array(ids, dtype=object)
        arr_lo = np.array(los, dtype=np.float64)
        arr_hi = np.array(his, dtype=np.float64)
        best, worst = adversarial_ranks(arr_lo, arr_hi)
        got = labels_from_ranks(best, worst, b_seats).tolist()
        exp = loop_adversarial(ids, los, his, b_seats)
        if got != exp:
            raise SystemExit(f"adversarial edge mismatch {name}: {got} != {exp}")
        if np.any((best > worst)):
            raise SystemExit(f"best exceeds worst in {name}")
        b2, w2 = userid_ranks(arr_id, arr_lo, arr_hi)
        got_id = labels_from_ranks(b2, w2, b_seats).tolist()
        exp_id = loop_userid(ids, los, his, b_seats)
        if got_id != exp_id:
            raise SystemExit(f"userid edge mismatch {name}: {got_id} != {exp_id}")
        # Adversarial positive certificates remain positive under user-id order.
        for adv, uid_lab in zip(got, got_id, strict=True):
            if adv == "guaranteed_in" and uid_lab != "guaranteed_in":
                raise SystemExit(f"containment gin broken in {name}")
            if adv == "guaranteed_out" and uid_lab != "guaranteed_out":
                raise SystemExit(f"containment gout broken in {name}")
    # The two-user equal-risk case is unresolved adversarially and resolved by id.
    ids = ["b", "a"]
    los = [0.4, 0.4]
    his = [0.4, 0.4]
    adv = labels_from_ranks(*adversarial_ranks(np.array(los), np.array(his)), 1).tolist()
    uid = labels_from_ranks(*userid_ranks(np.array(ids, dtype=object), np.array(los), np.array(his)), 1).tolist()
    if adv != ["unresolved", "unresolved"] or uid != ["guaranteed_out", "guaranteed_in"]:
        raise SystemExit(f"expected id resolution, got adv={adv} uid={uid}")
    if budget(0, 0.05) != 0 or budget(6754, 0.05) != 338 or budget(6749, 0.05) != 337:
        raise SystemExit("budget arithmetic mismatch")


def load_records() -> tuple[dict[str, Any], dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    blob = json.loads(AGG_PATH.read_text(encoding="utf-8"))
    if not isinstance(blob, dict):
        raise SystemExit("aggregate root is not an object")
    models = {}
    meta_keys = []
    scalars: dict[str, Any] = {}
    for key, value in blob.items():
        if key in MODELS and isinstance(value, dict):
            models[key] = value
        else:
            meta_keys.append(key)
            if isinstance(value, (str, int, float, bool)) or value is None or isinstance(value, list):
                scalars[key] = value
    missing = [m for m in MODELS if m not in models]
    if missing:
        raise SystemExit(f"missing models {missing}")
    return {"root_keys": sorted(blob), "non_model_keys": meta_keys, "scalars": scalars}, models


def align(models: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> dict[str, Any]:
    user_sets = {model: set(rows) for model, rows in models.items()}
    all_users = sorted(set.union(*user_sets.values()))
    same_ids = user_sets["TransparentActionPFA"] == user_sets["ActionTokenGRU"]
    index = {uid: i for i, uid in enumerate(all_users)}
    n = len(all_users)
    pack: dict[str, Any] = {
        "user_ids": np.array(all_users, dtype=object),
        "n_users": n,
        "same_user_ids_across_consumers": same_ids,
        "per_model_user_counts": {m: len(s) for m, s in user_sets.items()},
        "record_keys": {},
        "invalid_intervals": [],
        "missing_schema_rows": [],
        "nonfinite": [],
    }
    for model, rows in models.items():
        sample_uid = next(iter(rows))
        sample_schema = next(iter(rows[sample_uid]))
        pack["record_keys"][model] = sorted(rows[sample_uid][sample_schema])
        for schema in SCHEMAS:
            eligible = np.zeros(n, dtype=bool)
            los = np.full(n, np.nan)
            his = np.full(n, np.nan)
            actual = np.full(n, np.nan)
            actual_present = np.zeros(n, dtype=bool)
            stored = np.full(n, "", dtype=object)
            present = np.zeros(n, dtype=bool)
            for uid, by_schema in rows.items():
                rec = by_schema.get(schema)
                if rec is None:
                    pack["missing_schema_rows"].append({"model": model, "schema": schema})
                    continue
                i = index[uid]
                present[i] = True
                eligible[i] = bool(rec.get("p2_eligible"))
                stored[i] = "" if rec.get("p2_status") is None else str(rec.get("p2_status"))
                for dest, flag, field in (
                    (los, "lo", "p2_risk_min"),
                    (his, "hi", "p2_risk_max"),
                    (actual, "actual", "p2_actual_risk"),
                ):
                    value = rec.get(field)
                    if value is None:
                        continue
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        pack["nonfinite"].append({"model": model, "schema": schema, "field": field})
                        continue
                    if not np.isfinite(number):
                        pack["nonfinite"].append({"model": model, "schema": schema, "field": field})
                        continue
                    dest[i] = number
                    if field == "p2_actual_risk":
                        actual_present[i] = True
                if eligible[i]:
                    if not (np.isfinite(los[i]) and np.isfinite(his[i])):
                        pack["invalid_intervals"].append({"model": model, "schema": schema, "reason": "eligible_missing_endpoint"})
                    elif los[i] > his[i]:
                        pack["invalid_intervals"].append({"model": model, "schema": schema, "reason": "lower_gt_upper"})
            pack[(model, schema)] = {
                "present": present,
                "eligible": eligible,
                "lo": los,
                "hi": his,
                "actual": actual,
                "actual_present": actual_present,
                "stored_status": stored,
            }
    return pack


def subset_labels(ids: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray, kappa: float, rule: str) -> tuple[np.ndarray, int]:
    sel = np.flatnonzero(mask)
    b_seats = budget(int(len(sel)), kappa)
    if rule == "adversarial":
        best, worst = adversarial_ranks(lo[sel], hi[sel])
    elif rule == "ascending_user_id":
        best, worst = userid_ranks(ids[sel], lo[sel], hi[sel])
    else:
        raise ValueError(rule)
    return labels_from_ranks(best, worst, b_seats), b_seats


def realized_top(ids: np.ndarray, actual: np.ndarray, present: np.ndarray, b_seats: int) -> np.ndarray:
    sel = np.flatnonzero(present)
    if len(sel) == 0 or b_seats <= 0:
        return np.zeros(len(ids), dtype=bool)
    order = np.lexsort((ids[sel].astype(str), -actual[sel]))
    chosen = sel[order[:b_seats]]
    out = np.zeros(len(ids), dtype=bool)
    out[chosen] = True
    return out


def validate_against_loop(pack: dict[str, Any]) -> dict[str, Any]:
    cell = pack[("TransparentActionPFA", "S0")]
    ids = pack["user_ids"]
    mask = cell["eligible"]
    sel = np.flatnonzero(mask)[:40]
    id_list = [str(x) for x in ids[sel]]
    lo = [float(x) for x in cell["lo"][sel]]
    hi = [float(x) for x in cell["hi"][sel]]
    b_seats = budget(len(sel), PRIMARY_KAPPA)
    adv = labels_from_ranks(*adversarial_ranks(cell["lo"][sel], cell["hi"][sel]), b_seats).tolist()
    uid = labels_from_ranks(*userid_ranks(ids[sel], cell["lo"][sel], cell["hi"][sel]), b_seats).tolist()
    ok_adv = adv == loop_adversarial(id_list, lo, hi, b_seats)
    ok_uid = uid == loop_userid(id_list, lo, hi, b_seats)
    if not ok_adv or not ok_uid:
        raise SystemExit(f"real-sample rank mismatch adv={ok_adv} uid={ok_uid}")
    return {"sample": "first 40 eligible TransparentActionPFA S0 users in string-id order", "n": 40, "B": b_seats, "adversarial_matches_pairwise": ok_adv, "userid_matches_pairwise": ok_uid}


def native_table(pack: dict[str, Any], expected_extra: dict[tuple, tuple]) -> tuple[list[dict[str, Any]], bool]:
    rows = []
    ok = True
    for model in MODELS:
        for schema in SCHEMAS:
            cell = pack[(model, schema)]
            mask = cell["eligible"]
            n_el = int(mask.sum())
            for kappa in KAPPAS:
                labels, b_seats = subset_labels(pack["user_ids"], cell["lo"], cell["hi"], mask, kappa, "adversarial")
                c = counts(labels)
                if c["guaranteed_in"] + c["guaranteed_out"] + c["unresolved"] != n_el:
                    ok = False
                key = (model, schema, kappa)
                expected = NATIVE_005.get((model, schema)) if kappa == PRIMARY_KAPPA else expected_extra.get(key)
                observed = (n_el, b_seats, c["guaranteed_in"], c["guaranteed_out"], c["unresolved"])
                match = None if expected is None else observed == tuple(expected)
                if kappa == PRIMARY_KAPPA:
                    ok = ok and match is True
                no_ckpt = N_FULL_EXPECTED - n_el
                rows.append(
                    {
                        "model": model,
                        "schema": schema,
                        "kappa": kappa,
                        "tie_rule": "adversarial",
                        "n_full": N_FULL_EXPECTED,
                        "n_eligible": n_el,
                        "n_no_checkpoint": no_ckpt,
                        "B": b_seats,
                        "guaranteed_in": c["guaranteed_in"],
                        "guaranteed_out": c["guaranteed_out"],
                        "unresolved": c["unresolved"],
                        "nc_all_numerator": no_ckpt + c["unresolved"],
                        "nc_all": (no_ckpt + c["unresolved"]) / N_FULL_EXPECTED,
                        "expected_eligible": None if expected is None else int(expected[0]),
                        "expected_B": None if expected is None else int(expected[1]),
                        "expected_guaranteed_in": None if expected is None else int(expected[2]),
                        "expected_guaranteed_out": None if expected is None else int(expected[3]),
                        "expected_unresolved": None if expected is None else int(expected[4]),
                        "match": match,
                    }
                )
    primary_ok = ok and all(r["match"] is True for r in rows if r["kappa"] == PRIMARY_KAPPA)
    return rows, primary_ok


def load_expected_sensitivity() -> dict[tuple, tuple]:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    out = {}
    kappa_block = bundle.get("sensitivity", {}).get("kappa", {})
    for key, models in kappa_block.items():
        kappa = float(key)
        for bundle_name, schemas in models.items():
            model = {v: k for k, v in BUNDLE_MODEL.items()}[bundle_name]
            for schema, rec in schemas.items():
                out[(model, schema, kappa)] = (
                    int(rec["p2_eligible_users"]),
                    int(rec["p2_seats_B"]),
                    int(rec["p2_guaranteed_in"]),
                    int(rec["p2_guaranteed_out"]),
                    int(rec["p2_unresolved"]),
                )
    return out


def cohort_definition(pack: dict[str, Any]) -> dict[str, Any]:
    ids = pack["user_ids"]
    per_schema = {}
    masks = {}
    for schema in SCHEMAS:
        both = np.ones(pack["n_users"], dtype=bool)
        for model in MODELS:
            both &= pack[(model, schema)]["eligible"]
            per_schema[f"{model}_{schema}"] = int(pack[(model, schema)]["eligible"].sum())
        masks[schema] = both
    common = np.ones(pack["n_users"], dtype=bool)
    for schema in SCHEMAS:
        common &= masks[schema]
    # Eligibility identical across consumers?
    eligibility_same = {}
    for schema in SCHEMAS:
        eligibility_same[schema] = bool(np.array_equal(pack[("TransparentActionPFA", schema)]["eligible"], pack[("ActionTokenGRU", schema)]["eligible"]))
    nested = {}
    for model in MODELS:
        flags = [pack[(model, schema)]["eligible"] for schema in SCHEMAS]
        nested[model] = {
            "S3_subset_S2": bool(np.all(~flags[3] | flags[2])),
            "S2_subset_S1": bool(np.all(~flags[2] | flags[1])),
            "S1_subset_S0": bool(np.all(~flags[1] | flags[0])),
        }
    pairwise = masks["S0"] & masks["S1"]
    invalid = pack["invalid_intervals"]
    finite_common = True
    for model in MODELS:
        for schema in SCHEMAS:
            cell = pack[(model, schema)]
            if not np.all(np.isfinite(cell["lo"][common]) & np.isfinite(cell["hi"][common]) & (cell["lo"][common] <= cell["hi"][common])):
                finite_common = False
    return {
        "n_full_observed": pack["n_users"],
        "n_full_declared": N_FULL_EXPECTED,
        "same_user_ids_across_consumers": pack["same_user_ids_across_consumers"],
        "eligibility_identical_across_consumers": eligibility_same,
        "eligible_counts": per_schema,
        "schema_intersection_counts": {schema: int(masks[schema].sum()) for schema in SCHEMAS},
        "common_n": int(common.sum()),
        "excluded_from_common": int(pack["n_users"] - int(common.sum())),
        "pairwise_s0_s1_n": int(pairwise.sum()),
        "nested_eligible_sets": nested,
        "invalid_interval_count": len(invalid),
        "invalid_interval_reasons": {str(k): int(v) for k, v in pd.Series([row["reason"] for row in invalid]).value_counts().items()} if invalid else {},
        "nonfinite_count": len(pack["nonfinite"]),
        "missing_schema_row_count": len(pack["missing_schema_rows"]),
        "common_intervals_finite_and_ordered": finite_common,
        "budgets": {str(kappa): budget(int(common.sum()), kappa) for kappa in KAPPAS},
        "pairwise_budgets": {str(kappa): budget(int(pairwise.sum()), kappa) for kappa in KAPPAS},
        "common_mask": common,
        "pairwise_mask": pairwise,
        "schema_masks": masks,
    }


def capacity_rows(pack: dict[str, Any], mask: np.ndarray, cohort: str, schemas: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    n_common = int(mask.sum())
    ids = pack["user_ids"]
    for kappa in KAPPAS:
        b_seats = budget(n_common, kappa)
        for model in MODELS:
            stored = {}
            for schema in schemas:
                cell = pack[(model, schema)]
                labels, b_check = subset_labels(ids, cell["lo"], cell["hi"], mask, kappa, "adversarial")
                if b_check != b_seats:
                    raise SystemExit("budget drifted inside a common cohort")
                c = counts(labels)
                selected = realized_top(ids, cell["actual"], mask & cell["actual_present"], b_seats)
                actual = cell["actual"]
                lo = cell["lo"]
                hi = cell["hi"]
                inside = mask & cell["actual_present"] & (actual >= lo) & (actual <= hi)
                outside = mask & cell["actual_present"] & ~((actual >= lo) & (actual <= hi))
                inside_atol = mask & cell["actual_present"] & (actual >= lo - 1e-12) & (actual <= hi + 1e-12)
                missing_actual = int((mask & ~cell["actual_present"]).sum())
                gin = (labels == "guaranteed_in")
                gout = (labels == "guaranteed_out")
                sel_idx = np.flatnonzero(mask)
                selected_sub = selected[sel_idx]
                sub_outside = outside[sel_idx]
                sub_inside = inside[sel_idx]
                false_gin_mask = gin & ~selected_sub
                false_gout_mask = gout & selected_sub
                stored[schema] = labels
                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "schema": schema,
                        "kappa": kappa,
                        "tie_rule": "adversarial",
                        "n_cohort": n_common,
                        "B": b_seats,
                        "guaranteed_in": c["guaranteed_in"],
                        "guaranteed_out": c["guaranteed_out"],
                        "unresolved": c["unresolved"],
                        "nc_common": c["unresolved"] / n_common if n_common else None,
                        "nc_full_common_numerator": (N_FULL_EXPECTED - n_common) + c["unresolved"],
                        "nc_full_common": ((N_FULL_EXPECTED - n_common) + c["unresolved"]) / N_FULL_EXPECTED,
                        "actual_present": int((mask & cell["actual_present"]).sum()),
                        "actual_missing": missing_actual,
                        "actual_inside_exact": int(inside.sum()),
                        "actual_outside_exact": int(outside.sum()),
                        "actual_outside_atol_1e_12": int((mask & cell["actual_present"] & ~inside_atol).sum()),
                        "containment_tolerance": "primary comparison is exact; atol 1e-12 is reported separately and is not used for membership",
                        "false_gin": int(false_gin_mask.sum()),
                        "false_gout": int(false_gout_mask.sum()),
                        "false_gin_with_noncontainment": int((false_gin_mask & sub_outside).sum()),
                        "false_gout_with_noncontainment": int((false_gout_mask & sub_outside).sum()),
                        "false_gin_with_containment": int((false_gin_mask & sub_inside).sum()),
                        "false_gout_with_containment": int((false_gout_mask & sub_inside).sum()),
                        "realized_population_matches_certificate": missing_actual == 0,
                        "realized_tie_rule": "ascending_string_user_id",
                    }
                )
            pack[("labels", cohort, model, kappa)] = stored
    return rows


def transition_rows(pack: dict[str, Any], mask: np.ndarray, cohort: str) -> list[dict[str, Any]]:
    pairs = (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3"))
    rows = []
    n = int(mask.sum())
    for kappa in KAPPAS:
        for model in MODELS:
            labels = pack[("labels", cohort, model, kappa)]
            for src, dst in pairs:
                a = labels[src]
                b = labels[dst]
                if len(a) != n or len(b) != n:
                    raise SystemExit("transition length mismatch")
                for la in LABELS:
                    for lb in LABELS:
                        rows.append(
                            {
                                "cohort": cohort,
                                "model": model,
                                "kappa": kappa,
                                "from_schema": src,
                                "to_schema": dst,
                                "from_label": la,
                                "to_label": lb,
                                "n": int(np.sum((a == la) & (b == lb))),
                            }
                        )
                changed = int(np.sum(a != b))
                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "kappa": kappa,
                        "from_schema": src,
                        "to_schema": dst,
                        "from_label": "_summary",
                        "to_label": "n_changed",
                        "n": changed,
                    }
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "kappa": kappa,
                        "from_schema": src,
                        "to_schema": dst,
                        "from_label": "_summary",
                        "to_label": "unresolved_to_guaranteed",
                        "n": int(np.sum((a == "unresolved") & (b != "unresolved"))),
                    }
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "kappa": kappa,
                        "from_schema": src,
                        "to_schema": dst,
                        "from_label": "_summary",
                        "to_label": "guaranteed_to_unresolved",
                        "n": int(np.sum((a != "unresolved") & (b == "unresolved"))),
                    }
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "kappa": kappa,
                        "from_schema": src,
                        "to_schema": dst,
                        "from_label": "_summary",
                        "to_label": "guaranteed_in_to_guaranteed_out",
                        "n": int(np.sum((a == "guaranteed_in") & (b == "guaranteed_out"))),
                    }
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "kappa": kappa,
                        "from_schema": src,
                        "to_schema": dst,
                        "from_label": "_summary",
                        "to_label": "guaranteed_out_to_guaranteed_in",
                        "n": int(np.sum((a == "guaranteed_out") & (b == "guaranteed_in"))),
                    }
                )
                nine = sum(int(np.sum((a == la) & (b == lb))) for la in LABELS for lb in LABELS)
                if nine != n:
                    raise SystemExit(f"transition accounting {model} {kappa} {src}->{dst} {nine} != {n}")
    return rows


def tie_rows(pack: dict[str, Any], mask: np.ndarray, cohort: str, schemas: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    violations = []
    ids = pack["user_ids"]
    n = int(mask.sum())
    for kappa in KAPPAS:
        b_seats = budget(n, kappa)
        for model in MODELS:
            for schema in schemas:
                cell = pack[(model, schema)]
                adv, _b = subset_labels(ids, cell["lo"], cell["hi"], mask, kappa, "adversarial")
                uid, _b2 = subset_labels(ids, cell["lo"], cell["hi"], mask, kappa, "ascending_user_id")
                if len(adv) != len(uid):
                    raise SystemExit("tie length mismatch")
                for adv_lab, uid_lab in zip(adv.tolist(), uid.tolist(), strict=True):
                    if adv_lab == "guaranteed_in" and uid_lab != "guaranteed_in":
                        violations.append({"cohort": cohort, "model": model, "schema": schema, "kappa": kappa, "kind": "gin"})
                    if adv_lab == "guaranteed_out" and uid_lab != "guaranteed_out":
                        violations.append({"cohort": cohort, "model": model, "schema": schema, "kappa": kappa, "kind": "gout"})
                ca, cu = counts(adv), counts(uid)
                changed = int(np.sum(adv != uid))
                rows.append(
                    {
                        "cohort": cohort,
                        "model": model,
                        "schema": schema,
                        "kappa": kappa,
                        "n_cohort": n,
                        "B": b_seats,
                        "adversarial_guaranteed_in": ca["guaranteed_in"],
                        "adversarial_guaranteed_out": ca["guaranteed_out"],
                        "adversarial_unresolved": ca["unresolved"],
                        "userid_guaranteed_in": cu["guaranteed_in"],
                        "userid_guaranteed_out": cu["guaranteed_out"],
                        "userid_unresolved": cu["unresolved"],
                        "n_labels_changed": changed,
                        "n_gin_changed": int(np.sum((adv == "guaranteed_in") != (uid == "guaranteed_in"))),
                        "n_gout_changed": int(np.sum((adv == "guaranteed_out") != (uid == "guaranteed_out"))),
                        "n_unresolved_changed": int(np.sum((adv == "unresolved") != (uid == "unresolved"))),
                        "nc_adversarial": ca["unresolved"] / n if n else None,
                        "nc_userid": cu["unresolved"] / n if n else None,
                        "delta_nc_userid_minus_adversarial": (cu["unresolved"] - ca["unresolved"]) / n if n else None,
                    }
                )
                for la in LABELS:
                    for lb in LABELS:
                        rows.append(
                            {
                                "cohort": cohort,
                                "model": model,
                                "schema": schema,
                                "kappa": kappa,
                                "n_cohort": n,
                                "B": b_seats,
                                "transition_from_adversarial": la,
                                "transition_to_userid": lb,
                                "n_transition": int(np.sum((adv == la) & (uid == lb))),
                            }
                        )
    return rows, violations


def status_audit(pack: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple, int] = defaultdict(int)
    boundary = 0
    total = 0
    for model in MODELS:
        for schema in SCHEMAS:
            cell = pack[(model, schema)]
            mask = cell["eligible"]
            labels, b_seats = subset_labels(pack["user_ids"], cell["lo"], cell["hi"], mask, PRIMARY_KAPPA, "adversarial")
            best, worst = adversarial_ranks(cell["lo"][mask], cell["hi"][mask])
            stored = cell["stored_status"][mask]
            derived = labels
            for i in range(len(derived)):
                tag = str(stored[i])
                if tag == "":
                    tag = "missing"
                if tag == "ambiguous":
                    tag = "unresolved"
                got = str(derived[i])
                if tag != got:
                    total += 1
                    on_boundary = int(worst[i]) == b_seats and got == "guaranteed_in" and tag == "unresolved"
                    if on_boundary:
                        boundary += 1
                    groups[(model, schema, tag, got, bool(on_boundary))] += 1
    return {
        "kappa": PRIMARY_KAPPA,
        "tie_rule": "adversarial",
        "n_eligible_tags_disagreeing": total,
        "n_worst_equals_B_derived_gin_stored_unresolved": boundary,
        "by_model_schema_stored_derived_boundary": [
            {
                "model": k[0],
                "schema": k[1],
                "stored_status": k[2],
                "derived_status": k[3],
                "worst_equals_B_and_derived_gin_stored_unresolved": k[4],
                "n": v,
            }
            for k, v in sorted(groups.items())
        ],
        "stored_tags_not_used_for_new_membership": True,
    }


def checkpoint_audit(meta: dict[str, Any], pack: dict[str, Any], common: np.ndarray) -> dict[str, Any]:
    keys = sorted({k for ks in pack["record_keys"].values() for k in ks})
    identity_fields = [
        k
        for k in keys
        if any(tok in k.lower() for tok in ("timestamp", "problem", "checkpoint", "event_index", "event_id", "item_id", "decision_index"))
    ]
    pairs = (("S0", "S1"), ("S1", "S2"), ("S2", "S3"), ("S0", "S3"))
    width: dict[str, dict[str, int]] = {}
    actual_equal: dict[str, dict[str, int]] = {}
    interval_equal: dict[str, dict[str, int]] = {}
    actual_equal_all_four: dict[str, int] = {}
    n_common = int(common.sum())
    for model in MODELS:
        width[model] = {}
        for schema in SCHEMAS:
            cell = pack[(model, schema)]
            width[model][schema] = int(np.sum(common & np.isfinite(cell["lo"]) & (cell["lo"] == cell["hi"])))
        actual_equal[model] = {}
        interval_equal[model] = {}
        same_all = common & pack[(model, "S0")]["actual_present"]
        base = pack[(model, "S0")]["actual"]
        for src, dst in pairs:
            left = pack[(model, src)]
            right = pack[(model, dst)]
            actual_equal[model][f"{src}_{dst}"] = int(np.sum(common & left["actual_present"] & right["actual_present"] & (left["actual"] == right["actual"])))
            interval_equal[model][f"{src}_{dst}"] = int(np.sum(common & np.isfinite(left["lo"]) & (left["lo"] == right["lo"]) & (left["hi"] == right["hi"])))
        for schema in SCHEMAS[1:]:
            cell = pack[(model, schema)]
            same_all = same_all & cell["actual_present"] & (cell["actual"] == base)
        actual_equal_all_four[model] = int(np.sum(same_all))
    return {
        "root_keys": meta["root_keys"],
        "non_model_keys": meta["non_model_keys"],
        "root_scalars": meta["scalars"],
        "per_schema_record_keys": pack["record_keys"],
        "n_events_is_a_count_not_an_event_identifier": "n_events" in keys,
        "identity_fields": identity_fields,
        "event_identity_available": bool(identity_fields),
        "equal_aggregate_values_are_not_checkpoint_identifiers": True,
        "common_n": n_common,
        "zero_width_intervals_in_common_cohort": width,
        "stored_actual_risk_equal_across_schema_pairs": actual_equal,
        "stored_actual_risk_equal_across_all_four_schemas": actual_equal_all_four,
        "stored_interval_endpoints_equal_across_schema_pairs": interval_equal,
        "statement": "The new comparison fixes the eligible users and capacity budget but retains schema-specific last-supported checkpoints. It is not a matched-event or matched-time comparison.",
    }


def bootstrap(pack: dict[str, Any], mask: np.ndarray) -> list[dict[str, Any]]:
    idx = np.flatnonzero(mask)
    n = int(len(idx))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n, size=(N_BOOT, n))
    point = {}
    samples = {model: {schema: np.empty(N_BOOT) for schema in SCHEMAS} for model in MODELS}
    for model in MODELS:
        for schema in SCHEMAS:
            labels, _b = subset_labels(pack["user_ids"], pack[(model, schema)]["lo"], pack[(model, schema)]["hi"], mask, PRIMARY_KAPPA, "adversarial")
            point[(model, schema)] = float(np.mean(labels == "unresolved"))
    failures = 0
    for r in range(N_BOOT):
        take = idx[draws[r]]
        b_seats = budget(n, PRIMARY_KAPPA)
        if b_seats != budget(len(take), PRIMARY_KAPPA):
            failures += 1
        for model in MODELS:
            for schema in SCHEMAS:
                cell = pack[(model, schema)]
                best, worst = adversarial_ranks(cell["lo"][take], cell["hi"][take])
                labels = labels_from_ranks(best, worst, b_seats)
                samples[model][schema][r] = float(np.mean(labels == "unresolved"))
    rows = []
    for model in MODELS:
        for schema in SCHEMAS:
            arr = samples[model][schema]
            rows.append(
                {
                    "cohort": "four_schema_intersection",
                    "model": model,
                    "contrast": schema,
                    "kappa": PRIMARY_KAPPA,
                    "tie_rule": "adversarial",
                    "point_nc": point[(model, schema)],
                    "boot_mean": float(arr.mean()),
                    "boot_p2_5": float(np.quantile(arr, 0.025)),
                    "boot_p97_5": float(np.quantile(arr, 0.975)),
                    "n_resamples": N_BOOT,
                    "n_failures": failures,
                    "seed": SEED,
                    "unit": "learner",
                    "note": "Percentile interval under paired learner resampling. Not a causal schema effect and not a certificate-validity guarantee.",
                }
            )
        diff = samples[model]["S1"] - samples[model]["S0"]
        rows.append(
            {
                "cohort": "four_schema_intersection",
                "model": model,
                "contrast": "NC_S1_minus_NC_S0",
                "kappa": PRIMARY_KAPPA,
                "tie_rule": "adversarial",
                "point_nc": point[(model, "S1")] - point[(model, "S0")],
                "boot_mean": float(diff.mean()),
                "boot_p2_5": float(np.quantile(diff, 0.025)),
                "boot_p97_5": float(np.quantile(diff, 0.975)),
                "n_resamples": N_BOOT,
                "n_failures": failures,
                "seed": SEED,
                "unit": "learner",
                "note": "Same resampled positions are applied to every schema. Duplicated draws are separate ranking positions.",
            }
        )
    return rows


def write_checksums() -> dict[str, Any]:
    rows = []
    for path in sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "output_checksums.json"):
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "algorithm": "sha256",
        "excluded": ["results/assistments/capacity_certification/output_checksums.json"],
        "n_files": len(rows),
        "files": rows,
    }
    (OUT / "output_checksums.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_rank_edges()
    print("edges ok", flush=True)
    meta, models = load_records()
    pack = align(models)
    print("loaded", pack["n_users"], "invalid", len(pack["invalid_intervals"]), flush=True)
    if pack["invalid_intervals"] or pack["nonfinite"] or pack["n_users"] != N_FULL_EXPECTED:
        (OUT / "cohort_readiness.json").write_text(
            json.dumps({"blocked": True, "n_users": pack["n_users"], "invalid": len(pack["invalid_intervals"]), "nonfinite": len(pack["nonfinite"])}, indent=2) + "\n",
            encoding="utf-8",
        )
        if pack["invalid_intervals"] or pack["nonfinite"]:
            raise SystemExit(3)
    sample_check = validate_against_loop(pack)
    expected = load_expected_sensitivity()
    native_rows, native_ok = native_table(pack, expected)
    pd.DataFrame(native_rows).to_csv(OUT / "native_capacity_reproduction.csv", index=False)
    print("native", native_ok, flush=True)
    if not native_ok:
        (OUT / "integrity_checks.json").write_text(json.dumps({"native_pass": False, "sample_check": sample_check}, indent=2) + "\n", encoding="utf-8")
        write_checksums()
        raise SystemExit(4)

    cohort = cohort_definition(pack)
    common = cohort.pop("common_mask")
    pairwise = cohort.pop("pairwise_mask")
    schema_masks = cohort.pop("schema_masks")
    (OUT / "cohort_readiness.json").write_text(json.dumps(cohort, indent=2) + "\n", encoding="utf-8")
    print("common", cohort["common_n"], "pairwise", cohort["pairwise_s0_s1_n"], flush=True)

    common_rows = capacity_rows(pack, common, "four_schema_intersection", SCHEMAS)
    pd.DataFrame(common_rows).to_csv(OUT / "common_cohort_capacity.csv", index=False)
    pair_rows = capacity_rows(pack, pairwise, "s0_s1_intersection", ("S0", "S1"))
    pd.DataFrame(pair_rows).to_csv(OUT / "pairwise_s0_s1_capacity.csv", index=False)
    trans = transition_rows(pack, common, "four_schema_intersection")
    pd.DataFrame(trans).to_csv(OUT / "capacity_label_transitions.csv", index=False)

    tie_common, viol_common = tie_rows(pack, common, "four_schema_intersection", SCHEMAS)
    tie_pair, viol_pair = tie_rows(pack, pairwise, "s0_s1_intersection", ("S0", "S1"))
    native_tie_rows = []
    native_viol = []
    ids = pack["user_ids"]
    for model in MODELS:
        for schema in SCHEMAS:
            mask = pack[(model, schema)]["eligible"]
            n = int(mask.sum())
            for kappa in KAPPAS:
                b_seats = budget(n, kappa)
                cell = pack[(model, schema)]
                adv, _ = subset_labels(ids, cell["lo"], cell["hi"], mask, kappa, "adversarial")
                uid, _ = subset_labels(ids, cell["lo"], cell["hi"], mask, kappa, "ascending_user_id")
                for adv_lab, uid_lab in zip(adv.tolist(), uid.tolist(), strict=True):
                    if adv_lab == "guaranteed_in" and uid_lab != "guaranteed_in":
                        native_viol.append({"cohort": "native_eligible", "model": model, "schema": schema, "kappa": kappa, "kind": "gin"})
                    if adv_lab == "guaranteed_out" and uid_lab != "guaranteed_out":
                        native_viol.append({"cohort": "native_eligible", "model": model, "schema": schema, "kappa": kappa, "kind": "gout"})
                ca, cu = counts(adv), counts(uid)
                native_tie_rows.append(
                    {
                        "cohort": "native_eligible",
                        "model": model,
                        "schema": schema,
                        "kappa": kappa,
                        "n_cohort": n,
                        "B": b_seats,
                        "adversarial_guaranteed_in": ca["guaranteed_in"],
                        "adversarial_guaranteed_out": ca["guaranteed_out"],
                        "adversarial_unresolved": ca["unresolved"],
                        "userid_guaranteed_in": cu["guaranteed_in"],
                        "userid_guaranteed_out": cu["guaranteed_out"],
                        "userid_unresolved": cu["unresolved"],
                        "n_labels_changed": int(np.sum(adv != uid)),
                        "n_gin_changed": int(np.sum((adv == "guaranteed_in") != (uid == "guaranteed_in"))),
                        "n_gout_changed": int(np.sum((adv == "guaranteed_out") != (uid == "guaranteed_out"))),
                        "n_unresolved_changed": int(np.sum((adv == "unresolved") != (uid == "unresolved"))),
                        "nc_adversarial": ca["unresolved"] / n,
                        "nc_userid": cu["unresolved"] / n,
                        "delta_nc_userid_minus_adversarial": (cu["unresolved"] - ca["unresolved"]) / n,
                    }
                )
    pd.DataFrame(tie_common + tie_pair + native_tie_rows).to_csv(OUT / "tie_rule_sensitivity.csv", index=False)

    native_realized = []
    for schema in SCHEMAS:
        schema_mask = pack[("TransparentActionPFA", schema)]["eligible"]
        if not np.array_equal(schema_mask, pack[("ActionTokenGRU", schema)]["eligible"]):
            raise SystemExit(f"native realized mask differs by consumer at {schema}")
        native_realized.extend(capacity_rows(pack, schema_mask, "native_eligible", (schema,)))
    pd.DataFrame(common_rows + native_realized).to_csv(OUT / "realized_risk_validity.csv", index=False)
    audit = status_audit(pack)
    (OUT / "status_tag_discrepancy_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    check_info = checkpoint_audit(meta, pack, common)
    (OUT / "checkpoint_comparability.json").write_text(json.dumps(check_info, indent=2) + "\n", encoding="utf-8")
    print("bootstrap", flush=True)
    boot_rows = bootstrap(pack, common)
    pd.DataFrame(boot_rows).to_csv(OUT / "paired_bootstrap_intervals.csv", index=False)

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    hashes = {
        "assist_heldout_user_aggregates.json": sha256_file(AGG_PATH),
        "p2_by_schema_model.csv": sha256_file(P2_PATH),
        "confirmatory_sealed_bundle.json": sha256_file(BUNDLE_PATH),
        "assist_heldout_validity.json": sha256_file(VALIDITY_PATH),
        "primary_frontier.json": sha256_file(FRONTIER_PATH),
        "script": sha256_file(Path(__file__).resolve()),
    }
    integrity = {
        "native_primary_pass": True,
        "native_sensitivity_pass": all(r["match"] is True for r in native_rows if r["match"] is not None),
        "rank_edge_tests_pass": True,
        "real_sample_pairwise_check": sample_check,
        "invalid_intervals": len(pack["invalid_intervals"]),
        "tie_containment_violations": {
            "common": len(viol_common),
            "pairwise": len(viol_pair),
            "native": len(native_viol),
        },
        "status_tag_disagreements": audit["n_eligible_tags_disagreeing"],
        "status_boundary_pattern": audit["n_worst_equals_B_derived_gin_stored_unresolved"],
        "event_identity_available": check_info["event_identity_available"],
        "bootstrap_replicates": N_BOOT,
        "bootstrap_failures": boot_rows[0]["n_failures"],
        "seed": SEED,
        "versions": versions,
        "hashes": hashes,
        "command": "python3 -u scripts/assistments/capacity_certification.py",
    }
    (OUT / "integrity_checks.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment": "pipeline",
        "models": list(MODELS),
        "schemas": list(SCHEMAS),
        "kappas": list(KAPPAS),
        "primary_tie_rule": "adversarial",
        "sensitivity_tie_rule": "ascending_string_user_id",
        "common_cohort": "intersection of eligible users across S0-S3 and both consumers",
        "budget": "max(1, round(kappa * n)) with Python round-half-to-even",
        "seed": SEED,
        "n_boot": N_BOOT,
        "versions": versions,
        "hashes": hashes,
        "command": "python3 -u scripts/assistments/capacity_certification.py",
        "checkpoint_statement": check_info["statement"],
    }
    (OUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums = write_checksums()
    bad = []
    for row in checksums["files"]:
        path = ROOT / row["path"]
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            bad.append(row["path"])
    self_listed = any(row["path"].endswith("output_checksums.json") for row in checksums["files"])
    manifest_sha = sha256_file(OUT / "output_checksums.json")
    print(json.dumps({"checksum_files": checksums["n_files"], "mismatches": bad, "self_listed": self_listed, "manifest_sha256": manifest_sha, "tie_violations": len(viol_common) + len(viol_pair) + len(native_viol)}, indent=2))
    if bad or self_listed or viol_common or viol_pair or native_viol:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
