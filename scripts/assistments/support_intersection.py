#!/usr/bin/env python3
"""Exact support-loss intersection and bootstrap-input checks for the factorial ablation.

Reads the frozen ablation score cache and per-user state. No training, no neural
forward pass, and no modification of the ablation outputs.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "chronorep", ROOT / "scripts/assistments/chronological_replication.py"
)
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)

FIELDABL = ROOT / "results/assistments/release_field_ablation"
OUT = FIELDABL / "suppint_support_qa"
CACHE = FIELDABL / "private_event_cache"
VALID = 939819
RELEASES = ("R00", "R10", "R01", "R11")
MODELS = ("TransparentActionPFA", "ActionTokenGRU", "CausalTokenAttentionKT")
TAUS = (0.70, 0.80, 0.90)
EDGES = (("R00", "R10"), ("R00", "R01"), ("R10", "R11"), ("R01", "R11"), ("R00", "R11"))
PATTERNS = ("0000", "1000", "1100", "1010", "1110", "1111")
EXPECTED_NC = {
    "TransparentActionPFA": (22974, 36163, 56296, 77892),
    "ActionTokenGRU": (129243, 99349, 87470, 93047),
    "CausalTokenAttentionKT": (176477, 136891, 138271, 131770),
}
EXPECTED_INTERACTION = {
    "TransparentActionPFA": 8407,
    "ActionTokenGRU": 35471,
    "CausalTokenAttentionKT": 33085,
}
EXPECTED_SET_INTERACTION = {
    "TransparentActionPFA": 155,
    "ActionTokenGRU": 27219,
    "CausalTokenAttentionKT": 24833,
}
GRU_SHA = "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
ATT_SHA = "8728fb05b33399df24daaaa48a76b3c6d6870aa2cee8419909cdeda5f88c0aaa"


def dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, default=_json) + "\n", encoding="utf-8")


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_listed_checksums() -> dict:
    manifest = json.loads((FIELDABL / "output_checksums.json").read_text())
    bad = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.exists() or chronorep.sha256_file(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            bad.append(item["path"])
    return {"n_files": manifest["n_files"], "mismatches": bad}


def status_from_scores(lo: np.ndarray, hi: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supported = np.isfinite(lo) & np.isfinite(hi)
    stable = supported & ((lo >= tau) | (hi < tau))
    sett = supported & ~stable
    unsupported = ~supported
    return unsupported, stable, sett


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("preflight", flush=True)
    pre = chronorep.preflight()
    gru = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
    att = ROOT / "results/assistments/attention_consumer/causal_token_attention_seed17.pt"
    pre["gru_sha256"] = chronorep.sha256_file(gru)
    pre["attention_sha256"] = chronorep.sha256_file(att)
    pre["gru_match"] = pre["gru_sha256"] == GRU_SHA
    pre["attention_match"] = pre["attention_sha256"] == ATT_SHA
    listed = verify_listed_checksums()
    pre["fieldabl_listed_checksums"] = listed
    pre["pass"] = bool(pre["pass"] and pre["gru_match"] and pre["attention_match"] and not listed["mismatches"])
    dump("preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_OR_STATISTICAL_INCONSISTENCY", "preflight": False})
        write_checksums()
        raise SystemExit(2)

    source = np.memmap(CACHE / "source_row.i8", dtype=np.int64, mode="r", shape=(VALID,))
    scores = np.memmap(CACHE / "scores.f8", dtype=np.float64, mode="r", shape=(VALID, 3, 4, 3))
    state = np.load(CACHE / "running_state.npz")
    print("align events", flush=True)
    grouped = chronorep.load_confirmatory()
    users = sorted(grouped)
    user_of = np.empty(VALID, dtype=np.int32)
    n_valid = np.zeros(len(users), dtype=np.int32)
    invalid = 0
    cursor = 0
    order_mismatches = 0
    for user_i, uid in enumerate(users):
        by_start, _by_end = chronorep.order_valid(grouped[uid])
        invalid += sum(1 for row in grouped[uid] if not row["valid"])
        n_valid[user_i] = len(by_start)
        for row in by_start:
            if int(source[cursor]) != int(row["source_row"]):
                order_mismatches += 1
            user_of[cursor] = user_i
            cursor += 1
    if cursor != VALID or invalid != 91 or order_mismatches or len(users) != 6766:
        dump("numerical_invariants.json", {
            "status": "BLOCKED_NUMERICAL_OR_STATISTICAL_INCONSISTENCY",
            "events": cursor, "invalid": invalid, "order_mismatches": order_mismatches, "users": len(users),
        })
        write_checksums()
        raise SystemExit(3)
    if not np.array_equal(n_valid, state["n_valid"]):
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_OR_STATISTICAL_INCONSISTENCY", "n_valid_match": False})
        write_checksums()
        raise SystemExit(3)

    support = np.isfinite(scores[:, :, :, 0]) & np.isfinite(scores[:, :, :, 1])
    support_same_across_models = bool(np.array_equal(support[:, 0, :], support[:, 1, :]) and np.array_equal(support[:, 0, :], support[:, 2, :]))
    mask = support[:, 0, :]
    b = mask[:, 0]
    h = mask[:, 1]
    a = mask[:, 2]
    j = mask[:, 3]
    forbidden = {
        "R11_without_R10": int(np.sum(j & ~h)),
        "R11_without_R01": int(np.sum(j & ~a)),
        "R10_without_R00": int(np.sum(h & ~b)),
        "R01_without_R00": int(np.sum(a & ~b)),
    }
    c = int(np.sum(b & ~h & ~a))
    joint_only = int(np.sum(h & a & ~j))
    l_h = int(np.sum(b & ~h))
    l_a = int(np.sum(b & ~a))
    b_minus_j = int(np.sum(b & ~j))
    count_interaction = b_minus_j - l_h - l_a
    patterns = {
        "0000": ~b,
        "1000": b & ~h & ~a,
        "1100": b & h & ~a,
        "1010": b & ~h & a,
        "1110": h & a & ~j,
        "1111": j,
    }
    counts = {name: int(flag.sum()) for name, flag in patterns.items()}
    pattern_ok = (
        counts["0000"] == 20408
        and counts["1000"] == c
        and counts["1100"] == 35205 - c
        and counts["1010"] == 14012 - c
        and counts["1110"] == 8252 + c
        and counts["1111"] == 861942
        and sum(counts.values()) == VALID
        and all(value == 0 for value in forbidden.values())
        and joint_only == 8252 + c
        and l_h == 14012
        and l_a == 35205
        and b_minus_j == 57469
        and int(b.sum()) == 919411
        and int(h.sum()) == 905399
        and int(a.sum()) == 884206
        and int(j.sum()) == 861942
    )
    covered = np.zeros(VALID, dtype=bool)
    overlap = 0
    for flag in patterns.values():
        overlap += int(np.sum(covered & flag))
        covered |= flag
    partition_ok = overlap == 0 and int(covered.sum()) == VALID
    write_csv("support_six_pattern_contingency.csv", [
        {"pattern": name, "r00": int(name[0]), "r10": int(name[1]), "r01": int(name[2]), "r11": int(name[3]), "events": counts[name]}
        for name in PATTERNS
    ])
    dump("support_loss_intersection.json", {
        "B": int(b.sum()), "H": int(h.sum()), "A": int(a.sum()), "J": int(j.sum()),
        "L_H": l_h, "L_A": l_a, "B_minus_J": b_minus_j,
        "c_L_H_intersection_L_A": c,
        "joint_only_H_and_A_not_J": joint_only,
        "count_interaction_B_minus_J_minus_L_H_minus_L_A": count_interaction,
        "identity_joint_only_equals_8252_plus_c": joint_only == 8252 + c,
        "original_8252_equals_joint_only_only_if_c_is_zero": True,
        "c_is_zero": c == 0,
        "forbidden": forbidden,
        "patterns": counts,
        "pattern_sum": sum(counts.values()),
        "patterns_are_mutually_exclusive_and_exhaustive": partition_ok,
        "support_identical_across_consumers": support_same_across_models,
        "interpretation": (
            "8,252 is the unsupported-count interaction |B\\J|-|B\\H|-|B\\A|. "
            "The event count supported by both single-field releases but not the joint release is 8252+c."
        ),
    })

    print("learner accounting", flush=True)
    learner_patterns = np.column_stack([
        np.bincount(user_of[flag], minlength=len(users)).astype(np.int32) for flag in (patterns[name] for name in PATTERNS)
    ])
    private = OUT / "private_learner_support.npz"
    np.savez_compressed(private, pattern_counts=learner_patterns, n_valid=n_valid)
    learner_sum_ok = bool(np.array_equal(learner_patterns.sum(axis=1), n_valid))
    marginal_from_patterns = {
        "R00": learner_patterns[:, 1] + learner_patterns[:, 2] + learner_patterns[:, 3] + learner_patterns[:, 4] + learner_patterns[:, 5],
        "R10": learner_patterns[:, 2] + learner_patterns[:, 4] + learner_patterns[:, 5],
        "R01": learner_patterns[:, 3] + learner_patterns[:, 4] + learner_patterns[:, 5],
        "R11": learner_patterns[:, 5],
    }
    supported_match = all(np.array_equal(marginal_from_patterns[release], state["supported_n"][:, 0, i]) for i, release in enumerate(RELEASES))
    supported_same_models = all(
        np.array_equal(state["supported_n"][:, 0, i], state["supported_n"][:, model, i])
        for model in range(1, 3) for i in range(4)
    )
    zero_users = {release: int(np.sum(marginal_from_patterns[release] == 0)) for release in RELEASES}
    summed = {name: int(learner_patterns[:, i].sum()) for i, name in enumerate(PATTERNS)}
    invalid_in_source = int(np.sum(~np.isin(source, np.array([row["source_row"] for rows in grouped.values() for row in rows if row["valid"]], dtype=np.int64))))

    disk_unsupported_sum = int(state["status"][:, :, :, :, 0].sum())
    status_match = []
    per_learner_identity = True
    per_learner_nc_identity = True
    for model_i in range(3):
        for release_i in range(4):
            for tau_i, tau in enumerate(TAUS):
                unsupported, stable, sett = status_from_scores(scores[:, model_i, release_i, 0], scores[:, model_i, release_i, 1], tau)
                stable_u = np.bincount(user_of[stable], minlength=len(users)).astype(np.int32)
                set_u = np.bincount(user_of[sett], minlength=len(users)).astype(np.int32)
                unsup_u = np.bincount(user_of[unsupported], minlength=len(users)).astype(np.int32)
                disk_stable = state["status"][:, model_i, release_i, tau_i, 1]
                disk_set = state["status"][:, model_i, release_i, tau_i, 2]
                disk_unsup = state["status"][:, model_i, release_i, tau_i, 0]
                restored = n_valid - state["supported_n"][:, model_i, release_i]
                ok_bins = np.array_equal(stable_u, disk_stable) and np.array_equal(set_u, disk_set) and np.array_equal(unsup_u, restored)
                ok_identity = np.array_equal(restored + disk_stable + disk_set, n_valid)
                ok_nc = np.array_equal(restored + disk_set, unsup_u + set_u)
                per_learner_identity = per_learner_identity and bool(ok_identity)
                per_learner_nc_identity = per_learner_nc_identity and bool(ok_nc)
                status_match.append(bool(ok_bins))
    dump("learner_support_consistency.json", {
        "learners": len(users),
        "learner_pattern_counts_sum_to_valid_targets": learner_sum_ok,
        "learner_marginals_match_stored_supported_counts": bool(supported_match),
        "supported_counts_identical_across_consumers": bool(supported_same_models),
        "summed_learner_patterns": summed,
        "summed_patterns_match_event_contingency": summed == counts,
        "learners_with_zero_supported_targets": zero_users,
        "invalid_interval_events_in_support_cache": invalid_in_source,
        "learner_ids_exported": False,
        "private_cache": "results/assistments/release_field_ablation/support_intersection/private_learner_support.npz",
    })

    print("bootstrap", flush=True)
    restored_status = np.array(state["status"], copy=True)
    for model_i in range(3):
        for release_i in range(4):
            restored = n_valid - state["supported_n"][:, model_i, release_i]
            for tau_i in range(3):
                restored_status[:, model_i, release_i, tau_i, 0] = restored
    reproduced = reproduce_bootstrap(restored_status, n_valid, state["matched_prob"])
    original = list(csv.DictReader((FIELDABL / "learner_bootstrap_intervals.csv").open()))
    comparisons = []
    exact = True
    for new, old in zip(reproduced, original):
        same = (
            new["analysis"] == old["analysis"]
            and new["model"] == old["model"]
            and new["quantity"] == old["quantity"]
            and new["point_rate"] == old["point_rate"]
            and new["ci_low"] == old["ci_low"]
            and new["ci_high"] == old["ci_high"]
        )
        exact = exact and same
        comparisons.append({
            "analysis": old["analysis"], "model": old["model"], "quantity": old["quantity"],
            "original_point_rate": old["point_rate"], "recomputed_point_rate": new["point_rate"],
            "original_ci_low": old["ci_low"], "recomputed_ci_low": new["ci_low"],
            "original_ci_high": old["ci_high"], "recomputed_ci_high": new["ci_high"],
            "exact_match": same,
        })
    if len(reproduced) != len(original):
        exact = False
    unrepaired_point = float((state["status"][:, 0, 0, 1, 0] + state["status"][:, 0, 0, 1, 2]).sum() / VALID)
    dump("bootstrap_input_integrity.json", {
        "disk_status_unsupported_bin_sum": disk_unsupported_sum,
        "disk_unsupported_bin_was_not_rewritten": disk_unsupported_sum == 0,
        "restoration_formula": "valid_events - supported_events",
        "restoration_applied_in_fieldabl_before_bootstrap_call": True,
        "restoration_code_path": "main calls repair_unsupported(result) before bootstrap(result); score_population is the only writer of running_state.npz",
        "published_bootstrap_uses_corrected_in_memory_counts": exact,
        "unrepaired_pfa_r00_event_nc_rate": unrepaired_point,
        "published_pfa_r00_event_nc_rate": original[0]["point_rate"],
        "unrepaired_point_differs_from_published_point": unrepaired_point != float(original[0]["point_rate"]),
        "per_learner_restored_partition_complete": per_learner_identity and all(status_match),
        "per_learner_event_nc_equals_unsupported_plus_set_valued": per_learner_nc_identity,
        "identical_learner_draws_across_consumers": True,
        "seed": 20312,
        "replicates": 2000,
        "event_iid_bootstrap": False,
    })
    write_csv("bootstrap_reproduction_comparison.csv", comparisons)

    print("transitions", flush=True)
    recomputed = np.zeros_like(state["transitions"])
    for model_i in range(3):
        for tau_i, tau in enumerate(TAUS):
            codes = []
            for release_i in range(4):
                unsupported, stable, sett = status_from_scores(scores[:, model_i, release_i, 0], scores[:, model_i, release_i, 1], tau)
                code = np.zeros(VALID, dtype=np.int8)
                code[stable] = 1
                code[sett] = 2
                codes.append(code)
            for edge_i, (left, right) in enumerate(EDGES):
                left_i = RELEASES.index(left)
                right_i = RELEASES.index(right)
                pair = codes[left_i] * 3 + codes[right_i]
                counts_pair = np.bincount(pair, minlength=9)
                recomputed[model_i, tau_i, edge_i] = counts_pair.reshape(3, 3)
    transition_equal = bool(np.array_equal(recomputed, state["transitions"]))
    transition_report = []
    transition_ok = transition_equal
    for model_i, model in enumerate(MODELS):
        for tau_i, tau in enumerate(TAUS):
            for edge_i, (left, right) in enumerate(EDGES):
                matrix = recomputed[model_i, tau_i, edge_i]
                left_i = RELEASES.index(left)
                right_i = RELEASES.index(right)
                row_totals = [int(matrix[i, :].sum()) for i in range(3)]
                col_totals = [int(matrix[:, i].sum()) for i in range(3)]
                left_status = [int(np.sum(status_from_scores(scores[:, model_i, left_i, 0], scores[:, model_i, left_i, 1], tau)[k])) for k in range(3)]
                right_status = [int(np.sum(status_from_scores(scores[:, model_i, right_i, 0], scores[:, model_i, right_i, 1], tau)[k])) for k in range(3)]
                loss = int(matrix[1, 0] + matrix[2, 0])
                mask_loss = int(np.sum(support[:, model_i, left_i] & ~support[:, model_i, right_i]))
                cell_ok = (
                    int(matrix.sum()) == VALID
                    and row_totals == left_status
                    and col_totals == right_status
                    and int(matrix[1, 2]) == 0
                    and int(matrix[0, 1] + matrix[0, 2]) == 0
                    and loss == mask_loss
                )
                transition_ok = transition_ok and cell_ok
                if tau == 0.80:
                    transition_report.append({
                        "model": model, "tau": tau, "left": left, "right": right,
                        "matrix_sum": int(matrix.sum()),
                        "row_marginals": row_totals, "column_marginals": col_totals,
                        "stable_to_set": int(matrix[1, 2]),
                        "unsupported_to_supported": int(matrix[0, 1] + matrix[0, 2]),
                        "gross_support_loss": loss, "mask_support_loss": mask_loss,
                        "set_to_stable": int(matrix[2, 1]),
                        "pass": cell_ok,
                    })
    dump("transition_matrix_integrity.json", {
        "recomputed_from_cached_scores_matches_stored_matrices": transition_equal,
        "all_edges_and_thresholds_pass": transition_ok,
        "primary_tau_edges": transition_report,
        "neural_rescore": False,
    })

    component_rows = []
    factorial_ok = True
    for model_i, model in enumerate(MODELS):
        totals = []
        for release_i in range(4):
            unsupported, stable, sett = status_from_scores(scores[:, model_i, release_i, 0], scores[:, model_i, release_i, 1], 0.80)
            totals.append((int(unsupported.sum()), int(stable.sum()), int(sett.sum())))
        nc = [item[0] + item[2] for item in totals]
        unsup = [item[0] for item in totals]
        sett = [item[2] for item in totals]
        interaction_nc = nc[3] - nc[1] - nc[2] + nc[0]
        interaction_u = unsup[3] - unsup[1] - unsup[2] + unsup[0]
        interaction_s = sett[3] - sett[1] - sett[2] + sett[0]
        match = (
            tuple(nc) == EXPECTED_NC[model]
            and interaction_nc == EXPECTED_INTERACTION[model]
            and interaction_u == 8252
            and interaction_s == EXPECTED_SET_INTERACTION[model]
            and interaction_nc == interaction_u + interaction_s
        )
        factorial_ok = factorial_ok and match
        component_rows.append({
            "model": model,
            "NC00": nc[0], "NC10": nc[1], "NC01": nc[2], "NC11": nc[3],
            "hint_only": nc[1] - nc[0], "attempt_only": nc[2] - nc[0],
            "hint_after_attempt": nc[3] - nc[2], "attempt_after_hint": nc[3] - nc[1],
            "joint": nc[3] - nc[0], "interaction_nc": interaction_nc,
            "interaction_unsupported": interaction_u, "interaction_set_valued": interaction_s,
            "interaction_nc_equals_sum_of_components": interaction_nc == interaction_u + interaction_s,
            "matches_ablation": match,
        })
    write_csv("factorial_component_interactions.csv", component_rows)

    checks = {
        "source_and_fieldabl_checksums": pre["pass"],
        "event_alignment": order_mismatches == 0 and cursor == VALID and invalid == 91,
        "support_cardinalities_and_patterns": pattern_ok and partition_ok and support_same_across_models,
        "learner_support": learner_sum_ok and bool(supported_match) and summed == counts and invalid_in_source == 0,
        "restored_learner_status": per_learner_identity and per_learner_nc_identity and all(status_match),
        "bootstrap_exact": exact,
        "factorial": factorial_ok,
        "transitions": transition_ok,
    }
    status = "COMPLETE" if all(checks.values()) and c == 0 else (
        "COMPLETE_WITH_NOTES" if all(checks.values()) else "BLOCKED_NUMERICAL_OR_STATISTICAL_INCONSISTENCY"
    )
    dump("numerical_invariants.json", {"checks": checks, "c": c, "joint_only": joint_only, "status": status})
    dump("execution_manifest.json", {
        "experiment": "pipeline",
        "script": "scripts/assistments/support_intersection.py",
        "script_sha256": chronorep.sha256_file(Path(__file__).resolve()),
        "neural_forward_passes": 0,
        "training": False,
        "fieldabl_files_overwritten": False,
        "status": status,
    })
    checksums = write_checksums()
    print(status, "c", c, "joint_only", joint_only, "checksum_mismatches", len(checksums["mismatches"]), flush=True)
    if not all(checks.values()) or checksums["mismatches"]:
        raise SystemExit(5)


def reproduce_bootstrap(status: np.ndarray, n_valid: np.ndarray, matched: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(20312)
    n_users = len(n_valid)
    draws = rng.integers(0, n_users, size=(2000, n_users))
    rows = []
    valid = n_valid.astype(np.float64)
    for model_i, model in enumerate(MODELS):
        samples = {}
        point = {}
        for release_i, release in enumerate(RELEASES):
            nc = status[:, model_i, release_i, 1, 0].astype(np.float64) + status[:, model_i, release_i, 1, 2].astype(np.float64)
            point[release] = float(nc.sum() / VALID)
            samples[release] = nc[draws].sum(axis=1) / valid[draws].sum(axis=1)
        derived = {
            "NC00": samples["R00"], "NC10": samples["R10"], "NC01": samples["R01"], "NC11": samples["R11"],
            "delta_H": samples["R10"] - samples["R00"], "delta_A": samples["R01"] - samples["R00"],
            "delta_H_given_A": samples["R11"] - samples["R01"], "delta_A_given_H": samples["R11"] - samples["R10"],
            "interaction": samples["R11"] - samples["R10"] - samples["R01"] + samples["R00"],
            "joint": samples["R11"] - samples["R00"],
        }
        points = {
            "NC00": point["R00"], "NC10": point["R10"], "NC01": point["R01"], "NC11": point["R11"],
            "delta_H": point["R10"] - point["R00"], "delta_A": point["R01"] - point["R00"],
            "delta_H_given_A": point["R11"] - point["R01"], "delta_A_given_H": point["R11"] - point["R10"],
            "interaction": point["R11"] - point["R10"] - point["R01"] + point["R00"],
            "joint": point["R11"] - point["R00"],
        }
        for name, values in derived.items():
            low, high = np.percentile(values, [2.5, 97.5])
            rows.append({
                "analysis": "event_nc_learner_bootstrap", "model": model, "quantity": name,
                "point_rate": format(points[name], ".17g"), "ci_low": format(float(low), ".17g"), "ci_high": format(float(high), ".17g"),
            })
    common = np.flatnonzero(np.isfinite(matched[:, 0, 0, 0]) & np.all(np.isfinite(matched[:, 0, :, 0]), axis=1))
    for model_i, model in enumerate(MODELS):
        for release_i, release in enumerate(RELEASES):
            cell = matched[common, model_i, release_i, :]
            lo0 = 1.0 - cell[:, 1]
            hi0 = 1.0 - cell[:, 0]
            point_counts = chronorep.membership_counts(lo0, hi0, 0.05)
            unresolved = np.empty(2000, dtype=np.float64)
            for _replicate, draw in enumerate(rng.integers(0, len(common), size=(2000, len(common)))):
                unresolved[_replicate] = chronorep.membership_counts(lo0[draw], hi0[draw], 0.05)["unresolved"] / len(common)
            low, high = np.percentile(unresolved, [2.5, 97.5])
            rows.append({
                "analysis": "matched_capacity_unresolved_rate", "model": model, "quantity": f"{release}_unresolved_rate",
                "point_rate": format(point_counts["unresolved"] / len(common), ".17g"),
                "ci_low": format(float(low), ".17g"), "ci_high": format(float(high), ".17g"),
            })
    return rows


def write_checksums() -> dict:
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {
        "algorithm": "sha256",
        "excluded": ["results/assistments/release_field_ablation/support_intersection/output_checksums.json"],
        "n_files": len(files),
        "files": files,
    })
    bad = [item["path"] for item in files if chronorep.sha256_file(ROOT / item["path"]) != item["sha256"]]
    return {"n_files": len(files), "mismatches": bad}


if __name__ == "__main__":
    main()
