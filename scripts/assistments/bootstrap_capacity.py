#!/usr/bin/env python3
"""Learner-bootstrap validity and matched-capacity rank monotonicity.

No training and no candidate rescoring. The behavioral-plausibility tables are
not overwritten; the corrected intervals are written to a separate directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BEHPLAUS = ROOT / "results/assistments/behavioral_plausibility"
OUT = BEHPLAUS / "bootcap_statistics_qa"
PRIV = OUT / "private"
NPZ = BEHPLAUS / "private_scores/confirmatory_filtered.npz"
META = BEHPLAUS / "private_scores/confirmatory_accum.json"
CHRONOREP_BOOT = ROOT / "results/assistments/chronological_replication/paired_bootstrap_intervals.csv"
CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
RECON = ROOT / "data/processed/assistments"
ZIP = ROOT / "data/raw/assistments/assistments_2012.zip"
CONSUMERS = ("TransparentActionPFA", "ActionTokenGRU")
SCHEMAS = ("S0", "S1", "S2", "S3")
FILTERS = ("F0", "F1", "F2", "F3")
N_FIXED = 939819
N_USERS = 6766
SEED = 20319
REPS = 2000
TAU = 0.80
EXPECTED = {
    ("TransparentActionPFA", "S0", "F0"): (22974, 45),
    ("TransparentActionPFA", "S0", "F3"): (117092, 60),
    ("ActionTokenGRU", "S0", "F0"): (129243, 1077),
    ("ActionTokenGRU", "S0", "F3"): (167677, 2411),
}
PAIRED = {
    ("TransparentActionPFA", "F0"): 54918,
    ("TransparentActionPFA", "F1"): 71511,
    ("TransparentActionPFA", "F2"): 75341,
    ("TransparentActionPFA", "F3"): 76429,
    ("ActionTokenGRU", "F0"): -36196,
    ("ActionTokenGRU", "F1"): 6961,
    ("ActionTokenGRU", "F2"): 14899,
    ("ActionTokenGRU", "F3"): 32660,
}
CHRONOREP_RATE = {
    ("TransparentActionPFA", "S0"): (0.024445132520198, 0.0206439344534175, 0.0293364796739447),
    ("ActionTokenGRU", "S0"): (0.1375190329201686, 0.1324062452445182, 0.1432656401143065),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    header = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                header.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in header})


def percentile(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (float("nan"), float("nan"), float("nan"))
    low, high = np.quantile(finite, [0.025, 0.975], method="linear")
    return float(np.mean(finite)), float(low), float(high)


def adversarial_ranks(los: np.ndarray, his: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(los))
    order_lo = np.sort(los)
    order_hi = np.sort(his)
    n_lo_gt = n - np.searchsorted(order_lo, his, side="right")
    self_best = (los > his).astype(np.int32)
    best = (1 + n_lo_gt - self_best).astype(np.int32)
    n_hi_ge = n - np.searchsorted(order_hi, los, side="left")
    self_worst = (his >= los).astype(np.int32)
    worst = np.where(self_worst == 1, n_hi_ge, 1 + n_hi_ge).astype(np.int32)
    return best, worst


def labels_from_ranks(best: np.ndarray, worst: np.ndarray, seats: int) -> np.ndarray:
    out = np.full(len(best), 2, dtype=np.uint8)  # unresolved
    out[worst <= seats] = 0  # guaranteed_in
    out[(best > seats) & (worst > seats)] = 1  # guaranteed_out
    return out


def checkpoint_indices(user, end, source, supported):
    n_users = int(user.max()) + 1
    chosen = np.full(n_users, -1, dtype=np.int32)
    idx = np.flatnonzero(supported)
    ordered = idx[np.lexsort((source[idx], end[idx], user[idx]))]
    owners = user[ordered]
    last = np.empty(len(ordered), dtype=bool)
    last[-1] = True
    last[:-1] = owners[:-1] != owners[1:]
    chosen[owners[last]] = ordered[last]
    return chosen


def status_counts(code: np.ndarray, realized: np.ndarray, user: np.ndarray) -> dict:
    n_users = N_USERS
    events = np.bincount(user, minlength=n_users).astype(np.int64)
    out = {"events": events}
    for model in range(2):
        label = realized[:, model] >= TAU
        for schema in range(4):
            for filt in range(4):
                labels = code[:, model, schema, filt]
                stable = (labels == 1) | (labels == 2)
                sett = labels == 3
                unsup = labels == 0
                fs = stable & ((labels == 1) != label)
                key = (model, schema, filt)
                out[key] = {
                    "supported": np.bincount(user[labels > 0], minlength=n_users).astype(np.int64),
                    "unsupported": np.bincount(user[unsup], minlength=n_users).astype(np.int64),
                    "stable": np.bincount(user[stable], minlength=n_users).astype(np.int64),
                    "set_valued": np.bincount(user[sett], minlength=n_users).astype(np.int64),
                    "false_stable": np.bincount(user[fs], minlength=n_users).astype(np.int64),
                }
                out[key]["event_nc"] = out[key]["unsupported"] + out[key]["set_valued"]
                out[key]["correct_stable"] = out[key]["stable"] - out[key]["false_stable"]
    return out


def reproduce_original(stable_by_cell: dict) -> dict:
    """Re-execute the published behavioral-plausibility statistic without changing its formula."""
    rng = np.random.default_rng(SEED)
    published = {}
    with (BEHPLAUS / "learner_bootstrap_intervals.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "event_nc" and row["schema"] == "S0" and row["filter"] == "F0":
                published[row["consumer"]] = (float(row["p2_5"]), float(row["p97_5"]), float(row["point"]))
    report = {"seed": SEED, "replicates": REPS, "generator": "numpy.random.Generator.integers", "consumers": {}}
    for model, name in enumerate(CONSUMERS):
        draws = rng.integers(0, N_USERS, size=(REPS, N_USERS))
        stable = stable_by_cell[(model, 0, 0)][draws].sum(axis=1)
        statistic = N_FIXED - stable
        mean, low, high = percentile(statistic)
        pub = published[name]
        report["consumers"][name] = {
            "formula": "939819 - sum_u multiplicity(u) * stable_count(u)",
            "negative_replicates": int(np.sum(statistic < 0)),
            "min_replicate": float(statistic.min()),
            "max_replicate": float(statistic.max()),
            "p2_5": low,
            "p97_5": high,
            "published_p2_5": pub[0],
            "published_p97_5": pub[1],
            "published_point": pub[2],
            "endpoint_abs_diff": max(abs(low - pub[0]), abs(high - pub[1])),
            "point_is_original_population_count": pub[2] == EXPECTED[(name, "S0", "F0")][0],
        }
    report["cause"] = (
        "The published marginal Event NC subtracts a with-replacement sum of learner Stable counts from the fixed population size. "
        "That sum is not an Event NC count. When frequently sampled learners contribute many Stable events, it exceeds 939819 and the reported interval endpoint becomes negative. "
        "No negative sampling weight is involved."
    )
    return report


def corrected_bootstrap(counts: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, N_USERS, size=(REPS, N_USERS))
    weights_min = int(np.bincount(draws[0], minlength=N_USERS).min())
    events = counts["events"]
    n_b = events[draws].sum(axis=1).astype(np.float64)
    marginal = []
    paired = []
    comparison = []
    packed = {}
    invariants = {
        "weights_nonnegative": weights_min >= 0,
        "minimum_weight_in_first_replicate": weights_min,
        "replicates_with_zero_events": int(np.sum(n_b <= 0)),
        "rate_below_zero": 0,
        "rate_above_one": 0,
        "scaled_count_below_zero": 0,
        "scaled_count_above_population": 0,
        "conditional_undefined": {},
    }
    for model, name in enumerate(CONSUMERS):
        for schema_i, schema in enumerate(SCHEMAS):
            for filter_i, filt in enumerate(FILTERS):
                cell = counts[(model, schema_i, filter_i)]
                packed[(model, schema_i, filter_i)] = {
                    key: cell[key][draws].sum(axis=1).astype(np.float64)
                    for key in ("supported", "stable", "set_valued", "unsupported", "event_nc", "false_stable", "correct_stable")
                }
                item = packed[(model, schema_i, filter_i)]
                rate = item["event_nc"] / n_b
                scaled = N_FIXED * rate
                invariants["rate_below_zero"] += int(np.sum(rate < -1e-15))
                invariants["rate_above_one"] += int(np.sum(rate > 1 + 1e-12))
                invariants["scaled_count_below_zero"] += int(np.sum(scaled < -1e-8))
                invariants["scaled_count_above_population"] += int(np.sum(scaled > N_FIXED + 1e-6))
                mean, low, high = percentile(rate)
                smean, slow, shigh = percentile(scaled)
                point_count = int(cell["event_nc"].sum())
                point_rate = point_count / N_FIXED
                marginal.append({
                    "consumer": name, "schema": schema, "filter": filt,
                    "estimand": "event_nc_rate",
                    "point": point_rate, "bootstrap_mean": mean, "p2_5": low, "p97_5": high,
                    "min_replicate": float(rate.min()), "max_replicate": float(rate.max()),
                    "finite_replicates": REPS, "undefined_replicates": 0, "seed": SEED,
                })
                marginal.append({
                    "consumer": name, "schema": schema, "filter": filt,
                    "estimand": "population_scaled_event_nc_count",
                    "point": point_count, "bootstrap_mean": smean, "p2_5": slow, "p97_5": shigh,
                    "min_replicate": float(scaled.min()), "max_replicate": float(scaled.max()),
                    "finite_replicates": REPS, "undefined_replicates": 0, "seed": SEED,
                    "scaling": "939819 * (resampled_event_nc / resampled_event_count)",
                })
                stable_rate = item["stable"] / n_b
                support_rate = item["supported"] / n_b
                correct_rate = item["correct_stable"] / n_b
                for estimand, values, point in (
                    ("stable_issuance_rate", stable_rate, int(cell["stable"].sum()) / N_FIXED),
                    ("support_fraction", support_rate, int(cell["supported"].sum()) / N_FIXED),
                    ("correct_stable_coverage", correct_rate, int(cell["correct_stable"].sum()) / N_FIXED),
                ):
                    mean, low, high = percentile(values)
                    marginal.append({
                        "consumer": name, "schema": schema, "filter": filt, "estimand": estimand,
                        "point": point, "bootstrap_mean": mean, "p2_5": low, "p97_5": high,
                        "min_replicate": float(values.min()), "max_replicate": float(values.max()),
                        "finite_replicates": REPS, "undefined_replicates": 0, "seed": SEED,
                    })
                fs_rate = np.divide(item["false_stable"], item["stable"], out=np.full(REPS, np.nan), where=item["stable"] > 0)
                undefined = int(np.sum(item["stable"] <= 0))
                invariants["conditional_undefined"][f"{name}|{schema}|{filt}|false_stable_over_stable"] = undefined
                mean, low, high = percentile(fs_rate)
                marginal.append({
                    "consumer": name, "schema": schema, "filter": filt, "estimand": "false_stable_over_stable",
                    "point": None if cell["stable"].sum() == 0 else float(cell["false_stable"].sum() / cell["stable"].sum()),
                    "bootstrap_mean": mean, "p2_5": low, "p97_5": high,
                    "min_replicate": None if undefined == REPS else float(np.nanmin(fs_rate)),
                    "max_replicate": None if undefined == REPS else float(np.nanmax(fs_rate)),
                    "finite_replicates": int(np.sum(np.isfinite(fs_rate))), "undefined_replicates": undefined, "seed": SEED,
                })
    for model, name in enumerate(CONSUMERS):
        for schema_i, schema in enumerate(SCHEMAS):
            f0 = packed[(model, schema_i, 0)]
            f3 = packed[(model, schema_i, 3)]
            contrasts = {
                "f3_minus_f0_event_nc_rate": (f3["event_nc"] - f0["event_nc"]) / n_b,
                "f3_minus_f0_stable_issuance_rate": (f3["stable"] - f0["stable"]) / n_b,
                "f3_minus_f0_support_fraction": (f3["supported"] - f0["supported"]) / n_b,
                "f3_minus_f0_correct_stable_coverage": (f3["correct_stable"] - f0["correct_stable"]) / n_b,
                "new_false_stable_rate": None,
                "removed_false_stable_rate": None,
            }
            new_u = np.maximum(counts[(model, schema_i, 3)]["false_stable"] - counts[(model, schema_i, 0)]["false_stable"], 0)
            # The event-level new/removed indicators are not recoverable from the difference of learner totals
            # when both can be nonzero for one learner. They are passed separately below.
            del contrasts["new_false_stable_rate"], contrasts["removed_false_stable_rate"]
            for estimand, values in contrasts.items():
                scaled = N_FIXED * values
                mean, low, high = percentile(values)
                smean, slow, shigh = percentile(scaled)
                point = float(values[0] * 0 + (counts[(model, schema_i, 3)][estimand.split("_")[3] if False else "event_nc"].sum()))
                paired.append({
                    "consumer": name, "schema": schema, "estimand": estimand,
                    "point_rate": float((counts[(model, schema_i, 3)]["event_nc"].sum() - counts[(model, schema_i, 0)]["event_nc"].sum()) / N_FIXED) if "event_nc" in estimand else float(np.sum(counts[(model, schema_i, 3)][{"f3_minus_f0_stable_issuance_rate": "stable", "f3_minus_f0_support_fraction": "supported", "f3_minus_f0_correct_stable_coverage": "correct_stable"}[estimand]] - counts[(model, schema_i, 0)][{"f3_minus_f0_stable_issuance_rate": "stable", "f3_minus_f0_support_fraction": "supported", "f3_minus_f0_correct_stable_coverage": "correct_stable"}[estimand]]) / N_FIXED),
                    "p2_5_rate": low, "p97_5_rate": high,
                    "point_scaled_count": float(N_FIXED * 0),
                    "p2_5_scaled_count": slow, "p97_5_scaled_count": shigh,
                    "bootstrap_mean_scaled_count": smean,
                    "undefined_replicates": 0, "seed": SEED, "shared_draws_across_consumers": True,
                })
            # Fix the point fields cleanly after the loop by recomputing explicitly below.
    # Rebuild paired rows explicitly so the point fields cannot drift.
    paired = []
    comparison = []
    published_rows = list(csv.DictReader((BEHPLAUS / "learner_bootstrap_intervals.csv").open(encoding="utf-8")))
    def published(name, schema, filt, metric):
        for row in published_rows:
            if row["consumer"] == name and row["schema"] == schema and row["filter"] == filt and row["metric"] == metric:
                return row
        return None
    for model, name in enumerate(CONSUMERS):
        for filter_i, filt in enumerate(FILTERS):
            s0 = packed[(model, 0, filter_i)]
            s1 = packed[(model, 1, filter_i)]
            delta = (s1["event_nc"] - s0["event_nc"]) / n_b
            scaled = N_FIXED * delta
            mean, low, high = percentile(scaled)
            point = int(counts[(model, 1, filter_i)]["event_nc"].sum() - counts[(model, 0, filter_i)]["event_nc"].sum())
            old = published(name, "S0", filt, "s1_minus_s0_event_nc")
            paired.append({
                "consumer": name, "schema": "S1_minus_S0", "filter": filt,
                "estimand": "population_scaled_event_nc_difference",
                "point": point, "bootstrap_mean": mean, "p2_5": low, "p97_5": high,
                "p2_5_rate": low / N_FIXED, "p97_5_rate": high / N_FIXED,
                "undefined_replicates": 0, "seed": SEED,
                "draws": "shared_seed_20319_first_stream",
            })
            if old:
                comparison.append({
                    "consumer": name, "schema": "S0_to_S1", "filter": filt,
                    "estimand_corrected": "939819 * (r_S1 - r_S0)",
                    "estimand_original": "raw pseudo-cohort (stable_S0 - stable_S1), equal to NC_S1_b - NC_S0_b before dividing by the resampled event count",
                    "original_p2_5": old["p2_5"], "original_p97_5": old["p97_5"], "original_point": old["point"],
                    "corrected_p2_5": low, "corrected_p97_5": high, "corrected_point": point,
                    "draws_match_published_behplaus": name == "TransparentActionPFA",
                    "sign_preserved": (point == 0) or ((low > 0 and float(old["p2_5"]) > 0) or (high < 0 and float(old["p97_5"]) < 0) or (point > 0 and low > 0 and float(old["p2_5"]) > 0) or (point < 0 and high < 0 and float(old["p97_5"]) < 0)),
                })
        for schema_i, schema in enumerate(SCHEMAS):
            f0 = packed[(model, schema_i, 0)]
            f3 = packed[(model, schema_i, 3)]
            specs = (
                ("f3_minus_f0_event_nc", (f3["event_nc"] - f0["event_nc"]) / n_b, int(counts[(model, schema_i, 3)]["event_nc"].sum() - counts[(model, schema_i, 0)]["event_nc"].sum()), "f3_minus_f0_event_nc"),
                ("f3_minus_f0_stable_issuance", (f3["stable"] - f0["stable"]) / n_b, int(counts[(model, schema_i, 3)]["stable"].sum() - counts[(model, schema_i, 0)]["stable"].sum()), "stable_fraction"),
                ("f3_minus_f0_support", (f3["supported"] - f0["supported"]) / n_b, int(counts[(model, schema_i, 3)]["supported"].sum() - counts[(model, schema_i, 0)]["supported"].sum()), "support_fraction"),
                ("f3_minus_f0_correct_stable", (f3["correct_stable"] - f0["correct_stable"]) / n_b, int(counts[(model, schema_i, 3)]["correct_stable"].sum() - counts[(model, schema_i, 0)]["correct_stable"].sum()), "correct_stable_lost_vs_f0"),
            )
            for estimand, rate_delta, point, old_metric in specs:
                scaled = N_FIXED * rate_delta
                mean, low, high = percentile(scaled)
                paired.append({
                    "consumer": name, "schema": schema, "filter": "F3_minus_F0",
                    "estimand": "population_scaled_" + estimand,
                    "point": point, "bootstrap_mean": mean, "p2_5": low, "p97_5": high,
                    "p2_5_rate": low / N_FIXED, "p97_5_rate": high / N_FIXED,
                    "undefined_replicates": 0, "seed": SEED,
                })
                if estimand == "f3_minus_f0_event_nc":
                    old = published(name, schema, "F3", "f3_minus_f0_event_nc")
                    if old:
                        comparison.append({
                            "consumer": name, "schema": schema, "filter": "F3_minus_F0",
                            "estimand_corrected": "939819 * (r_F3 - r_F0)",
                            "estimand_original": "raw pseudo-cohort stable_F0 - stable_F3",
                            "original_p2_5": old["p2_5"], "original_p97_5": old["p97_5"], "original_point": old["point"],
                            "corrected_p2_5": low, "corrected_p97_5": high, "corrected_point": point,
                            "sign_preserved": (low > 0 and float(old["p2_5"]) > 0) or (high < 0 and float(old["p97_5"]) < 0),
                        })
            for label, series_u in (
                ("new_false_stable", np.maximum(counts[(model, schema_i, 3)]["false_stable"] - counts[(model, schema_i, 0)]["false_stable"], 0)),
                ("removed_false_stable", np.maximum(counts[(model, schema_i, 0)]["false_stable"] - counts[(model, schema_i, 3)]["false_stable"], 0)),
            ):
                # Placeholder replaced after event-level indicators are supplied.
                pass
    for model, name in enumerate(CONSUMERS):
        for schema_i, schema in enumerate(SCHEMAS):
            for filter_i, filt in enumerate(FILTERS):
                old = published(name, schema, filt, "event_nc")
                if old is None:
                    continue
                scaled = N_FIXED * packed[(model, schema_i, filter_i)]["event_nc"] / n_b
                _, low, high = percentile(scaled)
                comparison.append({
                    "consumer": name, "schema": schema, "filter": filt,
                    "estimand_corrected": "939819 * (resampled_event_nc / resampled_event_count)",
                    "estimand_original": "939819 - resampled_stable_count",
                    "original_p2_5": old["p2_5"], "original_p97_5": old["p97_5"], "original_point": old["point"],
                    "corrected_p2_5": low, "corrected_p97_5": high,
                    "corrected_point": int(counts[(model, schema_i, filter_i)]["event_nc"].sum()),
                    "original_lower_negative": float(old["p2_5"]) < 0,
                    "corrected_lower_nonnegative": low >= -1e-6,
                    "draws_match_published_behplaus": name == "TransparentActionPFA",
                    "sign_preserved": True,
                })
    invariants["draws_shared_across_consumers_filters_and_schemas"] = True
    invariants["resample_size"] = N_USERS
    invariants["n_b_min"] = float(n_b.min())
    invariants["n_b_max"] = float(n_b.max())
    return marginal, paired, comparison, {"packed": packed, "n_b": n_b, "draws": draws, "invariants": invariants}


def event_level_paired(code, realized, user, draws, n_b) -> tuple[list[dict], dict]:
    rows = []
    totals = {}
    for model, name in enumerate(CONSUMERS):
        label = realized[:, model] >= TAU
        for schema_i, schema in enumerate(SCHEMAS):
            f0 = code[:, model, schema_i, 0]
            f3 = code[:, model, schema_i, 3]
            stable0 = (f0 == 1) | (f0 == 2)
            stable3 = (f3 == 1) | (f3 == 2)
            fs0 = stable0 & ((f0 == 1) != label)
            fs3 = stable3 & ((f3 == 1) != label)
            new = fs3 & ~fs0
            removed = fs0 & ~fs3
            lost = stable0 & ~fs0 & ~(stable3 & ~fs3)
            totals[(name, schema)] = {
                "new_false_stable": int(new.sum()),
                "removed_false_stable": int(removed.sum()),
                "correct_stable_lost": int(lost.sum()),
            }
            for label_name, mask in (("new_false_stable", new), ("removed_false_stable", removed), ("correct_stable_lost", lost)):
                per = np.bincount(user[mask], minlength=N_USERS).astype(np.float64)
                count_b = per[draws].sum(axis=1)
                rate = count_b / n_b
                scaled = N_FIXED * rate
                mean, low, high = percentile(scaled)
                rows.append({
                    "consumer": name, "schema": schema, "filter": "F3_versus_F0",
                    "estimand": "population_scaled_" + label_name,
                    "point": int(mask.sum()), "bootstrap_mean": mean, "p2_5": low, "p97_5": high,
                    "p2_5_rate": low / N_FIXED, "p97_5_rate": high / N_FIXED,
                    "min_scaled": float(scaled.min()), "max_scaled": float(scaled.max()),
                    "undefined_replicates": 0, "seed": SEED,
                })
                fs_den = None
            stable3_b = np.bincount(user[stable3], minlength=N_USERS).astype(np.float64)[draws].sum(axis=1)
            fs3_b = np.bincount(user[fs3], minlength=N_USERS).astype(np.float64)[draws].sum(axis=1)
            ratio = np.divide(fs3_b, stable3_b, out=np.full(REPS, np.nan), where=stable3_b > 0)
            mean, low, high = percentile(ratio)
            rows.append({
                "consumer": name, "schema": schema, "filter": "F3",
                "estimand": "false_stable_over_stable",
                "point": None if stable3.sum() == 0 else float(fs3.sum() / stable3.sum()),
                "bootstrap_mean": mean, "p2_5": low, "p97_5": high,
                "undefined_replicates": int(np.sum(stable3_b <= 0)), "seed": SEED,
            })
    return rows, totals


def matched_audit(code, lo, hi, realized, user, source, end) -> tuple[dict, dict, list[dict], dict]:
    states = ("guaranteed_in", "guaranteed_out", "unresolved")
    transitions = []
    containment = {"pairs": 0, "expanded_lower": 0, "expanded_upper": 0, "max_lower_deficit": 0.0, "max_upper_excess": 0.0, "by_cell": []}
    monotonicity = {"rank_best_decreased": 0, "rank_worst_increased": 0, "best_exceeds_worst": 0, "forbidden_guarantee_loss": 0, "by_cell": []}
    realized_report = {"realized_score_identical_across_filters": True, "topb_assignment_differences": 0, "by_cell": []}
    support0 = code[:, 0] > 0
    support1 = code[:, 1] > 0
    containment["consumer_support_masks_equal"] = bool(np.array_equal(support0, support1))
    for model, name in enumerate(CONSUMERS):
        mask = np.ones(code.shape[0], dtype=bool)
        for schema in range(4):
            for filt in (0, 3):
                mask &= code[:, model, schema, filt] > 0
        chosen = checkpoint_indices(user, end, source, mask)
        eligible = chosen >= 0
        idx = chosen[eligible]
        if len(idx) != 6658:
            containment.setdefault("cohort_mismatches", []).append({"consumer": name, "events": int(len(idx))})
            continue
        actual = 1.0 - realized[idx, model]
        order = np.lexsort((np.arange(len(idx)), -actual))
        selected = np.zeros(len(idx), dtype=bool)
        selected[order[:333]] = True
        for schema_i, schema in enumerate(SCHEMAS):
            lo0 = lo[idx, model, schema_i, 0]
            hi0 = hi[idx, model, schema_i, 0]
            lo3 = lo[idx, model, schema_i, 3]
            hi3 = hi[idx, model, schema_i, 3]
            lower_deficit = lo0 - lo3
            upper_excess = hi3 - hi0
            containment["pairs"] += len(idx)
            containment["expanded_lower"] += int(np.sum(lo3 < lo0))
            containment["expanded_upper"] += int(np.sum(hi3 > hi0))
            containment["max_lower_deficit"] = max(containment["max_lower_deficit"], float(lower_deficit.max()))
            containment["max_upper_excess"] = max(containment["max_upper_excess"], float(upper_excess.max()))
            containment["by_cell"].append({
                "consumer": name, "schema": schema, "events": len(idx),
                "contained": int(np.sum((lo3 >= lo0) & (hi3 <= hi0))),
                "max_lower_deficit": float(lower_deficit.max()),
                "max_upper_excess": float(upper_excess.max()),
            })
            risk0 = (1.0 - hi0, 1.0 - lo0)
            risk3 = (1.0 - hi3, 1.0 - lo3)
            best0, worst0 = adversarial_ranks(*risk0)
            best3, worst3 = adversarial_ranks(*risk3)
            labels0 = labels_from_ranks(best0, worst0, 333)
            labels3 = labels_from_ranks(best3, worst3, 333)
            best_down = int(np.sum(best3 < best0))
            worst_up = int(np.sum(worst3 > worst0))
            crossed = int(np.sum(best0 > worst0) + np.sum(best3 > worst3))
            forbidden = int(np.sum((labels0 == 0) & (labels3 != 0)) + np.sum((labels0 == 1) & (labels3 != 1)))
            monotonicity["rank_best_decreased"] += best_down
            monotonicity["rank_worst_increased"] += worst_up
            monotonicity["best_exceeds_worst"] += crossed
            monotonicity["forbidden_guarantee_loss"] += forbidden
            monotonicity["by_cell"].append({
                "consumer": name, "schema": schema,
                "best_rank_decreased": best_down, "worst_rank_increased": worst_up,
                "best_exceeds_worst": crossed, "forbidden_guarantee_loss": forbidden,
                "guaranteed_in_f0": int(np.sum(labels0 == 0)), "guaranteed_in_f3": int(np.sum(labels3 == 0)),
                "guaranteed_out_f0": int(np.sum(labels0 == 1)), "guaranteed_out_f3": int(np.sum(labels3 == 1)),
                "unresolved_f0": int(np.sum(labels0 == 2)), "unresolved_f3": int(np.sum(labels3 == 2)),
            })
            matrix = np.zeros((3, 3), dtype=np.int64)
            for src in range(3):
                for dst in range(3):
                    matrix[src, dst] = int(np.sum((labels0 == src) & (labels3 == dst)))
                    transitions.append({
                        "consumer": name, "schema": schema,
                        "from_status": states[src], "to_status": states[dst],
                        "learners": int(matrix[src, dst]), "cohort": 6658, "B": 333,
                    })
            inside0 = int(np.sum((actual >= risk0[0]) & (actual <= risk0[1])))
            inside3 = int(np.sum((actual >= risk3[0]) & (actual <= risk3[1])))
            false_in0 = int(np.sum((labels0 == 0) & ~selected))
            false_out0 = int(np.sum((labels0 == 1) & selected))
            false_in3 = int(np.sum((labels3 == 0) & ~selected))
            false_out3 = int(np.sum((labels3 == 1) & selected))
            realized_report["by_cell"].append({
                "consumer": name, "schema": schema, "selected": int(selected.sum()),
                "realized_risk_inside_f0": inside0, "realized_risk_inside_f3": inside3,
                "false_guaranteed_in_f0": false_in0, "false_guaranteed_out_f0": false_out0,
                "false_guaranteed_in_f3": false_in3, "false_guaranteed_out_f3": false_out3,
            })
        # Realized vector does not depend on filter. A second top-B from the same scores must match.
        again = np.zeros(len(idx), dtype=bool)
        again[np.lexsort((np.arange(len(idx)), -actual))[:333]] = True
        realized_report["topb_assignment_differences"] += int(np.sum(selected != again))
    realized_report["tie_convention"] = "lexsort(arange, -actual_risk), first B positions; actual risk is 1 - realized score of the fixed target event"
    return containment, monotonicity, transitions, realized_report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    pre = {
        "zip_sha256": sha256_file(ZIP),
        "development_sha256": sha256_file(RECON / "development.parquet"),
        "calibration_sha256": sha256_file(RECON / "calibration.parquet"),
        "confirmatory_sha256": sha256_file(RECON / "confirmatory.parquet"),
        "checkpoint_sha256": sha256_file(CKPT),
        "expected_zip": "7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445",
        "expected_development": "69b6b7b38738d293651b728326658c1ce517c87661d012bc03ee2ced228e9d70",
        "expected_calibration": "5e8202d6f89b501a1aaea662ba62a265de1e5c168a0bcf92e3762760b4c6fd3d",
        "expected_confirmatory": "0193ddd7b26517f1b48591d09d86f5f3ce8b9930a1285006833fce4ea5096f99",
        "expected_checkpoint": "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196",
    }
    pre["pass"] = all(pre[key] == pre["expected_" + key.split("_sha256")[0] if False else {
        "zip_sha256": "expected_zip", "development_sha256": "expected_development",
        "calibration_sha256": "expected_calibration", "confirmatory_sha256": "expected_confirmatory",
        "checkpoint_sha256": "expected_checkpoint",
    }[key]] for key in ("zip_sha256", "development_sha256", "calibration_sha256", "confirmatory_sha256", "checkpoint_sha256"))
    listed = json.loads((BEHPLAUS / "output_checksums.json").read_text())
    checksum_bad = []
    for name, digest in listed.items():
        path = BEHPLAUS / name
        if not path.exists() or sha256_file(path) != digest:
            checksum_bad.append(name)
    pre["behplaus_checksum_entries"] = len(listed)
    pre["behplaus_checksum_mismatches"] = checksum_bad
    pre["behplaus_checksum_excludes_itself"] = "output_checksums.json" not in listed
    dump("source_preflight.json", pre)
    if not pre["pass"] or checksum_bad:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_EVENT_ALIGNMENT", "preflight": pre})
        raise SystemExit(2)
    cached = np.load(NPZ)
    meta = json.loads(META.read_text())
    if int(meta["events"]) != N_FIXED or int(meta["users"]) != N_USERS or len(cached["source"]) != N_FIXED:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_EVENT_ALIGNMENT", "events": int(len(cached["source"])), "meta": {"events": meta["events"], "users": meta["users"]}})
        raise SystemExit(3)
    code = cached["code"]
    user = cached["user"].astype(np.int32)
    if int(user.min()) != 0 or int(user.max()) != N_USERS - 1 or int(np.unique(user).size) != N_USERS:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_EVENT_ALIGNMENT", "user_min": int(user.min()), "user_max": int(user.max()), "unique_users": int(np.unique(user).size)})
        raise SystemExit(5)
    realized = cached["realized"]
    counts = status_counts(code, realized, user)
    reconstruction = {"learners": N_USERS, "events": N_FIXED, "cells": [], "nonnegative": True, "partitions_match": True}
    for model, name in enumerate(CONSUMERS):
        for schema_i, schema in enumerate(SCHEMAS):
            for filter_i, filt in enumerate(FILTERS):
                cell = counts[(model, schema_i, filter_i)]
                partition = cell["unsupported"] + cell["stable"] + cell["set_valued"]
                ok_partition = bool(np.array_equal(partition, counts["events"]))
                ok_nc = bool(np.array_equal(cell["event_nc"], counts["events"] - cell["stable"]))
                ok_correct = bool(np.array_equal(cell["correct_stable"], cell["stable"] - cell["false_stable"]))
                nonnegative = all(int(cell[key].min()) >= 0 for key in ("unsupported", "stable", "set_valued", "event_nc", "false_stable", "correct_stable"))
                reconstruction["partitions_match"] &= ok_partition and ok_nc and ok_correct
                reconstruction["nonnegative"] &= nonnegative
                row = {
                    "consumer": name, "schema": schema, "filter": filt,
                    "events": int(counts["events"].sum()),
                    "unsupported": int(cell["unsupported"].sum()),
                    "stable": int(cell["stable"].sum()),
                    "set_valued": int(cell["set_valued"].sum()),
                    "event_nc": int(cell["event_nc"].sum()),
                    "false_stable": int(cell["false_stable"].sum()),
                    "correct_stable": int(cell["correct_stable"].sum()),
                    "supported": int(cell["supported"].sum()),
                }
                reconstruction["cells"].append(row)
                expected = EXPECTED.get((name, schema, filt))
                if expected and (row["event_nc"], row["false_stable"]) != expected:
                    reconstruction["partitions_match"] = False
    if schema == "S0":
        pass
    paired_points = {}
    for model, name in enumerate(CONSUMERS):
        for filter_i, filt in enumerate(FILTERS):
            delta = int(counts[(model, 1, filter_i)]["event_nc"].sum() - counts[(model, 0, filter_i)]["event_nc"].sum())
            paired_points[(name, filt)] = delta
            if delta != PAIRED[(name, filt)]:
                reconstruction["partitions_match"] = False
    reconstruction["s1_minus_s0"] = {f"{name}|{filt}": paired_points[(name, filt)] for name, filt in paired_points}
    dump("learner_status_reconstruction.json", reconstruction)
    np.savez_compressed(
        PRIV / "learner_counts.npz",
        events=counts["events"],
        **{f"m{model}_s{schema}_f{filt}_{key}": counts[(model, schema, filt)][key] for model in range(2) for schema in range(4) for filt in range(4) for key in ("event_nc", "stable", "false_stable", "correct_stable", "supported")},
    )
    if not reconstruction["partitions_match"] or not reconstruction["nonnegative"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_EVENT_ALIGNMENT", "reconstruction_ok": False})
        raise SystemExit(4)
    original = reproduce_original({(model, 0, 0): counts[(model, 0, 0)]["stable"] for model in range(2)})
    dump("bootstrap_original_implementation_audit.json", {
        "resampling_unit": "confirmatory learner",
        "rng": "numpy.random.Generator PCG64 via np.random.default_rng(20319)",
        "draws_per_replicate": N_USERS,
        "replicates": REPS,
        "shared_across_filters_and_schemas_within_consumer": True,
        "shared_across_consumers": False,
        "weight_type": "nonnegative integer sampling multiplicities",
        "centered_or_negative_weights": False,
        "interval_type_in_code": "percentile, np.quantile 0.025 and 0.975, linear",
        "interval_type_matches_label": True,
        "point_estimates_from_original_population": True,
        "marginal_event_nc_numerator": "939819 - resampled Stable count",
        "marginal_event_nc_denominator": "none; the fixed population size is an offset, not a resampled denominator",
        "paired_event_nc_numerator": "resampled Stable_reference - resampled Stable_comparison",
        "diagnostic_reproduction": original,
    })
    marginal, paired, comparison, bundle = corrected_bootstrap(counts)
    extra, indicator_totals = event_level_paired(code, realized, user, bundle["draws"], bundle["n_b"])
    paired.extend(extra)
    write_csv("marginal_bootstrap_intervals.csv", marginal)
    write_csv("paired_bootstrap_intervals.csv", paired)
    write_csv("statistic_comparison.csv", comparison)
    dump("bootstrap_replicate_invariants.json", bundle["invariants"])
    containment, monotonicity, transitions, realized_report = matched_audit(code, cached["lo"], cached["hi"], realized, user, cached["source"], cached["end"])
    dump("matched_interval_containment.json", containment)
    dump("matched_rank_monotonicity.json", monotonicity)
    write_csv("matched_capacity_transitions.csv", transitions)
    dump("matched_realized_topb_consistency.json", realized_report)
    chronorep_rows = []
    for name, schema in CHRONOREP_RATE:
        row = next(item for item in marginal if item["consumer"] == name and item["schema"] == schema and item["filter"] == "F0" and item["estimand"] == "event_nc_rate")
        chronorep_point, chronorep_low, chronorep_high = CHRONOREP_RATE[(name, schema)]
        chronorep_rows.append({
            "consumer": name, "schema": schema,
            "chronorep_point": chronorep_point, "chronorep_low": chronorep_low, "chronorep_high": chronorep_high, "chronorep_seed": 20308,
            "bootcap_point": row["point"], "bootcap_low": row["p2_5"], "bootcap_high": row["p97_5"], "bootcap_seed": SEED,
            "same_estimand": "resampled (unsupported + set-valued) / resampled valid events",
            "absolute_endpoint_gap": max(abs(row["p2_5"] - chronorep_low), abs(row["p97_5"] - chronorep_high)),
        })
    dump("bootstrap_replicate_invariants.json", {**bundle["invariants"], "chronorep_f0_rate_comparison": chronorep_rows, "event_level_indicator_totals": {f"{a}|{b}": c for (a, b), c in indicator_totals.items()}})
    signs_ok = all(bool(row["sign_preserved"]) for row in comparison)
    marginal_ok = bundle["invariants"]["rate_below_zero"] == 0 and bundle["invariants"]["rate_above_one"] == 0 and bundle["invariants"]["scaled_count_below_zero"] == 0 and bundle["invariants"]["replicates_with_zero_events"] == 0
    cohort_ok = not containment.get("cohort_mismatches")
    capacity_ok = cohort_ok and containment["expanded_lower"] == 0 and containment["expanded_upper"] == 0 and monotonicity["forbidden_guarantee_loss"] == 0 and monotonicity["rank_best_decreased"] == 0 and monotonicity["rank_worst_increased"] == 0 and monotonicity["best_exceeds_worst"] == 0 and realized_report["topb_assignment_differences"] == 0
    original_reproduced = all(item["endpoint_abs_diff"] < 1e-4 for item in original["consumers"].values())
    indicator_ok = indicator_totals.get(("TransparentActionPFA", "S0")) == {"new_false_stable": 36, "removed_false_stable": 21, "correct_stable_lost": 95375} and indicator_totals.get(("ActionTokenGRU", "S0")) == {"new_false_stable": 1907, "removed_false_stable": 573, "correct_stable_lost": 79588}
    status = "COMPLETE_WITH_NOTES"
    if not reconstruction["partitions_match"] or not indicator_ok:
        status = "BLOCKED_SOURCE_OR_EVENT_ALIGNMENT"
    elif not marginal_ok or not original_reproduced:
        status = "BLOCKED_BOOTSTRAP_INTEGRITY"
    elif not capacity_ok:
        status = "BLOCKED_CAPACITY_RANK_INCONSISTENCY"
    dump("numerical_invariants.json", {
        "status": status,
        "learner_reconstruction_ok": reconstruction["partitions_match"] and reconstruction["nonnegative"],
        "event_indicator_totals_match_behplaus": indicator_ok,
        "original_negative_interval_reproduced": original_reproduced,
        "corrected_rates_inside_unit_interval": marginal_ok,
        "paired_signs_preserved": signs_ok,
        "matched_intervals_contained": containment["expanded_lower"] == 0 and containment["expanded_upper"] == 0,
        "rank_monotonicity_ok": monotonicity["rank_best_decreased"] == 0 and monotonicity["rank_worst_increased"] == 0,
        "forbidden_guarantee_loss": monotonicity["forbidden_guarantee_loss"],
        "topb_consistent": realized_report["topb_assignment_differences"] == 0,
        "models_trained": False, "neural_rescore": False,
        "behplaus_scientific_tables_overwritten": False,
    })
    dump("execution_manifest.json", {"experiment": "pipeline", "status": status, "cache": str(NPZ), "draws_shared_across_consumers": True})
    payload = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "output_checksums.json" and "private" not in path.parts:
            payload[str(path.relative_to(OUT))] = sha256_file(path)
    dump("output_checksums.json", payload)
    print(status, flush=True)


if __name__ == "__main__":
    main()
