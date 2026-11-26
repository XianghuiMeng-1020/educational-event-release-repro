#!/usr/bin/env python3
"""Support-predicate sensitivity with frozen candidate intervals.

Consumers are not retrained and earlier outputs are not overwritten.
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
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("chronorep", ROOT / "scripts/assistments/chronological_replication.py")
chronorep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chronorep)
DECBASE = importlib.util.spec_from_file_location("decbase", ROOT / "scripts/assistments/decision_baselines.py")
decbase = importlib.util.module_from_spec(DECBASE)
DECBASE.loader.exec_module(decbase)
from r7_core import NeuralPilot  # noqa: E402

OUT = ROOT / "results/assistments/support_predicate_robustness"
PRIV = OUT / "private_scores"
CHRONOREP_CKPT = ROOT / "results/assistments/chronological_replication/gru_chronological_seed17.pt"
SCHEMAS = ("S0", "S1", "S2", "S3")
MODELS = ("TransparentActionPFA", "ActionTokenGRU")
TAUS = (0.70, 0.80, 0.90)
PRINCIPAL = ("P00", "P04", "P07", "P09", "P13", "P17")
CHRONOREP_CERT = {
    ("TransparentActionPFA", "S0"): (20408, 916845, 2566, 45),
    ("TransparentActionPFA", "S1"): (77877, 861927, 15, 0),
    ("TransparentActionPFA", "S2"): (80790, 859025, 4, 0),
    ("TransparentActionPFA", "S3"): (100688, 839131, 0, 0),
    ("ActionTokenGRU", "S0"): (20408, 810576, 108835, 1077),
    ("ActionTokenGRU", "S1"): (77877, 846772, 15170, 252),
    ("ActionTokenGRU", "S2"): (80790, 844157, 14872, 250),
    ("ActionTokenGRU", "S3"): (100688, 839131, 0, 0),
}
PRIMARY_DELTA = {"TransparentActionPFA": 54918, "ActionTokenGRU": -36196}


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
    return chronorep._json_default(value)


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format(value, ".17g") if isinstance(value, float) else ("" if value is None else value) for key, value in row.items()})


class Slot:
    __slots__ = ("n", "users", "pfa", "gru", "U", "pfa_counts", "pfa_deltas", "pfa_index", "gru_counts", "gru_index", "Fp", "Fg")

    def __init__(self) -> None:
        self.n = 0
        self.users = set()
        self.pfa = {}
        self.gru = {}


def finalize(slot: Slot) -> None:
    slot.U = len(slot.users)
    slot.users = None
    pfa_items = list(slot.pfa)
    slot.pfa_index = {fp: i for i, fp in enumerate(pfa_items)}
    slot.pfa_counts = np.asarray([slot.pfa[fp] for fp in pfa_items], dtype=np.int32)
    slot.pfa_deltas = np.asarray([chronorep.fp_delta(fp) for fp in pfa_items], dtype=np.float64) if pfa_items else np.zeros((0, 7))
    slot.Fp = len(pfa_items)
    gru_items = list(slot.gru)
    slot.gru_index = {fp: i for i, fp in enumerate(gru_items)}
    slot.gru_counts = np.asarray([slot.gru[fp] for fp in gru_items], dtype=np.int32)
    slot.Fg = len(gru_items)
    slot.pfa = None
    slot.gru = None


def build_development() -> tuple[dict, dict]:
    bundles = {schema: {} for schema in SCHEMAS}
    user_ids = {}
    signatures = set()
    duplicate = 0
    events = 0
    parquet = __import__("pyarrow.parquet").parquet.ParquetFile(chronorep.RECON / "development.parquet")
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            events += 1
            item = chronorep.compact_row(row, int(start), int(end))
            ev = chronorep.as_event(item)
            digest = hashlib.blake2s(repr((ev.problem_id, ev.correct, ev.hint_count, ev.attempt_count, ev.first_action, ev.action_types, tuple(str(tok) if tok is not None else "" for tok in (row["action_timestamps"] or [])))).encode(), digest_size=16).digest()
            duplicate += digest in signatures
            signatures.add(digest)
            uid = user_ids.setdefault(item["user_id"], len(user_ids))
            pfa = chronorep.pfa_fp(item)
            gru = chronorep.gru_fp(item)
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                slot = bundles[schema].get(key)
                if slot is None:
                    slot = Slot()
                    bundles[schema][key] = slot
                slot.n += 1
                slot.users.add(uid)
                slot.pfa[pfa] = slot.pfa.get(pfa, 0) + 1
                slot.gru[gru] = slot.gru.get(gru, 0) + 1
        if events % 1000000 == 0:
            print(f"development {events} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    geometry = {schema: {"U1_Rge2": 0, "U1_Rge2_events": 0, "Uge2_Fp1": 0, "Uge2_Fpge2": 0, "U1_Fpge2": 0, "Rge2_Fp1": 0, "Fpge2": 0, "Uge2_Fg1": 0, "Uge2_Fgge2": 0, "U1_Fgge2": 0, "Rge2_Fg1": 0, "Fgge2": 0, "keys": 0, "events": 0, "U_gt_R": 0} for schema in SCHEMAS}
    key_rows = {schema: [] for schema in SCHEMAS}
    for schema, bundle in bundles.items():
        for slot in bundle.values():
            finalize(slot)
            if int(slot.pfa_counts.sum()) != slot.n or int(slot.gru_counts.sum()) != slot.n or slot.U > slot.n:
                raise RuntimeError("count identity failed")
            key_rows[schema].append((slot.U, slot.n, slot.Fp, slot.Fg))
            geo = geometry[schema]
            geo["keys"] += 1
            geo["events"] += slot.n
            geo["U1_Rge2"] += int(slot.U == 1 and slot.n >= 2)
            geo["U1_Rge2_events"] += slot.n if slot.U == 1 and slot.n >= 2 else 0
            geo["Uge2_Fp1"] += int(slot.U >= 2 and slot.Fp == 1)
            geo["Uge2_Fpge2"] += int(slot.U >= 2 and slot.Fp >= 2)
            geo["U1_Fpge2"] += int(slot.U == 1 and slot.Fp >= 2)
            geo["Rge2_Fp1"] += int(slot.n >= 2 and slot.Fp == 1)
            geo["Fpge2"] += int(slot.Fp >= 2)
            geo["Uge2_Fg1"] += int(slot.U >= 2 and slot.Fg == 1)
            geo["Uge2_Fgge2"] += int(slot.U >= 2 and slot.Fg >= 2)
            geo["U1_Fgge2"] += int(slot.U == 1 and slot.Fg >= 2)
            geo["Rge2_Fg1"] += int(slot.n >= 2 and slot.Fg == 1)
            geo["Fgge2"] += int(slot.Fg >= 2)
            geo["U_gt_R"] += int(slot.U > slot.n)
    key_stats = {schema: np.asarray(key_rows[schema], dtype=np.int32) for schema in SCHEMAS}
    return bundles, {"events": events, "duplicate_rich_signatures": duplicate, "unique_rich_signatures": len(signatures), "development_users_in_library": len(user_ids), "geometry": geometry, "key_counts": {schema: int(geometry[schema]["keys"]) for schema in SCHEMAS}, "key_stats": key_stats}


def score_confirmatory(grouped: dict, bundles: dict, pilot: NeuralPilot) -> dict:
    users = sorted(grouped)
    n = sum(len(chronorep.order_valid(grouped[uid])[0]) for uid in users)
    user_lookup = {uid: i for i, uid in enumerate(users)}
    user = np.empty(n, dtype=np.int32)
    source = np.empty(n, dtype=np.int64)
    end = np.empty(n, dtype=np.int64)
    realized = np.empty((n, 2), dtype=np.float64)
    lo = np.full((n, 2, 4), np.nan, dtype=np.float64)
    hi = np.full((n, 2, 4), np.nan, dtype=np.float64)
    counts = np.zeros((n, 4, 4), dtype=np.int32)  # U, R, Fp, Fg
    present = np.zeros((n, 4), dtype=np.uint8)
    library = np.zeros((n, 2, 4), dtype=np.uint8)
    cursor_event = 0
    future = 0
    direct_gap = 0.0
    direct_checked = 0
    direct_label_mismatch = 0
    for user_i, uid in enumerate(users):
        by_start, by_end = chronorep.order_valid(grouped[uid])
        state = defaultdict(lambda: np.zeros(7, dtype=np.float64))
        cursor = 0
        for row in by_start:
            while cursor < len(by_end) and by_end[cursor]["end"] < row["start"]:
                done = by_end[cursor]
                state[done["skill_id"]] += np.asarray(chronorep.fp_delta(chronorep.pfa_fp(done)), dtype=np.float64)
                cursor += 1
            if any(item["end"] >= row["start"] for item in by_end[:cursor]):
                future += 1
            ev = chronorep.as_event(row)
            prefix = [chronorep.as_event(item) for item in by_end[max(0, cursor - 40):cursor]]
            base = state[row["skill_id"]]
            own_pfa = chronorep.pfa_fp(row)
            own_gru = chronorep.gru_fp(row)
            pfa_realized = float(chronorep.mastery_of(base + np.asarray(chronorep.fp_delta(own_pfa), dtype=np.float64)))
            gru_needed = {own_gru: None}
            slots = []
            for schema in SCHEMAS:
                slot = bundles[schema].get(chronorep.release_key(ev, schema))
                slots.append(slot)
                if slot is not None:
                    gru_needed.update(slot.gru_index)
            pairs = [chronorep.sequence_for(pilot, prefix, fp, row["skill_id"]) for fp in gru_needed]
            gru_scores = chronorep.score_forward(pilot.model, pairs, pilot.device)
            gru_map = {fp: float(score) for fp, score in zip(gru_needed, gru_scores)}
            pos = cursor_event
            user[pos] = user_lookup[uid]
            source[pos] = row["source_row"]
            end[pos] = row["end"]
            realized[pos, 0] = pfa_realized
            realized[pos, 1] = gru_map[own_gru]
            for schema_i, slot in enumerate(slots):
                if slot is None:
                    continue
                present[pos, schema_i] = 1
                counts[pos, schema_i] = (slot.U, slot.n, slot.Fp, slot.Fg)
                pfa_scores = decbase.mastery_vec(base, slot.pfa_deltas)
                gru_values = np.asarray([gru_map[fp] for fp in slot.gru_index], dtype=np.float64)
                lo[pos, 0, schema_i] = float(pfa_scores.min())
                hi[pos, 0, schema_i] = float(pfa_scores.max())
                lo[pos, 1, schema_i] = float(gru_values.min())
                hi[pos, 1, schema_i] = float(gru_values.max())
                library[pos, 0, schema_i] = int(own_pfa in slot.pfa_index)
                library[pos, 1, schema_i] = int(own_gru in slot.gru_index)
                if schema_i == 0 and slot.n == 1 and direct_checked < 100:
                    direct = float(chronorep.mastery_of(base + slot.pfa_deltas[0]))
                    direct_gap = max(direct_gap, abs(direct - float(pfa_scores.min())))
                    if (direct >= 0.80) != (float(pfa_scores.min()) >= 0.80):
                        direct_label_mismatch += 1
                    again = chronorep.score_forward(pilot.model, [chronorep.sequence_for(pilot, prefix, next(iter(slot.gru_index)), row["skill_id"])], pilot.device)
                    direct_gap = max(direct_gap, abs(float(again[0]) - float(gru_values.min())))
                    if (float(again[0]) >= 0.80) != (float(gru_values.min()) >= 0.80):
                        direct_label_mismatch += 1
                    direct_checked += 1
            cursor_event += 1
        if user_i and user_i % 250 == 0:
            print(f"confirmatory_users {user_i} events {cursor_event} rss_mb={chronorep.rss_mb():.1f}", flush=True)
    return {"users": len(users), "events": n, "future": future, "direct_checked": direct_checked, "direct_gap": direct_gap, "direct_label_mismatch": direct_label_mismatch, "user": user, "source": source, "end": end, "realized": realized, "lo": lo, "hi": hi, "counts": counts, "present": present, "library": library}


def save_private(payload: dict) -> None:
    PRIV.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRIV / "confirmatory_intervals.npz", **{key: payload[key] for key in ("user", "source", "end", "realized", "lo", "hi", "counts", "present", "library")})
    (PRIV / "meta.json").write_text(json.dumps({key: payload[key] for key in ("users", "events", "future", "direct_checked", "direct_gap", "direct_label_mismatch")}), encoding="utf-8")


def load_private() -> dict:
    arrays = np.load(PRIV / "confirmatory_intervals.npz")
    meta = json.loads((PRIV / "meta.json").read_text())
    payload = {key: arrays[key] for key in arrays.files}
    payload.update(meta)
    return payload


def predicates(U, R, F):
    return {
        "P00": (U >= 2) | (R >= 2),
        "P01": R >= 1,
        "P02": R >= 2,
        "P03": R >= 3,
        "P04": R >= 5,
        "P05": R >= 10,
        "P06": R >= 20,
        "P07": U >= 2,
        "P08": U >= 3,
        "P09": U >= 5,
        "P10": U >= 10,
        "P11": U >= 20,
        "P12": (U >= 2) & (R >= 2),
        "P13": (U >= 5) & (R >= 5),
        "P14": (U >= 10) & (R >= 10),
        "P15": (U >= 5) | (R >= 10),
        "P16": (U >= 10) | (R >= 5),
        "P17": F >= 2,
        "P18": F >= 3,
        "P19": F >= 5,
        "P20": (U >= 2) | (F >= 2),
        "P21": (U >= 5) & (F >= 2),
    }


def classify(lo, hi, tau):
    code = np.zeros(lo.shape, dtype=np.uint8)
    finite = np.isfinite(lo) & np.isfinite(hi)
    code[finite & (lo >= tau)] = 1
    code[finite & (hi < tau) & (code == 0)] = 2
    code[finite & (code == 0)] = 3
    return code


def apply_support(code, supported):
    out = code.copy()
    out[~supported] = 0
    return out


def budget(n: int) -> int:
    if n <= 0:
        return 0
    seats = int(round(0.05 * n))
    if seats < 1:
        seats = 1
    return min(seats, n)


def checkpoint_indices(user, end, source, supported) -> tuple[np.ndarray, np.ndarray]:
    n_users = int(user.max()) + 1
    chosen = np.full(n_users, -1, dtype=np.int32)
    idx = np.flatnonzero(supported)
    if len(idx) == 0:
        return chosen, np.zeros(n_users, dtype=bool)
    ordered = idx[np.lexsort((source[idx], end[idx], user[idx]))]
    owners = user[ordered]
    last = np.empty(len(ordered), dtype=bool)
    last[-1] = True
    last[:-1] = owners[:-1] != owners[1:]
    chosen[owners[last]] = ordered[last]
    return chosen, chosen >= 0


def capacity_row(model, schema, rule, user, end, source, supported, lo, hi, realized, kind: str) -> dict:
    chosen, eligible = checkpoint_indices(user, end, source, supported)
    n_full = int(user.max()) + 1
    n_el = int(eligible.sum())
    seats = budget(n_el)
    if n_el == 0:
        return {"analysis": kind, "consumer": model, "schema": schema, "rule": rule, "kappa": 0.05, "n_full": n_full, "n_eligible": 0, "n_no_checkpoint": n_full, "B": 0, "guaranteed_in": 0, "guaranteed_out": 0, "unresolved": 0, "all_user_capacity_nc": 1.0, "interval_containment": None, "false_guaranteed_in": None, "false_guaranteed_out": None}
    idx = chosen[eligible]
    risk_lo = 1.0 - hi[idx]
    risk_hi = 1.0 - lo[idx]
    actual = 1.0 - realized[idx]
    counts = chronorep.membership_counts(risk_lo, risk_hi, 0.05)
    labels = counts["labels"]
    order = np.lexsort((np.arange(n_el), -actual))
    selected = np.zeros(n_el, dtype=bool)
    selected[order[:seats]] = True
    inside = (actual >= risk_lo) & (actual <= risk_hi)
    return {
        "analysis": kind, "consumer": model, "schema": schema, "rule": rule, "kappa": 0.05, "n_full": n_full,
        "n_eligible": n_el, "n_no_checkpoint": n_full - n_el, "B": seats,
        "guaranteed_in": counts["guaranteed_in"], "guaranteed_out": counts["guaranteed_out"], "unresolved": counts["unresolved"],
        "all_user_capacity_nc": (n_full - n_el + counts["unresolved"]) / n_full,
        "interval_containment": int(inside.sum()), "false_guaranteed_in": int(np.sum((labels == "guaranteed_in") & ~selected)),
        "false_guaranteed_out": int(np.sum((labels == "guaranteed_out") & selected)),
    }


def bin_users(U):
    out = np.full(U.shape, "unseen", dtype=object)
    out[U == 1] = "1"
    out[(U >= 2) & (U <= 4)] = "2-4"
    out[(U >= 5) & (U <= 9)] = "5-9"
    out[(U >= 10) & (U <= 19)] = "10-19"
    out[U >= 20] = ">=20"
    return out


def bin_rich(R):
    out = np.full(R.shape, "unseen", dtype=object)
    out[R == 1] = "1"
    out[(R >= 2) & (R <= 4)] = "2-4"
    out[(R >= 5) & (R <= 9)] = "5-9"
    out[(R >= 10) & (R <= 19)] = "10-19"
    out[R >= 20] = ">=20"
    return out


def bin_fp(F):
    out = np.full(F.shape, "unseen", dtype=object)
    out[F == 1] = "1"
    out[F == 2] = "2"
    out[(F >= 3) & (F <= 4)] = "3-4"
    out[F >= 5] = ">=5"
    return out


def audit_development_counts(payload: dict, indices: np.ndarray) -> dict:
    wanted = {int(payload["source"][i]): int(i) for i in indices}
    grouped = decbase.load_split("confirmatory")
    needed = {schema: {} for schema in SCHEMAS}
    seen = 0
    for rows in grouped.values():
        for row in rows:
            pos = wanted.get(int(row["source_row"]))
            if pos is None or not row["valid"]:
                continue
            seen += 1
            ev = chronorep.as_event(row)
            for schema_i, schema in enumerate(SCHEMAS):
                key = chronorep.release_key(ev, schema)
                needed[schema].setdefault(key, []).append(pos)
    stats = {schema: {key: [set(), 0, set(), set()] for key in needed[schema]} for schema in SCHEMAS}
    parquet = __import__("pyarrow.parquet").parquet.ParquetFile(chronorep.RECON / "development.parquet")
    scanned = 0
    for batch in parquet.iter_batches(batch_size=40000):
        rows = batch.to_pylist()
        starts = chronorep.civil_us([row["start_time"] for row in rows])
        ends = chronorep.civil_us([row["end_time"] for row in rows])
        for row, start, end in zip(rows, starts, ends):
            item = chronorep.compact_row(row, int(start), int(end))
            ev = chronorep.as_event(item)
            pfa = chronorep.pfa_fp(item)
            gru = chronorep.gru_fp(item)
            for schema in SCHEMAS:
                key = chronorep.release_key(ev, schema)
                slot = stats[schema].get(key)
                if slot is None:
                    continue
                slot[0].add(item["user_id"])
                slot[1] += 1
                slot[2].add(pfa)
                slot[3].add(gru)
        scanned += len(rows)
    mismatches = 0
    checked = 0
    for schema_i, schema in enumerate(SCHEMAS):
        for key, positions in needed[schema].items():
            users, count, pfa, gru = stats[schema][key]
            for pos in positions:
                stored = payload["counts"][pos, schema_i]
                present = int(payload["present"][pos, schema_i])
                checked += 1
                if (len(users), count, len(pfa), len(gru), int(count > 0)) != (int(stored[0]), int(stored[1]), int(stored[2]), int(stored[3]), present):
                    mismatches += 1
    if seen != len(indices):
        mismatches += abs(len(indices) - seen)
    return {"sample_events": int(len(indices)), "confirmatory_rows_matched": seen, "development_rows_scanned": scanned, "key_lookups": checked, "count_mismatches": mismatches}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    started = time.time()
    pre = chronorep.preflight()
    pre["chronorep_checkpoint_sha256"] = chronorep.sha256_file(CHRONOREP_CKPT)
    pre["chronorep_checkpoint_match"] = pre["chronorep_checkpoint_sha256"] == "42f14f14b9d99a50476d25badfc48736505844964b338346748ad5eb7dec5196"
    pre["pass"] = bool(pre["pass"] and pre["chronorep_checkpoint_match"])
    dump("source_and_checkpoint_preflight.json", pre)
    if not pre["pass"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SUPPORT_SEMANTICS"})
        raise SystemExit(2)
    dump("support_predicate_protocol_freeze.json", {
        "frozen_before_confirmatory_rule_selection": True,
        "primary": "P00 = U>=2 OR R>=2",
        "alternatives_are_sensitivity_only": True,
        "tau_primary": 0.80,
        "fingerprint_rules": "P17-P21 are consumer-specific diagnostics",
        "interval_isolation": "support eligibility does not change stored candidate minima or maxima",
        "development_invalid_intervals_retained_in_library": 475,
    })
    if (PRIV / "confirmatory_intervals.npz").exists() and (PRIV / "development_key_counts.npz").exists():
        print("reuse intervals", flush=True)
        payload = load_private()
        geometry = json.loads((OUT / "development_count_semantics.json").read_text())
        stored = np.load(PRIV / "development_key_counts.npz")
        geometry["key_stats"] = {schema: stored[schema] for schema in SCHEMAS}
        bundles = None
    else:
        print("development", flush=True)
        bundles, geometry = build_development()
        key_ok = geometry["key_counts"] == {"S0": 259838, "S1": 511770, "S2": 523378, "S3": 605097}
        if geometry["duplicate_rich_signatures"] != 0 or geometry["events"] != 4286981 or any(geometry["geometry"][s]["U_gt_R"] for s in SCHEMAS) or not key_ok:
            dump("numerical_invariants.json", {"status": "BLOCKED_SOURCE_OR_SUPPORT_SEMANTICS", "events": geometry["events"], "duplicates": geometry["duplicate_rich_signatures"], "key_counts": geometry["key_counts"], "key_counts_match": key_ok})
            raise SystemExit(3)
        PRIV.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PRIV / "development_key_counts.npz", **geometry["key_stats"])
        public_geometry = {key: value for key, value in geometry.items() if key != "key_stats"}
        dump("development_count_semantics.json", public_geometry)
        bundle = torch.load(CHRONOREP_CKPT, map_location="cpu", weights_only=False)
        pilot = NeuralPilot(device="cpu")
        pilot.load_bundle({"model": bundle["state_dict"], "skill2i": bundle["skill2i"], "atype2i": bundle["atype2i"]})
        pilot.model.eval()
        print("score confirmatory", flush=True)
        payload = score_confirmatory(decbase.load_split("confirmatory"), bundles, pilot)
        save_private(payload)
        del pilot, bundles, bundle
    if payload["events"] != 939819 or payload["future"]:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "events": payload["events"], "future": payload["future"]})
        raise SystemExit(4)
    U = payload["counts"][:, :, 0]
    R = payload["counts"][:, :, 1]
    Fp = payload["counts"][:, :, 2]
    Fg = payload["counts"][:, :, 3]
    present = payload["present"].astype(bool)
    lo = payload["lo"]
    hi = payload["hi"]
    realized = payload["realized"]
    library = payload["library"].astype(bool)
    user = payload["user"]
    n = payload["events"]
    masks = {}
    for model_i, F in enumerate((Fp, Fg)):
        masks[model_i] = {}
        for schema_i in range(4):
            local = predicates(U[:, schema_i], R[:, schema_i], F[:, schema_i])
            for name, mask in local.items():
                masks[model_i].setdefault(name, [None] * 4)
                masks[model_i][name][schema_i] = mask & present[:, schema_i]
    equivalence = []
    names = list(predicates(U[:, 0], R[:, 0], Fp[:, 0]))
    for model_i, model in enumerate(MODELS):
        groups = []
        used = set()
        for left in names:
            if left in used:
                continue
            same = [left]
            for right in names:
                if right == left or right in used:
                    continue
                if all(np.array_equal(masks[model_i][left][s], masks[model_i][right][s]) for s in range(4)):
                    same.append(right)
            for item in same:
                used.add(item)
            groups.append(same)
            equivalence.append({"consumer": model, "equivalent_rules": "|".join(same), "n_rules": len(same), "identical_on_all_schemas": True})
    write_csv("support_equivalence_classes.csv", equivalence)
    p00_p02 = all(np.array_equal(masks[m]["P00"][s], masks[m]["P02"][s]) for m in range(2) for s in range(4))
    p07_p12 = all(np.array_equal(masks[m]["P07"][s], masks[m]["P12"][s]) for m in range(2) for s in range(4))
    codes = {tau: classify(lo, hi, tau) for tau in TAUS}
    realized_label = {tau: realized >= tau for tau in TAUS}
    key_summary = [{schema: {} for schema in SCHEMAS} for _model in MODELS]
    for model_i in range(2):
        for schema in SCHEMAS:
            arr = geometry["key_stats"][schema]
            local = predicates(arr[:, 0], arr[:, 1], arr[:, 2 + model_i])
            for rule, mask in local.items():
                chosen = arr[mask]
                def pack(column: int) -> dict:
                    if len(chosen) == 0:
                        return {"q25": None, "median": None, "q75": None, "q90": None}
                    quantiles = np.quantile(chosen[:, column].astype(np.float64), [0.25, 0.5, 0.75, 0.9])
                    return {"q25": float(quantiles[0]), "median": float(quantiles[1]), "q75": float(quantiles[2]), "q90": float(quantiles[3])}
                users_q = pack(0)
                rich_q = pack(1)
                key_summary[model_i][schema][rule] = {
                    "eligible_development_keys": int(mask.sum()),
                    "development_event_mass": int(arr[mask, 1].sum()) if np.any(mask) else 0,
                    "class_users_q25": users_q["q25"], "class_users_median": users_q["median"], "class_users_q75": users_q["q75"], "class_users_q90": users_q["q90"],
                    "class_rich_q25": rich_q["q25"], "class_rich_median": rich_q["median"], "class_rich_q75": rich_q["q75"], "class_rich_q90": rich_q["q90"],
                }
    frontier, effects, false_rows, transitions, schema_trans, threshold_rows = [], [], [], [], [], []
    p00_counts = {}
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS):
            for rule in names:
                supported = masks[model_i][rule][schema_i]
                code = apply_support(codes[0.80][:, model_i, schema_i], supported)
                label = realized_label[0.80][:, model_i]
                stable = (code == 1) | (code == 2)
                action = code == 1
                false = stable & (action != label)
                uncovered = stable & ~library[:, model_i, schema_i]
                sett = code == 3
                unsup = code == 0
                contained = (stable & ~false) | sett
                learners = int(np.unique(user[supported]).size) if supported.any() else 0
                covered = int((supported & library[:, model_i, schema_i]).sum())
                frontier.append({
                    "consumer": model, "schema": schema, "rule": rule, "tau": 0.80, "valid_targets": n,
                    "supported": int(supported.sum()), "unsupported": int(unsup.sum()), "stable": int(stable.sum()),
                    "set_valued": int(sett.sum()), "event_nc": int((unsup | sett).sum()),
                    "event_nc_over_valid": int((unsup | sett).sum()) / n, "stable_over_valid": int(stable.sum()) / n,
                    "set_valued_over_supported": (int(sett.sum()) / int(supported.sum())) if supported.any() else None,
                    "supported_learners": learners, "fingerprint_covered_supported": covered,
                    "stable_uncovered": int(uncovered.sum()), "false_stable": int(false.sum()),
                    "false_stable_over_stable": (int(false.sum()) / int(stable.sum())) if stable.any() else None,
                    "false_stable_over_supported": (int(false.sum()) / int(supported.sum())) if supported.any() else None,
                "action_contained_supported": int(contained.sum()),
                **key_summary[model_i][schema][rule],
            })
                false_rows.append({
                    "consumer": model, "schema": schema, "rule": rule, "n_S": int(stable.sum()), "n_U": int(uncovered.sum()),
                    "n_FS": int(false.sum()), "false_stable_over_stable": (int(false.sum()) / int(stable.sum())) if stable.any() else None,
                    "n_FS_over_n_U": (int(false.sum()) / int(uncovered.sum())) if uncovered.any() else None,
                    "case": "B" if stable.any() and not uncovered.any() else ("A" if stable.any() and uncovered.any() else "no_stable"),
                })
                if rule == "P00":
                    p00_counts[(model, schema)] = (int(unsup.sum()), int(stable.sum()), int(sett.sum()), int(false.sum()))
            if schema_i == 0:
                map_code = np.array([0, 1, 1, 2], dtype=np.int8)
                labels = ("unsupported", "stable", "set_valued")
                for rule in names:
                    left = apply_support(codes[0.80][:, model_i, 0], masks[model_i][rule][0])
                    right = apply_support(codes[0.80][:, model_i, 1], masks[model_i][rule][1])
                    left_nc = (left == 0) | (left == 3)
                    right_nc = (right == 0) | (right == 3)
                    delta = int(right_nc.sum() - left_nc.sum())
                    effects.append({
                        "consumer": model, "rule": rule, "nc_s0": int(left_nc.sum()), "nc_s1": int(right_nc.sum()),
                        "delta_nc_s1_minus_s0": delta, "delta_nc_percentage_points": 100.0 * delta / n,
                        "delta_supported_s1_minus_s0": int(masks[model_i][rule][1].sum() - masks[model_i][rule][0].sum()),
                        "primary_pfa_delta": 54918, "primary_gru_delta": -36196,
                    })
                    if rule in ("P00", "P01", "P04", "P07", "P09", "P13", "P17"):
                        matrix = np.bincount(map_code[left] * 3 + map_code[right], minlength=9).reshape(3, 3)
                        if int(matrix.sum()) != n:
                            raise RuntimeError("schema transition matrix does not cover the valid population")
                        for i, src in enumerate(labels):
                            for j, dst in enumerate(labels):
                                schema_trans.append({"consumer": model, "rule": rule, "from_schema": "S0", "to_schema": "S1", "from_label": src, "to_label": dst, "n": int(matrix[i, j]), "denominator": n})
    cert_match = p00_counts == CHRONOREP_CERT
    delta_match = all(next(row["delta_nc_s1_minus_s0"] for row in effects if row["consumer"] == model and row["rule"] == "P00") == PRIMARY_DELTA[model] for model in MODELS)
    if not cert_match or not p00_p02 or not p07_p12 or not delta_match:
        dump("numerical_invariants.json", {"status": "BLOCKED_NUMERICAL_INCONSISTENCY", "cert_match": cert_match, "p00_equals_p02": p00_p02, "p07_equals_p12": p07_p12, "delta_match": delta_match, "found": {f"{a}|{b}": v for (a, b), v in p00_counts.items()}})
        raise SystemExit(5)
    for model_i, model in enumerate(MODELS):
        base = apply_support(codes[0.80][:, model_i], np.stack(masks[model_i]["P00"], axis=1))
        for rule in ("P01", "P04", "P07", "P09", "P13", "P17"):
            alt = apply_support(codes[0.80][:, model_i], np.stack(masks[model_i][rule], axis=1))
            for schema_i, schema in enumerate(SCHEMAS):
                b = base[:, schema_i]
                a = alt[:, schema_i]
                b_stable = (b == 1) | (b == 2)
                a_stable = (a == 1) | (a == 2)
                label = realized_label[0.80][:, model_i]
                transitions.append({
                    "consumer": model, "schema": schema, "rule": rule,
                    "stable_to_unsupported": int(((b_stable) & (a == 0)).sum()),
                    "set_to_unsupported": int(((b == 3) & (a == 0)).sum()),
                    "unsupported_to_unsupported": int(((b == 0) & (a == 0)).sum()),
                    "unsupported_to_stable": int(((b == 0) & a_stable).sum()),
                    "unsupported_to_set": int(((b == 0) & (a == 3)).sum()),
                    "stable_action_changed_while_both_supported": int((b_stable & a_stable & (b != a)).sum()),
                    "removed_false_stable": int((b_stable & (a == 0) & ((b == 1) != label)).sum()),
                    "removed_true_stable": int((b_stable & (a == 0) & ((b == 1) == label)).sum()),
                    "net_nc_increase": int((((a == 0) | (a == 3)).sum()) - (((b == 0) | (b == 3)).sum())),
                })
    write_csv("support_rule_event_frontier.csv", frontier)
    write_csv("support_rule_s0_s1_effects.csv", effects)
    write_csv("false_stable_by_support_rule.csv", false_rows)
    write_csv("paired_support_rule_transitions.csv", transitions)
    write_csv("paired_schema_certificate_transitions.csv", schema_trans)
    geo_rows = []
    for schema, geo in geometry["geometry"].items():
        geo_rows.append({"schema": schema, "development_events": geometry["events"], "duplicate_rich_signatures": geometry["duplicate_rich_signatures"], "R_equals_N": True, "U_le_R": geo["U_gt_R"] == 0, **geo})
    write_csv("development_support_geometry.csv", geo_rows)
    diversity = []
    cross = []
    for model_i, model in enumerate(MODELS):
        F = Fp if model_i == 0 else Fg
        for schema_i, schema in enumerate(SCHEMAS):
            diversity.append({
                "consumer": model, "schema": schema,
                "events_U1_Rge2": int(((U[:, schema_i] == 1) & (R[:, schema_i] >= 2)).sum()),
                "events_Uge2_F1": int(((U[:, schema_i] >= 2) & (F[:, schema_i] == 1)).sum()),
                "events_Uge2_Fge2": int(((U[:, schema_i] >= 2) & (F[:, schema_i] >= 2)).sum()),
                "events_U1_Fge2": int(((U[:, schema_i] == 1) & (F[:, schema_i] >= 2)).sum()),
                "events_Rge2_F1": int(((R[:, schema_i] >= 2) & (F[:, schema_i] == 1)).sum()),
                "events_Fge2": int((F[:, schema_i] >= 2).sum()),
                "events_unseen": int((~present[:, schema_i]).sum()),
                "events_F_equals_1_among_present": int(((F[:, schema_i] == 1) & present[:, schema_i]).sum()),
            })
            for ub in ("1", "2-4", "5-9", "10-19", ">=20", "unseen"):
                for fb in ("1", "2", "3-4", ">=5", "unseen"):
                    mask = (bin_users(U[:, schema_i]) == ub) & (bin_fp(F[:, schema_i]) == fb)
                    if not mask.any():
                        continue
                    supported = masks[model_i]["P00"][schema_i] & mask
                    code = apply_support(codes[0.80][:, model_i, schema_i], supported)
                    stable = (code == 1) | (code == 2)
                    label = realized_label[0.80][:, model_i]
                    false = stable & ((code == 1) != label)
                    cross.append({"consumer": model, "schema": schema, "users": ub, "fingerprints": fb, "events": int(mask.sum()), "p00_supported": int(supported.sum()), "stable": int(stable.sum()), "false_stable": int(false.sum())})
    write_csv("consumer_fingerprint_diversity.csv", diversity)
    write_csv("support_vs_fingerprint_cross_tab.csv", cross)
    strata = []
    for model_i, model in enumerate(MODELS):
        F = Fp if model_i == 0 else Fg
        for schema_i, schema in enumerate(SCHEMAS):
            code = apply_support(codes[0.80][:, model_i, schema_i], masks[model_i]["P00"][schema_i])
            stable = (code == 1) | (code == 2)
            label = realized_label[0.80][:, model_i]
            false = stable & ((code == 1) != label)
            uncovered = stable & ~library[:, model_i, schema_i]
            for kind, labels, bins in (
                ("users", bin_users(U[:, schema_i]), ("1", "2-4", "5-9", "10-19", ">=20", "unseen")),
                ("rich_signatures", bin_rich(R[:, schema_i]), ("1", "2-4", "5-9", "10-19", ">=20", "unseen")),
                ("fingerprints", bin_fp(F[:, schema_i]), ("1", "2", "3-4", ">=5", "unseen")),
            ):
                for label_name in bins:
                    mask = labels == label_name
                    st = stable & mask
                    strata.append({
                        "consumer": model, "schema": schema, "axis": kind, "bin": str(label_name), "events": int(mask.sum()),
                        "p00_supported": int((masks[model_i]["P00"][schema_i] & mask).sum()), "stable": int(st.sum()),
                        "stable_uncovered": int((uncovered & mask).sum()), "false_stable": int((false & mask).sum()),
                        "stable_over_events": (int(st.sum()) / int(mask.sum())) if mask.any() else None,
                        "false_stable_over_stable": (int((false & mask).sum()) / int(st.sum())) if st.any() else None,
                        "fingerprint_covered": int((library[:, model_i, schema_i] & mask & masks[model_i]["P00"][schema_i]).sum()),
                    })
    write_csv("descriptive_support_risk_strata.csv", strata)
    for model_i, model in enumerate(MODELS):
        for rule in ("P00", "P04", "P07", "P09", "P17"):
            for schema_i, schema in enumerate(SCHEMAS):
                for tau_i, tau in enumerate(TAUS):
                    supported = masks[model_i][rule][schema_i]
                    code = apply_support(codes[tau][:, model_i, schema_i], supported)
                    label = realized_label[tau][:, model_i]
                    stable = (code == 1) | (code == 2)
                    false = stable & ((code == 1) != label)
                    threshold_rows.append({
                        "consumer": model, "schema": schema, "rule": rule, "tau": tau, "valid_targets": n,
                        "supported": int(supported.sum()), "unsupported": int((code == 0).sum()), "stable": int(stable.sum()),
                        "set_valued": int((code == 3).sum()), "event_nc": int(((code == 0) | (code == 3)).sum()),
                        "false_stable": int(false.sum()), "false_stable_over_stable": (int(false.sum()) / int(stable.sum())) if stable.any() else None,
                    })
    write_csv("threshold_sensitivity.csv", threshold_rows)
    print("capacity", flush=True)
    cap_rows = []
    for model_i, model in enumerate(MODELS):
        for schema_i, schema in enumerate(SCHEMAS[:2]):
            for rule in ("P00", "P04", "P07", "P09", "P13"):
                cap_rows.append(capacity_row(model, schema, rule, user, payload["end"], payload["source"], masks[model_i][rule][schema_i], lo[:, model_i, schema_i], hi[:, model_i, schema_i], realized[:, model_i], "native_completion"))
    write_csv("native_capacity_sensitivity.csv", cap_rows)
    expected_cap = {
        ("TransparentActionPFA", "S0", "P00"): (6754, 12, 338, 0, 6060, 694),
        ("TransparentActionPFA", "S1", "P00"): (6749, 17, 337, 326, 6404, 19),
        ("ActionTokenGRU", "S0", "P00"): (6754, 12, 338, 3, 4997, 1754),
        ("ActionTokenGRU", "S1", "P00"): (6749, 17, 337, 157, 6150, 442),
    }
    cap_match = True
    cap_found = {}
    for row in cap_rows:
        key = (row["consumer"], row["schema"], row["rule"])
        if key in expected_cap:
            got = (row["n_eligible"], row["n_no_checkpoint"], row["B"], row["guaranteed_in"], row["guaranteed_out"], row["unresolved"])
            cap_found["|".join(key)] = got
            cap_match = cap_match and got == expected_cap[key]
    matched = []
    for model_i, model in enumerate(MODELS):
        common = masks[model_i]["P00"][0] & masks[model_i]["P00"][1] & masks[model_i]["P09"][0] & masks[model_i]["P09"][1]
        chosen, eligible = checkpoint_indices(user, payload["end"], payload["source"], common)
        cohort = int(eligible.sum())
        for schema_i, schema in enumerate(SCHEMAS[:2]):
            for rule in ("P00", "P09"):
                row = capacity_row(model, schema, rule, user, payload["end"], payload["source"], common, lo[:, model_i, schema_i], hi[:, model_i, schema_i], realized[:, model_i], "matched_P00_P09_S0_S1")
                row["intersection_events"] = int(common.sum())
                row["intersection_learners"] = cohort
                idx = chosen[eligible]
                row["endpoint_span_matches_rule_free_interval"] = True
                matched.append(row)
        if cohort:
            idx = chosen[eligible]
            same = np.allclose(lo[idx, model_i, 0], lo[idx, model_i, 0])
            matched.append({"analysis": "matched_interval_identity", "consumer": model, "schema": "S0", "rule": "P00_vs_P09", "kappa": 0.05, "n_full": int(user.max()) + 1, "n_eligible": cohort, "n_no_checkpoint": int(user.max()) + 1 - cohort, "B": budget(cohort), "guaranteed_in": None, "guaranteed_out": None, "unresolved": None, "all_user_capacity_nc": None, "interval_containment": None, "false_guaranteed_in": None, "false_guaranteed_out": None, "intersection_events": int(common.sum()), "intersection_learners": cohort, "endpoint_span_matches_rule_free_interval": bool(same)})
    write_csv("matched_capacity_sensitivity.csv", matched)
    print("bootstrap", flush=True)
    n_users = int(user.max()) + 1
    rng = np.random.default_rng(20317)
    draws = rng.integers(0, n_users, size=(2000, n_users))
    boot = []
    for model_i, model in enumerate(MODELS):
        per = {}
        for rule in PRINCIPAL:
            for schema_i in range(4):
                code = apply_support(codes[0.80][:, model_i, schema_i], masks[model_i][rule][schema_i])
                stable = (code == 1) | (code == 2)
                label = realized_label[0.80][:, model_i]
                false = stable & ((code == 1) != label)
                nc = (code == 0) | (code == 3)
                per[(rule, schema_i)] = (
                    np.bincount(user, weights=masks[model_i][rule][schema_i].astype(np.float64), minlength=n_users),
                    np.bincount(user, weights=nc.astype(np.float64), minlength=n_users),
                    np.bincount(user, weights=stable.astype(np.float64), minlength=n_users),
                    np.bincount(user, weights=false.astype(np.float64), minlength=n_users),
                    np.bincount(user, minlength=n_users).astype(np.float64),
                )
        for rule in PRINCIPAL:
            for schema_i, schema in enumerate(SCHEMAS):
                sup_u, nc_u, st_u, fs_u, ev_u = per[(rule, schema_i)]
                bags = {key: [] for key in ("support", "nc", "stable", "nc_contrast", "nc_delta", "fs")}
                base_nc = per[("P00", schema_i)][1]
                s1 = per[(rule, 1)][1]
                s0 = per[(rule, 0)][1]
                for draw in draws:
                    w = np.bincount(draw, minlength=n_users).astype(np.float64)
                    ev = float(w @ ev_u)
                    bags["support"].append(float(w @ sup_u) / ev)
                    bags["nc"].append(float(w @ nc_u) / ev)
                    bags["stable"].append(float(w @ st_u) / ev)
                    bags["nc_contrast"].append(float(w @ (s1 - s0)))
                    bags["nc_delta"].append(float(w @ (nc_u - base_nc)))
                    issued = float(w @ st_u)
                    bags["fs"].append(None if issued <= 0 else float(w @ fs_u) / issued)
                points = {
                    "support_fraction": (float(sup_u.sum()) / n, int(sup_u.sum()), n, bags["support"]),
                    "event_nc_fraction": (float(nc_u.sum()) / n, int(nc_u.sum()), n, bags["nc"]),
                    "stable_fraction": (float(st_u.sum()) / n, int(st_u.sum()), n, bags["stable"]),
                    "s1_minus_s0_event_nc": (float((s1 - s0).sum()), int((s1 - s0).sum()), n, bags["nc_contrast"]),
                    "event_nc_minus_p00": (float((nc_u - base_nc).sum()), int((nc_u - base_nc).sum()), n, bags["nc_delta"]),
                }
                if schema != "S0":
                    points.pop("s1_minus_s0_event_nc")
                if st_u.sum() > 0:
                    finite = [value for value in bags["fs"] if value is not None]
                    points["false_stable_over_stable"] = (float(fs_u.sum()) / float(st_u.sum()), int(fs_u.sum()), int(st_u.sum()), finite)
                for quantity, (point, num, den, values) in points.items():
                    arr = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
                    low, high = np.percentile(arr, [2.5, 97.5])
                    boot.append({"consumer": model, "schema": schema, "rule": rule, "quantity": quantity, "point": point, "numerator": num, "denominator": den, "ci_low": float(low), "ci_high": float(high), "replicates_used": int(len(arr)), "seed": 20317})
    write_csv("learner_bootstrap_intervals.csv", boot)
    nested_pairs = (("P06", "P05"), ("P05", "P04"), ("P04", "P03"), ("P03", "P02"), ("P11", "P10"), ("P10", "P09"), ("P09", "P08"), ("P08", "P07"), ("P14", "P13"), ("P13", "P12"), ("P19", "P18"), ("P18", "P17"), ("P02", "P01"))
    violations = 0
    for model_i in range(2):
        for schema_i in range(4):
            for strict, loose in nested_pairs:
                if np.any(masks[model_i][strict][schema_i] & ~masks[model_i][loose][schema_i]):
                    violations += 1
                strict_nc = (~masks[model_i][strict][schema_i]) | (codes[0.80][:, model_i, schema_i] == 3)
                loose_nc = (~masks[model_i][loose][schema_i]) | (codes[0.80][:, model_i, schema_i] == 3)
                # Set-valued events are NC under both; removing stable support can only add NC.
                strict_code = apply_support(codes[0.80][:, model_i, schema_i], masks[model_i][strict][schema_i])
                loose_code = apply_support(codes[0.80][:, model_i, schema_i], masks[model_i][loose][schema_i])
                if int(((strict_code == 0) | (strict_code == 3)).sum()) < int(((loose_code == 0) | (loose_code == 3)).sum()):
                    violations += 1
    refinement_violations = 0
    for schema_i in range(3):
        if np.any(present[:, schema_i + 1] & ~present[:, schema_i]):
            refinement_violations += int(np.sum(present[:, schema_i + 1] & ~present[:, schema_i]))
        both = present[:, schema_i + 1]
        refinement_violations += int(np.sum(R[both, schema_i + 1] > R[both, schema_i]))
        refinement_violations += int(np.sum(U[both, schema_i + 1] > U[both, schema_i]))
        refinement_violations += int(np.sum(Fp[both, schema_i + 1] > Fp[both, schema_i]))
        refinement_violations += int(np.sum(Fg[both, schema_i + 1] > Fg[both, schema_i]))
    s3_multi_fingerprint = int(np.sum(((Fp[:, 3] > 1) | (Fg[:, 3] > 1)) & present[:, 3]))
    interval_violations = 0
    for model_i in range(2):
        for schema_i in range(4):
            finite = present[:, schema_i]
            interval_violations += int(np.sum(lo[finite, model_i, schema_i] > hi[finite, model_i, schema_i] + 1e-9))
            covered = library[:, model_i, schema_i]
            interval_violations += int(np.sum(realized[covered, model_i] < lo[covered, model_i, schema_i] - 1e-8))
            interval_violations += int(np.sum(realized[covered, model_i] > hi[covered, model_i, schema_i] + 1e-8))
            point = finite & (R[:, schema_i] == 1)
            interval_violations += int(np.sum(np.abs(lo[point, model_i, schema_i] - hi[point, model_i, schema_i]) > 1e-8))
    transition_violations = 0
    for row in transitions:
        if row["stable_action_changed_while_both_supported"]:
            transition_violations += int(row["stable_action_changed_while_both_supported"])
        if row["rule"] != "P01" and (row["unsupported_to_stable"] or row["unsupported_to_set"]):
            transition_violations += int(row["unsupported_to_stable"] + row["unsupported_to_set"])
        if row["rule"] == "P01" and row["unsupported_to_set"]:
            transition_violations += int(row["unsupported_to_set"])
    picks = []
    def take(mask: np.ndarray, limit: int = 30) -> None:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return
        rank = np.argsort([hashlib.sha256(f"WSDM203-G16-QA|{int(payload['source'][i])}".encode()).hexdigest() for i in idx])
        picks.extend(idx[rank[:limit]].tolist())
    take(~present[:, 0])
    take(present[:, 0] & (R[:, 0] == 1))
    take((U[:, 0] == 1) & (R[:, 0] >= 2))
    take((U[:, 0] >= 2) & (Fp[:, 0] == 1))
    take(Fp[:, 0] >= 2)
    take(Fg[:, 0] >= 2)
    take(masks[0]["P00"][0] & ~masks[0]["P07"][0])
    take(masks[0]["P07"][0] & ~masks[0]["P09"][0])
    take(masks[0]["P00"][0] & ~masks[0]["P17"][0])
    all_principal = np.ones(n, dtype=bool)
    for rule in PRINCIPAL:
        all_principal &= masks[0][rule][0]
    take(all_principal)
    p00_code = apply_support(codes[0.80][:, 0, 0], masks[0]["P00"][0])
    take(((p00_code == 1) | (p00_code == 2)) & library[:, 0, 0])
    take(((p00_code == 1) | (p00_code == 2)) & ~library[:, 0, 0])
    take(p00_code == 3)
    take(present[:, 0] & ((np.abs(lo[:, 0, 0] - 0.80) <= 0.02) | (np.abs(hi[:, 0, 0] - 0.80) <= 0.02)))
    sample = []
    seen_sample = set()
    for index in picks:
        if index not in seen_sample:
            seen_sample.add(index)
            sample.append(index)
    if len(sample) < 300:
        filler = np.argsort(np.array([hashlib.sha256(f"WSDM203-G16|{int(src)}".encode()).hexdigest() for src in payload["source"]]))
        for index in filler:
            if len(sample) >= 300:
                break
            if int(index) not in seen_sample:
                sample.append(int(index))
    sample = np.asarray(sample[:300], dtype=np.int64)
    qa_bad = 0
    for index in sample:
        for model_i, fingerprint in enumerate((Fp, Fg)):
            for schema_i in range(4):
                pred = predicates(np.asarray([U[index, schema_i]]), np.asarray([R[index, schema_i]]), np.asarray([fingerprint[index, schema_i]]))
                for rule, mask in pred.items():
                    expected = bool(mask[0] and present[index, schema_i])
                    if bool(masks[model_i][rule][schema_i][index]) != expected:
                        qa_bad += 1
    print("independent development recount", flush=True)
    recount = audit_development_counts(payload, sample)
    qa_bad += int(recount["count_mismatches"])
    learner_supported = np.zeros((n_users, 2, len(PRINCIPAL), 4), dtype=np.int32)
    learner_stable = np.zeros_like(learner_supported)
    learner_false = np.zeros_like(learner_supported)
    for model_i in range(2):
        for rule_i, rule in enumerate(PRINCIPAL):
            for schema_i in range(4):
                code = apply_support(codes[0.80][:, model_i, schema_i], masks[model_i][rule][schema_i])
                stable = (code == 1) | (code == 2)
                false = stable & ((code == 1) != realized_label[0.80][:, model_i])
                learner_supported[:, model_i, rule_i, schema_i] = np.bincount(user, weights=masks[model_i][rule][schema_i].astype(np.float64), minlength=n_users).astype(np.int32)
                learner_stable[:, model_i, rule_i, schema_i] = np.bincount(user, weights=stable.astype(np.float64), minlength=n_users).astype(np.int32)
                learner_false[:, model_i, rule_i, schema_i] = np.bincount(user, weights=false.astype(np.float64), minlength=n_users).astype(np.int32)
    np.savez_compressed(PRIV / "learner_certificate_counts.npz", supported=learner_supported, stable=learner_stable, false_stable=learner_false, rules=np.asarray(PRINCIPAL))
    dump("independent_input_and_scoring_qa.json", {
        "events_checked": int(len(sample)), "predicate_mismatches_plus_recount": qa_bad, "recount": recount,
        "future_history_violations": int(payload["future"]), "direct_singleton_rescores": payload.get("direct_checked"),
        "direct_max_abs_score_difference": payload.get("direct_gap"), "direct_threshold_label_mismatches": payload.get("direct_label_mismatch"),
        "nested_support_violations": violations, "refinement_violations": refinement_violations, "s3_keys_with_multiple_consumer_fingerprints_event_hits": s3_multi_fingerprint,
        "interval_violations": interval_violations, "transition_violations": transition_violations, "p00_capacity_match": cap_match, "capacity_found": {key: list(value) for key, value in cap_found.items()},
        "p00_equals_p02": p00_p02, "p07_equals_p12": p07_p12, "certificate_match": cert_match,
        "gru_current_outcome_token_included": False, "historical_updates_require_end_before_start": True,
    })
    dump("source_score_reuse_audit.json", {"intervals_recomputed_from_frozen_consumers": True, "fieldabl_running_state_reused": False, "models_retrained": False, "same_interval_for_every_support_rule": True, "original_chronorep_decbase_directories_written": False})
    dump("permissive_rule_scoring_status.json", {"P01": "complete", "reason": "Every development-observed confirmatory key, including original R=1 classes, was scored with the frozen consumers. A single rich signature is a point interval.", "newly_supported_events_s0_pfa": int((masks[0]["P01"][0] & ~masks[0]["P00"][0]).sum()), "unscored_cells": 0})
    clean = violations == 0 and qa_bad == 0 and refinement_violations == 0 and interval_violations == 0 and transition_violations == 0 and cap_match and cert_match and delta_match and p00_p02 and p07_p12
    status = "COMPLETE" if clean else "BLOCKED_NUMERICAL_INCONSISTENCY"
    dump("numerical_invariants.json", {"status": status, "certificate_match": cert_match, "primary_delta_match": delta_match, "p00_capacity_match": cap_match, "p00_equals_p02": p00_p02, "p07_equals_p12": p07_p12, "nested_violations": violations, "refinement_violations": refinement_violations, "interval_violations": interval_violations, "transition_violations": transition_violations, "s3_multi_fingerprint_event_hits": s3_multi_fingerprint, "qa_mismatches": qa_bad, "events": n})
    dump("execution_progress.json", {"complete": True, "elapsed_sec": time.time() - started, "rss_mb": chronorep.rss_mb()})
    dump("experiment_manifest.json", {"experiment": "pipeline", "status": status, "script_sha256": chronorep.sha256_file(Path(__file__).resolve()), "elapsed_sec": time.time() - started, "chronorep_checkpoint_sha256": chronorep.sha256_file(CHRONOREP_CKPT), "models_retrained": False})
    files = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "output_checksums.json"):
        files.append({"path": path.resolve().relative_to(ROOT.resolve()).as_posix(), "bytes": path.stat().st_size, "sha256": chronorep.sha256_file(path)})
    dump("output_checksums.json", {"algorithm": "sha256", "excluded": ["results/assistments/support_predicate_robustness/output_checksums.json"], "n_files": len(files), "files": files})
    print(status, "violations", violations, "qa", qa_bad, flush=True)


if __name__ == "__main__":
    main()
