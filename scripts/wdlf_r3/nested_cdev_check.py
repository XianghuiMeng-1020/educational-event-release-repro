#!/usr/bin/env python3
"""Verify empirical C_dev nesting for ASSISTments nested releases (shared development records)."""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "wdlf_r7"))

from r7_core import Event, release_key, rich_signature  # noqa: E402

SEALED = ROOT / "prospective" / "wdlf_r7" / "sealed"
OUT = ROOT / "results" / "wdlf_r3"
SCHEMAS = ["S0", "S1", "S2", "S3"]


def fps_for_schema(schema: str, sigs: list) -> set[tuple]:
    """Decision fingerprints used by validity (schema-agnostic collapse of sig[5] actions)."""
    out = set()
    for sig in sigs:
        sig = tuple(sig) if isinstance(sig, list) else sig
        # transparent-style: (correct, hint, attempt, actions)
        # neural-style: (skill, actions) — for nesting of C* we compare full rich sigs projected
        out.add(sig)
    return out


def main() -> None:
    with (SEALED / "dicts_dev_compact.pkl").open("rb") as f:
        dicts = pickle.load(f)["dicts"]
    with (SEALED / "splits" / "confirmatory.pkl").open("rb") as f:
        events: list[Event] = pickle.load(f)

    # For each confirmatory event, check C_dev fingerprint inclusion across nested schemas
    # using the SAME rich signatures stored under each schema key.
    violations = {f"{a}->{b}": 0 for a, b in zip(SCHEMAS, SCHEMAS[1:])}
    checked = {k: 0 for k in violations}
    both_supported = {k: 0 for k in violations}
    # Also verify: every rich sig under fine key also maps under coarse key's C_dev
    # by re-deriving from development events if available — here we check confirmatory
    # eventwise: if fine supported and coarse supported, every fine C_dev sig's release
    # under coarse equals coarse key, and fine sig set ⊆ coarse sig set when projected
    # to coarse observation. Stronger check: for each event with both keys supported,
    # the set of compatible rich signatures under fine is a subset of those under coarse.

    def entry_sigs(entry):
        if entry is None:
            return None, 0
        if isinstance(entry, dict) and "sigs" in entry:
            return [tuple(x) if isinstance(x, list) else x for x in entry["sigs"]], int(
                entry.get("n_users", 2)
            )
        return [tuple(x) if isinstance(x, list) else x for x in entry.keys()], 2

    def supported(entry):
        sigs, n_users = entry_sigs(entry) if entry is not None else (None, 0)
        if sigs is None:
            return False, []
        if n_users < 2 and len(sigs) < 2:
            return False, []
        return True, sigs

    for ev in events:
        keys = {s: release_key(ev, s) for s in SCHEMAS}
        for c, f in zip(SCHEMAS, SCHEMAS[1:]):
            ok_c, sigs_c = supported(dicts[c].get(keys[c]))
            ok_f, sigs_f = supported(dicts[f].get(keys[f]))
            if not (ok_c and ok_f):
                continue
            both_supported[f"{c}->{f}"] += 1
            checked[f"{c}->{f}"] += 1
            set_c, set_f = set(sigs_c), set(sigs_f)
            # Nested release: every fine-compatible rich record maps to coarse key,
            # so fine C_dev ⊆ coarse C_dev as sets of rich signatures.
            if not set_f.issubset(set_c):
                violations[f"{c}->{f}"] += 1

    # Pairwise global check over all fine keys: project each fine key's sigs to coarse release
    # and ensure each appears in coarse dictionary class.
    key_level_violations = {f"{a}->{b}": 0 for a, b in zip(SCHEMAS, SCHEMAS[1:])}
    key_level_checked = {k: 0 for k in key_level_violations}
    # Build reverse: for each rich sig in fine dict, compute coarse key via a synthetic event — hard.
    # Instead sample: for each fine key entry, for each sig, reconstruct Event-like fields from sig
    # and check release_key under coarse is in coarse dict and sig in that class.
    from r7_core import syn_from_sig  # local

    # Need a template event — use first confirmatory event's non-action fields
    template = events[0]
    for c, f in zip(SCHEMAS, SCHEMAS[1:]):
        for fkey, entry in dicts[f].items():
            ok_f, sigs_f = supported(entry)
            if not ok_f:
                continue
            for sig in sigs_f:
                key_level_checked[f"{c}->{f}"] += 1
                syn = syn_from_sig(template, sig)
                # Preserve skill/problem from sig if encoded
                ckey = release_key(syn, c)
                ok_c, sigs_c = supported(dicts[c].get(ckey))
                if not ok_c or sig not in set(sigs_c):
                    key_level_violations[f"{c}->{f}"] += 1

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            "Empirical nesting: for shared development libraries, rich-signature sets "
            "satisfy C_dev,fine(z_f)subseteq C_dev,coarse(z_c) on confirmatory events "
            "where both keys are supported; plus key-level projection check via syn_from_sig."
        ),
        "eventwise_both_supported": both_supported,
        "eventwise_subset_violations": violations,
        "eventwise_zero_violations": all(v == 0 for v in violations.values()),
        "key_level_checked": key_level_checked,
        "key_level_violations": key_level_violations,
        "key_level_zero_violations": all(v == 0 for v in key_level_violations.values()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "nested_cdev_check.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
