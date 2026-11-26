"""Production corrected P2 engine (Implementation A).

POST-UNSEAL CORRECTIVE REANALYSIS
NOT ORIGINAL CONFIRMATORY EVIDENCE
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.policies.p2_reference_contract import (
    P2ReferenceContract,
    assert_reference_independence,
    build_primary_contract,
)


def collapse_max_risk(user_keys: np.ndarray, risks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order: dict[str, float] = {}
    for uid, risk in zip(user_keys, risks):
        key = str(uid)
        r = float(risk)
        prev = order.get(key)
        if prev is None or r > prev:
            order[key] = r
    users = np.array(list(order.keys()), dtype=object)
    vals = np.array([order[u] for u in users], dtype=np.float64)
    return users, vals


def select_capacity_list(
    users: np.ndarray,
    risks: np.ndarray,
    *,
    capacity: float,
    tie_keys: np.ndarray | None = None,
) -> list[str]:
    if not 0.0 <= capacity <= 1.0:
        raise ValueError("capacity must be in [0, 1]")
    if len(users) == 0:
        return []
    ties = np.array([str(u) for u in users], dtype=object) if tie_keys is None else np.asarray(tie_keys, dtype=object)
    idx = np.lexsort((ties, -np.asarray(risks, dtype=np.float64)))
    ranked = users[idx]
    n_select = int(round(len(ranked) * capacity))
    if capacity > 0 and n_select == 0:
        n_select = 1
    return [str(u) for u in ranked[:n_select]]


def baseline_transform(probs: np.ndarray, baseline: str, seed: int) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if baseline == "B3":
        return np.abs(p - 0.5)
    if baseline == "B4":
        lo = np.clip(p, 1e-9, 1.0 - 1e-9)
        return -(lo * np.log(lo) + (1.0 - lo) * np.log(1.0 - lo))
    if baseline == "B9":
        import random

        rng = random.Random(int(seed))
        return np.fromiter((rng.random() for _ in range(len(p))), dtype=np.float64, count=len(p))
    return p.copy()


def compute_p2_metrics_v2(
    *,
    evaluated_probs: np.ndarray,
    reference_probs: np.ndarray,
    user_keys: np.ndarray,
    baseline: str,
    seed: int,
    capacity: float = 0.05,
    contract: P2ReferenceContract | None = None,
    dataset: str = "",
    model: str = "",
    evaluated_representation: str = "",
) -> dict[str, Any]:
    """Corrected P2: evaluated list vs independent full-trace reference list."""
    if contract is None:
        contract = build_primary_contract(
            dataset=dataset,
            model=model,
            evaluated_representation=evaluated_representation,
            seed=seed,
            capacity=capacity,
        )
    users = np.asarray(user_keys, dtype=object)
    # Guard on caller-provided arrays before defensive copies.
    assert_reference_independence(
        contract=contract,
        evaluated_probs=np.asarray(evaluated_probs),
        reference_probs=np.asarray(reference_probs),
        evaluated_users=users,
        reference_users=users,
    )
    ev = np.asarray(evaluated_probs, dtype=np.float64).copy()
    rf = np.asarray(reference_probs, dtype=np.float64).copy()
    scores = baseline_transform(ev, baseline, seed)
    risks = 1.0 - scores
    ref_risks = 1.0 - rf
    u_ev, r_ev = collapse_max_risk(users, risks)
    u_rf, r_rf = collapse_max_risk(users, ref_risks)
    selected = select_capacity_list(u_ev, r_ev, capacity=capacity)
    reference = select_capacity_list(u_rf, r_rf, capacity=capacity)
    sel_set, ref_set = set(selected), set(reference)
    if not ref_set:
        regret = 0.0
        replacement = 0.0
    else:
        regret = float(np.mean([0.0 if u in sel_set else 1.0 for u in reference]))
        replacement = 1.0 - len(sel_set.intersection(ref_set)) / len(ref_set)
    unsafe = float(sum(1 for u in selected if u not in ref_set) / max(1, len(selected)))
    # Boundary separation on point scores (diagnostic)
    if selected and len(u_ev) > len(selected):
        sel_scores = [1.0 - float(r_ev[list(u_ev).index(u)]) for u in selected if u in set(map(str, u_ev))]
        uns = [str(u) for u in u_ev if str(u) not in sel_set]
        uns_scores = [1.0 - float(r_ev[list(u_ev).index(u)]) for u in uns]
        boundary_sep = float(min(sel_scores) - max(uns_scores)) if sel_scores and uns_scores else float("nan")
    else:
        boundary_sep = float("nan")
    return {
        "policy_regret": float(regret),
        "fixed_capacity_list_replacement": float(replacement),
        "unsafe_automation_rate": float(unsafe),
        "list_identical": selected == reference,
        "list_identifiable": selected == reference,
        "eligible_population": int(len(u_ev)),
        "seat_count": int(len(selected)),
        "boundary_separation_point": boundary_sep,
        "selected_list": selected,
        "reference_list": reference,
        "contract": contract.to_dict(),
        "label": "POST-UNSEAL CORRECTIVE REANALYSIS - NOT ORIGINAL CONFIRMATORY EVIDENCE",
    }
