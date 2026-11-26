"""Capacity policy as an identified membership set (WDLF Auth R2).

Decision object: membership in a fixed-size support shortlist of size
k = round(|U| * kappa).

For each user u, a feasible risk set S_u (discrete) or interval [L_u, U_u]
is induced by compatible-trace propagation of that user's released history.
Risk = 1 - mastery_proxy (higher risk → more likely selected).

Labels:
  guaranteed_in  — selected under every joint feasible realization
  guaranteed_out — selected under no joint feasible realization
  ambiguous      — selection depends on which compatible completions occur

Independence assumption for certification with intervals:
  adversary may choose each user's risk independently within their feasible
  set/interval (worst-case product uncertainty). This is conservative relative
  to shared latent shocks.

Old G6P R0-vs-R1 replacement remains a SECONDARY external reference diagnostic.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CapacityMembership:
    user_id: str
    status: str  # guaranteed_in | guaranteed_out | ambiguous
    risk_min: float
    risk_max: float
    worst_rank: int  # 1 = highest risk
    best_rank: int
    k: int


def _n_select(n_users: int, capacity: float) -> int:
    k = int(round(n_users * float(capacity)))
    if capacity > 0 and k == 0:
        k = 1
    return min(max(k, 0), n_users)


def robust_rank_bounds(
    user_ids: Sequence[str],
    risk_mins: Sequence[float],
    risk_maxs: Sequence[float],
) -> dict[str, tuple[int, int]]:
    """Return (best_rank, worst_rank) for each user (1 = most at-risk).

    Best rank: maximize own risk, minimize others (use risk_max for u, risk_min for others).
    Worst rank: minimize own risk, maximize others.
    Ties broken by user_id ascending (same as G6P deterministic rule).
    """
    users = list(user_ids)
    n = len(users)
    out: dict[str, tuple[int, int]] = {}
    for i, u in enumerate(users):
        # Best-case rank for u
        own_best = float(risk_maxs[i])
        better = 0
        for j, v in enumerate(users):
            if j == i:
                continue
            rj = float(risk_mins[j])
            if (rj > own_best) or (rj == own_best and v < u):
                better += 1
        best_rank = better + 1
        # Worst-case rank for u
        own_worst = float(risk_mins[i])
        better_w = 0
        for j, v in enumerate(users):
            if j == i:
                continue
            rj = float(risk_maxs[j])
            if (rj > own_worst) or (rj == own_worst and v < u):
                better_w += 1
        worst_rank = better_w + 1
        out[u] = (best_rank, worst_rank)
    assert len(out) == n
    return out


def certify_capacity_membership(
    user_ids: Sequence[str],
    risk_mins: Sequence[float],
    risk_maxs: Sequence[float],
    *,
    capacity: float = 0.05,
) -> list[CapacityMembership]:
    if len(user_ids) != len(risk_mins) or len(user_ids) != len(risk_maxs):
        raise ValueError("length mismatch")
    for lo, hi in zip(risk_mins, risk_maxs):
        if float(lo) - float(hi) > 1e-12:
            raise ValueError("risk_min must be <= risk_max")
    k = _n_select(len(user_ids), capacity)
    ranks = robust_rank_bounds(user_ids, risk_mins, risk_maxs)
    results: list[CapacityMembership] = []
    for i, u in enumerate(user_ids):
        best_rank, worst_rank = ranks[u]
        if worst_rank <= k:
            status = "guaranteed_in"
        elif best_rank > k:
            status = "guaranteed_out"
        else:
            status = "ambiguous"
        results.append(
            CapacityMembership(
                user_id=str(u),
                status=status,
                risk_min=float(risk_mins[i]),
                risk_max=float(risk_maxs[i]),
                worst_rank=worst_rank,
                best_rank=best_rank,
                k=k,
            )
        )
    return results


def brute_force_membership(
    user_ids: Sequence[str],
    feasible_risks: Sequence[Sequence[float]],
    *,
    capacity: float = 0.05,
) -> dict[str, str]:
    """Enumerate Cartesian product of discrete feasible risks; exact membership.

    Returns status in {guaranteed_in, guaranteed_out, ambiguous}.
    """
    users = [str(u) for u in user_ids]
    sets = [sorted({float(x) for x in s}) for s in feasible_risks]
    if any(len(s) == 0 for s in sets):
        raise ValueError("empty feasible set")
    k = _n_select(len(users), capacity)
    selected_counts = {u: 0 for u in users}
    n_worlds = 0
    for risks in itertools.product(*sets):
        n_worlds += 1
        # rank by risk desc, user_id asc
        order = sorted(range(len(users)), key=lambda i: (-risks[i], users[i]))
        chosen = {users[i] for i in order[:k]}
        for u in chosen:
            selected_counts[u] += 1
    out: dict[str, str] = {}
    for u in users:
        c = selected_counts[u]
        if c == n_worlds:
            out[u] = "guaranteed_in"
        elif c == 0:
            out[u] = "guaranteed_out"
        else:
            out[u] = "ambiguous"
    return out


def interval_cert_vs_brute(
    user_ids: Sequence[str],
    feasible_risks: Sequence[Sequence[float]],
    *,
    capacity: float,
) -> tuple[list[CapacityMembership], dict[str, str], bool]:
    """Compare interval robust bounds to discrete brute force.

    Interval method uses min/max of each feasible set. It must be sound
    (never call guaranteed_in/out incorrectly) but may be conservative
    (label ambiguous when brute force is decisive).
    """
    mins = [min(s) for s in feasible_risks]
    maxs = [max(s) for s in feasible_risks]
    interval = certify_capacity_membership(
        user_ids, mins, maxs, capacity=capacity
    )
    brute = brute_force_membership(user_ids, feasible_risks, capacity=capacity)
    sound = True
    for m in interval:
        b = brute[m.user_id]
        if m.status == "guaranteed_in" and b != "guaranteed_in":
            sound = False
        if m.status == "guaranteed_out" and b != "guaranteed_out":
            sound = False
        # ambiguous interval may refine under discrete — OK
    return interval, brute, sound
