"""Immutable P2 reference contract (V2).

POST-UNSEAL CORRECTIVE REANALYSIS support.
NOT ORIGINAL CONFIRMATORY EVIDENCE.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

FULL_REPS = {"A-FULL", "E-FULL", "EEDI-R1"}
IDENTITY_BASELINES = frozenset({"B0", "B1", "B2", "B5", "B6", "B7", "B8", "B10"})


@dataclass(frozen=True)
class P2ReferenceContract:
    evaluated_method_score_id: str
    evaluated_representation: str
    evaluated_checkpoint: str
    reference_score_id: str
    reference_representation: str
    reference_checkpoint: str
    eligible_population_id: str
    capacity: float
    tie_rule: str
    utility: str
    list_construction: str
    regret_definition: str
    replacement_definition: str
    placebo_self_check: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def full_rep_for_dataset(dataset: str) -> str:
    ds = dataset.lower()
    if "assistments" in ds:
        return "A-FULL"
    if "ednet" in ds:
        return "E-FULL"
    if "eedi" in ds:
        # Prospective full event-semantic reference (EEDI-R1), not a separate FULL stage.
        return "EEDI-R1"
    raise ValueError(f"No FULL representation mapping for dataset={dataset}")


def build_primary_contract(
    *,
    dataset: str,
    model: str,
    evaluated_representation: str,
    seed: int,
    capacity: float = 0.05,
    tie_rule: str = "primary_user_hash",
    placebo_self_check: bool = False,
) -> P2ReferenceContract:
    ref_rep = (
        evaluated_representation
        if placebo_self_check
        else full_rep_for_dataset(dataset)
    )
    ckpt = f"{dataset}|{model}|s{seed}"
    return P2ReferenceContract(
        evaluated_method_score_id=f"prob|{evaluated_representation}|{ckpt}",
        evaluated_representation=evaluated_representation,
        evaluated_checkpoint=ckpt,
        reference_score_id=f"prob|{ref_rep}|{ckpt}",
        reference_representation=ref_rep,
        reference_checkpoint=ckpt,
        eligible_population_id=f"users|{dataset}|{model}|s{seed}",
        capacity=float(capacity),
        tie_rule=tie_rule,
        utility="indicator_membership_in_reference_list",
        list_construction="collapse_max_risk_then_lexsort_risk_desc_tie_asc",
        regret_definition="mean_abs(selected_membership - 1) over reference list",
        replacement_definition="1 - |sel ∩ ref| / |ref|",
        placebo_self_check=placebo_self_check,
    )


def assert_reference_independence(
    *,
    contract: P2ReferenceContract,
    evaluated_probs: np.ndarray,
    reference_probs: np.ndarray,
    evaluated_users: np.ndarray,
    reference_users: np.ndarray,
) -> None:
    """Hard-fail on unintentional aliasing / invalid reference provenance."""
    if contract.placebo_self_check:
        return
    if contract.reference_representation not in FULL_REPS:
        raise RuntimeError(
            f"P2_REFERENCE_CONTRACT_VIOLATION: reference_representation="
            f"{contract.reference_representation} is not FULL"
        )
    if not contract.reference_checkpoint:
        raise RuntimeError("P2_REFERENCE_CONTRACT_VIOLATION: missing reference checkpoint")
    if evaluated_probs is reference_probs:
        raise RuntimeError(
            "P2_REFERENCE_CONTRACT_VIOLATION: evaluated and reference arrays alias"
        )
    if np.shares_memory(evaluated_probs, reference_probs):
        raise RuntimeError(
            "P2_REFERENCE_CONTRACT_VIOLATION: evaluated/reference share memory"
        )
    if (
        contract.evaluated_representation != contract.reference_representation
        and evaluated_probs.shape == reference_probs.shape
        and np.allclose(evaluated_probs, reference_probs)
        and contract.evaluated_representation not in FULL_REPS
    ):
        # Same values from different intended sources is still invalid for scientific cells
        raise RuntimeError(
            "P2_REFERENCE_CONTRACT_VIOLATION: evaluated and reference probability "
            "columns are numerically identical while independence is required"
        )
    eu = np.asarray([str(x) for x in evaluated_users])
    ru = np.asarray([str(x) for x in reference_users])
    if set(eu.tolist()) != set(ru.tolist()):
        raise RuntimeError(
            "P2_REFERENCE_CONTRACT_VIOLATION: eligible populations differ"
        )
