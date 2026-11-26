"""Compatible-trace decision propagation (WDLF Auth R2).

Contract A (local current-event completion) at the post-ingest / next-step
checkpoint. Development-only option templates supply C_dev(z).
"""

from src.identification.compatible_trace.capacity_identified_set import (
    CapacityMembership,
    certify_capacity_membership,
    robust_rank_bounds,
)
from src.identification.compatible_trace.certificate_v2 import (
    CertificateV2,
    certify_from_propagated_scores,
)
from src.identification.compatible_trace.dkt_propagation import (
    DKTPropagationResult,
    propagate_dkt_local_options,
)
from src.identification.compatible_trace.pfa_propagation import (
    PFAState,
    merge_equivalent_states,
    propagate_pfa_local_options,
    pfa_ingest_event,
    pfa_next_step_proba,
    pfa_state_from_history,
)

__all__ = [
    "CapacityMembership",
    "CertificateV2",
    "DKTPropagationResult",
    "PFAState",
    "certify_capacity_membership",
    "certify_from_propagated_scores",
    "merge_equivalent_states",
    "propagate_dkt_local_options",
    "propagate_pfa_local_options",
    "pfa_ingest_event",
    "pfa_next_step_proba",
    "pfa_state_from_history",
    "robust_rank_bounds",
]
