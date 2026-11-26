"""Certificate labels for genuine compatible-trace propagation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from src.policies.mastery_policy import mastery_decision


@dataclass(frozen=True)
class CertificateV2:
    """R2 certificate. Unsupported is separate from action ambiguity.

    Labels:
      unsupported — no development-compatible completion
      stable — all propagated completions induce the same action
      set-valued — ≥2 distinct actions among propagated completions
      uncertified — search incomplete / unbound; must NOT be called stable
      abstain — explicit safety/size rule (optional)
    """

    status: str
    actions: tuple[str, ...]
    scores: tuple[float, ...]
    n_candidates: int
    released_key: str = ""
    search_mode: str = "exact"  # exact | certified_bound | partial
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def certify_from_propagated_scores(
    released_key: str,
    scores: Iterable[float],
    *,
    threshold: float = 0.80,
    search_mode: str = "exact",
    max_set_size: int = 5,
    policy: Callable[[float, float], str] | None = None,
) -> CertificateV2:
    score_list = [float(s) for s in scores]
    decide = policy or mastery_decision
    if not score_list:
        return CertificateV2(
            status="unsupported",
            actions=(),
            scores=(),
            n_candidates=0,
            released_key=released_key,
            search_mode=search_mode,
            rationale="empty_compatible_set",
        )
    actions = tuple(sorted({decide(s, threshold) for s in score_list}))
    if search_mode == "partial":
        # Partial search may detect set-valued, but singleton ≠ Stable.
        if len(actions) > 1:
            status = "set-valued" if len(actions) <= max_set_size else "abstain"
        else:
            status = "uncertified"
        return CertificateV2(
            status=status,
            actions=actions,
            scores=tuple(score_list),
            n_candidates=len(score_list),
            released_key=released_key,
            search_mode=search_mode,
            rationale="partial_search_no_stability_certificate",
        )
    if len(actions) == 1:
        status = "stable"
    elif len(actions) <= max_set_size:
        status = "set-valued"
    else:
        status = "abstain"
    return CertificateV2(
        status=status,
        actions=actions,
        scores=tuple(score_list),
        n_candidates=len(score_list),
        released_key=released_key,
        search_mode=search_mode,
        rationale="exact_or_certified_propagation",
    )
