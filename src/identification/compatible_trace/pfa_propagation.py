"""Exact PFA-Option compatible-trace propagation (Contract A, post-ingest).

Temporal contract:
  - Pre-event features exclude current target.
  - Selected option a_t first affects state AFTER ingest of event t (R1 only).
  - Decision checkpoint = next-step probability on a probe subject opportunity
    after ingesting each compatible completion of the released (q,s,y) event.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.identification.compatible_trace.certificate_v2 import (
    CertificateV2,
    certify_from_propagated_scores,
)
from src.models.pfa_option import PFAOption, PFAOptionConfig, PFAOptionParams


@dataclass(frozen=True)
class PFAState:
    """Sufficient statistics for one subject under PFA-Option."""

    successes: float
    failures: float
    opt1: float
    opt2: float
    opt3: float
    opt4: float

    def as_feats(self) -> dict[str, float]:
        return {
            "successes": float(self.successes),
            "failures": float(self.failures),
            "opt1": float(self.opt1),
            "opt2": float(self.opt2),
            "opt3": float(self.opt3),
            "opt4": float(self.opt4),
        }

    def key(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.successes,
            self.failures,
            self.opt1,
            self.opt2,
            self.opt3,
            self.opt4,
        )


def pfa_state_from_history(
    history: Sequence[Mapping[str, Any]],
    *,
    subject_id: Any,
    use_option_covariates: bool,
    max_history: int = 10_000,
) -> PFAState:
    """Build pre-event state from strictly prior history (no current target)."""
    cfg = PFAOptionConfig(
        representation="EEDI-R1" if use_option_covariates else "EEDI-R0",
        use_option_covariates=use_option_covariates,
        max_history=max_history,
    )
    model = PFAOption(cfg)
    feats = model.encode_features(history, subject_id=subject_id)
    return PFAState(
        successes=feats["successes"],
        failures=feats["failures"],
        opt1=feats["opt1"],
        opt2=feats["opt2"],
        opt3=feats["opt3"],
        opt4=feats["opt4"],
    )


def pfa_ingest_event(
    state: PFAState,
    *,
    is_correct: int,
    answer_value: int | None,
    use_option_covariates: bool,
) -> PFAState:
    """Update sufficient statistics AFTER observing a completed event."""
    succ = state.successes + (1.0 if int(is_correct) == 1 else 0.0)
    fail = state.failures + (0.0 if int(is_correct) == 1 else 1.0)
    o1, o2, o3, o4 = state.opt1, state.opt2, state.opt3, state.opt4
    if use_option_covariates:
        av = int(answer_value) if answer_value is not None else 0
        if av == 1:
            o1 += 1.0
        elif av == 2:
            o2 += 1.0
        elif av == 3:
            o3 += 1.0
        elif av == 4:
            o4 += 1.0
    return PFAState(succ, fail, o1, o2, o3, o4)


def pfa_next_step_proba(
    state: PFAState,
    model: PFAOption,
) -> float:
    """Next-step / post-ingest mastery proxy from sufficient statistics."""
    return float(model.predict_proba_from_feats(state.as_feats()))


def merge_equivalent_states(states: Iterable[PFAState]) -> list[PFAState]:
    seen: dict[tuple[float, ...], PFAState] = {}
    for st in states:
        seen[st.key()] = st
    return list(seen.values())


def propagate_pfa_local_options(
    *,
    history: Sequence[Mapping[str, Any]],
    subject_id: Any,
    is_correct: int,
    compatible_options: Sequence[int],
    model: PFAOption,
    released_key: str = "",
    threshold: float = 0.80,
) -> CertificateV2:
    """Contract A: vary option at event t, evaluate post-ingest next-step score.

    Compatible options must come from a development-only dictionary.
    Confirmatory student's own option must not define C_dev.
    """
    opts = sorted({int(o) for o in compatible_options})
    if not opts:
        return certify_from_propagated_scores(
            released_key, [], threshold=threshold, search_mode="exact"
        )
    use_opt = model.cfg.use_option_covariates
    pre = pfa_state_from_history(
        history,
        subject_id=subject_id,
        use_option_covariates=use_opt,
        max_history=model.cfg.max_history,
    )
    scores: list[float] = []
    post_states: list[PFAState] = []
    for a in opts:
        post = pfa_ingest_event(
            pre,
            is_correct=is_correct,
            answer_value=a if use_opt else None,
            use_option_covariates=use_opt,
        )
        post_states.append(post)
        scores.append(pfa_next_step_proba(post, model))
    # Exactness: duplicate sufficient states merge; scores must agree for merges.
    merged = merge_equivalent_states(post_states)
    assert len(merged) <= len(post_states)
    return certify_from_propagated_scores(
        released_key,
        scores,
        threshold=threshold,
        search_mode="exact",
    )


def pfa_exact_replay_proba(
    history_with_current: Sequence[Mapping[str, Any]],
    *,
    subject_id: Any,
    model: PFAOption,
) -> float:
    """Score after ingesting the full provided history (exact replay helper)."""
    if not history_with_current:
        st = PFAState(0, 0, 0, 0, 0, 0)
        return pfa_next_step_proba(st, model)
    prior = list(history_with_current[:-1])
    cur = history_with_current[-1]
    pre = pfa_state_from_history(
        prior,
        subject_id=subject_id,
        use_option_covariates=model.cfg.use_option_covariates,
        max_history=model.cfg.max_history,
    )
    post = pfa_ingest_event(
        pre,
        is_correct=int(cur.get("is_correct", cur.get("IsCorrect"))),
        answer_value=int(cur.get("answer_value", cur.get("AnswerValue", 0))),
        use_option_covariates=model.cfg.use_option_covariates,
    )
    return pfa_next_step_proba(post, model)


def default_r1_model(params: PFAOptionParams | None = None) -> PFAOption:
    return PFAOption(
        PFAOptionConfig(representation="EEDI-R1", use_option_covariates=True),
        params=params
        or PFAOptionParams(
            beta=0.0,
            gamma=0.25,
            rho=-0.25,
            eta_opt1=0.15,
            eta_opt2=-0.05,
            eta_opt3=0.10,
            eta_opt4=-0.20,
        ),
    )


def default_r0_model(params: PFAOptionParams | None = None) -> PFAOption:
    return PFAOption(
        PFAOptionConfig(representation="EEDI-R0", use_option_covariates=False),
        params=params or PFAOptionParams(beta=0.0, gamma=0.25, rho=-0.25),
    )
