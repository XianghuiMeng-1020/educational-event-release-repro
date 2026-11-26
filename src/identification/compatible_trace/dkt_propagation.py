"""CategoricalDKT compatible-trace propagation (Contract A, post-ingest).

Under local completion, |C_dev(z)| is tiny (typically 2–4 options), so exact
complete search over candidate option tokens is feasible: fork from cached
pre-event hidden state, ingest each candidate token, read next-step logit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from src.identification.compatible_trace.certificate_v2 import (
    CertificateV2,
    certify_from_propagated_scores,
)
from src.models.categorical_dkt import CategoricalDKT, interaction_token


@dataclass(frozen=True)
class DKTPropagationResult:
    certificate: CertificateV2
    scores: tuple[float, ...]
    search_mode: str
    n_candidates: int
    forked_from_cache: bool


@torch.no_grad()
def _hidden_after_tokens(model: CategoricalDKT, tokens: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return LSTM (h_n, c_n) after consuming tokens (batch=1)."""
    if not tokens:
        h = torch.zeros(1, 1, model.cfg.hidden)
        c = torch.zeros(1, 1, model.cfg.hidden)
        return h, c
    x = torch.tensor([list(tokens)], dtype=torch.long)
    emb = model.embedding(x)
    _, (h_n, c_n) = model.lstm(emb)
    return h_n, c_n


@torch.no_grad()
def _next_step_proba_from_state(
    model: CategoricalDKT,
    h_n: torch.Tensor,
    c_n: torch.Tensor,
) -> float:
    """Mastery proxy = sigmoid(W h) after ingest (next-step checkpoint)."""
    # h_n: (1, batch=1, hidden)
    logit = model.output(model.dropout(h_n.squeeze(0))).squeeze(-1)
    return float(torch.sigmoid(logit).reshape(-1)[0].item())


@torch.no_grad()
def propagate_dkt_local_options(
    model: CategoricalDKT,
    *,
    prior_tokens: Sequence[int],
    question_id: int,
    is_correct: int,
    compatible_options: Sequence[int],
    released_key: str = "",
    threshold: float = 0.80,
) -> DKTPropagationResult:
    """Exact local search: vary option token at event t, score post-ingest state.

    For EEDI-R0, option identity is not in the token; all candidates collapse to
    the same correctness token — Stable is then exact but scientifically notes
    that R0 cannot carry option semantics (structural for R0 representation,
    not a copied-score artifact of the certificate layer).
    """
    opts = sorted({int(o) for o in compatible_options})
    if not opts:
        cert = certify_from_propagated_scores(
            released_key, [], threshold=threshold, search_mode="exact"
        )
        return DKTPropagationResult(cert, (), "exact", 0, True)

    model.eval()
    h0, c0 = _hidden_after_tokens(model, prior_tokens)
    scores: list[float] = []
    for a in opts:
        if model.cfg.representation == "EEDI-R0":
            tok = interaction_token(
                question_id,
                representation="EEDI-R0",
                is_correct=is_correct,
                n_questions=model.cfg.n_questions,
            )
        else:
            tok = interaction_token(
                question_id,
                representation="EEDI-R1",
                answer_value=a,
                n_questions=model.cfg.n_questions,
            )
        emb = model.embedding(torch.tensor([[tok]], dtype=torch.long))
        _, (h1, c1) = model.lstm(emb, (h0, c0))
        scores.append(_next_step_proba_from_state(model, h1, c1))

    cert = certify_from_propagated_scores(
        released_key,
        scores,
        threshold=threshold,
        search_mode="exact",
    )
    return DKTPropagationResult(
        certificate=cert,
        scores=tuple(scores),
        search_mode="exact",
        n_candidates=len(scores),
        forked_from_cache=True,
    )


@torch.no_grad()
def dkt_exact_replay_sequence_probas(
    model: CategoricalDKT,
    tokens: Sequence[int],
) -> list[float]:
    """Full forward probs at each position (for agreement checks)."""
    model.eval()
    if not tokens:
        return []
    x = torch.tensor([list(tokens)], dtype=torch.long)
    return [float(p) for p in model.predict_proba(x).reshape(-1).tolist()]
