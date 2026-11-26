"""Categorical-response DKT for Eedi prospective study.

EEDI-R0 token: (question, binary correctness)
EEDI-R1 token: (question, selected answer option)
Same LSTM architecture family; target = next-step correctness under temporal contract.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CategoricalDKTConfig:
    representation: str  # EEDI-R0 or EEDI-R1
    n_questions: int
    emb_dim: int = 64
    hidden: int = 64
    dropout: float = 0.1
    max_history: int = 200
    tuning_budget_epochs: int = 5

    @property
    def n_response_codes(self) -> int:
        return 2 if self.representation == "EEDI-R0" else 4

    @property
    def n_tokens(self) -> int:
        # token 0 = pad; others = question × response_code
        return 1 + self.n_questions * self.n_response_codes

    def config_hash(self) -> str:
        material = (
            f"CategoricalDKT|{self.representation}|nq={self.n_questions}"
            f"|emb={self.emb_dim}|h={self.hidden}|drop={self.dropout}"
            f"|max_h={self.max_history}|epochs={self.tuning_budget_epochs}"
        ).encode()
        return hashlib.sha256(material).hexdigest()


def interaction_token(
    question_id: int,
    *,
    representation: str,
    is_correct: int | None = None,
    answer_value: int | None = None,
    n_questions: int,
) -> int:
    """Encode observed interaction. Token 0 reserved for pad."""
    q = int(question_id)
    if q <= 0 or q > n_questions:
        return 0
    if representation == "EEDI-R0":
        code = 1 if int(is_correct) == 1 else 0
        n_codes = 2
    elif representation == "EEDI-R1":
        av = int(answer_value)
        if av not in (1, 2, 3, 4):
            return 0
        code = av - 1
        n_codes = 4
    else:
        raise ValueError(representation)
    return (q - 1) * n_codes + code + 1


def build_categorical_dkt_sequences(
    question_ids: Sequence[int],
    *,
    representation: str,
    is_correct: Sequence[int],
    answer_values: Sequence[int] | None,
    n_questions: int,
    max_len: int,
) -> dict[str, Any]:
    qs = [int(x) for x in question_ids]
    ys = [int(x) for x in is_correct]
    avs = [int(x) for x in answer_values] if answer_values is not None else [0] * len(qs)
    T = len(qs)
    if T < 2:
        return {
            "input_tokens": [0] * max_len,
            "target_correct": [-1] * max_len,
            "query_questions": [0] * max_len,
            "padding_mask": [False] * max_len,
            "n_valid": 0,
            "representation": representation,
        }
    inp: list[int] = []
    tgt: list[int] = []
    qry: list[int] = []
    for t in range(T - 1):
        inp.append(
            interaction_token(
                qs[t],
                representation=representation,
                is_correct=ys[t],
                answer_value=avs[t],
                n_questions=n_questions,
            )
        )
        tgt.append(ys[t + 1])
        qry.append(qs[t + 1])
    n_valid = min(len(inp), max_len)
    while len(inp) < max_len:
        inp.append(0)
        tgt.append(-1)
        qry.append(0)
    inp, tgt, qry = inp[:max_len], tgt[:max_len], qry[:max_len]
    return {
        "input_tokens": inp,
        "target_correct": tgt,
        "query_questions": qry,
        "padding_mask": [i < n_valid for i in range(max_len)],
        "n_valid": n_valid,
        "representation": representation,
        "tensor_fingerprint": hashlib.sha256(
            (",".join(map(str, inp[:n_valid]))).encode()
        ).hexdigest(),
    }


class CategoricalDKT(nn.Module):
    name = "CategoricalDKT"

    def __init__(self, cfg: CategoricalDKTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.n_tokens, cfg.emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(cfg.emb_dim, cfg.hidden, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.output = nn.Linear(cfg.hidden, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        hidden, _ = self.lstm(emb)
        return self.output(self.dropout(hidden)).squeeze(-1)

    def predict_proba(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(tokens))

    def input_encoder_id(self) -> str:
        return f"cat_dkt_encoder|{self.cfg.representation}|{self.cfg.config_hash()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": asdict(self.cfg),
            "config_hash": self.cfg.config_hash(),
            "input_encoder_id": self.input_encoder_id(),
            "n_tokens": self.cfg.n_tokens,
        }
