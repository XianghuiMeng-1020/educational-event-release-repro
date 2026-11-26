"""Published-method knowledge-tracing modules (DKT, SAKT, AKT).

The classes below reimplement the defining mechanisms that were inspected in
the cited papers and, for AKT, in arghosh/AKT commit 01b7e8d7. They are not
renames of the chronological GRU or the causal token Transformer.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


N_SKILLS = 265  # frozen ids 0..264, with 0 = unknown skill
MAX_HISTORY = 40


def bkt_step(mastery: float, outcome: int, learn: float, guess: float, slip: float) -> tuple[float, float]:
    """Predict correctness, then apply the observed outcome and the learning transition."""
    predict = mastery * (1.0 - slip) + (1.0 - mastery) * guess
    predict = min(max(predict, 1e-8), 1.0 - 1e-8)
    if int(outcome) == 1:
        posterior = mastery * (1.0 - slip) / predict
    else:
        posterior = mastery * slip / (1.0 - predict)
    posterior = min(max(posterior, 0.0), 1.0)
    nxt = posterior + (1.0 - posterior) * learn
    return predict, min(max(nxt, 0.0), 1.0)


def bkt_predict(prior: float, learn: float, guess: float, slip: float, history: np.ndarray) -> float:
    mastery = float(prior)
    for outcome in history:
        _predict, mastery = bkt_step(mastery, int(outcome), learn, guess, slip)
    predict, _mastery = bkt_step(mastery, 0, learn, guess, slip)
    # The returned probability is the pre-update prediction. The dummy outcome is discarded.
    return float(mastery * (1.0 - slip) + (1.0 - mastery) * guess)


def bkt_nll(params: np.ndarray, histories: list[np.ndarray], targets: np.ndarray) -> float:
    prior, learn, guess, slip = [float(value) for value in params]
    loss = 0.0
    for history, target in zip(histories, targets):
        probability = bkt_predict(prior, learn, guess, slip, history)
        probability = min(max(probability, 1e-6), 1.0 - 1e-6)
        loss -= math.log(probability if int(target) == 1 else 1.0 - probability)
    return loss


def bkt_predict_matrix(prior: float, learn: float, guess: float, slip: float, history: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    mastery = np.full(history.shape[0], float(prior), dtype=np.float64)
    for step in range(history.shape[1]):
        active = step < lengths
        outcome = history[:, step]
        predict = mastery * (1.0 - slip) + (1.0 - mastery) * guess
        predict = np.clip(predict, 1e-8, 1.0 - 1e-8)
        posterior_correct = mastery * (1.0 - slip) / predict
        posterior_incorrect = mastery * slip / (1.0 - predict)
        posterior = np.where(outcome == 1, posterior_correct, posterior_incorrect)
        nxt = posterior + (1.0 - posterior) * learn
        mastery = np.where(active, nxt, mastery)
    return mastery * (1.0 - slip) + (1.0 - mastery) * guess


def bkt_nll_matrix(params: np.ndarray, history: np.ndarray, lengths: np.ndarray, targets: np.ndarray) -> float:
    prior, learn, guess, slip = [float(value) for value in params]
    probability = np.clip(bkt_predict_matrix(prior, learn, guess, slip, history, lengths), 1e-6, 1.0 - 1e-6)
    picked = np.where(targets == 1, probability, 1.0 - probability)
    return float(-np.log(picked).sum())


def fit_bkt_skill(history: np.ndarray, lengths: np.ndarray, targets: np.ndarray) -> dict:
    from scipy.optimize import minimize
    start = np.array([0.30, 0.10, 0.10, 0.10], dtype=np.float64)
    bounds = [(1e-3, 1.0 - 1e-3), (1e-3, 1.0 - 1e-3), (1e-3, 0.5), (1e-3, 0.5)]
    result = minimize(bkt_nll_matrix, start, args=(history, lengths, targets), method="L-BFGS-B", bounds=bounds, options={"maxiter": 80, "ftol": 1e-8})
    return {
        "parameters": [float(value) for value in result.x],
        "success": bool(result.success),
        "n_iter": int(result.nit),
        "nll": float(result.fun),
        "n": int(len(targets)),
    }


class DKT(nn.Module):
    """Piech et al. 2015 DKT: LSTM over past skill-response inputs, read out by the current skill."""

    def __init__(self, n_skills: int = N_SKILLS, hidden: int = 200):
        super().__init__()
        self.n_skills = n_skills
        self.hidden = hidden
        self.lstm = nn.LSTM(2 * n_skills, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, n_skills)

    def forward(self, hist_skill: torch.Tensor, hist_correct: torch.Tensor, hist_len: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch, steps = hist_skill.shape
        skill = hist_skill.clamp(0, self.n_skills - 1)
        index = hist_correct.long() * self.n_skills + skill
        eye = torch.zeros(batch, steps, 2 * self.n_skills, device=hist_skill.device)
        valid = torch.arange(steps, device=hist_skill.device)[None, :] < hist_len[:, None]
        eye[valid] = torch.nn.functional.one_hot(index[valid], 2 * self.n_skills).float()
        packed = nn.utils.rnn.pack_padded_sequence(eye, hist_len.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        _output, (hidden, _cell) = self.lstm(packed)
        # Empty histories stay at the zero initial state because pack skips them.
        empty = hist_len == 0
        state = hidden[-1]
        state = torch.where(empty[:, None], torch.zeros_like(state), state)
        logits = self.readout(state)
        chosen = logits.gather(1, query.long().clamp(0, self.n_skills - 1)[:, None]).squeeze(1)
        return torch.sigmoid(chosen)


class SAKTKC(nn.Module):
    """SAKT at concept level: the current skill queries past skill-response interactions."""

    def __init__(self, n_skills: int = N_SKILLS, d_model: int = 64, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.2):
        super().__init__()
        self.n_skills = n_skills
        self.interaction = nn.Embedding(2 * n_skills + 1, d_model, padding_idx=0)
        self.query = nn.Embedding(n_skills + 1, d_model, padding_idx=0)
        self.position = nn.Embedding(MAX_HISTORY + 2, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.out = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hist_skill: torch.Tensor, hist_correct: torch.Tensor, hist_len: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch, steps = hist_skill.shape
        positions = torch.arange(1, steps + 1, device=hist_skill.device)[None, :].expand(batch, steps)
        interaction_id = (hist_correct.long() * self.n_skills + hist_skill.long() + 1).clamp(max=2 * self.n_skills)
        valid = positions <= hist_len[:, None]
        interaction_id = torch.where(valid, interaction_id, torch.zeros_like(interaction_id))
        keys = self.interaction(interaction_id) + self.position(positions.clamp(max=MAX_HISTORY))
        query_pos = (hist_len + 1).clamp(max=MAX_HISTORY + 1)
        query_vector = self.query(query.long() + 1) + self.position(query_pos)
        # A single current-skill query attends only to real past interactions.
        safe_valid = valid.clone()
        empty = hist_len == 0
        safe_valid[empty, 0] = True
        attended, _weights = self.attention(
            query_vector[:, None, :], keys, keys, key_padding_mask=~safe_valid, need_weights=False,
        )
        attended = attended.squeeze(1)
        attended = torch.where(empty[:, None], torch.zeros_like(attended), attended)
        hidden = self.norm1(query_vector + self.dropout(attended))
        hidden = self.norm2(hidden + self.ff(hidden))
        return torch.sigmoid(self.out(hidden).squeeze(1))


class AKTNonRasch(nn.Module):
    """AKT concept-ID variant from arghosh/AKT commit 01b7e8d7, with n_pid = -1.

    Monotonic distance-decay attention, a question encoder, a knowledge encoder,
    and a knowledge retriever are preserved. The current response is not a key
    or value for the prediction position.
    """

    def __init__(self, n_question: int = N_SKILLS, d_model: int = 256, n_blocks: int = 1, n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.05):
        super().__init__()
        self.n_question = n_question
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_embed = nn.Embedding(n_question + 1, d_model)
        self.qa_embed = nn.Embedding(2, d_model)
        self.knowledge = nn.ModuleList([AKTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)])
        self.retriever = nn.ModuleList([AKTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks * 2)])
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.reset()

    def reset(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, skill: torch.Tensor, response: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        question = self.q_embed(skill.long())
        interaction = self.qa_embed(response.long().clamp(0, 1)) + question
        knowledge = interaction
        for block in self.knowledge:
            knowledge = block(knowledge, knowledge, knowledge, real, allow_current=True, feedforward=True)
        hidden = question
        first = True
        for block in self.retriever:
            if first:
                hidden = block(hidden, hidden, hidden, real, allow_current=True, feedforward=False)
                first = False
            else:
                hidden = block(hidden, hidden, knowledge, real, allow_current=False, feedforward=True, zero_first=True)
                first = True
        lengths = real.long().sum(dim=1) - 1
        chosen = hidden[torch.arange(skill.shape[0], device=skill.device), lengths.clamp(min=0)]
        question_now = question[torch.arange(skill.shape[0], device=skill.device), lengths.clamp(min=0)]
        return torch.sigmoid(self.out(torch.cat([chosen, question_now], dim=-1)).squeeze(-1))


class AKTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = AKTAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, real, allow_current: bool, feedforward: bool, zero_first: bool = False):
        attended = self.attention(query, key, value, real, allow_current, zero_first)
        hidden = self.norm1(query + self.dropout(attended))
        if feedforward:
            hidden = self.norm2(hidden + self.dropout(self.ff(hidden)))
        return hidden


class AKTAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        nn.init.xavier_uniform_(self.gammas)

    def forward(self, query, key, value, real, allow_current: bool, zero_first: bool):
        batch, seqlen, _dim = query.shape
        k = self.k_linear(key).view(batch, seqlen, self.n_heads, self.d_k).transpose(1, 2)
        q = self.k_linear(query).view(batch, seqlen, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch, seqlen, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        positions = torch.arange(seqlen, device=query.device)
        distance = (positions[None, :] - positions[:, None]).abs().float()
        can = positions[None, :] <= positions[:, None] if allow_current else positions[None, :] < positions[:, None]
        can = can[None, None, :, :] & real[:, None, None, :]
        blocked = ~can
        with torch.no_grad():
            base = scores.masked_fill(blocked, -1e32)
            weights = torch.softmax(base, dim=-1) * can.float()
            cumulative = torch.cumsum(weights, dim=-1)
            total = weights.sum(dim=-1, keepdim=True)
            dist_scores = torch.clamp((total - cumulative) * distance, min=0.0).sqrt()
        gamma = -F.softplus(self.gammas).view(1, self.n_heads, 1, 1)
        effect = torch.clamp(torch.exp(torch.clamp(dist_scores * gamma, min=-20.0, max=20.0)), min=1e-5, max=1e5)
        scores = scores * effect
        scores = scores.masked_fill(blocked, -1e32)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(can, weights, torch.zeros_like(weights))
        if zero_first:
            weights = weights.clone()
            weights[:, :, 0, :] = 0.0
        weights = self.dropout(weights)
        mixed = torch.matmul(weights, v)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seqlen, -1)
        return self.out_proj(mixed)
