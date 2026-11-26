"""Shared R7 release-frontier core (frozen protocol helpers)."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

SCHEMAS = ["S0", "S1", "S2", "S3"]
TAU = 0.80
KAPPA = 0.05
MIN_USER_EVENTS = 3
SPLIT_SEED = "wdlf-r7-assistments-2012-CONFIRMATORY-v2-TO_FREEZE"
DEV_FRAC, CAL_FRAC = 0.70, 0.15
NEURAL_SEED = 17
MAX_SEQ_ACTIONS = 32
MAX_HIST_TOKENS = 64
MAX_TRAIN_EVENTS_PER_USER = 40
MAX_EVAL_PREFIX_EVENTS = 40


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def user_bucket(user_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SEED}|{user_id}".encode()).hexdigest()
    u = int(digest[:12], 16) / float(16**12)
    if u < DEV_FRAC:
        return "development"
    if u < DEV_FRAC + CAL_FRAC:
        return "calibration"
    return "confirmatory"


def bin_correct(raw: Any) -> Optional[int]:
    try:
        v = float(raw)
    except Exception:
        return None
    if v >= 1.0:
        return 1
    if v <= 0.0:
        return 0
    return None


@dataclass(frozen=True)
class Event:
    user_id: str
    problem_id: str
    skill_id: str
    correct: int
    hint_count: int
    attempt_count: int
    first_action: str
    action_types: tuple[str, ...]
    action_timestamps: tuple[str, ...]
    split: str


def release_key(ev: Event, schema: str) -> tuple:
    if schema == "S0":
        return ("S0", ev.problem_id, str(ev.correct))
    if schema == "S1":
        return ("S1", ev.problem_id, str(ev.correct), str(ev.hint_count), str(ev.attempt_count))
    if schema == "S2":
        return (
            "S2",
            ev.problem_id,
            str(ev.correct),
            str(ev.hint_count),
            str(ev.attempt_count),
            str(ev.first_action),
        )
    if schema == "S3":
        return (
            "S3",
            ev.problem_id,
            str(ev.correct),
            str(ev.hint_count),
            str(ev.attempt_count),
            str(ev.first_action),
            ev.action_types,
        )
    if schema == "S4":
        return (
            "S4",
            ev.problem_id,
            str(ev.correct),
            str(ev.hint_count),
            str(ev.attempt_count),
            str(ev.first_action),
            ev.action_types,
            ev.action_timestamps,
        )
    raise KeyError(schema)


def project_down(key_hi: tuple, from_s: str, to_s: str) -> tuple:
    order = ["S0", "S1", "S2", "S3", "S4"]
    if order.index(to_s) > order.index(from_s):
        raise ValueError("downward only")
    d: dict[str, Any] = {"problem_id": key_hi[1], "correct": key_hi[2]}
    if from_s in {"S1", "S2", "S3", "S4"}:
        d["hint_count"], d["attempt_count"] = key_hi[3], key_hi[4]
    if from_s in {"S2", "S3", "S4"}:
        d["first_action"] = key_hi[5]
    if from_s in {"S3", "S4"}:
        d["action_types"] = key_hi[6]
    if from_s == "S4":
        d["action_timestamps"] = key_hi[7]
    if to_s == "S0":
        return ("S0", d["problem_id"], d["correct"])
    if to_s == "S1":
        return ("S1", d["problem_id"], d["correct"], d["hint_count"], d["attempt_count"])
    if to_s == "S2":
        return (
            "S2",
            d["problem_id"],
            d["correct"],
            d["hint_count"],
            d["attempt_count"],
            d["first_action"],
        )
    if to_s == "S3":
        return (
            "S3",
            d["problem_id"],
            d["correct"],
            d["hint_count"],
            d["attempt_count"],
            d["first_action"],
            d["action_types"],
        )
    return key_hi


def rich_signature(ev: Event) -> tuple:
    return (
        ev.problem_id,
        ev.correct,
        ev.hint_count,
        ev.attempt_count,
        ev.first_action,
        ev.action_types,
        ev.action_timestamps,
    )


def syn_from_sig(ev: Event, sig: tuple) -> Event:
    return Event(
        user_id=ev.user_id,
        problem_id=sig[0],
        skill_id=ev.skill_id,
        correct=int(sig[1]),
        hint_count=int(sig[2]),
        attempt_count=int(sig[3]),
        first_action=str(sig[4]),
        action_types=tuple(sig[5]),
        action_timestamps=tuple(sig[6]),
        split=ev.split,
    )


def semantic_cost(ev: Event, schema: str) -> dict[str, float]:
    n_fields = {"S0": 2, "S1": 4, "S2": 5, "S3": 6, "S4": 7}[schema]
    payload = 0.0
    if schema == "S2":
        payload = 1.0
    elif schema in {"S3", "S4"}:
        payload = float(len(ev.action_types))
    if schema == "S0":
        s = f"S0|{ev.problem_id}|{ev.correct}"
    elif schema == "S1":
        s = f"S1|{ev.problem_id}|{ev.correct}|{ev.hint_count}|{ev.attempt_count}"
    elif schema == "S2":
        s = f"S2|{ev.problem_id}|{ev.correct}|{ev.hint_count}|{ev.attempt_count}|{ev.first_action}"
    elif schema == "S3":
        s = (
            f"S3|{ev.problem_id}|{ev.correct}|{ev.hint_count}|{ev.attempt_count}|"
            f"{ev.first_action}|{'/'.join(ev.action_types)}"
        )
    else:
        s = (
            f"S4|{ev.problem_id}|{ev.correct}|{ev.hint_count}|{ev.attempt_count}|"
            f"{ev.first_action}|{'/'.join(ev.action_types)}|{'/'.join(ev.action_timestamps)}"
        )
    return {
        "n_semantic_fields": float(n_fields),
        "action_payload_len": payload,
        "canonical_utf8_bytes": float(len(s.encode("utf-8"))),
    }


class TransparentActionPFA:
    def __init__(self) -> None:
        self.skill_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "n": 0.0,
                "c": 0.0,
                "hints": 0.0,
                "attempts": 0.0,
                "ans": 0.0,
                "hint_before_answer": 0.0,
                "first_is_hint": 0.0,
            }
        )

    def clone(self) -> "TransparentActionPFA":
        m = TransparentActionPFA()
        for k, v in self.skill_stats.items():
            m.skill_stats[k] = dict(v)
        return m

    def update(self, ev: Event) -> None:
        st = self.skill_stats[ev.skill_id]
        st["n"] += 1.0
        st["c"] += float(ev.correct)
        st["hints"] += float(ev.hint_count)
        st["attempts"] += float(max(ev.attempt_count, 1))
        st["ans"] += float(sum(1 for t in ev.action_types if t == "answer"))
        hba = 0
        for t in ev.action_types:
            if t == "answer":
                break
            if t == "hint":
                hba += 1
        st["hint_before_answer"] += float(hba)
        st["first_is_hint"] += float(1.0 if ev.action_types and ev.action_types[0] == "hint" else 0.0)

    def mastery(self, skill_id: str) -> float:
        st = self.skill_stats[skill_id]
        if st["n"] <= 0:
            return 0.5
        base = (st["c"] + 1.0) / (st["n"] + 2.0)
        score = (
            base
            - 0.05 * st["hints"] / st["n"]
            - 0.04 * max(st["attempts"] / st["n"] - 1.0, 0.0)
            - 0.03 * max(st["ans"] / st["n"] - 1.0, 0.0)
            - 0.06 * st["hint_before_answer"] / st["n"]
            - 0.05 * st["first_is_hint"] / st["n"]
        )
        return max(0.01, min(0.99, score))


class ActionTokenGRU(nn.Module):
    def __init__(self, n_skills: int, n_atypes: int, emb: int = 32, hid: int = 64):
        super().__init__()
        self.skill_emb = nn.Embedding(n_skills + 1, emb, padding_idx=0)
        self.atype_emb = nn.Embedding(n_atypes + 3, emb, padding_idx=0)
        self.gru = nn.GRU(emb * 2, hid, batch_first=True)
        self.out = nn.Linear(hid, 1)
        self.hid = hid
        self.outcome0 = n_atypes + 1
        self.outcome1 = n_atypes + 2

    def embed(self, skill_ids: torch.Tensor, atype_ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.skill_emb(skill_ids), self.atype_emb(atype_ids)], dim=-1)

    def forward_from_state(
        self, skill_ids: torch.Tensor, atype_ids: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(skill_ids, atype_ids)
        out, h = self.gru(x, h0)
        score = torch.sigmoid(self.out(out[:, -1, :])).squeeze(-1)
        return score, h


class NeuralPilot:
    def __init__(self, device: Optional[str] = None) -> None:
        self.skill2i = {"<pad>": 0}
        self.atype2i = {"<pad>": 0}
        self.model: Optional[ActionTokenGRU] = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def fit(self, train_events: list[Event], epochs: int = 2, user_cap: int = 3000) -> dict[str, Any]:
        random.seed(NEURAL_SEED)
        torch.manual_seed(NEURAL_SEED)
        for e in train_events:
            if e.skill_id not in self.skill2i:
                self.skill2i[e.skill_id] = len(self.skill2i)
            for t in e.action_types:
                if t not in self.atype2i:
                    self.atype2i[t] = len(self.atype2i)
        n_atypes = len(self.atype2i) - 1
        self.model = ActionTokenGRU(len(self.skill2i) - 1, n_atypes).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        by_user: dict[str, list[Event]] = defaultdict(list)
        for e in train_events:
            by_user[e.user_id].append(e)
        users = [u for u, h in by_user.items() if len(h) >= 2]
        loss_hist = []
        self.model.train()
        for _ in range(epochs):
            random.shuffle(users)
            total = 0.0
            n = 0
            for u in users[:user_cap]:
                hist = by_user[u][-MAX_TRAIN_EVENTS_PER_USER:]
                for t, ev in enumerate(hist):
                    sk, at = self._tensorize(self._tokens(hist[:t], ev, False))
                    if sk is None:
                        continue
                    pred, _ = self.model.forward_from_state(sk, at, None)
                    y = torch.tensor([float(ev.correct)], device=self.device)
                    loss = nn.functional.binary_cross_entropy(pred, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += float(loss.item())
                    n += 1
            loss_hist.append(total / max(n, 1))
        self.model.eval()
        return {
            "epochs": epochs,
            "user_cap": min(len(users), user_cap),
            "max_train_events_per_user": MAX_TRAIN_EVENTS_PER_USER,
            "n_skills": len(self.skill2i) - 1,
            "n_action_types": n_atypes,
            "loss_hist": loss_hist,
            "param_count": sum(p.numel() for p in self.model.parameters()),
            "seed": NEURAL_SEED,
        }

    def _tokens(self, prefix: list[Event], current: Optional[Event], include_current_outcome: bool):
        assert self.model is not None
        skills: list[int] = []
        atypes: list[int] = []
        for ev in prefix:
            sid = self.skill2i.get(ev.skill_id, 0)
            for t in ev.action_types[:MAX_SEQ_ACTIONS]:
                skills.append(sid)
                atypes.append(self.atype2i.get(t, 0))
            skills.append(sid)
            atypes.append(self.model.outcome1 if ev.correct == 1 else self.model.outcome0)
        if current is not None:
            sid = self.skill2i.get(current.skill_id, 0)
            for t in current.action_types[:MAX_SEQ_ACTIONS]:
                skills.append(sid)
                atypes.append(self.atype2i.get(t, 0))
            if include_current_outcome:
                skills.append(sid)
                atypes.append(self.model.outcome1 if current.correct == 1 else self.model.outcome0)
        if not skills:
            return [], []
        return skills[-MAX_HIST_TOKENS:], atypes[-MAX_HIST_TOKENS:]

    def _tensorize(self, pair):
        skills, atypes = pair
        if not skills:
            return None, None
        return (
            torch.tensor([skills], dtype=torch.long, device=self.device),
            torch.tensor([atypes], dtype=torch.long, device=self.device),
        )

    @torch.no_grad()
    def mastery_branch(self, prefix: list[Event], cand: Event) -> float:
        sk, at = self._tensorize(self._tokens(prefix, cand, False))
        if sk is None:
            return 0.5
        pred, _ = self.model.forward_from_state(sk, at, None)
        return float(pred.item())

    @torch.no_grad()
    def mastery_branches_shared_prefix(self, prefix: list[Event], cands: list[Event]) -> list[float]:
        """Exact multi-candidate eval equivalent to joint last-MAX_HIST_TOKENS encoding.

        Batches candidates of equal sequence length (exact; same logits as serial).
        """
        if not cands:
            return []
        assert self.model is not None
        pref_skills, pref_atypes = self._tokens(prefix, None, False)
        out: list[float] = [0.5] * len(cands)
        by_len: dict[int, list[tuple[int, list[int], list[int]]]] = defaultdict(list)
        for i, cand in enumerate(cands):
            c_skills: list[int] = []
            c_atypes: list[int] = []
            sid = self.skill2i.get(cand.skill_id, 0)
            for t in cand.action_types[:MAX_SEQ_ACTIONS]:
                c_skills.append(sid)
                c_atypes.append(self.atype2i.get(t, 0))
            if not c_skills:
                continue
            keep = MAX_HIST_TOKENS - len(c_skills)
            if keep > 0 and pref_skills:
                ps, pa = pref_skills[-keep:], pref_atypes[-keep:]
            else:
                ps, pa = [], []
            skills = ps + c_skills
            atypes = pa + c_atypes
            by_len[len(skills)].append((i, skills, atypes))
        for items in by_len.values():
            sk = torch.tensor([x[1] for x in items], dtype=torch.long, device=self.device)
            at = torch.tensor([x[2] for x in items], dtype=torch.long, device=self.device)
            pred, _ = self.model.forward_from_state(sk, at, None)
            for j, (i, _, _) in enumerate(items):
                out[i] = float(pred[j].item())
        return out

    def state_dict_bundle(self) -> dict[str, Any]:
        assert self.model is not None
        return {
            "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "skill2i": self.skill2i,
            "atype2i": self.atype2i,
            "seed": NEURAL_SEED,
        }

    def load_bundle(self, bundle: dict[str, Any]) -> None:
        self.skill2i = dict(bundle["skill2i"])
        self.atype2i = dict(bundle["atype2i"])
        n_atypes = len(self.atype2i) - 1
        self.model = ActionTokenGRU(len(self.skill2i) - 1, n_atypes).to(self.device)
        self.model.load_state_dict(bundle["model"])
        self.model.eval()


def p2_certificate(intervals: dict[str, tuple[float, float]], kappa: float = KAPPA) -> dict[str, Any]:
    eligible = sorted(intervals.items(), key=lambda kv: kv[0])
    n_el = len(eligible)
    B = int(round(kappa * n_el)) if n_el else 0
    if n_el and B < 1:
        B = 1
    gu_in = gu_out = unresolved = 0
    if n_el and B:
        iv = [(uid, lo, hi) for uid, (lo, hi) in eligible]
        for uid, lo, hi in iv:
            best = 1 + sum(1 for u2, lo2, hi2 in iv if u2 != uid and lo2 > hi)
            worst = 1 + sum(1 for u2, lo2, hi2 in iv if u2 != uid and hi2 >= lo)
            if worst <= B:
                gu_in += 1
            elif best > B:
                gu_out += 1
            else:
                unresolved += 1
    return {
        "eligible": n_el,
        "seats_B": B,
        "guaranteed_in": gu_in,
        "guaranteed_out": gu_out,
        "unresolved": unresolved,
    }


def evaluate_split(
    events: list[Event],
    eval_split: str,
    dicts: dict[str, dict[tuple, Counter]],
    dict_users: Optional[dict[str, dict[tuple, set]]],
    model_name: str,
    neural: Optional[NeuralPilot] = None,
    tau: float = TAU,
    kappa: float = KAPPA,
    return_intervals: bool = False,
) -> dict[str, Any]:
    eval_users = sorted({e.user_id for e in events if e.split == eval_split})
    eval_user_set = set(eval_users)
    user_hist: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        if e.user_id in eval_user_set:
            user_hist[e.user_id].append(e)
    n_all = len(eval_users)
    out: dict[str, Any] = {}
    all_intervals: dict[str, dict[str, tuple[float, float]]] = {}
    for s in SCHEMAS:
        stable = set_valued = unsupported = 0
        costs: list[dict[str, float]] = []
        intervals: dict[str, tuple[float, float]] = {}
        for ui, uid in enumerate(eval_users):
            if ui and ui % 500 == 0:
                print(
                    f"EVAL_PROGRESS model={model_name} schema={s} users={ui}/{n_all}",
                    flush=True,
                )
            hist = user_hist[uid]
            tmodel = TransparentActionPFA()
            last_interval = None
            for t, ev in enumerate(hist):
                costs.append(semantic_cost(ev, s))
                k = release_key(ev, s)
                entry = dicts[s].get(k)
                # Supports classic Counter dicts or compact {'sigs','n_users'} entries.
                if entry is None:
                    unsupported += 1
                    if model_name == "transparent":
                        tmodel.update(ev)
                    continue
                if isinstance(entry, dict) and "sigs" in entry:
                    sigs = entry["sigs"]
                    n_users = entry["n_users"]
                else:
                    sigs = list(entry.keys())
                    n_users = len(dict_users[s][k]) if dict_users is not None else 2
                if n_users < 2 and len(sigs) < 2:
                    unsupported += 1
                    if model_name == "transparent":
                        tmodel.update(ev)
                    continue
                decisions = set()
                risks = []
                syns = [syn_from_sig(ev, sig) for sig in sigs]
                if model_name == "transparent":
                    # Exact collapse: TransparentActionPFA ignores timestamps; unique
                    # update fingerprints yield identical mastery.
                    seen_fp: set[tuple] = set()
                    uniq_syns: list[Event] = []
                    for syn in syns:
                        fp = (
                            syn.correct,
                            syn.hint_count,
                            syn.attempt_count,
                            syn.action_types,
                        )
                        if fp in seen_fp:
                            continue
                        seen_fp.add(fp)
                        uniq_syns.append(syn)
                    for syn in uniq_syns:
                        # In-place update/restore (exact; avoids full-state clone cost).
                        st = tmodel.skill_stats[ev.skill_id]
                        backup = dict(st)
                        tmodel.update(syn)
                        p = tmodel.mastery(ev.skill_id)
                        tmodel.skill_stats[ev.skill_id] = backup
                        decisions.add("progress" if p >= tau else "review")
                        risks.append(1.0 - p)
                else:
                    assert neural is not None
                    # Exact collapse: neural branch tokens ignore timestamps.
                    seen_fp = set()
                    uniq_syns = []
                    for syn in syns:
                        fp = (syn.skill_id, syn.action_types[:MAX_SEQ_ACTIONS])
                        if fp in seen_fp:
                            continue
                        seen_fp.add(fp)
                        uniq_syns.append(syn)
                    # Frozen history truncation for neural prefix (token cap also applies).
                    prefix = hist[max(0, t - MAX_EVAL_PREFIX_EVENTS) : t]
                    ps = neural.mastery_branches_shared_prefix(prefix, uniq_syns)
                    for p in ps:
                        decisions.add("progress" if p >= tau else "review")
                        risks.append(1.0 - p)
                if len(decisions) == 1:
                    stable += 1
                else:
                    set_valued += 1
                last_interval = (min(risks), max(risks))
                if model_name == "transparent":
                    tmodel.update(ev)
            if last_interval is not None:
                intervals[uid] = last_interval
        n_events = stable + set_valued + unsupported
        p2 = p2_certificate(intervals, kappa=kappa)
        no_ckpt = n_all - p2["eligible"]
        mean_cost = {
            k: sum(c[k] for c in costs) / len(costs) if costs else None
            for k in ("n_semantic_fields", "action_payload_len", "canonical_utf8_bytes")
        }
        out[s] = {
            "model": model_name,
            "split": eval_split,
            "n_events": n_events,
            "supported": stable + set_valued,
            "unsupported": unsupported,
            "support_rate": (stable + set_valued) / n_events if n_events else None,
            "stable": stable,
            "set_valued": set_valued,
            "stable_given_supported": stable / (stable + set_valued) if (stable + set_valued) else None,
            "set_valued_given_supported": set_valued / (stable + set_valued) if (stable + set_valued) else None,
            "event_certified_stable_coverage": stable / n_events if n_events else None,
            "event_noncertification_rate": (unsupported + set_valued) / n_events if n_events else None,
            "p2_full_users": n_all,
            "p2_eligible_users": p2["eligible"],
            "p2_eligibility_rate": p2["eligible"] / n_all if n_all else None,
            "p2_no_checkpoint_users": no_ckpt,
            "p2_seats_B": p2["seats_B"],
            "p2_guaranteed_in": p2["guaranteed_in"],
            "p2_guaranteed_out": p2["guaranteed_out"],
            "p2_unresolved": p2["unresolved"],
            "p2_unresolved_rate_eligible": p2["unresolved"] / p2["eligible"] if p2["eligible"] else None,
            "p2_all_user_noncertification_rate": (p2["unresolved"] + no_ckpt) / n_all if n_all else None,
            "cost": mean_cost,
        }
        all_intervals[s] = intervals
    if return_intervals:
        return {"metrics": out, "intervals": all_intervals}
    return out


def pareto_analysis(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Dominate on cost (lower better), coverage (higher better), noncerts (lower better)."""

    def dominates(a, b):
        # a dominates b if a is <= cost, >= coverage, <= both noncerts, and strict in one
        ca, cb = a["cost_fields"], b["cost_fields"]
        cov_a, cov_b = a["event_certified_stable_coverage"], b["event_certified_stable_coverage"]
        ena, enb = a["event_noncertification_rate"], b["event_noncertification_rate"]
        pna, pnb = a["p2_all_user_noncertification_rate"], b["p2_all_user_noncertification_rate"]
        weakly = ca <= cb and cov_a >= cov_b and ena <= enb and pna <= pnb
        strict = ca < cb or cov_a > cov_b or ena < enb or pna < pnb
        return weakly and strict

    nondom = []
    dom = []
    for i, a in enumerate(cells):
        if any(dominates(cells[j], a) for j in range(len(cells)) if j != i):
            dom.append(a["id"])
        else:
            nondom.append(a["id"])
    return {"nondominated": nondom, "dominated": dom, "convention": "lower_cost_higher_coverage_lower_noncert"}
