"""PFA-Option: option-aware generalized logistic / PFA family for Eedi.

EEDI-R0: prior success/failure counts only.
EEDI-R1: additionally uses prespecified option/distractor-history covariates.
Current target answer is excluded from all history features.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class PFAOptionParams:
    beta: float = 0.0
    gamma: float = 0.25
    rho: float = -0.25
    # Option-history covariates (R1 only; must be zeroed for R0)
    eta_opt1: float = 0.0
    eta_opt2: float = 0.0
    eta_opt3: float = 0.0
    eta_opt4: float = 0.0


@dataclass(frozen=True)
class PFAOptionConfig:
    representation: str  # EEDI-R0 or EEDI-R1
    use_option_covariates: bool
    max_history: int = 10_000

    def config_hash(self) -> str:
        material = (
            f"PFA-Option|{self.representation}|opt={int(self.use_option_covariates)}"
            f"|max_h={self.max_history}"
        ).encode()
        return hashlib.sha256(material).hexdigest()


class PFAOption:
    """Honest name: not vanilla PFA when option covariates are active."""

    name = "PFA-Option"

    def __init__(
        self,
        cfg: PFAOptionConfig,
        params: PFAOptionParams | None = None,
    ) -> None:
        self.cfg = cfg
        if cfg.representation == "EEDI-R0" and cfg.use_option_covariates:
            raise ValueError("EEDI-R0 must not enable option covariates")
        if cfg.representation == "EEDI-R1" and not cfg.use_option_covariates:
            raise ValueError("EEDI-R1 must enable option covariates")
        self.params = params or PFAOptionParams()

    def encode_features(
        self,
        history: Sequence[Mapping[str, Any]],
        *,
        subject_id: Any,
    ) -> dict[str, float]:
        """Build features from strictly prior history only (caller enforces prior)."""
        succ = 0.0
        fail = 0.0
        opt = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for ev in history[-self.cfg.max_history :]:
            if str(ev.get("subject_id", ev.get("SubjectId"))) != str(subject_id):
                continue
            y = int(ev.get("is_correct", ev.get("IsCorrect")))
            if y == 1:
                succ += 1.0
            else:
                fail += 1.0
            if self.cfg.use_option_covariates:
                av = int(ev.get("answer_value", ev.get("AnswerValue")))
                if av in opt:
                    opt[av] += 1.0
        feats = {
            "successes": succ,
            "failures": fail,
            "opt1": opt[1] if self.cfg.use_option_covariates else 0.0,
            "opt2": opt[2] if self.cfg.use_option_covariates else 0.0,
            "opt3": opt[3] if self.cfg.use_option_covariates else 0.0,
            "opt4": opt[4] if self.cfg.use_option_covariates else 0.0,
        }
        return feats

    def feature_vector(self, feats: Mapping[str, float]) -> tuple[float, ...]:
        base = (float(feats["successes"]), float(feats["failures"]))
        if self.cfg.representation == "EEDI-R0":
            return base + (0.0, 0.0, 0.0, 0.0)
        return base + (
            float(feats["opt1"]),
            float(feats["opt2"]),
            float(feats["opt3"]),
            float(feats["opt4"]),
        )

    def predict_proba_from_feats(self, feats: Mapping[str, float]) -> float:
        p = self.params
        logit = (
            p.beta
            + p.gamma * math.log1p(feats["successes"])
            + p.rho * math.log1p(feats["failures"])
        )
        if self.cfg.use_option_covariates:
            logit += (
                p.eta_opt1 * feats["opt1"]
                + p.eta_opt2 * feats["opt2"]
                + p.eta_opt3 * feats["opt3"]
                + p.eta_opt4 * feats["opt4"]
            )
        return _sigmoid(logit)

    def input_encoder_id(self) -> str:
        return f"pfa_option_encoder|{self.cfg.representation}|{self.cfg.config_hash()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": asdict(self.cfg),
            "config_hash": self.cfg.config_hash(),
            "params": asdict(self.params),
            "input_encoder_id": self.input_encoder_id(),
        }
