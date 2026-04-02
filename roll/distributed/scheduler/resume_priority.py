import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ResumeScoreWeights:
    age: float = 1.0
    hist: float = 0.001
    aff: float = 0.5
    load: float = 0.5
    fair: float = 0.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "ResumeScoreWeights":
        cfg = cfg or {}
        return cls(
            age=float(cfg.get("age", cls.age)),
            hist=float(cfg.get("hist", cls.hist)),
            aff=float(cfg.get("aff", cls.aff)),
            load=float(cfg.get("load", cls.load)),
            fair=float(cfg.get("fair", cls.fair)),
        )


def compute_resume_score(
    *,
    pause_age_s: float,
    history_len_tokens: float,
    is_last_backend: float,
    worker_load: float,
    fairness_bonus: float,
    weights: ResumeScoreWeights,
) -> float:
    return (
        weights.age * math.log1p(max(0.0, pause_age_s))
        - weights.hist * history_len_tokens
        + weights.aff * is_last_backend
        - weights.load * worker_load
        + weights.fair * fairness_bonus
    )
