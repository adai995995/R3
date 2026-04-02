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


@dataclass
class RequestPriorityWeights:
    age: float = 1.0
    hist: float = 0.001
    hit: float = 0.5
    rebuild: float = 0.0
    fair: float = 0.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "RequestPriorityWeights":
        cfg = cfg or {}
        return cls(
            age=float(cfg.get("age", cls.age)),
            hist=float(cfg.get("hist", cls.hist)),
            hit=float(cfg.get("hit", cls.hit)),
            rebuild=float(cfg.get("rebuild", cls.rebuild)),
            fair=float(cfg.get("fair", cls.fair)),
        )


def compute_request_priority(
    *,
    pause_age_s: float,
    history_len_tokens: float,
    hit_prob: float,
    rebuild_cost: float,
    fairness_bonus: float,
    weights: RequestPriorityWeights,
) -> float:
    return (
        weights.age * math.log1p(max(0.0, pause_age_s))
        - weights.hist * history_len_tokens
        + weights.hit * hit_prob
        - weights.rebuild * rebuild_cost
        + weights.fair * fairness_bonus
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
