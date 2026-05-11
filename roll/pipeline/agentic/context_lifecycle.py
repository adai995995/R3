from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ContextState(str, Enum):
    GPU_HOT = "gpu_hot"
    CPU_WARM = "cpu_warm"
    EVICTED = "evicted"


@dataclass
class ContextRecord:
    rid: str
    worker_url: Optional[str]
    state: ContextState
    created_ts: float
    updated_ts: float
    expires_at: Optional[float] = None
    estimated_tokens: int = 0
    reload_count: int = 0
    last_reload_latency_s: float = 0.0


class ContextLifecycleManager:
    """Control-plane state for resume-aware context lifecycle decisions.

    This manager does not claim that KV cache has been retained by SGLang by
    itself. It tracks the desired lifecycle state and exposes metrics so the
    actual worker/gateway KV pin/offload/reload APIs can be wired in behind the
    same interface.
    """

    def __init__(self, token_budget: int = 0, memory_budget_bytes: Optional[int] = None):
        # `memory_budget_bytes` is accepted as a legacy alias, but the unit here
        # is intentionally tokens until a real KV-byte estimator is wired in.
        if memory_budget_bytes is not None and token_budget == 0:
            token_budget = memory_budget_bytes
        self.token_budget = max(0, int(token_budget))
        self._records: "OrderedDict[str, ContextRecord]" = OrderedDict()
        self._eviction_count = 0
        self._expiration_count = 0
        self._reload_count = 0
        self._reload_latency_sum_s = 0.0

    def pin_context(
        self,
        rid: str,
        worker_url: Optional[str],
        ttl_s: Optional[float] = None,
        estimated_tokens: int = 0,
        estimated_bytes: Optional[int] = None,
    ) -> ContextRecord:
        if estimated_bytes is not None and estimated_tokens == 0:
            estimated_tokens = estimated_bytes
        now = time.time()
        record = ContextRecord(
            rid=rid,
            worker_url=worker_url,
            state=ContextState.GPU_HOT,
            created_ts=now,
            updated_ts=now,
            expires_at=now + ttl_s if ttl_s is not None else None,
            estimated_tokens=max(0, int(estimated_tokens)),
        )
        self._records[rid] = record
        self._records.move_to_end(rid)
        self._evict_to_budget()
        return record

    def retain_context(self, rid: str, ttl_s: Optional[float] = None) -> Optional[ContextRecord]:
        record = self._records.get(rid)
        if record is None:
            return None
        now = time.time()
        record.updated_ts = now
        if ttl_s is not None:
            record.expires_at = now + ttl_s
        self._records.move_to_end(rid)
        return record

    def offload_context(self, rid: str) -> Optional[ContextRecord]:
        record = self._records.get(rid)
        if record is None:
            return None
        record.state = ContextState.CPU_WARM
        record.updated_ts = time.time()
        self._records.move_to_end(rid)
        return record

    def reload_context(self, rid: str, worker_url: Optional[str], latency_s: float = 0.0) -> Optional[ContextRecord]:
        record = self._records.get(rid)
        if record is None or record.state == ContextState.EVICTED:
            return None
        latency_s = max(0.0, float(latency_s))
        record.state = ContextState.GPU_HOT
        record.worker_url = worker_url
        record.updated_ts = time.time()
        record.reload_count += 1
        record.last_reload_latency_s = latency_s
        self._reload_count += 1
        self._reload_latency_sum_s += latency_s
        self._records.move_to_end(rid)
        self._evict_to_budget()
        return record

    def unpin_context(self, rid: str) -> Optional[ContextRecord]:
        record = self._records.get(rid)
        if record is None:
            return None
        record.state = ContextState.EVICTED
        record.updated_ts = time.time()
        return self._records.pop(rid)

    def classify_resume(self, rid: str, worker_url: Optional[str]) -> str:
        record = self._records.get(rid)
        if record is None or record.state == ContextState.EVICTED:
            return "full_prefill"
        if self._is_expired(record):
            self._expire_context(rid)
            return "full_prefill"
        if record.state == ContextState.CPU_WARM:
            return "cpu_reload"
        if record.worker_url == worker_url:
            self._records.move_to_end(rid)
            return "gpu_hit"
        return "full_prefill"

    def collect_metrics(self, prefix: str = "context") -> Dict[str, float]:
        self.prune_expired()
        gpu_hot = sum(1 for r in self._records.values() if r.state == ContextState.GPU_HOT)
        cpu_warm = sum(1 for r in self._records.values() if r.state == ContextState.CPU_WARM)
        estimated_tokens = sum(r.estimated_tokens for r in self._records.values())
        reload_latency_mean = (
            self._reload_latency_sum_s / self._reload_count if self._reload_count > 0 else 0.0
        )
        return {
            f"{prefix}/gpu_hot_count": float(gpu_hot),
            f"{prefix}/cpu_warm_count": float(cpu_warm),
            f"{prefix}/record_count": float(len(self._records)),
            f"{prefix}/estimated_tokens": float(estimated_tokens),
            f"{prefix}/token_budget": float(self.token_budget),
            f"{prefix}/eviction_count": float(self._eviction_count),
            f"{prefix}/expiration_count": float(self._expiration_count),
            f"{prefix}/reload_count": float(self._reload_count),
            f"{prefix}/reload_latency_mean_s": float(reload_latency_mean),
        }

    def prune_expired(self) -> int:
        expired = [rid for rid, record in self._records.items() if self._is_expired(record)]
        for rid in expired:
            self._expire_context(rid)
        return len(expired)

    def _evict_to_budget(self) -> None:
        if self.token_budget <= 0:
            return
        while self._estimated_tokens() > self.token_budget and self._records:
            _, record = self._records.popitem(last=False)
            record.state = ContextState.EVICTED
            record.updated_ts = time.time()
            self._eviction_count += 1

    def _estimated_tokens(self) -> int:
        return sum(r.estimated_tokens for r in self._records.values())

    def _expire_context(self, rid: str) -> None:
        record = self._records.pop(rid, None)
        if record is not None:
            record.state = ContextState.EVICTED
            record.updated_ts = time.time()
            self._expiration_count += 1

    @staticmethod
    def _is_expired(record: ContextRecord) -> bool:
        return record.expires_at is not None and time.time() > record.expires_at
