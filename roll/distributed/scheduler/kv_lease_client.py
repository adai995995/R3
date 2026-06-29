"""HTTP client for gateway/SGLang KV lease and resume lookup (L2/L3, Phase D)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

_INPROC_WORKER_PREFIX = "inproc://"


def is_inproc_worker_url(url: Optional[str]) -> bool:
    return isinstance(url, str) and url.startswith(_INPROC_WORKER_PREFIX)


def lookup_result_from_worker_payload(data: Dict[str, Any]) -> LookupResumeResult:
    """Normalize Ray/HTTP worker KV lookup payloads."""
    if not isinstance(data, dict):
        return LookupResumeResult()
    return LookupResumeResult.from_json(data)


@dataclass
class LookupResumeResult:
    """Engine-side resume/KV observability (L2)."""

    found: bool = False
    hit_tokens: int = 0
    resident_blocks: int = 0
    estimated_prefill_tokens: int = 0
    cache_confidence: float = 0.0
    lease_remaining_s: Optional[float] = None
    worker_url: Optional[str] = None
    memory_pressure: Optional[float] = None
    worker_confirmed: bool = False
    lookup_source: str = "miss"
    lease_model_version: Optional[int] = None
    request_model_version: Optional[int] = None
    model_version_match: Optional[bool] = None
    stale_version_blocked: bool = False
    engine_kv_pinned_tokens: Optional[int] = None
    engine_kv_evicted_tokens: Optional[int] = None
    engine_kv_evicted_pinned_tokens: Optional[int] = None
    engine_kv_lease_hit: Optional[int] = None
    engine_kv_lease_miss: Optional[int] = None
    engine_kv_lease_stale_version_blocked: Optional[int] = None

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "LookupResumeResult":
        if not isinstance(data, dict):
            return cls()
        hit = data.get("hit_tokens") or data.get("matched_prefix_tokens") or 0
        prefill = data.get("estimated_prefill_tokens") or data.get("prefill_tokens") or 0
        conf = data.get("cache_confidence") or data.get("confidence")
        remaining = data.get("lease_remaining_s") or data.get("remaining_s")
        mem = data.get("memory_pressure")
        worker_confirmed = data.get("worker_confirmed")
        lookup_source = data.get("lookup_source")
        def _opt_int(key: str) -> Optional[int]:
            value = data.get(key)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        model_match = data.get("model_version_match")
        if model_match is not None:
            model_match = bool(model_match)
        return cls(
            found=bool(data.get("found", False)),
            hit_tokens=int(hit) if hit is not None else 0,
            resident_blocks=int(data.get("resident_blocks") or 0),
            estimated_prefill_tokens=int(prefill) if prefill is not None else 0,
            cache_confidence=float(conf) if conf is not None else 0.0,
            lease_remaining_s=float(remaining) if remaining is not None else None,
            worker_url=data.get("worker_url") if isinstance(data.get("worker_url"), str) else None,
            memory_pressure=float(mem) if mem is not None else None,
            worker_confirmed=bool(worker_confirmed) if worker_confirmed is not None else False,
            lookup_source=lookup_source if isinstance(lookup_source, str) and lookup_source else "miss",
            lease_model_version=_opt_int("lease_model_version") or _opt_int("kv_lease_model_version") or _opt_int("model_version"),
            request_model_version=_opt_int("request_model_version") or _opt_int("route_model_version"),
            model_version_match=model_match,
            stale_version_blocked=bool(data.get("stale_version_blocked", False)),
            engine_kv_pinned_tokens=_opt_int("engine_kv_pinned_tokens"),
            engine_kv_evicted_tokens=_opt_int("engine_kv_evicted_tokens"),
            engine_kv_evicted_pinned_tokens=_opt_int("engine_kv_evicted_pinned_tokens"),
            engine_kv_lease_hit=_opt_int("engine_kv_lease_hit"),
            engine_kv_lease_miss=_opt_int("engine_kv_lease_miss"),
            engine_kv_lease_stale_version_blocked=_opt_int("engine_kv_lease_stale_version_blocked"),
        )


def _norm_url(url: str) -> str:
    return str(url or "").rstrip("/")


async def lookup_resume(
    client: httpx.AsyncClient,
    gateway_url: str,
    trajectory_id: str,
    *,
    lookup_path_template: str = "/kv/resume/{trajectory_id}",
    worker_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 2.0,
    model_version: Optional[int] = None,
) -> LookupResumeResult:
    """GET resume/KV state before dispatch. Returns empty result if API missing."""
    gw = _norm_url(gateway_url)
    if not gw or not trajectory_id:
        return LookupResumeResult()
    path = lookup_path_template.format(trajectory_id=quote(trajectory_id, safe=""))
    url = f"{gw}{path}"
    params: Dict[str, str] = {}
    if worker_url:
        params["worker_url"] = worker_url
    if model_version is not None:
        params["model_version"] = str(int(model_version))
    try:
        resp = await client.get(url, params=params or None, headers=headers, timeout=timeout_s)
        if resp.status_code == 404:
            return LookupResumeResult(found=False)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            data = data["data"]
        return LookupResumeResult.from_json(data if isinstance(data, dict) else {})
    except Exception as e:
        logger.debug("lookup_resume failed for %s: %s", trajectory_id, e)
        return LookupResumeResult(found=False)


async def lookup_resume_worker(
    client: httpx.AsyncClient,
    worker_url: str,
    trajectory_id: str,
    *,
    lookup_path_template: str = "/internal/kv/resume/{trajectory_id}",
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 2.0,
    model_version: Optional[int] = None,
) -> LookupResumeResult:
    """GET engine-authoritative resume state from a single worker."""
    base = _norm_url(worker_url)
    if not base or not trajectory_id:
        return LookupResumeResult()
    path = lookup_path_template.format(trajectory_id=quote(trajectory_id, safe=""))
    try:
        params = {"model_version": str(int(model_version))} if model_version is not None else None
        resp = await client.get(f"{base}{path}", params=params, headers=headers, timeout=timeout_s)
        if resp.status_code == 404:
            return LookupResumeResult(found=False)
        resp.raise_for_status()
        data = resp.json()
        result = LookupResumeResult.from_json(data if isinstance(data, dict) else {})
        if result.lookup_source == "miss":
            result.lookup_source = "worker" if result.found else "miss"
        result.worker_confirmed = True
        return result
    except Exception as e:
        logger.debug("lookup_resume_worker failed for %s: %s", trajectory_id, e)
        return LookupResumeResult(found=False)


async def set_kv_lease(
    client: httpx.AsyncClient,
    gateway_url: str,
    *,
    trajectory_id: str,
    ttl_s: float,
    lease_score: float,
    worker_url: Optional[str] = None,
    belief_level: Optional[str] = None,
    model_version: Optional[int] = None,
    lease_path: str = "/kv/lease",
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 2.0,
) -> bool:
    """POST lease registration (L3). No-op on failure."""
    gw = _norm_url(gateway_url)
    if not gw or not trajectory_id:
        return False
    body: Dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "ttl_s": float(ttl_s),
        "lease_score": float(lease_score),
    }
    if worker_url:
        body["worker_url"] = worker_url
    if belief_level:
        body["belief_level"] = belief_level
    if model_version is not None:
        body["model_version"] = int(model_version)
    if model_version is not None:
        body["model_version"] = int(model_version)
    try:
        resp = await client.post(
            f"{gw}{lease_path}",
            json=body,
            headers=headers,
            timeout=timeout_s,
        )
        if resp.status_code in (200, 201, 204):
            return True
        if resp.status_code == 404:
            logger.debug("set_kv_lease endpoint not found: %s", lease_path)
            return False
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.debug("set_kv_lease failed for %s: %s", trajectory_id, e)
        return False


async def set_kv_lease_worker(
    client: httpx.AsyncClient,
    worker_url: str,
    *,
    trajectory_id: str,
    ttl_s: float,
    lease_score: float,
    belief_level: Optional[str] = None,
    model_version: Optional[int] = None,
    lease_path: str = "/internal/kv/lease",
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 2.0,
) -> bool:
    """POST lease directly to worker (Phase D primary path, form A)."""
    base = _norm_url(worker_url)
    if not base or not trajectory_id:
        return False
    body: Dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "ttl_s": float(ttl_s),
        "lease_score": float(lease_score),
    }
    if belief_level:
        body["belief_level"] = belief_level
    try:
        resp = await client.post(
            f"{base}{lease_path}",
            json=body,
            headers=headers,
            timeout=timeout_s,
        )
        if resp.status_code in (200, 201, 204):
            return True
        if resp.status_code == 404:
            logger.debug("set_kv_lease_worker endpoint not found: %s", lease_path)
            return False
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.debug("set_kv_lease_worker failed for %s: %s", trajectory_id, e)
        return False


async def delete_kv_lease(
    client: httpx.AsyncClient,
    gateway_url: str,
    trajectory_id: str,
    *,
    lease_path_template: str = "/kv/lease/{trajectory_id}",
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 2.0,
) -> bool:
    gw = _norm_url(gateway_url)
    if not gw or not trajectory_id:
        return False
    path = lease_path_template.format(trajectory_id=quote(trajectory_id, safe=""))
    try:
        resp = await client.delete(f"{gw}{path}", headers=headers, timeout=timeout_s)
        return resp.status_code in (200, 204, 404)
    except Exception as e:
        logger.debug("delete_kv_lease failed for %s: %s", trajectory_id, e)
        return False


async def delete_kv_lease_worker(
    client: httpx.AsyncClient,
    worker_url: str,
    trajectory_id: str,
    *,
    lease_path_template: str = "/internal/kv/lease/{trajectory_id}",
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 2.0,
) -> bool:
    base = _norm_url(worker_url)
    if not base or not trajectory_id:
        return False
    path = lease_path_template.format(trajectory_id=quote(trajectory_id, safe=""))
    try:
        resp = await client.delete(f"{base}{path}", headers=headers, timeout=timeout_s)
        return resp.status_code in (200, 204, 404)
    except Exception as e:
        logger.debug("delete_kv_lease_worker failed for %s: %s", trajectory_id, e)
        return False
