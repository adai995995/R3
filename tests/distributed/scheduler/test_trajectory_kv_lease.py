"""Unit tests for Phase D trajectory KV lease manager (SGLang-side logic mirror)."""

import time

from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from sglang.srt.mem_cache.trajectory_kv_lease import TrajectoryKvLeaseManager


def test_cache_before_lease_then_set_lease_and_lookup():
    """B1: generate caches KV before tool-suspend POST; set_lease must bind cached node."""
    mgr = TrajectoryKvLeaseManager(lease_lambda=100.0)
    root = TreeNode()
    root.key = []
    root.value = []
    leaf = TreeNode()
    leaf.parent = root
    leaf.key = [1, 2, 3]
    leaf.value = [10, 11, 12]
    root.children[1] = leaf

    mgr.on_req_cached("traj-a", leaf)
    mgr.set_lease("traj-a", ttl_s=30.0, lease_score=0.9)

    assert float(getattr(leaf, "lease_pin_score", 0)) >= 0.9
    out = mgr.lookup_resume("traj-a", evictable_size=100, protected_size=50)
    assert out["found"] is True
    assert out["hit_tokens"] >= 3


def test_set_lease_pins_path_and_lookup():
    mgr = TrajectoryKvLeaseManager(lease_lambda=100.0)
    root = TreeNode()
    root.key = []
    root.value = []
    leaf = TreeNode()
    leaf.parent = root
    leaf.key = [1, 2, 3]
    leaf.value = [10, 11, 12]
    root.children[1] = leaf

    mgr.set_lease("traj-a", ttl_s=30.0, lease_score=0.9)
    mgr.on_req_cached("traj-a", leaf)

    assert float(getattr(leaf, "lease_pin_score", 0)) >= 0.9
    out = mgr.lookup_resume("traj-a", evictable_size=100, protected_size=50)
    assert out["found"] is True
    assert out["hit_tokens"] >= 3
    assert 0.0 <= out["memory_pressure"] <= 1.0


def test_expired_lease_allows_evict():
    mgr = TrajectoryKvLeaseManager()
    node = TreeNode()
    node.last_access_time = time.time() - 1000
    node.lease_pin_score = 0.9
    node.lease_pin_expires_at = time.time() - 1.0
    assert mgr.can_evict_node(node) is True


def test_active_lease_blocks_evict():
    mgr = TrajectoryKvLeaseManager()
    node = TreeNode()
    node.last_access_time = time.time()
    node.lease_pin_score = 0.8
    node.lease_pin_expires_at = time.time() + 60.0
    assert mgr.can_evict_node(node) is False


def test_radix_cache_has_kv_lease_manager():
    cache = RadixCache(None, None, page_size=1, disable=True)
    assert hasattr(cache, "kv_lease")
    assert isinstance(cache.kv_lease, TrajectoryKvLeaseManager)
