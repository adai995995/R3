from roll.distributed.scheduler.trajectory_value import merge_resume_lease_ttl_score


def test_merge_resume_lease_ttl_score_takes_max():
    route_meta = {
        "resume_lease_ttl_s": 30.0,
        "resume_lease_score": 0.2,
        "pending_resume_lease_ttl_s": 12.0,
        "pending_resume_lease_score": 0.5,
    }
    ttl, score, used_pending = merge_resume_lease_ttl_score(route_meta)
    assert ttl == 30.0
    assert score == 0.5
    assert used_pending is True


def test_merge_store_pending_not_less_than_resume():
    route_meta = {
        "resume_lease_ttl_s": 25.0,
        "resume_lease_score": 0.3,
    }
    ttl, _, _ = merge_resume_lease_ttl_score(
        route_meta, store_pending_ttl=40.0, store_pending_score=0.1
    )
    assert ttl == 40.0
