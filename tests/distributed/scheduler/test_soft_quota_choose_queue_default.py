from roll.distributed.scheduler.soft_quota_utils import choose_queue_for_soft_quota


def test_choose_queue_skip_on_empty():
    assert (
        choose_queue_for_soft_quota(
            resume_empty=False,
            normal_empty=True,
            quota_resume_target=3,
            quota_normal_target=1,
            quota_cursor=0,
            oldest_resume_wait_s=0.0,
            oldest_normal_wait_s=0.0,
            resume_max_queue_wait_s=0.0,
            normal_max_queue_wait_s=0.0,
        )
        == "resume"
    )
    assert (
        choose_queue_for_soft_quota(
            resume_empty=True,
            normal_empty=False,
            quota_resume_target=3,
            quota_normal_target=1,
            quota_cursor=0,
            oldest_resume_wait_s=0.0,
            oldest_normal_wait_s=0.0,
            resume_max_queue_wait_s=0.0,
            normal_max_queue_wait_s=0.0,
        )
        == "normal"
    )


def test_choose_queue_timeout_escape_hatch():
    # Normal starved: should choose normal regardless of quota cursor.
    assert (
        choose_queue_for_soft_quota(
            resume_empty=False,
            normal_empty=False,
            quota_resume_target=3,
            quota_normal_target=1,
            quota_cursor=0,
            oldest_resume_wait_s=0.0,
            oldest_normal_wait_s=10.0,
            resume_max_queue_wait_s=0.0,
            normal_max_queue_wait_s=1.0,
        )
        == "normal"
    )
    # Resume starved: should choose resume regardless of quota cursor.
    assert (
        choose_queue_for_soft_quota(
            resume_empty=False,
            normal_empty=False,
            quota_resume_target=0,
            quota_normal_target=1,
            quota_cursor=0,
            oldest_resume_wait_s=10.0,
            oldest_normal_wait_s=0.0,
            resume_max_queue_wait_s=1.0,
            normal_max_queue_wait_s=0.0,
        )
        == "resume"
    )


def test_choose_queue_quota_cycle():
    # quota 2:1 => cursor 0,1 => resume; cursor 2 => normal
    assert (
        choose_queue_for_soft_quota(
            resume_empty=False,
            normal_empty=False,
            quota_resume_target=2,
            quota_normal_target=1,
            quota_cursor=0,
            oldest_resume_wait_s=0.0,
            oldest_normal_wait_s=0.0,
            resume_max_queue_wait_s=0.0,
            normal_max_queue_wait_s=0.0,
        )
        == "resume"
    )
    assert (
        choose_queue_for_soft_quota(
            resume_empty=False,
            normal_empty=False,
            quota_resume_target=2,
            quota_normal_target=1,
            quota_cursor=1,
            oldest_resume_wait_s=0.0,
            oldest_normal_wait_s=0.0,
            resume_max_queue_wait_s=0.0,
            normal_max_queue_wait_s=0.0,
        )
        == "resume"
    )
    assert (
        choose_queue_for_soft_quota(
            resume_empty=False,
            normal_empty=False,
            quota_resume_target=2,
            quota_normal_target=1,
            quota_cursor=2,
            oldest_resume_wait_s=0.0,
            oldest_normal_wait_s=0.0,
            resume_max_queue_wait_s=0.0,
            normal_max_queue_wait_s=0.0,
        )
        == "normal"
    )

