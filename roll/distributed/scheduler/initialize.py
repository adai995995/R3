import os
import subprocess
import sys
import time

import ray

from roll.distributed.scheduler.driver_utils import (
    get_driver_rank,
    get_driver_master_addr,
    get_driver_node_name,
    get_driver_master_port,
    get_driver_world_size,
    get_driver_dashboard_port,
    get_ray_status,
    is_ray_cluster_running,
    wait_for_nodes,
)
from roll.distributed.scheduler.log_monitor import LogMonitorListener
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.platforms import current_platform

logger = get_logger()

def _ray_stop_and_cleanup(tmp_dir: str) -> None:
    # Best-effort cleanup for stale local Ray sessions that can cause
    # "Session name ... does not match persisted value" assertion failures.
    try:
        subprocess.run("ray stop --force", shell=True, capture_output=True)
    except Exception:
        pass
    if tmp_dir:
        try:
            subprocess.run(f"rm -rf {tmp_dir}", shell=True, capture_output=True)
        except Exception:
            pass


def start_ray_cluster():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()
    node_name = get_driver_node_name()
    dashboard_port = get_driver_dashboard_port()
    ray_tmp_dir = (os.getenv("RAY_TMPDIR") or "/tmp/ray").strip()
    ray_tmp_dir_flag = f" --temp-dir={ray_tmp_dir}" if ray_tmp_dir else ""

    # Ray's GPU autodetection can fail in some container/runtime setups even when
    # CUDA is usable (e.g. NVML init quirks). Allow forcing GPU count so the
    # scheduler can still request "GPU" resources.
    forced_num_gpus = os.getenv("RAY_NUM_GPUS", "").strip()
    if forced_num_gpus:
        ray_num_gpus_flag = f" --num-gpus={forced_num_gpus}"
    else:
        ray_num_gpus_flag = ""
        try:
            import torch

            dc = int(torch.cuda.device_count())
            if dc > 0:
                ray_num_gpus_flag = f" --num-gpus={dc}"
        except Exception:
            pass

    if is_ray_cluster_running():
        logger.info("Ray cluster already initialized")
        return False

    if rank == 0:
        # IMPORTANT: start Ray using the current Python so that raylet/workers
        # inherit the same venv/site-packages (e.g. openreward) as the driver.
        #
        # NOTE: `python -m ray` is not a stable entrypoint across Ray versions
        # (some builds do not provide ray.__main__). The canonical module
        # entrypoint is `ray.scripts.scripts`.
        ray_cli = f"\"{sys.executable}\" -m ray.scripts.scripts"
        cmd = (
            f"{ray_cli} start --head "
            f"--port={master_port} "
            f"--node-name={node_name} "
            f"--dashboard-port={dashboard_port} "
            "--disable-usage-stats"
            f"{ray_tmp_dir_flag}"
            f"{ray_num_gpus_flag}"
        )
    else:
        # fix: 处理大规模下可能会出现的head/worker node创建顺序不一致问题
        time.sleep(5)
        ray_cli = f"\"{sys.executable}\" -m ray.scripts.scripts"
        cmd = (
            f"{ray_cli} start "
            f"--address={master_addr}:{master_port} "
            f"--node-name={node_name} "
            f"--dashboard-port={dashboard_port} "
            "--disable-usage-stats"
            f"{ray_tmp_dir_flag}"
            f"{ray_num_gpus_flag}"
        )

    logger.info(f"Starting ray cluster: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        stderr_text = ""
        try:
            stderr_text = (ret.stderr or b"").decode("utf-8", errors="ignore")
        except Exception:
            stderr_text = str(ret.stderr)

        # Self-heal stale local session conflicts (common with --network host and port reuse).
        if "does not match persisted value" in stderr_text or "Perhaps there was an error connecting to Redis" in stderr_text:
            logger.warning(
                "Ray start failed due to a stale local session conflict; "
                "running `ray stop --force` and cleaning temp dir, then retrying once."
            )
            _ray_stop_and_cleanup(ray_tmp_dir)
            ret2 = subprocess.run(cmd, shell=True, capture_output=True)
            if ret2.returncode == 0:
                return True
            logger.error(f"Failed to start ray cluster after cleanup: {cmd}")
            logger.error(f"ret.stdout: {ret2.stdout}")
            logger.error(f"ret.stderr: {ret2.stderr}")
            sys.exit(1)

        logger.error(f"Failed to start ray cluster: {cmd}")
        logger.error(f"ret.stdout: {ret.stdout}")
        logger.error(f"ret.stderr: {ret.stderr}")
        sys.exit(1)
    return True


def init():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()

    # Prefer an existing Ray cluster if user provides an explicit address.
    # This is common in shared single-node environments where another Ray head
    # is already running (port 6379 by default).
    ray_address = os.getenv("RAY_ADDRESS", "").strip()
    # NOTE: We intentionally avoid shelling out to `ray start` here.
    # - In some environments the Ray CLI can fail due to dependency conflicts
    #   (click/typer/sentinel issues) even though the Ray Python API works.
    # - `ray.init()` (with address=None) will start a local Ray runtime using
    #   the current Python executable, ensuring workers inherit the same venv
    #   and can import packages like `openreward`.
    manual_start = False

    runtime_env = {
        "env_vars": current_platform.get_custom_env_vars(),
    }

    if not ray.is_initialized():
        # For local init, also respect RAY_TMPDIR/RAY_NUM_GPUS when provided.
        ray_tmp_dir = (os.getenv("RAY_TMPDIR") or "/tmp/ray").strip()
        forced_num_gpus = os.getenv("RAY_NUM_GPUS", "").strip()
        num_gpus = None
        if forced_num_gpus:
            try:
                num_gpus = int(forced_num_gpus)
            except Exception:
                num_gpus = None
        if num_gpus is None:
            try:
                import torch

                dc = int(torch.cuda.device_count())
                if dc > 0:
                    num_gpus = dc
            except Exception:
                num_gpus = None

        ray.init(
            address=(ray_address or None),
            namespace=RAY_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=True,
            runtime_env=runtime_env,
            _temp_dir=ray_tmp_dir or None,
            num_gpus=num_gpus,
        )
        logger.info("Ray cluster initialized")

    logger.info(f"Current ray cluster resources: {ray.available_resources()}")
