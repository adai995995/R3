"""Patch SGLang model-update NCCL group for Ray single-GPU workers.

SGLang's ModelRunner.init_weights_update_group uses dist.barrier(..., device_ids=[rank])
where rank is the global NCCL rank (1..N). Each InferWorker only sees one GPU as cuda:0,
so rank>=1 triggers "CUDA error: invalid device ordinal".

Apply from run_scheduler_process (sglang::scheduler child) before ModelRunner loads.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_model_update_pg_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import torch
    import torch.distributed as dist

    from roll.utils.collective.pg_utils import init_custom_process_group as roll_init_pg

    import sglang.srt.utils as srt_utils

    if hasattr(srt_utils, "init_custom_process_group"):
        srt_utils.init_custom_process_group = roll_init_pg

    import sglang.srt.model_executor.model_runner as model_runner

    model_runner.init_custom_process_group = roll_init_pg

    def init_weights_update_group_roll(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert torch.distributed.is_initialized(), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank
        logger.info(
            "init custom process group (roll patch): master_address=%s, master_port=%s, "
            "rank_offset=%s, rank=%s, world_size=%s, group_name=%s, backend=%s, local_cuda=%s",
            master_address,
            master_port,
            rank_offset,
            rank,
            world_size,
            group_name,
            backend,
            torch.cuda.current_device(),
        )

        try:
            self._model_update_group = roll_init_pg(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            local_device = torch.cuda.current_device()
            dist.barrier(group=self._model_update_group, device_ids=[local_device])
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    model_runner.ModelRunner.init_weights_update_group = init_weights_update_group_roll
    _PATCHED = True
    logger.info("Applied ROLL model_update_pg patch to sglang ModelRunner")
