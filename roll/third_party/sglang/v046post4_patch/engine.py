import os
import multiprocessing as mp

import sglang.srt.entrypoints.engine as engine_module
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
)


# Remove signal handler. singla.signal in python can only run in MainThread which fails when using Ray Async Actor.
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Set mp start method
    mp.set_start_method("spawn", force=True)

def run_scheduler_process(*args, **kwargs):
    # torchao is an optional dependency in some environments but certain SGLang
    # builds import it unconditionally during ModelRunner initialization.
    # For ROLL's bf16 smoke runs, we can safely skip torchao if it's unavailable.
    try:
        import sglang.srt.layers.torchao_utils as _torchao_utils

        _orig_apply = getattr(_torchao_utils, "apply_torchao_config_to_model", None)

        if callable(_orig_apply):
            def _safe_apply_torchao_config_to_model(*a, **k):  # type: ignore[no-redef]
                try:
                    return _orig_apply(*a, **k)
                except ModuleNotFoundError as e:
                    if str(e).startswith("No module named 'torchao'"):
                        print("[roll][sglang] torchao not installed; skip torchao config")
                        return None
                    raise

            _torchao_utils.apply_torchao_config_to_model = _safe_apply_torchao_config_to_model  # type: ignore[attr-defined]
            # Some SGLang versions import the symbol into model_runner module namespace:
            #   from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
            # Patch that reference too (best-effort) so calls don't bypass our shim.
            try:
                import sglang.srt.model_executor.model_runner as _model_runner
                if hasattr(_model_runner, "apply_torchao_config_to_model"):
                    _model_runner.apply_torchao_config_to_model = _safe_apply_torchao_config_to_model  # type: ignore[attr-defined]
            except Exception as e:
                print(f"[roll][sglang] torchao model_runner shim not applied: {e}")
    except Exception as e:
        # Best-effort only: if module path differs across SGLang versions, ignore.
        print(f"[roll][sglang] torchao shim not applied: {e}")

    # ROLL patches fp8 utils to support specific SGLang versions.
    # SGLang APIs in fp8/moe frequently change; import failures should not prevent
    # non-fp8 runs (e.g. bf16) from starting the server.
    try:
        from roll.third_party.sglang import fp8
        fp8.monkey_patch_fp8()
    except Exception as e:
        # Avoid hard-crash on import-time incompatibility.
        # The scheduler can still run without fp8 monkey patches.
        print(f"[roll][sglang] skip fp8 monkey patch due to: {e}")

    from sglang.srt.managers.scheduler import run_scheduler_process
    return run_scheduler_process(*args, **kwargs)

def run_data_parallel_controller_process(*args, **kwargs):
    import sys
    sys.modules['sglang.srt.managers.data_parallel_controller'].__dict__['run_scheduler_process'] = run_scheduler_process

    from sglang.srt.managers.data_parallel_controller import run_data_parallel_controller_process
    return run_data_parallel_controller_process(*args, **kwargs)

class _roll_launch_subprocesses(object):
    def __init__(self, _launch_subprocesses):
        self._launch_subprocesses = _launch_subprocesses
    
    def __call__(self, *args, **kwargs):
        import sys

        sys.modules['sglang.srt.entrypoints.engine'].__dict__['_set_envs_and_config'] = _set_envs_and_config
        sys.modules['sglang.srt.entrypoints.engine'].__dict__['run_scheduler_process'] = run_scheduler_process
        sys.modules['sglang.srt.entrypoints.engine'].__dict__['run_data_parallel_controller_process'] = run_data_parallel_controller_process
        return self._launch_subprocesses(*args, **kwargs)


engine_module._launch_subprocesses = _roll_launch_subprocesses(engine_module._launch_subprocesses)
