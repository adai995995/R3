from __future__ import annotations

patch = None


def _is_dev_build(version: str) -> bool:
    # Local editable installs often expose "0.0.0.dev..." or "...+g<sha>".
    return ".dev" in version or version.startswith("0.0.0.dev")


def _prepatch_torch_for_sgl_kernel() -> None:
    """Make SGLang import robust when sgl_kernel custom op schemas are missing.

    Some SGLang versions unconditionally call torch.library.register_fake on
    sgl_kernel::* ops during import. If the op schema is not registered in torch,
    that crashes the whole worker import path.
    """

    try:
        # In some Ray/multiprocessing worker contexts, dynamic linker may pick an
        # older system CUDA runtime (e.g. from /usr/local/cuda) which can be
        # incompatible with the CUDA runtime bundled by PyTorch wheels.
        # Preloading the wheel-provided libcudart avoids undefined symbols when
        # importing torch in spawned processes.
        try:
            import ctypes

            _cudart = "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
            ctypes.CDLL(_cudart, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

        import torch

        # Define minimal schemas so the operator exists in torch's dispatcher.
        try:
            lib = torch.library.Library("sgl_kernel", "DEF")
        except Exception:
            lib = None

        if lib is not None:
            for schema in (
                "sgl_per_tensor_quant_fp8(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s, bool is_static) -> ()",
                "sgl_per_token_quant_fp8(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s) -> ()",
                "sgl_per_token_group_quant_fp8(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s, int group_size) -> ()",
                "sgl_per_token_group_quant_8bit(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s, int group_size) -> ()",
                "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? bias) -> Tensor",
                "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype) -> Tensor",
                "moe_fused_gate(Tensor input_tensor, Tensor bias, int num_expert_group, int topk_group, int topk) -> (Tensor, Tensor)",
            ):
                try:
                    lib.define(schema)
                except Exception:
                    pass

        # Wrap register_fake to only suppress missing-op errors for sgl_kernel.
        # Support both call styles:
        # - decorator: register_fake(opname)(fn)
        # - direct:    register_fake(opname, fn)
        orig = torch.library.register_fake

        def safe_register_fake(opname: str, fn=None):
            def deco(f):
                try:
                    return orig(opname)(f)
                except RuntimeError as e:
                    msg = str(e)
                    if opname.startswith("sgl_kernel::") and "does not exist" in msg:
                        return f
                    raise

            if fn is not None:
                return deco(fn)
            return deco

        torch.library.register_fake = safe_register_fake
    except Exception:
        # Best effort: never block bring-up.
        return


def _select_patch(version: str):
    if version == "0.4.6.post4":
        from roll.third_party.sglang import v046post4_patch

        return v046post4_patch
    if version in {"0.4.6.post1", "0.4.6.post5"}:
        from roll.third_party.sglang import v046post4_patch

        return v046post4_patch
    if version == "0.4.10.post2":
        from roll.third_party.sglang import v0410post2_patch

        return v0410post2_patch
    if version == "0.5.2":
        from roll.third_party.sglang import v052_patch

        return v052_patch
    if version in {"0.5.4.post2", "0.5.5.post3"}:
        from roll.third_party.sglang import v054_patch

        return v054_patch

    # Best-effort compatibility for local dev builds (e.g., custom forks).
    # v054_patch targets the newer `sglang.srt.entrypoints.engine` API surface.
    if _is_dev_build(version):
        from roll.third_party.sglang import v054_patch

        return v054_patch

    return None


# Patch torch first, then import sglang.
_prepatch_torch_for_sgl_kernel()

import sglang as sgl

patch = _select_patch(sgl.__version__)
if patch is None:
    raise NotImplementedError(
        "Scale aligner version is not supported. "
        f"sglang.__version__={sgl.__version__!r}. "
        "Supported: 0.4.6.post{1,4,5}, 0.4.10.post2, 0.5.2, 0.5.4.post2, 0.5.5.post3. "
        "If you are using a local dev build, ensure the fork is compatible with the 0.5.x runtime API."
    )