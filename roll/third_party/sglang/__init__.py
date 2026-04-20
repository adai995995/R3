from __future__ import annotations

import sglang as sgl

patch = None


def _is_dev_build(version: str) -> bool:
    # Local editable installs often expose "0.0.0.dev..." or "...+g<sha>".
    return ".dev" in version or version.startswith("0.0.0.dev")


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


patch = _select_patch(sgl.__version__)
if patch is None:
    raise NotImplementedError(
        "Scale aligner version is not supported. "
        f"sglang.__version__={sgl.__version__!r}. "
        "Supported: 0.4.6.post{1,4,5}, 0.4.10.post2, 0.5.2, 0.5.4.post2, 0.5.5.post3. "
        "If you are using a local dev build, ensure the fork is compatible with the 0.5.x runtime API."
    )