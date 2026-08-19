"""Fail-fast compatibility checks for the intentionally pinned V1 API."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version


def _minor(package: str) -> tuple[int, int] | None:
    try:
        value = version(package)
    except PackageNotFoundError:
        return None
    match = re.match(r"^(\d+)\.(\d+)", value)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def verify_runtime(strict: bool) -> None:
    if not strict:
        return
    vllm_minor = _minor("vllm")
    if vllm_minor not in (None, (0, 18)):
        raise RuntimeError(
            f"CacheBlend V1 supports vLLM 0.18.x, found {version('vllm')}. "
            "Set strict_version_check=false only for compatibility development."
        )
    ascend_minor = _minor("vllm-ascend")
    if ascend_minor not in (None, (0, 18)):
        raise RuntimeError(
            f"CacheBlend V1 supports vLLM-Ascend 0.18.x, found {version('vllm-ascend')}"
        )
