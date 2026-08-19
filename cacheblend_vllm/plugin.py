"""vLLM general-plugin entry point."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_REGISTERED = False


def register() -> None:
    """Register transparent Qwen3 model subclasses exactly once per process."""
    global _REGISTERED
    with _LOCK:
        if _REGISTERED:
            return
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            "Qwen3ForCausalLM",
            "cacheblend_vllm.models:CacheBlendQwen3ForCausalLM",
        )
        ModelRegistry.register_model(
            "Qwen3MoeForCausalLM",
            "cacheblend_vllm.models:CacheBlendQwen3MoeForCausalLM",
        )
        _REGISTERED = True
        logger.info("Registered CacheBlend Qwen3 model adapters")
