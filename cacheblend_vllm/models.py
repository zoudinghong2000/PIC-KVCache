"""Transparent vLLM Qwen3 subclasses used to expose the serving model.

The normal model forward and weight-loading paths are inherited unchanged.
Only CacheBlend engines publish the raw model object to a process-local weak
registry before vLLM constructs the KV connector.
"""

from __future__ import annotations

from typing import Any

from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from .registry import register_model


def _maybe_register(vllm_config: Any, model: Any) -> None:
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if transfer is None or transfer.kv_connector != "CacheBlendConnectorV1":
        return
    engine_id = transfer.engine_id
    if not engine_id:
        raise ValueError("CacheBlendConnectorV1 requires kv_transfer_config.engine_id")
    register_model(str(engine_id), model)


class CacheBlendQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _maybe_register(vllm_config, self)


class CacheBlendQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _maybe_register(vllm_config, self)
