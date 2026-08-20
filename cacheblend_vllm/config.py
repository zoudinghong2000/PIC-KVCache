"""Configuration owned by the out-of-tree CacheBlend connector."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class CacheBlendConfig:
    chunk_size: int = 256
    local_cpu_gb: float = 16.0
    min_retrieve_tokens: int = 256
    min_hit_ratio: float = 0.10
    # Profitability thresholds are workload-specific. Keep library defaults
    # neutral and let deployment/benchmark profiles opt into stricter gates.
    min_saved_tokens: int = 0
    max_apc_prefix_to_hit_ratio: float = 0.0
    check_layers: tuple[int, ...] = (1,)
    recompute_ratios: tuple[float, ...] = (0.15,)
    save_decode_cache: bool = False
    tp_global_selection: bool = True
    event_pipeline: bool = True
    fused_segment_copy: bool = True
    cache_attention_mask: bool = True
    lookup_timeout_ms: int = 10_000
    store_workers: int = 1
    max_inflight_store_batches: int = 8
    strict_version_check: bool = True
    ipc_root: str = "/tmp/cacheblend-v1"
    model_scope: str | None = None

    @classmethod
    def from_extra_config(cls, values: dict[str, Any] | None) -> CacheBlendConfig:
        raw = dict(values or {})
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown CacheBlend config fields: {unknown}")
        for key in ("check_layers", "recompute_ratios"):
            if key in raw:
                value = raw[key]
                if isinstance(value, (int, float)):
                    value = [value]
                raw[key] = tuple(value)
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.local_cpu_gb <= 0:
            raise ValueError("local_cpu_gb must be positive")
        if self.min_retrieve_tokens < 0 or self.min_saved_tokens < 0:
            raise ValueError("token thresholds cannot be negative")
        if not 0.0 <= self.min_hit_ratio <= 1.0:
            raise ValueError("min_hit_ratio must be between 0 and 1")
        if self.max_apc_prefix_to_hit_ratio < 0:
            raise ValueError("max_apc_prefix_to_hit_ratio cannot be negative")
        if self.lookup_timeout_ms <= 0:
            raise ValueError("lookup_timeout_ms must be positive")
        if self.store_workers <= 0:
            raise ValueError("store_workers must be positive")
        if self.max_inflight_store_batches <= 0:
            raise ValueError("max_inflight_store_batches must be positive")
        if not self.check_layers:
            raise ValueError("at least one check layer is required")
        if len(self.recompute_ratios) not in (1, len(self.check_layers)):
            raise ValueError("recompute_ratios must contain one value or one per check layer")
        if any(layer < 0 for layer in self.check_layers):
            raise ValueError("check layer indices cannot be negative")
        if any(not 0.0 <= ratio <= 1.0 for ratio in self.recompute_ratios):
            raise ValueError("recompute ratios must be between 0 and 1")

    @property
    def max_local_cpu_bytes(self) -> int:
        return int(self.local_cpu_gb * 1024**3)
