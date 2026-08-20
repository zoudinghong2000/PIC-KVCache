"""Qwen3 CacheBlend V1 execution on Ascend.

This module is intentionally independent of LMCache-Ascend. It uses vLLM's
loaded Qwen3 layers, torch-npu fused attention, and pure-torch paged KV
gather/scatter. Optional native transfer kernels can replace the latter without
changing the connector contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from .config import CacheBlendConfig
from .kv_layout import gather_paged_kv, scatter_paged_kv
from .storage import LocalPinnedCPUStore
from .trace import PipelineTracer
from .types import BlendPlan

logger = init_logger(__name__)


def _device_name() -> str:
    return "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"


def _embed(model, input_ids: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if callable(embeddings):
            return embeddings(input_ids)
    if hasattr(model, "embed_input_ids"):
        return model.embed_input_ids(input_ids)
    if hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens(input_ids)
    raise AttributeError(f"{type(model).__name__} has no supported embedding API")


def _shuffle_rope_halves(tensor: torch.Tensor, head_size: int, neox: bool) -> torch.Tensor:
    original = tensor.shape
    tensor = tensor.reshape(tensor.shape[0], -1, head_size)
    if neox:
        first, second = torch.chunk(tensor, 2, dim=-1)
        return torch.cat((second, first), dim=-1).reshape(original)
    first, second = tensor[..., ::2], tensor[..., 1::2]
    return torch.stack((second, first), dim=-1).reshape(original)


def relocate_key(
    rotary_emb,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor:
    """Move an already-RoPE-encoded K tensor with one delta rotation.

    Standard RoPE is a group rotation, so ``R(new) R(old)^-1`` can be applied
    directly.  This avoids running the serving model's generic RoPE kernel
    twice for every cached layer.
    """
    if torch.equal(old_positions, new_positions):
        return key
    head_size = int(rotary_emb.head_size)
    rotary_dim = int(getattr(rotary_emb, "rotary_dim", head_size))
    neox = bool(rotary_emb.is_neox_style)
    if not hasattr(rotary_emb, "cos_sin_cache"):
        flat = key.reshape(key.shape[0], -1)
        shuffled = _shuffle_rope_halves(flat, head_size, neox)
        _, inverse_once = rotary_emb(old_positions, shuffled, shuffled)
        unrotated = _shuffle_rope_halves(inverse_once, head_size, neox)
        _, relocated = rotary_emb(new_positions, unrotated, unrotated)
        return relocated.reshape_as(key)
    cache = rotary_emb.cos_sin_cache.to(device=key.device, dtype=key.dtype)
    old_cos_sin = cache.index_select(0, old_positions.flatten())
    new_cos_sin = cache.index_select(0, new_positions.flatten())
    old_cos, old_sin = old_cos_sin.chunk(2, dim=-1)
    new_cos, new_sin = new_cos_sin.chunk(2, dim=-1)
    delta_cos = new_cos * old_cos + new_sin * old_sin
    delta_sin = new_sin * old_cos - new_cos * old_sin

    original_shape = key.shape
    shaped = key.reshape(key.shape[0], -1, head_size)
    rotated, passthrough = shaped[..., :rotary_dim], shaped[..., rotary_dim:]
    delta_cos = delta_cos.unsqueeze(1)
    delta_sin = delta_sin.unsqueeze(1)
    if neox:
        first, second = torch.chunk(rotated, 2, dim=-1)
        relocated = torch.cat(
            (first * delta_cos - second * delta_sin, second * delta_cos + first * delta_sin),
            dim=-1,
        )
    else:
        first, second = rotated[..., ::2], rotated[..., 1::2]
        relocated = torch.stack(
            (first * delta_cos - second * delta_sin, second * delta_cos + first * delta_sin),
            dim=-1,
        ).flatten(-2)
    if passthrough.numel():
        relocated = torch.cat((relocated, passthrough), dim=-1)
    return relocated.reshape(original_shape)


def validate_rope_support(model, rotary_emb) -> None:
    """Fail closed for RoPE variants not validated by CacheBlend.

    LMCache-Ascend only enables key relocation for full-dimensional, unscaled
    RoPE. Keep the same boundary here: silently relocating an unsupported key
    is worse than declining to start the connector.
    """
    head_size = int(rotary_emb.head_size)
    rotary_dim = int(getattr(rotary_emb, "rotary_dim", head_size))
    if rotary_dim != head_size:
        raise ValueError(
            "CacheBlend requires rotary_dim == head_size; "
            f"got rotary_dim={rotary_dim}, head_size={head_size}"
        )

    model_config = getattr(model, "config", None)
    rope_scaling = getattr(model_config, "rope_scaling", None)
    if rope_scaling:
        raise ValueError("CacheBlend does not support rope_scaling")
    partial_factor = getattr(model_config, "partial_rotary_factor", 1.0)
    partial_factor = 1.0 if partial_factor is None else float(partial_factor)
    if partial_factor != 1.0:
        raise ValueError(
            "CacheBlend requires partial_rotary_factor=1.0; "
            f"got {partial_factor}"
        )

    cache = getattr(rotary_emb, "cos_sin_cache", None)
    if not isinstance(cache, torch.Tensor) or cache.ndim != 2:
        raise ValueError("CacheBlend requires a two-dimensional RoPE cos_sin_cache")
    if cache.shape[0] == 0:
        raise ValueError("CacheBlend requires a non-empty RoPE cos_sin_cache")
    if cache.shape[-1] != rotary_dim:
        raise ValueError(
            "CacheBlend RoPE cache width does not match rotary_dim: "
            f"cache={cache.shape[-1]}, rotary_dim={rotary_dim}"
        )
    # The config check above rejects frequency scaling. This additional check
    # catches amplitude-scaled custom classes that omit their metadata, where
    # R(new) R(old)^T is not an inverse relocation.
    sample_indices = torch.tensor(
        sorted({0, max(0, cache.shape[0] - 1)}),
        dtype=torch.long,
        device=cache.device,
    )
    sample = cache.index_select(0, sample_indices).float()
    cosine, sine = sample.chunk(2, dim=-1)
    max_norm_error = (cosine.square() + sine.square() - 1.0).abs().max().item()
    if max_norm_error > 0.05:
        raise ValueError(
            "CacheBlend requires unit-norm, unscaled RoPE; "
            f"maximum sampled norm error is {max_norm_error:.4f}"
        )


@dataclass(slots=True)
class BlendExecutionState:
    compute_start: int
    positions: torch.Tensor
    gap_mask: torch.Tensor
    important_indices: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


class PagedBlendLoader:
    """Double-buffer layerwise CPU->NPU load and paged-cache scatter."""

    def __init__(
        self,
        kv_caches: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
        store: LocalPinnedCPUStore,
        request_id: str,
        plan: BlendPlan,
        slot_mapping: torch.Tensor,
        block_size: int,
        config: CacheBlendConfig,
        tracer: PipelineTracer | None = None,
        load_stream: object | None = None,
    ):
        self.kv_caches = kv_caches
        self.store = store
        self.request_id = request_id
        self.plan = plan
        self.slot_mapping = slot_mapping[: plan.allocation_end]
        self.block_size = block_size
        self.config = config
        self.tracer = tracer
        self._buffers: dict[int, torch.Tensor] = {}
        self._events: dict[int, object] = {}
        self._staging: list[torch.Tensor] = []
        self._reuse_events: dict[int, object] = {}
        self._hit_staging: torch.Tensor | None = None
        self._device_positions: tuple[torch.Tensor, torch.Tensor] | None = None
        self._old_positions_cpu = torch.cat(
            [
                torch.arange(segment.source_start, segment.source_start + segment.length)
                for segment in self.plan.segments
            ]
        )
        self._new_positions_cpu = torch.cat(
            [torch.arange(segment.target_start, segment.target_end) for segment in self.plan.segments]
        )
        self._load_stream = None
        if _device_name() == "npu" and config.event_pipeline:
            # The connector normally supplies one persistent Load stream. Keep
            # a fallback for direct loader construction in tests/tools.
            self._load_stream = load_stream or torch.npu.Stream()
        if self.tracer is not None and self.tracer.enabled:
            self.tracer.emit(
                "loader_initialized",
                request_id=request_id,
                apc_prefix_tokens=plan.apc_prefix_tokens,
                allocation_tokens=plan.allocation_tokens,
                hit_tokens=plan.hit_tokens,
                gap_tokens=plan.gap_tokens,
                segments=len(plan.segments),
            )

    def prefetch(self, layer_id: int, rotary_emb) -> None:
        if layer_id in self._buffers or layer_id >= len(self.kv_caches):
            return
        trace_enabled = self.tracer is not None and self.tracer.enabled
        started_ns = time.monotonic_ns() if trace_enabled else 0

        def load() -> None:
            kv_layer = self.kv_caches[layer_id]
            prefix = self.plan.apc_prefix_tokens
            sample_slots = self.slot_mapping[: prefix or 1]
            sample = gather_paged_kv(kv_layer, sample_slots, self.block_size)
            if not self._staging:
                shape = (2, self.plan.allocation_end, *sample.shape[2:])
                self._staging = [
                    torch.empty(shape, dtype=sample.dtype, device=sample.device),
                    torch.empty(shape, dtype=sample.dtype, device=sample.device),
                ]
            buffer_index = layer_id % 2
            reuse_event = self._reuse_events.pop(buffer_index, None)
            if reuse_event is not None and self._load_stream is not None:
                self._load_stream.wait_event(reuse_event)
            buffer = self._staging[buffer_index]
            if prefix:
                buffer[:, :prefix].copy_(sample)
            self._zero_gaps(buffer)
            if self.config.fused_segment_copy:
                self._load_segments_fused(layer_id, rotary_emb, buffer)
            else:
                self._load_segments_individually(layer_id, rotary_emb, buffer)
            self._buffers[layer_id] = buffer

        if self._load_stream is None:
            load()
            if trace_enabled:
                assert self.tracer is not None
                self.tracer.emit(
                    "layer_prefetch_enqueued",
                    request_id=self.request_id,
                    layer_id=layer_id,
                    duration_us=(time.monotonic_ns() - started_ns) / 1000,
                )
            return
        with torch.npu.stream(self._load_stream):
            load()
            event = torch.npu.Event()
            event.record(self._load_stream)
            self._events[layer_id] = event
        if trace_enabled:
            assert self.tracer is not None
            self.tracer.emit(
                "layer_prefetch_enqueued",
                request_id=self.request_id,
                layer_id=layer_id,
                duration_us=(time.monotonic_ns() - started_ns) / 1000,
            )

    def _zero_gaps(self, buffer: torch.Tensor) -> None:
        cursor = self.plan.apc_prefix_tokens
        for segment in self.plan.segments:
            if segment.target_start > cursor:
                buffer[:, cursor : segment.target_start].zero_()
            cursor = max(cursor, segment.target_end)
        if cursor < self.plan.allocation_end:
            buffer[:, cursor : self.plan.allocation_end].zero_()

    def _load_segments_individually(
        self,
        layer_id: int,
        rotary_emb,
        buffer: torch.Tensor,
    ) -> None:
        for segment in self.plan.segments:
            host = self.store.get_layer(self.request_id, segment, layer_id)
            device_kv = host.to(device=buffer.device, non_blocking=True)
            old_positions = torch.arange(
                segment.source_start,
                segment.source_start + segment.length,
                device=buffer.device,
                dtype=torch.long,
            )
            new_positions = torch.arange(
                segment.target_start,
                segment.target_end,
                device=buffer.device,
                dtype=torch.long,
            )
            device_kv[0] = relocate_key(rotary_emb, old_positions, new_positions, device_kv[0])
            buffer[:, segment.target_start : segment.target_end].copy_(device_kv)

    def _load_segments_fused(
        self,
        layer_id: int,
        rotary_emb,
        buffer: torch.Tensor,
    ) -> None:
        hosts = [
            self.store.get_layer(self.request_id, segment, layer_id)
            for segment in self.plan.segments
        ]
        if not hosts:
            return
        shape = (2, self.plan.hit_tokens, *hosts[0].shape[2:])
        if self._hit_staging is None:
            self._hit_staging = torch.empty(
                shape,
                dtype=hosts[0].dtype,
                device=buffer.device,
            )
        device_kv = self._hit_staging
        offset = 0
        for host, segment in zip(hosts, self.plan.segments, strict=True):
            end = offset + segment.length
            # Saved chunks are views into a larger pinned batch. Their K and V
            # planes are contiguous even when the combined [2, T, ...] view is
            # not. Copy the two planes directly into reusable NPU staging and
            # avoid constructing a second fused CPU buffer for every layer.
            device_kv[0, offset:end].copy_(host[0], non_blocking=True)
            device_kv[1, offset:end].copy_(host[1], non_blocking=True)
            offset += segment.length
        if self._device_positions is None:
            self._device_positions = (
                self._old_positions_cpu.to(
                    device=buffer.device,
                    dtype=torch.long,
                    non_blocking=True,
                ),
                self._new_positions_cpu.to(
                    device=buffer.device,
                    dtype=torch.long,
                    non_blocking=True,
                ),
            )
        old_positions, new_positions = self._device_positions
        device_kv[0] = relocate_key(rotary_emb, old_positions, new_positions, device_kv[0])
        if self.plan.allocation_end % self.block_size == 0:
            paged = (
                buffer[0].view(-1, self.block_size, *buffer.shape[2:]),
                buffer[1].view(-1, self.block_size, *buffer.shape[2:]),
            )
            scatter_paged_kv(
                device_kv,
                paged,
                new_positions,
                self.block_size,
            )
        else:
            buffer[:, new_positions] = device_kv

    def get(self, layer_id: int) -> torch.Tensor:
        event = self._events.pop(layer_id, None)
        if event is not None:
            torch.npu.current_stream().wait_event(event)
        if self.tracer is not None and self.tracer.enabled:
            self.tracer.emit(
                "layer_prefetch_wait_enqueued",
                request_id=self.request_id,
                layer_id=layer_id,
                had_event=event is not None,
            )
        return self._buffers.pop(layer_id)

    def commit(self, layer_id: int, contiguous: torch.Tensor) -> None:
        # Native APC already owns the prefix pages. Only the blended suffix is
        # overwritten so its block ownership remains entirely with vLLM.
        start = self.plan.apc_prefix_tokens
        if start >= self.plan.allocation_end:
            return
        scatter_paged_kv(
            contiguous[:, start:],
            self.kv_caches[layer_id],
            self.slot_mapping[start:],
            self.block_size,
        )
        if self._load_stream is not None:
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            self._reuse_events[layer_id % 2] = event
        if self.tracer is not None and self.tracer.enabled:
            self.tracer.emit(
                "layer_committed",
                request_id=self.request_id,
                layer_id=layer_id,
                suffix_tokens=self.plan.allocation_end - start,
            )


class Qwen3BlendExecutor:
    def __init__(self, model, config: CacheBlendConfig, block_size: int):
        self.model = model
        self.config = config
        if block_size <= 0:
            raise ValueError("vLLM KV cache block_size must be positive")
        self.block_size = block_size
        self.layers = list(model.model.layers[model.model.start_layer : model.model.end_layer])
        if not self.layers:
            raise RuntimeError("Qwen3 model has no local decoder layers")
        if max(config.check_layers) >= len(self.layers):
            raise ValueError(
                "CacheBlend check layer exceeds the local Qwen3 layer range: "
                f"check_layers={config.check_layers}, layers={len(self.layers)}"
            )
        validate_rope_support(model, self.layers[0].self_attn.rotary_emb)
        try:
            from vllm.distributed.parallel_state import get_tp_group

            self.tp_group = get_tp_group()
        except (AssertionError, ImportError):
            self.tp_group = None

    def run(
        self,
        request_id: str,
        tokens: list[int],
        plan: BlendPlan,
        loader: PagedBlendLoader,
    ) -> None:
        if plan.allocation_end <= plan.apc_prefix_tokens:
            return
        device = _device_name()
        input_ids = torch.tensor(
            tokens[plan.apc_prefix_tokens : plan.allocation_end],
            dtype=torch.long,
            device=device,
        )
        hidden_states = _embed(self.model, input_ids)
        residual = None
        positions = torch.arange(
            plan.apc_prefix_tokens,
            plan.allocation_end,
            device=hidden_states.device,
            dtype=torch.long,
        )
        gap_mask = torch.ones(plan.allocation_end, device=hidden_states.device, dtype=torch.bool)
        gap_mask[: plan.apc_prefix_tokens] = False
        for segment in plan.segments:
            gap_mask[segment.target_start : segment.target_end] = False
        state = BlendExecutionState(plan.apc_prefix_tokens, positions, gap_mask)

        loader.prefetch(0, self.layers[0].self_attn.rotary_emb)
        for layer_id, layer in enumerate(self.layers):
            old_kv = loader.get(layer_id)
            if layer_id + 1 < len(self.layers):
                loader.prefetch(layer_id + 1, self.layers[layer_id + 1].self_attn.rotary_emb)
            hidden_states, residual, state, old_kv = self._compute_layer(
                layer_id,
                layer,
                hidden_states,
                residual,
                state,
                old_kv,
                request_id=request_id,
                tracer=loader.tracer,
            )
            if loader.tracer is not None and loader.tracer.enabled:
                loader.tracer.emit(
                    "layer_compute_enqueued",
                    request_id=request_id,
                    layer_id=layer_id,
                    computed_tokens=int(hidden_states.shape[0]),
                    check_layer=layer_id in self.config.check_layers,
                )
            loader.commit(layer_id, old_kv)
        logger.info(
            "CacheBlend request %s prepared allocation=%d hits=%d gaps=%d segments=%d",
            request_id,
            plan.allocation_tokens,
            plan.hit_tokens,
            plan.gap_tokens,
            len(plan.segments),
        )

    def _compute_layer(
        self,
        layer_id: int,
        layer,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        state: BlendExecutionState,
        old_kv: torch.Tensor,
        request_id: str | None = None,
        tracer: PipelineTracer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, BlendExecutionState, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)

        qkv, _ = layer.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [layer.self_attn.q_size, layer.self_attn.kv_size, layer.self_attn.kv_size],
            dim=-1,
        )
        query_indices = (
            torch.arange(q.shape[0], device=q.device)
            if state.important_indices is None
            else state.important_indices
        )
        absolute_indices = state.compute_start + query_indices
        q, k = self._qk_post_process(layer.self_attn, state.positions, q, k)

        flat_old_k = old_kv[0].reshape(old_kv.shape[1], -1)
        flat_old_v = old_kv[1].reshape(old_kv.shape[1], -1)
        check_layer = layer_id in self.config.check_layers
        strategy = self.config.selection_strategy
        active_gap_mask = state.gap_mask[absolute_indices]
        deviation_selected: torch.Tensor | None = None
        sparse_q_selected: torch.Tensor | None = None
        sparse_q_stats: dict[str, int | bool] = {}

        if check_layer and strategy in {"kv_deviation", "compare"}:
            deviation_scores = torch.sum(
                (k.float() - flat_old_k[absolute_indices].float()) ** 2,
                dim=-1,
            )
            deviation_selected = self._select(
                deviation_scores,
                active_gap_mask,
                layer_id,
            )

        if check_layer and strategy in {"sparse_q", "compare"}:
            attn = layer.self_attn.attn
            num_heads = int(attn.num_heads)
            num_kv_heads = int(attn.num_kv_heads)
            head_size = int(attn.head_size)
            # The boundary score must see the fresh Keys produced by dense
            # contextualization, while APC Keys remain in their native pages.
            score_keys = flat_old_k if strategy == "sparse_q" else flat_old_k.clone()
            score_keys[absolute_indices] = k
            query_scores, forced_mask, sparse_q_stats = self._sparse_q_scores(
                q.view(-1, num_heads, head_size),
                score_keys.view(-1, num_kv_heads, head_size),
                absolute_indices,
                active_gap_mask,
                float(attn.impl.scale),
            )
            sparse_q_selected = self._select(
                query_scores,
                active_gap_mask,
                layer_id,
                forced_mask=forced_mask,
            )

        if check_layer and strategy == "compare":
            assert deviation_selected is not None and sparse_q_selected is not None
            self._trace_selection_comparison(
                tracer,
                request_id,
                layer_id,
                deviation_selected,
                sparse_q_selected,
                sparse_q_stats,
            )

        # K-deviation preserves the original CacheBlend behavior: prune before
        # the check-layer attention. Compare mode observes Sparse-Q but follows
        # this same path so it cannot change model output.
        if check_layer and strategy in {"kv_deviation", "compare"}:
            assert deviation_selected is not None
            selected = deviation_selected
            q, k, v = q[selected], k[selected], v[selected]
            residual = residual[selected]
            query_indices = query_indices[selected]
            absolute_indices = state.compute_start + query_indices
            state.important_indices = query_indices
            state.positions = state.positions[selected]
            state.attention_mask = None

        flat_old_k[absolute_indices] = k
        flat_old_v[absolute_indices] = v
        old_kv[0] = flat_old_k.reshape_as(old_kv[0])
        old_kv[1] = flat_old_v.reshape_as(old_kv[1])

        attn = layer.self_attn.attn
        num_heads = int(attn.num_heads)
        num_kv_heads = int(attn.num_kv_heads)
        head_size = int(attn.head_size)
        query = q.view(-1, num_heads, head_size)
        key = old_kv[0].view(-1, num_kv_heads, head_size)
        value = old_kv[1].view(-1, num_kv_heads, head_size)
        attn_output = self._attention(
            query,
            key,
            value,
            absolute_indices,
            float(attn.impl.scale),
            state,
        ).reshape(-1, num_heads * head_size)
        hidden_states, _ = layer.self_attn.o_proj(attn_output)
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)

        # SparseX uses the boundary layer itself for dense contextualization,
        # then applies the Query-aware set to every later layer.
        if check_layer and strategy == "sparse_q":
            assert sparse_q_selected is not None
            hidden_states = hidden_states[sparse_q_selected]
            residual = residual[sparse_q_selected]
            query_indices = query_indices[sparse_q_selected]
            state.important_indices = query_indices
            state.positions = state.positions[sparse_q_selected]
            state.attention_mask = None
            if tracer is not None and tracer.enabled:
                tracer.emit(
                    "selection_finished",
                    request_id=request_id,
                    layer_id=layer_id,
                    strategy="sparse_q",
                    selected_tokens=sparse_q_selected.numel(),
                    **sparse_q_stats,
                )
        return hidden_states, residual, state, old_kv

    def _sparse_q_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        absolute_indices: torch.Tensor,
        gap_mask: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | bool]]:
        """Aggregate causal Sparse-Q attention into one score per active token.

        Query chunks bound the temporary ``[KV heads, groups, Q, context]``
        probability tensor. The result remains local to this TP rank; `_select`
        performs the existing all-reduce before global Top-K selection.
        """
        query_mask = gap_mask.clone()
        forced_mask = gap_mask.clone()
        gap_count = torch.where(gap_mask)[0].numel()
        has_gap = bool(query_mask.any().item())
        tail_fallback = not has_gap
        if tail_fallback:
            tail = min(self.config.tail_query_tokens, query.shape[0])
            query_mask[-tail:] = True
            forced_mask[-tail:] = True

        radius = self.config.overflow_blocks * self.block_size
        if radius and has_gap:
            # A prefix-sum window expands each non-reuse interval by the
            # configured number of cache blocks without a host-side index loop.
            values = gap_mask.to(torch.int32)
            prefix = torch.cat((values.new_zeros(1), values.cumsum(0)))
            positions = torch.arange(values.numel(), device=values.device)
            left = (positions - radius).clamp_min(0)
            right = (positions + radius + 1).clamp_max(values.numel())
            forced_mask |= prefix[right] > prefix[left]

        sparse_indices = torch.where(query_mask)[0]
        sparse_query = query.index_select(0, sparse_indices)
        sparse_positions = absolute_indices.index_select(0, sparse_indices)
        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        if num_heads % num_kv_heads:
            raise ValueError(
                "Sparse-Q requires query heads to be divisible by KV heads; "
                f"got {num_heads} and {num_kv_heads}"
            )
        groups = num_heads // num_kv_heads
        key_transposed = key.permute(1, 2, 0).contiguous()
        key_positions = torch.arange(key.shape[0], device=query.device)
        full_scores = torch.zeros(key.shape[0], dtype=torch.float32, device=query.device)
        chunk_size = self.config.query_score_chunk_size

        for start in range(0, sparse_query.shape[0], chunk_size):
            end = min(start + chunk_size, sparse_query.shape[0])
            query_chunk = sparse_query[start:end]
            query_chunk = query_chunk.view(end - start, num_kv_heads, groups, -1)
            query_chunk = query_chunk.permute(1, 2, 0, 3).contiguous()
            logits = torch.matmul(query_chunk, key_transposed[:, None]).float()
            logits.mul_(scale)
            causal_mask = sparse_positions[start:end, None] < key_positions[None, :]
            logits.masked_fill_(causal_mask[None, None], torch.finfo(logits.dtype).min)
            full_scores.add_(torch.softmax(logits, dim=-1).sum(dim=(0, 1, 2)))

        return (
            full_scores.index_select(0, absolute_indices),
            forced_mask,
            {
                "active_tokens": query.shape[0],
                "gap_tokens": gap_count,
                "cached_tokens": query.shape[0] - gap_count,
                "sparse_query_tokens": sparse_indices.numel(),
                "tail_fallback": tail_fallback,
                "overflow_blocks": self.config.overflow_blocks,
            },
        )

    @staticmethod
    def _trace_selection_comparison(
        tracer: PipelineTracer | None,
        request_id: str | None,
        layer_id: int,
        deviation_selected: torch.Tensor,
        sparse_q_selected: torch.Tensor,
        sparse_q_stats: dict[str, int | bool],
    ) -> None:
        if tracer is None or not tracer.enabled:
            return
        size = max(
            int(deviation_selected.max().item()) if deviation_selected.numel() else -1,
            int(sparse_q_selected.max().item()) if sparse_q_selected.numel() else -1,
        ) + 1
        deviation_mask = torch.zeros(
            size,
            dtype=torch.bool,
            device=deviation_selected.device,
        )
        sparse_q_mask = torch.zeros_like(deviation_mask)
        deviation_mask[deviation_selected] = True
        sparse_q_mask[sparse_q_selected] = True
        overlap = int((deviation_mask & sparse_q_mask).sum().item())
        union = int((deviation_mask | sparse_q_mask).sum().item())
        tracer.emit(
            "selection_compared",
            request_id=request_id,
            layer_id=layer_id,
            deviation_tokens=deviation_selected.numel(),
            sparse_q_tokens=sparse_q_selected.numel(),
            overlap_tokens=overlap,
            jaccard=overlap / max(union, 1),
            **sparse_q_stats,
        )

    @staticmethod
    def _qk_post_process(attn_layer, positions, q, k):
        q_heads = q.view(*q.shape[:-1], q.shape[-1] // attn_layer.head_dim, attn_layer.head_dim)
        q = attn_layer.q_norm(q_heads).view(q.shape)
        k_heads = k.view(*k.shape[:-1], k.shape[-1] // attn_layer.head_dim, attn_layer.head_dim)
        k = attn_layer.k_norm(k_heads).view(k.shape)
        return attn_layer.rotary_emb(positions, q, k)

    def _select(
        self,
        scores: torch.Tensor,
        gap_mask: torch.Tensor,
        layer_id: int | None = None,
        forced_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tp_group = self.tp_group
        use_tp = (
            self.config.tp_global_selection and tp_group is not None and tp_group.world_size > 1
        )
        if use_tp:
            scores = tp_group.all_reduce(scores)
        base_mask = gap_mask.clone()
        if forced_mask is not None:
            if forced_mask.shape != gap_mask.shape:
                raise ValueError("forced selection mask must match the active token shape")
            base_mask |= forced_mask
        cached_count = scores.numel() - torch.where(gap_mask)[0].numel()
        check_index = (
            self.config.check_layers.index(layer_id) if layer_id in self.config.check_layers else 0
        )
        ratio_index = min(len(self.config.recompute_ratios) - 1, check_index)
        topk = int(cached_count * self.config.recompute_ratios[ratio_index])
        available_count = scores.numel() - torch.where(base_mask)[0].numel()
        if cached_count and available_count:
            topk = min(max(topk, 1), available_count)
        else:
            topk = 0
        selected_count = scores.numel() - available_count + topk
        choose_here = not use_tp or tp_group.rank_in_group == 0
        if choose_here:
            selected_mask = base_mask
            if topk:
                cached_scores = scores.masked_fill(base_mask, float("-inf"))
                selected_mask[torch.topk(cached_scores, k=topk).indices] = True
            selected = torch.where(selected_mask)[0]
        else:
            selected = torch.empty(selected_count, dtype=torch.long, device=scores.device)
        return tp_group.broadcast(selected, src=0) if use_tp else selected

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_positions: torch.Tensor,
        scale: float,
        state: BlendExecutionState,
    ) -> torch.Tensor:
        k_positions = torch.arange(key.shape[0], device=query.device)
        cache_mask = self.config.cache_attention_mask and state.important_indices is not None
        mask = state.attention_mask if cache_mask else None
        if mask is None:
            mask = (q_positions[:, None] < k_positions[None, :])[None, None]
            if cache_mask:
                state.attention_mask = mask
        if query.device.type == "npu":
            from torch_npu import npu_fused_infer_attention_score

            return npu_fused_infer_attention_score(
                query=query[None].contiguous(),
                key=key[None].contiguous(),
                value=value[None].contiguous(),
                atten_mask=mask,
                actual_seq_lengths=None,
                actual_seq_lengths_kv=None,
                num_heads=query.shape[1],
                num_key_value_heads=key.shape[1],
                scale=scale,
                input_layout="BSND",
                sparse_mode=0,
                softmax_lse_flag=False,
            )[0].reshape_as(query)

        repeats = query.shape[1] // key.shape[1]
        key = (
            key[:, :, None]
            .expand(-1, -1, repeats, -1)
            .reshape(key.shape[0], query.shape[1], key.shape[-1])
        )
        value = (
            value[:, :, None]
            .expand(-1, -1, repeats, -1)
            .reshape(value.shape[0], query.shape[1], value.shape[-1])
        )
        weights = torch.einsum("qhd,khd->hqk", query, key) * scale
        weights = weights.masked_fill(mask[0], torch.finfo(weights.dtype).min)
        probs = torch.softmax(weights.float(), dim=-1).to(query.dtype)
        return torch.einsum("hqk,khd->qhd", probs, value)
