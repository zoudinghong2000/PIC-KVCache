"""Qwen3 CacheBlend V1 execution on Ascend.

This module is intentionally independent of LMCache-Ascend. It uses vLLM's
loaded Qwen3 layers, torch-npu fused attention, and pure-torch paged KV
gather/scatter. Optional native transfer kernels can replace the latter without
changing the connector contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from .config import CacheBlendConfig
from .kv_layout import gather_paged_kv, scatter_paged_kv
from .storage import LocalPinnedCPUStore
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
    """Move an already-RoPE-encoded K tensor to new absolute positions."""
    if torch.equal(old_positions, new_positions):
        return key
    head_size = int(rotary_emb.head_size)
    neox = bool(rotary_emb.is_neox_style)
    flat = key.reshape(key.shape[0], -1)
    shuffled = _shuffle_rope_halves(flat, head_size, neox)
    _, inverse_once = rotary_emb(old_positions, shuffled, shuffled)
    unrotated = _shuffle_rope_halves(inverse_once, head_size, neox)
    _, relocated = rotary_emb(new_positions, unrotated, unrotated)
    return relocated.reshape_as(key)


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
    ):
        self.kv_caches = kv_caches
        self.store = store
        self.request_id = request_id
        self.plan = plan
        self.slot_mapping = slot_mapping[: plan.allocation_end]
        self.block_size = block_size
        self.config = config
        self._buffers: dict[int, torch.Tensor] = {}
        self._events: dict[int, object] = {}
        self._load_stream = None
        if _device_name() == "npu" and config.event_pipeline:
            self._load_stream = torch.npu.Stream()

    def prefetch(self, layer_id: int, rotary_emb) -> None:
        if layer_id in self._buffers or layer_id >= len(self.kv_caches):
            return

        def load() -> None:
            kv_layer = self.kv_caches[layer_id]
            prefix = self.plan.apc_prefix_tokens
            if prefix:
                prefix_kv = gather_paged_kv(kv_layer, self.slot_mapping[:prefix], self.block_size)
                shape = (2, self.plan.allocation_end, *prefix_kv.shape[2:])
                buffer = torch.zeros(shape, dtype=prefix_kv.dtype, device=prefix_kv.device)
                buffer[:, :prefix].copy_(prefix_kv)
            else:
                sample = gather_paged_kv(kv_layer, self.slot_mapping[:1], self.block_size)
                shape = (2, self.plan.allocation_end, *sample.shape[2:])
                buffer = torch.zeros(shape, dtype=sample.dtype, device=sample.device)
            if self.config.fused_segment_copy:
                self._load_segments_fused(layer_id, rotary_emb, buffer)
            else:
                self._load_segments_individually(layer_id, rotary_emb, buffer)
            self._buffers[layer_id] = buffer

        if self._load_stream is None:
            load()
            return
        with torch.npu.stream(self._load_stream):
            load()
            event = torch.npu.Event()
            event.record(self._load_stream)
            self._events[layer_id] = event

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
        try:
            fused_host = torch.empty(
                shape,
                dtype=hosts[0].dtype,
                device="cpu",
                pin_memory=True,
            )
        except RuntimeError:
            fused_host = torch.empty(shape, dtype=hosts[0].dtype, device="cpu")
        offset = 0
        old_position_parts: list[torch.Tensor] = []
        new_position_parts: list[torch.Tensor] = []
        for host, segment in zip(hosts, self.plan.segments, strict=True):
            fused_host[:, offset : offset + segment.length].copy_(host)
            old_position_parts.append(
                torch.arange(segment.source_start, segment.source_start + segment.length)
            )
            new_position_parts.append(torch.arange(segment.target_start, segment.target_end))
            offset += segment.length
        old_positions = torch.cat(old_position_parts).to(
            device=buffer.device, dtype=torch.long, non_blocking=True
        )
        new_positions = torch.cat(new_position_parts).to(
            device=buffer.device, dtype=torch.long, non_blocking=True
        )
        device_kv = fused_host.to(device=buffer.device, non_blocking=True)
        device_kv[0] = relocate_key(rotary_emb, old_positions, new_positions, device_kv[0])
        buffer[:, new_positions] = device_kv

    def get(self, layer_id: int) -> torch.Tensor:
        event = self._events.pop(layer_id, None)
        if event is not None:
            torch.npu.current_stream().wait_event(event)
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


class Qwen3BlendExecutor:
    def __init__(self, model, config: CacheBlendConfig):
        self.model = model
        self.config = config
        self.layers = list(model.model.layers[model.model.start_layer : model.model.end_layer])
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

        if not self.layers:
            raise RuntimeError("Qwen3 model has no local decoder layers")
        loader.prefetch(0, self.layers[0].self_attn.rotary_emb)
        for layer_id, layer in enumerate(self.layers):
            old_kv = loader.get(layer_id)
            if layer_id + 1 < len(self.layers):
                loader.prefetch(layer_id + 1, self.layers[layer_id + 1].self_attn.rotary_emb)
            hidden_states, residual, state, old_kv = self._compute_layer(
                layer_id, layer, hidden_states, residual, state, old_kv
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
        if layer_id in self.config.check_layers:
            scores = torch.sum(
                (k.float() - flat_old_k[absolute_indices].float()) ** 2,
                dim=-1,
            )
            selected = self._select(
                scores,
                state.gap_mask[absolute_indices],
                layer_id,
            )
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
        return hidden_states, residual, state, old_kv

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
    ) -> torch.Tensor:
        tp_group = self.tp_group
        use_tp = (
            self.config.tp_global_selection and tp_group is not None and tp_group.world_size > 1
        )
        if use_tp:
            scores = tp_group.all_reduce(scores)
        gap_positions = torch.where(gap_mask)[0]
        cached_count = scores.numel() - gap_positions.numel()
        check_index = (
            self.config.check_layers.index(layer_id) if layer_id in self.config.check_layers else 0
        )
        ratio_index = min(len(self.config.recompute_ratios) - 1, check_index)
        topk = int(cached_count * self.config.recompute_ratios[ratio_index])
        if cached_count:
            topk = min(max(topk, 1), cached_count)
        selected_count = gap_positions.numel() + topk
        choose_here = not use_tp or tp_group.rank_in_group == 0
        if choose_here:
            selected_mask = gap_mask.clone()
            if topk:
                cached_scores = scores.masked_fill(gap_mask, float("-inf"))
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
