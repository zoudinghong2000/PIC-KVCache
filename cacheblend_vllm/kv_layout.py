"""Pure-torch gather/scatter helpers for vLLM paged KV layouts."""

from __future__ import annotations

from collections.abc import Sequence

import torch

KVLayer = torch.Tensor | Sequence[torch.Tensor]


def _separate(kv_layer: KVLayer) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(kv_layer, torch.Tensor):
        return None
    if len(kv_layer) != 2:
        raise ValueError(f"separate KV layout requires two tensors, got {len(kv_layer)}")
    key, value = kv_layer
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("separate KV layout must contain tensors")
    return key, value


def _slots(slot_mapping: torch.Tensor, device: torch.device) -> torch.Tensor:
    return slot_mapping.to(device=device, dtype=torch.long, non_blocking=True)


def _gather_ascend_separate(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    """Use CANN's paged-cache gather instead of NPU advanced indexing.

    ``npu_paged_cache_load`` accepts a logical block table.  Reconstructing it
    from the connector's ordered slot list also handles a concatenation of
    independently cached, block-aligned chunks.
    """
    if key_cache.device.type != "npu" or key_cache.ndim != 4:
        return None
    try:
        import torch_npu
    except ImportError:
        return None

    slots_cpu = slot_mapping.detach().to(device="cpu", dtype=torch.long)
    if slots_cpu.numel() == 0:
        shape = (2, 0, *key_cache.shape[2:])
        return torch.empty(shape, dtype=key_cache.dtype, device=key_cache.device)
    blocks = slots_cpu // block_size
    offsets = slots_cpu % block_size
    block_starts = torch.ones_like(blocks, dtype=torch.bool)
    block_starts[1:] = offsets[1:].eq(0)
    logical_blocks = blocks[block_starts]
    expected_blocks = (int(offsets[0]) + slots_cpu.numel() + block_size - 1) // block_size
    if logical_blocks.numel() != expected_blocks:
        return None

    device = key_cache.device
    block_table = logical_blocks.to(device=device, dtype=torch.int32).unsqueeze(0)
    context_lens = torch.tensor([slots_cpu.numel()], dtype=torch.int32, device=device)
    seq_starts = offsets[:1].to(device=device, dtype=torch.int32)
    key = torch.empty(
        (slots_cpu.numel(), *key_cache.shape[2:]),
        dtype=key_cache.dtype,
        device=device,
    )
    value = torch.empty_like(key)
    torch_npu.atb.npu_paged_cache_load(
        key_cache,
        value_cache,
        block_table,
        context_lens,
        seq_starts=seq_starts,
        key=key,
        value=value,
    )
    return torch.stack((key, value), dim=0)


def gather_paged_kv(
    kv_layer: KVLayer,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather paged KV as contiguous ``[2, tokens, ...]``."""
    separate = _separate(kv_layer)
    if separate is not None:
        key, value = separate
        native = _gather_ascend_separate(key, value, slot_mapping, block_size)
        if native is not None:
            return native
        slots = _slots(slot_mapping, key.device)
        blocks, offsets = slots // block_size, slots % block_size
        return torch.stack((key[blocks, offsets], value[blocks, offsets]), dim=0)
    assert isinstance(kv_layer, torch.Tensor)
    slots = _slots(slot_mapping, kv_layer.device)
    blocks, offsets = slots // block_size, slots % block_size
    if kv_layer.shape[0] == 2:
        return kv_layer[:, blocks, offsets]
    if kv_layer.ndim >= 4 and kv_layer.shape[1] == 2:
        return kv_layer[blocks, :, offsets].transpose(0, 1)
    raise ValueError(f"unsupported vLLM KV layout {tuple(kv_layer.shape)}")


def scatter_paged_kv(
    contiguous: torch.Tensor,
    kv_layer: KVLayer,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter ``[2, tokens, ...]`` into a vLLM paged KV layer."""
    if contiguous.shape[0] != 2:
        raise ValueError("contiguous KV must have K/V as its first dimension")
    separate = _separate(kv_layer)
    if separate is not None:
        key, value = separate
        if key.device.type == "npu" and key.ndim == 4:
            try:
                from vllm_ascend.device.device_op import DeviceOperator

                slots = slot_mapping.to(device=key.device, dtype=torch.int32, non_blocking=True)
                DeviceOperator.reshape_and_cache(contiguous[0], contiguous[1], key, value, slots)
                return
            except (ImportError, AttributeError):
                # Preserve the pure-torch path on older torch-npu releases.
                pass
        slots = _slots(slot_mapping, key.device)
        blocks, offsets = slots // block_size, slots % block_size
        key[blocks, offsets] = contiguous[0]
        value[blocks, offsets] = contiguous[1]
        return
    assert isinstance(kv_layer, torch.Tensor)
    slots = _slots(slot_mapping, kv_layer.device)
    blocks, offsets = slots // block_size, slots % block_size
    if kv_layer.shape[0] == 2:
        kv_layer[:, blocks, offsets] = contiguous
        return
    if kv_layer.ndim >= 4 and kv_layer.shape[1] == 2:
        kv_layer[blocks, :, offsets] = contiguous.transpose(0, 1)
        return
    raise ValueError(f"unsupported vLLM KV layout {tuple(kv_layer.shape)}")
