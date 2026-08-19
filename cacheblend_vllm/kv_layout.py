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


def gather_paged_kv(
    kv_layer: KVLayer,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather paged KV as contiguous ``[2, tokens, ...]``."""
    separate = _separate(kv_layer)
    if separate is not None:
        key, value = separate
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
