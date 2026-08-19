"""Per-rank pinned-CPU CacheBlend storage.

The store publishes a fingerprint only after every KV layer for a chunk has
been committed. Readers pin entries for the lifetime of a request so eviction
cannot invalidate an in-flight blend plan.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

from .matcher import FingerprintRecord, TokenRangeMatcher
from .types import BlendSegment, SegmentId


@dataclass(slots=True)
class _Entry:
    record: FingerprintRecord
    num_layers: int
    layers: dict[int, torch.Tensor] = field(default_factory=dict)
    reserved_layers: set[int] = field(default_factory=set)
    size_bytes: int = 0
    pin_count: int = 0
    committed: bool = False
    last_access: float = field(default_factory=time.monotonic)


class LocalPinnedCPUStore:
    def __init__(self, chunk_size: int, max_bytes: int, pin_memory: bool = True):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.matcher = TokenRangeMatcher(chunk_size)
        self.max_bytes = int(max_bytes)
        self.pin_memory = pin_memory
        self._entries: OrderedDict[SegmentId, _Entry] = OrderedDict()
        self._bytes = 0
        self._request_pins: dict[str, tuple[SegmentId, ...]] = {}
        self._lock = threading.RLock()

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def committed_entries(self) -> int:
        return len(self.matcher)

    def begin_put(
        self,
        model_scope: str,
        source_start: int,
        tokens: Sequence[int],
        num_layers: int,
    ) -> SegmentId:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        record = self.matcher.make_record(model_scope, source_start, tokens)
        with self._lock:
            existing = self._entries.get(record.segment_id)
            if existing is not None:
                if existing.record.tokens != record.tokens:
                    raise RuntimeError("content-hash collision for an existing segment id")
                if existing.num_layers != num_layers:
                    raise ValueError("num_layers changed for an existing segment")
                self._touch(record.segment_id, existing)
                return record.segment_id
            self._entries[record.segment_id] = _Entry(record=record, num_layers=num_layers)
        return record.segment_id

    def put_layer(self, segment_id: SegmentId, layer_id: int, kv: torch.Tensor) -> bool:
        """Store one layer and return whether the segment became committed."""
        with self._lock:
            entry = self._entries.get(segment_id)
            if entry is None:
                raise KeyError(segment_id)
            if not 0 <= layer_id < entry.num_layers:
                raise ValueError(f"layer {layer_id} is outside [0, {entry.num_layers})")

        cpu_tensor = kv.detach().to(device="cpu", copy=True).contiguous()
        if self.pin_memory and not cpu_tensor.is_pinned():
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except RuntimeError:
                # CPU-only development and CI environments may not expose a pin
                # allocator. Correctness does not depend on pinning.
                pass

        return self.put_layer_host(segment_id, layer_id, cpu_tensor)

    def reserve_layer(self, segment_id: SegmentId, layer_id: int) -> bool:
        """Reserve a layer for an asynchronous transfer.

        Returns false when the layer is already present or being transferred.
        """
        with self._lock:
            entry = self._entries.get(segment_id)
            if entry is None:
                raise KeyError(segment_id)
            if not 0 <= layer_id < entry.num_layers:
                raise ValueError(f"layer {layer_id} is outside [0, {entry.num_layers})")
            if layer_id in entry.layers or layer_id in entry.reserved_layers:
                return False
            entry.reserved_layers.add(layer_id)
            return True

    def cancel_layer(self, segment_id: SegmentId, layer_id: int) -> None:
        with self._lock:
            entry = self._entries.get(segment_id)
            if entry is None:
                return
            entry.reserved_layers.discard(layer_id)
            if not entry.layers and not entry.reserved_layers and not entry.pin_count:
                self._remove_unlocked(segment_id, entry)

    def put_layer_host(
        self,
        segment_id: SegmentId,
        layer_id: int,
        cpu_tensor: torch.Tensor,
    ) -> bool:
        """Publish an owned, contiguous CPU tensor without another copy."""
        if cpu_tensor.device.type != "cpu":
            raise ValueError("put_layer_host requires a CPU tensor")
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()
        with self._lock:
            entry = self._entries.get(segment_id)
            if entry is None:
                return False
            if not 0 <= layer_id < entry.num_layers:
                raise ValueError(f"layer {layer_id} is outside [0, {entry.num_layers})")
            entry.reserved_layers.discard(layer_id)
            previous = entry.layers.get(layer_id)
            if previous is not None:
                self._bytes -= previous.numel() * previous.element_size()
                entry.size_bytes -= previous.numel() * previous.element_size()
            entry.layers[layer_id] = cpu_tensor
            added = cpu_tensor.numel() * cpu_tensor.element_size()
            entry.size_bytes += added
            self._bytes += added
            entry.last_access = time.monotonic()
            became_committed = not entry.committed and len(entry.layers) == entry.num_layers
            if became_committed:
                entry.committed = True
                self.matcher.register(entry.record)
            self._touch(segment_id, entry)
            self._evict_unlocked()
            return became_committed

    def abort_put(self, segment_id: SegmentId) -> None:
        with self._lock:
            entry = self._entries.get(segment_id)
            if entry is None or entry.pin_count:
                return
            self._remove_unlocked(segment_id, entry)

    def match(
        self,
        model_scope: str,
        tokens: Sequence[int],
        start_offset: int,
    ) -> tuple[BlendSegment, ...]:
        candidates = self.matcher.match(model_scope, tokens, start_offset)
        return self.validate(candidates)

    def validate(self, segments: Iterable[BlendSegment]) -> tuple[BlendSegment, ...]:
        valid: list[BlendSegment] = []
        with self._lock:
            for segment in segments:
                entry = self._entries.get(segment.segment_id)
                if entry is None or not entry.committed:
                    continue
                self._touch(segment.segment_id, entry)
                valid.append(segment)
        return tuple(valid)

    def pin(self, request_id: str, segments: Iterable[BlendSegment]) -> tuple[BlendSegment, ...]:
        """Atomically pin the committed subset of ``segments`` for a request."""
        requested = tuple(segments)
        with self._lock:
            self.release(request_id)
            valid: list[BlendSegment] = []
            ids: list[SegmentId] = []
            for segment in requested:
                entry = self._entries.get(segment.segment_id)
                if entry is None or not entry.committed:
                    continue
                entry.pin_count += 1
                self._touch(segment.segment_id, entry)
                valid.append(segment)
                ids.append(segment.segment_id)
            if ids:
                self._request_pins[request_id] = tuple(ids)
            return tuple(valid)

    def release(self, request_id: str) -> None:
        with self._lock:
            ids = self._request_pins.pop(request_id, ())
            for segment_id in ids:
                entry = self._entries.get(segment_id)
                if entry is not None:
                    entry.pin_count = max(0, entry.pin_count - 1)
            self._evict_unlocked()

    def get_layer(self, request_id: str, segment: BlendSegment, layer_id: int) -> torch.Tensor:
        with self._lock:
            if segment.segment_id not in self._request_pins.get(request_id, ()):
                raise RuntimeError(f"segment {segment.segment_id} is not pinned by {request_id}")
            entry = self._entries.get(segment.segment_id)
            if entry is None or not entry.committed:
                raise KeyError(segment.segment_id)
            try:
                tensor = entry.layers[layer_id]
            except KeyError as error:
                raise KeyError(
                    f"layer {layer_id} is not stored for {segment.segment_id}"
                ) from error
            self._touch(segment.segment_id, entry)
            return tensor

    def clear(self) -> None:
        with self._lock:
            for segment_id in list(self._entries):
                self.matcher.unregister(segment_id)
            self._entries.clear()
            self._request_pins.clear()
            self._bytes = 0

    def _touch(self, segment_id: SegmentId, entry: _Entry) -> None:
        entry.last_access = time.monotonic()
        self._entries.move_to_end(segment_id)

    def _remove_unlocked(self, segment_id: SegmentId, entry: _Entry) -> None:
        self.matcher.unregister(segment_id)
        self._entries.pop(segment_id, None)
        self._bytes -= entry.size_bytes

    def _evict_unlocked(self) -> None:
        if self._bytes <= self.max_bytes:
            return
        for segment_id, entry in list(self._entries.items()):
            if self._bytes <= self.max_bytes:
                break
            # An in-progress layerwise write is atomic from the matcher's point
            # of view. Evict it only through abort_put; otherwise a later layer
            # would write into a segment that disappeared mid-commit.
            if entry.pin_count or not entry.committed:
                continue
            self._remove_unlocked(segment_id, entry)
