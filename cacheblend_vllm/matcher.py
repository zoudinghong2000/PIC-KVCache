"""Thread-safe, content-addressed token-range matcher."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .hashing import content_hash, rolling_hash_values
from .types import BlendSegment, SegmentId


@dataclass(frozen=True, slots=True)
class FingerprintRecord:
    segment_id: SegmentId
    tokens: tuple[int, ...]


class TokenRangeMatcher:
    """Match fixed-size stored chunks at every query-token offset."""

    def __init__(self, chunk_size: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self._by_scope_hash: dict[tuple[str, int], list[FingerprintRecord]] = defaultdict(list)
        self._scope_hashes: dict[str, set[int]] = defaultdict(set)
        self._by_id: dict[SegmentId, FingerprintRecord] = {}
        self._lock = threading.RLock()

    def make_record(
        self,
        model_scope: str,
        source_start: int,
        tokens: Sequence[int],
    ) -> FingerprintRecord:
        if len(tokens) != self.chunk_size:
            raise ValueError(
                f"fingerprint requires exactly {self.chunk_size} tokens, got {len(tokens)}"
            )
        token_tuple = tuple(int(token) for token in tokens)
        segment_id = SegmentId(
            model_scope=model_scope,
            content_hash=content_hash(token_tuple),
            source_start=source_start,
            token_count=len(token_tuple),
        )
        return FingerprintRecord(segment_id=segment_id, tokens=token_tuple)

    def register(self, record: FingerprintRecord) -> None:
        with self._lock:
            previous = self._by_id.get(record.segment_id)
            if previous == record:
                return
            if previous is not None:
                self.unregister(record.segment_id)
            self._by_id[record.segment_id] = record
            key = (record.segment_id.model_scope, record.segment_id.content_hash)
            self._by_scope_hash[key].append(record)
            self._scope_hashes[record.segment_id.model_scope].add(record.segment_id.content_hash)

    def unregister(self, segment_id: SegmentId) -> None:
        with self._lock:
            record = self._by_id.pop(segment_id, None)
            if record is None:
                return
            key = (segment_id.model_scope, segment_id.content_hash)
            records = self._by_scope_hash[key]
            records[:] = [candidate for candidate in records if candidate.segment_id != segment_id]
            if not records:
                del self._by_scope_hash[key]
                scope_hashes = self._scope_hashes[segment_id.model_scope]
                scope_hashes.discard(segment_id.content_hash)
                if not scope_hashes:
                    del self._scope_hashes[segment_id.model_scope]

    def match(
        self,
        model_scope: str,
        tokens: Sequence[int],
        start_offset: int = 0,
    ) -> tuple[BlendSegment, ...]:
        """Return leftmost-greedy, non-overlapping exact token matches."""
        if start_offset < 0 or start_offset > len(tokens):
            raise ValueError("start_offset is outside the token sequence")
        suffix = tokens[start_offset:]
        matches: list[BlendSegment] = []
        next_allowed = start_offset
        with self._lock:
            known_hashes = self._scope_hashes.get(model_scope)
            if not known_hashes:
                return ()
            hashes = rolling_hash_values(suffix, self.chunk_size)
            known = np.fromiter(
                known_hashes,
                dtype=np.uint64,
                count=len(known_hashes),
            )
            candidate_starts = np.flatnonzero(np.isin(hashes, known))
            for relative_start_value in candidate_starts:
                relative_start = int(relative_start_value)
                value = int(hashes[relative_start])
                target_start = start_offset + relative_start
                if target_start < next_allowed:
                    continue
                candidates = self._by_scope_hash.get((model_scope, value), ())
                if not candidates:
                    continue
                query_tokens = tuple(
                    int(token) for token in tokens[target_start : target_start + self.chunk_size]
                )
                record = next(
                    (candidate for candidate in candidates if candidate.tokens == query_tokens),
                    None,
                )
                if record is None:
                    continue
                matches.append(
                    BlendSegment(segment_id=record.segment_id, target_start=target_start)
                )
                next_allowed = target_start + self.chunk_size
        return tuple(matches)

    def validate(self, segments: Iterable[BlendSegment]) -> tuple[BlendSegment, ...]:
        with self._lock:
            return tuple(segment for segment in segments if segment.segment_id in self._by_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)
