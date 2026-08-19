"""Serializable CacheBlend request and segment types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True, slots=True)
class SegmentId:
    model_scope: str
    content_hash: int
    source_start: int
    token_count: int

    def __post_init__(self) -> None:
        if not self.model_scope:
            raise ValueError("model_scope cannot be empty")
        if self.source_start < 0 or self.token_count <= 0:
            raise ValueError("invalid source segment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_scope": self.model_scope,
            "content_hash": self.content_hash,
            "source_start": self.source_start,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SegmentId:
        return cls(
            model_scope=str(value["model_scope"]),
            content_hash=int(value["content_hash"]),
            source_start=int(value["source_start"]),
            token_count=int(value["token_count"]),
        )


@dataclass(frozen=True, order=True, slots=True)
class BlendSegment:
    segment_id: SegmentId
    target_start: int

    def __post_init__(self) -> None:
        if self.target_start < 0:
            raise ValueError("target_start cannot be negative")

    @property
    def source_start(self) -> int:
        return self.segment_id.source_start

    @property
    def length(self) -> int:
        return self.segment_id.token_count

    @property
    def target_end(self) -> int:
        return self.target_start + self.length

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id.to_dict(),
            "target_start": self.target_start,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BlendSegment:
        return cls(
            segment_id=SegmentId.from_dict(value["segment_id"]),
            target_start=int(value["target_start"]),
        )


@dataclass(frozen=True, slots=True)
class BlendPlan:
    apc_prefix_tokens: int
    allocation_end: int
    hit_tokens: int
    segments: tuple[BlendSegment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.apc_prefix_tokens < 0 or self.allocation_end < self.apc_prefix_tokens:
            raise ValueError("invalid CacheBlend allocation span")
        ordered = tuple(sorted(self.segments, key=lambda segment: segment.target_start))
        previous_end = self.apc_prefix_tokens
        total = 0
        for segment in ordered:
            if segment.target_start < self.apc_prefix_tokens:
                raise ValueError("segments must start after the APC prefix")
            if segment.target_start < previous_end:
                raise ValueError("segments cannot overlap")
            if segment.target_end > self.allocation_end:
                raise ValueError("segment exceeds allocation span")
            total += segment.length
            previous_end = segment.target_end
        if total != self.hit_tokens:
            raise ValueError(f"hit_tokens={self.hit_tokens} does not match segments={total}")
        if ordered != self.segments:
            raise ValueError("segments must be sorted")

    @classmethod
    def from_segments(
        cls,
        apc_prefix_tokens: int,
        segments: Iterable[BlendSegment],
    ) -> BlendPlan:
        ordered = tuple(sorted(set(segments), key=lambda segment: segment.target_start))
        if not ordered:
            return cls(apc_prefix_tokens, apc_prefix_tokens, 0, ())
        return cls(
            apc_prefix_tokens=apc_prefix_tokens,
            allocation_end=max(segment.target_end for segment in ordered),
            hit_tokens=sum(segment.length for segment in ordered),
            segments=ordered,
        )

    @property
    def allocation_tokens(self) -> int:
        return self.allocation_end - self.apc_prefix_tokens

    @property
    def gap_tokens(self) -> int:
        return self.allocation_tokens - self.hit_tokens

    def passes_gate(
        self,
        min_retrieve_tokens: int,
        min_hit_ratio: float,
        min_saved_tokens: int,
    ) -> bool:
        if self.hit_tokens < min_retrieve_tokens:
            return False
        if self.hit_tokens < min_saved_tokens:
            return False
        return self.hit_tokens / max(self.allocation_tokens, 1) >= min_hit_ratio

    def to_dict(self) -> dict[str, Any]:
        return {
            "apc_prefix_tokens": self.apc_prefix_tokens,
            "allocation_end": self.allocation_end,
            "hit_tokens": self.hit_tokens,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BlendPlan:
        return cls(
            apc_prefix_tokens=int(value["apc_prefix_tokens"]),
            allocation_end=int(value["allocation_end"]),
            hit_tokens=int(value["hit_tokens"]),
            segments=tuple(
                BlendSegment.from_dict(segment) for segment in value.get("segments", [])
            ),
        )
