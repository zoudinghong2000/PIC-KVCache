"""Deterministic rolling hashes for arbitrary-offset token matching."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

MASK64 = (1 << 64) - 1
POLY_BASE = 1_000_003


def _token_value(token: int) -> int:
    return (int(token) + 1) & MASK64


def content_hash(tokens: Sequence[int]) -> int:
    value = 0
    for token in tokens:
        value = ((value * POLY_BASE) + _token_value(token)) & MASK64
    return value


def rolling_hashes(tokens: Sequence[int], window: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, hash)`` for every full window in ``tokens``."""
    if window <= 0:
        raise ValueError("window must be positive")
    if len(tokens) < window:
        return
    current = content_hash(tokens[:window])
    yield 0, current
    leading_power = pow(POLY_BASE, window - 1, 1 << 64)
    for start in range(1, len(tokens) - window + 1):
        outgoing = (_token_value(tokens[start - 1]) * leading_power) & MASK64
        current = (current - outgoing) & MASK64
        current = ((current * POLY_BASE) + _token_value(tokens[start + window - 1])) & MASK64
        yield start, current
