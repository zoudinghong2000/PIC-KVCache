"""Deterministic rolling hashes for arbitrary-offset token matching."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import lru_cache

import numpy as np

MASK64 = (1 << 64) - 1
POLY_BASE = 1_000_003


def _token_value(token: int) -> int:
    return (int(token) + 1) & MASK64


def content_hash(tokens: Sequence[int]) -> int:
    value = 0
    for token in tokens:
        value = ((value * POLY_BASE) + _token_value(token)) & MASK64
    return value


@lru_cache(maxsize=16)
def _window_powers(window: int) -> np.ndarray:
    modulus = 1 << 64
    return np.fromiter(
        (pow(POLY_BASE, exponent, modulus) for exponent in range(window - 1, -1, -1)),
        dtype=np.uint64,
        count=window,
    )


def rolling_hashes(tokens: Sequence[int], window: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, hash)`` for every full window in ``tokens``."""
    hashes = rolling_hash_values(tokens, window)
    for start, value in enumerate(hashes):
        yield start, int(value)


def rolling_hash_values(tokens: Sequence[int], window: int) -> np.ndarray:
    """Return every full-window hash as a contiguous uint64 array."""
    if window <= 0:
        raise ValueError("window must be positive")
    if len(tokens) < window:
        return np.empty(0, dtype=np.uint64)
    # NumPy's uint64 matrix product intentionally wraps modulo 2**64, exactly
    # matching content_hash.  A sliding-window view is zero-copy and moves the
    # O(tokens * window) arithmetic out of the scheduler's Python loop.
    values = np.asarray(tokens, dtype=np.uint64) + np.uint64(1)
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    return windows @ _window_powers(window)
