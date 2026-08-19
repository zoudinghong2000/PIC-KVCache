"""Process-local registry joining vLLM model construction to the connector."""

from __future__ import annotations

import threading
import weakref
from typing import Any

_MODELS: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
_LOCK = threading.RLock()


def register_model(engine_id: str, model: Any) -> None:
    if not engine_id:
        raise ValueError("engine_id cannot be empty")
    with _LOCK:
        existing = _MODELS.get(engine_id)
        if existing is not None and existing is not model:
            raise RuntimeError(f"a different model is already registered for engine {engine_id}")
        _MODELS[engine_id] = model


def get_model(engine_id: str) -> Any:
    with _LOCK:
        model = _MODELS.get(engine_id)
    if model is None:
        raise RuntimeError(
            "CacheBlend model was not registered before connector initialization. "
            "Ensure the cacheblend_models vLLM general plugin is loaded."
        )
    return model


def unregister_model(engine_id: str) -> None:
    with _LOCK:
        _MODELS.pop(engine_id, None)


def clear_for_test() -> None:
    with _LOCK:
        _MODELS.clear()
