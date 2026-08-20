"""Opt-in JSONL tracing for the CacheBlend data pipeline.

Tracing is disabled unless ``CACHEBLEND_TRACE_DIR`` is set.  The trace records
host-side scheduling/enqueue events; it intentionally does not synchronize NPU
streams just to obtain timings.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class PipelineTracer:
    """Write one line-buffered trace per process with negligible disabled cost."""

    def __init__(self, component: str):
        self.component = component
        self._lock = threading.Lock()
        self._file = None
        root = os.getenv("CACHEBLEND_TRACE_DIR")
        if not root:
            return
        trace_root = Path(root)
        trace_root.mkdir(parents=True, exist_ok=True)
        path = trace_root / f"{component}-pid-{os.getpid()}.jsonl"
        self._file = path.open("a", encoding="utf-8", buffering=1)
        self.emit("trace_started")

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def emit(self, event: str, request_id: str | None = None, **fields: Any) -> None:
        output = self._file
        if output is None:
            return
        payload: dict[str, Any] = {
            "timestamp_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "component": self.component,
            "event": event,
        }
        if request_id is not None:
            payload["request_id"] = request_id
        payload.update(fields)
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self._file is not None:
                self._file.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            output = self._file
            self._file = None
        if output is not None:
            output.close()
