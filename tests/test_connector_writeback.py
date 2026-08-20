import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import torch

from cacheblend_vllm.connector import CacheBlendConnectorV1
from cacheblend_vllm.trace import PipelineTracer


class RecordingStream:
    def __init__(self):
        self.events = []

    def wait_event(self, event):
        self.events.append(event)


class BlockingEvent:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def synchronize(self):
        self.entered.set()
        assert self.release.wait(timeout=5)


class RecordingEvent:
    def __init__(self):
        self.recorded_on = None

    def record(self, stream):
        self.recorded_on = stream


class RecordingStore:
    def __init__(self):
        self.layers = []
        self.cancelled = []

    def put_layer_host(self, segment_id, layer_id, tensor):
        self.layers.append((segment_id, layer_id, tensor))
        return True

    def cancel_layer(self, segment_id, layer_id):
        self.cancelled.append((segment_id, layer_id))


def test_wait_for_save_only_orders_paged_gather(monkeypatch):
    connector = object.__new__(CacheBlendConnectorV1)
    gather_event = object()
    current_stream = RecordingStream()
    connector._last_gather_event = gather_event
    connector._pending_offloads = []
    connector._trace = PipelineTracer("test")
    connector._save_lock = threading.Lock()
    connector._save_futures = set()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: current_stream)

    connector.wait_for_save()

    assert current_stream.events == [gather_event]
    assert connector._last_gather_event is None


def test_wait_for_save_defers_d2h_behind_end_of_forward_barrier(monkeypatch):
    connector = object.__new__(CacheBlendConnectorV1)
    gather_event = object()
    current_stream = RecordingStream()
    contiguous = torch.empty(1)
    pending = [("segment", 0, 1)]
    connector._last_gather_event = gather_event
    connector._pending_offloads = [("request", pending, contiguous, 123)]
    connector._trace = PipelineTracer("test")
    connector._save_lock = threading.Lock()
    connector._save_futures = set()
    enqueued = []

    monkeypatch.setattr(torch.npu, "current_stream", lambda: current_stream)
    monkeypatch.setattr(torch.npu, "Event", RecordingEvent)
    monkeypatch.setattr(
        connector,
        "_save_contiguous_batch",
        lambda batch, source, ready_event=None, **kwargs: enqueued.append(
            (batch, source, ready_event, kwargs)
        ),
    )

    connector.wait_for_save()

    assert current_stream.events == [gather_event]
    assert connector._pending_offloads == []
    assert enqueued[0][0] is pending
    assert enqueued[0][1] is contiguous
    assert enqueued[0][2].recorded_on is current_stream
    assert enqueued[0][3]["request_id"] == "request"


def test_async_publish_holds_device_staging_until_offload_completes():
    connector = object.__new__(CacheBlendConnectorV1)
    connector._store = RecordingStore()
    connector._trace = PipelineTracer("test")
    event = BlockingEvent()
    host = torch.arange(8).reshape(2, 4)
    device_staging = torch.empty(1)
    staging_ref = weakref.ref(device_staging)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            connector._publish_after_event,
            event,
            [("segment", 3, 4)],
            host,
            device_staging,
        )
        assert event.entered.wait(timeout=5)
        del device_staging
        gc.collect()
        assert staging_ref() is not None

        event.release.set()
        assert future.result(timeout=5)

    del future
    gc.collect()
    assert staging_ref() is None
    assert connector._store.cancelled == []
    assert connector._store.layers[0][:2] == ("segment", 3)
    assert torch.equal(connector._store.layers[0][2], host)
