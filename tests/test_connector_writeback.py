import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch

from cacheblend_vllm.connector import CacheBlendConnectorV1
from cacheblend_vllm.trace import PipelineTracer


class RecordingStream:
    def __init__(self):
        self.events = []
        self.streams = []
        self.synchronize_calls = 0

    def wait_event(self, event):
        self.events.append(event)

    def wait_stream(self, stream):
        self.streams.append(stream)

    def synchronize(self):
        self.synchronize_calls += 1


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
        self.synchronize_calls = 0

    def record(self, stream):
        self.recorded_on = stream

    def synchronize(self):
        self.synchronize_calls += 1


class RecordingStore:
    def __init__(self):
        self.layers = []
        self.cancelled = []

    def put_layer_host(self, segment_id, layer_id, tensor):
        self.layers.append((segment_id, layer_id, tensor))
        return True

    def cancel_layer(self, segment_id, layer_id):
        self.cancelled.append((segment_id, layer_id))


class RecordingTracer:
    enabled = True

    def __init__(self):
        self.events = []

    def emit(self, event, request_id=None, **fields):
        self.events.append((event, request_id, fields))


def test_wait_for_save_synchronizes_single_store_stream():
    connector = object.__new__(CacheBlendConnectorV1)
    store_stream = RecordingStream()
    connector._store_stream = store_stream
    connector._trace = PipelineTracer("test")
    connector._save_lock = threading.Lock()
    connector._save_futures = set()
    connector._pending_store_request_ids = set()

    connector.wait_for_save()

    assert store_stream.synchronize_calls == 1
    assert connector._save_futures == set()


def test_enqueue_store_gathers_and_offloads_on_one_stream(monkeypatch):
    connector = object.__new__(CacheBlendConnectorV1)
    connector._store = RecordingStore()
    connector._store_stream = RecordingStream()
    connector._trace = RecordingTracer()
    connector._save_lock = threading.Lock()
    connector._save_futures = set()
    connector._pending_store_request_ids = set()
    connector._save_slots = threading.Semaphore(2)
    connector._save_pool = ThreadPoolExecutor(max_workers=1)
    connector.block_size = 2
    serving_stream = RecordingStream()
    events = []
    original_empty = torch.empty

    def make_event():
        event = RecordingEvent()
        events.append(event)
        return event

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    gathered = torch.arange(8).reshape(2, 4)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: serving_stream)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext(stream))
    monkeypatch.setattr(torch.npu, "Event", make_event)
    monkeypatch.setattr(torch, "empty", unpinned_empty)
    monkeypatch.setattr(
        "cacheblend_vllm.connector.gather_paged_kv",
        lambda kv_layer, slots, block_size: gathered,
    )

    connector._enqueue_store_batch(
        [("segment", 0, 4)],
        object(),
        torch.arange(4),
        request_id="request",
    )
    connector.wait_for_save()
    connector._save_pool.shutdown()

    assert connector._store_stream.streams == [serving_stream]
    assert connector._store_stream.synchronize_calls == 1
    assert events[0].recorded_on is connector._store_stream
    assert events[0].synchronize_calls == 1
    assert torch.equal(connector._store.layers[0][2], gathered)
    assert any(
        event == "save_store_wait_finished" and request_id == "request"
        for event, request_id, _ in connector._trace.events
    )


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
