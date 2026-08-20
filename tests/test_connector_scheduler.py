import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from cacheblend_vllm.config import CacheBlendConfig
from cacheblend_vllm.connector import CacheBlendConnectorV1, RequestMetadata
from cacheblend_vllm.hashing import content_hash
from cacheblend_vllm.trace import PipelineTracer
from cacheblend_vllm.types import BlendPlan, SegmentId


def make_connector():
    connector = object.__new__(CacheBlendConnectorV1)
    connector.config = CacheBlendConfig.from_extra_config(
        {"chunk_size": 4, "strict_version_check": False}
    )
    connector.block_size = 2
    connector._plans = {}
    connector._load_requests = {}
    connector._request_objects = {}
    connector._scheduler_requests = {}
    connector._lookup = None
    connector._save_segments = {}
    return connector


class CountingStore:
    def __init__(self):
        self.calls = []

    def begin_put(self, model_scope, source_start, tokens, num_layers):
        token_tuple = tuple(tokens)
        self.calls.append((model_scope, source_start, token_tuple, num_layers))
        return SegmentId(model_scope, content_hash(token_tuple), source_start, len(token_tuple))


def output(new=(), cached=None, scheduled=None, preempted=None):
    if cached is None:
        cached = SimpleNamespace(
            req_ids=[],
            resumed_req_ids=set(),
            new_block_ids=[],
            num_computed_tokens=[],
        )
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=scheduled or {},
        finished_req_ids=set(),
        preempted_req_ids=preempted,
    )


def test_metadata_tracks_full_blocks_across_chunked_prefill():
    connector = make_connector()
    request = SimpleNamespace(
        request_id="r",
        prompt_token_ids=list(range(10)),
        _all_token_ids=list(range(10)),
        lora_request=None,
        mm_features=[],
        prompt_embeds=None,
    )
    connector._request_objects["r"] = request
    connector._plans["r"] = BlendPlan(0, 0, 0, ())

    first = SimpleNamespace(
        req_id="r",
        prompt_token_ids=request.prompt_token_ids,
        block_ids=([3, 5, 7],),
        num_computed_tokens=4,
    )
    metadata = connector.build_connector_meta(output([first], scheduled={"r": 2}))
    assert metadata.requests[0].save_end == 6
    assert metadata.requests[0].slot_mapping.tolist() == [6, 7, 10, 11, 14, 15]

    cached = SimpleNamespace(
        req_ids=["r"],
        resumed_req_ids=set(),
        new_block_ids=[([9],)],
        num_computed_tokens=[6],
    )
    metadata = connector.build_connector_meta(output(cached=cached, scheduled={"r": 2}))
    assert metadata.requests[0].plan is None
    assert metadata.requests[0].save_end == 8
    assert metadata.requests[0].slot_mapping.tolist() == [6, 7, 10, 11, 14, 15, 18, 19]


def test_non_token_identified_request_is_not_saved():
    connector = make_connector()
    request = SimpleNamespace(
        request_id="mm",
        prompt_token_ids=[1, 2, 3],
        _all_token_ids=[1, 2, 3],
        lora_request=None,
        mm_features=[object()],
        prompt_embeds=None,
    )
    connector._request_objects["mm"] = request
    new = SimpleNamespace(
        req_id="mm",
        prompt_token_ids=request.prompt_token_ids,
        block_ids=([1, 2],),
        num_computed_tokens=0,
    )
    metadata = connector.build_connector_meta(output([new], scheduled={"mm": 3}))
    assert not metadata.requests[0].save
    assert metadata.requests[0].save_end == 3


def test_save_segment_hashes_are_built_once_per_request():
    connector = make_connector()
    connector._store = CountingStore()
    connector._kv_caches = [object(), object()]
    connector.model_scope = "model"
    request = RequestMetadata(
        request_id="r",
        token_ids=list(range(12)),
        slot_mapping=torch.arange(12),
        plan=None,
        save=True,
        save_end=12,
    )

    first = connector._request_save_segments(request, 8)
    assert [start for start, _ in first] == [0, 4]
    assert len(connector._store.calls) == 2

    grown = connector._request_save_segments(request, 12)
    assert [start for start, _ in grown] == [0, 4, 8]
    assert len(connector._store.calls) == 3

    assert connector._request_save_segments(request, 12) == grown
    assert len(connector._store.calls) == 3


def test_scheduler_lookup_is_submitted_and_polled_without_blocking(monkeypatch):
    connector = make_connector()
    connector._role = KVConnectorRole.SCHEDULER
    connector._lookup_pool = ThreadPoolExecutor(max_workers=1)
    connector._lookup_futures = {}
    connector._lookup_state_lock = threading.Lock()
    connector._cancelled_lookup_ids = set()
    connector._trace = PipelineTracer("test")
    entered = threading.Event()
    release = threading.Event()

    def lookup_plan(_request_id, _tokens, apc_prefix_tokens):
        entered.set()
        assert release.wait(timeout=5)
        return BlendPlan(apc_prefix_tokens, apc_prefix_tokens, 0, ())

    monkeypatch.setattr(connector, "_lookup_plan", lookup_plan)
    request = SimpleNamespace(
        request_id="async",
        prompt_token_ids=list(range(12)),
        lora_request=None,
        mm_features=[],
        prompt_embeds=None,
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    assert entered.wait(timeout=2)
    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)

    release.set()
    deadline = time.monotonic() + 2
    result = (None, True)
    while result[1] and time.monotonic() < deadline:
        result = connector.get_num_new_matched_tokens(request, 0)
        time.sleep(0.001)
    assert result == (0, False)

    connector._lookup_pool.shutdown(wait=True)
    connector._lookup_pool = None
