from types import SimpleNamespace

from cacheblend_vllm.config import CacheBlendConfig
from cacheblend_vllm.connector import CacheBlendConnectorV1
from cacheblend_vllm.types import BlendPlan


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
    connector._lookup_futures = {}
    connector._lookup = None
    return connector


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
