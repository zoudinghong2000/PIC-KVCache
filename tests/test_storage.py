import torch

from cacheblend_vllm.storage import LocalPinnedCPUStore


def put(store, scope, source, token_base, value):
    tokens = list(range(token_base, token_base + 4))
    segment_id = store.begin_put(scope, source, tokens, num_layers=2)
    assert not store.put_layer(segment_id, 0, torch.full((2, 4, 2), value))
    assert store.match(scope, tokens, 0) == ()
    assert store.put_layer(segment_id, 1, torch.full((2, 4, 2), value + 1))
    return segment_id, tokens


def test_partial_entries_are_not_visible_and_committed_entries_are_pinnable():
    store = LocalPinnedCPUStore(4, max_bytes=4096, pin_memory=False)
    segment_id, tokens = put(store, "m", 0, 10, 1)
    segment = store.match("m", tokens, 0)[0]
    assert segment.segment_id == segment_id
    assert store.pin("r", [segment]) == (segment,)
    assert torch.equal(store.get_layer("r", segment, 1), torch.full((2, 4, 2), 2))
    store.release("r")


def test_lru_does_not_evict_pinned_entries():
    bytes_per_entry = 2 * 2 * 4 * 2 * 4  # two float32 tensors shaped [2,4,2]
    store = LocalPinnedCPUStore(4, max_bytes=bytes_per_entry, pin_memory=False)
    _, tokens1 = put(store, "m", 0, 10, 1.0)
    segment1 = store.match("m", tokens1, 0)[0]
    store.pin("r", [segment1])
    _, tokens2 = put(store, "m", 4, 20, 2.0)
    assert store.match("m", tokens1, 0) == (segment1,)
    assert store.match("m", tokens2, 0) == ()
    store.release("r")
    _, tokens3 = put(store, "m", 8, 30, 3.0)
    assert store.current_bytes <= store.max_bytes
    assert store.match("m", tokens1, 0) == ()
    assert store.match("m", tokens3, 0)


def test_async_layer_reservation_is_unique_and_publishes_owned_host_tensor():
    store = LocalPinnedCPUStore(4, max_bytes=4096, pin_memory=False)
    segment_id = store.begin_put("m", 0, [1, 2, 3, 4], num_layers=1)
    assert store.reserve_layer(segment_id, 0)
    assert not store.reserve_layer(segment_id, 0)
    parent = torch.ones(2, 8, 2)
    host = parent[:, 2:6]
    assert not host.is_contiguous()
    assert store.put_layer_host(segment_id, 0, host)
    segment = store.match("m", [1, 2, 3, 4], 0)[0]
    store.pin("r", [segment])
    assert store.get_layer("r", segment, 0) is host
