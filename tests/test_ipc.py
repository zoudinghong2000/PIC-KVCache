from pathlib import Path

import torch

from cacheblend_vllm.ipc import LookupServer, TensorParallelLookup
from cacheblend_vllm.storage import LocalPinnedCPUStore


def populate(store, source, tokens, rank):
    segment_id = store.begin_put("m", source, tokens, num_layers=1)
    store.put_layer(segment_id, 0, torch.full((2, len(tokens), 1), rank))


def test_tp_lookup_uses_exact_intersection_and_pins(tmp_path: Path):
    stores = [LocalPinnedCPUStore(4, 4096, pin_memory=False) for _ in range(2)]
    for rank, store in enumerate(stores):
        populate(store, 0, [1, 2, 3, 4], rank)
    populate(stores[0], 4, [5, 6, 7, 8], 0)

    uris = [f"inproc://cacheblend-test-{id(stores)}-{rank}" for rank in range(2)]
    servers = [
        LookupServer(str(tmp_path), "engine", rank, store, uri=uris[rank])
        for rank, store in enumerate(stores)
    ]
    for server in servers:
        server.start()
    try:
        lookup = TensorParallelLookup(uris)
        segments = lookup.lookup_and_prefetch("request", "m", [0, 1, 2, 3, 4, 5, 6, 7, 8], 1)
        assert [(value.target_start, value.source_start) for value in segments] == [(1, 0)]
        for store in stores:
            assert store.get_layer("request", segments[0], 0) is not None
        lookup.cancel("request")
    finally:
        for server in servers:
            server.close()
