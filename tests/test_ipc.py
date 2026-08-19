from pathlib import Path

import torch

from cacheblend_vllm.ipc import LookupServer, TensorParallelLookup
from cacheblend_vllm.storage import LocalPinnedCPUStore


def populate(store, source, tokens, rank):
    segment_id = store.begin_put("m", source, tokens, num_layers=1)
    store.put_layer(segment_id, 0, torch.full((2, len(tokens), 1), rank))


class RecordingStore(LocalPinnedCPUStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_calls = []

    def match(self, model_scope, tokens, start_offset):
        self.match_calls.append((len(tokens), start_offset))
        return super().match(model_scope, tokens, start_offset)


def test_tp_lookup_uses_exact_intersection_and_pins(tmp_path: Path):
    stores = [
        RecordingStore(4, 4096, pin_memory=False),
        LocalPinnedCPUStore(4, 4096, pin_memory=False),
    ]
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
    lookup = TensorParallelLookup(uris)
    try:
        sockets = tuple(client._socket for client in lookup.clients)
        segments = lookup.lookup_and_prefetch("request", "m", [0, 1, 2, 3, 4, 5, 6, 7, 8], 1)
        assert [(value.target_start, value.source_start) for value in segments] == [(1, 0)]
        assert stores[0].match_calls == [(8, 0)]
        for store in stores:
            assert store.get_layer("request", segments[0], 0) is not None
        lookup.cancel("request")
        assert tuple(client._socket for client in lookup.clients) == sockets
    finally:
        lookup.close()
        for server in servers:
            server.close()
