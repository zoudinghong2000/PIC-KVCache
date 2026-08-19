# CacheBlend vLLM plugin

This repository is an out-of-tree CacheBlend V1 implementation for vLLM 0.18
and vLLM-Ascend 0.18. It owns the token-range index, per-rank pinned-CPU KV
store, scheduler/worker RPC, Qwen3 side executor, and Ascend transfer pipeline.
There is no runtime import or package dependency on LMCache or LMCache-Ascend.

The current target is deliberately narrow:

- one host, TP1 or TP2, and pipeline parallel size 1;
- Qwen3 dense and Qwen3-MoE on Ascend;
- native vLLM APC first, followed by non-contiguous CacheBlend reuse;
- fixed-size prompt chunks in a per-TP-rank pinned-CPU LRU cache;
- token-only requests. LoRA, multimodal inputs, and prompt embeddings bypass
  CacheBlend so they cannot contaminate the base-model cache.

## Install

Use the same Python environment as vLLM and vLLM-Ascend:

```bash
python -m pip install --no-deps -e /home/zdh/cacheblend-vllm
```

The package installs one vLLM general-plugin entry point. If `VLLM_PLUGINS` is
unset, vLLM discovers both Ascend and CacheBlend normally. If an allow-list is
required, it must retain all Ascend entries:

```bash
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,cacheblend_models
```

## Run

```bash
vllm serve MODEL \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector":"CacheBlendConnectorV1",
    "kv_connector_module_path":"cacheblend_vllm.connector",
    "kv_role":"kv_both",
    "kv_load_failure_policy":"recompute",
    "kv_connector_extra_config":{
      "chunk_size":256,
      "local_cpu_gb":16,
      "min_retrieve_tokens":256,
      "min_hit_ratio":0.10,
      "check_layers":[1],
      "recompute_ratios":[0.15],
      "async_prefetch":true,
      "async_fingerprint":true,
      "tp_global_selection":true,
      "event_pipeline":true,
      "fused_segment_copy":true,
      "cache_attention_mask":true
    }
  }'
```

`kv_load_failure_policy=recompute` lets vLLM invalidate the connector-owned
blocks and retry locally if a layer load fails. Use `fail` while debugging when
the original exception is preferable.

The scheduler first accepts vLLM's contiguous APC prefix. Rank 0 then performs
an arbitrary-offset token-range lookup, every TP rank validates the exact same
segments, and all ranks pin the intersection. The Connector runs a Qwen3
layerwise side forward for the allocated span, forcing every gap token plus the
configured fraction of high-error cache hits to recompute. It scatters the
completed suffix into vLLM-owned pages before the regular one-token forward.

## Configuration

| Field | Default | Meaning |
|---|---:|---|
| `chunk_size` | `256` | Stored and matched token range size |
| `local_cpu_gb` | `16` | Per-rank pinned-CPU cache capacity |
| `min_retrieve_tokens` | `256` | Minimum exact hit tokens to activate blending |
| `min_hit_ratio` | `0.10` | Minimum hits / allocated blend span |
| `check_layers` | `[1]` | Layers that score cached K error |
| `recompute_ratios` | `[0.15]` | Cached-token fraction retained at each check layer |
| `save_decode_cache` | `false` | Store decode KV as well as complete prompt chunks |
| `ipc_root` | `/tmp/cacheblend-v1` | Local scheduler/worker Unix-socket root |
| `model_scope` | automatic | Optional explicit cache namespace |
| `strict_version_check` | `true` | Require the vLLM 0.18 package line |

Unknown fields fail at startup. The implementation also fails fast for
pipeline parallelism because its layerwise executor currently assumes all
decoder layers are local.

## Validate

```bash
cd /home/zdh/cacheblend-vllm
ruff check cacheblend_vllm tests
python -m pytest -q tests
python -m pip wheel --no-deps -w /tmp/cacheblend-wheel .
```

For an NPU acceptance run, compare APC and CacheBlend on the same prompt order,
TP layout, model, and cache warm-up. Check output token equality/top-logprob
agreement first, then TTFT and end-to-end latency. The handoff baseline uses
Qwen3-30B-A3B TP2 with the defaults above.

## Provenance

The design and portions of the algorithms are derived from the Apache-2.0
LMCache and LMCache-Ascend projects, specifically LMCache commit `da5c247b`
and LMCache-Ascend commit `903bab1`. See [NOTICE](NOTICE) and [LICENSE](LICENSE).
