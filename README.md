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
python -m pip install --no-deps -e .
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
      "min_saved_tokens":8192,
      "max_apc_prefix_to_hit_ratio":8.0,
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
| `min_saved_tokens` | `8192` | Minimum exact hits needed to amortize the side forward |
| `max_apc_prefix_to_hit_ratio` | `8.0` | Skip when gathering the APC prefix dominates hits; `0` disables |
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

The smoke runner starts a real vLLM-Ascend server. With no request file it runs
a short synthetic reuse case. Supplying a replay file enables the full runner;
`REPLAY_SCRIPT` can point at a checkout other than the local default:

```bash
ASCEND_RT_VISIBLE_DEVICES=2,3 \
MODEL=/path/to/qwen3-30b-a3b \
MAX_MODEL_LEN=98304 \
MAX_NUM_BATCHED_TOKENS=8192 \
LOCAL_CPU_GB=16 \
REQUESTS_FILE=/path/to/requests.jsonl \
bash scripts/run_npu_smoke.sh
```

## Validated result

The standalone plugin was validated on 509 ordered requests using
Qwen3-30B-A3B, Ascend TP2/EP2, PP1, 98,304 maximum model length, and a 16 GiB
per-rank CPU cache. All 509 requests completed without an error. The records
reported a 1,755,107-token external CacheBlend allocation span and 10,909,568
native APC hit tokens. The external metric is the scheduler-owned continuous
span and can include recomputed gaps; it is not the exact-hit count.

| Arm | Total TTFT | Median | p90 | p99 | External span |
|---|---:|---:|---:|---:|---:|
| Standalone plugin | 320.827 s | 0.406 s | 1.588 s | 2.568 s | 1,755,107 |
| LMCache-based CacheBlend baseline | 307.462 s | 0.349 s | 1.512 s | 2.776 s | 1,858,313 |
| Native APC baseline | 337.026 s | 0.269 s | 2.009 s | 4.285 s | n/a |

These are single-run acceptance figures, not a general benchmark. Profitability
gating, direct segment-to-NPU staging, and suffix-only persistent TP lookup cut
the plugin's previous 378.334-second result by 15.2%. It is 4.8% faster than
native APC on this trace and remains 4.3% above the LMCache-based implementation.
The remaining aggregate gap is concentrated in lookup-miss writeback and its
main-stream dependency.

## Provenance

The design and portions of the algorithms are derived from the Apache-2.0
LMCache and LMCache-Ascend projects, specifically LMCache commit `da5c247b`
and LMCache-Ascend commit `903bab1`. See [NOTICE](NOTICE) and [LICENSE](LICENSE).
