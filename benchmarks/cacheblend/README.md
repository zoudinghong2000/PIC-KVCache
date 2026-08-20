# CacheBlend benchmark

This benchmark creates deterministic token-level documents and measures three
server configurations against exactly the same workload:

| Arm | Native APC | CacheBlend connector | Purpose |
|---|---:|---:|---|
| `no_cache` | off | off | Full-prefill baseline |
| `apc` | on | off | vLLM contiguous-prefix baseline |
| `cacheblend` | on | on | Non-contiguous document reuse |

It is intentionally separate from the 509-request acceptance replay in the
root README. This workload answers *why* a request is faster or slower, while
the replay checks compatibility with a real request stream.

## Workload phases

Every arm receives the same ordered phases:

1. `engine_warmup`: an equal-size, disjoint, unmeasured miss warms lazy kernels
   and the save pipeline without populating any measured document.
2. `populate`: each document is sent alone. The CacheBlend arm asynchronously
   writes complete KV chunks to its pinned-CPU store.
3. `blend`: already-populated documents are reordered and concatenated. Native
   APC cannot describe these interior hits, while CacheBlend can.
4. `apc_repeat`: exact repeats of blend prompts show the native APC lower bound.
5. `cold`: fresh documents with the same token count show the full-prefill
   baseline without restarting the server.

The default blend request contains four 4,096-token documents. Native APC can
usually consume the first populated document; the remaining three still exceed
the benchmark profile's `min_saved_tokens=8192` gate (the library default is
neutral). The complete prompt
remains below the default `MAX_MODEL_LEN=32768` after separators and the
question.

## Run the clean comparison

Install the package in the same environment as vLLM and vLLM-Ascend, then run:

```bash
cd /path/to/PIC-KVCache
python -m pip install --no-deps -e .

ASCEND_RT_VISIBLE_DEVICES=2,3 \
MODEL=/path/to/qwen3-30b-a3b \
bash benchmarks/cacheblend/run_suite.sh
```

The suite starts and stops one server per arm. Useful environment variables
include:

| Variable | Default | Meaning |
|---|---|---|
| `ARMS` | `no_cache apc cacheblend` | Space-separated arms to run |
| `RUN_ROOT` | timestamped directory | Result directory |
| `TP_SIZE` | `2` | Tensor-parallel size |
| `MAX_MODEL_LEN` | `32768` | vLLM maximum context |
| `MAX_NUM_SEQS` | `1` | Benchmark concurrency; raise it for serving tests |
| `LOCAL_CPU_GB` | `16` | CacheBlend CPU cache per TP rank |
| `MIN_SAVED_TOKENS` | `8192` | Benchmark-only profitability gate |
| `MAX_APC_PREFIX_TO_HIT_RATIO` | `8.0` | Benchmark-only APC/hit gate |
| `STORE_WORKERS` | `1` | Host publication worker count |
| `MAX_INFLIGHT_STORE_BATCHES` | `8` | Store queue backpressure limit |
| `NUM_DOCUMENTS` | `4` | Warm document corpus size |
| `DOCUMENT_TOKENS` | `4096` | Exact tokens per generated document |
| `DOCUMENTS_PER_QUERY` | `4` | Documents in each blend/cold prompt |
| `BLEND_REQUESTS` | `4` | Non-contiguous reuse requests |
| `APC_REPEATS` | `2` | Exact prompt repeats |
| `COLD_REQUESTS` | `2` | Equal-size miss requests |

Keep `BLEND_REQUESTS <= NUM_DOCUMENTS`. Each measured blend then starts with a
different populated document, preventing native APC from reusing a
multi-document prefix produced by an earlier blend request. Increase both
values together when more samples are needed.

For a dense model, set `ENABLE_EXPERT_PARALLEL=0`. If the Ascend environment
is not already configured, pass `CANN_ENV=/path/to/set_env.sh`. Other standard
vLLM/Ascend environment variables are inherited.

One arm can be run independently:

```bash
BENCH_ARM=cacheblend \
RUN_DIR=/tmp/cacheblend-clean \
MODEL=/path/to/model \
bash benchmarks/cacheblend/run_arm.sh
```

## Trace the data pipeline

Tracing is opt-in because JSONL writes add host overhead. Use a separate run
for understanding scheduling and enqueue order:

```bash
ARMS=cacheblend \
TRACE_CACHEBLEND=1 \
RUN_ROOT=/tmp/cacheblend-trace \
MODEL=/path/to/model \
bash benchmarks/cacheblend/run_suite.sh
```

Most trace durations are host-side function/enqueue times and are not kernel
execution times. `save_store_wait_finished` is the exception: it includes the
synchronized single-Store completion boundary. Use `msprof` or a
PyTorch/torch-npu profiler after the host trace identifies a suspicious layer
or transfer stage.

## Result files

Each arm contains:

- `manifest.json`: workload parameters, document token counts/digests, and
  deterministic blend orders;
- `records.jsonl`: request-level TTFT, latency, usage, response ID, and metric
  deltas;
- `summary.json`: phase aggregates and cold/blend speedup;
- `server.log` and `client.log`: complete execution logs;
- `report.md`: the arm summary;
- `traces/*.jsonl`: optional raw scheduler and worker events;
- `pipeline_timeline.jsonl`: optional events grouped by vLLM request ID.

The analyzer joins each server-side request ID back to its client phase and
name. TP worker events therefore appear once per rank; this is expected and is
useful for spotting rank skew.

The suite root also contains `report.md`, which compares all selected arms.
Regenerate a report without rerunning inference with:

```bash
python -m benchmarks.cacheblend.analyze \
  --run no_cache=benchmark-results/RUN/no_cache \
  --run apc=benchmark-results/RUN/apc \
  --run cacheblend=benchmark-results/RUN/cacheblend \
  --output benchmark-results/RUN/report.md
```

See [the pipeline walkthrough](../../docs/BENCHMARK_WALKTHROUGH.md) for the
event-to-code map and an optimization checklist.
