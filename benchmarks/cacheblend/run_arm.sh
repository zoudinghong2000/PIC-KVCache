#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BENCH_ARM="${BENCH_ARM:-cacheblend}"
MODEL="${MODEL:-/home/zdh/qwen3-30b-a3b}"
MODEL_NAME="${MODEL_NAME:-cacheblend-benchmark}"
TOKENIZER="${TOKENIZER:-$MODEL}"
VLLM_PORT="${VLLM_PORT:-8123}"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2,3}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
LOCAL_CPU_GB="${LOCAL_CPU_GB:-16}"
MIN_SAVED_TOKENS="${MIN_SAVED_TOKENS:-8192}"
MAX_APC_PREFIX_TO_HIT_RATIO="${MAX_APC_PREFIX_TO_HIT_RATIO:-8.0}"
LOOKUP_TIMEOUT_MS="${LOOKUP_TIMEOUT_MS:-10000}"
STORE_WORKERS="${STORE_WORKERS:-1}"
MAX_INFLIGHT_STORE_BATCHES="${MAX_INFLIGHT_STORE_BATCHES:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1200}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
TRACE="${TRACE:-0}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/benchmark-results/${BENCH_ARM}}"

case "$BENCH_ARM" in
  no_cache|apc|cacheblend) ;;
  *) echo "BENCH_ARM must be no_cache, apc, or cacheblend" >&2; exit 2 ;;
esac

mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
SERVER_LOG="$RUN_DIR/server.log"
CLIENT_LOG="$RUN_DIR/client.log"
VLLM_PID=""

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
unset LMCACHE_CONFIG_FILE LMCACHE_LOG_LEVEL
export PYTHONPATH=""

if [[ -n "${CANN_ENV:-}" ]]; then
  [[ -f "$CANN_ENV" ]] || { echo "Missing CANN_ENV: $CANN_ENV" >&2; exit 2; }
  # shellcheck disable=SC1090
  source "$CANN_ENV"
elif [[ -f /usr/local/Ascend/cann-8.5.1/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-8.5.1/set_env.sh
fi

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export VLLM_USE_V1=1
ASCEND_PLUGINS="${VLLM_ASCEND_PLUGINS:-ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling}"
if [[ "$BENCH_ARM" == cacheblend ]]; then
  export VLLM_PLUGINS="${VLLM_PLUGINS:-${ASCEND_PLUGINS},cacheblend_models}"
else
  export VLLM_PLUGINS="${VLLM_PLUGINS:-$ASCEND_PLUGINS}"
fi
export PYTHONHASHSEED=0
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

if [[ "$BENCH_ARM" == cacheblend && "$TRACE" == 1 ]]; then
  export CACHEBLEND_TRACE_DIR="$RUN_DIR/traces"
  mkdir -p "$CACHEBLEND_TRACE_DIR"
else
  unset CACHEBLEND_TRACE_DIR
fi

stop_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 45); do
    pgrep -s "$pid" >/dev/null 2>&1 || return 0
    sleep 1
  done
  pkill -TERM -s "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -s "$pid" >/dev/null 2>&1 || return 0
    sleep 1
  done
  pkill -KILL -s "$pid" 2>/dev/null || true
}

cleanup() {
  set +e
  stop_group "$VLLM_PID"
  echo "benchmark artifacts: $RUN_DIR"
}
trap cleanup EXIT INT TERM

KV_CONFIG="{\"kv_connector\":\"CacheBlendConnectorV1\",\"kv_connector_module_path\":\"cacheblend_vllm.connector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"chunk_size\":256,\"local_cpu_gb\":${LOCAL_CPU_GB},\"min_retrieve_tokens\":256,\"min_hit_ratio\":0.10,\"min_saved_tokens\":${MIN_SAVED_TOKENS},\"max_apc_prefix_to_hit_ratio\":${MAX_APC_PREFIX_TO_HIT_RATIO},\"check_layers\":[1],\"recompute_ratios\":[0.15],\"tp_global_selection\":true,\"event_pipeline\":true,\"fused_segment_copy\":true,\"cache_attention_mask\":true,\"lookup_timeout_ms\":${LOOKUP_TIMEOUT_MS},\"store_workers\":${STORE_WORKERS},\"max_inflight_store_batches\":${MAX_INFLIGHT_STORE_BATCHES}}}"

VLLM_ARGS=(
  serve "$MODEL"
  --host 127.0.0.1
  --port "$VLLM_PORT"
  --served-model-name "$MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --data-parallel-size 1
  --seed 0
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --trust-remote-code
  --language-model-only
  --enable-chunked-prefill
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true}'
)
if [[ "$ENABLE_EXPERT_PARALLEL" == 1 ]]; then
  VLLM_ARGS+=(--enable-expert-parallel)
fi
if [[ "$BENCH_ARM" == apc || "$BENCH_ARM" == cacheblend ]]; then
  VLLM_ARGS+=(--enable-prefix-caching)
fi
if [[ "$BENCH_ARM" == cacheblend ]]; then
  VLLM_ARGS+=(--kv-transfer-config "$KV_CONFIG")
fi

setsid vllm "${VLLM_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
VLLM_PID=$!
echo "started arm=$BENCH_ARM pid=$VLLM_PID devices=$VISIBLE_DEVICES log=$SERVER_LOG"

attempts=$((STARTUP_TIMEOUT_SECONDS / 5))
for _ in $(seq 1 "$attempts"); do
  if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited during startup" >&2
    tail -n 200 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null
echo "server healthy"

python -m benchmarks.cacheblend.benchmark \
  --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
  --metrics-url "http://127.0.0.1:${VLLM_PORT}/metrics" \
  --model "$MODEL_NAME" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$RUN_DIR" \
  --num-documents "${NUM_DOCUMENTS:-4}" \
  --document-tokens "${DOCUMENT_TOKENS:-4096}" \
  --documents-per-query "${DOCUMENTS_PER_QUERY:-4}" \
  --blend-requests "${BLEND_REQUESTS:-4}" \
  --apc-repeats "${APC_REPEATS:-2}" \
  --cold-requests "${COLD_REQUESTS:-2}" \
  --output-tokens "${OUTPUT_TOKENS:-1}" \
  --seed "${BENCHMARK_SEED:-0}" \
  --settle-seconds "${SETTLE_SECONDS:-0.50}" \
  --timeout "$REQUEST_TIMEOUT_SECONDS" 2>&1 | tee "$CLIENT_LOG"

python -m benchmarks.cacheblend.analyze \
  --run "$BENCH_ARM=$RUN_DIR" \
  --output "$RUN_DIR/report.md" >/dev/null
echo "report: $RUN_DIR/report.md"
