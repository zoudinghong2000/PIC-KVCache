#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

MODEL="${MODEL:-/home/zdh/qwen3-30b-a3b}"
MODEL_NAME="${MODEL_NAME:-qwen3-coder-30b-a3b}"
VLLM_PORT="${VLLM_PORT:-8123}"
VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-2,3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
LOCAL_CPU_GB="${LOCAL_CPU_GB:-1}"
REQUESTS_FILE="${REQUESTS_FILE:-}"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-/home/zdh/mcts_kv_replay/replay_tree.py}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1800}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1200}"
START_TURN="${START_TURN:-}"
END_TURN="${END_TURN:-}"

[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 2; }
if [[ -n "$REQUESTS_FILE" ]]; then
  [[ -f "$REQUESTS_FILE" ]] || { echo "Missing requests: $REQUESTS_FILE" >&2; exit 2; }
  [[ -f "$REPLAY_SCRIPT" ]] || { echo "Missing replay script: $REPLAY_SCRIPT" >&2; exit 2; }
fi

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
unset LMCACHE_CONFIG_FILE LMCACHE_LOG_LEVEL
export PYTHONPATH=""
source /usr/local/Ascend/cann-8.5.1/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend
export VLLM_USE_V1=1
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,cacheblend_models
export PYTHONHASHSEED=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

SMOKE_DIR="$(mktemp -d /tmp/cacheblend-plugin-smoke.XXXXXX)"
SERVER_LOG="$SMOKE_DIR/server.log"
VLLM_PID=""

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
  echo "smoke artifacts: $SMOKE_DIR"
}
trap cleanup EXIT INT TERM

KV_CONFIG="{\"kv_connector\":\"CacheBlendConnectorV1\",\"kv_connector_module_path\":\"cacheblend_vllm.connector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"chunk_size\":256,\"local_cpu_gb\":${LOCAL_CPU_GB},\"min_retrieve_tokens\":256,\"min_hit_ratio\":0.10,\"check_layers\":[1],\"recompute_ratios\":[0.15],\"async_prefetch\":true,\"async_fingerprint\":true,\"tp_global_selection\":true,\"event_pipeline\":true,\"fused_segment_copy\":true,\"cache_attention_mask\":true}}"

setsid vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --served-model-name "$MODEL_NAME" \
  --tensor-parallel-size 2 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --seed 0 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --language-model-only \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true}' \
  --kv-transfer-config "$KV_CONFIG" >"$SERVER_LOG" 2>&1 &
VLLM_PID=$!
echo "started pid=$VLLM_PID devices=$VISIBLE_DEVICES log=$SERVER_LOG"

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

if [[ -n "$REQUESTS_FILE" ]]; then
  RECORDS="$SMOKE_DIR/records.jsonl"
  REPLAY_RANGE_ARGS=()
  [[ -z "$START_TURN" ]] || REPLAY_RANGE_ARGS+=(--start-turn "$START_TURN")
  [[ -z "$END_TURN" ]] || REPLAY_RANGE_ARGS+=(--end-turn "$END_TURN")
  python "$REPLAY_SCRIPT" \
    --requests "$REQUESTS_FILE" \
    --output "$RECORDS" \
    --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
    --metrics-url "http://127.0.0.1:${VLLM_PORT}/metrics" \
    --model "$MODEL_NAME" \
    --timeout "$REQUEST_TIMEOUT_SECONDS" \
    --settle-seconds 0.05 \
    --max-tokens 1 \
    "${REPLAY_RANGE_ARGS[@]}"
  python - "$RECORDS" "$SMOKE_DIR/summary.json" <<'PY'
import json
import statistics
import sys
from pathlib import Path

records_path, summary_path = sys.argv[1:]
rows = [json.loads(line) for line in Path(records_path).read_text().splitlines() if line]
ttfts = [float(row["ttft_seconds"]) for row in rows if "error" not in row]
metrics = {}
for row in rows:
    for name, value in row.get("metrics_delta", {}).items():
        metrics[name] = metrics.get(name, 0.0) + value
summary = {
    "request_count": len(rows),
    "error_count": sum("error" in row for row in rows),
    "prompt_tokens": sum(row.get("usage", {}).get("prompt_tokens", 0) for row in rows),
    "total_ttft_seconds": sum(ttfts),
    "median_ttft_seconds": statistics.median(ttfts) if ttfts else None,
    "total_request_seconds": sum(row.get("total_seconds", 0.0) for row in rows),
    "metrics_delta": dict(sorted(metrics.items())),
}
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
else
  python - "$VLLM_PORT" "$MODEL_NAME" <<'PY'
import json
import sys
import time
import urllib.request

port, model = sys.argv[1:]
shared = " ".join(f"cacheblend_token_{index:04d}" for index in range(400))
first = "First source document. " + shared + " End of source."
second = "A completely different request prefix. " + first

def complete(prompt):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    return time.perf_counter() - started, payload

for label, prompt in (("populate", first), ("reuse", second)):
    elapsed, payload = complete(prompt)
    usage = payload.get("usage", {})
    text = payload["choices"][0].get("text", "")
    print(label, "seconds=", round(elapsed, 4), "usage=", usage, "text=", repr(text))

with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as response:
    metrics = response.read().decode()
for line in metrics.splitlines():
    if "external_prefix_cache" in line and not line.startswith("#"):
        print("metric", line)
PY
fi

echo "CacheBlend log lines:"
rg -n "CacheBlend|cacheblend" "$SERVER_LOG" | tail -n 80 || true
