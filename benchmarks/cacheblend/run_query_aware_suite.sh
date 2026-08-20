#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/benchmark-results/query-aware-$(date -u +%Y%m%dT%H%M%SZ)}"
QUERY_AWARE_MODES="${QUERY_AWARE_MODES:-kv_deviation sparse_q compare}"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

COMMON_ENV=(QUALITY=1 TRACE=1)
ANALYZE_ARGS=()

env "${COMMON_ENV[@]}" \
  BENCH_ARM=no_cache \
  RUN_DIR="$RUN_ROOT/full_recompute" \
  bash benchmarks/cacheblend/run_arm.sh
ANALYZE_ARGS+=(--run "full_recompute=$RUN_ROOT/full_recompute")

for mode in $QUERY_AWARE_MODES; do
  if [[ "$mode" == kv_deviation ]]; then
    check_layer="${KV_DEVIATION_CHECK_LAYER:-1}"
  else
    check_layer="${SPARSE_Q_CHECK_LAYER:-6}"
  fi
  env "${COMMON_ENV[@]}" \
    BENCH_ARM=cacheblend \
    SELECTION_STRATEGY="$mode" \
    CHECK_LAYER="$check_layer" \
    RUN_DIR="$RUN_ROOT/$mode" \
    bash benchmarks/cacheblend/run_arm.sh
  ANALYZE_ARGS+=(--run "$mode=$RUN_ROOT/$mode")
done

python -m benchmarks.cacheblend.analyze \
  "${ANALYZE_ARGS[@]}" \
  --output "$RUN_ROOT/report.md"
echo "query-aware report: $RUN_ROOT/report.md"
