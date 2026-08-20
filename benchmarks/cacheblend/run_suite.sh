#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ARMS="${ARMS:-no_cache apc cacheblend}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/benchmark-results/$(date -u +%Y%m%dT%H%M%SZ)}"
TRACE_CACHEBLEND="${TRACE_CACHEBLEND:-0}"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

ANALYZE_ARGS=()
for arm in $ARMS; do
  trace=0
  if [[ "$arm" == cacheblend ]]; then
    trace="$TRACE_CACHEBLEND"
  fi
  BENCH_ARM="$arm" RUN_DIR="$RUN_ROOT/$arm" TRACE="$trace" \
    bash benchmarks/cacheblend/run_arm.sh
  ANALYZE_ARGS+=(--run "$arm=$RUN_ROOT/$arm")
done

python -m benchmarks.cacheblend.analyze \
  "${ANALYZE_ARGS[@]}" \
  --output "$RUN_ROOT/report.md"
echo "suite report: $RUN_ROOT/report.md"
