#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Benchmark wrapper: run examples on hardware, then parse per-round latency
# emitted by the test framework directly (see simpler_setup/scene_test.py::
# _log_round_timings). Worker.run now returns a RunTiming struct so we no
# longer scrape device logs for sched/orch timestamps.
#
# Usage:
#   ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>]
#
# Edit the EXAMPLE_CASES map below to control which examples and cases to run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Examples to benchmark and their case lists, per runtime.
# ---------------------------------------------------------------------------

declare -A TMR_EXAMPLE_CASES=(
    [alternating_matmul_add]="Case1"
    [benchmark_bgemm]="Case0"
    [paged_attention_unroll]="Case1,Case2"
    [paged_attention_unroll_manual_scope]="Case1,Case2"
    [batch_paged_attention]="Case1"
    [spmd_paged_attention]="Case1,Case2"
)
TMR_EXAMPLE_ORDER=(
    alternating_matmul_add
    benchmark_bgemm
    paged_attention_unroll
    paged_attention_unroll_manual_scope
    batch_paged_attention
    spmd_paged_attention
)

DEVICE_ID=0
ROUNDS=100
PLATFORM=a2a3
RUNTIME=tensormap_and_ringbuffer
VERBOSE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform) PLATFORM="$2"; shift 2 ;;
        -d|--device)   DEVICE_ID="$2"; shift 2 ;;
        -n|--rounds)   ROUNDS="$2"; shift 2 ;;
        -r|--runtime)  RUNTIME="$2"; shift 2 ;;
        -v|--verbose)  VERBOSE=1; shift ;;
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run examples and report per-round host/device wall
emitted directly by Worker.run().

Options:
  -p, --platform  Platform (default: a2a3)
  -d, --device    Device ID (default: 0)
  -n, --rounds    Rounds per example (default: 100)
  -r, --runtime   Runtime (default: tensormap_and_ringbuffer)
  -v, --verbose   Save test output to a timestamped log
  -h, --help      Show this help

Device-wall numbers are zero unless the runtime was built with
PTO2_PROFILING=ON AND the run is invoked with --enable-l2-swimlane.
Pass --enable-l2-swimlane to populate the device column.
USAGE
            exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

VERBOSE_LOG=""
if [[ $VERBOSE -eq 1 ]]; then
    mkdir -p "$PROJECT_ROOT/outputs"
    VERBOSE_LOG="$PROJECT_ROOT/outputs/benchmark_$(date +%Y%m%d_%H%M%S).log"
    echo "Verbose log: $VERBOSE_LOG"
fi

ARCH="${PLATFORM%%sim}"
EXAMPLES_DIRS=(
    "$PROJECT_ROOT/tests/st/${ARCH}/${RUNTIME}"
    "$PROJECT_ROOT/examples/${ARCH}/${RUNTIME}"
)

case "$RUNTIME" in
    tensormap_and_ringbuffer)
        declare -n EXAMPLE_CASES=TMR_EXAMPLE_CASES
        EXAMPLE_ORDER=("${TMR_EXAMPLE_ORDER[@]}")
        ;;
    *)
        echo "ERROR: unknown runtime '$RUNTIME'." >&2
        exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Parse "Avg Host: X.X us  |  Avg Device: Y.Y us  (N rounds)" emitted by
# _log_round_timings. Returns "host_avg device_avg" (device may be "-").
# ---------------------------------------------------------------------------
parse_avgs() {
    local output="$1"
    local host_avg device_avg
    host_avg=$(echo "$output" | grep -oE 'Avg Host: [0-9.]+' | tail -1 | awk '{print $3}')
    device_avg=$(echo "$output" | grep -oE 'Avg Device: [0-9.]+' | tail -1 | awk '{print $3}')
    [[ -z "$host_avg"   ]] && host_avg="-"
    [[ -z "$device_avg" ]] && device_avg="-"
    echo "$host_avg $device_avg"
}

run_bench() {
    local example="$1" example_dir="$2" case_name="${3:-}"
    [[ -n "$case_name" ]] && echo "  ---- $case_name ----"

    local test_file
    test_file=$(find "$example_dir" -maxdepth 1 -name 'test_*.py' -print -quit 2>/dev/null || true)
    if [[ -z "$test_file" ]]; then
        echo "  SKIPPED: no test_*.py found in $example_dir"
        return
    fi

    local run_cmd=(
        python3 "$test_file"
        --platform "$PLATFORM" --device "$DEVICE_ID"
        --rounds "$ROUNDS" --skip-golden
    )
    if [[ -n "$case_name" ]]; then
        run_cmd+=(--case "$case_name" --manual include)
    fi
    run_cmd+=("${EXTRA_ARGS[@]}")

    local output rc=0
    output=$("${run_cmd[@]}" 2>&1) || rc=$?
    if [[ -n "$VERBOSE_LOG" ]]; then
        echo "==== $example ${case_name} ====" >> "$VERBOSE_LOG"
        echo "$output" >> "$VERBOSE_LOG"
    fi
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED: exit code $rc"
        ((FAIL++)) || true
        return
    fi

    local avgs host_avg device_avg
    avgs=$(parse_avgs "$output")
    host_avg=$(echo "$avgs" | awk '{print $1}')
    device_avg=$(echo "$avgs" | awk '{print $2}')

    if [[ "$host_avg" == "-" ]]; then
        echo "  (no per-round timing — was --rounds > 1?)"
        ((FAIL++)) || true
        return
    fi

    echo "  Avg Host: $host_avg us  |  Avg Device: $device_avg us  ($ROUNDS rounds)"
    ((PASS++)) || true

    local label="$example"
    [[ -n "$case_name" ]] && label="$example ($case_name)"
    SUMMARY_NAMES+=("$label")
    SUMMARY_HOST+=("$host_avg")
    SUMMARY_DEVICE+=("$device_avg")
}

PASS=0
FAIL=0
SUMMARY_NAMES=()
SUMMARY_HOST=()
SUMMARY_DEVICE=()

echo ""
echo "Runtime: $RUNTIME"

for example in "${EXAMPLE_ORDER[@]}"; do
    case_list="${EXAMPLE_CASES[$example]:-}"

    EXAMPLE_DIR=""
    for dir in "${EXAMPLES_DIRS[@]}"; do
        candidate="$dir/$example"
        if [[ -d "$candidate" ]] && ls "$candidate"/test_*.py 1>/dev/null 2>&1; then
            EXAMPLE_DIR="$candidate"; break
        fi
    done

    echo ""
    echo "================================================================"
    echo "  $example"
    echo "================================================================"

    if [[ -z "$EXAMPLE_DIR" ]]; then
        echo "  SKIP: not found"
        ((FAIL++)) || true
        continue
    fi

    if [[ -z "${case_list:-}" ]]; then
        run_bench "$example" "$EXAMPLE_DIR"
    else
        IFS=',' read -ra cases <<< "$case_list"
        for c in "${cases[@]}"; do
            run_bench "$example" "$EXAMPLE_DIR" "$c"
        done
    fi
done

if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    echo ""
    echo "================================================================"
    echo "  Performance Summary ($RUNTIME)"
    echo "================================================================"
    echo ""
    printf "  %-40s  %12s  %12s\n" "Example" "Host (us)" "Device (us)"
    printf "  %-40s  %12s  %12s\n" "----------------------------------------" "------------" "------------"
    for i in "${!SUMMARY_NAMES[@]}"; do
        printf "  %-40s  %12s  %12s\n" "${SUMMARY_NAMES[$i]}" "${SUMMARY_HOST[$i]}" "${SUMMARY_DEVICE[$i]}"
    done
fi

TOTAL=$((PASS + FAIL))
echo ""
echo "================================================================"
echo "  Benchmark complete ($RUNTIME): $PASS passed, $FAIL failed ($TOTAL total)"
echo "================================================================"

if [[ -n "$VERBOSE_LOG" ]]; then
    echo "  Verbose log saved to: $VERBOSE_LOG"
fi

[[ $FAIL -eq 0 ]]
