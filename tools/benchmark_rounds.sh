#!/usr/bin/env bash
# Benchmark wrapper: run examples on hardware,
# then parse device-log timing lines to report per-round latency.
#
# Usage:
#   ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>]
#
# Edit the EXAMPLE_CASES maps below to control which examples and cases to run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXAMPLE="$PROJECT_ROOT/examples/scripts/run_example.py"

# ---------------------------------------------------------------------------
# Examples to benchmark and their case lists, per runtime.
# Key   = directory name under tests/device_tests/<platform>/<runtime>/
# Value = comma-separated case names to run (empty string = run DEFAULT_CASE)
# ---------------------------------------------------------------------------

# --- tensormap_and_ringbuffer ---
declare -A TMR_EXAMPLE_CASES=(
    [alternating_matmul_add]=""
    [benchmark_bgemm]=""
    [paged_attention_unroll]="Case1,Case2"
    [batch_paged_attention]=""
    [paged_attention]=""
)
TMR_EXAMPLE_ORDER=(
    alternating_matmul_add
    benchmark_bgemm
    paged_attention_unroll
    batch_paged_attention
    paged_attention
)

# --- aicpu_build_graph ---
declare -A ABG_EXAMPLE_CASES=(
    [paged_attention_unroll]="Case1,Case2"
)
ABG_EXAMPLE_ORDER=(
    paged_attention_unroll
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DEVICE_ID=0
ROUNDS=10
PLATFORM=a2a3
RUNTIME=tensormap_and_ringbuffer
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_ID="$2"
            shift 2
            ;;
        -n|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -r|--runtime)
            RUNTIME="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run all examples and report per-round timing from device logs

Usage:
  ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>]

Options:
  -p, --platform Platform to run on (default: a2a3)
  -d, --device   Device ID (default: 0)
  -n, --rounds   Override number of rounds for each example (default: 10)
  -r, --runtime  Runtime to benchmark: tensormap_and_ringbuffer (default), aicpu_build_graph
  -h, --help     Show this help

All other options are passed through to run_example.py (e.g. --case).

Edit the EXAMPLE_CASES map at the top of this script to control which
examples and cases to benchmark.

Output:
  Average elapsed time in microseconds for each example.
USAGE
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Derive arch from platform and set examples directory
# ---------------------------------------------------------------------------
EXAMPLES_DIR="$PROJECT_ROOT/tests/device_tests/${PLATFORM}/${RUNTIME}"

# Select example cases and order based on runtime
case "$RUNTIME" in
    tensormap_and_ringbuffer)
        declare -n EXAMPLE_CASES=TMR_EXAMPLE_CASES
        EXAMPLE_ORDER=("${TMR_EXAMPLE_ORDER[@]}")
        ;;
    aicpu_build_graph)
        declare -n EXAMPLE_CASES=ABG_EXAMPLE_CASES
        EXAMPLE_ORDER=("${ABG_EXAMPLE_ORDER[@]}")
        ;;
    *)
        echo "ERROR: unknown runtime '$RUNTIME'. Use tensormap_and_ringbuffer or aicpu_build_graph."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Resolve device log directory (mirrors run_example.py / device_log_resolver.py)
# ---------------------------------------------------------------------------
if [[ -n "${ASCEND_WORK_PATH:-}" ]]; then
    LOG_ROOT="$ASCEND_WORK_PATH/log/debug"
    if [[ ! -d "$LOG_ROOT" ]]; then
        LOG_ROOT="$HOME/ascend/log/debug"
    fi
else
    LOG_ROOT="$HOME/ascend/log/debug"
fi
DEVICE_LOG_DIR="$LOG_ROOT/device-${DEVICE_ID}"

# ---------------------------------------------------------------------------
# parse_timing <log_file>
#   Grep for orch_start / end lines, compute per-round elapsed, print summary.
# ---------------------------------------------------------------------------
parse_timing() {
    local log_file="$1"

    local timing
    timing=$(grep -E 'Thread [0-9]+: (orch_start|orch_end|sched_end|orch_stage_end)=' "$log_file" || true)

    if [[ -z "$timing" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi

    echo "$timing" | awk '
    function flush_round() {
        if (round >= 0 && max_end > 0 && min_start > 0) {
            results[round] = (max_end - min_start) / 50.0
            count++
        }
    }
    BEGIN { round = 0; min_start = 0; max_end = 0; count = 0 }
    /orch_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in seen) {
            flush_round()
            round++
            min_start = 0
            max_end = 0
            delete seen
        }
        seen[tid] = 1
        match($0, /orch_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_start == 0 || val < min_start) min_start = val
    }
    /end=/ {
        match($0, /end=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_end) max_end = val
    }
    END {
        flush_round()
        if (count == 0) { print "  (no rounds parsed)"; exit 1 }

        printf "  %-8s  %12s\n", "Round", "Elapsed (us)"
        printf "  %-8s  %12s\n", "-----", "------------"
        sum_v = 0
        min_v = results[0]
        max_v = results[0]
        for (i = 0; i < count; i++) {
            printf "  %-8d  %12.1f\n", i, results[i]
            sum_v += results[i]
            if (results[i] < min_v) min_v = results[i]
            if (results[i] > max_v) max_v = results[i]
        }
        printf "\n  Avg: %.1f us  (%d rounds)\n", sum_v / count, count
        if (count > 2) {
            trimmed = (sum_v - min_v - max_v) / (count - 2)
            printf "  Trimmed Avg: %.1f us  (excluding min=%.1f, max=%.1f)\n", trimmed, min_v, max_v
        }
    }'
}

# ---------------------------------------------------------------------------
# wait_for_new_log <pre_run_logs_file>
#   Wait up to 15s for a new .log file in DEVICE_LOG_DIR. Prints the path.
# ---------------------------------------------------------------------------
wait_for_new_log() {
    local pre_file="$1"
    local new_log=""
    local deadline=$((SECONDS + 15))

    while [[ $SECONDS -lt $deadline ]]; do
        if [[ -d "$DEVICE_LOG_DIR" ]]; then
            new_log=$(comm -13 "$pre_file" <(ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort) 2>/dev/null | tail -1 || true)
            if [[ -n "$new_log" ]]; then
                echo "$new_log"
                return 0
            fi
        fi
        sleep 0.5
    done

    # Fallback: newest log
    if [[ -d "$DEVICE_LOG_DIR" ]]; then
        new_log=$(ls -t "$DEVICE_LOG_DIR"/*.log 2>/dev/null | head -1 || true)
        if [[ -n "$new_log" ]]; then
            echo "$new_log"
            return 0
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# run_bench <example> <kernels_dir> <golden> [case_name]
#   Run one benchmark invocation and parse timing from the resulting log.
#   Sets global PASS / FAIL counters.
# ---------------------------------------------------------------------------
run_bench() {
    local example="$1" kernels_dir="$2" golden="$3" case_name="${4:-}"

    if [[ -n "$case_name" ]]; then
        echo "  ---- $case_name ----"
    fi

    # Snapshot existing logs
    local pre_log_file
    pre_log_file=$(mktemp)
    trap 'rm -f -- "$pre_log_file"' RETURN
    ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort > "$pre_log_file" || true

    # Build run command
    local run_cmd=(
        python3 "$RUN_EXAMPLE"
        -k "$kernels_dir" -g "$golden"
        -p "$PLATFORM" -d "$DEVICE_ID"
        -n "$ROUNDS"
    )
    if [[ -n "$case_name" ]]; then
        run_cmd+=(--case "$case_name")
    fi
    run_cmd+=("${EXTRA_ARGS[@]}")

    # Run example
    if ! "${run_cmd[@]}" > /dev/null 2>&1; then
        echo "  FAILED: run_example.py returned non-zero"
        ((FAIL++)) || true
        return
    fi

    # Find new device log
    local new_log
    new_log=$(wait_for_new_log "$pre_log_file")

    if [[ -z "$new_log" ]]; then
        echo "  FAILED: no device log found in $DEVICE_LOG_DIR"
        ((FAIL++)) || true
        return
    fi

    echo "  Log: $new_log"
    if parse_timing "$new_log"; then
        ((PASS++)) || true
    else
        ((FAIL++)) || true
    fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

echo ""
echo "Runtime: $RUNTIME"
echo "Tests dir: $EXAMPLES_DIR"

for example in "${EXAMPLE_ORDER[@]}"; do
    case_list="${EXAMPLE_CASES[$example]:-}"

    EXAMPLE_DIR="$EXAMPLES_DIR/$example"
    KERNELS_DIR="$EXAMPLE_DIR/kernels"
    GOLDEN="$EXAMPLE_DIR/golden.py"

    echo ""
    echo "================================================================"
    echo "  $example"
    echo "================================================================"

    if [[ ! -f "$GOLDEN" || ! -d "$KERNELS_DIR" ]]; then
        echo "  SKIP: missing kernels/ or golden.py"
        ((FAIL++)) || true
        continue
    fi

    if [[ -z "${case_list:-}" ]]; then
        run_bench "$example" "$KERNELS_DIR" "$GOLDEN"
    else
        IFS=',' read -ra cases <<< "$case_list"
        for c in "${cases[@]}"; do
            run_bench "$example" "$KERNELS_DIR" "$GOLDEN" "$c"
        done
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL))
echo ""
echo "================================================================"
echo "  Benchmark complete ($RUNTIME): $PASS passed, $FAIL failed ($TOTAL total)"
echo "================================================================"

[[ $FAIL -eq 0 ]]
