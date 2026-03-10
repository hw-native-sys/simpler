#!/usr/bin/env bash
# Benchmark wrapper: run examples on a2a3 hardware,
# then parse device-log timing lines to report per-round latency.
#
# Usage:
#   ./tools/benchmark_rounds.sh [-d <device>] [-n <rounds>]
#
# Runs all examples listed in EXAMPLES array and prints timing for each.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXAMPLE="$PROJECT_ROOT/examples/scripts/run_example.py"
EXAMPLES_DIR="$PROJECT_ROOT/tests/device_tests/tensormap_and_ringbuffer"

# ---------------------------------------------------------------------------
# Examples to benchmark (paths relative to examples/tensormap_and_ringbuffer/)
# Each entry is just the directory name; kernels/ and golden.py are implied.
# ---------------------------------------------------------------------------
EXAMPLES=(
    alternating_matmul_add
    benchmark_bgemm
    batch_paged_attention
    paged_attention
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DEVICE_ID=0
ROUNDS=10
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--device)
            DEVICE_ID="$2"
            shift 2
            ;;
        -n|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run all examples and report per-round timing from device logs

Usage:
  ./tools/benchmark_rounds.sh [-d <device>] [-n <rounds>]

Options:
  -d, --device   Device ID (default: 0)
  -n, --rounds   Override number of rounds for each example (default: 10)
  -h, --help     Show this help

All other options are passed through to run_example.py (e.g. --case).

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
    timing=$(grep -E 'Thread=[0-9]+ (orch_start|end)=' "$log_file" || true)

    if [[ -z "$timing" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi

    echo "$timing" | awk '
    /orch_start=/ {
        if (round > 0 && max_end > 0) {
            elapsed = max_end - orch_start
            results[round - 1] = elapsed / 50.0
            count++
        }
        round++
        match($0, /orch_start=([0-9]+)/, m)
        orch_start = m[1] + 0
        max_end = 0
    }
    /end=/ {
        match($0, /end=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_end) max_end = val
    }
    END {
        if (round > 0 && max_end > 0) {
            results[round - 1] = (max_end - orch_start) / 50.0
            count++
        }
        if (count == 0) { print "  (no rounds parsed)"; exit 1 }

        printf "  %-8s  %12s\n", "Round", "Elapsed (us)"
        printf "  %-8s  %12s\n", "-----", "------------"
        sum_v = 0
        for (i = 0; i < count; i++) {
            printf "  %-8d  %12.1f\n", i, results[i]
            sum_v += results[i]
        }
        printf "\n  Avg: %.1f us  (%d rounds)\n", sum_v / count, count
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
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

for example in "${EXAMPLES[@]}"; do
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

    # Snapshot existing logs
    PRE_LOG_FILE=$(mktemp)
    ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort > "$PRE_LOG_FILE" || true

    # Run example
    if ! python3 "$RUN_EXAMPLE" \
            -k "$KERNELS_DIR" -g "$GOLDEN" \
            -p a2a3 -d "$DEVICE_ID" \
            -n "$ROUNDS" \
            "${EXTRA_ARGS[@]}" > /dev/null 2>&1; then
        echo "  FAILED: run_example.py returned non-zero"
        rm -f "$PRE_LOG_FILE"
        ((FAIL++)) || true
        continue
    fi

    # Find new device log
    NEW_LOG=$(wait_for_new_log "$PRE_LOG_FILE")
    rm -f "$PRE_LOG_FILE"

    if [[ -z "$NEW_LOG" ]]; then
        echo "  FAILED: no device log found in $DEVICE_LOG_DIR"
        ((FAIL++)) || true
        continue
    fi

    echo "  Log: $NEW_LOG"
    if parse_timing "$NEW_LOG"; then
        ((PASS++)) || true
    else
        ((FAIL++)) || true
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Benchmark complete: $PASS passed, $FAIL failed (${#EXAMPLES[@]} total)"
echo "================================================================"

[[ $FAIL -eq 0 ]]
