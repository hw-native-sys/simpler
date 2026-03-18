#!/usr/bin/env bash
# Benchmark wrapper: run examples on hardware,
# then parse device-log timing lines to report per-round latency.
#
# Usage:
#   ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>]
#
# Edit the EXAMPLE_CASES map below to control which examples and cases to run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXAMPLE="$PROJECT_ROOT/examples/scripts/run_example.py"

# ---------------------------------------------------------------------------
# Examples to benchmark and their case lists.
# Key   = directory name under tests/device_tests/<platform>/tensormap_and_ringbuffer/
# Value = comma-separated case names to run (empty string = run DEFAULT_CASE)
#
# Available cases per example (from golden.py ALL_CASES):
#   alternating_matmul_add : Case1, Case2
#   benchmark_bgemm        : Case0, Case1, Case2, Case3, Case4
#   paged_attention_unroll : Case1, Case2, Case3
#   batch_paged_attention  : Case1, Case2, Case3
#   paged_attention        : Case1, Case2, Case3, Case4, Case5, Case6
# ---------------------------------------------------------------------------
declare -A EXAMPLE_CASES=(
    [alternating_matmul_add]=""
    [benchmark_bgemm]=""
    [paged_attention_unroll]="Case1,Case2"
    [batch_paged_attention]=""
    [paged_attention]=""
)

# Ordered list to control benchmark execution order
EXAMPLE_ORDER=(
    alternating_matmul_add
    benchmark_bgemm
    paged_attention_unroll
    batch_paged_attention
    paged_attention
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DEVICE_ID=0
ROUNDS=10
PLATFORM=a2a3
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
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run all examples and report per-round timing from device logs

Usage:
  ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>]

Options:
  -p, --platform Platform to run on (default: a2a3)
  -d, --device   Device ID (default: 0)
  -n, --rounds   Override number of rounds for each example (default: 10)
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
EXAMPLES_DIR="$PROJECT_ROOT/tests/device_tests/${PLATFORM}/tensormap_and_ringbuffer"

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
    timing=$(grep -E 'Thread=[0-9]+ (orch_start|end)=|BENCHMARK: thread=[0-9]+ end=|Thread [0-9]+: Scheduler summary: total_time=' "$log_file" || true)

    if [[ -z "$timing" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi

    echo "$timing" | awk '
    function flush_round() {
        if (round >= 0 && max_end > 0 && min_orch_start > 0) {
            r_elapsed[round] = (max_end - min_orch_start) / 50.0

            # Orch wall-clock: BENCHMARK end (= last orch finish) - earliest orch start
            if (benchmark_end > 0)
                r_orch[round] = (benchmark_end - min_orch_start) / 50.0

            # Sched wall-clock: max(sched_end) - min(sched_start)
            # Derive sched_start per thread: end_cycle - total_time_us * 50
            min_ss = 0; max_se = 0
            for (tid in sched_total_us) {
                if (tid in thread_end) {
                    ss = thread_end[tid] - sched_total_us[tid] * 50
                    if (min_ss == 0 || ss < min_ss) min_ss = ss
                    if (thread_end[tid] > max_se) max_se = thread_end[tid]
                }
            }
            if (max_se > 0 && min_ss > 0)
                r_sched[round] = (max_se - min_ss) / 50.0

            count++
        }
    }
    BEGIN {
        round = 0; min_orch_start = 0; max_end = 0; benchmark_end = 0; count = 0
    }
    /orch_start=/ {
        match($0, /Thread=([0-9]+)/, tm)
        tid = tm[1] + 0
        if (tid in orch_seen) {
            flush_round()
            round++
            min_orch_start = 0; max_end = 0; benchmark_end = 0
            delete orch_seen; delete thread_end; delete sched_total_us
        }
        orch_seen[tid] = 1
        match($0, /orch_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_orch_start == 0 || val < min_orch_start) min_orch_start = val
    }
    # Scheduler/orch thread end cycle (uppercase Thread=)
    /Thread=[0-9]+ end=/ {
        match($0, /Thread=([0-9]+)/, tm); tid = tm[1] + 0
        match($0, /end=([0-9]+)/, m); val = m[1] + 0
        thread_end[tid] = val
        if (val > max_end) max_end = val
    }
    # Orchestrator benchmark end cycle (lowercase thread=, last orch to finish)
    /BENCHMARK:.*end=/ {
        match($0, /end=([0-9]+)/, m)
        benchmark_end = m[1] + 0
    }
    # Scheduler summary: extract thread id and total_time for sched_start derivation
    /Scheduler summary: total_time=/ {
        match($0, /Thread ([0-9]+):/, tm); tid = tm[1] + 0
        match($0, /total_time=([0-9.]+)us/, m)
        sched_total_us[tid] = m[1] + 0.0
    }
    END {
        flush_round()
        if (count == 0) { print "  (no rounds parsed)"; exit 1 }

        # Detect whether orch/sched data is available
        has_detail = 0
        for (i = 0; i < count; i++) {
            if (r_orch[i] > 0 || r_sched[i] > 0) { has_detail = 1; break }
        }

        if (has_detail) {
            printf "  %-8s  %12s  %12s  %12s\n", "Round", "Elapsed (us)", "Orch (us)", "Sched (us)"
            printf "  %-8s  %12s  %12s  %12s\n", "-----", "------------", "---------", "----------"
        } else {
            printf "  %-8s  %12s\n", "Round", "Elapsed (us)"
            printf "  %-8s  %12s\n", "-----", "------------"
        }
        sum_e = 0; min_e = r_elapsed[0]; max_e = r_elapsed[0]
        sum_o = 0; min_o = r_orch[0]; max_o = r_orch[0]
        sum_s = 0; min_s = r_sched[0]; max_s = r_sched[0]
        for (i = 0; i < count; i++) {
            if (has_detail)
                printf "  %-8d  %12.1f  %12.1f  %12.1f\n", i, r_elapsed[i], r_orch[i], r_sched[i]
            else
                printf "  %-8d  %12.1f\n", i, r_elapsed[i]
            sum_e += r_elapsed[i]
            if (r_elapsed[i] < min_e) min_e = r_elapsed[i]
            if (r_elapsed[i] > max_e) max_e = r_elapsed[i]
            sum_o += r_orch[i]
            if (r_orch[i] < min_o) min_o = r_orch[i]
            if (r_orch[i] > max_o) max_o = r_orch[i]
            sum_s += r_sched[i]
            if (r_sched[i] < min_s) min_s = r_sched[i]
            if (r_sched[i] > max_s) max_s = r_sched[i]
        }
        printf "\n  Avg: %.1f us  (%d rounds)\n", sum_e / count, count
        if (count > 2) {
            trimmed = (sum_e - min_e - max_e) / (count - 2)
            printf "  Trimmed Avg: %.1f us  (excluding min=%.1f, max=%.1f)\n", trimmed, min_e, max_e
        }
        if (has_detail) {
            printf "\n  Orch  Avg: %.1f us", sum_o / count
            if (count > 2) {
                trimmed_o = (sum_o - min_o - max_o) / (count - 2)
                printf "  Trimmed: %.1f us", trimmed_o
            }
            printf "  Total: %.1f us\n", sum_o

            printf "  Sched Avg: %.1f us", sum_s / count
            if (count > 2) {
                trimmed_s = (sum_s - min_s - max_s) / (count - 2)
                printf "  Trimmed: %.1f us", trimmed_s
            }
            printf "  Total: %.1f us\n", sum_s
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
echo "  Benchmark complete: $PASS passed, $FAIL failed ($TOTAL total)"
echo "================================================================"

[[ $FAIL -eq 0 ]]
