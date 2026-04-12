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
# Key   = directory name under tests/st/<platform>/<runtime>/
# Value = comma-separated case names to run (empty string = run DEFAULT_CASE)
# ---------------------------------------------------------------------------

# --- tensormap_and_ringbuffer ---
declare -A TMR_EXAMPLE_CASES=(
    [alternating_matmul_add]=""
    [benchmark_bgemm]=""
    [paged_attention]="Case1,Case2"
    [paged_attention_unroll]="Case1,Case2"
    [batch_paged_attention]=""
)
TMR_EXAMPLE_ORDER=(
    alternating_matmul_add
    benchmark_bgemm
    paged_attention
    paged_attention_unroll
    batch_paged_attention
)

# --- aicpu_build_graph ---
declare -A ABG_EXAMPLE_CASES=(
    [paged_attention]="Case1,Case2"
    [paged_attention_unroll]="Case1,Case2"
)
ABG_EXAMPLE_ORDER=(
    paged_attention
    paged_attention_unroll
)

# --- tensormap_and_ringbuffer_partial_manual ---
declare -A TMR_PARTIAL_MANUAL_EXAMPLE_CASES=(
    [alternating_matmul_add_partial_manual]=""
    [benchmark_bgemm_partial_manual]=""
    [paged_attention_partial_manual]="Case1,Case2"
    [paged_attention_unroll_partial_manual]="Case1,Case2"
    [batch_paged_attention_partial_manual]=""
)
TMR_PARTIAL_MANUAL_EXAMPLE_ORDER=(
    alternating_matmul_add_partial_manual
    benchmark_bgemm_partial_manual
    paged_attention_partial_manual
    paged_attention_unroll_partial_manual
    batch_paged_attention_partial_manual
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DEVICE_ID=0
ROUNDS=100
PLATFORM=a2a3
RUNTIME=tensormap_and_ringbuffer
VERBOSE=0
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
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run all examples and report per-round timing from device logs

Usage:
  ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>] [-v]

Options:
  -p, --platform Platform to run on (default: a2a3)
  -d, --device   Device ID (default: 0)
  -n, --rounds   Override number of rounds for each example (default: 100)
  -r, --runtime  Runtime to benchmark: tensormap_and_ringbuffer (default),
                 tensormap_and_ringbuffer_partial_manual,
                 aicpu_build_graph
  -v, --verbose  Save detailed run_example.py output to a timestamped log file
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
# Verbose logging setup
# ---------------------------------------------------------------------------
VERBOSE_LOG=""
if [[ $VERBOSE -eq 1 ]]; then
    mkdir -p "$PROJECT_ROOT/outputs"
    VERBOSE_LOG="$PROJECT_ROOT/outputs/benchmark_$(date +%Y%m%d_%H%M%S).log"
    echo "Verbose log: $VERBOSE_LOG"
fi

vlog() {
    if [[ -n "$VERBOSE_LOG" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$VERBOSE_LOG"
    fi
}

# ---------------------------------------------------------------------------
# Derive arch from platform and set examples directory
# ---------------------------------------------------------------------------
TESTS_RUNTIME_DIR="$RUNTIME"

# Clock frequency (MHz) for converting cycle counts to microseconds
case "$PLATFORM" in
    a2a3) FREQ=50 ;;
    a5)   FREQ=1000 ;;
    *)    echo "ERROR: unsupported platform '$PLATFORM'. Use a2a3 or a5."; exit 1 ;;
esac

# Select example cases and order based on runtime
case "$RUNTIME" in
    tensormap_and_ringbuffer)
        declare -n EXAMPLE_CASES=TMR_EXAMPLE_CASES
        EXAMPLE_ORDER=("${TMR_EXAMPLE_ORDER[@]}")
        ;;
    tensormap_and_ringbuffer_partial_manual)
        TESTS_RUNTIME_DIR="tensormap_and_ringbuffer"
        declare -n EXAMPLE_CASES=TMR_PARTIAL_MANUAL_EXAMPLE_CASES
        EXAMPLE_ORDER=("${TMR_PARTIAL_MANUAL_EXAMPLE_ORDER[@]}")
        ;;
    aicpu_build_graph)
        declare -n EXAMPLE_CASES=ABG_EXAMPLE_CASES
        EXAMPLE_ORDER=("${ABG_EXAMPLE_ORDER[@]}")
        ;;
    *)
        echo "ERROR: unknown runtime '$RUNTIME'. Use tensormap_and_ringbuffer, tensormap_and_ringbuffer_partial_manual, or aicpu_build_graph."
        exit 1
        ;;
esac

EXAMPLES_DIR="$PROJECT_ROOT/tests/st/${PLATFORM}/${TESTS_RUNTIME_DIR}"

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
    timing=$(grep -E 'Thread [0-9]+: (sched_start|orch_start|orch_end|sched_end|orch_stage_end)' "$log_file" || true)

    if [[ -z "$timing" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi

    echo "$timing" | awk -v freq="$FREQ" '
    function new_round() {
        flush_round()
        round++
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
        delete sched_seen
        delete orch_seen
    }
    function flush_round() {
        if (round >= 0 && max_end > 0 && min_start > 0) {
            results[round] = (max_end - min_start) / freq
            if (max_sched_end > 0 && min_sched_start > 0)
                sched_results[round] = (max_sched_end - min_sched_start) / freq
            if (max_orch_end > 0 && min_orch_start > 0)
                orch_results[round] = (max_orch_end - min_orch_start) / freq
            count++
        }
    }
    BEGIN {
        round = 0; count = 0
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
        has_sched = 0; has_orch_end = 0
    }
    /sched_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in sched_seen) new_round()
        sched_seen[tid] = 1
        has_sched = 1
        match($0, /sched_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_sched_start == 0 || val < min_sched_start) min_sched_start = val
        if (min_start == 0 || val < min_start) min_start = val
    }
    /orch_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in orch_seen) new_round()
        orch_seen[tid] = 1
        match($0, /orch_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_orch_start == 0 || val < min_orch_start) min_orch_start = val
        if (min_start == 0 || val < min_start) min_start = val
    }
    /sched_end[^=]*=/ {
        match($0, /sched_end[^=]*=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_sched_end) max_sched_end = val
        if (val > max_end) max_end = val
    }
    /orch_end=/ {
        match($0, /orch_end=([0-9]+)/, m)
        val = m[1] + 0
        has_orch_end = 1
        if (val > max_orch_end) max_orch_end = val
        if (val > max_end) max_end = val
    }
    /orch_stage_end=/ {
        match($0, /orch_stage_end=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_end) max_end = val
    }
    END {
        flush_round()
        if (count == 0) { print "  (no rounds parsed)"; exit 1 }

        show_sched = has_sched
        show_orch = has_orch_end

        # Header
        hdr = sprintf("  %-8s  %12s", "Round", "Elapsed (us)")
        sep = sprintf("  %-8s  %12s", "-----", "------------")
        if (show_sched) { hdr = hdr sprintf("  %12s", "Sched (us)"); sep = sep sprintf("  %12s", "----------") }
        if (show_orch)  { hdr = hdr sprintf("  %12s", "Orch (us)");  sep = sep sprintf("  %12s", "---------")  }
        print hdr; print sep

        sum_v = 0; min_v = results[0]; max_v = results[0]
        sum_s = 0; min_s = sched_results[0]; max_s = sched_results[0]
        sum_o = 0; min_o = orch_results[0]; max_o = orch_results[0]

        for (i = 0; i < count; i++) {
            line = sprintf("  %-8d  %12.1f", i, results[i])
            sum_v += results[i]
            if (results[i] < min_v) min_v = results[i]
            if (results[i] > max_v) max_v = results[i]
            if (show_sched) {
                line = line sprintf("  %12.1f", sched_results[i])
                sum_s += sched_results[i]
                if (sched_results[i] < min_s) min_s = sched_results[i]
                if (sched_results[i] > max_s) max_s = sched_results[i]
            }
            if (show_orch) {
                line = line sprintf("  %12.1f", orch_results[i])
                sum_o += orch_results[i]
                if (orch_results[i] < min_o) min_o = orch_results[i]
                if (orch_results[i] > max_o) max_o = orch_results[i]
            }
            print line
        }

        printf "\n  Avg: %.1f us", sum_v / count
        if (show_sched) printf "  |  Sched Avg: %.1f us", sum_s / count
        if (show_orch)  printf "  |  Orch Avg: %.1f us", sum_o / count
        printf "  (%d rounds)\n", count

        TRIM = 10
        if (count > 2 * TRIM) {
            # Insertion sort for each metric
            for (i = 0; i < count; i++) sv[i] = results[i]
            for (i = 1; i < count; i++) {
                k = sv[i]; j = i - 1
                while (j >= 0 && sv[j] > k) { sv[j+1] = sv[j]; j-- }
                sv[j+1] = k
            }
            tc = count - 2 * TRIM; ts = 0
            for (i = TRIM; i < count - TRIM; i++) ts += sv[i]
            printf "  Trimmed Avg: %.1f us  (dropped %d low + %d high, %d rounds used)\n", ts / tc, TRIM, TRIM, tc

            if (show_sched) {
                for (i = 0; i < count; i++) ss[i] = sched_results[i]
                for (i = 1; i < count; i++) {
                    k = ss[i]; j = i - 1
                    while (j >= 0 && ss[j] > k) { ss[j+1] = ss[j]; j-- }
                    ss[j+1] = k
                }
                ts2 = 0
                for (i = TRIM; i < count - TRIM; i++) ts2 += ss[i]
                printf "  Sched Trimmed Avg: %.1f us  (dropped %d low + %d high)\n", ts2 / tc, TRIM, TRIM
            }
            if (show_orch) {
                for (i = 0; i < count; i++) so[i] = orch_results[i]
                for (i = 1; i < count; i++) {
                    k = so[i]; j = i - 1
                    while (j >= 0 && so[j] > k) { so[j+1] = so[j]; j-- }
                    so[j+1] = k
                }
                ts3 = 0
                for (i = TRIM; i < count - TRIM; i++) ts3 += so[i]
                printf "  Orch Trimmed Avg: %.1f us  (dropped %d low + %d high)\n", ts3 / tc, TRIM, TRIM
            }
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
        -n "$ROUNDS" --skip-golden
    )
    if [[ -n "$case_name" ]]; then
        run_cmd+=(--case "$case_name")
    fi
    run_cmd+=("${EXTRA_ARGS[@]}")

    # Run example
    vlog "Running: ${run_cmd[*]}"
    local rc=0
    if [[ -n "$VERBOSE_LOG" ]]; then
        local run_output
        run_output=$("${run_cmd[@]}" 2>&1) || rc=$?
        if [[ -n "$run_output" ]]; then echo "$run_output" >> "$VERBOSE_LOG"; fi
    else
        "${run_cmd[@]}" > /dev/null 2>&1 || rc=$?
    fi
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED: run_example.py returned non-zero"
        vlog "FAILED: exit code $rc"
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
    local timing_output
    local parse_rc=0
    timing_output=$(parse_timing "$new_log") || parse_rc=$?
    echo "$timing_output"

    if [[ $parse_rc -ne 0 ]]; then
        ((FAIL++)) || true
        return
    fi
    ((PASS++)) || true

    # Extract averages for summary table
    local label="$example"
    [[ -n "$case_name" ]] && label="$example ($case_name)"

    local avg_line
    avg_line=$(echo "$timing_output" | grep "^  Avg:" || true)
    local avg_elapsed="-" avg_sched="-" avg_orch="-"
    if [[ -n "$avg_line" ]]; then
        avg_elapsed=$(echo "$avg_line" | awk '{print $2}')
        avg_sched=$(echo "$avg_line" | grep -o 'Sched Avg: [0-9.]*' | awk '{print $3}') || avg_sched="-"
        avg_orch=$(echo "$avg_line" | grep -o 'Orch Avg: [0-9.]*' | awk '{print $3}') || avg_orch="-"
    fi

    SUMMARY_NAMES+=("$label")
    SUMMARY_ELAPSED+=("$avg_elapsed")
    SUMMARY_SCHED+=("$avg_sched")
    SUMMARY_ORCH+=("$avg_orch")
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

# Summary collection arrays
SUMMARY_NAMES=()
SUMMARY_ELAPSED=()
SUMMARY_SCHED=()
SUMMARY_ORCH=()

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
# Performance Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    # Check if any sched/orch data exists across all runs
    _has_sched=0
    _has_orch=0
    for _i in "${!SUMMARY_NAMES[@]}"; do
        [[ "${SUMMARY_SCHED[$_i]}" != "-" ]] && _has_sched=1
        [[ "${SUMMARY_ORCH[$_i]}" != "-" ]] && _has_orch=1
    done

    echo ""
    echo "================================================================"
    echo "  Performance Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    # Header
    _hdr=$(printf "  %-40s  %12s" "Example" "Elapsed (us)")
    _sep=$(printf "  %-40s  %12s" "----------------------------------------" "------------")
    if [[ $_has_sched -eq 1 ]]; then
        _hdr=$(printf "%s  %12s" "$_hdr" "Sched (us)")
        _sep=$(printf "%s  %12s" "$_sep" "------------")
    fi
    if [[ $_has_orch -eq 1 ]]; then
        _hdr=$(printf "%s  %12s" "$_hdr" "Orch (us)")
        _sep=$(printf "%s  %12s" "$_sep" "------------")
    fi
    echo "$_hdr"
    echo "$_sep"

    # Rows
    for _i in "${!SUMMARY_NAMES[@]}"; do
        _row=$(printf "  %-40s  %12s" "${SUMMARY_NAMES[$_i]}" "${SUMMARY_ELAPSED[$_i]}")
        if [[ $_has_sched -eq 1 ]]; then
            _row=$(printf "%s  %12s" "$_row" "${SUMMARY_SCHED[$_i]}")
        fi
        if [[ $_has_orch -eq 1 ]]; then
            _row=$(printf "%s  %12s" "$_row" "${SUMMARY_ORCH[$_i]}")
        fi
        echo "$_row"
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL))
echo ""
echo "================================================================"
echo "  Benchmark complete ($RUNTIME): $PASS passed, $FAIL failed ($TOTAL total)"
echo "================================================================"

if [[ -n "$VERBOSE_LOG" ]]; then
    echo "  Verbose log saved to: $VERBOSE_LOG"
fi

[[ $FAIL -eq 0 ]]
