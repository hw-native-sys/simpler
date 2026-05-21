#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Benchmark wrapper: run examples on hardware and report per-round latency
# in five columns:
#   - Host   from RunTiming (host_wall_us)
#   - Device from RunTiming (device_wall_us, AICPU orch mailbox)
#   - Total  device-log: full span across sched/orch events
#   - Sched  device-log: sched_start -> sched_end
#   - Orch   device-log: orch_start  -> orch_end
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
# Key   = directory name under tests/st/<platform>/<runtime>/
# Value = comma-separated case names to run (empty string = run DEFAULT_CASE)
# ---------------------------------------------------------------------------

# --- tensormap_and_ringbuffer ---
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
  -r, --runtime  Runtime to benchmark: tensormap_and_ringbuffer (default)
  -v, --verbose  Save detailed test_*.py output to a timestamped log file
  -h, --help     Show this help

All other options are passed through to the underlying `python test_*.py`
invocation (e.g. --case).

Edit the EXAMPLE_CASES map at the top of this script to control which
examples and cases to benchmark.

Output:
  Per-round and average latency (microseconds) in 5 columns:
  Host, Device (from RunTiming) + Total, Sched, Orch (from device log).
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
# Derive arch from platform and set examples directories
# ---------------------------------------------------------------------------
# Search both examples/ (migrated tests) and tests/st/ (legacy tests)
ARCH="${PLATFORM%%sim}"  # strip "sim" suffix if present
EXAMPLES_DIRS=(
    "$PROJECT_ROOT/tests/st/${ARCH}/${RUNTIME}"
    "$PROJECT_ROOT/examples/${ARCH}/${RUNTIME}"
)

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
    *)
        echo "ERROR: unknown runtime '$RUNTIME'. Use tensormap_and_ringbuffer."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Resolve CANN device log directory: $ASCEND_WORK_PATH/log/debug or ~/ascend/log/debug
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
# parse_timing <fw_stdout_file> <device_log_file>
#   Merge per-round timing from two sources into a 5-column table:
#     - Host (us)   from RunTiming (framework `_log_round_timings` stdout)
#     - Device (us) from RunTiming (AICPU mailbox orch_start/end)
#     - Total (us)  device-log: max(end) - min(start) across sched/orch events
#     - Sched (us)  device-log: sched_start -> sched_end
#     - Orch (us)   device-log: orch_start -> orch_end
# ---------------------------------------------------------------------------
parse_timing() {
    local fw_file="$1"
    local log_file="$2"

    # Slice device log to just timing lines (keeps awk input small).
    local dev_timing_file
    dev_timing_file=$(mktemp)
    trap 'rm -f -- "$dev_timing_file"' RETURN
    grep -E 'Thread [0-9]+: (sched_start|orch_start|orch_end|sched_end|orch_stage_end)' \
        "$log_file" > "$dev_timing_file" 2>/dev/null || true

    if [[ ! -s "$fw_file" && ! -s "$dev_timing_file" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi

    awk -v freq="$FREQ" '
    function flush_round() {
        if (round >= 0 && max_end > 0 && min_start > 0) {
            total_results[round] = (max_end - min_start) / freq
            if (max_sched_end > 0 && min_sched_start > 0)
                sched_results[round] = (max_sched_end - min_sched_start) / freq
            if (max_orch_end > 0 && min_orch_start > 0)
                orch_results[round] = (max_orch_end - min_orch_start) / freq
            dev_count++
        }
    }
    function new_round() {
        flush_round()
        round++
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
        delete sched_seen
        delete orch_seen
    }
    function trimmed(label, arr, n, trim,    i, j, k, tc, ts) {
        for (i = 2; i <= n; i++) {
            k = arr[i]; j = i - 1
            while (j >= 1 && arr[j] > k) { arr[j+1] = arr[j]; j-- }
            arr[j+1] = k
        }
        tc = n - 2 * trim; ts = 0
        for (i = trim + 1; i <= n - trim; i++) ts += arr[i]
        printf "  %s Trimmed Avg: %.1f us  (dropped %d low + %d high, %d rounds used)\n", \
               label, ts / tc, trim, trim, tc
    }
    BEGIN {
        round = 0; dev_count = 0
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
        has_sched = 0; has_orch_end = 0
        fw_n = 0; in_fw = 0
    }
    # First file: framework `_log_round_timings` stdout (Host / Device per round).
    # Header may be concatenated with the test runner status line (uses end=""),
    # so anchor on "Round...Host (us)" anywhere on the line, not column 0.
    FNR == NR {
        if (match($0, /Round +Host \(us\)/))   { in_fw = 1; next }
        if (in_fw && /^  -+$/)                 next
        if (in_fw && /Avg Host:/)              { in_fw = 0; next }
        if (in_fw && NF == 0)                  { in_fw = 0; next }
        if (in_fw && match($0, /^  +([0-9]+) +([0-9.]+)( +([0-9.]+))?/, m)) {
            r = m[1] + 0
            fw_host[r] = m[2] + 0
            if (m[4] != "") fw_dev[r] = m[4] + 0
            if (r + 1 > fw_n) fw_n = r + 1
        }
        next
    }
    # Second file: device-log timing lines
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

        show_host  = (fw_n > 0)
        show_dev   = 0
        for (i = 0; i < fw_n; i++) if ((i in fw_dev) && fw_dev[i] > 0) { show_dev = 1; break }
        has_total  = (dev_count > 0)
        show_sched = has_sched
        show_orch  = has_orch_end

        if (!has_total && fw_n == 0) {
            print "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
            exit 1
        }

        n_rounds = (dev_count > fw_n) ? dev_count : fw_n

        # Header / separator
        hdr = sprintf("  %-6s", "Round")
        sep = sprintf("  %-6s", "------")
        if (show_host)  { hdr = hdr sprintf("  %12s", "Host (us)");   sep = sep sprintf("  %12s", "------------") }
        if (show_dev)   { hdr = hdr sprintf("  %12s", "Device (us)"); sep = sep sprintf("  %12s", "------------") }
        if (has_total)  { hdr = hdr sprintf("  %12s", "Total (us)");  sep = sep sprintf("  %12s", "------------") }
        if (show_sched) { hdr = hdr sprintf("  %12s", "Sched (us)");  sep = sep sprintf("  %12s", "------------") }
        if (show_orch)  { hdr = hdr sprintf("  %12s", "Orch (us)");   sep = sep sprintf("  %12s", "------------") }
        print hdr; print sep

        cnt_host = 0; cnt_dev = 0; cnt_tot = 0; cnt_sch = 0; cnt_orc = 0
        sum_host = 0; sum_dev = 0; sum_tot = 0; sum_sch = 0; sum_orc = 0

        for (i = 0; i < n_rounds; i++) {
            row = sprintf("  %-6d", i)
            if (show_host) {
                if (i in fw_host) {
                    row = row sprintf("  %12.1f", fw_host[i])
                    sum_host += fw_host[i]; cnt_host++; host_arr[cnt_host] = fw_host[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_dev) {
                if ((i in fw_dev) && fw_dev[i] > 0) {
                    row = row sprintf("  %12.1f", fw_dev[i])
                    sum_dev += fw_dev[i]; cnt_dev++; dev_arr[cnt_dev] = fw_dev[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (has_total) {
                if (i in total_results) {
                    row = row sprintf("  %12.1f", total_results[i])
                    sum_tot += total_results[i]; cnt_tot++; tot_arr[cnt_tot] = total_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_sched) {
                if (i in sched_results) {
                    row = row sprintf("  %12.1f", sched_results[i])
                    sum_sch += sched_results[i]; cnt_sch++; sch_arr[cnt_sch] = sched_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_orch) {
                if (i in orch_results) {
                    row = row sprintf("  %12.1f", orch_results[i])
                    sum_orc += orch_results[i]; cnt_orc++; orc_arr[cnt_orc] = orch_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            print row
        }

        # Averages: Host | Device | Total | Sched | Orch
        avg_line = ""; avg_sep = ""
        if (show_host  && cnt_host > 0) { avg_line = avg_line avg_sep sprintf("Host Avg: %.1f us",   sum_host / cnt_host); avg_sep = "  |  " }
        if (show_dev   && cnt_dev > 0)  { avg_line = avg_line avg_sep sprintf("Device Avg: %.1f us", sum_dev  / cnt_dev);  avg_sep = "  |  " }
        if (has_total  && cnt_tot > 0)  { avg_line = avg_line avg_sep sprintf("Total Avg: %.1f us",  sum_tot  / cnt_tot);  avg_sep = "  |  " }
        if (show_sched && cnt_sch > 0)  { avg_line = avg_line avg_sep sprintf("Sched Avg: %.1f us",  sum_sch  / cnt_sch);  avg_sep = "  |  " }
        if (show_orch  && cnt_orc > 0)  { avg_line = avg_line avg_sep sprintf("Orch Avg: %.1f us",   sum_orc  / cnt_orc);  avg_sep = "  |  " }
        printf "\n  %s  (%d rounds)\n", avg_line, n_rounds

        TRIM = 10
        if (cnt_host > 2 * TRIM) trimmed("Host",   host_arr, cnt_host, TRIM)
        if (cnt_dev  > 2 * TRIM) trimmed("Device", dev_arr,  cnt_dev,  TRIM)
        if (cnt_tot  > 2 * TRIM) trimmed("Total",  tot_arr,  cnt_tot,  TRIM)
        if (cnt_sch  > 2 * TRIM) trimmed("Sched",  sch_arr,  cnt_sch,  TRIM)
        if (cnt_orc  > 2 * TRIM) trimmed("Orch",   orc_arr,  cnt_orc,  TRIM)
    }' "$fw_file" "$dev_timing_file"
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
# run_bench <example> <example_dir> [case_name]
#   Run one benchmark invocation (via `python test_*.py`) and parse timing
#   from the resulting log. Skips the example if it has no test_*.py.
#   Sets global PASS / FAIL counters.
# ---------------------------------------------------------------------------
run_bench() {
    local example="$1" example_dir="$2" case_name="${3:-}"

    if [[ -n "$case_name" ]]; then
        echo "  ---- $case_name ----"
    fi

    # Snapshot existing logs
    local pre_log_file fw_stdout_file
    pre_log_file=$(mktemp)
    fw_stdout_file=$(mktemp)
    trap 'rm -f -- "$pre_log_file" "$fw_stdout_file"' RETURN
    ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort > "$pre_log_file" || true

    # Build run command using test_*.py
    local test_file
    test_file=$(find "$example_dir" -maxdepth 1 -name 'test_*.py' -print -quit 2>/dev/null || true)

    local run_cmd
    if [[ -n "$test_file" ]]; then
        run_cmd=(
            python3 "$test_file"
            --platform "$PLATFORM" --device "$DEVICE_ID"
            --rounds "$ROUNDS" --skip-golden
        )
    else
        echo "  SKIPPED: no test_*.py found in $example_dir"
        return
    fi
    if [[ -n "$case_name" ]]; then
        run_cmd+=(--case "$case_name")
        [[ -n "$test_file" ]] && run_cmd+=(--manual include)
    fi
    run_cmd+=("${EXTRA_ARGS[@]}")

    # Run example, capturing stdout/stderr for Host/Device timing parse
    vlog "Running: ${run_cmd[*]}"
    local rc=0
    "${run_cmd[@]}" > "$fw_stdout_file" 2>&1 || rc=$?
    if [[ -n "$VERBOSE_LOG" && -s "$fw_stdout_file" ]]; then
        cat "$fw_stdout_file" >> "$VERBOSE_LOG"
    fi
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED: benchmark run returned non-zero"
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
    timing_output=$(parse_timing "$fw_stdout_file" "$new_log") || parse_rc=$?
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
    avg_line=$(echo "$timing_output" | grep -E '(Host|Device|Total|Sched|Orch) Avg:' | grep -v 'Trimmed' | head -1 || true)
    local avg_host="-" avg_device="-" avg_total="-" avg_sched="-" avg_orch="-"
    if [[ -n "$avg_line" ]]; then
        avg_host=$(  echo "$avg_line" | grep -oE 'Host Avg: [0-9.]+'   || true); avg_host=${avg_host##* }
        avg_device=$(echo "$avg_line" | grep -oE 'Device Avg: [0-9.]+' || true); avg_device=${avg_device##* }
        avg_total=$( echo "$avg_line" | grep -oE 'Total Avg: [0-9.]+'  || true); avg_total=${avg_total##* }
        avg_sched=$( echo "$avg_line" | grep -oE 'Sched Avg: [0-9.]+'  || true); avg_sched=${avg_sched##* }
        avg_orch=$(  echo "$avg_line" | grep -oE 'Orch Avg: [0-9.]+'   || true); avg_orch=${avg_orch##* }
        [[ -z "$avg_host" ]]   && avg_host="-"
        [[ -z "$avg_device" ]] && avg_device="-"
        [[ -z "$avg_total" ]]  && avg_total="-"
        [[ -z "$avg_sched" ]]  && avg_sched="-"
        [[ -z "$avg_orch" ]]   && avg_orch="-"
    fi

    SUMMARY_NAMES+=("$label")
    SUMMARY_HOST+=("$avg_host")
    SUMMARY_DEVICE+=("$avg_device")
    SUMMARY_TOTAL+=("$avg_total")
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
SUMMARY_HOST=()
SUMMARY_DEVICE=()
SUMMARY_TOTAL=()
SUMMARY_SCHED=()
SUMMARY_ORCH=()

echo ""
echo "Runtime: $RUNTIME"

for example in "${EXAMPLE_ORDER[@]}"; do
    case_list="${EXAMPLE_CASES[$example]:-}"

    # Search for example: prefer test_*.py (new style), fall back to golden.py (legacy).
    # tests/st/ is searched before examples/ since benchmarks use production-scale cases.
    EXAMPLE_DIR=""
    for dir in "${EXAMPLES_DIRS[@]}"; do
        candidate="$dir/$example"
        if [[ -d "$candidate" ]] && ls "$candidate"/test_*.py 1>/dev/null 2>&1; then
            EXAMPLE_DIR="$candidate"
            break
        fi
    done
    if [[ -z "$EXAMPLE_DIR" ]]; then
        for dir in "${EXAMPLES_DIRS[@]}"; do
            candidate="$dir/$example"
            if [[ -f "$candidate/golden.py" && -d "$candidate/kernels" ]]; then
                EXAMPLE_DIR="$candidate"
                break
            fi
        done
    fi

    echo ""
    echo "================================================================"
    echo "  $example"
    echo "================================================================"

    if [[ -z "$EXAMPLE_DIR" ]]; then
        echo "  SKIP: not found in any search directory"
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

# ---------------------------------------------------------------------------
# Performance Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    # Show only columns that have at least one non-"-" value
    _has_host=0; _has_device=0; _has_total=0; _has_sched=0; _has_orch=0
    for _i in "${!SUMMARY_NAMES[@]}"; do
        [[ "${SUMMARY_HOST[$_i]}"   != "-" ]] && _has_host=1
        [[ "${SUMMARY_DEVICE[$_i]}" != "-" ]] && _has_device=1
        [[ "${SUMMARY_TOTAL[$_i]}"  != "-" ]] && _has_total=1
        [[ "${SUMMARY_SCHED[$_i]}"  != "-" ]] && _has_sched=1
        [[ "${SUMMARY_ORCH[$_i]}"   != "-" ]] && _has_orch=1
    done

    echo ""
    echo "================================================================"
    echo "  Performance Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    _hdr=$(printf "  %-40s" "Example")
    _sep=$(printf "  %-40s" "----------------------------------------")
    if [[ $_has_host   -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Host (us)");   _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_device -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Device (us)"); _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_total  -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Total (us)");  _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_sched  -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Sched (us)");  _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_orch   -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Orch (us)");   _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    echo "$_hdr"
    echo "$_sep"

    for _i in "${!SUMMARY_NAMES[@]}"; do
        _row=$(printf "  %-40s" "${SUMMARY_NAMES[$_i]}")
        if [[ $_has_host   -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_HOST[$_i]}");   fi
        if [[ $_has_device -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_DEVICE[$_i]}"); fi
        if [[ $_has_total  -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_TOTAL[$_i]}");  fi
        if [[ $_has_sched  -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_SCHED[$_i]}");  fi
        if [[ $_has_orch   -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_ORCH[$_i]}");   fi
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
