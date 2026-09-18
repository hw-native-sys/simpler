#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Benchmark wrapper: run examples on hardware and report per-round latency.
# All columns come from the `[STRACE]` markers the run emits to stderr, parsed
# by `strace_timing --rounds-table` (no CANN device log is read):
#   - Host      chip.run span (host wall, incl. Python)
#   - Device    chip.run.runner_run.device_wall span (full on-NPU AICPU wall)
#   - Effective orch∪sched merged window, from the TMR orch/sched markers'
#               device-domain ts+dur (the old device-log "Total", now pure-marker)
#   - Orch      TMR chip.run.runner_run.device_wall.orch span
#   - Sched     TMR chip.run.runner_run.device_wall.sched span
#
# Usage:
#   ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>] [--serial-orch-sched]
#
# Edit the architecture/runtime EXAMPLE_CASES lists below to control which examples and cases to run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Examples to benchmark and their case lists, per architecture and runtime.
# Each entry is <directory-name>=<comma-separated-case-names>.
# An empty case list runs DEFAULT_CASE.
# ---------------------------------------------------------------------------

# --- a2a3 + tensormap_and_ringbuffer ---
A2A3_TMR_EXAMPLE_CASES=(
    "alternating_matmul_add=Case1"
    "benchmark_bgemm=Case0"
    "paged_attention_unroll=Case1,Case2"
    "paged_attention_unroll_manual_scope=Case1,Case2"
    "batch_paged_attention=Case1"
    # spmd_paged_attention temporarily disabled: pre-existing onboard stall
    # (507018 S1:running-stalled), reproduces on baseline — see KNOWN_ISSUES.md.
    # "spmd_paged_attention=Case1,Case2"
    # The only case here with a per-task fanin degree above 1.5; every other
    # entry is a near-chain, so an O(fanout) regression in wiring, dependency-pool
    # allocation, or the completion fanout walk is invisible without it.
    "sliding_window_deps=Dense16"
    "qwen3_14b_decode=StressBatch16Seq3500"
)

# --- a2a3 + host_build_graph ---
A2A3_HBG_EXAMPLE_CASES=(
    "alternating_matmul_add=Case1"
    "benchmark_bgemm=Case0"
    "paged_attention_unroll=Case1,Case2"
    "paged_attention_unroll_manual_scope=Case1,Case2"
    "batch_paged_attention=Case1"
    "qwen3_14b_decode=GraphExecutionBatch16Seq3500"
)

# --- a5 + tensormap_and_ringbuffer ---
A5_TMR_EXAMPLE_CASES=(
    "alternating_matmul_add=Case1"
    "benchmark_bgemm=Case0"
    "paged_attention_unroll=Case1,Case2"
    "paged_attention_unroll_manual_scope=Case1,Case2"
    "batch_paged_attention=Case1"
    # The only case here with a per-task fanin degree above 1.5; every other
    # entry is a near-chain, so an O(fanout) regression in wiring, dependency-pool
    # allocation, or the completion fanout walk is invisible without it.
    "sliding_window_deps=Dense16"
    "qwen3_14b_decode=StressBatch16Seq3500"
)

# --- a5 + host_build_graph ---
A5_HBG_EXAMPLE_CASES=(
    "alternating_matmul_add=Case1"
    "benchmark_bgemm=Case0"
    "paged_attention_unroll=Case1,Case2"
    "paged_attention_unroll_manual_scope=Case1,Case2"
    "batch_paged_attention=Case1"
    "qwen3_14b_decode=GraphExecutionBatch16Seq3500"
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DEVICE_ID=0
ROUNDS=100
PLATFORM=a2a3
RUNTIME=tensormap_and_ringbuffer
VERBOSE=0
SERIAL_ORCH_SCHED=0
EXTRA_ARGS=()
EXTRA_ARGS_COUNT=0

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
        --serial-orch-sched)
            SERIAL_ORCH_SCHED=1
            shift
            ;;
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run all examples and report per-round timing from [STRACE] markers

Usage:
  ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>] [-v] [--serial-orch-sched]

Options:
  -p, --platform Platform to run on (default: a2a3)
  -d, --device   Device ID (default: 0)
  -n, --rounds   Override number of rounds for each example (default: 100)
  -r, --runtime  Runtime to benchmark: tensormap_and_ringbuffer (default),
                 host_build_graph
  -v, --verbose  Save detailed test_*.py output to a timestamped log file
  --serial-orch-sched
                 Run each TMR case twice: default parallel mode, then serial
                 orch->sched mode with SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE=1.
  -h, --help     Show this help

All other options are passed through to the underlying `python test_*.py`
invocation (e.g. --case).

Edit the architecture/runtime EXAMPLE_CASES lists at the top of this script
to control which examples and cases to benchmark.

Output:
  Per-round and average latency (microseconds), all from the [STRACE] markers:
  TMR: Host, Device, Effective, Orch, Sched.
  HBG: Host, Device. Columns without captured markers are omitted.
  All metrics are parsed by strace_timing --rounds-table.
USAGE
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            EXTRA_ARGS_COUNT=$((EXTRA_ARGS_COUNT + 1))
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

# Validate platform (a2a3 / a5; the "sim" suffix was already stripped into ARCH).
case "$PLATFORM" in
    a2a3|a2a3sim|a5|a5sim) ;;
    *) echo "ERROR: unsupported platform '$PLATFORM'. Use a2a3 or a5."; exit 1 ;;
esac

# Select example cases and order based on architecture and runtime.
case "$ARCH:$RUNTIME" in
    a2a3:tensormap_and_ringbuffer)
        EXAMPLE_CASES=("${A2A3_TMR_EXAMPLE_CASES[@]}")
        DEFAULT_MODE=parallel
        ;;
    a2a3:host_build_graph)
        EXAMPLE_CASES=("${A2A3_HBG_EXAMPLE_CASES[@]}")
        DEFAULT_MODE=default
        ;;
    a5:tensormap_and_ringbuffer)
        EXAMPLE_CASES=("${A5_TMR_EXAMPLE_CASES[@]}")
        DEFAULT_MODE=parallel
        ;;
    a5:host_build_graph)
        EXAMPLE_CASES=("${A5_HBG_EXAMPLE_CASES[@]}")
        DEFAULT_MODE=default
        ;;
    *)
        echo "ERROR: unknown runtime '$RUNTIME'. Use tensormap_and_ringbuffer or host_build_graph."
        exit 1
        ;;
esac

if [[ $SERIAL_ORCH_SCHED -eq 1 && "$RUNTIME" == "host_build_graph" ]]; then
    echo "ERROR: --serial-orch-sched is only supported by tensormap_and_ringbuffer."
    exit 1
fi

# ---------------------------------------------------------------------------
# parse_timing <fw_stdout_file>
#   Render per-round timing from the [STRACE] markers the run teed into
#   $fw_stdout_file (host stderr, via 2>&1). All columns come from one source —
#   the markers, parsed by `strace_timing --rounds-table`:
#     - Host (us)      [STRACE] chip.run span (host wall, incl. Python)
#     - Device (us)    [STRACE] device_wall span (full on-NPU AICPU wall)
#     - Effective (us) orch∪sched merged window, from orch/sched device-domain ts+dur
#     - Orch (us)      [STRACE] device_wall.orch span
#     - Sched (us)     [STRACE] device_wall.sched span
#   Effective/Orch/Sched come from TMR's AICPU phase subdivision.
#   No CANN device log is read.
# ---------------------------------------------------------------------------
parse_timing() {
    local fw_file="$1"

    local out
    out=$(python3 -m simpler_setup.tools.strace_timing "$fw_file" --rounds-table 2>/dev/null || true)

    if [[ -z "$out" || "$out" == *"No [STRACE] markers found."* ]]; then
        echo "  (no benchmark timing data — are SIMPLER_HOST_STRACE markers visible?)"
        return 1
    fi
    echo "$out"
}

# ---------------------------------------------------------------------------
# run_bench <example> <example_dir> [case_name] [mode]
#   Run one benchmark invocation (via `python test_*.py`) and parse timing
#   from the [STRACE] markers in its stderr. Skips the example if it has no
#   test_*.py. Sets global PASS / FAIL counters.
# ---------------------------------------------------------------------------

# Enough to carry a Python traceback plus the exception line that follows it,
# which is where a failing run states its own cause.
FAILURE_TAIL_LINES=25

run_bench() {
    local example="$1" example_dir="$2" case_name="${3:-}"
    local mode="${4:-parallel}"

    if [[ -n "$case_name" ]]; then
        echo "  ---- $case_name [$mode] ----"
    else
        echo "  ---- $mode ----"
    fi

    local fw_stdout_file
    fw_stdout_file=$(mktemp)
    trap 'rm -f -- "$fw_stdout_file"' RETURN

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
    # Bash 3.2 with `set -u` rejects expanding an empty indexed array.
    if [[ $EXTRA_ARGS_COUNT -gt 0 ]]; then
        run_cmd+=("${EXTRA_ARGS[@]}")
    fi

    # Run example, capturing stdout+stderr — the [STRACE] markers are on stderr.
    vlog "Running: ${run_cmd[*]}"
    local rc=0
    if [[ "$mode" == "serial" ]]; then
        vlog "Environment: SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE=1"
        SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE=1 "${run_cmd[@]}" > "$fw_stdout_file" 2>&1 || rc=$?
    else
        "${run_cmd[@]}" > "$fw_stdout_file" 2>&1 || rc=$?
    fi
    if [[ -n "$VERBOSE_LOG" && -s "$fw_stdout_file" ]]; then
        cat "$fw_stdout_file" >> "$VERBOSE_LOG"
    fi
    if [[ $rc -ne 0 ]]; then
        # The captured output is the only record of why the run failed, and it
        # dies with $fw_stdout_file on return — so echo its tail here rather
        # than leaving the exit code to stand in for the cause.
        echo "  FAILED: exit $rc — ${run_cmd[*]}"
        if [[ -s "$fw_stdout_file" ]]; then
            echo "  ---- last $FAILURE_TAIL_LINES lines of output ----"
            tail -n "$FAILURE_TAIL_LINES" "$fw_stdout_file" | sed 's/^/  | /'
            echo "  ---- end of output ----"
            [[ -z "$VERBOSE_LOG" ]] && echo "  (re-run with -v to keep the full output)"
        else
            echo "  (the run produced no output)"
        fi
        vlog "FAILED: exit code $rc"
        ((FAIL++)) || true
        return
    fi

    local timing_output
    local parse_rc=0
    timing_output=$(parse_timing "$fw_stdout_file") || parse_rc=$?
    echo "$timing_output"

    if [[ $parse_rc -ne 0 ]]; then
        ((FAIL++)) || true
        return
    fi
    ((PASS++)) || true

    # Extract averages for summary table (from `strace_timing --rounds-table`,
    # which prints "Avg <Metric>: N us").
    local label="$example"
    [[ -n "$case_name" ]] && label="$example ($case_name)"

    local avg_line
    avg_line=$(echo "$timing_output" | grep -E 'Avg (Host|Device|Orch|Sched|Effective):' | grep -v 'Trimmed' | head -1 || true)
    local avg_host="-" avg_device="-" avg_effective="-" avg_sched="-" avg_orch="-"
    if [[ -n "$avg_line" ]]; then
        avg_host=$(     echo "$avg_line" | grep -oE 'Avg Host: [0-9.]+'      || true); avg_host=${avg_host##* }
        avg_device=$(   echo "$avg_line" | grep -oE 'Avg Device: [0-9.]+'    || true); avg_device=${avg_device##* }
        avg_effective=$(echo "$avg_line" | grep -oE 'Avg Effective: [0-9.]+' || true); avg_effective=${avg_effective##* }
        avg_sched=$(    echo "$avg_line" | grep -oE 'Avg Sched: [0-9.]+'     || true); avg_sched=${avg_sched##* }
        avg_orch=$(     echo "$avg_line" | grep -oE 'Avg Orch: [0-9.]+'      || true); avg_orch=${avg_orch##* }
        [[ -z "$avg_host" ]]      && avg_host="-"
        [[ -z "$avg_device" ]]    && avg_device="-"
        [[ -z "$avg_effective" ]] && avg_effective="-"
        [[ -z "$avg_sched" ]]     && avg_sched="-"
        [[ -z "$avg_orch" ]]      && avg_orch="-"
    fi

    SUMMARY_NAMES+=("$label")
    SUMMARY_MODE+=("$mode")
    SUMMARY_HOST+=("$avg_host")
    SUMMARY_DEVICE+=("$avg_device")
    SUMMARY_EFFECTIVE+=("$avg_effective")
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
SUMMARY_MODE=()
SUMMARY_HOST=()
SUMMARY_DEVICE=()
SUMMARY_EFFECTIVE=()
SUMMARY_SCHED=()
SUMMARY_ORCH=()

find_example_dir() {
    local example="$1" candidate dir
    for dir in "${EXAMPLES_DIRS[@]}"; do
        candidate="$dir/$example"
        if [[ -d "$candidate" ]] && ls "$candidate"/test_*.py 1>/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    for dir in "${EXAMPLES_DIRS[@]}"; do
        candidate="$dir/$example"
        if [[ -f "$candidate/golden.py" && -d "$candidate/kernels" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

for example_case in "${EXAMPLE_CASES[@]}"; do
    example="${example_case%%=*}"
    if ! find_example_dir "$example" >/dev/null; then
        echo "ERROR: benchmark workload '$example' not found for $ARCH + $RUNTIME."
        exit 1
    fi
done

echo ""
echo "Runtime: $RUNTIME"

for example_case in "${EXAMPLE_CASES[@]}"; do
    example="${example_case%%=*}"
    case_list="${example_case#*=}"

    # tests/st/ is searched before examples/ since benchmarks use production-scale cases.
    EXAMPLE_DIR=$(find_example_dir "$example")

    echo ""
    echo "================================================================"
    echo "  $example"
    echo "================================================================"

    if [[ -z "${case_list:-}" ]]; then
        run_bench "$example" "$EXAMPLE_DIR" "" "$DEFAULT_MODE"
        if [[ $SERIAL_ORCH_SCHED -eq 1 ]]; then
            run_bench "$example" "$EXAMPLE_DIR" "" "serial"
        fi
    else
        IFS=',' read -ra cases <<< "$case_list"
        for c in "${cases[@]}"; do
            run_bench "$example" "$EXAMPLE_DIR" "$c" "$DEFAULT_MODE"
            if [[ $SERIAL_ORCH_SCHED -eq 1 ]]; then
                run_bench "$example" "$EXAMPLE_DIR" "$c" "serial"
            fi
        done
    fi
done

# ---------------------------------------------------------------------------
# Performance Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    # Show only columns that have at least one non-"-" value
    _has_host=0; _has_device=0; _has_orch=0; _has_sched=0; _has_effective=0
    for _i in "${!SUMMARY_NAMES[@]}"; do
        [[ "${SUMMARY_HOST[$_i]}"      != "-" ]] && _has_host=1
        [[ "${SUMMARY_DEVICE[$_i]}"    != "-" ]] && _has_device=1
        [[ "${SUMMARY_ORCH[$_i]}"      != "-" ]] && _has_orch=1
        [[ "${SUMMARY_SCHED[$_i]}"     != "-" ]] && _has_sched=1
        [[ "${SUMMARY_EFFECTIVE[$_i]}" != "-" ]] && _has_effective=1
    done

    echo ""
    echo "================================================================"
    echo "  Performance Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    _hdr=$(printf "  %-40s  %-8s" "Example" "Mode")
    _sep=$(printf "  %-40s  %-8s" "----------------------------------------" "--------")
    if [[ $_has_host      -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Host (us)");      _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_device    -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Device (us)");    _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_effective -eq 1 ]]; then _hdr=$(printf "%s  %14s" "$_hdr" "Effective (us)"); _sep=$(printf "%s  %14s" "$_sep" "--------------"); fi
    if [[ $_has_orch      -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Orch (us)");      _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_sched     -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Sched (us)");     _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    echo "$_hdr"
    echo "$_sep"

    for _i in "${!SUMMARY_NAMES[@]}"; do
        _row=$(printf "  %-40s" "${SUMMARY_NAMES[$_i]}")
        _row=$(printf "%s  %-8s" "$_row" "${SUMMARY_MODE[$_i]}")
        if [[ $_has_host      -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_HOST[$_i]}");      fi
        if [[ $_has_device    -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_DEVICE[$_i]}");    fi
        if [[ $_has_effective -eq 1 ]]; then _row=$(printf "%s  %14s" "$_row" "${SUMMARY_EFFECTIVE[$_i]}"); fi
        if [[ $_has_orch      -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_ORCH[$_i]}");      fi
        if [[ $_has_sched     -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_SCHED[$_i]}");     fi
        echo "$_row"
    done

    if [[ $SERIAL_ORCH_SCHED -eq 1 ]]; then
        echo ""
        echo "================================================================"
        echo "  Serial vs Parallel Delta ($RUNTIME)"
        echo "================================================================"
        echo ""
        printf "  %-40s  %-8s  %12s  %12s  %12s  %12s\n" \
            "Example" "Metric" "Parallel" "Serial" "Delta" "Change (%)"
        printf "  %-40s  %-8s  %12s  %12s  %12s  %12s\n" \
            "----------------------------------------" "--------" "------------" "------------" "------------" "------------"

        print_delta_row() {
            local name="$1" metric="$2" base="$3" serial="$4"
            [[ "$base" == "-" || "$serial" == "-" ]] && return
            awk -v name="$name" -v metric="$metric" -v base="$base" -v serial="$serial" '
                BEGIN {
                    delta = serial - base
                    change = (base == 0) ? 0 : delta * 100.0 / base
                    printf "  %-40s  %-8s  %12.1f  %12.1f  %+12.1f  %+12.2f\n", name, metric, base, serial, delta, change
                }'
        }

        for _i in "${!SUMMARY_NAMES[@]}"; do
            [[ "${SUMMARY_MODE[$_i]}" == "serial" ]] || continue
            _base_idx=""
            for _j in "${!SUMMARY_NAMES[@]}"; do
                if [[ "${SUMMARY_NAMES[$_j]}" == "${SUMMARY_NAMES[$_i]}" && "${SUMMARY_MODE[$_j]}" == "parallel" ]]; then
                    _base_idx="$_j"
                    break
                fi
            done
            [[ -n "$_base_idx" ]] || continue
            print_delta_row "${SUMMARY_NAMES[$_i]}" "Host"   "${SUMMARY_HOST[$_base_idx]}"   "${SUMMARY_HOST[$_i]}"
            print_delta_row "${SUMMARY_NAMES[$_i]}" "Device" "${SUMMARY_DEVICE[$_base_idx]}" "${SUMMARY_DEVICE[$_i]}"
            print_delta_row "${SUMMARY_NAMES[$_i]}" "Effective" "${SUMMARY_EFFECTIVE[$_base_idx]}" "${SUMMARY_EFFECTIVE[$_i]}"
            print_delta_row "${SUMMARY_NAMES[$_i]}" "Orch"      "${SUMMARY_ORCH[$_base_idx]}"      "${SUMMARY_ORCH[$_i]}"
            print_delta_row "${SUMMARY_NAMES[$_i]}" "Sched"     "${SUMMARY_SCHED[$_base_idx]}"     "${SUMMARY_SCHED[$_i]}"
        done
    fi
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
