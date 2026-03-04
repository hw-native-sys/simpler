#!/bin/bash

# Parse arguments
PLATFORM=""
DEVICE_RANGE=""
PARALLEL=false
RUNTIME=""
PTO_ISA_COMMIT=""
TIMEOUT=600  # 10 minutes default

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_RANGE="$2"
            shift 2
            ;;
        -r|--runtime)
            RUNTIME="$2"
            shift 2
            ;;
        -c|--pto-isa-commit)
            PTO_ISA_COMMIT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate runtime if specified
if [[ -n "$RUNTIME" ]]; then
    VALID_RUNTIMES=("host_build_graph" "aicpu_build_graph" "tensormap_and_ringbuffer")
    RUNTIME_VALID=false
    for r in "${VALID_RUNTIMES[@]}"; do
        if [[ "$RUNTIME" == "$r" ]]; then
            RUNTIME_VALID=true
            break
        fi
    done
    if [[ "$RUNTIME_VALID" == "false" ]]; then
        echo "Unknown runtime: $RUNTIME"
        echo "Valid runtimes: ${VALID_RUNTIMES[*]}"
        exit 1
    fi
fi

# Parse device range (e.g., "5-8" or "5")
if [[ "$DEVICE_RANGE" == *-* ]]; then
    IFS='-' read -r DEV_START DEV_END <<< "$DEVICE_RANGE"
    DEVICES=()
    for ((i=DEV_START; i<=DEV_END; i++)); do
        DEVICES+=("$i")
    done
else
    DEVICES=("${DEVICE_RANGE:-0}")
fi
NUM_DEVICES=${#DEVICES[@]}

OS=$(uname -s)
echo "Running tests on $OS..."

OVERALL_EXIT=0

# Run pytest synchronously first
if [[ -d "tests" && "$OS" == "Linux" && "$PLATFORM" != "a2a3sim" ]]; then
    echo "Running pytest tests..."
    if ! pytest tests -v; then
        echo "PYTEST FAILED"
        OVERALL_EXIT=1
    fi
fi

# Setup temp directory for logs and results
LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ci_parallel_$$.XXXXXX")
RESULTS_FILE="${LOG_DIR}/results.txt"
touch "$RESULTS_FILE"

cleanup() {
    kill $WATCHDOG_PID 2>/dev/null
    kill 0 2>/dev/null
    rm -rf "$LOG_DIR"
    exit 130
}
trap cleanup INT TERM
trap 'kill $WATCHDOG_PID 2>/dev/null; rm -rf "$LOG_DIR"' EXIT

# Watchdog: abort CI if it exceeds the timeout
(
    sleep "$TIMEOUT"
    echo ""
    echo "========================================"
    echo "[CI] TIMEOUT: exceeded ${TIMEOUT}s ($(( TIMEOUT / 60 ))min) limit, aborting"
    echo "========================================"
    kill -TERM $$ 2>/dev/null
) &
WATCHDOG_PID=$!

# commit_flag starts empty (try latest PTO-ISA first).
# If -c is given AND a test fails, pin_pto_isa_on_failure sets commit_flag.
commit_flag=()

# Pin PTO-ISA to the specified commit on first failure.
# On first failure: cleans cached clone, sets commit_flag, returns 0 (caller retries).
# On subsequent failures (already pinned): returns 1 (real failure).
pin_pto_isa_on_failure() {
    if [[ -z "$PTO_ISA_COMMIT" ]]; then
        return 1  # No fallback commit configured
    fi
    if [[ ${#commit_flag[@]} -gt 0 ]]; then
        return 1  # Already pinned, real failure
    fi
    echo "[CI] First failure detected, pinning PTO-ISA to commit $PTO_ISA_COMMIT"
    rm -rf examples/scripts/_deps/pto-isa
    commit_flag=(-c "$PTO_ISA_COMMIT")
    return 0  # Pinned, caller should retry
}

# ---- Discover all tasks ----
EXAMPLES_DIR="examples"
DEVICE_TESTS_DIR="tests/device_tests"

declare -a HW_TASK_NAMES=()
declare -a HW_TASK_DIRS=()
declare -a SIM_TASK_NAMES=()
declare -a SIM_TASK_DIRS=()

# Discover examples
while IFS= read -r -d '' example_dir; do
    [[ "$example_dir" == *"/scripts" ]] && continue
    kernel_config="${example_dir}/kernels/kernel_config.py"
    golden="${example_dir}/golden.py"
    [[ -f "$kernel_config" && -f "$golden" ]] || continue

    example_name="${example_dir#$EXAMPLES_DIR/}"

    # Filter by runtime if specified
    if [[ -n "$RUNTIME" && "$example_name" != "$RUNTIME"/* ]]; then
        continue
    fi

    if [[ -n "$PLATFORM" ]]; then
        if [[ "$PLATFORM" == "a2a3" ]]; then
            HW_TASK_NAMES+=("example:${example_name}")
            HW_TASK_DIRS+=("${example_dir}")
        else
            SIM_TASK_NAMES+=("example:${example_name}")
            SIM_TASK_DIRS+=("${example_dir}")
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        SIM_TASK_NAMES+=("example:${example_name}")
        SIM_TASK_DIRS+=("${example_dir}")
    else
        HW_TASK_NAMES+=("example:${example_name}")
        HW_TASK_DIRS+=("${example_dir}")
        SIM_TASK_NAMES+=("example:${example_name}")
        SIM_TASK_DIRS+=("${example_dir}")
    fi
done < <(find "$EXAMPLES_DIR" -mindepth 1 -type d -print0 | sort -z)

# Discover device tests (hardware only)
if [[ -d "$DEVICE_TESTS_DIR" ]]; then
    RUN_DEVICE_TESTS=false
    [[ "$PLATFORM" == "a2a3" ]] && RUN_DEVICE_TESTS=true
    [[ -z "$PLATFORM" && "$OS" == "Linux" ]] && RUN_DEVICE_TESTS=true

    if [[ "$RUN_DEVICE_TESTS" == "true" ]]; then
        while IFS= read -r -d '' test_dir; do
            kernel_config="${test_dir}/kernels/kernel_config.py"
            golden="${test_dir}/golden.py"
            [[ -f "$kernel_config" && -f "$golden" ]] || continue
            test_name="${test_dir#$DEVICE_TESTS_DIR/}"
            # Filter by runtime if specified
            if [[ -n "$RUNTIME" && "$test_name" != "$RUNTIME"/* ]]; then
                continue
            fi
            HW_TASK_NAMES+=("device_test:${test_name}")
            HW_TASK_DIRS+=("${test_dir}")
        done < <(find "$DEVICE_TESTS_DIR" -mindepth 1 -type d -print0 | sort -z)
    else
        echo "Skipping device tests (a2a3 hardware only)"
    fi
fi

echo "Discovered ${#HW_TASK_NAMES[@]} hardware tasks, ${#SIM_TASK_NAMES[@]} simulation tasks"

MAX_RETRIES=3

# ---- Unified task runner ----
# Runs a single task and records the result.
# Log naming: ${safe_name}_${platform}_attempt${attempt}.log
# Result format: name|platform|PASS/FAIL|device:X|attempt:N|Xs
run_task() {
    local name="$1" dir="$2" platform="$3" attempt="$4" device_id="$5"
    local safe_name="${name//[:\/]/_}"
    local task_log="${LOG_DIR}/${safe_name}_${platform}_attempt${attempt}.log"
    local start_time=$SECONDS

    local -a cmd
    cmd=(python examples/scripts/run_example.py
        -k "${dir}/kernels" -g "${dir}/golden.py"
        -p "$platform" "${commit_flag[@]}")
    [[ -n "$device_id" ]] && cmd+=(-d "$device_id")

    # Progress to stdout (not captured in log)
    echo "[${platform}${device_id:+:dev${device_id}}] Running: $name (attempt $attempt)"

    # Command output to log file only
    "${cmd[@]}" > "$task_log" 2>&1
    local rc=$?
    local elapsed=$(( SECONDS - start_time ))

    local status
    if [[ $rc -eq 0 ]]; then
        status="PASS"
        echo "[${platform}${device_id:+:dev${device_id}}] PASS: $name (${elapsed}s)"
    else
        status="FAIL"
        echo "[${platform}${device_id:+:dev${device_id}}] FAIL: $name (${elapsed}s)"
    fi
    echo "${name}|${platform}|${status}|device:${device_id:-sim}|attempt:${attempt}|${elapsed}s" \
        >> "$RESULTS_FILE"
    return $rc
}

# ---- SIM executor ----
# run_sim_tasks <attempt> <idx1> <idx2> ...
# Sets SIM_FAILURES to array of failed indices after return.
run_sim_tasks() {
    local attempt="$1"; shift
    local indices=("$@")
    local sim_marker="${LOG_DIR}/sim_results_$$.txt"
    local run_parallel="$PARALLEL"
    > "$sim_marker"

    # Pinned retries share one _deps/pto-isa clone path; parallel clone races can fail.
    if [[ "$attempt" -gt 0 && ${#commit_flag[@]} -gt 0 && "$run_parallel" == "true" ]]; then
        echo "[CI] SIM retry uses pinned PTO-ISA; running retries sequentially to avoid clone races"
        run_parallel=false
    fi

    if [[ "$run_parallel" == "true" ]]; then
        local -a pids=()
        for idx in "${indices[@]}"; do
            (
                if run_task "${SIM_TASK_NAMES[$idx]}" "${SIM_TASK_DIRS[$idx]}" a2a3sim "$attempt"; then
                    echo "${idx}|PASS" >> "$sim_marker"
                else
                    echo "${idx}|FAIL" >> "$sim_marker"
                fi
            ) &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    else
        for idx in "${indices[@]}"; do
            if run_task "${SIM_TASK_NAMES[$idx]}" "${SIM_TASK_DIRS[$idx]}" a2a3sim "$attempt"; then
                echo "${idx}|PASS" >> "$sim_marker"
            else
                echo "${idx}|FAIL" >> "$sim_marker"
            fi
        done
    fi

    SIM_FAILURES=()
    while IFS='|' read -r idx result; do
        [[ "$result" == "FAIL" ]] && SIM_FAILURES+=("$idx")
    done < "$sim_marker"
}

# ---- HW executor: continuous shared queue ----
# run_hw_tasks <idx1> <idx2> ...
# Workers pop "idx:attempt" entries, run, re-enqueue on failure.
# Sets HW_FAILURES to array of indices that exhausted MAX_RETRIES after return.
run_hw_tasks() {
    local indices=("$@")
    local queue="${LOG_DIR}/hw_queue_$$.txt"
    local lock="${LOG_DIR}/hw_queue_$$.lock"
    local hw_marker="${LOG_DIR}/hw_results_$$.txt"
    > "$queue"
    > "$hw_marker"

    # Seed queue
    for idx in "${indices[@]}"; do
        echo "${idx}:0" >> "$queue"
    done

    # Launch one worker per device
    local -a pids=()
    for d in $(seq 0 $((NUM_DEVICES - 1))); do
        local device_id="${DEVICES[$d]}"
        (
            while true; do
                # Atomically pop the next entry from the queue
                entry=$(flock "$lock" bash -c "
                    entry=\$(head -n1 \"$queue\" 2>/dev/null)
                    if [[ -z \"\$entry\" ]]; then exit 1; fi
                    sed -i '1d' \"$queue\"
                    echo \"\$entry\"
                ") || break

                IFS=':' read -r idx attempt <<< "$entry"

                if run_task "${HW_TASK_NAMES[$idx]}" "${HW_TASK_DIRS[$idx]}" a2a3 "$attempt" "$device_id"; then
                    echo "${idx}|PASS" >> "$hw_marker"
                else
                    next=$((attempt + 1))
                    if [[ $next -lt $MAX_RETRIES ]]; then
                        flock "$lock" bash -c "echo '${idx}:${next}' >> \"$queue\""
                    else
                        echo "${idx}|FAIL" >> "$hw_marker"
                    fi
                fi
            done
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done

    HW_FAILURES=()
    while IFS='|' read -r idx result; do
        [[ "$result" == "FAIL" ]] && HW_FAILURES+=("$idx")
    done < "$hw_marker"
}

# ---- Main flow: two-pass per phase ----

# SIM phase
if [[ ${#SIM_TASK_NAMES[@]} -gt 0 ]]; then
    ALL_SIM=($(seq 0 $((${#SIM_TASK_NAMES[@]} - 1))))
    echo "---- SIM: ${#ALL_SIM[@]} tasks ----"
    run_sim_tasks 0 "${ALL_SIM[@]}"
    if [[ ${#SIM_FAILURES[@]} -gt 0 ]] && pin_pto_isa_on_failure; then
        echo "[CI] Retrying ${#SIM_FAILURES[@]} SIM failures with pinned PTO-ISA"
        run_sim_tasks 1 "${SIM_FAILURES[@]}"
    fi
fi

# HW phase
if [[ ${#HW_TASK_NAMES[@]} -gt 0 ]]; then
    ALL_HW=($(seq 0 $((${#HW_TASK_NAMES[@]} - 1))))
    echo "---- HW: ${#ALL_HW[@]} tasks on ${NUM_DEVICES} devices ----"
    run_hw_tasks "${ALL_HW[@]}"
    if [[ ${#HW_FAILURES[@]} -gt 0 ]] && pin_pto_isa_on_failure; then
        echo "[CI] Retrying ${#HW_FAILURES[@]} HW failures with pinned PTO-ISA"
        run_hw_tasks "${HW_FAILURES[@]}"
    fi
fi

# ---- Print summary ----
# Deduplicate results: a task may have multiple entries (fail then pass on retry).
# Keep the last result per task name+platform — the final outcome.
# Use composite key (task_name|platform) so SIM and HW results don't collide.
declare -A FINAL_RESULTS=()
declare -A FINAL_DISPLAY=()
declare -A FINAL_PLATFORM=()
declare -A FINAL_DEVICE=()
declare -A FINAL_ATTEMPT=()
declare -A FINAL_TIMING=()
declare -a TASK_ORDER=()

while IFS='|' read -r task_name platform result extra1 extra2 timing; do
    key="${task_name}|${platform}"
    if [[ -z "${FINAL_RESULTS[$key]+x}" ]]; then
        TASK_ORDER+=("$key")
    fi
    FINAL_RESULTS["$key"]="$result"
    FINAL_DISPLAY["$key"]="$task_name"
    FINAL_PLATFORM["$key"]="$platform"
    FINAL_DEVICE["$key"]="${extra1#device:}"
    FINAL_ATTEMPT["$key"]="${extra2#attempt:}"
    FINAL_TIMING["$key"]="$timing"
done < "$RESULTS_FILE"

FAIL_COUNT=0
PASS_COUNT=0
declare -a FAIL_KEYS=()
for key in "${TASK_ORDER[@]}"; do
    result="${FINAL_RESULTS[$key]}"
    if [[ "$result" == "FAIL" ]]; then
        ((FAIL_COUNT++))
        FAIL_KEYS+=("$key")
    else
        ((PASS_COUNT++))
    fi
done

# Print failure logs first (long output goes here, before the summary table)
for key in "${FAIL_KEYS[@]}"; do
    IFS='|' read -r task_name platform <<< "$key"
    safe_name="${task_name//[:\/]/_}"
    for attempt_log in "${LOG_DIR}/${safe_name}_${platform}_attempt"*.log; do
        if [[ -f "$attempt_log" ]]; then
            echo "--- LOG: ${task_name} ($(basename "$attempt_log")) ---"
            cat "$attempt_log"
            echo "--- END ---"
            echo ""
        fi
    done
done

# Print clean summary table last so it is always visible
echo ""

if [[ -t 1 ]]; then
    COLOR_RED=$'\033[31m'
    COLOR_GREEN=$'\033[32m'
    COLOR_RESET=$'\033[0m'
else
    COLOR_RED=""
    COLOR_GREEN=""
    COLOR_RESET=""
fi

TASK_COL_WIDTH=4
for key in "${TASK_ORDER[@]}"; do
    task_name="${FINAL_DISPLAY[$key]}"
    if [[ ${#task_name} -gt $TASK_COL_WIDTH ]]; then
        TASK_COL_WIDTH=${#task_name}
    fi
done
if [[ $TASK_COL_WIDTH -lt 40 ]]; then TASK_COL_WIDTH=40; fi
if [[ $TASK_COL_WIDTH -gt 72 ]]; then TASK_COL_WIDTH=72; fi

SUMMARY_TITLE="CI RESULTS SUMMARY"
SUMMARY_HEADER=$(printf "%-*s %-8s %-6s %-7s %-6s %s" \
    "$TASK_COL_WIDTH" "TASK" "PLATFORM" "DEVICE" "ATTEMPT" "TIME" "RESULT")
SUMMARY_WIDTH=${#SUMMARY_HEADER}
if [[ ${#SUMMARY_TITLE} -gt $SUMMARY_WIDTH ]]; then
    SUMMARY_WIDTH=${#SUMMARY_TITLE}
fi
SUMMARY_BORDER=$(printf '%*s' "$SUMMARY_WIDTH" '' | tr ' ' '=')

TITLE_PAD_LEFT=$(( (SUMMARY_WIDTH - ${#SUMMARY_TITLE}) / 2 ))
TITLE_PAD_RIGHT=$(( SUMMARY_WIDTH - ${#SUMMARY_TITLE} - TITLE_PAD_LEFT ))
SUMMARY_TITLE_LINE=$(printf "%*s%s%*s" \
    "$TITLE_PAD_LEFT" "" "$SUMMARY_TITLE" "$TITLE_PAD_RIGHT" "")

echo "$SUMMARY_BORDER"
echo "$SUMMARY_TITLE_LINE"
echo "$SUMMARY_BORDER"

TASK_DIVIDER=$(printf '%*s' "$TASK_COL_WIDTH" '' | tr ' ' '-')
printf "%s\n" "$SUMMARY_HEADER"
printf "%-*s %-8s %-6s %-7s %-6s %s\n" "$TASK_COL_WIDTH" "$TASK_DIVIDER" "--------" "------" "-------" "----" "------"

for key in "${TASK_ORDER[@]}"; do
    task_name="${FINAL_DISPLAY[$key]}"
    result="${FINAL_RESULTS[$key]}"

    if [[ ${#task_name} -gt $TASK_COL_WIDTH ]]; then
        task_display="${task_name:0:$((TASK_COL_WIDTH - 3))}..."
    else
        task_display="$task_name"
    fi

    platform="${FINAL_PLATFORM[$key]}"
    device="${FINAL_DEVICE[$key]}"
    attempt="${FINAL_ATTEMPT[$key]}"
    timing="${FINAL_TIMING[$key]}"

    if [[ "$result" == "FAIL" ]]; then
        printf "%-*s %-8s %-6s %-7s %-6s %sFAIL%s\n" \
            "$TASK_COL_WIDTH" "$task_display" "$platform" "$device" "$attempt" "$timing" \
            "$COLOR_RED" "$COLOR_RESET"
    else
        printf "%-*s %-8s %-6s %-7s %-6s %sPASS%s\n" \
            "$TASK_COL_WIDTH" "$task_display" "$platform" "$device" "$attempt" "$timing" \
            "$COLOR_GREEN" "$COLOR_RESET"
    fi
done

echo "$SUMMARY_BORDER"
echo "Total: $((PASS_COUNT + FAIL_COUNT))  Passed: $PASS_COUNT  Failed: $FAIL_COUNT"
echo "$SUMMARY_BORDER"

if [[ $FAIL_COUNT -gt 0 || $OVERALL_EXIT -ne 0 ]]; then
    exit 1
else
    echo "All tests passed!"
fi
