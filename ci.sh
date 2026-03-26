#!/bin/bash

# Parse arguments
PLATFORM=""
DEVICE_RANGE=""
PARALLEL=false
RUNTIME=""
PTO_ISA_COMMIT=""
TIMEOUT=600  # 10 minutes default
CLONE_PROTOCOL="https"  # Default to HTTPS in CI

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
        --clone-protocol)
            CLONE_PROTOCOL="$2"
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

# Discover all runtimes from src/{arch}/runtime/
ALL_RUNTIMES=()
for arch_dir in src/*/runtime; do
    if [[ -d "$arch_dir" ]]; then
        for rt_dir in "$arch_dir"/*; do
            if [[ -d "$rt_dir" && -f "$rt_dir/build_config.py" ]]; then
                rt_name=$(basename "$rt_dir")
                # Add to list if not already present
                if [[ ! " ${ALL_RUNTIMES[*]} " =~ " ${rt_name} " ]]; then
                    ALL_RUNTIMES+=("$rt_name")
                fi
            fi
        done
    fi
done

# Validate runtime if specified
if [[ -n "$RUNTIME" ]]; then
    RUNTIME_VALID=false
    for r in "${ALL_RUNTIMES[@]}"; do
        if [[ "$RUNTIME" == "$r" ]]; then
            RUNTIME_VALID=true
            break
        fi
    done
    if [[ "$RUNTIME_VALID" == "false" ]]; then
        echo "Unknown runtime: $RUNTIME"
        echo "Valid runtimes: ${ALL_RUNTIMES[*]}"
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

# Dynamically discover platform-to-runtime mapping from src/ directory structure
# Use parallel arrays instead of associative arrays for bash 3 compatibility
PLATFORM_KEYS=()
PLATFORM_VALUES=()

for arch_dir in src/*/; do
    [[ -d "$arch_dir" ]] || continue
    arch=$(basename "$arch_dir")

    # Get runtimes for this architecture
    runtimes=()
    rt_dir="${arch_dir}runtime"
    if [[ -d "$rt_dir" ]]; then
        for rt in "$rt_dir"/*; do
            if [[ -d "$rt" && -f "$rt/build_config.py" ]]; then
                runtimes+=($(basename "$rt"))
            fi
        done
    fi

    # Sort runtimes
    IFS=$'\n' runtimes=($(sort <<<"${runtimes[*]}"))
    unset IFS
    runtime_str="${runtimes[*]}"

    # Add platforms if they exist
    if [[ -d "${arch_dir}platform/onboard" ]]; then
        PLATFORM_KEYS+=("$arch")
        PLATFORM_VALUES+=("$runtime_str")
    fi
    if [[ -d "${arch_dir}platform/sim" ]]; then
        PLATFORM_KEYS+=("${arch}sim")
        PLATFORM_VALUES+=("$runtime_str")
    fi
done

# Helper function to get platform runtimes (bash 3 compatible)
get_platform_runtimes() {
    local platform="$1"
    local i
    for i in "${!PLATFORM_KEYS[@]}"; do
        if [[ "${PLATFORM_KEYS[$i]}" == "$platform" ]]; then
            echo "${PLATFORM_VALUES[$i]}"
            return 0
        fi
    done
    echo ""
}

# Setup temp directory for logs and results
LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ci_parallel_$$.XXXXXX")
RESULTS_FILE="${LOG_DIR}/results.txt"
touch "$RESULTS_FILE"

cleanup() {
    kill $WATCHDOG_PID 2>/dev/null
    # Kill only our child processes, not the entire process group.
    # kill 0 would kill the GitHub Actions runner on self-hosted machines.
    pkill -TERM -P $$ 2>/dev/null
    rm -rf "$LOG_DIR"
    exit 130
}
trap cleanup INT TERM
trap 'kill $WATCHDOG_PID 2>/dev/null; pkill -TERM -P $$ 2>/dev/null; rm -rf "$LOG_DIR"' EXIT

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
DEVICE_TESTS_DIR="tests/st"

declare -a HW_TASK_NAMES=()
declare -a HW_TASK_DIRS=()
declare -a HW_TASK_PLATS=()
declare -a SIM_TASK_NAMES=()
declare -a SIM_TASK_DIRS=()
declare -a SIM_TASK_PLATS=()

# Discover examples
while IFS= read -r -d '' example_dir; do
    [[ "$example_dir" == *"/scripts" ]] && continue
    kernel_config="${example_dir}/kernels/kernel_config.py"
    golden="${example_dir}/golden.py"
    [[ -f "$kernel_config" && -f "$golden" ]] || continue

    example_name="${example_dir#$EXAMPLES_DIR/}"
    example_arch="${example_name%%/*}"  # Extract arch (a2a3/a5) from path
    example_rest="${example_name#*/}"
    example_runtime="${example_rest%%/*}"  # Extract runtime from path

    # Filter by runtime if specified
    if [[ -n "$RUNTIME" && "$example_runtime" != "$RUNTIME" ]]; then
        continue
    fi

    # Filter by platform's arch and supported runtimes
    if [[ -n "$PLATFORM" ]]; then
        platform_base="${PLATFORM%sim}"
        if [[ "$example_arch" != "$platform_base" ]]; then
            continue  # Skip examples not matching platform arch
        fi
        platform_runtimes="$(get_platform_runtimes "$PLATFORM")"
        if [[ ! " $platform_runtimes " =~ " $example_runtime " ]]; then
            continue  # Skip unsupported runtime for this platform
        fi
    fi

    if [[ -n "$PLATFORM" ]]; then
        if [[ "$PLATFORM" =~ sim$ ]]; then
            SIM_TASK_NAMES+=("example:${example_name}")
            SIM_TASK_DIRS+=("${example_dir}")
            SIM_TASK_PLATS+=("${PLATFORM}")
        else
            HW_TASK_NAMES+=("example:${example_name}")
            HW_TASK_DIRS+=("${example_dir}")
            HW_TASK_PLATS+=("${PLATFORM}")
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        SIM_TASK_NAMES+=("example:${example_name}")
        SIM_TASK_DIRS+=("${example_dir}")
        SIM_TASK_PLATS+=("${example_arch}sim")
    else
        HW_TASK_NAMES+=("example:${example_name}")
        HW_TASK_DIRS+=("${example_dir}")
        HW_TASK_PLATS+=("${example_arch}")
        SIM_TASK_NAMES+=("example:${example_name}")
        SIM_TASK_DIRS+=("${example_dir}")
        SIM_TASK_PLATS+=("${example_arch}sim")
    fi
done < <(find "$EXAMPLES_DIR" -mindepth 1 -type d -print0 | sort -z)

# Discover device tests (hardware only)
if [[ -d "$DEVICE_TESTS_DIR" ]]; then
    RUN_DEVICE_TESTS=false
    [[ -n "$PLATFORM" && ! "$PLATFORM" =~ sim$ ]] && RUN_DEVICE_TESTS=true
    [[ -z "$PLATFORM" && "$OS" == "Linux" ]] && RUN_DEVICE_TESTS=true

    if [[ "$RUN_DEVICE_TESTS" == "true" ]]; then
        while IFS= read -r -d '' test_dir; do
            kernel_config="${test_dir}/kernels/kernel_config.py"
            golden="${test_dir}/golden.py"
            [[ -f "$kernel_config" && -f "$golden" ]] || continue
            test_name="${test_dir#$DEVICE_TESTS_DIR/}"
            test_arch="${test_name%%/*}"  # Extract arch (a2a3/a5) from path
            test_rest="${test_name#*/}"
            test_runtime="${test_rest%%/*}"  # Extract runtime from path

            # Filter by runtime if specified
            if [[ -n "$RUNTIME" && "$test_runtime" != "$RUNTIME" ]]; then
                continue
            fi

            # Filter by platform's arch and supported runtimes
            if [[ -n "$PLATFORM" ]]; then
                platform_base="${PLATFORM%sim}"
                if [[ "$test_arch" != "$platform_base" ]]; then
                    continue  # Skip tests not matching platform arch
                fi
                platform_runtimes="$(get_platform_runtimes "$PLATFORM")"
                if [[ ! " $platform_runtimes " =~ " $test_runtime " ]]; then
                    continue  # Skip unsupported runtime for this platform
                fi
            fi

            HW_TASK_NAMES+=("device_test:${test_name}")
            HW_TASK_DIRS+=("${test_dir}")
            HW_TASK_PLATS+=("${PLATFORM:-${test_arch}}")
        done < <(find "$DEVICE_TESTS_DIR" -mindepth 1 -type d -print0 | sort -z)
    else
        echo "Skipping device tests (hardware platforms only)"
    fi
fi

echo "Discovered ${#HW_TASK_NAMES[@]} hardware tasks, ${#SIM_TASK_NAMES[@]} simulation tasks"

MAX_RETRIES=3

# ---- Unified task runner ----
# Runs a single task and records the result.
# Log naming: ${safe_name}_${platform}_attempt${attempt}.log
# Result format: name|platform|PASS/FAIL|device:X|attempt:N|Xs
run_task() {
    local name="$1" dir="$2" platform="$3" attempt="$4" device_id="$5" print_log_on_fail="${6:-true}"
    local safe_name="${name//[:\/]/_}"
    local task_log="${LOG_DIR}/${safe_name}_${platform}_attempt${attempt}.log"
    local start_time=$SECONDS

    local -a cmd
    cmd=(python examples/scripts/run_example.py
        -k "${dir}/kernels" -g "${dir}/golden.py"
        -p "$platform" --clone-protocol "$CLONE_PROTOCOL" "${commit_flag[@]}")
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
        if [[ "$print_log_on_fail" == "true" ]]; then
            echo "--- LOG: $name (attempt $attempt) ---"
            cat "$task_log"
            echo "--- END ---"
        fi
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
                if run_task "${SIM_TASK_NAMES[$idx]}" "${SIM_TASK_DIRS[$idx]}" "${SIM_TASK_PLATS[$idx]}" "$attempt"; then
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
            if run_task "${SIM_TASK_NAMES[$idx]}" "${SIM_TASK_DIRS[$idx]}" "${SIM_TASK_PLATS[$idx]}" "$attempt"; then
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

                if run_task "${HW_TASK_NAMES[$idx]}" "${HW_TASK_DIRS[$idx]}" "${HW_TASK_PLATS[$idx]}" "$attempt" "$device_id" "false"; then
                    echo "${idx}|PASS" >> "$hw_marker"
                else
                    next=$((attempt + 1))
                    if [[ $next -lt $MAX_RETRIES ]]; then
                        flock "$lock" bash -c "echo '${idx}:${next}' >> \"$queue\""
                    else
                        echo "${idx}|FAIL" >> "$hw_marker"
                        local safe_name="${HW_TASK_NAMES[$idx]//[:\/]/_}"
                        local last_log="${LOG_DIR}/${safe_name}_${HW_TASK_PLATS[$idx]}_attempt${attempt}.log"
                        echo "--- LOG: ${HW_TASK_NAMES[$idx]} (attempt $attempt) ---"
                        cat "$last_log"
                        echo "--- END ---"
                        echo "[${HW_TASK_PLATS[$idx]}:dev${device_id}] Device quarantined after exhausting retries"
                        break
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
# Use parallel arrays for bash 3 compatibility
FINAL_KEYS=()
FINAL_RESULTS=()
FINAL_DISPLAY=()
FINAL_PLATFORM=()
FINAL_DEVICE=()
FINAL_ATTEMPT=()
FINAL_TIMING=()
TASK_ORDER=()

# Helper functions for key-value operations (bash 3 compatible)
find_key_index() {
    local key="$1"
    local i
    for i in "${!FINAL_KEYS[@]}"; do
        if [[ "${FINAL_KEYS[$i]}" == "$key" ]]; then
            echo "$i"
            return 0
        fi
    done
    echo ""
}

set_result() {
    local key="$1"
    local result="$2"
    local display="$3"
    local platform="$4"
    local device="$5"
    local attempt="$6"
    local timing="$7"

    local idx
    idx=$(find_key_index "$key")

    if [[ -z "$idx" ]]; then
        # New key
        FINAL_KEYS+=("$key")
        FINAL_RESULTS+=("$result")
        FINAL_DISPLAY+=("$display")
        FINAL_PLATFORM+=("$platform")
        FINAL_DEVICE+=("$device")
        FINAL_ATTEMPT+=("$attempt")
        FINAL_TIMING+=("$timing")
        TASK_ORDER+=("$key")
    else
        # Update existing key
        FINAL_RESULTS[$idx]="$result"
        FINAL_DISPLAY[$idx]="$display"
        FINAL_PLATFORM[$idx]="$platform"
        FINAL_DEVICE[$idx]="$device"
        FINAL_ATTEMPT[$idx]="$attempt"
        FINAL_TIMING[$idx]="$timing"
    fi
}

while IFS='|' read -r task_name platform result extra1 extra2 timing; do
    key="${task_name}|${platform}"
    device="${extra1#device:}"
    attempt="${extra2#attempt:}"
    set_result "$key" "$result" "$task_name" "$platform" "$device" "$attempt" "$timing"
done < "$RESULTS_FILE"

FAIL_COUNT=0
PASS_COUNT=0
FAIL_KEYS=()
for i in "${!TASK_ORDER[@]}"; do
    result="${FINAL_RESULTS[$i]}"
    if [[ "$result" == "FAIL" ]]; then
        ((FAIL_COUNT++))
        FAIL_KEYS+=("$i")
    else
        ((PASS_COUNT++))
    fi
done

# Print failure logs first (long output goes here, before the summary table)
for i in "${FAIL_KEYS[@]}"; do
    key="${TASK_ORDER[$i]}"
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
for i in "${!TASK_ORDER[@]}"; do
    task_name="${FINAL_DISPLAY[$i]}"
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

for i in "${!TASK_ORDER[@]}"; do
    task_name="${FINAL_DISPLAY[$i]}"
    result="${FINAL_RESULTS[$i]}"

    if [[ ${#task_name} -gt $TASK_COL_WIDTH ]]; then
        task_display="${task_name:0:$((TASK_COL_WIDTH - 3))}..."
    else
        task_display="$task_name"
    fi

    platform="${FINAL_PLATFORM[$i]}"
    device="${FINAL_DEVICE[$i]}"
    attempt=$(( FINAL_ATTEMPT[$i] + 1 ))
    timing="${FINAL_TIMING[$i]}"

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
