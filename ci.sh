#!/bin/bash

# Parse arguments
PLATFORM=""
DEVICE_RANGE=""
PARALLEL=false

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
    kill 0 2>/dev/null
    rm -rf "$LOG_DIR"
    exit 130
}
trap cleanup INT TERM
trap 'rm -rf "$LOG_DIR"' EXIT

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
            HW_TASK_NAMES+=("device_test:${test_name}")
            HW_TASK_DIRS+=("${test_dir}")
        done < <(find "$DEVICE_TESTS_DIR" -mindepth 1 -type d -print0 | sort -z)
    else
        echo "Skipping device tests (a2a3 hardware only)"
    fi
fi

echo "Discovered ${#HW_TASK_NAMES[@]} hardware tasks, ${#SIM_TASK_NAMES[@]} simulation tasks"

MAX_RETRIES=3

# Run a single HW task on a specific device.
# Writes result to RESULTS_FILE as: name:a2a3|PASS_or_FAIL|device:ID|round:N
# Usage: run_hw_task <name> <dir> <device_id> <round>
run_hw_task() {
    local name="$1"
    local dir="$2"
    local device_id="$3"
    local round="$4"
    local safe_name="${name//[:\/]/_}"
    local task_log="${LOG_DIR}/${safe_name}_hw_round${round}.log"
    local start_time=$SECONDS

    {
        echo "========================================"
        echo "[Device $device_id] Running: $name (round $round/$MAX_RETRIES)"
        echo "========================================"
        python examples/scripts/run_example.py \
            -k "${dir}/kernels" -g "${dir}/golden.py" \
            -p a2a3 -d "$device_id"
    } > "$task_log" 2>&1
    local rc=$?
    local elapsed=$(( SECONDS - start_time ))

    if [[ $rc -eq 0 ]]; then
        echo "${name}:a2a3|PASS|device:${device_id}|round:${round}|${elapsed}s" >> "$RESULTS_FILE"
    else
        echo "${name}:a2a3|FAIL|device:${device_id}|round:${round}|${elapsed}s" >> "$RESULTS_FILE"
    fi
    return $rc
}

# ---- Sequential mode ----
if [[ "$PARALLEL" == "false" ]]; then
    # HW tasks: run with retry across different devices
    for i in "${!HW_TASK_NAMES[@]}"; do
        name="${HW_TASK_NAMES[$i]}"
        dir="${HW_TASK_DIRS[$i]}"
        passed=false
        for round in $(seq 1 $MAX_RETRIES); do
            dev_idx=$(( (round - 1) % NUM_DEVICES ))
            device_id="${DEVICES[$dev_idx]}"
            if run_hw_task "$name" "$dir" "$device_id" "$round"; then
                passed=true
                break
            fi
        done
    done
    # SIM tasks
    for i in "${!SIM_TASK_NAMES[@]}"; do
        name="${SIM_TASK_NAMES[$i]}"
        dir="${SIM_TASK_DIRS[$i]}"
        echo "========================================"
        echo "Running: $name (a2a3sim)"
        echo "========================================"
        if python examples/scripts/run_example.py \
            -k "${dir}/kernels" -g "${dir}/golden.py" \
            -p a2a3sim; then
            echo "${name}:a2a3sim|PASS" >> "$RESULTS_FILE"
        else
            echo "${name}:a2a3sim|FAIL" >> "$RESULTS_FILE"
        fi
    done
else
    # ---- Parallel mode ----
    # Sim tasks: launch all in parallel (no device constraint)
    declare -a SIM_PIDS=()
    for i in "${!SIM_TASK_NAMES[@]}"; do
        name="${SIM_TASK_NAMES[$i]}"
        dir="${SIM_TASK_DIRS[$i]}"
        safe_name="${name//[:\/]/_}"
        log_file="${LOG_DIR}/${safe_name}_sim.log"

        (
            echo "========================================"
            echo "Running: $name (a2a3sim)"
            echo "========================================"
            if python examples/scripts/run_example.py \
                -k "${dir}/kernels" -g "${dir}/golden.py" -p a2a3sim; then
                echo "${name}:a2a3sim|PASS" >> "$RESULTS_FILE"
            else
                echo "${name}:a2a3sim|FAIL" >> "$RESULTS_FILE"
            fi
        ) > "$log_file" 2>&1 &
        SIM_PIDS+=($!)
    done

    # HW tasks: dynamic work-stealing with retry
    # Each device worker atomically grabs tasks from a shared queue.
    # This naturally balances load — faster workers pick up more tasks.

    PENDING_INDICES=()
    for i in "${!HW_TASK_NAMES[@]}"; do
        PENDING_INDICES+=("$i")
    done

    QUEUE_LOCK="${LOG_DIR}/queue.lock"

    for round in $(seq 1 $MAX_RETRIES); do
        [[ ${#PENDING_INDICES[@]} -eq 0 ]] && break

        echo "---- HW round $round/$MAX_RETRIES: ${#PENDING_INDICES[@]} tasks ----"

        TASK_QUEUE="${LOG_DIR}/task_queue_round${round}.txt"
        ROUND_MARKER="${LOG_DIR}/round_${round}_results.txt"
        printf '%s\n' "${PENDING_INDICES[@]}" > "$TASK_QUEUE"
        touch "$ROUND_MARKER"

        # Launch one worker per device; each grabs tasks dynamically
        declare -a HW_PIDS=()
        for d in $(seq 0 $((NUM_DEVICES - 1))); do
            device_id="${DEVICES[$d]}"
            worker_log="${LOG_DIR}/device_${device_id}_round${round}.log"

            (
                while true; do
                    # Atomically pop the next task index from the queue
                    idx=$(flock "$QUEUE_LOCK" bash -c "
                        idx=\$(head -n1 \"$TASK_QUEUE\" 2>/dev/null)
                        if [[ -z \"\$idx\" ]]; then exit 1; fi
                        sed -i '1d' \"$TASK_QUEUE\"
                        echo \"\$idx\"
                    ") || break

                    name="${HW_TASK_NAMES[$idx]}"
                    dir="${HW_TASK_DIRS[$idx]}"
                    run_hw_task "$name" "$dir" "$device_id" "$round"
                    rc=$?
                    if [[ $rc -eq 0 ]]; then
                        echo "${idx}|PASS" >> "$ROUND_MARKER"
                    else
                        echo "${idx}|FAIL" >> "$ROUND_MARKER"
                    fi
                done
            ) > "$worker_log" 2>&1 &
            HW_PIDS+=($!)
        done

        # Wait for this round to finish
        for pid in "${HW_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done

        # Collect failures for next round
        NEXT_PENDING=()
        while IFS='|' read -r idx result; do
            if [[ "$result" == "FAIL" ]]; then
                NEXT_PENDING+=("$idx")
            fi
        done < "$ROUND_MARKER"

        PENDING_INDICES=("${NEXT_PENDING[@]}")
    done

    # Wait for sim tasks too
    for pid in "${SIM_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
fi

# ---- Print summary ----
# Deduplicate results: a task may have multiple entries (fail then pass on retry).
# Keep the last result per task name — the final outcome.
declare -A FINAL_RESULTS=()
declare -A FINAL_EXTRA=()
declare -a TASK_ORDER=()

while IFS='|' read -r task_name result extra1 extra2 timing; do
    if [[ -z "${FINAL_RESULTS[$task_name]+x}" ]]; then
        TASK_ORDER+=("$task_name")
    fi
    FINAL_RESULTS["$task_name"]="$result"
    FINAL_EXTRA["$task_name"]="${extra1:+$extra1, }${extra2:+$extra2, }${timing}"
done < "$RESULTS_FILE"

echo ""
echo "========================================"
echo "          CI RESULTS SUMMARY"
echo "========================================"
printf "%-55s %s\n" "TASK" "RESULT"
printf "%-55s %s\n" "----" "------"

FAIL_COUNT=0
PASS_COUNT=0
for task_name in "${TASK_ORDER[@]}"; do
    result="${FINAL_RESULTS[$task_name]}"
    extra="${FINAL_EXTRA[$task_name]}"
    if [[ "$result" == "FAIL" ]]; then
        printf "%-55s \033[31mFAIL\033[0m  (%s)\n" "$task_name" "$extra"
        ((FAIL_COUNT++))
        # Print all round logs inline
        safe_name="${task_name//[:\/]/_}"
        for round_log in "${LOG_DIR}/${safe_name}_hw_round"*.log "${LOG_DIR}/${safe_name}_sim.log"; do
            if [[ -f "$round_log" ]]; then
                echo "--- LOG: $(basename "$round_log") ---"
                cat "$round_log"
                echo "--- END ---"
                echo ""
            fi
        done
    else
        printf "%-55s \033[32mPASS\033[0m  (%s)\n" "$task_name" "$extra"
        ((PASS_COUNT++))
    fi
done

echo "========================================"
echo "Total: $((PASS_COUNT + FAIL_COUNT))  Passed: $PASS_COUNT  Failed: $FAIL_COUNT"
echo "========================================"

if [[ $FAIL_COUNT -gt 0 || $OVERALL_EXIT -ne 0 ]]; then
    exit 1
fi
echo "All tests passed!"
