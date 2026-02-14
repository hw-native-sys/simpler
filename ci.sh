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

# Run a single HW task with retry across different devices (max 3 attempts).
# Writes final result to RESULTS_FILE. Each attempt logged separately.
# Usage: run_hw_task_with_retry <name> <dir> [initial_device_id]
run_hw_task_with_retry() {
    local name="$1"
    local dir="$2"
    local initial_device="${3:-${DEVICES[0]}}"
    local safe_name="${name//[:\/]/_}"
    local tried_devices=()
    local failed_devices=()

    for attempt in $(seq 1 $MAX_RETRIES); do
        # First attempt uses the assigned device, retries pick next untried
        local device_id=""
        if [[ $attempt -eq 1 ]]; then
            device_id="$initial_device"
        else
            for dev in "${DEVICES[@]}"; do
                local already_tried=false
                for tried in "${tried_devices[@]}"; do
                    [[ "$tried" == "$dev" ]] && { already_tried=true; break; }
                done
                if [[ "$already_tried" == "false" ]]; then
                    device_id="$dev"
                    break
                fi
            done
        fi

        # No untried device left
        if [[ -z "$device_id" ]]; then
            echo "${name}:a2a3|FAIL|failed_on:${failed_devices[*]}|no_device_left" >> "$RESULTS_FILE"
            return 1
        fi

        tried_devices+=("$device_id")
        local task_log="${LOG_DIR}/${safe_name}_hw_attempt${attempt}.log"

        {
            echo "========================================"
            echo "[Device $device_id] Running: $name (attempt $attempt/$MAX_RETRIES)"
            echo "========================================"
            python examples/scripts/run_example.py \
                -k "${dir}/kernels" -g "${dir}/golden.py" \
                -p a2a3 -d "$device_id"
        } > "$task_log" 2>&1
        local rc=$?

        if [[ $rc -eq 0 ]]; then
            echo "${name}:a2a3|PASS|device:${device_id}|attempt:${attempt}" >> "$RESULTS_FILE"
            return 0
        else
            failed_devices+=("$device_id")
            echo "[Retry] $name failed on device $device_id (attempt $attempt/$MAX_RETRIES)" >&2
        fi
    done

    # All retries exhausted
    echo "${name}:a2a3|FAIL|failed_on:${failed_devices[*]}|attempts:${MAX_RETRIES}" >> "$RESULTS_FILE"
    return 1
}

# ---- Sequential mode ----
if [[ "$PARALLEL" == "false" ]]; then
    for i in "${!HW_TASK_NAMES[@]}"; do
        run_hw_task_with_retry "${HW_TASK_NAMES[$i]}" "${HW_TASK_DIRS[$i]}"
    done
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
    declare -a WORKER_PIDS=()

    # Launch sim tasks in parallel (no device constraint)
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
        WORKER_PIDS+=($!)
    done

    # Launch HW tasks with round-robin device assignment
    # Each task runs with retry logic: on failure, retries on a different device
    for d in $(seq 0 $((NUM_DEVICES - 1))); do
        device_id="${DEVICES[$d]}"

        # Collect task indices for this device slot
        slot_indices=()
        for i in "${!HW_TASK_NAMES[@]}"; do
            if [[ $((i % NUM_DEVICES)) -eq $d ]]; then
                slot_indices+=("$i")
            fi
        done
        [[ ${#slot_indices[@]} -eq 0 ]] && continue

        worker_log="${LOG_DIR}/device_${device_id}_worker.log"

        (
            for idx in "${slot_indices[@]}"; do
                run_hw_task_with_retry "${HW_TASK_NAMES[$idx]}" "${HW_TASK_DIRS[$idx]}" "$device_id"
            done
        ) > "$worker_log" 2>&1 &
        WORKER_PIDS+=($!)
    done

    # Wait for all workers
    for pid in "${WORKER_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
fi

# ---- Print summary ----
echo ""
echo "========================================"
echo "          CI RESULTS SUMMARY"
echo "========================================"
printf "%-55s %s\n" "TASK" "RESULT"
printf "%-55s %s\n" "----" "------"

FAIL_COUNT=0
PASS_COUNT=0
while IFS='|' read -r task_name result extra1 extra2; do
    if [[ "$result" == "FAIL" ]]; then
        printf "%-55s \033[31mFAIL\033[0m  (%s)\n" "$task_name" "${extra1:+$extra1 }${extra2}"
        ((FAIL_COUNT++))
        # Print all attempt logs inline
        safe_name="${task_name//[:\/]/_}"
        for attempt_log in "${LOG_DIR}/${safe_name}_hw_attempt"*.log "${LOG_DIR}/${safe_name}_sim.log"; do
            if [[ -f "$attempt_log" ]]; then
                echo "--- LOG: $(basename "$attempt_log") ---"
                cat "$attempt_log"
                echo "--- END ---"
                echo ""
            fi
        done
    else
        local_info=""
        [[ -n "$extra1" ]] && local_info=" ($extra1, $extra2)"
        printf "%-55s \033[32mPASS\033[0m%s\n" "$task_name" "$local_info"
        ((PASS_COUNT++))
    fi
done < "$RESULTS_FILE"

echo "========================================"
echo "Total: $((PASS_COUNT + FAIL_COUNT))  Passed: $PASS_COUNT  Failed: $FAIL_COUNT"
echo "========================================"

if [[ $FAIL_COUNT -gt 0 || $OVERALL_EXIT -ne 0 ]]; then
    exit 1
fi
echo "All tests passed!"
