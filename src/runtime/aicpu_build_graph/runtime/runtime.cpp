/**
 * Runtime Class - Implementation
 *
 * Task dependency management with circular ready queue.
 * Follows patterns from pto_runtime.c for consistency.
 */

#include "runtime.h"

// =============================================================================
// Constructor
// =============================================================================

Runtime::Runtime() {
    // NOTE: host_api is initialized in InitRuntime() (host-only code)
    // because the CApi functions don't exist when compiled for device.

    // Initialize task array (cannot use memset with atomic members)
    for (int i = 0; i < RUNTIME_MAX_TASKS; i++) {
        tasks[i].task_id = 0;
        tasks[i].func_id = 0;
        tasks[i].num_args = 0;
        tasks[i].function_bin_addr = 0;
        tasks[i].core_type = CoreType::AIV;  // Default to AIV
        tasks[i].fanin = 0;
        tasks[i].fanout_count = 0;
        tasks[i].start_time = 0;
        tasks[i].end_time = 0;
        tasks[i].published = 0;
        tasks[i].completed = 0;
        memset(tasks[i].args, 0, sizeof(tasks[i].args));
        memset(tasks[i].fanout, 0, sizeof(tasks[i].fanout));
    }
    next_task_id = 0;
    initial_ready_count = 0;
    worker_count = 0;
    sche_cpu_num = 1;
    tensor_pair_count = 0;
    device_alloc_count = 0;

    orch_argc = 0;
    memset(orch_args, 0, sizeof(orch_args));
    memset(kernel_addrs, 0, sizeof(kernel_addrs));
    aicpu_orch_so_size = 0;
    memset(aicpu_orch_so_storage, 0, sizeof(aicpu_orch_so_storage));
    memset(aicpu_orch_func_name, 0, sizeof(aicpu_orch_func_name));
    strncpy(aicpu_orch_func_name, "build_graph_aicpu", sizeof(aicpu_orch_func_name) - 1);
    build_mode = 1;  // default to concurrent build||schedule
    aicpu_build_api = {};
}

bool Runtime::try_set_aicpu_orch_so(const void* data, size_t size) {
    if (data == nullptr || size == 0) {
        aicpu_orch_so_size = 0;
        return false;
    }
    if (size > sizeof(aicpu_orch_so_storage)) {
        fprintf(stderr,
            "[Runtime] ERROR: AICPU orchestration plugin too large (%zu > %zu)\n",
            size,
            sizeof(aicpu_orch_so_storage));
        aicpu_orch_so_size = 0;
        return false;
    }
    memcpy(aicpu_orch_so_storage, data, size);
    aicpu_orch_so_size = static_cast<uint32_t>(size);
    return true;
}

void Runtime::set_aicpu_orch_so(const void* data, size_t size) { (void)try_set_aicpu_orch_so(data, size); }

const void* Runtime::get_aicpu_orch_so_data() const { return aicpu_orch_so_size > 0 ? aicpu_orch_so_storage : nullptr; }

size_t Runtime::get_aicpu_orch_so_size() const { return static_cast<size_t>(aicpu_orch_so_size); }

// =============================================================================
// Task Management
// =============================================================================

int Runtime::add_task(uint64_t* args, int num_args, int func_id, CoreType core_type) {
    // Check bounds
    if (next_task_id >= RUNTIME_MAX_TASKS) {
        fprintf(stderr, "[Runtime] ERROR: Task table full (max=%d)\n", RUNTIME_MAX_TASKS);
        return -1;
    }

    if (num_args > RUNTIME_MAX_ARGS) {
        fprintf(stderr, "[Runtime] ERROR: Too many args (%d > %d)\n", num_args, RUNTIME_MAX_ARGS);
        return -1;
    }

    // Allocate task
    int task_id = next_task_id++;
    Task* task = &tasks[task_id];

    // Initialize task fields
    task->task_id = task_id;
    task->func_id = func_id;
    task->num_args = num_args;
    if (args && num_args > 0) {
        memcpy(task->args, args, num_args * sizeof(uint64_t));
    }
    task->function_bin_addr = 0;  // Set by AICPU builder for this runtime
    task->core_type = core_type;  // Set core type
    task->fanin = 0;
    task->fanout_count = 0;
    task->published.store(0, std::memory_order_release);
    task->completed.store(0, std::memory_order_release);
    memset(task->fanout, 0, sizeof(task->fanout));

    return task_id;
}

void Runtime::add_successor(int from_task, int to_task) {
    // Validate task IDs
    if (from_task < 0 || from_task >= next_task_id) {
        fprintf(stderr, "[Runtime] ERROR: Invalid from_task ID %d\n", from_task);
        return;
    }

    if (to_task < 0 || to_task >= next_task_id) {
        fprintf(stderr, "[Runtime] ERROR: Invalid to_task ID %d\n", to_task);
        return;
    }

    Task* from = &tasks[from_task];
    Task* to = &tasks[to_task];

    // Add to_task to from_task's fanout
    if (from->fanout_count >= RUNTIME_MAX_FANOUT) {
        fprintf(stderr, "[Runtime] ERROR: Fanout overflow for task %d (max=%d)\n", from_task, RUNTIME_MAX_FANOUT);
        return;
    }

    from->fanout[from->fanout_count++] = to_task;
    to->fanin++;
}

void Runtime::add_successor_conditional(int from_task, int to_task) {
    // Validate task IDs
    if (from_task < 0 || from_task >= next_task_id) {
        fprintf(stderr, "[Runtime] ERROR: Invalid from_task ID %d\n", from_task);
        return;
    }

    if (to_task < 0 || to_task >= next_task_id) {
        fprintf(stderr, "[Runtime] ERROR: Invalid to_task ID %d\n", to_task);
        return;
    }

    Task* from = &tasks[from_task];
    Task* to = &tasks[to_task];

    if (from->fanout_count >= RUNTIME_MAX_FANOUT) {
        fprintf(stderr, "[Runtime] ERROR: Fanout overflow for task %d (max=%d)\n", from_task, RUNTIME_MAX_FANOUT);
        return;
    }

    from->fanout[from->fanout_count++] = to_task;

    // If predecessor is already completed, dependency is already satisfied.
    if (from->completed.load(std::memory_order_acquire) == 0) {
        to->fanin.fetch_add(1, std::memory_order_acq_rel);
    }
}

// =============================================================================
// Query Methods
// =============================================================================

Task* Runtime::get_task(int task_id) {
    if (task_id < 0 || task_id >= next_task_id) {
        return nullptr;
    }
    return &tasks[task_id];
}

int Runtime::get_task_count() const { return next_task_id; }

int Runtime::get_initial_ready_tasks(int* ready_tasks) {
    initial_ready_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        if (tasks[i].fanin == 0) {
            initial_ready_tasks[initial_ready_count] = i;
            if (ready_tasks != nullptr) {
                ready_tasks[initial_ready_count] = i;
            }
            initial_ready_count++;
        }
    }
    return initial_ready_count;
}

// =============================================================================
// Utility Methods
// =============================================================================

void Runtime::print_runtime() const {
    printf(
        "\n===================================================================="
        "============\n");
    printf("[Runtime] Task Runtime Status\n");
    printf(
        "======================================================================"
        "==========\n");
    printf("  Total tasks: %d\n", next_task_id);

    // Print initially ready tasks
    printf("\nInitially Ready Tasks (fanin==0):\n");
    printf(
        "----------------------------------------------------------------------"
        "----------\n");
    printf("  ");
    int ready_count = 0;
    for (int i = 0; i < next_task_id; i++) {
        if (tasks[i].fanin.load() == 0) {
            if (ready_count > 0) printf(", ");
            printf("%d", i);
            ready_count++;
        }
    }
    if (ready_count == 0) {
        printf("(none)");
    }
    printf("\n  Count: %d\n", ready_count);

    printf("\nTask Table:\n");
    printf(
        "----------------------------------------------------------------------"
        "----------\n");

    for (int i = 0; i < next_task_id; i++) {
        const Task* t = &tasks[i];

        printf("  Task %d: func_id=%d, fanin=%d, fanout=%d, args=%d [",
            i,
            t->func_id,
            t->fanin.load(),
            t->fanout_count,
            t->num_args);

        // Print fanout list
        for (int j = 0; j < t->fanout_count; j++) {
            printf("%d%s", t->fanout[j], j < t->fanout_count - 1 ? "," : "");
        }
        printf("]\n");
    }

    printf(
        "======================================================================"
        "==========\n\n");
}

// =============================================================================
// Tensor Pair Management
// =============================================================================

void Runtime::record_tensor_pair(void* host_ptr, void* dev_ptr, size_t size) {
    if (tensor_pair_count >= RUNTIME_MAX_TENSOR_PAIRS) {
        fprintf(stderr, "[Runtime] ERROR: Tensor pairs full (max=%d)\n", RUNTIME_MAX_TENSOR_PAIRS);
        return;
    }
    tensor_pairs[tensor_pair_count].host_ptr = host_ptr;
    tensor_pairs[tensor_pair_count].dev_ptr = dev_ptr;
    tensor_pairs[tensor_pair_count].size = size;
    tensor_pair_count++;
    printf("Recorded tensor pair: host=%p dev=%p size=%zu\n", host_ptr, dev_ptr, size);
}

void Runtime::record_device_alloc(void* dev_ptr) {
    if (dev_ptr == nullptr) {
        return;
    }
    for (int i = 0; i < device_alloc_count; ++i) {
        if (device_allocs[i].dev_ptr == dev_ptr) {
            return;
        }
    }
    if (device_alloc_count >= RUNTIME_MAX_TENSOR_PAIRS) {
        fprintf(stderr, "[Runtime] ERROR: Device allocs full (max=%d)\n", RUNTIME_MAX_TENSOR_PAIRS);
        return;
    }
    device_allocs[device_alloc_count].dev_ptr = dev_ptr;
    device_alloc_count++;
}

TensorPair* Runtime::get_tensor_pairs() { return tensor_pairs; }

int Runtime::get_tensor_pair_count() const { return tensor_pair_count; }

DeviceAlloc* Runtime::get_device_allocs() { return device_allocs; }

int Runtime::get_device_alloc_count() const { return device_alloc_count; }

void Runtime::clear_tensor_pairs() { tensor_pair_count = 0; }

void Runtime::clear_device_allocs() { device_alloc_count = 0; }
