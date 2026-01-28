/**
 * Runtime Class - Implementation
 *
 * Task dependency management with circular ready queue.
 * Follows patterns from pto_runtime.c for consistency.
 */

#include "runtime.h"

// External C API functions for device memory operations (defined in pto_runtime_c_api.cpp)
extern "C" {
void* DeviceMalloc_CApi(size_t size);
void DeviceFree_CApi(void* devPtr);
int CopyToDevice_CApi(void* devPtr, const void* hostPtr, size_t size);
int CopyFromDevice_CApi(void* hostPtr, const void* devPtr, size_t size);
}

// =============================================================================
// Constructor
// =============================================================================

Runtime::Runtime() {
    // Initialize task array (cannot use memset with atomic members)
    for (int i = 0; i < RUNTIME_MAX_TASKS; i++) {
        tasks[i].task_id = 0;
        tasks[i].func_id = 0;
        tasks[i].num_args = 0;
        tasks[i].functionBinAddr = 0;
        tasks[i].core_type = 0;
        tasks[i].fanin = 0;
        tasks[i].fanout_count = 0;
        tasks[i].start_time = 0;
        tasks[i].end_time = 0;
        memset(tasks[i].args, 0, sizeof(tasks[i].args));
        memset(tasks[i].fanout, 0, sizeof(tasks[i].fanout));
    }
    next_task_id = 0;
    initial_ready_count = 0;
    worker_count = 0;
    block_dim = 0;
    scheCpuNum = 1;
    tensor_pair_count = 0;
}

// =============================================================================
// Task Management
// =============================================================================

int Runtime::add_task(uint64_t* args, int num_args, int func_id, int core_type) {
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
    task->functionBinAddr = 0;    // Will be set by host before copying to device
    task->core_type = core_type;  // Set core type (0=AIC, 1=AIV)
    task->fanin = 0;
    task->fanout_count = 0;
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
// Device Memory Management
// =============================================================================

void* Runtime::DeviceMalloc(size_t size) {
    return DeviceMalloc_CApi(size);
}

void Runtime::DeviceFree(void* devPtr) {
    DeviceFree_CApi(devPtr);
}

int Runtime::CopyToDevice(void* devPtr, const void* hostPtr, size_t size) {
    return CopyToDevice_CApi(devPtr, hostPtr, size);
}

int Runtime::CopyFromDevice(void* hostPtr, const void* devPtr, size_t size) {
    return CopyFromDevice_CApi(hostPtr, devPtr, size);
}

// =============================================================================
// Tensor Pair Management
// =============================================================================

void Runtime::RecordTensorPair(void* hostPtr, void* devPtr, size_t size) {
    if (tensor_pair_count >= RUNTIME_MAX_TENSOR_PAIRS) {
        fprintf(stderr, "[Runtime] ERROR: Tensor pairs full (max=%d)\n", RUNTIME_MAX_TENSOR_PAIRS);
        return;
    }
    tensor_pairs[tensor_pair_count].hostPtr = hostPtr;
    tensor_pairs[tensor_pair_count].devPtr = devPtr;
    tensor_pairs[tensor_pair_count].size = size;
    tensor_pair_count++;
    printf("Recorded tensor pair: host=%p dev=%p size=%zu\n", hostPtr, devPtr, size);
}

TensorPair* Runtime::GetTensorPairs() {
    return tensor_pairs;
}

int Runtime::GetTensorPairCount() const {
    return tensor_pair_count;
}

void Runtime::ClearTensorPairs() {
    tensor_pair_count = 0;
}
