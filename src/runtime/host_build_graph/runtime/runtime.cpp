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
        memset(tasks[i].args, 0, sizeof(tasks[i].args));
        memset(tasks[i].fanout, 0, sizeof(tasks[i].fanout));
    }
    next_task_id = 0;
    initial_ready_count = 0;
    worker_count = 0;
    sche_cpu_num = 1;
    tensor_pair_count = 0;

    // Initialize function address mapping
    for (int i = 0; i < RUNTIME_MAX_FUNC_ID; i++) {
        func_id_to_addr_[i] = 0;
    }
}

// =============================================================================
// Task Management
// =============================================================================

int Runtime::add_task(uint64_t* args, int num_args, int func_id, CoreType core_type) {
    // Check bounds
    if (next_task_id >= RUNTIME_MAX_TASKS) {
        LOG_ERROR("[Runtime] Task table full (max=%d)", RUNTIME_MAX_TASKS);
        return -1;
    }

    if (num_args > RUNTIME_MAX_ARGS) {
        LOG_ERROR("[Runtime] Too many args (%d > %d)", num_args, RUNTIME_MAX_ARGS);
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
    task->function_bin_addr = 0;    // Will be set by host before copying to device
    task->core_type = core_type;    // Set core type
    task->fanin = 0;
    task->fanout_count = 0;
    memset(task->fanout, 0, sizeof(task->fanout));

    return task_id;
}

void Runtime::add_successor(int from_task, int to_task) {
    // Validate task IDs
    if (from_task < 0 || from_task >= next_task_id) {
        LOG_ERROR("[Runtime] Invalid from_task ID %d", from_task);
        return;
    }

    if (to_task < 0 || to_task >= next_task_id) {
        LOG_ERROR("[Runtime] Invalid to_task ID %d", to_task);
        return;
    }

    Task* from = &tasks[from_task];
    Task* to = &tasks[to_task];

    // Add to_task to from_task's fanout
    if (from->fanout_count >= RUNTIME_MAX_FANOUT) {
        LOG_ERROR("[Runtime] Fanout overflow for task %d (max=%d)", from_task, RUNTIME_MAX_FANOUT);
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
    LOG_DEBUG("\n================================================================================");
    LOG_DEBUG("[Runtime] Task Runtime Status");
    LOG_DEBUG("========================================================================");
    LOG_DEBUG("  Total tasks: %d", next_task_id);

    // Print initially ready tasks
    LOG_DEBUG("\nInitially Ready Tasks (fanin==0):");
    LOG_DEBUG("----------------------------------------------------------------------");
    
    // Build ready tasks string
    char ready_tasks_str[1024] = "  ";
    int offset = 2;
    int ready_count = 0;
    for (int i = 0; i < next_task_id && offset < 1000; i++) {
        if (tasks[i].fanin.load() == 0) {
            if (ready_count > 0) {
                offset += snprintf(ready_tasks_str + offset, sizeof(ready_tasks_str) - offset, ", ");
            }
            offset += snprintf(ready_tasks_str + offset, sizeof(ready_tasks_str) - offset, "%d", i);
            ready_count++;
        }
    }
    if (ready_count == 0) {
        snprintf(ready_tasks_str, sizeof(ready_tasks_str), "  (none)");
    }
    LOG_DEBUG("%s", ready_tasks_str);
    LOG_DEBUG("  Count: %d", ready_count);

    LOG_DEBUG("\nTask Table:");
    LOG_DEBUG("----------------------------------------------------------------------");

    for (int i = 0; i < next_task_id; i++) {
        const Task* t = &tasks[i];

        // Build fanout string
        char fanout_str[512];
        int fo_offset = 0;
        for (int j = 0; j < t->fanout_count && fo_offset < 500; j++) {
            fo_offset += snprintf(fanout_str + fo_offset, sizeof(fanout_str) - fo_offset, 
                                  "%d%s", t->fanout[j], j < t->fanout_count - 1 ? "," : "");
        }

        LOG_DEBUG("  Task %d: func_id=%d, fanin=%d, fanout=%d, args=%d [%s]",
            i,
            t->func_id,
            t->fanin.load(),
            t->fanout_count,
            t->num_args,
            fanout_str);
    }

    LOG_DEBUG("========================================================================");
}

// =============================================================================
// Tensor Pair Management
// =============================================================================

void Runtime::record_tensor_pair(void* host_ptr, void* dev_ptr, size_t size) {
    if (tensor_pair_count >= RUNTIME_MAX_TENSOR_PAIRS) {
        LOG_ERROR("[Runtime] Tensor pairs full (max=%d)", RUNTIME_MAX_TENSOR_PAIRS);
        return;
    }
    tensor_pairs[tensor_pair_count].host_ptr = host_ptr;
    tensor_pairs[tensor_pair_count].dev_ptr = dev_ptr;
    tensor_pairs[tensor_pair_count].size = size;
    tensor_pair_count++;
    LOG_DEBUG("Recorded tensor pair: host=%p dev=%p size=%zu", host_ptr, dev_ptr, size);
}

TensorPair* Runtime::get_tensor_pairs() {
    return tensor_pairs;
}

int Runtime::get_tensor_pair_count() const {
    return tensor_pair_count;
}

void Runtime::clear_tensor_pairs() {
    tensor_pair_count = 0;
}

// =============================================================================
// Performance Profiling
// =============================================================================

void Runtime::complete_perf_records(PerfBuffer* perf_buf) {
    uint32_t count = perf_buf->count;

    for (uint32_t i = 0; i < count; i++) {
        PerfRecord* record = &perf_buf->records[i];
        uint32_t task_id = record->task_id;

        // Query Task by task_id (O(1) array indexing)
        Task* task = get_task(task_id);
        if (task != nullptr) {
            record->fanout_count = task->fanout_count;

            for (int32_t j = 0; j < task->fanout_count; j++) {
                record->fanout[j] = task->fanout[j];
            }
        } else {
            record->fanout_count = 0;
        }
    }
}
