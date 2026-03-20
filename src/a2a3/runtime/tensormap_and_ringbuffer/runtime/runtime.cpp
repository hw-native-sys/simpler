/**
 * Runtime Class - Implementation
 *
 * Device execution and handshake control.
 * Task graph construction is handled by PTO2Runtime.
 */

#include "runtime.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "common/unified_log.h"

// =============================================================================
// Constructor
// =============================================================================

Runtime::Runtime() {
    // NOTE: host_api is initialized in InitRuntime() (host-only code)
    // because the CApi functions don't exist when compiled for device.

    // Initialize handshake buffers
    memset(workers, 0, sizeof(workers));
    worker_count = 0;
    sche_cpu_num = 1;
    orch_thread_num = 1;
    ready_queue_shards = RUNTIME_DEFAULT_READY_QUEUE_SHARDS;
    pto2_task_window_size = 0;
    pto2_heap_size = 0;
    pto2_dep_pool_size = 0;
    orch_to_sched = false;

    // Initialize tensor pairs
    tensor_pair_count = 0;

    // Initialize device orchestration state
    orch_built_on_host_ = true;
    pto2_gm_sm_ptr_ = nullptr;
    pto2_gm_heap_ptr_ = nullptr;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        pto2_ring_slot_states_ptrs_[r] = nullptr;
    }
    orch_args_ = nullptr;
    orch_arg_count_ = 0;

    // Initialize device orchestration SO binary
    device_orch_so_size_ = 0;

    // Initialize kernel binary tracking
    registered_kernel_count_ = 0;

    // Initialize function address mapping
    for (int i = 0; i < RUNTIME_MAX_FUNC_ID; i++) {
        func_id_to_addr_[i] = 0;
    }
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
    LOG_INFO("Recorded tensor pair: host=%p dev=%p size=%zu", host_ptr, dev_ptr, size);
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
// Device orchestration
// =============================================================================

bool Runtime::get_orch_built_on_host() const { return orch_built_on_host_; }
void* Runtime::get_pto2_gm_sm_ptr() const { return pto2_gm_sm_ptr_; }
void* Runtime::get_pto2_gm_heap_ptr() const { return pto2_gm_heap_ptr_; }
uint64_t* Runtime::get_orch_args() const {
    // Return embedded storage directly (not the pointer) so device code gets correct device address
    // When Runtime is copied to device memory, computing address relative to 'this' gives valid device address
    return orch_arg_count_ > 0 ? const_cast<uint64_t*>(orch_args_storage_) : nullptr;
}
int Runtime::get_orch_arg_count() const { return orch_arg_count_; }
void Runtime::set_orch_built_on_host(bool v) { orch_built_on_host_ = v; }
void Runtime::set_pto2_gm_sm_ptr(void* p) { pto2_gm_sm_ptr_ = p; }
void Runtime::set_pto2_gm_heap(void* p) { pto2_gm_heap_ptr_ = p; }
void Runtime::set_pto2_ring_slot_states_ptr(int ring_id, void* p) {
    if (ring_id >= 0 && ring_id < PTO2_MAX_RING_DEPTH) {
        pto2_ring_slot_states_ptrs_[ring_id] = p;
    }
}
void Runtime::set_orch_args(uint64_t* args, int count) {
    orch_arg_count_ = count <= RUNTIME_MAX_ARGS ? count : RUNTIME_MAX_ARGS;
    if (args && orch_arg_count_ > 0) {
        memcpy(orch_args_storage_, args, (size_t)orch_arg_count_ * sizeof(uint64_t));
        // Note: We no longer store orch_args_ pointer as it would contain host address
        // get_orch_args() now computes address from embedded storage directly
    }
}

// Device orchestration SO binary (for dlopen on AICPU thread 3)
// Copies data to internal storage to avoid lifetime issues with Python ctypes arrays
void Runtime::set_device_orch_so(const void* data, size_t size) {
    if (data == nullptr || size == 0) {
        device_orch_so_size_ = 0;
        return;
    }
    if (size > RUNTIME_MAX_ORCH_SO_SIZE) {
        LOG_ERROR("[Runtime] Orchestration SO too large (%zu > %d)", size, RUNTIME_MAX_ORCH_SO_SIZE);
        device_orch_so_size_ = 0;
        return;
    }
    memcpy(device_orch_so_storage_, data, size);
    device_orch_so_size_ = size;
}

const void* Runtime::get_device_orch_so_data() const {
    return device_orch_so_size_ > 0 ? device_orch_so_storage_ : nullptr;
}

size_t Runtime::get_device_orch_so_size() const {
    return device_orch_so_size_;
}

uint64_t Runtime::get_function_bin_addr(int func_id) const {
    if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
    return func_id_to_addr_[func_id];
}

void Runtime::set_function_bin_addr(int func_id, uint64_t addr) {
    if (func_id >= 0 && func_id < RUNTIME_MAX_FUNC_ID) {
        func_id_to_addr_[func_id] = addr;
        if (addr != 0 && registered_kernel_count_ < RUNTIME_MAX_FUNC_ID) {
            registered_kernel_func_ids_[registered_kernel_count_++] = func_id;
        }
    }
}

int Runtime::get_registered_kernel_count() const {
    return registered_kernel_count_;
}

int Runtime::get_registered_kernel_func_id(int index) const {
    if (index < 0 || index >= registered_kernel_count_) return -1;
    return registered_kernel_func_ids_[index];
}

void Runtime::clear_registered_kernels() {
    registered_kernel_count_ = 0;
}

// =============================================================================
// Performance Profiling
// =============================================================================

void Runtime::complete_perf_records(PerfBuffer* perf_buf) {
    // Get PTO2 shared memory context
    void* sm_base = get_pto2_gm_sm_ptr();
    if (sm_base == nullptr) {
        return;
    }

    PTO2SharedMemoryHeader* header = static_cast<PTO2SharedMemoryHeader*>(sm_base);
    uint32_t count = perf_buf->count;

    for (uint32_t i = 0; i < count; i++) {
        PerfRecord* record = &perf_buf->records[i];

        // Already filled by AICPU side at task completion time.
        if (record->fanout_filled != 0) {
            continue;
        }

        PTO2TaskId task_id{record->mixed_task_id};
        uint8_t ring_id = task_id.ring();
        if (ring_id >= PTO2_MAX_RING_DEPTH) {
            continue;
        }

        PTO2TaskSlotState* slot_states = static_cast<PTO2TaskSlotState*>(pto2_ring_slot_states_ptrs_[ring_id]);
        if (slot_states == nullptr) {
            continue;
        }

        int32_t window_mask = static_cast<int32_t>(header->rings[ring_id].task_window_size) - 1;
        int32_t slot = static_cast<int32_t>(task_id.local()) & window_mask;
        PTO2TaskSlotState& ss = slot_states[slot];

        // Fill fanout by traversing the linked list (best-effort: no lock, see aicpu_executor.cpp)
        PTO2DepListEntry* cur = ss.fanout_head;
        record->fanout_count = 0;
        while (cur != nullptr && record->fanout_count < RUNTIME_MAX_FANOUT) {
            record->fanout[record->fanout_count++] =
                static_cast<uint64_t>(cur->slot_state->task->mixed_task_id);
            cur = cur->next;
        }
        record->fanout_filled = 1;
    }
}
