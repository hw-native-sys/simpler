/**
 * Runtime Class - Implementation
 *
 * Device execution and handshake control.
 * Task graph construction is handled by PTO2Runtime.
 */

#include "runtime.h"
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
    ready_queue_shards = RUNTIME_DEFAULT_READY_QUEUE_SHARDS;

    // Initialize tensor pairs
    tensor_pair_count = 0;

    // Initialize device orchestration state
    orch_built_on_host_ = true;
    pto2_gm_sm_ptr_ = nullptr;
    pto2_gm_heap_ptr_ = nullptr;
    orch_args_ = nullptr;
    orch_arg_count_ = 0;

    // Initialize device orchestration SO binary
    device_orch_so_size_ = 0;

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
    }
}

// =============================================================================
// Performance Profiling
// =============================================================================

void Runtime::complete_perf_records(PerfBuffer* perf_buf) {
    // Get PTO2 shared memory context
    void* sm_base = get_pto2_gm_sm_ptr();
    if (sm_base == nullptr) {
        // No PTO2 context, cannot complete records
        return;
    }

    // Get PTO2 data structures
    PTO2SharedMemoryHeader* header = static_cast<PTO2SharedMemoryHeader*>(sm_base);
    PTO2TaskDescriptor* task_descriptors = reinterpret_cast<PTO2TaskDescriptor*>(
        static_cast<char*>(sm_base) + header->task_descriptors_offset);
    PTO2DepListEntry* dep_list_pool = reinterpret_cast<PTO2DepListEntry*>(
        static_cast<char*>(sm_base) + header->dep_list_pool_offset);
    int32_t window_mask = header->task_window_size - 1;

    uint32_t count = perf_buf->count;

    for (uint32_t i = 0; i < count; i++) {
        PerfRecord* record = &perf_buf->records[i];
        int32_t task_id = record->task_id;

        // Get TaskDescriptor from PTO2 shared memory
        int32_t slot = task_id & window_mask;
        PTO2TaskDescriptor* task = &task_descriptors[slot];

        // Fill fanout information by traversing the linked list
        record->fanout_count = 0;
        int32_t fanout_offset = task->fanout_head;

        while (fanout_offset != 0 && record->fanout_count < RUNTIME_MAX_FANOUT) {
            PTO2DepListEntry* entry = &dep_list_pool[fanout_offset];
            record->fanout[record->fanout_count++] = entry->task_id;
            fanout_offset = entry->next_offset;
        }
    }
}
