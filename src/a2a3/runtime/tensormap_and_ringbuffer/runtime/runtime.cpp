/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Runtime Class - Implementation
 *
 * Device execution and handshake control.
 * Task graph construction is handled by PTO2Runtime.
 */

#include "runtime.h"  // NOLINT(build/include_subdir)

#include "common/unified_log.h"
#include "pto_runtime2_types.h"  // NOLINT(build/include_subdir)
#include "pto_shared_memory.h"   // NOLINT(build/include_subdir)

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
    pto2_slot_states_ptr_ = nullptr;
    memset(async_context_addrs_, 0, sizeof(async_context_addrs_));
    orch_args_storage_.clear();

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

void Runtime::record_tensor_pair(void *host_ptr, void *dev_ptr, size_t size) {
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

TensorPair *Runtime::get_tensor_pairs() { return tensor_pairs; }

int Runtime::get_tensor_pair_count() const { return tensor_pair_count; }

void Runtime::clear_tensor_pairs() { tensor_pair_count = 0; }

// =============================================================================
// Device orchestration
// =============================================================================

bool Runtime::get_orch_built_on_host() const { return orch_built_on_host_; }
void *Runtime::get_pto2_gm_sm_ptr() const { return pto2_gm_sm_ptr_; }
void *Runtime::get_pto2_gm_heap_ptr() const { return pto2_gm_heap_ptr_; }
const ChipStorageTaskArgs &Runtime::get_orch_args() const { return orch_args_storage_; }
void Runtime::set_orch_built_on_host(bool v) { orch_built_on_host_ = v; }
void Runtime::set_pto2_gm_sm_ptr(void *p) { pto2_gm_sm_ptr_ = p; }
void Runtime::set_pto2_gm_heap(void *p) { pto2_gm_heap_ptr_ = p; }
void Runtime::set_pto2_slot_states_ptr(void *p) { pto2_slot_states_ptr_ = p; }
void Runtime::set_async_context_addr(PTO2AsyncEngine engine, uint64_t addr) {
    if (engine < PTO2_NUM_ASYNC_ENGINES) {
        async_context_addrs_[engine] = addr;
    }
}
uint64_t Runtime::get_async_context_addr(PTO2AsyncEngine engine) const {
    if (engine < PTO2_NUM_ASYNC_ENGINES) {
        return async_context_addrs_[engine];
    }
    return 0;
}
void Runtime::set_orch_args(const ChipStorageTaskArgs &args) { orch_args_storage_ = args; }

// Device orchestration SO binary (for dlopen on AICPU thread 3)
// Copies data to internal storage to avoid lifetime issues with Python ctypes arrays
void Runtime::set_device_orch_so(const void *data, size_t size) {
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

const void *Runtime::get_device_orch_so_data() const {
    return device_orch_so_size_ > 0 ? device_orch_so_storage_ : nullptr;
}

size_t Runtime::get_device_orch_so_size() const { return device_orch_so_size_; }

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

int Runtime::get_registered_kernel_count() const { return registered_kernel_count_; }

int Runtime::get_registered_kernel_func_id(int index) const {
    if (index < 0 || index >= registered_kernel_count_) return -1;
    return registered_kernel_func_ids_[index];
}

void Runtime::clear_registered_kernels() { registered_kernel_count_ = 0; }
