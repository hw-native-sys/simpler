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

#include "runtime.h"

#include "common/unified_log.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// =============================================================================
// Constructor
// =============================================================================

Runtime::Runtime() {
    // NOTE: host_api is initialized in InitRuntime() (host-only code)
    // because the CApi functions don't exist when compiled for device.

    // Initialize handshake buffers
    memset(workers, 0, sizeof(workers));
    worker_count = 0;
    aicpu_thread_num = 1;
    ready_queue_shards = RUNTIME_DEFAULT_READY_QUEUE_SHARDS;
    memset(aicpu_allowed_cpus, 0, sizeof(aicpu_allowed_cpus));
    aicpu_allowed_cpu_count = 0;
    aicpu_launch_count = 0;
    orch_to_sched = false;
    host_total_tasks = 0;

    // Initialize shared-memory / orchestration argument plumbing
    gm_sm_ptr_ = nullptr;
    gm_heap_ptr_ = nullptr;
    slot_states_ptr_ = nullptr;
    orch_args_storage_.clear();
    prebuilt_arena_base_ = nullptr;
    prebuilt_runtime_offset_ = 0;

    active_callable_id_ = -1;
    dev_orch_so_addr_ = 0;
    dev_orch_so_size_ = 0;
    device_orch_func_name_[0] = '\0';
    device_orch_config_name_[0] = '\0';

    // Initialize kernel binary tracking
    registered_kernel_count_ = 0;

    // Initialize function address mapping
    for (int i = 0; i < RUNTIME_MAX_FUNC_ID; i++) {
        func_id_to_addr_[i] = 0;
    }
}

// =============================================================================
// Shared-memory / orchestration argument plumbing
// =============================================================================

void *Runtime::get_gm_sm_ptr() const { return gm_sm_ptr_; }
void *Runtime::get_gm_heap_ptr() const { return gm_heap_ptr_; }
const ChipStorageTaskArgs &Runtime::get_orch_args() const { return orch_args_storage_; }
void Runtime::set_gm_sm_ptr(void *p) { gm_sm_ptr_ = p; }
void Runtime::set_gm_heap(void *p) { gm_heap_ptr_ = p; }
void Runtime::set_slot_states_ptr(void *p) { slot_states_ptr_ = p; }
void Runtime::set_orch_args(const ChipStorageTaskArgs &args) { orch_args_storage_ = args; }

void Runtime::set_prebuilt_arena(void *arena_base, size_t runtime_off) {
    prebuilt_arena_base_ = arena_base;
    prebuilt_runtime_offset_ = runtime_off;
}
void *Runtime::get_prebuilt_arena_base() const { return prebuilt_arena_base_; }
size_t Runtime::get_prebuilt_runtime_offset() const { return prebuilt_runtime_offset_; }

// Orchestration metadata written by the platform host (DeviceRunner) at
// callable registration. host_build_graph runs the orchestrator on the host so
// the device side never reads these back, but the platform registration path is
// shared with tensormap_and_ringbuffer and still writes them.
void Runtime::set_dev_orch_so(uint64_t dev_addr, uint64_t size) {
    dev_orch_so_addr_ = dev_addr;
    dev_orch_so_size_ = size;
}

void Runtime::set_active_callable_id(int32_t callable_id) { active_callable_id_ = callable_id; }

int32_t Runtime::get_active_callable_id() const { return active_callable_id_; }

void Runtime::set_device_orch_func_name(const char *name) {
    if (name == nullptr) {
        device_orch_func_name_[0] = '\0';
        return;
    }
    std::strncpy(device_orch_func_name_, name, RUNTIME_MAX_ORCH_SYMBOL_NAME - 1);
    device_orch_func_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME - 1] = '\0';
}

void Runtime::set_device_orch_config_name(const char *name) {
    if (name == nullptr) {
        device_orch_config_name_[0] = '\0';
        return;
    }
    std::strncpy(device_orch_config_name_, name, RUNTIME_MAX_ORCH_SYMBOL_NAME - 1);
    device_orch_config_name_[RUNTIME_MAX_ORCH_SYMBOL_NAME - 1] = '\0';
}

uint64_t Runtime::get_function_bin_addr(int func_id) const {
    if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
    return func_id_to_addr_[func_id];
}

void Runtime::set_function_bin_addr(int func_id, uint64_t addr) {
    if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
        LOG_ERROR("[Runtime] func_id=%d is out of range [0, %d)", func_id, RUNTIME_MAX_FUNC_ID);
        return;
    }
    if (addr != 0 && func_id_to_addr_[func_id] == 0) {
        if (registered_kernel_count_ < RUNTIME_MAX_FUNC_ID) {
            registered_kernel_func_ids_[registered_kernel_count_++] = func_id;
        } else {
            LOG_ERROR(
                "[Runtime] Registration limit reached (%d). Cannot track func_id=%d for cleanup.", RUNTIME_MAX_FUNC_ID,
                func_id
            );
        }
    }
    func_id_to_addr_[func_id] = addr;
}

void Runtime::replay_function_bin_addr(int func_id, uint64_t addr) {
    if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
        LOG_ERROR("[Runtime] func_id=%d is out of range [0, %d)", func_id, RUNTIME_MAX_FUNC_ID);
        return;
    }
    func_id_to_addr_[func_id] = addr;
}

int Runtime::get_registered_kernel_count() const { return registered_kernel_count_; }

int Runtime::get_registered_kernel_func_id(int index) const {
    if (index < 0 || index >= registered_kernel_count_) return -1;
    return registered_kernel_func_ids_[index];
}

void Runtime::clear_registered_kernels() { registered_kernel_count_ = 0; }

// host_build_graph's device image is the whole Runtime object (host-orch builds
// the graph on the host, but the entire Runtime is still rtMemcpy'd to device).
size_t runtime_device_copy_size(const Runtime &) { return sizeof(Runtime); }
