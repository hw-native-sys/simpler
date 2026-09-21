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
 * Task graph construction is handled by RuntimeContext.
 */

#include "runtime.h"

#include "runtime_types.h"
#include "shared_memory.h"

// =============================================================================
// Constructor
// =============================================================================

Runtime::Runtime() {
    // NOTE: host_api is initialized in InitRuntime() (host-only code)
    // because the CApi functions don't exist when compiled for device.

    // Initialize the device-copied descriptor (`dev`).
    memset(dev.workers, 0, sizeof(dev.workers));
    dev.worker_count = 0;
    dev.aicpu_thread_num = 1;
    memset(dev.aicpu_allowed_cpus, 0, sizeof(dev.aicpu_allowed_cpus));
    dev.aicpu_allowed_cpu_count = 0;
    dev.aicpu_launch_count = 0;
    dev.serial_orch_sched = false;
    dev.gm_sm_ptr_ = nullptr;
    dev.orch_args_storage_.clear();
    dev.prebuilt_arena_base_ = nullptr;
    dev.prebuilt_runtime_offset_ = 0;
    dev.active_callable_id_ = -1;
    dev.callable_table_addr_ = 0;
    dev.callable_table_len_ = 0;
}

// =============================================================================
// Device orchestration
// =============================================================================

void *Runtime::get_gm_sm_ptr() const { return dev.gm_sm_ptr_; }
const simpler::tmr::EntryArgsStorage &Runtime::get_orch_args() const { return dev.orch_args_storage_; }
void Runtime::set_gm_sm_ptr(void *p) { dev.gm_sm_ptr_ = p; }
// The one place a boundary ChipTensor becomes this runtime's Tensor. Called from
// the host, before any orchestration runs, so nothing inside the runtime — on the
// host or on the AICPU — ever holds the boundary form.
void Runtime::set_orch_args(const ChipStorageTaskArgs &args) {
    dev.orch_args_storage_.clear();
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        dev.orch_args_storage_.add_tensor(simpler::tmr::Tensor::from_boundary(args.tensor(i)));
    }
    for (int32_t i = 0; i < args.scalar_count(); ++i) {
        dev.orch_args_storage_.add_scalar(args.scalar(i));
    }
}

void Runtime::set_prebuilt_arena(void *arena_base, size_t runtime_off) {
    dev.prebuilt_arena_base_ = arena_base;
    dev.prebuilt_runtime_offset_ = runtime_off;
}
void *Runtime::get_prebuilt_arena_base() const { return dev.prebuilt_arena_base_; }
size_t Runtime::get_prebuilt_runtime_offset() const { return dev.prebuilt_runtime_offset_; }

void Runtime::set_active_callable_id(int32_t callable_id) { dev.active_callable_id_ = callable_id; }

int32_t Runtime::get_active_callable_id() const { return dev.active_callable_id_; }

uint64_t Runtime::get_function_bin_addr(int func_id) const {
    if (callable_table_host_ == nullptr || func_id < 0 || static_cast<uint32_t>(func_id) >= dev.callable_table_len_) {
        return 0;
    }
    return callable_table_host_[func_id];
}

void Runtime::set_callable_tables(
    const uint64_t *host_view, uint64_t object_table_addr, uint64_t entry_table_addr, uint32_t len
) {
    if (host_view == nullptr || object_table_addr == 0 || len == 0) {
        clear_callable_tables();
        return;
    }
    callable_table_host_ = host_view;
    callable_entry_table_addr_ = entry_table_addr;
    dev.callable_table_addr_ = object_table_addr;
    dev.callable_table_len_ = len;
}

void Runtime::clear_callable_tables() {
    callable_table_host_ = nullptr;
    callable_entry_table_addr_ = 0;
    dev.callable_table_addr_ = 0;
    dev.callable_table_len_ = 0;
}

uint64_t Runtime::callable_entry_table_addr() const { return callable_entry_table_addr_; }

uint32_t Runtime::callable_table_len() const { return dev.callable_table_len_; }

// A steady-state trb run uploads the `dev` descriptor before the handshake
// region (the rest of Runtime is host-only). Neither that region nor the gate
// tail carries a host value the device consumes: the AICore publishes its
// report, the AICPU writes the task pointer it answers with, and the AICPU
// zeroes the active gates and executes wmb() before publishing hs_setup_done_,
// with no register window open before that.
size_t runtime_device_copy_size(const Runtime &) { return offsetof(DeviceRuntimeLaunchDesc, workers); }

// The first publication onto an allocation adds the handshake region, so it
// starts from the ctor-zeroed host copy rather than from whatever rtMalloc
// left. It stops before the gates, whose host storage `Runtime()` never
// initializes.
size_t runtime_device_initialized_prefix_size(const Runtime &) {
    return offsetof(DeviceRuntimeLaunchDesc, teardown_gates);
}

size_t runtime_device_extent_size(const Runtime &) { return sizeof(DeviceRuntimeLaunchDesc); }
