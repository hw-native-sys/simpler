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

    // Initialize the device-copied descriptor (`dev`).
    memset(dev.workers, 0, sizeof(dev.workers));
    dev.worker_count = 0;
    dev.aicpu_thread_num = 1;
    dev.ready_queue_shards = RUNTIME_DEFAULT_READY_QUEUE_SHARDS;
    memset(dev.aicpu_allowed_cpus, 0, sizeof(dev.aicpu_allowed_cpus));
    dev.aicpu_allowed_cpu_count = 0;
    dev.aicpu_launch_count = 0;
    dev.serial_orch_sched = false;
    dev.drain_aba_test_mode = false;
    dev.gm_sm_ptr_ = nullptr;
    dev.orch_args_storage_.clear();
    dev.prebuilt_arena_base_ = nullptr;
    dev.prebuilt_runtime_offset_ = 0;
    dev.active_callable_id_ = -1;
    for (int i = 0; i < RUNTIME_MAX_FUNC_ID; i++) {
        dev.func_id_to_addr_[i] = 0;
    }
}

// =============================================================================
// Device orchestration
// =============================================================================

void *Runtime::get_gm_sm_ptr() const { return dev.gm_sm_ptr_; }
const ChipStorageTaskArgs &Runtime::get_orch_args() const { return dev.orch_args_storage_; }
void Runtime::set_gm_sm_ptr(void *p) { dev.gm_sm_ptr_ = p; }
void Runtime::set_orch_args(const ChipStorageTaskArgs &args) { dev.orch_args_storage_ = args; }

void Runtime::set_prebuilt_arena(void *arena_base, size_t runtime_off) {
    dev.prebuilt_arena_base_ = arena_base;
    dev.prebuilt_runtime_offset_ = runtime_off;
}
void *Runtime::get_prebuilt_arena_base() const { return dev.prebuilt_arena_base_; }
size_t Runtime::get_prebuilt_runtime_offset() const { return dev.prebuilt_runtime_offset_; }

void Runtime::set_active_callable_id(int32_t callable_id) { dev.active_callable_id_ = callable_id; }

int32_t Runtime::get_active_callable_id() const { return dev.active_callable_id_; }

uint64_t Runtime::get_function_bin_addr(int func_id) const {
    if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
    return dev.func_id_to_addr_[func_id];
}

void Runtime::replay_function_bin_addr(int func_id, uint64_t addr) {
    if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) {
        LOG_ERROR("[Runtime] func_id=%d is out of range [0, %d)", func_id, RUNTIME_MAX_FUNC_ID);
        return;
    }
    dev.func_id_to_addr_[func_id] = addr;
}

// trb's device image is just the `dev` descriptor (the rest of Runtime is
// host-only). Mirrors the host_build_graph definition (= sizeof(Runtime)).
size_t runtime_device_copy_size(const Runtime &) { return sizeof(DeviceRuntimeLaunchDesc); }
