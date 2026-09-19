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

#include "host_build_graph/runtime.h"

#include <cstring>

#include "common/unified_log.h"

// =============================================================================
// Constructor
// =============================================================================

Runtime::Runtime() {
    // Initialize the device-copied descriptor (`dev`).
    std::memset(dev.workers, 0, sizeof(dev.workers));
    dev.worker_count = 0;
    dev.aicpu_thread_num = 1;
    dev.ready_queue_shards = RUNTIME_DEFAULT_READY_QUEUE_SHARDS;
    std::memset(dev.aicpu_allowed_cpus, 0, sizeof(dev.aicpu_allowed_cpus));
    dev.aicpu_allowed_cpu_count = 0;
    dev.aicpu_launch_count = 0;
    dev.host_total_tasks = 0;
    dev.sm_image_bytes = 0;
    dev.gm_sm_ptr_ = nullptr;
    dev.prebuilt_arena_base_ = nullptr;
    dev.prebuilt_runtime_offset_ = 0;

    host_.orch_args_storage_.clear();
    host_.active_callable_id_ = -1;
    host_.pending_publication_ = {};

    // Initialize function address mapping
    for (int i = 0; i < RUNTIME_MAX_FUNC_ID; i++) {
        dev.func_id_to_addr_[i] = 0;
    }
}

// =============================================================================
// Shared-memory / orchestration argument plumbing
// =============================================================================

void *Runtime::get_gm_sm_ptr() const { return dev.gm_sm_ptr_; }
const simpler::hbg::EntryArgsStorage &Runtime::get_orch_args() const { return host_.orch_args_storage_; }
void Runtime::set_gm_sm_ptr(void *p) { dev.gm_sm_ptr_ = p; }
// The one place a boundary ChipTensor becomes this runtime's Tensor. Called from
// the host, before any orchestration runs, so nothing inside the runtime — on the
// host or on the AICPU — ever holds the boundary form.
void Runtime::set_orch_args(const ChipStorageTaskArgs &args) {
    host_.orch_args_storage_.clear();
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        host_.orch_args_storage_.add_tensor(simpler::hbg::Tensor::from_boundary(args.tensor(i)));
    }
    for (int32_t i = 0; i < args.scalar_count(); ++i) {
        host_.orch_args_storage_.add_scalar(args.scalar(i));
    }
}

void Runtime::set_prebuilt_arena(void *arena_base, size_t runtime_off) {
    dev.prebuilt_arena_base_ = arena_base;
    dev.prebuilt_runtime_offset_ = runtime_off;
}
void *Runtime::get_prebuilt_arena_base() const { return dev.prebuilt_arena_base_; }
size_t Runtime::get_prebuilt_runtime_offset() const { return dev.prebuilt_runtime_offset_; }

// The callable this runtime is stamped with. The orchestration .so and its entry
// symbols stay in the platform's `CallableArtifacts`, because host_build_graph
// resolves and runs them on the host.
void Runtime::set_active_callable_id(int32_t callable_id) { host_.active_callable_id_ = callable_id; }

int32_t Runtime::get_active_callable_id() const { return host_.active_callable_id_; }

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

void Runtime::clear_function_bin_addrs() {
    for (int i = 0; i < RUNTIME_MAX_FUNC_ID; i++) {
        dev.func_id_to_addr_[i] = 0;
    }
}

// host_build_graph ships the Runtime object without its host-only tail: the
// AICPU addresses fields inside that prefix directly, and nothing past it has a
// device reader.
size_t runtime_device_copy_size(const Runtime &) { return Runtime::device_image_bytes(); }
