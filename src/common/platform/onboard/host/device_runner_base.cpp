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
 * `DeviceRunnerBase` — tensor-memory wrappers + pooled arena accessors.
 *
 * Constructor wires the three arenas to call back into `mem_alloc_` via
 * the static trampolines declared in the header. Per-region commit is
 * still driven by the subclass's `setup_static_arena`.
 */

#include "device_runner_base.h"

#include <runtime/rt.h>

DeviceRunnerBase::DeviceRunnerBase() :
    gm_heap_arena_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_),
    gm_sm_arena_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_),
    runtime_arena_pool_(&arena_alloc_trampoline, &arena_free_trampoline, &mem_alloc_) {}

void *DeviceRunnerBase::allocate_tensor(std::size_t bytes) { return mem_alloc_.alloc(bytes); }

void DeviceRunnerBase::free_tensor(void *dev_ptr) {
    if (dev_ptr != nullptr) {
        mem_alloc_.free(dev_ptr);
    }
}

int DeviceRunnerBase::copy_to_device(void *dev_ptr, const void *host_ptr, std::size_t bytes) {
    return rtMemcpy(dev_ptr, bytes, host_ptr, bytes, RT_MEMCPY_HOST_TO_DEVICE);
}

int DeviceRunnerBase::copy_from_device(void *host_ptr, const void *dev_ptr, std::size_t bytes) {
    return rtMemcpy(host_ptr, bytes, dev_ptr, bytes, RT_MEMCPY_DEVICE_TO_HOST);
}

void *DeviceRunnerBase::acquire_pooled_gm_heap() {
    if (!gm_heap_arena_.is_committed()) return nullptr;
    return gm_heap_arena_.base();
}

void *DeviceRunnerBase::acquire_pooled_gm_sm() {
    if (!gm_sm_arena_.is_committed()) return nullptr;
    return gm_sm_arena_.base();
}

void *DeviceRunnerBase::acquire_pooled_runtime_arena() {
    // hbg calls setup_static_arena(...,0) and leaves runtime_arena_pool_
    // uncommitted — fail loudly if a caller asks for it anyway.
    if (!runtime_arena_pool_.is_committed()) return nullptr;
    return runtime_arena_pool_.base();
}
