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
 * Onboard host `DeviceRunnerBase` — common base class for a2a3 and a5
 * onboard `DeviceRunner`s.
 *
 * This module owns the host-side state and methods that are identical
 * between the two onboard arches today:
 *   - The `MemoryAllocator` and the three `DeviceArena`s (gm heap, PTO2
 *     SM, runtime arena) backing the per-Worker pooled regions.
 *   - The trivial tensor-memory wrappers (`allocate_tensor`,
 *     `free_tensor`, `copy_*_device`).
 *   - The arena-pool accessors (`acquire_pooled_gm_heap`, etc.).
 *
 * Subclasses (`{a2a3,a5}::DeviceRunner`) add arch-specific state
 * (streams, kernel args, profiling collectors, callable registration)
 * and override behaviorally divergent methods (the kernel launch path,
 * `finalize`).
 *
 * The migration plan in `.docs/ONBOARD_HOST_COMMON_REFACTOR.md` lays
 * out the further extractions (lifecycle / registration / profiling
 * init / c_api shims) that will progressively move methods + their
 * load-bearing state from the arch subclass into this base.
 */

#ifndef SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_BASE_H
#define SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_BASE_H

#include <cstddef>

#include "device_arena.h"
#include "host/memory_allocator.h"

/**
 * Common base class for both a2a3 and a5 onboard `DeviceRunner`s.
 *
 * Ctor + dtor are `protected` so this class can only be used as a base;
 * direct instantiation and `delete` through a base pointer are both
 * compile errors. The arch subclass's `DeviceRunner` is what
 * `destroy_device_context` sees, so the non-virtual `~DeviceRunnerBase`
 * is safe — it never runs as a virtual base destructor.
 */
class DeviceRunnerBase {
public:
    DeviceRunnerBase(const DeviceRunnerBase &) = delete;
    DeviceRunnerBase &operator=(const DeviceRunnerBase &) = delete;
    DeviceRunnerBase(DeviceRunnerBase &&) = delete;
    DeviceRunnerBase &operator=(DeviceRunnerBase &&) = delete;

    /** Allocate / free / copy on the per-Worker `MemoryAllocator` + CANN runtime. */
    void *allocate_tensor(std::size_t bytes);
    void free_tensor(void *dev_ptr);
    int copy_to_device(void *dev_ptr, const void *host_ptr, std::size_t bytes);
    int copy_from_device(void *host_ptr, const void *dev_ptr, std::size_t bytes);

    /**
     * Return the pooled GM heap / PTO2 SM / runtime arena base pointer.
     * `setup_static_arena` (arch subclass) must have already committed
     * the relevant region; otherwise returns nullptr. The runtime arena
     * accessor is trb-only — hbg's `setup_static_arena(...,0)` leaves
     * `runtime_arena_pool_` uncommitted and this returns nullptr.
     */
    void *acquire_pooled_gm_heap();
    void *acquire_pooled_gm_sm();
    void *acquire_pooled_runtime_arena();

protected:
    // Ctor / dtor are protected: this class is for inheritance only —
    // direct instantiation (`new DeviceRunnerBase()`) and polymorphic delete
    // (`delete (DeviceRunnerBase *)p`) are both compile errors.
    DeviceRunnerBase();
    ~DeviceRunnerBase() = default;

    /**
     * `DeviceArena` callback trampolines bridging from C-style
     * `void *(void *ctx, size_t)` / `void (void *ctx, void *)` to the
     * `MemoryAllocator` member function calls. The `ctx` opaque pointer
     * passed at arena construction time is `&mem_alloc_`.
     */
    static void *arena_alloc_trampoline(void *ctx, std::size_t size) {
        return static_cast<MemoryAllocator *>(ctx)->alloc(size);
    }
    static void arena_free_trampoline(void *ctx, void *p) { static_cast<MemoryAllocator *>(ctx)->free(p); }

    MemoryAllocator mem_alloc_;
    DeviceArena gm_heap_arena_;
    DeviceArena gm_sm_arena_;
    DeviceArena runtime_arena_pool_;
};

#endif  // SIMPLER_COMMON_PLATFORM_ONBOARD_HOST_DEVICE_RUNNER_BASE_H
