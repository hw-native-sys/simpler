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
 * Context-lifetime owner of the AICore argument block.
 *
 * `KernelArgsHelper` reallocates its device blocks per prepared run; this owner
 * allocates them exactly once and hands the same three device addresses to
 * every subsequent launch, so a steady-state launch performs no allocation and
 * no host-to-device copy.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "common/kernel_args.h"
#include "runtime.h"

/**
 * The complete vocabulary available to argument preparation — the third
 * restricted table alongside KernelContextOps and KernelLaunchOps, and the
 * only one of the three that may allocate. Synchronization, stream and event
 * operations are absent, so a launch-path operation cannot be written against
 * this table at all.
 *
 * Allocation, release and host-to-device copy arrive as operations rather
 * than a `MemoryAllocator &` because `MemoryAllocator::alloc` / `free` are
 * non-virtual and `rtMemcpy` is a free function: a host-only test can observe
 * the balance of this object's device traffic only if they are parameters.
 *
 * `fill_arch_fields` exists for the second reason, which allocation
 * indirection alone does not solve: the device-side fields that are neither
 * pure memory nor uniform across architectures. The register table comes from
 * an arch entry that takes its own allocator and a different signature on
 * a2a3 than on a5, and reaches the driver; `ffts_base_addr` exists only on
 * a2a3. Absorbing both at the table's construction point keeps the platform
 * knowledge with the platform, exactly as `KernelContextOps::event_flag` does
 * for ACL_EVENT_SYNC.
 */
struct PersistentArgsOps {
    void *context{nullptr};
    void *(*alloc)(void *context, size_t bytes){nullptr};
    int (*free_)(void *context, void *ptr){nullptr};
    int (*copy_h2d)(void *context, void *dst, size_t dst_bytes, const void *src, size_t src_bytes){nullptr};
    /**
     * Populate the arch-specific device fields of `args`: `regs` on both
     * arches (a2a3 additionally selecting AicoreRegKind::Ctrl) and
     * `ffts_base_addr` on a2a3. DFX fields stay zero. Anything this
     * allocates must come from `alloc` above, so release through `free_`
     * matches and a host-only test's counts balance.
     */
    int (*fill_arch_fields)(void *context, KernelArgs *args, uint64_t device_id){nullptr};

    bool valid() const {
        return alloc != nullptr && free_ != nullptr && copy_h2d != nullptr && fill_arch_fields != nullptr;
    }
};

/**
 * Prepare-once / reuse-N-times / release-at-close owner of the three device
 * blocks an AICore launch reads: the device `Runtime` image, the per-core
 * register table, and the device copy of `KernelArgs` itself.
 *
 * Every operation this object performs happens in `prepare_once` and
 * `finalize_once`. The destructor performs none, so an owner that is dropped
 * without `finalize_once` leaks its device blocks rather than freeing memory
 * an in-flight device program may still read.
 *
 * The ops table is stored at `prepare_once` because `finalize_once` and
 * `abandon` run from a teardown path that no longer has it, the same reason
 * `KernelExecutionState` stores its own.
 */
class PersistentKernelArgs {
public:
    PersistentKernelArgs() = default;
    ~PersistentKernelArgs() = default;
    PersistentKernelArgs(const PersistentKernelArgs &) = delete;
    PersistentKernelArgs &operator=(const PersistentKernelArgs &) = delete;

    /**
     * Allocate and populate the three device blocks. Idempotent: a second
     * call on a prepared owner returns 0 without allocating, which is what
     * makes it correct to call from every prepare_callable.
     *
     * A failure at any step releases whatever this call allocated, in reverse
     * order, and leaves the owner unprepared with all three fields back at
     * their unset values; the original failure code is returned.
     */
    int prepare_once(const Runtime &host_runtime, const PersistentArgsOps &ops, uint64_t device_id);

    bool is_prepared() const { return prepared_; }

    /** Device address of the `KernelArgs` copy AICore's kernel entry receives. */
    KernelArgs *device_k_args() const { return device_k_args_; }

    const KernelArgs &args() const { return args_; }

    /**
     * Release the three device blocks in reverse allocation order. Idempotent.
     *
     * A failed release keeps its address so a retry redoes only the remainder,
     * and leaves the owner prepared; the first failure code is returned.
     */
    int finalize_once();

    /**
     * Drop the device-address bookkeeping without calling the ops table. The
     * device blocks are gone with the reset or quarantined device that made
     * releasing them meaningless.
     */
    void abandon();

private:
    int release_block(void *block, int &first_error);

    KernelArgs args_{};
    KernelArgs *device_k_args_{nullptr};
    PersistentArgsOps ops_{};
    bool prepared_{false};
};
