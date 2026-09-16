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
 * Onboard host common helpers — shared between a2a3 and a5 onboard host
 * runtime libraries (`libhost_runtime.so`).
 *
 * Migration target for code that's line-identical between arches; arch-specific
 * extensions (e.g. a2a3's `init_ffts_base_addr`) live as free functions in
 * the arch's own `device_runner.h` rather than being declared here.
 *
 * Current contents:
 *   - `KernelArgsHelper`: host-side `KernelArgs` wrapper with device-memory
 *     management for the H2D `Runtime` and `KernelArgs` copies.
 *
 * Future migrations:
 *   - `DeviceRunnerBase` (lifecycle + registration + profiling init).
 *   - C-API common shims.
 */

#pragma once

#include <runtime/rt.h>

#include <cstdint>
#include <utility>

#include "common/kernel_args.h"  // arch-specific KernelArgs layout
#include "host/memory_allocator.h"
#include "runtime_c_api.h"
#include "runtime.h"

/**
 * Query both streams that form one onboard run without waiting.
 *
 * Completion is reported only after rtStreamQuery reports both the AICPU and
 * AICore streams complete. The return value is one of the
 * SIMPLER_NATIVE_RUN_POLL_* constants.
 */
int query_stream_pair_nonblocking(rtStream_t aicpu_stream, rtStream_t aicore_stream);

/**
 * The device blocks one pipeline slot reuses across every run it prepares.
 *
 * All three have a size fixed for the runner's lifetime — `sizeof(KernelArgs)`,
 * the runtime variant's device-copy length, and the architecture register
 * table — so a run rewrites their contents rather than reallocating them. A
 * slot admits at most one run at a time (`try_reserve_native_run` rejects a
 * second reservation on an occupied slot), so one block per slot needs no
 * further serialization.
 *
 * The runner owns these for its whole lifetime and releases them in
 * `finalize()`, alongside the collector resources that already work this way.
 */
struct SlotPersistentArgs {
    Runtime *runtime_args{nullptr};      // device copy of the Runtime prefix
    KernelArgs *device_k_args{nullptr};  // device copy of KernelArgs for AICore
    uint64_t regs{0};                    // architecture register table
    uint64_t runtime_bytes{0};           // committed length of runtime_args
    // Whether `regs` names a table whose contents reached the device. A failed
    // host-to-device copy whose rollback release also failed leaves the block
    // owned but unwritten, so `regs != 0` alone does not mean "usable": the
    // address is retained for release, and the next prepare must commit again.
    bool regs_committed{false};
};

/**
 * Helper class for managing `KernelArgs` with device memory.
 *
 * Wraps `KernelArgs` (defined per-arch in `common/kernel_args.h`) and provides
 * host-side initialization methods for publishing data to the device. The
 * `KernelArgs` value is per-run: every prepare refills it and copies it over.
 * The device blocks it names are not — they live in the slot's
 * `SlotPersistentArgs` and are only rewritten here. Separates device-memory
 * management (host-only) from the structure layout (shared with kernels).
 *
 * The helper provides implicit conversion to `KernelArgs *` for seamless use
 * with runtime APIs.
 *
 * Arch-specific extensions (a2a3-only `init_ffts_base_addr`, etc.) live as
 * free functions in the arch's own `device_runner.h`.
 */
struct KernelArgsHelper {
    KernelArgsHelper() = default;
    KernelArgsHelper(const KernelArgsHelper &) = delete;
    KernelArgsHelper &operator=(const KernelArgsHelper &) = delete;
    KernelArgsHelper(KernelArgsHelper &&other) noexcept :
        args(other.args),
        allocator_(std::exchange(other.allocator_, nullptr)),
        device_k_args_(std::exchange(other.device_k_args_, nullptr)) {
        other.args = KernelArgs{};
    }
    KernelArgsHelper &operator=(KernelArgsHelper &&) = delete;

    KernelArgs args;
    MemoryAllocator *allocator_{nullptr};
    KernelArgs *device_k_args_{nullptr};  // Device copy of KernelArgs for AICore

    /**
     * Publish the host runtime into the slot's device copy, committing that
     * copy on first use.
     *
     * @param host_runtime  Host-side runtime to copy to device.
     * @param allocator     Memory allocator to use.
     * @param slot          The slot's persistent device blocks.
     * @return 0 on success, error code on failure.
     */
    int init_runtime_args(const Runtime &host_runtime, MemoryAllocator &allocator, SlotPersistentArgs &slot);

    /**
     * Publish this run's `KernelArgs` into the slot's device copy, committing
     * that copy on first use. AICore's `KERNEL_ENTRY` expects a `KernelArgs *`
     * (not a `Runtime *`) so it can read the profiling enablement bits + ring
     * address tables and forward them into AICore platform state. Call this
     * after every `kernel_args.args.*` field is populated for the run.
     */
    int init_device_kernel_args(MemoryAllocator &allocator, SlotPersistentArgs &slot);

    /**
     * Drop this run's view of the slot's device blocks.
     *
     * The blocks themselves stay committed for the next run on this slot; only
     * the per-run `KernelArgs` stops naming them.
     */
    void release_run_view() {
        args.runtime_args = nullptr;
        device_k_args_ = nullptr;
    }

    /**
     * Clear device-pointer bookkeeping without calling the allocator.
     *
     * Used only by fatal teardown after reset/quarantine.
     */
    void abandon_after_device_failure() {
        args.runtime_args = nullptr;
        device_k_args_ = nullptr;
        allocator_ = nullptr;
    }

    /**
     * Implicit conversion operators for seamless use with runtime APIs.
     *
     * These allow `KernelArgsHelper` to be used wherever a payload
     * `KernelArgs *` is expected.
     */
    operator KernelArgs *() { return &args; }
    KernelArgs *operator&() { return &args; }
};

/**
 * Release one slot's persistent device blocks and clear its bookkeeping.
 *
 * Returns the first failing free's code, having attempted every block, so a
 * single failure cannot strand the rest. A block whose free fails keeps its
 * address so a retry reaches it again.
 */
int release_slot_persistent_args(SlotPersistentArgs &slot, MemoryAllocator &allocator);

/**
 * Drop one slot's persistent device blocks without calling the allocator.
 *
 * Used only by fatal teardown, where a reset has already invalidated the
 * device generation these addresses belong to.
 */
void abandon_slot_persistent_args(SlotPersistentArgs &slot);
