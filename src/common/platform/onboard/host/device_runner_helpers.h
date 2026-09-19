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
#include "host/runtime_launch_image.h"
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
 * The error either stream of one onboard run is already holding, or 0 for none.
 *
 * Non-blocking, and strictly weaker than a stream synchronize: it reports a
 * stream left in a sticky error state, which is what `rtStreamQuery` answers.
 * It does **not** detect a device exception raised by the work itself —
 * measured on a2a3, an AICPU kernel that returns a fatal status leaves both
 * streams reading drained and error-free here, and only
 * `aclrtSynchronizeStreamWithTimeout` produces the 507018. So this preserves
 * what a non-blocking poll used to report; it is not a substitute for the
 * synchronize on the drain path.
 *
 * Work still queued reads as no error, since a stream holds its error stickily.
 * Returns the AICPU error in preference to the AICore one, matching the order
 * `sync_stream_pair` reports them in.
 */
int query_stream_pair_error(rtStream_t aicpu_stream, rtStream_t aicore_stream);

/**
 * The device block one pipeline slot reuses across every run it prepares.
 *
 * Its size is fixed for the runner's lifetime — the runtime variant's
 * device-copy length — so a run rewrites its contents rather than
 * reallocating it. A slot admits at most one run at a time
 * (`try_reserve_native_run` rejects a second reservation on an occupied slot),
 * so one block per slot needs no further serialization. Per-slot rather than
 * per-runner because the copy is not ordered on the run stream: a prepared
 * successor would otherwise overwrite the image its predecessor is executing
 * against.
 *
 * The runner owns this for its whole lifetime and releases it in `finalize()`,
 * alongside the collector resources that already work this way.
 *
 * The AICore register tables are deliberately NOT here: they are device
 * constants, identical for every slot, and are owned per device context by
 * `DeviceRunnerBase::aicore_{ctrl,pmu}_reg_table_dev_`. Nor is a device copy of
 * `KernelArgs`: AICore receives everything it needs as launch arguments.
 */
struct SlotPersistentArgs {
    Runtime *runtime_args{nullptr};  // device copy of the Runtime prefix
    uint64_t runtime_bytes{0};       // committed length of runtime_args
};

/**
 * Helper class for managing `KernelArgs` with device memory.
 *
 * Wraps `KernelArgs` (defined per-arch in `common/kernel_args.h`) and provides
 * host-side initialization methods for publishing data to the device. The
 * `KernelArgs` value is per-run: every prepare refills it and copies it over.
 * The device blocks it names are not — they live in the slot's
 * `SlotPersistentArgs`, except the AICore register tables, which are owned per
 * device context by `DeviceRunnerBase`. Both are only rewritten here. Separates
 * device-memory management (host-only) from the structure layout (shared with
 * kernels).
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
        runtime_image_(std::move(other.runtime_image_)) {
        other.args = KernelArgs{};
    }
    KernelArgsHelper &operator=(KernelArgsHelper &&) = delete;

    KernelArgs args;
    MemoryAllocator *allocator_{nullptr};

    // Reserve the slot's destination and snapshot this invocation's device-read
    // descriptor. No bytes are published by preparation.
    int prepare_runtime_args(const Runtime &host_runtime, MemoryAllocator &allocator, SlotPersistentArgs &slot);

    // Consume the snapshot with a synchronous metadata H2D. The slot remains
    // owned even on failure; the run must not launch after an unsuccessful copy.
    int publish_runtime_args();

    /**
     * Drop this run's view of the slot's device blocks.
     *
     * The blocks themselves stay committed for the next run on this slot; only
     * the per-run `KernelArgs` stops naming them.
     */
    void release_run_view() {
        runtime_image_.clear();
        args.runtime_args = nullptr;
    }

    /**
     * Clear device-pointer bookkeeping without calling the allocator.
     *
     * Used only by fatal teardown after reset/quarantine.
     */
    void abandon_after_device_failure() {
        release_run_view();
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

private:
    RuntimeLaunchImage runtime_image_;
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
