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
#include <vector>

#include "common/kernel_args.h"  // arch-specific KernelArgs layout
#include "common/launch_entry_args.h"
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
 * Whether the launch route is open on `aicpu_stream`.
 *
 * True only when the stream answers that it is capturing nothing. A capturing
 * or invalidated stream, an unavailable answer, and a null stream all say no,
 * which routes the values through the descriptor instead — the behaviour every
 * run had before the launch route existed. Read-only: it neither readies nor
 * retires the stream pair, and asking costs the run nothing when the answer is
 * no.
 *
 * Pass the same stream handle the launch will submit on, resolved and about to
 * be used, so the answer describes the stream that actually carries the launch.
 */
bool launch_entry_args_permitted(rtStream_t aicpu_stream);

/**
 * How one capture-status answer routes this run, given as its two halves so the
 * mapping is stated once and separately from the call that obtains it.
 *
 * Only a successful query reporting no capture opens the launch route.
 */
bool launch_route_permitted_by_capture(int query_rc, int capture_status);

/**
 * The device block one pipeline slot reuses across every run it prepares.
 *
 * Its size is fixed for the runner's lifetime — the runtime variant's device
 * extent — so a run rewrites its contents rather than reallocating it. That
 * extent is never shorter than what a run uploads, and on a variant whose
 * descriptor ends in device-initialized storage it is longer: that range lives
 * inside the block but outside every copy into it. A slot admits at most one run
 * at a time
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
    Runtime *runtime_args{nullptr};  // device block holding the Runtime descriptor
    uint64_t runtime_bytes{0};       // committed length: the full device extent, not a published prefix

    // Whether THIS block's handshake region has been published once.
    //
    // It states a fact about the allocation named above, not about the slot:
    // the field lives here so it is adopted, released and abandoned with the
    // pointer it describes, and the two teardown paths reset the whole struct
    // rather than named fields, so a later block cannot inherit a predecessor's
    // verdict. Committed only after the copy carrying that region has
    // succeeded; a failed publication leaves it false and the next prepare
    // sends the longer prefix again.
    bool workers_initialized{false};

    // Host staging for this slot's AICPU launch package, on a runtime whose
    // entry values can travel as launch arguments. Grow-only across the runs a
    // slot serves, and host memory only — RTS makes its own device copy from it
    // during the launch call. Per slot rather than per run because a slot admits
    // one run at a time, which is the same reservation that protects the device
    // block above; and here rather than in the per-run helper so the growth is
    // paid once per slot instead of once per run.
    std::vector<std::byte> launch_package;
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
        runtime_image_(std::move(other.runtime_image_)),
        runtime_args_state_(std::exchange(other.runtime_args_state_, RuntimeArgsState::Empty)),
        initializing_slot_(std::exchange(other.initializing_slot_, nullptr)),
        slot_(std::exchange(other.slot_, nullptr)),
        plan_(other.plan_),
        launch_payload_(nullptr),
        launch_payload_bytes_(std::exchange(other.launch_payload_bytes_, 0)) {
        // The payload points either into the slot's staging or at this object's
        // own `args`, so the moved-to object re-derives it rather than
        // inheriting a pointer into the source.
        launch_payload_ = (other.launch_payload_ == nullptr)            ? nullptr :
                          (launch_payload_bytes_ == sizeof(KernelArgs)) ? static_cast<void *>(&args) :
                                                                          other.launch_payload_;
        other.launch_payload_ = nullptr;
        other.plan_ = LaunchEntryArgsPlan{};
        other.args = KernelArgs{};
    }
    KernelArgsHelper &operator=(KernelArgsHelper &&) = delete;

    KernelArgs args;
    MemoryAllocator *allocator_{nullptr};

    // Reserve the slot's destination and snapshot this invocation's device-read
    // descriptor. An unpublished snapshot rejects another prepare without
    // changing its source, destination, or allocator. After publication or
    // release, a fresh prepare withdraws the previous publication status.
    int prepare_runtime_args(const Runtime &host_runtime, MemoryAllocator &allocator, SlotPersistentArgs &slot);

    /**
     * Consume the snapshot with one synchronous metadata H2D, having recorded
     * which route this run's entry values take.
     *
     * `launch_route_permitted` says the caller has established that this
     * launch may carry the values as launch arguments — on the streams this
     * repo owns, that this run's own AICPU stream is not capturing. It is a
     * permission, not a request: a runtime with no launch route, and the first
     * publication onto a block (which sends the whole initialized prefix
     * anyway), stay on the descriptor route regardless.
     *
     * Everything published comes from the snapshot taken at prepare. The route
     * decision and this run's counts are patched into it; nothing is re-read
     * from the caller's `Runtime`, which by now may hold a successor's values.
     *
     * The slot remains owned even on failure. Callers must check the return
     * code and abort the run on error — the state stays unpublished, so no
     * kernel may be submitted. A repeated publish is rejected without another
     * copy.
     */
    int publish_runtime_args(bool launch_route_permitted);

    // A non-null destination alone may still contain a previous run's bytes.
    // This verdict covers only the Runtime descriptor, not late DFX publication.
    bool runtime_args_published() const {
        return runtime_args_state_ == RuntimeArgsState::Published && args.runtime_args != nullptr;
    }

    // Whether this run holds a snapshot that has yet to be published. A launch
    // entry admits this as well as Published, because publication is what the
    // launch does first; only a kernel submission requires Published.
    bool runtime_args_prepared() const {
        return runtime_args_state_ == RuntimeArgsState::Prepared && args.runtime_args != nullptr;
    }

    // What a launch entry admits: a run that has published, or one that still
    // can. Both arches gate on this, so the two cannot disagree about which
    // states reach a launch.
    bool launchable() const { return runtime_args_prepared() || runtime_args_published(); }

    /**
     * The AICPU launch payload and its length for this run.
     *
     * The envelope — this header followed by the entry values — when the launch
     * route was taken, and `KernelArgs` alone otherwise. Meaningful only once
     * `publish_runtime_args` has returned success; before that no route has
     * been recorded and this is null.
     *
     * Call it at the submission, not before: it re-copies the header from
     * `args` as it now stands, so fields armed after publication — the wall
     * buffer, the collector bases, this run's terminal bank — are the ones RTS
     * copies. That is the same point the AICore launch reads `args` from, and
     * it costs no second device copy. The entry region is untouched: those
     * bytes came from the prepare snapshot and stay that run's.
     */
    void *launch_payload();
    size_t launch_payload_bytes() const { return launch_payload_bytes_; }

    /**
     * Drop this run's view of the slot's device blocks.
     *
     * The blocks themselves stay committed for the next run on this slot; only
     * the per-run `KernelArgs` stops naming them.
     */
    void release_run_view() {
        runtime_image_.clear();
        runtime_args_state_ = RuntimeArgsState::Empty;
        args.runtime_args = nullptr;
        initializing_slot_ = nullptr;
        slot_ = nullptr;
        plan_ = LaunchEntryArgsPlan{};
        launch_payload_ = nullptr;
        launch_payload_bytes_ = 0;
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
    enum class RuntimeArgsState : uint8_t { Empty, Prepared, Published };

    RuntimeLaunchImage runtime_image_;
    RuntimeArgsState runtime_args_state_{RuntimeArgsState::Empty};

    // The slot whose pending snapshot carries the handshake region, or null
    // when the snapshot is an ordinary shorter one. `publish_runtime_args`
    // records the block as initialized through this, and only after its copy
    // returns success — so the fact is committed by the same call that earns
    // it, and no caller can commit it early by forgetting the order.
    SlotPersistentArgs *initializing_slot_{nullptr};

    // The slot this run prepared against, which owns both the destination and
    // the host staging a launch package is built in. Held for the whole run
    // because publication happens at launch, by which time the caller's
    // `Runtime` is no longer this run's only reader.
    SlotPersistentArgs *slot_{nullptr};

    // This run's entry-argument routing facts, read while the source was still
    // this run's. The launch side works from this and the snapshot alone.
    LaunchEntryArgsPlan plan_{};

    void *launch_payload_{nullptr};
    size_t launch_payload_bytes_{0};

    // Fill the slot's staging with this run's launch package, reading the entry
    // values out of the captured snapshot. Returns false when the snapshot does
    // not contain the windows the plan names.
    bool build_launch_package();
};

/**
 * This run's one descriptor publication, immediately before its launch.
 *
 * Asks `aicpu_stream` whether the launch route is open, then consumes the
 * snapshot prepare captured. Returns 0 only when the copy succeeded and the run
 * is Published — the state a kernel submission requires. A non-zero return is
 * the copy's own error and leaves the run unpublished with no launch payload,
 * so the caller reports it with the run NotStarted and submits nothing.
 *
 * Idempotent on an already-published run, which is what lets a launch path call
 * it unconditionally.
 */
int publish_for_launch(KernelArgsHelper &kernel_args, rtStream_t aicpu_stream);

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
