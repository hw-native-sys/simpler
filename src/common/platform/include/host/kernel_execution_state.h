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

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "worker/runtime_c_api.h"
#include "host/kernel_device_resources.h"

/**
 * The stream and event operations this state machine performs.
 *
 * Deliberately absent: device/stream synchronization, device reset, ACL
 * finalization, capture queries, model handles, and stream-to-model
 * attachment. What that buys is narrow and worth stating exactly — every call
 * *this object* makes is one of these, so its host-only tests cannot silently
 * exercise a forbidden operation. It is not the whole vocabulary of context
 * bring-up: binary load and the AICPU init handshake run outside this table
 * and the handshake does synchronize its own stream, which is permitted on the
 * cold path.
 */
struct KernelContextOps {
    void *context{nullptr};
    int (*get_current_device)(void *context, int *device_id) noexcept {nullptr};
    int (*get_current_context)(void *context, void **current) noexcept {nullptr};
    int (*create_hidden_stream)(void *context, void **stream) noexcept {nullptr};
    int (*destroy_hidden_stream)(void *context, void *stream) noexcept {nullptr};
    /**
     * Creates one context event. The creation flag is not a parameter: the
     * platform adapter that builds this table supplies its own constant, which
     * keeps KernelExecutionState free of CANN. A host-only test therefore
     * observes how many events are created and destroyed, and in what order —
     * not which flag the platform chose.
     */
    int (*create_event)(void *context, void **event) noexcept {nullptr};
    int (*destroy_event)(void *context, void *event) noexcept {nullptr};

    bool valid() const {
        return get_current_device != nullptr && get_current_context != nullptr && create_hidden_stream != nullptr &&
               destroy_hidden_stream != nullptr && create_event != nullptr && destroy_event != nullptr;
    }
};

enum class KernelContextPhase : uint8_t {
    New = 0,
    Initializing,
    Collecting,
    ReadyEnqueued,
    Poisoned,
    Closing,
    Closed,
};

/**
 * The two streams a context creates and owns. Both are private streams;
 * neither is the caller's. Stream creation alone does not establish capture
 * support or a captured-graph resource lease.
 */
enum class KernelStreamKind : uint8_t {
    Aicpu = 0,
    Aicore,
    Count,
};

/**
 * Reserved context events for chained caller ↔ aicpu ↔ aicore synchronization,
 * plus the call tail. Native eager ordering uses the separate per-slot
 * RunCallerOrder / RunCompletionFence event owners, not these handles.
 *
 * AicoreStart is recorded on the aicpu stream *before* the AICPU launch: the
 * AICPU orchestrator spins on AICore's handshake report, so an AicoreStart
 * recorded after the launch could only fire once the AICPU task completed,
 * which is a deadlock.
 *
 * An event may persist across many captured graphs, but each wait must share
 * the capture context of that event's most recent record. Waiting during
 * capture on a record issued before capture crosses the capture boundary and
 * is rejected by CANN (107024). Launch must therefore enqueue every record and
 * its matching wait inside the same capture; prepare must not publish a tail
 * for a later captured launch to consume.
 */
enum class KernelEventKind : uint8_t {
    Start = 0,   /* caller → aicpu fork */
    AicoreStart, /* aicpu → aicore fork */
    AicoreDone,  /* aicore → aicpu join */
    AicpuDone,   /* aicpu → caller join */
    SerialTail,  /* caller-visible call tail; the stream-switch gate reads it */
    Count,
};

/**
 * Context-lifetime state for the borrowed kernel execution mode.
 *
 * This object owns the dedicated AICPU/AICore streams and reserved event set.
 * A5 native eager runs borrow this stream pair; A2/A3 runs borrow a separate
 * RunStreamPair. Neither run bindings nor per-slot event owners destroy this
 * object's streams. The caller stream is never stored or destroyed here.
 * The owner must prove all users quiescent before close(); phase and poison
 * state alone are not completion proofs.
 *
 * Phase machine:
 *   New → (initialize) → Collecting → ReadyEnqueued
 *   Collecting/ReadyEnqueued → (poison, on partial-enqueue failure) → Poisoned
 *   Collecting/ReadyEnqueued/Poisoned → (close) → Closing → Closed
 * Resource preparation starts in Collecting; prepared resources must be frozen
 * before mark_ready_enqueued(). accepts_launch() is the sole launch predicate;
 * accepts_dispatch() covers preparation work and is not launch admission.
 * Initializing is held only inside initialize()'s critical section and the
 * lock is released with the phase already past it, so no caller observes it;
 * close()'s rejection of it is defensive. The stream and event sets are the
 * launch protocol's shared vocabulary, common to both runtimes, so
 * initialize() creates all of them unconditionally.
 *
 * A pre-enqueue validation failure leaves the phase unchanged. Poisoned
 * rejects dispatch but still accepts close. Closing is sticky: entered
 * before the first destructive teardown step, it rejects all dispatch, and a
 * cleanup failure stays in Closing for explicit close() retry — successfully
 * destroyed handles are nulled, so a retry redoes only the remainder.
 *
 * Error reporting keeps two slots so a controlled runtime error can never
 * mask a real teardown failure: last_runtime_error() latches the first
 * poison cause, unexpected_teardown_error() latches the first real cleanup
 * failure. A caller deciding an overall verdict consults them in that order
 * before reporting controlled success.
 *
 * The destructor performs no runtime calls. If a caller skips explicit close
 * while an ACLGraph can still reference these handles, freeing them would be
 * a use-after-free, so the handles leak instead. Refusing to destroy an
 * unclosed kernel runner is the public C API's half of that contract:
 * destroy_device_context refuses while this object owns stream/event handles
 * or the runner has an outstanding native run.
 */
class KernelExecutionState {
public:
    KernelExecutionState() = default;
    ~KernelExecutionState() = default;
    KernelExecutionState(const KernelExecutionState &) = delete;
    KernelExecutionState &operator=(const KernelExecutionState &) = delete;

    // Adopts the caller's current device/context. context_generation is the
    // nonzero caller identity checked by all later resource bindings.
    int initialize(int requested_device_id, const KernelContextOps &ops, uint64_t context_generation = 1);
    // First preparation is allowed only in Collecting. Repeats are idempotent
    // checks in Collecting/ReadyEnqueued: schema, arena kinds and offsets must
    // match, regions may not grow, and the returned views follow the requested
    // sizes. A failed rollback moves the context to Closing.
    int prepare_resources(const KernelResourceLayout &layout, const KernelResourceOps &ops);
    // Freezes the prepared allocation capacity. A context with prepared
    // resources must freeze before mark_ready_enqueued().
    int freeze_resources();
    // Copies the frozen views for registration preparation. Allowed while
    // Collecting or ReadyEnqueued; out owns its snapshot.
    int inspect_frozen_resources(
        int device_id, uint64_t generation, uint64_t schema, const uint64_t *required, size_t count,
        KernelResourceBinding &out
    ) const;
    // Copies the frozen views for a launch. Requires accepts_launch(); out owns
    // its snapshot, but the addresses are valid only while resources are live.
    int bind_resources_for_launch(
        int device_id, uint64_t generation, uint64_t schema, const uint64_t *required, size_t count,
        KernelResourceBinding &out
    ) const;
    bool resources_prepared() const;
    bool resources_frozen() const;
    /**
     * Check the caller's current device and RTS context without rebinding.
     * Before adoption there is nothing to validate. The borrowed identity
     * survives close: other runner owners may still need a cleanup retry
     * after this object's streams and events have been released.
     */
    int validate_current_context() const;
    // Enters the launch-ready phase. Idempotent in ReadyEnqueued. If resources
    // have been prepared, they must already be frozen.
    int mark_ready_enqueued();
    // Admission for preparation work (for example callable publication), open
    // in Collecting and ReadyEnqueued. This is not the launch predicate.
    bool accepts_dispatch() const;
    // Launch predicate: ReadyEnqueued, with prepared resources frozen. A
    // context that never prepared resources may launch without a binding.
    bool accepts_launch() const;
    void poison(int runtime_error);
    // Stops admission without releasing handles needed by runtime-specific cleanup.
    // Idempotent in Closing/Closed; close() completes resource release.
    int begin_close();
    // Releases resources, then streams/events. A failed release stays Closing
    // for an explicit retry; the destructor performs no runtime calls.
    int close();

    KernelContextPhase phase() const;
    int device_id() const;
    int last_runtime_error() const;
    int unexpected_teardown_error() const;
    bool has_live_resources() const;
    void *hidden_stream(KernelStreamKind kind) const;
    void *event(KernelEventKind kind) const;

private:
    int validate_current_context_locked() const;
    int cleanup_owned_resources_locked();
    bool has_live_resources_locked() const;
    bool accepts_launch_locked() const;

    mutable std::mutex mutex_;
    KernelContextPhase phase_{KernelContextPhase::New};
    int device_id_{-1};
    uint64_t context_generation_{0};
    KernelDeviceResources resources_;
    void *borrowed_context_{nullptr};
    int last_runtime_error_{0};
    int unexpected_teardown_error_{0};
    KernelContextOps ops_{};
    std::array<void *, static_cast<size_t>(KernelStreamKind::Count)> hidden_streams_{};
    std::array<void *, static_cast<size_t>(KernelEventKind::Count)> events_{};
};
