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
 * Runtime operations owned by the kernel-context lifecycle — the complete
 * vocabulary available to context init and close.
 *
 * Deliberately absent from this table: device/stream synchronization, device
 * reset, ACL finalization, capture queries, model handles, and
 * stream-to-model attachment. Keeping those operations unrepresentable makes
 * the host-only lifecycle tests an architectural guard instead of a mock
 * that can silently exercise forbidden behavior.
 */
struct KernelContextOps {
    void *context{nullptr};
    int (*get_current_device)(void *context, int *device_id) noexcept {nullptr};
    int (*create_hidden_stream)(void *context, void **stream) noexcept {nullptr};
    int (*destroy_hidden_stream)(void *context, void *stream) noexcept {nullptr};
    int (*create_event)(void *context, void **event) noexcept {nullptr};
    int (*destroy_event)(void *context, void *event) noexcept {nullptr};

    bool valid() const {
        return get_current_device != nullptr && create_hidden_stream != nullptr && destroy_hidden_stream != nullptr &&
               create_event != nullptr && destroy_event != nullptr;
    }
};

/**
 * Allocation-free operation table for one borrowed-stream kernel-mode
 * enqueue — the complete vocabulary available to a launch. Synchronization,
 * allocation, stream/event creation, capture inspection, and model
 * attachment cannot be expressed. The opaque context normally points to a
 * stack snapshot owned by the caller.
 *
 * The unrepresentability guarantee binds only launch code that routes every
 * runtime call through this table; routing through it is the launch
 * implementation's obligation, and its call-sequence tests are what hold the
 * obligation.
 */
struct KernelLaunchOps {
    void *context{nullptr};
    int (*wait_event)(void *context, void *stream, void *event) noexcept {nullptr};
    int (*memset_handshake)(void *context, void *stream) noexcept {nullptr};
    int (*record_event)(void *context, void *event, void *stream) noexcept {nullptr};
    int (*launch_aicpu)(void *context, void *stream) noexcept {nullptr};
    int (*launch_aicore)(void *context, void *stream) noexcept {nullptr};

    bool valid() const {
        return wait_event != nullptr && memset_handshake != nullptr && record_event != nullptr &&
               launch_aicpu != nullptr && launch_aicore != nullptr;
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
 * The two streams a context creates and owns. Only Aicore is hidden in the
 * capture sense; Aicpu is a dedicated private stream. Neither is the caller's.
 */
enum class KernelStreamKind : uint8_t {
    Aicpu = 0,
    Aicore,
    Count,
};

/**
 * One event per edge of the chained caller ⇄ aicpu ⇄ aicore synchronization,
 * plus the call tail. Caller and aicore are never adjacent, so no event joins
 * them directly and ACLGraph capture propagates in two hops.
 *
 * AicoreStart is recorded on the aicpu stream *before* the AICPU launch: the
 * AICPU orchestrator spins on AICore's handshake report, so an AicoreStart
 * recorded after the launch could only fire once the AICPU task completed,
 * which is a deadlock.
 *
 * An event may persist across many captured graphs, but each wait must share
 * the capture context of that event's most recent record. Waiting during
 * capture on a record issued before capture crosses the capture boundary and
 * is rejected by CANN (107024). Launch therefore enqueues every record and its
 * matching wait inside the same capture, and preparation publishes no tail for
 * a later captured launch to consume: it enqueues on the context's own aicpu
 * stream, which every launch also enqueues on, so stream FIFO carries that
 * ordering instead.
 */
enum class KernelEventKind : uint8_t {
    Start = 0,   /* caller → aicpu fork */
    AicoreStart, /* aicpu → aicore fork */
    AicoreDone,  /* aicore → aicpu join */
    AicpuDone,   /* aicpu → caller join */
    SerialTail,  /* caller-visible call tail; the stream-switch gate reads it */
    Revoke,      /* close-time context revocation completion, recorded on the aicpu stream */
    Count,
};

/**
 * Context-lifetime state for the borrowed kernel execution mode.
 *
 * This object owns the dedicated AICPU stream, hidden AICore stream and event set; every
 * graph-visible persistent execution resource belongs here rather than in a
 * per-invocation object. The caller stream is never stored or destroyed —
 * each launch receives it as a borrowed argument.
 *
 * Initialization advances New to Collecting; enqueueing readiness advances
 * Collecting to ReadyEnqueued. A
 * partial-enqueue failure poisons either live
 * phase. Close advances either live phase or Poisoned through Closing to
 * Closed.
 * Initializing is held only inside initialize()'s critical section and the
 * lock is released with the
 * phase already past it, so no caller observes it;
 * close()'s rejection of it is defensive. The stream and event sets
 * are the launch protocol's shared vocabulary, common to both runtimes, so initialize() creates all of them
 * unconditionally.
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

    int initialize(int requested_device_id, const KernelContextOps &ops, uint64_t context_generation = 1);
    int prepare_resources(const KernelResourceLayout &layout, const KernelResourceOps &ops);
    int freeze_resources();
    // The caller holds the context lease and serializes binding with close.
    int inspect_frozen_resources(
        int device_id, uint64_t generation, uint64_t schema, const uint64_t *required, size_t count,
        KernelResourceBinding &out
    ) const;
    int bind_resources_for_launch(
        int device_id, uint64_t generation, uint64_t schema, const uint64_t *required, size_t count,
        KernelResourceBinding &out
    ) const;
    bool resources_prepared() const;
    bool resources_frozen() const;
    int mark_ready_enqueued();
    void poison(int runtime_error);
    // Revoke Host admission without releasing handles. A kernel owner uses
    // this before asynchronous Device unregister, then calls close only after
    // its completion receipt is observed. The submission lease is still held.
    int begin_closing();
    int close();

    KernelContextPhase phase() const;
    bool accepts_dispatch() const;
    int device_id() const;
    int last_runtime_error() const;
    int unexpected_teardown_error() const;
    bool has_live_resources() const;
    void *hidden_stream(KernelStreamKind kind) const;
    void *event(KernelEventKind kind) const;

private:
    int cleanup_owned_resources_locked();
    bool has_live_resources_locked() const;

    mutable std::mutex mutex_;
    KernelContextPhase phase_{KernelContextPhase::New};
    int device_id_{-1};
    uint64_t context_generation_{0};
    KernelDeviceResources resources_;
    int last_runtime_error_{0};
    int unexpected_teardown_error_{0};
    KernelContextOps ops_{};
    std::array<void *, static_cast<size_t>(KernelStreamKind::Count)> hidden_streams_{};
    std::array<void *, static_cast<size_t>(KernelEventKind::Count)> events_{};
};
