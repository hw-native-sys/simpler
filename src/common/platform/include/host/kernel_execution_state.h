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
 * Which execution mode owns a device runner. A runner has exactly one mode
 * for its whole lifetime: program mode retains the historical
 * owned-device/reset semantics; kernel mode borrows the caller's
 * already-current device and may only ever release resources the kernel
 * context itself created.
 */
enum class ClaimedExecutionMode : uint8_t {
    Unclaimed = 0,
    Program,
    Kernel,
    Closed,
};

/**
 * Mutual-exclusion claim between the two modes. Claiming an already-claimed
 * mode again is idempotent; claiming the other mode is rejected with
 * PTO_RUNTIME_ERR_INVALID_STATE. A kernel claim whose initialization fails
 * before any graph-visible resource is published rolls back through
 * abort_kernel_initialization().
 */
class ExecutionModeClaimState {
public:
    int claim_program();
    int claim_kernel();
    int abort_kernel_initialization();
    int mark_closed();

    ClaimedExecutionMode mode() const;
    bool accepts_program_calls() const;
    bool accepts_kernel_calls() const;
    bool requires_explicit_kernel_close() const;

private:
    int claim(ClaimedExecutionMode requested);

    mutable std::mutex mutex_;
    ClaimedExecutionMode mode_{ClaimedExecutionMode::Unclaimed};
};

enum class KernelStreamKind : size_t {
    Aicpu = 0,
    Aicore,
    Count,
};

/**
 * Handle operations owned by the kernel-context lifecycle — the complete
 * vocabulary for creating/destroying streams and events. Device allocation
 * and release are isolated in KernelResourceOps for resource prepare/close.
 *
 * Deliberately absent from this table: device/stream synchronization, device
 * reset, ACL finalization, capture queries, model handles, and
 * stream-to-model attachment. Keeping those operations unrepresentable makes
 * the host-only lifecycle tests an architectural guard instead of a mock
 * that can silently exercise forbidden behavior.
 */
struct KernelContextOps {
    void *context{nullptr};
    int (*get_current_device)(void *context, int *device_id){nullptr};
    int (*create_stream)(void *context, KernelStreamKind kind, void **stream){nullptr};
    int (*destroy_stream)(void *context, KernelStreamKind kind, void *stream){nullptr};
    int (*create_event)(void *context, void **event){nullptr};
    int (*destroy_event)(void *context, void *event){nullptr};

    bool valid() const {
        return get_current_device != nullptr && create_stream != nullptr && destroy_stream != nullptr &&
               create_event != nullptr && destroy_event != nullptr;
    }
};

/**
 * Allocation-free operation table for one borrowed-stream kernel-mode
 * enqueue — the complete vocabulary available to a launch. Synchronization,
 * allocation, stream/event creation, capture inspection, and model
 * attachment cannot be expressed. The opaque context normally points to a
 * stack snapshot owned by the caller.
 */
struct KernelLaunchOps {
    void *context{nullptr};
    int (*wait_event)(void *context, void *stream, void *event) noexcept {nullptr};
    int (*memset_handshake)(void *context, void *stream) noexcept {nullptr};
    int (*record_event)(void *context, void *event, void *stream) noexcept {nullptr};
    int (*launch_aicpu)(void *context, void *stream) noexcept {nullptr};
    int (*launch_aicore)(void *context, void *stream) noexcept {nullptr};
    int (*cancel_waiting_aicore)(void *context, void *stream) noexcept {nullptr};

    bool valid() const {
        return wait_event != nullptr && memset_handshake != nullptr && record_event != nullptr &&
               launch_aicpu != nullptr && launch_aicore != nullptr && cancel_waiting_aicore != nullptr;
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

enum class KernelEventKind : size_t {
    PrepareTail = 0,
    Start,
    AicoreDone,
    AicpuDone,
    SerialTail,
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
 * Phase machine:
 *   New → (initialize) → Collecting ⇄ ReadyEnqueued
 *   Collecting/ReadyEnqueued → (poison, on partial-enqueue failure) → Poisoned
 *   Collecting/ReadyEnqueued/Poisoned → (close) → Closing → Closed
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
 * a use-after-free; the public integration must retain an unclosed kernel
 * runner and its allocator rather than invoke program teardown.
 */
class KernelExecutionState {
public:
    KernelExecutionState() = default;
    ~KernelExecutionState() = default;
    KernelExecutionState(const KernelExecutionState &) = delete;
    KernelExecutionState &operator=(const KernelExecutionState &) = delete;

    int initialize(int requested_device_id, const KernelContextOps &ops, uint64_t context_generation);
    int prepare_resources(const KernelResourceLayout &layout, const KernelResourceOps &ops);
    int freeze_resources();
    // Capture-external sealing reads frozen addresses before init work is ready.
    // The caller holds the context lease and serializes this operation with close.
    int inspect_frozen_resources(
        int device_id, uint64_t generation, uint64_t schema, const uint64_t *required, size_t count,
        KernelResourceBinding &out
    ) const;
    // Caller serializes binding/enqueue with close, and keeps the context alive
    // until all device work and captured graphs referencing these views end.
    int bind_resources_for_launch(
        int device_id, uint64_t generation, uint64_t schema, const uint64_t *required, size_t count,
        KernelResourceBinding &out
    ) const;
    bool resources_prepared() const;
    bool resources_frozen() const;
    int mark_ready_enqueued();
    void poison(int runtime_error);
    int close();

    KernelContextPhase phase() const;
    bool accepts_dispatch() const;
    int device_id() const;
    int last_runtime_error() const;
    int unexpected_teardown_error() const;
    bool has_live_resources() const;
    void *stream(KernelStreamKind kind) const;
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
    std::array<void *, static_cast<size_t>(KernelStreamKind::Count)> streams_{};
    std::array<void *, static_cast<size_t>(KernelEventKind::Count)> events_{};
};
