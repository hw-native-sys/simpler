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

#include "runtime_c_api.h"

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

enum class KernelStreamKind : uint8_t {
    Aicpu = 0,
    Aicore,
    Count,
};

enum class KernelEventKind : uint8_t {
    PrepareTail = 0,
    Start,
    AicoreDone,
    SerialTail,
    Count,
};

/**
 * Context-lifetime state for the borrowed kernel execution mode.
 *
 * This object owns the persistent hidden stream pair and event set; every
 * graph-visible persistent execution resource belongs here rather than in a
 * per-invocation object. The caller stream is never stored or destroyed —
 * each launch receives it as a borrowed argument.
 *
 * Phase machine:
 *   New → (initialize) → Collecting → ReadyEnqueued
 *   Collecting/ReadyEnqueued → (poison, on partial-enqueue failure) → Poisoned
 *   Collecting/ReadyEnqueued/Poisoned → (close) → Closing → Closed
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
 * a use-after-free, so the handles leak instead. The owner must retire graph
 * references and wait for queued uses to finish before close(), then keep
 * this object alive until close() succeeds.
 */
class KernelExecutionState {
public:
    KernelExecutionState() = default;
    ~KernelExecutionState() = default;
    KernelExecutionState(const KernelExecutionState &) = delete;
    KernelExecutionState &operator=(const KernelExecutionState &) = delete;

    int initialize(int requested_device_id, const KernelContextOps &ops);
    int mark_ready_enqueued();
    void poison(int runtime_error);
    int close();

    KernelContextPhase phase() const;
    bool accepts_dispatch() const;
    int device_id() const;
    int last_runtime_error() const;
    int unexpected_teardown_error() const;
    bool has_live_resources() const;

private:
    int cleanup_owned_resources_locked();
    bool has_live_resources_locked() const;

    mutable std::mutex mutex_;
    KernelContextPhase phase_{KernelContextPhase::New};
    int device_id_{-1};
    int last_runtime_error_{0};
    int unexpected_teardown_error_{0};
    KernelContextOps ops_{};
    std::array<void *, static_cast<size_t>(KernelStreamKind::Count)> hidden_streams_{};
    std::array<void *, static_cast<size_t>(KernelEventKind::Count)> events_{};
};
