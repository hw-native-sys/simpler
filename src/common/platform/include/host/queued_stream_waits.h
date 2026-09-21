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
#include <mutex>
#include <utility>

#include "native_run_execution.h"
#include "run_completion_fence.h"
#include "runtime_c_api.h"

/**
 * Every stream wait a run has queued on a completion boundary, and the evidence
 * that retires each one.
 *
 * Two shapes of wait exist, and they differ only in what proves the wait was
 * consumed:
 *
 * - **Own.** A run queues, on its own AICPU stream, a wait for its own AICore
 *   boundary, and records its AICPU boundary behind that wait. Its AICPU
 *   boundary therefore covers the whole operator — both kernels — rather than
 *   only the AICPU one. The run's own boundaries completing is the proof: the
 *   wait sits ahead of a boundary that has fired.
 * - **Cross-run.** A successor queues, on its own AICore stream, a wait for the
 *   predecessor's whole-operator boundary. Nothing about the predecessor
 *   completing proves the host has observed the successor's stream passing that
 *   wait, so the wait carries a proof event of its own, recorded into the
 *   successor stream immediately behind it. That event completes when the wait
 *   is consumed — shortly after the predecessor's boundary fires, and well
 *   before the successor's kernels finish.
 *
 * The reference protocol on the boundary itself belongs to
 * `RunCompletionFence`; this holds the tokens it mints and pairs each with the
 * evidence that may release it. The entries are keyed by the *waiting* run,
 * because a run queues at most one wait of each shape, while a predecessor may
 * be named by both its own wait and a successor's.
 *
 * A proof event is created on first use and reused, for the reason the fence
 * reuses its boundary events: a reuse-capable event re-records without a reset,
 * so creating one per join would add device calls to every dispatch for no gain.
 *
 * Threading: a drain discharges while a launch on another thread may be opening
 * an entry, so the whole table is mutex-guarded. Device operations are injected
 * so the state machine is exercisable without a device.
 */
class QueuedStreamWaits {
public:
    using StreamRole = RunCompletionFence::StreamRole;

    /** Which run's completion proves this wait was consumed. */
    enum class Shape : uint8_t {
        /** The waiter's own boundaries. */
        Own,
        /** A proof event recorded behind the wait. */
        CrossRun,
    };

    explicit QueuedStreamWaits(RunCompletionFence::DeviceEventOps ops) :
        ops_(std::move(ops)) {}
    QueuedStreamWaits(const QueuedStreamWaits &) = delete;
    QueuedStreamWaits &operator=(const QueuedStreamWaits &) = delete;

    /**
     * Take an entry for one wait and hand back the boundary event to queue it
     * on, before the wait is queued.
     *
     * Reserving ahead of the wait is what makes the failure path safe, and it is
     * the fence that refuses a boundary that was never recorded — so a queued
     * wait can never name an event that nothing will record. A cross-run entry
     * also creates its proof event here, ahead of any submission, so a run that
     * cannot be proved fails while it is still rollback-able.
     *
     * @param waiter          the run queueing the wait
     * @param boundary_owner  the run whose boundary is waited on; equal to
     *                        `waiter` for an `Own` wait
     * @param boundary_fence  `boundary_owner`'s fence
     */
    int open(
        Shape shape, const NativeRunIdentity &waiter, const NativeRunIdentity &boundary_owner,
        RunCompletionFence &boundary_fence, StreamRole boundary_role, StreamRole waiter_role, void **boundary_event_out
    ) {
        if (boundary_event_out == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        *boundary_event_out = nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        Entry *entry = free_entry(shape, waiter);
        if (entry == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        void *boundary_event = boundary_fence.boundary_event(boundary_owner, boundary_role);
        if (boundary_event == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        if (shape == Shape::CrossRun && entry->proof_event == nullptr) {
            void *event = nullptr;
            int rc = ops_.create(&event);
            if (rc != 0) return rc;
            entry->proof_event = event;
        }
        int rc = boundary_fence.reserve_wait_reference(boundary_owner, boundary_role, waiter_role, &entry->reference);
        if (rc != 0) return rc;
        entry->shape = shape;
        entry->waiter = waiter;
        entry->boundary_owner = boundary_owner;
        entry->fence = &boundary_fence;
        entry->proof_recorded = false;
        entry->occupied = true;
        *boundary_event_out = boundary_event;
        return 0;
    }

    /** Promote the entry whose stream wait reached the device queue. */
    int commit(Shape shape, const NativeRunIdentity &waiter) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry *entry = find(shape, waiter);
        if (entry == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        return entry->fence->commit_wait_reference(entry->reference);
    }

    /**
     * Record the proof event behind a committed cross-run wait, on the stream
     * that holds it.
     *
     * A failure leaves the reference committed and the entry live: the wait is
     * queued, so the only thing lost is the cheap proof, and the entry falls
     * back to the quiescence proof at discharge.
     */
    int record_proof(const NativeRunIdentity &waiter, void *waiter_stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry *entry = find(Shape::CrossRun, waiter);
        if (entry == nullptr || entry->proof_event == nullptr || waiter_stream == nullptr) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (!entry->reference.committed()) return PTO_RUNTIME_ERR_INTERNAL;
        int rc = ops_.record(entry->proof_event, waiter_stream);
        if (rc != 0) return rc;
        entry->proof_recorded = true;
        return 0;
    }

    /** Drop an entry whose stream wait was never queued. */
    int revoke(Shape shape, const NativeRunIdentity &waiter) {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry *entry = find(shape, waiter);
        if (entry == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        int rc = entry->fence->revoke_wait_reference(entry->reference);
        if (rc != 0) return rc;
        clear(*entry);
        return 0;
    }

    /**
     * Retire every wait naming a boundary of `boundary_owner` that the caller's
     * evidence covers.
     *
     * `boundaries_complete` is the caller's first-hand observation that
     * `boundary_owner`'s own boundaries completed. It retires that run's own
     * intra-run wait outright, and it is also the precondition for waiting on a
     * cross-run proof event: the boundary the successor waits on has fired, so
     * its stream is already past the wait and the proof can only be pending for
     * as long as the host takes to see it. Without that observation a cross-run
     * proof is only queried, never waited on.
     *
     * Reports the first device error. Retiring nothing is not an error — the
     * caller decides what an undischarged reference costs by asking
     * `holds_reference_to` afterwards.
     */
    int discharge(const NativeRunIdentity &boundary_owner, bool boundaries_complete, int timeout_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        int first_rc = 0;
        for (Entry &entry : entries_) {
            if (!entry.occupied || entry.boundary_owner != boundary_owner) continue;
            if (!entry.reference.committed()) continue;
            if (entry.shape == Shape::Own) {
                if (!boundaries_complete) continue;
                // The stream the proof is about comes from the token because
                // the table is the token's holder, not a caller passing a role
                // alongside it: an `Own` wait sits in the run's own AICPU
                // stream, whose boundary is one of the two just observed.
                int rc = entry.fence->release_wait_reference(entry.reference, entry.reference.waiter());
                if (rc != 0) {
                    if (first_rc == 0) first_rc = rc;
                    continue;
                }
                clear(entry);
                continue;
            }
            if (!entry.proof_recorded) continue;
            bool complete = false;
            int rc = ops_.query(entry.proof_event, &complete);
            if (rc != 0) {
                if (first_rc == 0) first_rc = rc;
                continue;
            }
            if (!complete && boundaries_complete) {
                rc = ops_.wait(entry.proof_event, timeout_ms);
                if (rc != 0) {
                    if (first_rc == 0) first_rc = rc;
                    continue;
                }
                complete = true;
            }
            if (!complete) continue;
            rc = entry.fence->release_wait_reference(entry.reference, entry.reference.waiter());
            if (rc != 0) {
                if (first_rc == 0) first_rc = rc;
                continue;
            }
            clear(entry);
        }
        return first_rc;
    }

    /**
     * Retire every wait naming a boundary of `boundary_owner` against a proof
     * that covers the device as a whole — a stream-pair synchronize the caller
     * itself issued and that returned success, which covers everything queued
     * behind the boundary as well as the boundary itself.
     */
    int discharge_on_quiescence(const NativeRunIdentity &boundary_owner) {
        std::lock_guard<std::mutex> lock(mutex_);
        int first_rc = 0;
        for (Entry &entry : entries_) {
            if (!entry.occupied || entry.boundary_owner != boundary_owner) continue;
            if (!entry.reference.committed()) continue;
            int rc = entry.fence->release_wait_reference_on_quiescence(entry.reference);
            if (rc != 0) {
                if (first_rc == 0) first_rc = rc;
                continue;
            }
            clear(entry);
        }
        return first_rc;
    }

    /** Whether any live wait names a boundary of `boundary_owner`. */
    bool holds_reference_to(const NativeRunIdentity &boundary_owner) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const Entry &entry : entries_) {
            if (entry.occupied && entry.boundary_owner == boundary_owner) return true;
        }
        return false;
    }

    /**
     * Forget every entry after a verified reset, without any device call.
     *
     * A reset invalidates every device reference to the generation the waits and
     * their events belonged to, which is the one proof that covers a queued wait
     * without the stream holding it ever completing. **The fences must already
     * have been abandoned** when this runs: dropping a token is the minting
     * fence's judgement, and a fence that still recognises its token refuses,
     * which is reported rather than overridden.
     */
    int abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        int first_rc = 0;
        for (Entry &entry : entries_) {
            // The events belonged to the invalidated generation; their handles
            // name nothing now, so they are forgotten rather than destroyed.
            entry.proof_event = nullptr;
            if (!entry.occupied) continue;
            if (entry.reference.valid()) {
                int rc = entry.fence->discard_stale_wait_reference(entry.reference);
                if (rc != 0) {
                    if (first_rc == 0) first_rc = rc;
                    continue;
                }
            }
            clear(entry);
        }
        return first_rc;
    }

    /** Destroy the proof events, keeping a handle whose destroy failed. */
    int release_events() {
        std::lock_guard<std::mutex> lock(mutex_);
        int first_rc = 0;
        for (Entry &entry : entries_) {
            if (entry.occupied) {
                if (first_rc == 0) first_rc = PTO_RUNTIME_ERR_INTERNAL;
                continue;
            }
            if (entry.proof_event == nullptr) continue;
            int rc = ops_.destroy(entry.proof_event);
            if (rc != 0) {
                if (first_rc == 0) first_rc = rc;
                continue;
            }
            entry.proof_event = nullptr;
        }
        return first_rc;
    }

    /** How many waits are live. */
    size_t live_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (const Entry &entry : entries_) {
            if (entry.occupied) ++count;
        }
        return count;
    }

private:
    struct Entry {
        Shape shape{Shape::Own};
        NativeRunIdentity waiter{};
        NativeRunIdentity boundary_owner{};
        RunCompletionFence *fence{nullptr};
        RunCompletionFence::WaitReference reference{};
        void *proof_event{nullptr};
        bool proof_recorded{false};
        bool occupied{false};
    };

    /** Keep the reused proof event; drop everything that belonged to one wait. */
    static void clear(Entry &entry) {
        entry.shape = Shape::Own;
        entry.waiter = NativeRunIdentity{};
        entry.boundary_owner = NativeRunIdentity{};
        entry.fence = nullptr;
        entry.proof_recorded = false;
        entry.occupied = false;
    }

    Entry *find(Shape shape, const NativeRunIdentity &waiter) {
        for (Entry &entry : entries_) {
            if (entry.occupied && entry.shape == shape && entry.waiter == waiter) return &entry;
        }
        return nullptr;
    }

    /**
     * An entry for a wait this run does not already hold.
     *
     * One run may hold one wait of each shape, and a second wait of the same
     * shape would be a second token against one queued wait — the thing the
     * whole reference model exists to prevent — so it is refused rather than
     * given a fresh entry.
     */
    Entry *free_entry(Shape shape, const NativeRunIdentity &waiter) {
        if (find(shape, waiter) != nullptr) return nullptr;
        for (Entry &entry : entries_) {
            if (!entry.occupied && !entry.reference.valid()) return &entry;
        }
        return nullptr;
    }

    mutable std::mutex mutex_;
    // One own wait and one cross-run wait per pipeline slot is the most the
    // launch path can queue, and the slot count is the runner's admission bound.
    std::array<Entry, PTO_PIPELINE_MAX_DEPTH * 2> entries_{};
    RunCompletionFence::DeviceEventOps ops_;
};
