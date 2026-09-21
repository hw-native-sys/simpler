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
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>

#include "runtime_c_api.h"
#include "worker/native_run_execution.h"

/**
 * Two passive device-timestamp markers per pipeline slot.
 *
 * A run's completion fence answers *whether* it finished. These answer *when*, on the device's own
 * counter, at two positions the fence brackets:
 *
 * - `AicoreStart` — on the AICore stream, after any cross-run wait and before the kernel launch:
 *   the instant that stream was released to begin this run's work.
 * - `WholeOperatorEnd` — on the AICPU stream, behind the wait on this run's *own* AICore boundary:
 *   an instant at which its AICore kernel had returned.
 *
 * Together they make one comparison expressible that neither the AICPU run wall nor any host
 * signal can: a successor's `AicoreStart` against its predecessor's `WholeOperatorEnd`.
 *
 * **No device stream waits on these.** Recording an event is not a wait and no stream ever waits on
 * one of these, so they add no ordering to the streams they sit in and cannot make a missing
 * production wait look satisfied. The production completion fence is untouched: its events stay
 * completion-only with their own contract, and these are separate handles with a separate creation
 * flag, because a timestamp needs a capability the completion flag does not promise.
 *
 * **Reading is two steps: this event's own completion, then its timestamp.** The device having
 * passed the record does not by itself make the timestamp retrievable, so `read()` establishes the
 * event's completion first, bounded by the caller's timeout, and then retrieves. A failure at
 * either step is unavailable with that rc.
 *
 * The completion step is a host call that blocks the host, and it is taken under this object's
 * mutex, so it can delay another position's record. What it does not do is reach a stream: it is
 * per event rather than a stream drain, and it queues nothing, so no device ordering edge exists
 * because of it. `read()` is called at a run's finalize, once that run's own completion has been
 * established.
 *
 * **Ownership is per slot for the runner's life, not per run.** A queued record names its event and
 * the API documents no guarantee that a *failed* record was not queued, so an event is never
 * destroyed per run. It is destroyed at teardown, and only once the streams that recorded it are
 * proven retired — a run stream whose own destroy failed is kept by `RunStreamPair` because it may
 * still hold instructions, and such a stream may still name a marker. Otherwise the events are
 * `retain()`ed for the process's remaining life and the caller reports it. The bootstrap stream
 * pair records none of these and is not part of that condition.
 *
 * **A reading is accepted only when it can be attributed to the run that asked.** Three conditions,
 * all per position:
 *
 * - that run recorded *this* position. The two positions are not recorded together — every run
 *   records `AicoreStart`, while `WholeOperatorEnd` exists only for a run that built a
 *   whole-operator boundary — so a run that recorded only the first must not be handed its
 *   predecessor's second, which no count could catch since no new record advanced it.
 * - exactly one record has happened since the last attributed reading. More means a generation
 *   went unread, and any value now may be that one.
 * - the value advances past the last attributed one.
 *
 * Failing either of the last two latches the position `uncertain` for good, as does a failed
 * synchronize or retrieval: once a recorded generation has gone unread, nothing later can be tied
 * to a generation. A bare "greater than the last successful reading" test does none of this — with
 * A read as 100, B recording 200 but its read failing, and C reading 200, it credits C with B's
 * time. The record count advances whenever `record` is *called*, whatever it returns, since a
 * failed record carries no promise the stream did not take it. Only `rc` is run-scoped; the counts,
 * the value bound and the latch belong to the event.
 *
 * **Every answer is explicit.** Not recorded by this run, event not created, record failed,
 * synchronize failed, retrieval failed, value did not advance, or uncertain — each is *unavailable*
 * with the rc that says why, never a time. A caller cannot read an absence or an error as an
 * ordering.
 *
 * Event operations are injected, so this state machine is exercisable without a device.
 */
class RunBoundaryMarks {
public:
    enum class Position : size_t { AicoreStart = 0, WholeOperatorEnd = 1, kCount = 2 };

    struct DeviceEventOps {
        /** Create a timestamp-capable event. */
        std::function<int(void **out_event)> create;
        std::function<int(void *event, void *stream)> record;
        /** Establish this event's own completion, bounded. Blocks the caller; queues nothing. */
        std::function<int(void *event, int timeout_ms)> synchronize;
        std::function<int(void *event, uint64_t *out_timestamp)> read_timestamp;
        std::function<int(void *event)> destroy;
    };

    /** One position's answer. `available` is the only thing that licenses reading `timestamp`. */
    struct Mark {
        bool available{false};
        uint64_t timestamp{0};
        int rc{0};
    };

    explicit RunBoundaryMarks(DeviceEventOps ops) :
        ops_(std::move(ops)) {}

    /**
     * Record one position for `identity` on `stream`.
     *
     * The first record for an identity claims *this position*, so a later read cannot mix two
     * runs. A create or record failure is remembered against that position rather than
     * propagated: the caller's run does not depend on a marker, and a marker the device may hold
     * must still be destroyed at teardown rather than dropped here.
     *
     * The record count advances whenever `ops_.record` is *called*, whatever it returns. A failed
     * record carries no promise that the stream did not take it, so it may still advance the
     * event's value — and a generation that may exist must be counted, or a later reading could be
     * attributed to the wrong run.
     */
    int record(Position position, const NativeRunIdentity &identity, void *stream) {
        if (identity.pipeline_slot >= PTO_PIPELINE_MAX_DEPTH || position == Position::kCount) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        PositionState &state = slots_[identity.pipeline_slot].positions[static_cast<size_t>(position)];
        // Claimed per position, not per slot: a run records `AicoreStart` always but
        // `WholeOperatorEnd` only when it built a whole-operator boundary, so the two positions of
        // one slot routinely belong to different runs. Only `rc` is run-scoped; the counters, the
        // value bound and the latch belong to the event, which outlives every run on this slot.
        if (state.recorded_by != identity) {
            state.recorded_by = identity;
            state.rc = 0;
        }
        if (state.event == nullptr) {
            int rc = ops_.create(&state.event);
            if (rc != 0) {
                state.event = nullptr;
                state.rc = rc;
                return rc;
            }
        }
        int rc = ops_.record(state.event, stream);
        ++state.records;
        state.rc = rc;
        return rc;
    }

    /**
     * What one position says for `identity`, or unavailable with the reason.
     *
     * `timeout_ms` bounds this event's own completion step, which blocks the calling host thread
     * and is taken under this object's mutex. Call at a run's finalize, once that run's own
     * completion has been established.
     */
    Mark read(Position position, const NativeRunIdentity &identity, int timeout_ms) {
        Mark mark;
        if (identity.pipeline_slot >= PTO_PIPELINE_MAX_DEPTH || position == Position::kCount) {
            mark.rc = PTO_RUNTIME_ERR_INTERNAL;
            return mark;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        PositionState &state = slots_[identity.pipeline_slot].positions[static_cast<size_t>(position)];
        // Before anything cached or fresh: this run has to be the one that recorded *this*
        // position. A run that records only `AicoreStart` must not be handed the predecessor's
        // `WholeOperatorEnd`, and the attribution count alone would do exactly that — it would
        // still read as already-attributed, because no new record advanced it.
        if (state.recorded_by != identity) {
            mark.rc = PTO_RUNTIME_ERR_INTERNAL;
            return mark;
        }
        if (state.uncertain) {
            mark.rc = PTO_RUNTIME_ERR_INVALID_STATE;
            return mark;
        }
        if (state.records == 0 || state.event == nullptr || state.rc != 0) {
            mark.rc = state.rc != 0 ? state.rc : PTO_RUNTIME_ERR_INTERNAL;
            return mark;
        }
        // Already attributed to this record. Answered from what was stored rather than read again,
        // so a second call is idempotent instead of failing the advance check below.
        if (state.records == state.attributed) {
            mark.timestamp = state.last_timestamp;
            mark.available = true;
            return mark;
        }
        // More than one record since the last attributed reading means a generation was never
        // read, and any value now may be that one. Nothing after it can be attributed.
        if (state.records != state.attributed + 1) {
            state.uncertain = true;
            mark.rc = PTO_RUNTIME_ERR_INVALID_STATE;
            return mark;
        }
        // This event's own completion, then its timestamp: the device having passed the record
        // does not by itself make the timestamp retrievable. The completion step blocks this
        // thread, bounded by `timeout_ms`, and queues nothing on any stream.
        int rc = ops_.synchronize(state.event, timeout_ms);
        if (rc != 0) {
            state.uncertain = true;
            mark.rc = rc;
            return mark;
        }
        uint64_t timestamp = 0;
        rc = ops_.read_timestamp(state.event, &timestamp);
        if (rc != 0) {
            // This generation will never be attributed, so a later value could be it.
            state.uncertain = true;
            mark.rc = rc;
            return mark;
        }
        if (timestamp <= state.last_timestamp) {
            // Either the record has not taken effect for the host yet or the value is the previous
            // generation's. Nothing here can tell those apart, and neither can a later read.
            state.uncertain = true;
            mark.rc = PTO_RUNTIME_ERR_INVALID_STATE;
            return mark;
        }
        state.attributed = state.records;
        state.last_timestamp = timestamp;
        mark.timestamp = timestamp;
        mark.available = true;
        return mark;
    }

    /**
     * Destroy every event, reporting the first failure and keeping a handle that survived.
     *
     * Refuses outright once `retain()` has been called: a stream that may still hold a queued
     * record naming one of these events is exactly the condition under which destroying it is
     * not allowed, and that condition does not expire.
     */
    int release() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (retained_) return PTO_RUNTIME_ERR_INVALID_STATE;
        int first_error = 0;
        for (SlotState &slot : slots_) {
            for (PositionState &state : slot.positions) {
                if (state.event == nullptr) continue;
                int rc = ops_.destroy(state.event);
                if (rc != 0) {
                    if (first_error == 0) first_error = rc;
                    continue;
                }
                // A destroyed event's successor would be a new one with its own value sequence,
                // so the attribution state goes with it.
                state = PositionState{};
            }
        }
        return first_error;
    }

    /**
     * Keep every event for the process's remaining life, because a stream that recorded one has
     * not been proven retired.
     *
     * The alternative is destroying an event a surviving stream may still name, which is the one
     * thing this class exists to avoid. Nothing here touches the device, so it is safe on the
     * teardown path that reaches it; the retention is the cost of that stream's own failed
     * destroy, and the caller reports it.
     */
    void retain() {
        std::lock_guard<std::mutex> lock(mutex_);
        retained_ = true;
    }

    /** Whether the events are being kept because a recording stream survived. */
    bool retained() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return retained_;
    }

    /** Forget every event after a device reset, without invoking destroy. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (SlotState &slot : slots_) {
            for (PositionState &state : slot.positions) {
                state = PositionState{};
            }
        }
        retained_ = false;
    }

    /** How many events are currently owned. */
    size_t live_event_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t live = 0;
        for (const SlotState &slot : slots_) {
            for (const PositionState &state : slot.positions) {
                if (state.event != nullptr) ++live;
            }
        }
        return live;
    }

private:
    // Everything but `rc` belongs to the event rather than to a run, because the event outlives
    // every run on its slot and a reading has to be attributable across that reuse.
    struct PositionState {
        void *event{nullptr};
        /** The run whose record this position currently holds, and the only one it answers for. */
        NativeRunIdentity recorded_by{};
        /** How many times this position has been recorded on this event. */
        uint64_t records{0};
        /** The record count the last attributed reading belonged to. */
        uint64_t attributed{0};
        /** That reading's value, and the lower bound the next one must pass. */
        uint64_t last_timestamp{0};
        /** A recorded generation went unread, so no later value can be tied to one. Latched. */
        bool uncertain{false};
        /** This run's own create/record outcome. The one field a new run resets. */
        int rc{0};
    };

    struct SlotState {
        std::array<PositionState, static_cast<size_t>(Position::kCount)> positions{};
    };

    mutable std::mutex mutex_;
    std::array<SlotState, PTO_PIPELINE_MAX_DEPTH> slots_{};
    bool retained_{false};
    DeviceEventOps ops_;
};
