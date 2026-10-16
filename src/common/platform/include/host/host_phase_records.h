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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"

/**
 * Pool of host phase records, written on host_build_graph's prepare path and read
 * by whichever diagnostics are enabled for the run.
 *
 * The pool is a platform resource with two independently gated readers — the
 * chip-swimlane export at level 4, and the per-event artifact the host trace
 * tooling consumes — so it belongs to neither of them. Storage and lifetime are
 * the platform's.
 *
 * Recording is a header-inline store into platform-allocated memory rather than
 * a call through HostApi: a pass performs a few hundred operations a few
 * microseconds apart on the very path whose budget is under measurement, so the
 * per-event cost must stay at two clock reads, one slot claim and a store.
 * HostApi carries only the once-per-pass arm and finish.
 *
 * Every producer thread of a pass writes concurrently — a Graph workload records
 * one thread per concurrent Definition alongside the submitting thread — so the
 * pool is a flat array of slots claimed by one atomic increment, not a rotating
 * active buffer. See `host_phase_pool_append` for what that replaced and why.
 */

/**
 * Record storage for one pass, written by every producer thread concurrently.
 *
 * A producer claims a whole buffer and fills it alone, so its records are
 * contiguous and no cache line is ever shared with another producer's record.
 * That is the whole design: the producers are independent — each records its own
 * Definition and counts only its own operations — so nothing about a record needs
 * coordination, and the per-record path has no shared write at all.
 *
 * `generation` invalidates a producer's cached buffer when a new pass is armed;
 * without it a producer left over from the previous pass would append at its old
 * offset into a buffer `arm()` has already cleared.
 *
 * At namespace scope alongside `HostPhaseRecord` and `HostPhaseRecordBuffer`, so
 * the header-inline producer below and its callers name all three the same way.
 */
struct HostPhaseRecordPool {
    std::atomic<uint32_t> next_buffer{0};
    std::atomic<uint32_t> generation{0};
    // Only touched when a producer is denied a buffer, i.e. never in a sized run.
    std::atomic<uint64_t> dropped{0};
    HostPhaseRecordBuffer *buffers{nullptr};
    uint32_t buffer_count{0};
};

/**
 * One producer's place in the pool: which buffer it owns and how much of it is
 * used. Thread-local, so reading and advancing it is private to that thread.
 */
struct HostPhaseProducerLane {
    const HostPhaseRecordPool *pool{nullptr};
    uint32_t generation{0};
    uint32_t buffer{0};
    uint32_t used{0};
    bool denied{false};
};

inline HostPhaseProducerLane &host_phase_producer_lane() {
    static thread_local HostPhaseProducerLane lane;
    return lane;
}

/**
 * Take the next unclaimed buffer as this producer's own.
 *
 * Cold path: once per PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER records, plus once
 * per pass per producer. Returns false when every buffer is claimed, which is the
 * pool's only lossy outcome; the lane remembers it so a denied producer does not
 * retry the atomic on every later record.
 */
inline bool host_phase_lane_claim_buffer(HostPhaseRecordPool *pool, HostPhaseProducerLane &lane, uint32_t generation) {
    const uint32_t buffer = pool->next_buffer.fetch_add(1, std::memory_order_relaxed);
    lane.pool = pool;
    lane.generation = generation;
    if (buffer >= pool->buffer_count) {
        lane.denied = true;
        return false;
    }
    lane.buffer = buffer;
    lane.used = 0;
    lane.denied = false;
    return true;
}

namespace simpler::dfx {

/**
 * Allocation, lifetime and reader access for one host phase pool.
 *
 * Held by the platform runner. `arm()` is called once per prepare pass, before
 * the first bind segment; `records()` is valid from `finish()` until the next
 * `arm()`.
 */
class HostPhaseRecordStore {
public:
    HostPhaseRecordStore() = default;
    // Neither copying nor moving is representable: pool_ holds raw pointers into
    // buffers_, and the producer holds &pool_ for the duration of a pass. A copy
    // would alias the source's buffers and a move would invalidate the pointer
    // already handed out, both silently.
    HostPhaseRecordStore(const HostPhaseRecordStore &) = delete;
    HostPhaseRecordStore &operator=(const HostPhaseRecordStore &) = delete;
    HostPhaseRecordStore(HostPhaseRecordStore &&) = delete;
    HostPhaseRecordStore &operator=(HostPhaseRecordStore &&) = delete;

    /**
     * Arm a fresh pass.
     *
     * @param collect_records  whether per-event records are wanted at all
     *                         (a per-event reader is enabled for this run)
     * @return the pool to write into, or nullptr when nothing is collected
     */
    HostPhaseRecordPool *arm(bool collect_records);

    // Append a native prepare retry to its finished, unpublished logical run.
    // Producers from the prior attempt must already have drained.
    HostPhaseRecordPool *resume();

    /**
     * Close the pass.
     *
     * @param submitted_tasks  what the producer reported submitting, for a
     *                         reader's completeness check
     * @param invocation_id    the run's `[STRACE]` invocation id, so an artifact
     *                         can be joined to that run's span tree
     */
    void finish(uint64_t submitted_tasks, uint64_t invocation_id);

    /**
     * Append this pass's records to `path` as one JSON Lines object.
     *
     * At most one line per pass: a second call before the next `arm()` is a no-op,
     * so paths that both end a run -- a device-run teardown and a run that stops
     * before launch -- can each write unconditionally without duplicating a pass.
     *
     * @return 0 on success, -1 when the path cannot be written
     */
    int write_records_jsonl(const std::string &path);

    /** Records in start-time order. Empty unless a pass was armed with collection. */
    std::vector<HostPhaseRecord> records() const;

    /** Records whose kind submits a task, in start-time order. */
    std::vector<HostPhaseRecord> submit_records() const;

    /** Records whose kind is a host-to-device transfer, in start-time order. */
    std::vector<HostPhaseRecord> device_upload_records() const;

    bool armed() const { return armed_; }
    bool finished() const { return finished_; }
    uint64_t submitted_tasks() const { return submitted_tasks_; }
    uint64_t invocation_id() const { return invocation_id_; }
    /** Records the producers emitted, whether or not each one found a buffer. */
    uint64_t total_records() const { return stored_records() + dropped_records(); }
    uint64_t dropped_records() const { return pool_.dropped.load(std::memory_order_relaxed); }
    /** Capacity in records, i.e. how many fit before any is dropped. */
    static constexpr size_t capacity() {
        return static_cast<size_t>(PLATFORM_HOST_PHASE_BUFFERS) * PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER;
    }

private:
    /** Records that reached a buffer, summed across the buffers producers filled. */
    uint64_t stored_records() const;

    HostPhaseRecordPool pool_{};
    std::vector<HostPhaseRecordBuffer> buffers_{};
    bool armed_{false};
    bool finished_{false};
    bool records_written_{false};
    uint64_t submitted_tasks_{0};
    uint64_t invocation_id_{0};
};

}  // namespace simpler::dfx

/**
 * Append one record. A null pool means this pass collects no records, which is
 * the common case and costs one branch.
 *
 * No shared write: the record goes into the buffer this thread owns, at the offset
 * this thread tracks, and `count` is published on that same private line. The one
 * atomic is a buffer claim, taken once per PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER
 * records, and it measures as free.
 *
 * That matters because all of this runs after the record's own end timestamp, so
 * whatever it costs is invisible except as a gap between two records. The
 * predecessor -- a global mutex over one shared active buffer -- cost the emitting
 * thread 3.3 us per record at eight producers against 0.1 us at one, and a flat
 * per-record slot claim still cost 1.1 us because a 40-byte record leaves adjacent
 * slots, owned by different threads, sharing a cache line. Per-producer buffers
 * are flat at ~50 ns from one producer to sixteen, which is what two
 * `clock_gettime` calls cost on their own.
 *
 * A written record is visible to a reader only after that producer stops, so a
 * reader must not read a pass before its producers are done.
 * `host_phase_trace_end()` guarantees that: it clears `active` and drains the
 * in-flight count to zero before `finish()`.
 *
 * @param start_ns,end_ns  host monotonic nanoseconds
 * @param payload          task id for kinds that submit a task, else a detail count
 * @param index            submit_idx for orchestrator operations, 0 for bind segments
 * @param thread_id        OS thread identity of the producer
 */
inline void host_phase_pool_append(
    HostPhaseRecordPool *pool, HostPhaseKind kind, uint64_t start_ns, uint64_t end_ns, uint64_t payload, uint32_t index,
    uint32_t thread_id = 0
) {
    if (pool == nullptr) return;
    HostPhaseProducerLane &lane = host_phase_producer_lane();
    const uint32_t generation = pool->generation.load(std::memory_order_acquire);
    const bool fresh_pass = lane.pool != pool || lane.generation != generation;
    if (fresh_pass || (!lane.denied && lane.used >= PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER)) {
        if (!host_phase_lane_claim_buffer(pool, lane, generation)) {
            pool->dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    } else if (lane.denied) {
        pool->dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    HostPhaseRecordBuffer &buffer = pool->buffers[lane.buffer];
    buffer.records[lane.used] =
        HostPhaseRecord{start_ns, end_ns, payload, static_cast<uint32_t>(kind), index, thread_id, 0};
    lane.used++;
    // This producer is the only writer of this buffer's count, and the reader
    // needs it to know how far the buffer is filled.
    buffer.count = lane.used;
}
