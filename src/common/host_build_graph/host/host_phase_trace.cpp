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
 * Prepare-path timing: per-kind counters and the binding to the platform's
 * record pool. See host_build_graph/host_phase_trace.h.
 */

#include "host_build_graph/host_phase_trace.h"

#include <stdio.h>

#include <atomic>
#include <array>
#include <cstdlib>
#include <mutex>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

// The thread-id fallback below runs only where SYS_gettid is unavailable, and it is
// the sole user of these two. SYS_gettid comes from <sys/syscall.h>, so this test
// has to follow that include.
#if !(defined(__linux__) && defined(SYS_gettid))
#include <functional>
#include <thread>
#endif

#include "common/host_api.h"
#include "common/log_clock.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "host/host_phase_records.h"

namespace {

// One producer's counters and in-flight claim, on its own cache lines.
//
// The producers of a bind are independent -- each records its own Definition and
// counts only its own operations -- so nothing here is shared, and the per-record
// path performs no atomic read-modify-write at all. Sharing them cost the emitting
// thread about 0.8 us per record at eight producers, purely in cache-line
// ownership: every producer of a Graph workload records the same kind
// (`record_sub_task`), so a per-kind counter is a single line eight threads fight
// over, and the `alignas(64)` that separates one kind from another does nothing
// about that.
//
// Plain integers, not atomics, and that is only sound because a lane has exactly
// one writer for the whole bind -- never a shared fallback -- and the reader sums
// the lanes after the bind is closed and its producers drained. A producer that
// cannot get a lane of its own does not record.
struct KindCounter {
    uint64_t total_ns{0};
    uint64_t count{0};
    uint64_t payload_sum{0};
    uint64_t first_start_ns{0};
};

struct alignas(64) ProducerLane {
    // Records accepted by `active` but not yet finished with the counters and the
    // pool. begin/end clear `active` and then drain the sum of these to zero, so a
    // record can never be reported by a bind it does not belong to.
    std::atomic<int32_t> in_flight{0};
    std::array<KindCounter, kHostPhaseKindCount> counters{};
};

// One lane per thread that can record: the submitting thread plus one per
// concurrently-recorded Definition, so the ceiling is GRAPH_MAX_DEFINITIONS + 1 =
// 17. Matches the pool's buffer count, which is sized the same way, so a producer
// that got a buffer also has a lane and neither denial is reachable in a sized
// run.
constexpr size_t kProducerLanes = PLATFORM_HOST_PHASE_BUFFERS;

struct TraceState {
    std::atomic<HostPhaseRecordPool *> pool{nullptr};
    const HostApi *api;
    std::atomic<bool> active{false};
    std::atomic<uint32_t> generation{0};
    std::atomic<uint32_t> next_lane{0};
    // Records dropped because every lane was taken. Unreachable while the lane
    // count covers the producer ceiling; counted rather than assumed so a workload
    // that outgrows it says so instead of silently under-reporting.
    std::atomic<uint64_t> lane_denied{0};
    std::mutex lifecycle_mutex;
    std::atomic<uint64_t> submitted_tasks{0};
    std::array<ProducerLane, kProducerLanes> lanes;
    // Each bind segment's attributes beyond its duration (byte counts, tensor
    // counts, this thread's fault and context-switch deltas), formatted when the
    // segment ends and printed at flush -- so a segment's values need not outlive
    // it and the log write stays off the measured path. `kBindAttrsCapacity` is
    // shared with every caller that formats one, because the copy in truncates
    // silently.
    std::array<std::array<char, kBindAttrsCapacity>, kHostPhaseKindCount> bind_attrs;
};

TraceState &state() {
    static TraceState s{};
    return s;
}

// This thread's lane for the current bind, or nullptr when every lane is taken.
//
// `generation` is what makes a cached lane safe to reuse: it is process-unique per
// bind, so a lane claimed under one bind is never mistaken for a claim under
// another. Returns the generation it claimed under, because the caller must
// re-check it once admitted -- see host_phase_record.
ProducerLane *producer_lane(TraceState &s, uint32_t &claimed_generation) {
    static thread_local ProducerLane *lane = nullptr;
    static thread_local uint32_t lane_generation = 0;
    const uint32_t generation = s.generation.load(std::memory_order_acquire);
    if (lane == nullptr || lane_generation != generation) {
        const uint32_t index = s.next_lane.fetch_add(1, std::memory_order_relaxed);
        lane = index < kProducerLanes ? &s.lanes[index] : nullptr;
        lane_generation = generation;
    }
    claimed_generation = generation;
    return lane;
}

// Publish-then-check against the trace lifecycle's clear-then-drain. Claiming a
// slot before reading `active` is what makes the pair race-free: a record that
// got in before the bind closed is waited for, and one that arrives after sees
// `active` false and withdraws. The claim is on this producer's own line, so it
// costs nothing to the others.
class RecordAdmission {
public:
    RecordAdmission(TraceState &s, ProducerLane &lane) :
        lane_(lane) {
        lane_.in_flight.fetch_add(1, std::memory_order_acq_rel);
        admitted_ = s.active.load(std::memory_order_acquire);
        if (!admitted_) lane_.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }
    ~RecordAdmission() {
        if (admitted_) lane_.in_flight.fetch_sub(1, std::memory_order_acq_rel);
    }

    RecordAdmission(const RecordAdmission &) = delete;
    RecordAdmission &operator=(const RecordAdmission &) = delete;

    explicit operator bool() const { return admitted_; }

private:
    ProducerLane &lane_;
    bool admitted_{false};
};

// Called with `active` already false, so no new record can be admitted. Only the
// recording worker can still be inside one, and commit joins it before a bind
// ends, so in practice this returns without spinning.
void drain_in_flight_records(TraceState &s) {
    for (ProducerLane &lane : s.lanes) {
        while (lane.in_flight.load(std::memory_order_acquire) != 0) {}
    }
}

bool is_bind_kind(uint32_t kind) { return kind < static_cast<uint32_t>(HostPhaseKind::OrchSubmitTask); }

// Segment names are stable for log consumers. Records use their
// measured timestamps; flush occurs after publication, outside chip.run.bind.
// Both bind and publication are depth-one children of the prepare span.
constexpr const char *kBindSpanName = "chip.run.bind";
constexpr int kBindSegmentSpanDepth = 2;

// Opt-in spelling shared by this runtime's switches, matching the runtime's
// other default-off switch, SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE, so that
// `=false` / `=off` / `=no` read as off rather than as a non-zero string.
bool env_opt_in(const char *name) {
    const char *env = std::getenv(name);
    return env != nullptr && (env[0] == '1' || env[0] == 't' || env[0] == 'T');
}

uint32_t current_thread_id() {
    // A thread's identity is fixed for its lifetime, so resolve it once. This
    // runs per record on the path being measured, where a syscall would inflate
    // the very durations it is annotating.
    static thread_local const uint32_t id = []() -> uint32_t {
#if defined(__linux__) && defined(SYS_gettid)
        return static_cast<uint32_t>(syscall(SYS_gettid));
#else
        return static_cast<uint32_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
    }();
    return id;
}

void record_counter(ProducerLane &lane, uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload) {
    KindCounter &counter = lane.counters[kind];
    if (counter.first_start_ns == 0) counter.first_start_ns = start_ns;
    counter.total_ns += end_ns - start_ns;
    counter.count++;
    counter.payload_sum += payload;
}

void append_record(TraceState &s, uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload, uint32_t index) {
    // No lock: the pool claims a slot per record with one atomic increment, so
    // concurrent recording threads do not serialize here. `pool` is republished
    // only by begin/end, which bracket every record via the admission drain.
    HostPhaseRecordPool *pool = s.pool.load(std::memory_order_acquire);
    if (pool == nullptr) return;
    host_phase_pool_append(
        pool, static_cast<HostPhaseKind>(kind), start_ns, end_ns, payload, index, current_thread_id()
    );
}

// A bind's totals for one kind, summed across the producers that recorded it.
// Valid only after the lanes are drained.
KindCounter merged_counter(const TraceState &s, uint32_t kind) {
    KindCounter merged{};
    for (const ProducerLane &lane : s.lanes) {
        const KindCounter &counter = lane.counters[kind];
        merged.total_ns += counter.total_ns;
        merged.count += counter.count;
        merged.payload_sum += counter.payload_sum;
        if (counter.first_start_ns != 0 &&
            (merged.first_start_ns == 0 || counter.first_start_ns < merged.first_start_ns)) {
            merged.first_start_ns = counter.first_start_ns;
        }
    }
    return merged;
}

void reset_lanes(TraceState &s) {
    for (ProducerLane &lane : s.lanes) {
        lane.counters = {};
    }
}

}  // namespace

bool host_phase_breakdown_enabled() {
    // Read once: the value cannot change within a process, and the orchestrator
    // asks per submit-level operation.
    static const bool enabled = env_opt_in("SIMPLER_HBG_BIND_BREAKDOWN_ENABLE");
    return enabled;
}

bool host_phase_records_enabled() {
    static const bool enabled = env_opt_in("SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE");
    return enabled;
}

// Strong definitions overriding the orchestrator core's weak fallbacks, which
// exist so builds without this translation unit still link.
__attribute__((visibility("hidden"))) uint64_t host_phase_now_ns() {
    if (!state().active.load(std::memory_order_acquire)) {
        return 0;
    }
    return static_cast<uint64_t>(simpler::log::monotonic_now_ns());
}

__attribute__((visibility("hidden"))) void
host_phase_record(uint64_t start_ns, uint64_t end_ns, uint32_t kind, uint64_t payload, uint32_t index) {
    TraceState &s = state();
    if (!s.active.load(std::memory_order_acquire) || kind >= kHostPhaseKindCount || end_ns < start_ns) {
        return;
    }
    // ORCH_PHASE_END calls this on every submit-level operation, so the load
    // above stays the whole cost when tracing is off. Admission then re-checks
    // under the in-flight claim, which is what actually closes the boundary.
    uint32_t claimed_generation = 0;
    ProducerLane *lane = producer_lane(s, claimed_generation);
    if (lane == nullptr) {
        s.lane_denied.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    RecordAdmission admission(s, *lane);
    if (!admission) {
        return;
    }
    // A bind can have closed and a new one opened between claiming the lane and
    // being admitted here, and the new bind resets the lane hand-out -- so a lane
    // claimed for the old bind may already belong to another producer, and writing
    // its plain-integer counters would be a race. A stale claim withdraws instead.
    if (s.generation.load(std::memory_order_acquire) != claimed_generation) {
        return;
    }
    record_counter(*lane, start_ns, end_ns, kind, payload);
    append_record(s, start_ns, end_ns, kind, payload, index);
}

void host_phase_record_bind(uint32_t kind, uint64_t start_ns, const char *attrs, uint64_t payload, uint64_t end_ns) {
    TraceState &s = state();
    if (!s.active.load(std::memory_order_acquire) || !is_bind_kind(kind)) {
        return;
    }
    uint32_t claimed_generation = 0;
    ProducerLane *lane = producer_lane(s, claimed_generation);
    if (lane == nullptr) {
        s.lane_denied.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    RecordAdmission admission(s, *lane);
    if (!admission) {
        return;
    }
    if (s.generation.load(std::memory_order_acquire) != claimed_generation) {
        return;
    }
    // Four early returns sit above this point, so a caller that reads counters at the
    // top of its own wrapper and lets the clock be taken here would span more than they
    // do. One that passes its own instant gets both over the same interval.
    const uint64_t end = end_ns != 0 ? end_ns : static_cast<uint64_t>(simpler::log::monotonic_now_ns());
    if (attrs != nullptr) {
        snprintf(s.bind_attrs[kind].data(), kBindAttrsCapacity, "%s", attrs);
    }
    record_counter(*lane, start_ns, end, kind, payload);
    append_record(s, start_ns, end, kind, payload, 0);
}

void host_phase_trace_begin(const void *host_api) {
    TraceState &s = state();
    std::scoped_lock lock(s.lifecycle_mutex);
    s.active.store(false, std::memory_order_release);
    drain_in_flight_records(s);
    s.api = static_cast<const HostApi *>(host_api);
    HostPhaseRecordPool *pool =
        s.api != nullptr ?
            static_cast<HostPhaseRecordPool *>(s.api->host_phase_pool_arm(host_phase_records_enabled())) :
            nullptr;
    s.pool.store(pool, std::memory_order_release);
    // Timestamps are taken whenever either sink wants them. Gating the clock on
    // one switch alone would leave the other sink recording zeros.
    s.submitted_tasks.store(0, std::memory_order_relaxed);
    s.lane_denied.store(0, std::memory_order_relaxed);
    reset_lanes(s);
    for (auto &attrs : s.bind_attrs) {
        attrs[0] = '\0';
    }
    // Retires the producers' cached lanes, so a thread that recorded in the last
    // bind claims afresh rather than keeping a lane whose counters were just
    // cleared under it. Published before `active`, so no record can see the new
    // bind with a stale lane. The counter is process-wide monotonic rather than
    // per-bind-reset, so no two binds ever share an identity.
    s.next_lane.store(0, std::memory_order_relaxed);
    s.generation.fetch_add(1, std::memory_order_release);
    s.active.store(s.pool != nullptr || host_phase_breakdown_enabled(), std::memory_order_release);
}

void host_phase_trace_note_submitted(uint64_t submitted_tasks) {
    state().submitted_tasks.store(submitted_tasks, std::memory_order_relaxed);
}

void host_phase_trace_end(const void *host_api) {
    TraceState &s = state();
    std::scoped_lock lifecycle_lock(s.lifecycle_mutex);
    if (s.api != host_api || !s.active.load(std::memory_order_relaxed)) {
        return;
    }
    s.active.store(false, std::memory_order_release);
    // Every record already past the `active` check finishes its counter and pool
    // work before the totals below are read and the pool is handed back, so no
    // lock is needed here either: the drain, not a mutex, is what makes the pool
    // quiescent.
    drain_in_flight_records(s);
    // Read the pool's own tally before handing it back, and drop the pointer
    // after: the pool belongs to the runner from finish() onward, and the bind is
    // over either way, so nothing here may outlive it. Clearing `active` also
    // makes a second end() without an intervening begin() a no-op rather than a
    // stale read plus a duplicate report.
    HostPhaseRecordPool *pool = s.pool.load(std::memory_order_relaxed);
    // Both ways a record can be lost: no buffer left in the pool, or no lane left
    // to count it in. The second is unreachable while the lane count covers the
    // producer ceiling, and is summed in rather than assumed away.
    const uint64_t dropped = (pool != nullptr ? pool->dropped.load(std::memory_order_relaxed) : 0) +
                             s.lane_denied.load(std::memory_order_relaxed);
    if (s.api != nullptr) {
        uint64_t inv = 0;
#if SIMPLER_HOST_STRACE
        // The invocation id the enclosing run's spans carry, so a per-event
        // artifact joins to them on (pid, inv).
        inv = simpler::strace::StraceScope::current_inv();
#endif
        s.api->host_phase_pool_finish(s.submitted_tasks.load(std::memory_order_relaxed), inv);
    }
    s.pool.store(nullptr, std::memory_order_release);
    if (!host_phase_breakdown_enabled()) {
        // Records were collected for a per-event reader alone, which has its own
        // report; this breakdown was not asked for.
        return;
    }

    for (uint32_t kind = 0; kind < kHostPhaseKindCount; ++kind) {
        const KindCounter counter = merged_counter(s, kind);
        const uint64_t count = counter.count;
        if (count == 0) {
            continue;
        }
        const uint64_t total_ns = counter.total_ns;
        const uint64_t payload_sum = counter.payload_sum;
        const uint64_t first_start_ns = counter.first_start_ns;
        const char *name = host_phase_kind_name(static_cast<HostPhaseKind>(kind));
        if (is_bind_kind(kind)) {
            // One occurrence per bind, so the summed time is the segment's own
            // duration. The span carries its own start because it is emitted at
            // the end of the bind, off the path being measured — the emit time
            // says nothing about when the segment ran. STRACE_HOST_SPAN_AT_A is
            // the form for exactly that; `chip.run` itself is emitted this way.
            if (count != 1) {
                LOG_WARN(
                    "bind segment %s recorded %llu times in one bind; its span reports their sum from the "
                    "earliest start",
                    name, static_cast<unsigned long long>(count)
                );
            }
            char span_name[SIMPLER_HOST_SPAN_NAME_CAPACITY];
            snprintf(span_name, sizeof(span_name), "%s.%s", kBindSpanName, name);
            STRACE_HOST_SPAN_AT_A(
                span_name, static_cast<long long>(first_start_ns), static_cast<long long>(total_ns),
                kBindSegmentSpanDepth, s.bind_attrs[kind].data()
            );
            continue;
        }
        // A summed cost share, not an interval: several of these kinds are
        // sub-operations of another, so there is no honest way to draw the total
        // on a timeline. Per-event bars come from the artifact instead.
        //
        // `detail_sum` appears only where `detail` counts something. The other
        // kinds put the identity of what they measured there — a task id, a Graph
        // key, the submission index — and printing a sum of identities invites a
        // reader to act on a number that means nothing.
        if (host_phase_kind_detail_is_quantity(static_cast<HostPhaseKind>(kind))) {
            LOG_TIMING(
                "host-orch phase=%s total_ns=%llu count=%llu detail_sum=%llu dropped=%llu", name,
                static_cast<unsigned long long>(total_ns), static_cast<unsigned long long>(count),
                static_cast<unsigned long long>(payload_sum), static_cast<unsigned long long>(dropped)
            );
            continue;
        }
        LOG_TIMING(
            "host-orch phase=%s total_ns=%llu count=%llu dropped=%llu", name, static_cast<unsigned long long>(total_ns),
            static_cast<unsigned long long>(count), static_cast<unsigned long long>(dropped)
        );
    }
}
