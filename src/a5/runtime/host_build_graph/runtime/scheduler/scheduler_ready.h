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

#include "scheduler_memory.h"
#include "scheduler_graph.h"
#include "scheduler_ssbuf.h"

enum class SchedulerRouteResult : uint64_t {
    READY_TO_ENQUEUE = 1,
    WAITING = 2,
    COMPLETED = 3,
    ERROR = 4,
};

enum class SchedulerPredicateResult : uint8_t {
    PASS = 0,
    FAIL = 1,
    MALFORMED = 2,
};

struct SchedulerWakeStats {
    uint64_t fanin_state_load_count{0};
    uint64_t wake_register_count{0};
    uint64_t wake_cas_retry_count{0};
    uint64_t wake_closed_retry_count{0};
    uint64_t wake_migrate_count{0};
    uint64_t wake_close_count{0};
};

struct SchedulerReadyStats {
    uint64_t enqueue_count{0};
    uint64_t batch_count{0};
    uint64_t pop_count{0};
    uint64_t steal_count{0};
    uint64_t cas_retry_count{0};
    uint64_t contention_giveup_count{0};
};

struct SchedulerCompletionStats {
    uint64_t enqueue_count{0};
    uint64_t resolve_count{0};
};

struct SchedulerReadyBatch {
    int64_t head{SCHEDULER_INBOX_EMPTY};
    int64_t tail{SCHEDULER_INBOX_EMPTY};
    uint64_t count{0};
};

inline __aicore__ void scheduler_ready_batch_reset(SchedulerReadyBatch *batch) {
    batch->head = SCHEDULER_INBOX_EMPTY;
    batch->tail = SCHEDULER_INBOX_EMPTY;
    batch->count = 0;
}

// A ready inbox addresses its two endpoints as int32 task refs inside one
// 64-bit word, low half the head. scheduler_plan_layout rejects a graph whose
// refs would not fit, and a push re-checks its own batch before publishing it.
//
// The endpoints are inclusive: the head names the oldest queued task and the
// tail the newest, both real tasks rather than slots or a sentinel. So
// `head == tail` is a queue holding exactly one task, and an empty queue is
// both endpoints at SCHEDULER_INBOX_EMPTY -- not the ring-buffer convention
// where the two meeting means empty.
inline __aicore__ uint64_t scheduler_ready_endpoints_pack(int64_t head, int64_t tail) {
    const uint64_t packed_head = static_cast<uint32_t>(static_cast<int32_t>(head));
    const uint64_t packed_tail = static_cast<uint32_t>(static_cast<int32_t>(tail));
    return packed_head | (packed_tail << 32);
}

inline __aicore__ int64_t scheduler_ready_endpoints_head(uint64_t packed) {
    return static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(packed)));
}

inline __aicore__ int64_t scheduler_ready_endpoints_tail(uint64_t packed) {
    return static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(packed >> 32)));
}

inline __aicore__ bool scheduler_ready_endpoint_fits(int64_t task_ref) {
    return task_ref >= SCHEDULER_INBOX_EMPTY && task_ref <= INT32_MAX;
}

// Either both endpoints name a task or neither does; one of each is corruption.
inline __aicore__ bool scheduler_ready_endpoints_consistent(uint64_t packed) {
    const int64_t head = scheduler_ready_endpoints_head(packed);
    const int64_t tail = scheduler_ready_endpoints_tail(packed);
    if (head < SCHEDULER_INBOX_EMPTY || tail < SCHEDULER_INBOX_EMPTY) return false;
    return (head == SCHEDULER_INBOX_EMPTY) == (tail == SCHEDULER_INBOX_EMPTY);
}

// A queue is empty at both endpoints at once, so either half answers for both.
// Every write to the pair is one CAS over the whole word, which is what keeps
// that true: no reachable state names a task at one endpoint and not the other.
inline __aicore__ bool scheduler_ready_endpoints_empty(uint64_t packed) {
    return scheduler_ready_endpoints_head(packed) == SCHEDULER_INBOX_EMPTY;
}

// How long a consumer waits for a push to store the link onto the tail it is
// standing on. The push stores it a few instructions after its CAS and AICore
// does not preempt, so exhausting this means the endpoints moved instead; the
// caller re-reads them.
inline constexpr uint32_t SCHEDULER_READY_LINK_SPIN_LIMIT = 256;

struct SchedulerReadyClaim {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t inbox_index{UINT64_MAX};
    SchedulerReadySource source{SchedulerReadySource::LOCAL};
    SchedulerPublicationMode publication_mode{SchedulerPublicationMode::DISPATCH};
    uint64_t state_probe_start_cycles{0};
    uint64_t state_probe_end_cycles{0};
};

struct SchedulerFreeSlotClaim {
    uint64_t worker_id{UINT64_MAX};
    uint32_t generation{0};
    uint32_t cluster_lane{UINT32_MAX};
    // Whether this claim is for a peer lane rather than for this core's own
    // running position. The two have separate state, payloads, generations and
    // publication tokens, so everything that touches a claim needs to know
    // which.
    bool mix{false};
    // Which of the target lane's two slots this dispatch goes in -- the one the
    // lane is not running. Meaningless for a claim that is not `mix`: a core
    // runs one task at a time out of a single position of its own.
    bool lane_slot{false};
    // Which block of the task this dispatch runs, and how many the task has.
    // Both travel with the claim because the claim already read them to advance
    // the task's cursor, and a Mix dispatch prepares three payloads from one
    // claim. A task that is not SPMD is not a special case here: it has one
    // block, so these are 0 and 1 and every ordinary dispatch site needs no
    // branch to stay correct.
    uint16_t block_idx{0};
    uint16_t block_num{1};
};

// Where a Mix dispatch is published: the slot of the target lane that the lane
// is not currently running. Only a Mix subtask is published through SSBUF at
// all -- it is the one dispatch a core does not fill for itself, so its holder
// has to be told. A core's own dispatch stays in local state, which is where
// its executor reads it.
// Which of the target lane's two slots a dispatch goes in: the one the lane is
// not running. A lane has exactly two, so this is a choice rather than an
// index, and saying so in the type is what makes it impossible to name a slot
// that does not exist.
inline __aicore__ bool
scheduler_lane_pending_slot(const SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, uint32_t cluster_lane) {
    return scheduler_ssbuf_load_relaxed(&ssbuf_region->lanes[cluster_lane].current) == 0;
}

inline __aicore__ SCHEDULER_SSBUF SchedulerSsbufDispatchControl *scheduler_mix_dispatch_control(
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, const SchedulerFreeSlotClaim &claim
) {
    return &ssbuf_region->lanes[claim.cluster_lane].dispatch[claim.lane_slot];
}

struct SchedulerLocalSlotState {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t generation{0};
    SchedulerDispatchSlotState state{SchedulerDispatchSlotState::EMPTY};
    uint8_t subtask_slot{UINT8_MAX};
    uint8_t mix_tracker{UINT8_MAX};
    // Whether this core took the dispatch out of a slot its coordinator fills
    // rather than claiming it for itself, and which of that lane's two slots it
    // came from. Together they decide where the completion goes: a coordinator
    // is counting this dispatch's siblings, and it tracks the two slots apart,
    // so the report has to say which one finished.
    bool from_mailbox{false};
    bool lane_slot{false};
    // The block this slot is running and how many the task has. The count is
    // what tells a completion whether the task needs cross-block aggregation at
    // all, which is why it is held here rather than read back from the task:
    // with timing off this path touches no SchedulerTaskMetadata, and in S1 the
    // same value is what the block counter is compared against.
    uint16_t block_idx{0};
    uint16_t block_num{1};
};

struct SchedulerWorkerTraceCache {
    uint64_t aicore_entry_cycles{0};
    uint64_t handshake_publish_cycles{0};
    uint64_t register_release_cycles{0};
    uint64_t descriptor_cache_observed_cycles{0};
};

// Immutable for one READY publication. Offsets address GM. Configuration and
// Scheduler-private mutable bookkeeping live in core-local storage; shared
// coordination state remains in GM.
struct SchedulerLocalConfig {
    __gm__ const SchedulerWorkerContext *shared_context{nullptr};
    const SchedulerGraphView *graph{nullptr};
    uint64_t task_controls_offset{0};
    uint64_t ready_inboxes_offset{0};
    uint64_t ready_directory_offset{0};
    uint64_t mix_payloads_offset{0};
    uint64_t mix_traces_offset{0};
    uint64_t graph_summary_offset{0};
    // The callable's registration-owned entry table, named by absolute address
    // because it is not part of the scheduler state these offsets index. The
    // address cannot be narrowed the way the offsets are; the length can, and
    // is validated below.
    uint64_t callable_addresses_address{0};
    uint32_t worker_contexts_offset{0};
    uint32_t callable_addresses_count{0};
    uint32_t task_metadata_offset{0};
    uint32_t dispatch_payloads_offset{0};
    uint16_t worker_ids[PLATFORM_CORES_PER_BLOCKDIM]{};
    uint16_t runtime_worker_count{0};
    uint16_t scheduler_index{UINT16_MAX};
    uint16_t scheduler_count{0};
    // Each cluster lane's index within its own core type, which is how every
    // ready queue that lane can hold is addressed. Read from the lane's own
    // worker context rather than derived, so nothing here assumes an AIC's rank
    // and its cluster index coincide.
    uint16_t cluster_type_ranks[PLATFORM_CORES_PER_BLOCKDIM]{};
    // Clusters in this run, which fixes how many of each queue's inboxes are
    // live: one per cluster for the AIC and Mix queues, two for the AIV queue.
    uint16_t cluster_count{0};
    uint8_t self_lane{UINT8_MAX};
    // This core's own type, which is also the only type of ready work it can
    // claim for itself. Held here so the dispatch path needs no worker-context
    // read to know which inbox is its own.
    uint32_t self_core_type{UINT32_MAX};
};

struct SchedulerLocalProfilingState {
    uint64_t trace_cells_offset{0};
    uint64_t activity_buffers_offset{0};
    // The trace of the task this core is running. One, because it runs one.
    SchedulerExecutorTaskTrace executor_trace{};
    SchedulerWorkerTraceCache worker_traces[PLATFORM_CORES_PER_BLOCKDIM]{};
    // Indexed by lane, which is what keeps a Mix subtask's timing distinct from
    // this core's own: a coordinator's own dispatch is lane 0 and the mailboxes
    // it fills are lanes 1 and 2.
    int32_t timing_slots[PLATFORM_CORES_PER_BLOCKDIM]{};
    uint32_t loop_iter{0};
    uint8_t worker_trace_valid_mask{0};

    inline __aicore__ SchedulerLocalProfilingState() {
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
            timing_slots[lane] = -1;
    }
};

// One tracker per outstanding Mix dispatch on this cluster's coordinator, which
// is what aggregates that dispatch's subtasks. A dispatch takes one slot on
// each lane it uses and every lane has two, so a cluster cannot have more than
// SCHEDULER_LANE_SLOT_COUNT of them in the air. Running out is not an error:
// placement fails and the task waits in the ready queue for the next round.
inline constexpr uint32_t SCHEDULER_MIX_TRACKER_COUNT = PLATFORM_CORES_PER_BLOCKDIM * SCHEDULER_LANE_SLOT_COUNT;
struct SchedulerMixTracker {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint8_t active_mask{0};
    uint8_t completed_mask{0};
};

struct SchedulerLocalState {
    SchedulerLocalConfig config{};
    SchedulerLocalProfilingState *profiling{nullptr};
    uint64_t pending_completed{0};
    SchedulerMixTracker mix_trackers[SCHEDULER_MIX_TRACKER_COUNT]{};
    uint32_t dispatch_sequence{0};
    bool has_mix{false};
    // Whether this graph has any SPMD task at all. Without it every core would
    // read the SPMD queue's endpoints once per pass, and ld_dev is strictly
    // serial: a graph that can never fill that queue must not pay for it.
    bool has_spmd{false};

    // The SPMD queue head this core last looked at, with the shape facts it
    // read from that task. A round that finds the head unchanged needs no
    // metadata read, which is what keeps the steady-state poll at the one
    // ld_dev of the endpoints: the alternative costs every core a cache-line
    // observe per round for as long as one task holds the head, which for a
    // large task is seconds.
    //
    // The completion path consults it too, and may therefore act on a view one
    // round old. That is safe in both directions: a head that has since
    // advanced suppresses one direct refill that need not have been, and a head
    // that has since arrived delays a cohort by one round, after which the
    // claim pass below refreshes this.
    int64_t spmd_head_ref{SCHEDULER_TASK_REF_INVALID};
    uint16_t spmd_head_block_num{0};
    uint8_t spmd_head_active_mask{0};
    uint8_t spmd_head_flags{0};

    inline __aicore__ explicit SchedulerLocalState(SchedulerLocalProfilingState *profile = nullptr) :
        profiling(profile) {}
    // True on the one core that coordinates this cluster's Mix tasks: its AIC.
    inline __aicore__ bool is_mix_coordinator() const { return config.self_lane == 0; }
    inline __aicore__ uint64_t worker_id() const {
        return config.self_lane < PLATFORM_CORES_PER_BLOCKDIM ? config.worker_ids[config.self_lane] : UINT64_MAX;
    }
    // This core's own inbox index in every queue it can hold.
    inline __aicore__ uint64_t self_type_rank() const {
        return config.self_lane < PLATFORM_CORES_PER_BLOCKDIM ? config.cluster_type_ranks[config.self_lane] :
                                                                UINT64_MAX;
    }
    // One payload buffer per core, because a core runs one task at a time and
    // claims the next only once this one is retired.
    inline __aicore__ uint64_t dispatch_payload_offset(uint32_t lane) const {
        return static_cast<uint64_t>(config.dispatch_payloads_offset) +
               static_cast<uint64_t>(config.worker_ids[lane]) * sizeof(DispatchPayload);
    }
    // A lane's payload and executor trace are addressed by its AIV rank and the
    // slot the dispatch sits in. Two of each per lane, because two dispatches
    // are live at once: the coordinator builds the pending one's payload while
    // the lane is still executing out of the other. One buffer per lane would
    // have the build overwrite the running kernel's arguments.
    //
    // Lane 0 -- the AIC -- has none: it runs its own Mix subtask out of its own
    // position and payload.
    inline __aicore__ uint64_t mix_payload_offset(uint32_t lane, bool lane_slot) const {
        return config.mix_payloads_offset +
               (static_cast<uint64_t>(config.cluster_type_ranks[lane]) * SCHEDULER_LANE_SLOT_COUNT + lane_slot) *
                   sizeof(DispatchPayload);
    }
    inline __aicore__ uint64_t mix_trace_offset(uint32_t lane, bool lane_slot) const {
        return config.mix_traces_offset +
               (static_cast<uint64_t>(config.cluster_type_ranks[lane]) * SCHEDULER_LANE_SLOT_COUNT + lane_slot) *
                   sizeof(SchedulerExecutorTaskTrace);
    }
    // The slot a claim names. An ordinary claim is always this core's own
    // running position; a Mix claim names the peer lane whose mailbox the
    // coordinator is filling.
    inline __aicore__ SchedulerLocalSlotState *slot_state(const SchedulerFreeSlotClaim &claim) {
        return claim.mix ? &mix_placed[claim.cluster_lane][claim.lane_slot] : &running;
    }
    inline __aicore__ uint64_t slot_payload_offset(const SchedulerFreeSlotClaim &claim) const {
        return claim.mix ? mix_payload_offset(claim.cluster_lane, claim.lane_slot) :
                           dispatch_payload_offset(claim.cluster_lane);
    }
    inline __aicore__ uint32_t next_slot_generation(const SchedulerFreeSlotClaim &claim) {
        return claim.mix ? ++mix_dispatch_sequences[claim.cluster_lane] : ++dispatch_sequence;
    }

    inline __aicore__ int32_t timing_slot(uint32_t lane) const {
        return profiling != nullptr ? profiling->timing_slots[lane] : -1;
    }
    inline __aicore__ void set_timing_slot(uint32_t lane, int32_t timing) {
        if (profiling != nullptr) profiling->timing_slots[lane] = timing;
    }
    inline __aicore__ bool sampled_task_timing(uint32_t lane) const {
        const int32_t timing = timing_slot(lane);
        return timing >= 0 && timing < SCHEDULER_TASK_TIMING_SLOT_COUNT;
    }

    // The task this core is executing or about to execute. Its only writer is
    // this core, which is what lets a peer fill the mailbox below without an
    // atomic between them: the coordinator never reaches in here.
    SchedulerLocalSlotState running{};
    // What this coordinator has put in each lane's two slots, held so it can
    // close a dispatch's tracker when that lane reports the subtask done. Two
    // per lane because two are live at once: the one the lane is running and
    // the one waiting behind it, which is the whole point of filling ahead.
    // Indexed the same way the lane indexes them, so a completion maps back by
    // matching the generation. Lane 0 is the coordinator itself: it runs its
    // own Mix subtask out of `running`, so its entries stay empty.
    SchedulerLocalSlotState mix_placed[PLATFORM_CORES_PER_BLOCKDIM][SCHEDULER_LANE_SLOT_COUNT]{};
    uint32_t mix_dispatch_sequences[PLATFORM_CORES_PER_BLOCKDIM]{};
    // The generation this lane last took out of each of its two slots. A slot
    // keeps whatever was published into it until the coordinator writes the
    // next dispatch there, so "filled" does not mean "new" -- the generation
    // does.
    uint32_t consumed_dispatch_generations[SCHEDULER_LANE_SLOT_COUNT]{};
    uint64_t consumed_completion_generation{0};
    uint64_t consumed_mix_completion_generations[PLATFORM_CORES_PER_BLOCKDIM]{};
    uint32_t local_completed_generation{0};
    bool local_ready{false};
    uint8_t owner_ready_queue_mask{0};
};

static_assert(SCHEDULER_WORKER_CAPACITY < UINT16_MAX, "worker IDs must fit the local config");
static_assert(PLATFORM_CORES_PER_BLOCKDIM <= 8, "worker trace mask must fit one byte");
static_assert(SCHEDULER_READY_QUEUE_COUNT <= 8, "owner ready mask must fit one byte");
static_assert(alignof(SchedulerLocalSlotState) == alignof(uint64_t));
static_assert(alignof(SchedulerLocalState) == alignof(uint64_t));
static_assert(sizeof(void *) != 8 || sizeof(SchedulerLocalConfig) == 120, "64-bit local config size changed");
// Tripwires against unintended growth of core-local state, not a wire contract:
// the executor already spills enough to fault on A5 when this grows carelessly
// (see the note on stage_task_trace_before_execution). Seven slot states per
// core -- the one it runs from, plus a coordinator's record of both slots of
// each lane -- means every byte added to one costs seven.
static_assert(sizeof(void *) != 8 || sizeof(SchedulerLocalSlotState) == 24, "64-bit local slot size changed");
static_assert(sizeof(void *) != 8 || sizeof(SchedulerLocalState) == 488, "64-bit local state size changed");
static_assert(
    sizeof(void *) != 8 || sizeof(SchedulerLocalProfilingState) == 184, "64-bit local profiling size changed"
);

// Called once after READY acquire, before any bootstrap or mailbox operation.
// Each invocation owns a fresh SchedulerLocalState; no cache survives a run.
// Failure leaves local partially initialized; the caller must not use it.
inline __aicore__ bool scheduler_initialize_local_config(
    __gm__ void *base, __gm__ SchedulerWorkerContext *context, const SchedulerGraphView *graph,
    SchedulerLocalState *local
) {
    if (base == nullptr || context == nullptr || graph == nullptr || local == nullptr) return false;
    for (uint64_t offset = 0; offset < offsetof(SchedulerWorkerContext, bootstrap_task_count); offset += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(context) + offset);
    if (context->scheduler_ssbuf_reserved0 != 0 || context->scheduler_ssbuf_reserved1 != 0 ||
        context->scheduler_ssbuf_reserved2 != 0 || context->topology_reserved[0] != 0)
        return false;
    SchedulerLocalConfig &config = local->config;
    const uint64_t payload_stride = sizeof(DispatchPayload);
    if (context->worker_contexts_offset > UINT32_MAX || context->callable_addresses_count > UINT32_MAX ||
        context->task_metadata_offset > UINT32_MAX || context->runtime_worker_count == 0 ||
        context->runtime_worker_count > SCHEDULER_WORKER_CAPACITY ||
        context->worker_index >= context->runtime_worker_count ||
        context->dispatch_payload_offset < context->worker_index * payload_stride)
        return false;
    const uint64_t payload_base = context->dispatch_payload_offset - context->worker_index * payload_stride;
    if (payload_base > UINT32_MAX) return false;
    config.dispatch_payloads_offset = static_cast<uint32_t>(payload_base);
    config.shared_context = context;
    config.graph = graph;
    config.task_controls_offset = context->task_controls_offset;
    config.task_metadata_offset = static_cast<uint32_t>(context->task_metadata_offset);
    config.ready_inboxes_offset = context->ready_inboxes_offset;
    config.ready_directory_offset = context->ready_directory_offset;
    if (local->profiling != nullptr) {
        local->profiling->trace_cells_offset = context->trace_cells_offset;
        local->profiling->activity_buffers_offset = context->activity_buffers_offset;
    }
    config.worker_contexts_offset = static_cast<uint32_t>(context->worker_contexts_offset);
    config.callable_addresses_address = context->callable_addresses_address;
    config.callable_addresses_count = static_cast<uint32_t>(context->callable_addresses_count);
    config.graph_summary_offset = context->graph_summary_offset;
    // The layout reserves the run control first, so no array it hands out can
    // begin at zero: a zero here means this worker's context was never wired up
    // and every Mix payload would be addressed from the state's own base.
    if (context->mix_payloads_offset == 0 || context->mix_traces_offset == 0) return false;
    config.mix_payloads_offset = context->mix_payloads_offset;
    config.mix_traces_offset = context->mix_traces_offset;
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, config.graph_summary_offset);
    scheduler_observe_cache_line(summary);
    local->has_mix = summary->mix_task_count != 0;
    local->has_spmd = summary->spmd_task_count != 0;
    const uint64_t runtime_worker_count = context->runtime_worker_count;
    const uint64_t scheduler_count = context->scheduler_count;
    const uint64_t scheduler_index = context->scheduler_index;
    if (runtime_worker_count == 0 || runtime_worker_count > SCHEDULER_WORKER_CAPACITY || scheduler_count == 0 ||
        scheduler_count > SCHEDULER_WORKER_CAPACITY || scheduler_index >= scheduler_count)
        return false;
    config.runtime_worker_count = static_cast<uint16_t>(runtime_worker_count);
    config.scheduler_count = static_cast<uint16_t>(scheduler_count);
    config.scheduler_index = static_cast<uint16_t>(scheduler_index);
    // One AIC per cluster, so a cluster count bounds the AIC and Mix queues
    // directly and the AIV queue by the lanes per cluster.
    const uint64_t cluster_count = context->cluster_count;
    if (cluster_count == 0 || cluster_count > SCHEDULER_CLUSTER_CAPACITY) return false;
    config.cluster_count = static_cast<uint16_t>(cluster_count);
    config.self_lane = UINT8_MAX;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        const uint64_t worker_id = context->cluster_worker_ids[lane];
        if (worker_id >= config.runtime_worker_count) return false;
        for (uint32_t previous = 0; previous < lane; ++previous)
            if (config.worker_ids[previous] == worker_id) return false;
        __gm__ SchedulerWorkerContext *target = scheduler_state_at<SchedulerWorkerContext>(
            base, config.worker_contexts_offset + worker_id * sizeof(SchedulerWorkerContext)
        );
        scheduler_observe_cache_line(target);
        const int32_t core_type = target->core_type;
        if (core_type != static_cast<int32_t>(lane == 0 ? CoreType::AIC : CoreType::AIV)) return false;
        if (target->dispatch_payload_offset != payload_base + worker_id * payload_stride) return false;
        if (target->worker_reserved != 0) return false;
        const int32_t rank = target->type_rank;
        if (rank < 0 ||
            static_cast<uint32_t>(rank) >= scheduler_ready_inbox_capacity(lane == 0 ? TaskType::AIC : TaskType::AIV))
            return false;
        config.cluster_type_ranks[lane] = static_cast<uint16_t>(rank);
        config.worker_ids[lane] = static_cast<uint16_t>(worker_id);
        if (worker_id == context->worker_index) config.self_lane = static_cast<uint8_t>(lane);
    }
    if (config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    // The loop above checked that lane 0 is the AIC and the rest are AIVs, so
    // the lane fixes the type without another worker-context read. The ready
    // queues are indexed by the same 0 = AIC / 1 = AIV order.
    config.self_core_type = config.self_lane == 0 ? 0U : 1U;
    // Every core schedules itself, so the published flag has nothing left to
    // disagree with; a zero would mean the topology never reached this worker.
    return context->is_scheduler != 0;
}

// A lane's own completions and the Mix subtask completions it reports are
// separate sequences in separate inboxes, so which one a generation belongs to
// has to travel with it. The Mix side is tracked per lane slot: a lane has two,
// and can finish what was in one while the other is already running, so a
// single value would let the second report erase the first before the
// coordinator ever saw it -- and the dispatch it erased would never be closed.
inline __aicore__ uint32_t scheduler_consumed_completion_generation(
    const SchedulerLocalState *scheduler_local_state, uint32_t cluster_lane, bool mix = false, bool lane_slot = false
) {
    if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) return 0;
    if (!mix) return static_cast<uint32_t>(scheduler_local_state->consumed_completion_generation);
    const uint32_t shift = lane_slot ? 32U : 0U;
    return static_cast<uint32_t>(scheduler_local_state->consumed_mix_completion_generations[cluster_lane] >> shift);
}

inline __aicore__ bool scheduler_completion_generation_is_new(
    const SchedulerLocalState *scheduler_local_state, uint32_t cluster_lane, uint32_t completed_generation,
    bool mix = false, bool lane_slot = false
) {
    return completed_generation != 0 &&
           completed_generation !=
               scheduler_consumed_completion_generation(scheduler_local_state, cluster_lane, mix, lane_slot);
}

inline __aicore__ void scheduler_mark_completion_consumed(
    SchedulerLocalState *scheduler_local_state, uint32_t cluster_lane, uint32_t completed_generation, bool mix = false,
    bool lane_slot = false
) {
    if (cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) return;
    if (!mix) {
        scheduler_local_state->consumed_completion_generation = completed_generation;
        return;
    }
    const uint32_t shift = lane_slot ? 32U : 0U;
    uint64_t &consumed = scheduler_local_state->consumed_mix_completion_generations[cluster_lane];
    consumed = (consumed & ~(UINT64_C(0xffffffff) << shift)) | (static_cast<uint64_t>(completed_generation) << shift);
}

inline __aicore__ void scheduler_local_ready_publish(SchedulerLocalState *scheduler_local_state) {
    scheduler_local_state->local_ready = true;
}

inline __aicore__ bool scheduler_local_ready_pop(SchedulerLocalState *scheduler_local_state, uint64_t *publication) {
    if (publication == nullptr || !scheduler_local_state->local_ready ||
        scheduler_local_state->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM)
        return false;
    scheduler_local_state->local_ready = false;
    const auto &slot = scheduler_local_state->running;
    // A local notification owns this slot until it is consumed here. Keep an
    // invalid state in the token so the Executor rejects it.
    *publication = (static_cast<uint64_t>(slot.generation) << 8) | static_cast<uint64_t>(slot.state);
    return true;
}

inline __aicore__ void scheduler_owner_queue_activate(SchedulerLocalState *scheduler_local_state, TaskType type) {
    scheduler_local_state->owner_ready_queue_mask |= UINT32_C(1) << scheduler_task_type_index(type);
}

inline __aicore__ void scheduler_owner_queue_deactivate(SchedulerLocalState *scheduler_local_state, TaskType type) {
    scheduler_local_state->owner_ready_queue_mask &= ~(UINT32_C(1) << scheduler_task_type_index(type));
}

inline __aicore__ uint64_t scheduler_dispatch_publication(uint32_t generation, SchedulerDispatchSlotState state) {
    return (static_cast<uint64_t>(generation) << 8) | static_cast<uint64_t>(state);
}

inline __aicore__ SchedulerDispatchSlotState scheduler_dispatch_state(uint64_t publication) {
    return static_cast<SchedulerDispatchSlotState>(publication & UINT64_C(0xff));
}

inline __aicore__ uint32_t scheduler_dispatch_generation(uint64_t publication) {
    return static_cast<uint32_t>(publication >> 8);
}

inline __aicore__ uint64_t scheduler_cycles() {
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
    return get_sys_cnt_aicore();
#else
    return 0;
#endif
}

inline __aicore__ uint32_t scheduler_core_type_index(int32_t core_type) {
    return core_type == static_cast<int32_t>(CoreType::AIC) ? 0U : 1U;
}

inline __aicore__ uint32_t scheduler_metadata_core_type_index(uint8_t subtask_slot) {
    return subtask_slot == 0 ? 0U : 1U;
}

inline __aicore__ uint8_t scheduler_metadata_single_subtask_slot(uint8_t active_mask) {
    if (active_mask == 1U) return 0;
    if (active_mask == 2U) return 1;
    return UINT8_MAX;
}

// Which core type a task of this shape runs on, or none when the shape is not
// one a single core takes whole. A Mix task has no single answer, which is why
// its coordinator is named by cluster lane instead.
inline __aicore__ bool scheduler_shape_runs_on_core_type(uint8_t active_mask, uint32_t self_core_type) {
    const uint8_t subtask = scheduler_metadata_single_subtask_slot(active_mask);
    return subtask < 3 && scheduler_metadata_core_type_index(subtask) == self_core_type;
}

// Whether the SPMD head this core last saw is a sync-start cohort it could run.
// The completion path consults this to hold a slot open for that cohort rather
// than refilling it with a successor it just resolved -- without it, a core
// that resolves a successor on every completion never has a free slot at the
// moment the claim pass looks, and the cohort assembles only once the rest of
// the reachable graph has drained.
inline __aicore__ bool scheduler_spmd_head_is_own_cohort(const SchedulerLocalState *local) {
    if (local == nullptr || scheduler_task_ref_is_null(local->spmd_head_ref)) return false;
    if (!scheduler_task_requires_sync_start(local->spmd_head_flags)) return false;
    if (scheduler_task_is_mix(local->spmd_head_flags)) return local->is_mix_coordinator();
    return scheduler_shape_runs_on_core_type(local->spmd_head_active_mask, local->config.self_core_type);
}

inline __aicore__ SchedulerPredicateResult
scheduler_evaluate_task_predicate(const SchedulerGraphView &graph, int64_t task_ref) {
    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_ref);
    __gm__ SchedulerDispatchPredicate *predicate =
        reinterpret_cast<__gm__ SchedulerDispatchPredicate *>(payload + SCHEDULER_GRAPH_PREDICATE_OFFSET);
    scheduler_observe_cache_line(predicate);
    if (predicate->op == 0) return SchedulerPredicateResult::PASS;
    if (!scheduler_dispatch_predicate_metadata_valid(predicate->addr, predicate->elem_size, predicate->op))
        return SchedulerPredicateResult::MALFORMED;

    __gm__ void *operand = reinterpret_cast<__gm__ void *>(predicate->addr);
    scheduler_observe_cache_line(operand);
    int64_t value = 0;
    switch (predicate->elem_size) {
    case 1:
        value = *reinterpret_cast<__gm__ int8_t *>(operand);
        break;
    case 2:
        value = *reinterpret_cast<__gm__ int16_t *>(operand);
        break;
    case 4:
        value = *reinterpret_cast<__gm__ int32_t *>(operand);
        break;
    case 8:
        value = *reinterpret_cast<__gm__ int64_t *>(operand);
        break;
    default:
        return SchedulerPredicateResult::MALFORMED;
    }
    switch (predicate->op) {
    case 1:
        return value == predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 2:
        return value != predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 3:
        return value > predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 4:
        return value < predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 5:
        return value >= predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    case 6:
        return value <= predicate->target ? SchedulerPredicateResult::PASS : SchedulerPredicateResult::FAIL;
    default:
        return SchedulerPredicateResult::MALFORMED;
    }
}

// `callable_count` is the table's own published length, not a capacity the
// scheduler state reserves: the table belongs to the callable's registration
// block, so an index past the length addresses memory no table occupies. The
// read needs no cache maintenance of its own — the AICore invalidates its
// whole data cache once at bootstrap, before any dispatch can reach here.
inline __aicore__ bool scheduler_lookup_callable_address(
    __gm__ uint64_t *callable_addresses, uint32_t callable_count, uint16_t kernel_id, uint64_t *callable_address
) {
    if (callable_addresses == nullptr || callable_address == nullptr || kernel_id >= callable_count) return false;
    *callable_address = callable_addresses[kernel_id];
    return *callable_address != 0;
}

inline __aicore__ uint64_t
scheduler_completion_id(const SchedulerLocalState *context, uint64_t local_completion_index) {
    return local_completion_index * context->config.runtime_worker_count + context->worker_id();
}

inline __aicore__ __gm__ SchedulerTaskControl *
scheduler_task_control_at(__gm__ void *scheduler_state_base, const SchedulerLocalState *context, int64_t task_ref) {
    return scheduler_state_at<SchedulerTaskControl>(
        scheduler_state_base,
        context->config.task_controls_offset + static_cast<uint64_t>(task_ref) * sizeof(SchedulerTaskControl)
    );
}

inline __aicore__ __gm__ SchedulerTaskMetadata *
scheduler_task_metadata_at(__gm__ void *scheduler_state_base, const SchedulerLocalState *context, int64_t task_ref) {
    return scheduler_state_at<SchedulerTaskMetadata>(
        scheduler_state_base,
        context->config.task_metadata_offset + static_cast<uint64_t>(task_ref) * sizeof(SchedulerTaskMetadata)
    );
}

inline __aicore__ __gm__ SchedulerReadyInbox *scheduler_ready_inbox_at(
    __gm__ void *scheduler_state_base, const SchedulerLocalState *context, TaskType type, uint64_t inbox_index
) {
    uint64_t linear = scheduler_ready_inbox_base(type) + inbox_index;
    return scheduler_state_at<SchedulerReadyInbox>(
        scheduler_state_base, context->config.ready_inboxes_offset + linear * sizeof(SchedulerReadyInbox)
    );
}

// Which core types may hold a queue: the AIC and Mix queues belong to AIC cores,
// the AIV queue to AIV cores. A core refreshes, claims from and is stolen from
// only the queues it can hold. The SPMD queue is held by every core: a block of
// it may be claimed by whichever core the task's own shape admits, and that is
// read from the task rather than fixed by the queue.
inline __aicore__ bool scheduler_ready_queue_holder(TaskType type, uint32_t self_core_type) {
    if (type == TaskType::SPMD) return true;
    return (type == TaskType::AIV) == (self_core_type == 1U);
}

// How many of a queue's inboxes this run actually uses. Bounds the directory's
// valid bits, so a shard never reports an owner the topology never created.
inline __aicore__ uint64_t scheduler_ready_inbox_count(const SchedulerLocalState *context, TaskType type) {
    if (type == TaskType::SPMD) return 1;
    const uint64_t clusters = context->config.cluster_count;
    return type == TaskType::AIV ? clusters * PLATFORM_AIV_CORES_PER_BLOCKDIM : clusters;
}

inline __aicore__ __gm__ SchedulerReadyDirectory *
scheduler_ready_directory_at(__gm__ void *scheduler_state_base, const SchedulerLocalState *context) {
    return scheduler_state_at<SchedulerReadyDirectory>(scheduler_state_base, context->config.ready_directory_offset);
}

inline __aicore__ __gm__ SchedulerWorkerContext *
scheduler_worker_context_at(__gm__ void *scheduler_state_base, const SchedulerLocalState *context, uint64_t worker_id) {
    return scheduler_state_at<SchedulerWorkerContext>(
        scheduler_state_base, context->config.worker_contexts_offset + worker_id * sizeof(SchedulerWorkerContext)
    );
}

// One activity buffer per Scheduler, and a Scheduler is a core, so the array is
// indexed by the worker this core runs on and spans every worker.
inline __aicore__ __gm__ SchedulerActivityBuffer *
scheduler_activity_buffer_at(__gm__ void *scheduler_state_base, const SchedulerLocalState *context) {
    if (context->profiling == nullptr || context->profiling->activity_buffers_offset == 0 ||
        context->config.scheduler_index >= SCHEDULER_WORKER_CAPACITY)
        return nullptr;
    return scheduler_state_at<SchedulerActivityBuffer>(
        scheduler_state_base,
        context->profiling->activity_buffers_offset + context->config.scheduler_index * sizeof(SchedulerActivityBuffer)
    );
}

inline __aicore__ void scheduler_append_idle_activity(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, uint64_t start_cycles, uint64_t end_cycles
) {
    __gm__ SchedulerActivityBuffer *buffer = scheduler_activity_buffer_at(scheduler_state_base, context);
    if (buffer == nullptr || end_cycles < start_cycles) return;
    const uint64_t capture_counts = scheduler_gm_query_u32_pair(&buffer->committed);
    const uint32_t committed = static_cast<uint32_t>(capture_counts);
    if (committed >= SCHEDULER_ACTIVITY_CAPACITY) {
        scheduler_gm_store(buffer->dropped, static_cast<uint32_t>(capture_counts >> 32) + 1);
        return;
    }
    __gm__ SchedulerIdleRecord *record = &buffer->records[committed];
    record->start_time = start_cycles;
    record->end_time = end_cycles;
    record->loop_iter = static_cast<uint32_t>(context->profiling->loop_iter);
    record->reserved = 0;
    scheduler_writeback_cache_line(record);
    scheduler_writeback_cache_line(&record->reserved);
    scheduler_cache_barrier();
    scheduler_gm_store(buffer->committed, committed + 1);
}

inline __aicore__ void scheduler_record_error(
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, SchedulerGraphResult status,
    const SchedulerGraphView *graph = nullptr, __gm__ const SchedulerWorkerContext *context = nullptr,
    SchedulerErrorSite error_site = SchedulerErrorSite::UNKNOWN
) {
    if (scheduler_gm_compare_exchange(run_control->error_claimed, UINT64_C(0), UINT64_C(1)) != 0) return;
    scheduler_gm_store(run_control->error_task_ref, static_cast<uint64_t>(task_ref));
    if (graph != nullptr) {
        scheduler_gm_store(run_control->error_graph_task_count, graph->task_count);
        scheduler_gm_store(run_control->error_storage_address, graph->storage_address);
        scheduler_gm_store(run_control->error_task_window_last_index, graph->task_window_last_index);
    }
    if (context != nullptr) {
        scheduler_gm_store(run_control->error_core_id, static_cast<uint64_t>(context->physical_core_id));
        scheduler_gm_store(run_control->error_core_type, static_cast<uint64_t>(context->core_type));
    }
    scheduler_gm_store(run_control->error_site, static_cast<uint64_t>(error_site));
    scheduler_gm_store(run_control->scheduler_error, static_cast<uint64_t>(status));
}

inline __aicore__ void scheduler_record_error(
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, SchedulerGraphResult status,
    const SchedulerGraphView *graph, const SchedulerLocalState *local,
    SchedulerErrorSite error_site = SchedulerErrorSite::UNKNOWN
) {
    scheduler_record_error(run_control, task_ref, status, graph, local->config.shared_context, error_site);
}

inline __aicore__ void scheduler_publish_waiter_metadata(
    __gm__ SchedulerTaskControl *control, int64_t next_waiter, int32_t next_fanin_index, int32_t waiting_producer
) {
    control->next_fanin_index = next_fanin_index;
    control->waiting_producer = waiting_producer;
    scheduler_publish_cache_line(&control->next_fanin_index);
    // The link lands after the fields it is read alongside, and as a single
    // atomic word: the waiter becomes reachable only once its link is readable.
    scheduler_gm_store(control->next_waiter, next_waiter);
}

// The chain link is published by another core's atomic, and ld_dev does not
// observe such a write until it has drained: reading it with ld_dev returns the
// pre-publication -1, which reads as the end of the chain and orphans every
// waiter behind this node. The read must therefore be an atomic too.
inline __aicore__ int64_t scheduler_observe_next_waiter(__gm__ SchedulerTaskControl *control) {
    return scheduler_gm_load(control->next_waiter);
}

inline __aicore__ SchedulerRouteResult scheduler_route_task(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, SchedulerWakeStats *stats,
    bool validate_current_state = true
) {
    if (!scheduler_task_ref_valid(graph, task_ref)) {
        scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_TASK_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_ref);
    if (validate_current_state) {
        // A BLOCKED task belongs to at most one producer wake list. Completion
        // detaches that list before routing the task to its next fanin.
        int64_t state = scheduler_gm_query(control->state);
        if (state == static_cast<int64_t>(SchedulerTaskState::DONE)) return SchedulerRouteResult::COMPLETED;
        if (state != static_cast<int64_t>(SchedulerTaskState::BLOCKED)) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::ROUTE_INVALID_STATE
            );
            return SchedulerRouteResult::ERROR;
        }
    }

    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_ref);
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
    int32_t next_fanin = control->next_fanin_index;
    if (fanin_count < 0 || next_fanin < 0 || next_fanin > fanin_count) {
        scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }

    while (next_fanin < fanin_count) {
        int32_t producer = scheduler_graph_fanin_id(graph, task_ref, next_fanin);
        if (!scheduler_task_ref_precedes(producer, task_ref)) {
            scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
            return SchedulerRouteResult::ERROR;
        }
        if (stats != nullptr) ++stats->fanin_state_load_count;
        __gm__ SchedulerTaskControl *producer_control =
            scheduler_task_control_at(scheduler_state_base, context, producer);
        if (scheduler_gm_query(producer_control->state) == static_cast<int64_t>(SchedulerTaskState::DONE)) {
            ++next_fanin;
            continue;
        }
        while (true) {
            int64_t observed = scheduler_gm_query(producer_control->wake_list_head);
            if (observed == SCHEDULER_WAKE_LIST_CLOSED) {
                if (stats != nullptr) ++stats->wake_closed_retry_count;
                ++next_fanin;
                break;
            }
            scheduler_publish_waiter_metadata(control, observed, next_fanin, producer);
            int64_t actual = scheduler_gm_compare_exchange(producer_control->wake_list_head, observed, task_ref);
            if (actual == observed) {
                if (stats != nullptr) ++stats->wake_register_count;
                return SchedulerRouteResult::WAITING;
            }
            if (stats != nullptr) ++stats->wake_cas_retry_count;
            if (actual == SCHEDULER_WAKE_LIST_CLOSED) {
                if (stats != nullptr) ++stats->wake_closed_retry_count;
                ++next_fanin;
                break;
            }
        }
    }
    return SchedulerRouteResult::READY_TO_ENQUEUE;
}

// The bootstrap barrier is closed for the whole of every core's registration
// pass: a core announces arrival only after its last registration, and the gate
// opens only once every core has arrived. No resolver can detach a wake list
// here, which is what makes publishing the new head before the waiter's link
// safe on this path and nowhere else. A producer the host completed inline
// carries no metadata flags at all, so the executability filter below is what
// keeps this pass off a wake list the host already closed.
inline __aicore__ SchedulerRouteResult scheduler_bootstrap_route_task(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, SchedulerWakeStats *stats
) {
    if (!scheduler_task_ref_valid(graph, task_ref)) {
        scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_TASK_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }
    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_ref);
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
    if (fanin_count < 0) {
        scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }

    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_ref);
    for (int32_t next_fanin = 0; next_fanin < fanin_count; ++next_fanin) {
        int32_t producer = scheduler_graph_fanin_id(graph, task_ref, next_fanin);
        if (!scheduler_task_ref_precedes(producer, task_ref)) {
            scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
            return SchedulerRouteResult::ERROR;
        }
        __gm__ SchedulerTaskMetadata *producer_metadata =
            scheduler_task_metadata_at(scheduler_state_base, context, producer);
        scheduler_observe_cache_line(producer_metadata);
        if (!scheduler_task_is_executable(producer_metadata->flags)) continue;

        __gm__ SchedulerTaskControl *producer_control =
            scheduler_task_control_at(scheduler_state_base, context, producer);
        int64_t previous = scheduler_gm_exchange(producer_control->wake_list_head, task_ref);
        if (previous < SCHEDULER_WAKE_LIST_OPEN) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::BOOTSTRAP_WAKE_INVALID_HEAD
            );
            return SchedulerRouteResult::ERROR;
        }
        control->next_fanin_index = next_fanin;
        control->waiting_producer = producer;
        scheduler_writeback_cache_line(&control->next_fanin_index);
        scheduler_gm_store(control->next_waiter, previous);
        if (stats != nullptr) ++stats->wake_register_count;
        return SchedulerRouteResult::WAITING;
    }

    return SchedulerRouteResult::READY_TO_ENQUEUE;
}

inline __aicore__ uint32_t scheduler_trace_index_at(
    __gm__ void *base, SchedulerLocalState *local, int64_t task_ref, uint8_t subtask, uint16_t block_idx
) {
    const auto *metadata = scheduler_task_metadata_at(base, local, task_ref);
    return scheduler_task_trace_index(metadata->trace_index_base, metadata->active_mask, subtask, block_idx);
}

inline __aicore__ bool scheduler_bootstrap_ready_batch_append(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, int64_t task_ref, SchedulerReadyBatch *batch,
    SchedulerReadyStats *stats, uint64_t profiling_level = 0
) {
    if (batch == nullptr || !scheduler_task_ref_valid(*context->config.graph, task_ref)) return false;
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_ref);
    scheduler_gm_store(control->next_waiter, SCHEDULER_INBOX_EMPTY);
    if (batch->head == SCHEDULER_INBOX_EMPTY) {
        batch->head = task_ref;
    } else {
        __gm__ SchedulerTaskControl *tail = scheduler_task_control_at(scheduler_state_base, context, batch->tail);
        scheduler_gm_store(tail->next_waiter, task_ref);
    }
    batch->tail = task_ref;
    if (scheduler_phase_timing_enabled(profiling_level)) {
        __gm__ SchedulerTaskTrace *cells =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, context->profiling->trace_cells_offset);
        const auto &metadata = *scheduler_task_metadata_at(scheduler_state_base, context, task_ref);
        const uint64_t now = scheduler_cycles();
        for (uint8_t subtask = 0; subtask < 3; ++subtask) {
            if ((metadata.active_mask & (1U << subtask)) == 0) continue;
            auto &trace = cells[scheduler_task_trace_index(metadata.trace_index_base, metadata.active_mask, subtask)];
            trace.ready_transition_cycles = now;
            scheduler_writeback_cache_line(&trace.ready_transition_cycles);
        }
    }
    ++batch->count;
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ bool scheduler_ready_batch_append(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, int64_t task_ref, SchedulerReadyBatch *batch,
    SchedulerReadyStats *stats, uint64_t profiling_level = 0
) {
    if (batch == nullptr || !scheduler_task_ref_valid(*context->config.graph, task_ref)) return false;
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_ref);
    scheduler_gm_store(control->next_waiter, SCHEDULER_INBOX_EMPTY);
    if (batch->head == SCHEDULER_INBOX_EMPTY) {
        batch->head = task_ref;
    } else {
        __gm__ SchedulerTaskControl *tail = scheduler_task_control_at(scheduler_state_base, context, batch->tail);
        scheduler_gm_store(tail->next_waiter, task_ref);
    }
    batch->tail = task_ref;
    if (scheduler_phase_timing_enabled(profiling_level)) {
        __gm__ SchedulerTaskTrace *cells =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, context->profiling->trace_cells_offset);
        const auto *metadata = scheduler_task_metadata_at(scheduler_state_base, context, task_ref);
        const uint8_t mask = metadata->active_mask;
        const uint32_t trace_base = metadata->trace_index_base;
        const uint64_t transition_cycles = scheduler_cycles();
        for (uint8_t subtask = 0; subtask < 3; ++subtask) {
            if ((mask & (1U << subtask)) == 0) continue;
            __gm__ SchedulerTaskTrace *trace = &cells[scheduler_task_trace_index(trace_base, mask, subtask)];
            scheduler_observe_cache_line(&trace->ready_transition_cycles);
            trace->ready_transition_cycles = transition_cycles;
            scheduler_publish_cache_line(&trace->ready_transition_cycles);
        }
    }
    ++batch->count;
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ void
scheduler_ready_directory_set(__gm__ SchedulerReadyDirectory *directory, TaskType type, uint64_t inbox_index) {
    uint64_t shard = inbox_index / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t bit = UINT64_C(1) << (inbox_index % SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD);
    scheduler_gm_fetch_or(directory->task_types[scheduler_task_type_index(type)][shard].bits, bit);
}

inline __aicore__ void
scheduler_ready_directory_clear(__gm__ SchedulerReadyDirectory *directory, TaskType type, uint64_t inbox_index) {
    uint64_t shard = inbox_index / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t bit = UINT64_C(1) << (inbox_index % SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD);
    scheduler_gm_fetch_and(directory->task_types[scheduler_task_type_index(type)][shard].bits, ~bit);
}

// Keep this core's directory bit consistent with its own inbox, so no other
// Scheduler misses stealable work or chases an inbox that has drained.
//
// The bit is shared state any publisher may set, while only this core clears
// it, which fixes how each direction may be short-circuited. A local mask
// saying "set" can only have come from this core, so the set is skippable --
// that is the steady-state path and it costs no atomic. The clear is not: a
// publisher that marked the directory left this core's mask untouched, so
// trusting the mask there would leak a bit that nothing else can retire.
//
// Re-reading the endpoints after the clear closes the other window: the bit
// gates every consumer, this core's own fill included, from looking at the
// inbox at all, and a push marks the directory only after its CAS has landed.
//
// The SPMD queue keeps no directory bit, so it has nothing to refresh: its one
// inbox is found by reading its endpoints, which is what a claim does anyway.
inline __aicore__ bool scheduler_refresh_ready_inbox_type(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, TaskType type,
    uint64_t *endpoints_snapshot = nullptr
) {
    if (context == nullptr || !scheduler_task_type_valid(type) || !scheduler_ready_queue_has_directory(type) ||
        !scheduler_ready_queue_holder(type, context->config.self_core_type))
        return false;
    const uint64_t own_index = context->self_type_rank();
    if (own_index >= scheduler_ready_inbox_count(context, type) || own_index >= scheduler_ready_inbox_capacity(type))
        return false;
    __gm__ SchedulerReadyInbox *inbox = scheduler_ready_inbox_at(scheduler_state_base, context, type, own_index);
    __gm__ SchedulerReadyDirectory *directory = scheduler_ready_directory_at(scheduler_state_base, context);
    uint64_t packed = scheduler_gm_query(inbox->endpoints);
    if (!scheduler_ready_endpoints_consistent(packed)) return false;
    if (scheduler_ready_endpoints_empty(packed)) {
        scheduler_ready_directory_clear(directory, type, own_index);
        scheduler_owner_queue_deactivate(context, type);
        scheduler_cache_barrier();
        packed = scheduler_gm_query(inbox->endpoints);
        if (!scheduler_ready_endpoints_consistent(packed)) return false;
    }
    if (!scheduler_ready_endpoints_empty(packed) &&
        (context->owner_ready_queue_mask & (UINT32_C(1) << scheduler_task_type_index(type))) == 0) {
        scheduler_ready_directory_set(directory, type, own_index);
        scheduler_owner_queue_activate(context, type);
    }
    if (endpoints_snapshot != nullptr) *endpoints_snapshot = packed;
    return true;
}

inline __aicore__ bool scheduler_refresh_ready_inbox(__gm__ void *scheduler_state_base, SchedulerLocalState *context) {
    const uint32_t active_mask = context->owner_ready_queue_mask;
    for (uint32_t index = 0; index < SCHEDULER_READY_QUEUE_COUNT; ++index) {
        if ((active_mask & (UINT32_C(1) << index)) == 0) continue;
        const TaskType type = static_cast<TaskType>(index);
        if (!scheduler_ready_queue_has_directory(type)) continue;
        if (!scheduler_ready_queue_holder(type, context->config.self_core_type)) continue;
        if (!scheduler_refresh_ready_inbox_type(scheduler_state_base, context, type)) return false;
    }
    return true;
}

// Which core's inbox a task of this type belongs in. The type of a ready task
// has nothing to do with which core resolved it -- an AIC finishing a matmul
// resolves the AIV activation that follows -- so a publisher routes by type
// rather than keeping what it produced.
//
// Every cluster holds all three core types, so the target is always a cluster
// peer and the task stays on this cluster's inboxes. AIV work alternates between
// the two AIV lanes by task ref parity, which spreads a fanout of same-type work
// instead of piling it on one lane.
//
// The result is the target's index within its own type, which is how its queue
// is addressed; no lookup turns it into anything else. The SPMD queue has one
// inbox for the whole machine, so every publisher names the same index.
inline __aicore__ uint64_t
scheduler_ready_push_target(const SchedulerLocalState *context, TaskType type, int64_t task_ref) {
    if (type == TaskType::SPMD) return 0;
    if (type == TaskType::AIV) {
        const uint32_t lane = 1U + static_cast<uint32_t>(task_ref & 1);
        return context->config.cluster_type_ranks[lane];
    }
    // AIC work, and Mix work which its cluster's AIC coordinates.
    return context->config.cluster_type_ranks[0];
}

// Append a locally built batch to the target core's ready queue.
//
// Any core may publish to any inbox, so the queue is a CAS-updated FIFO rather
// than a slot only its owner may store into. That is what retires the
// owner-pending list: it existed because a sole publisher could not append to a
// chain concurrent consumers were walking, and had to stage new work locally
// until its own inbox drained. Appending under CAS leaves nothing to stage.
//
// The order is append, not push-down: a consumer serves the oldest task first.
// Taking the newest instead would let a task that arrived later overtake queued
// work, and the mix path relies on the head being the oldest placeable task.
//
// Publication is two steps -- claim the tail, then link the old tail to the
// batch -- because the new tail is not known until the CAS lands. A consumer
// standing on the old tail therefore spins for that link instead of reading an
// unpublished one as the end of the queue.
inline __aicore__ bool scheduler_ready_batch_push(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, TaskType type, SchedulerReadyBatch *batch,
    SchedulerReadyStats *stats
) {
    if (batch == nullptr || context == nullptr || !scheduler_task_type_valid(type)) return false;
    if (batch->head == SCHEDULER_INBOX_EMPTY) return true;
    if (batch->tail < 0 || batch->head < 0) return false;
    if (!scheduler_ready_endpoint_fits(batch->head) || !scheduler_ready_endpoint_fits(batch->tail)) return false;

    const uint64_t target = scheduler_ready_push_target(context, type, batch->head);
    if (target >= scheduler_ready_inbox_count(context, type) || target >= scheduler_ready_inbox_capacity(type))
        return false;

    __gm__ SchedulerReadyInbox *inbox = scheduler_ready_inbox_at(scheduler_state_base, context, type, target);

    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        const uint64_t packed = scheduler_gm_query(inbox->endpoints);
        if (!scheduler_ready_endpoints_consistent(packed)) return false;
        const int64_t head = scheduler_ready_endpoints_head(packed);
        const int64_t tail = scheduler_ready_endpoints_tail(packed);
        const bool was_empty = scheduler_ready_endpoints_empty(packed);
        const uint64_t next_packed = scheduler_ready_endpoints_pack(was_empty ? batch->head : head, batch->tail);
        if (scheduler_gm_compare_exchange(inbox->endpoints, packed, next_packed) != packed) {
            if (stats != nullptr) ++stats->cas_retry_count;
            continue;
        }
        if (!was_empty) {
            __gm__ SchedulerTaskControl *old_tail = scheduler_task_control_at(scheduler_state_base, context, tail);
            scheduler_gm_store(old_tail->next_waiter, batch->head);
        }
        scheduler_cache_barrier();
        if (scheduler_ready_queue_has_directory(type))
            scheduler_ready_directory_set(scheduler_ready_directory_at(scheduler_state_base, context), type, target);
        scheduler_ready_batch_reset(batch);
        if (stats != nullptr) ++stats->batch_count;
        return true;
    }
    // The batch is still linked only to itself, so an exhausted retry budget
    // leaves no partially published chain behind; the caller reports it.
    return false;
}

struct SchedulerMixPlacement {
    SchedulerFreeSlotClaim slots[3]{};
    uint8_t subtasks[3]{};
    uint8_t count{0};
    uint8_t tracker{UINT8_MAX};
};

// Plan where a Mix task's subtasks go. Only the cluster's AIC plans one: the
// AIV subtasks land on Mix slots it alone writes, so capacity it probes here
// cannot be taken away before it claims, and its own subtask takes one of its
// own slots. That sole ownership is what lets a probe stand as a reservation
// and is why no back-pressure protocol is needed.
//
// Bootstrap established that lane 0 is the cluster's AIC and the rest are its
// AIVs, so the lane number fixes the core type and no worker context has to be
// read back on this path.
inline __aicore__ bool scheduler_plan_mix_placement(
    __gm__ void *base, SchedulerLocalState *local, int64_t task_ref, SchedulerMixPlacement *placement,
    const SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    *placement = {};
    if (!local->is_mix_coordinator() || ssbuf_region == nullptr) return false;
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    if (!scheduler_task_is_mix(metadata.flags)) return false;
    for (uint8_t i = 0; i < SCHEDULER_MIX_TRACKER_COUNT; ++i) {
        if (scheduler_task_ref_is_null(local->mix_trackers[i].task_ref)) {
            placement->tracker = i;
            break;
        }
    }
    if (placement->tracker == UINT8_MAX) return false;
    const bool both_aiv = (metadata.active_mask & 6U) == 6U;
    uint8_t used_lanes = 0;
    for (uint8_t subtask = 0; subtask < 3; ++subtask) {
        if ((metadata.active_mask & (1U << subtask)) == 0) continue;
        const bool mix_slot = subtask != 0;
        const uint8_t lane_begin = mix_slot ? 1 : 0;
        const uint8_t lane_end = mix_slot ? PLATFORM_CORES_PER_BLOCKDIM : 1;
        bool found = false;
        for (uint8_t lane = lane_begin; lane < lane_end && !found; ++lane) {
            if ((used_lanes & (1U << lane)) != 0) continue;
            // Two live AIV subtasks take a lane each; a lone one takes either.
            if (mix_slot && both_aiv && lane != subtask) continue;
            const uint64_t worker = local->config.worker_ids[lane];
            if (worker >= local->config.runtime_worker_count) continue;
            if (mix_slot) {
                // The lane's free slot is the one it is not running. Whether
                // this coordinator may write it is its own record's business,
                // not the index's: between placing a dispatch and the lane
                // taking it up the index does not move, so trusting the index
                // alone would let a second placement overwrite a first that was
                // never picked up -- losing that Mix task with nobody to report
                // it. The record is local and this core is its only writer, so
                // the test is exact.
                const bool lane_slot = scheduler_lane_pending_slot(ssbuf_region, lane);
                if (local->mix_placed[lane][lane_slot].state != SchedulerDispatchSlotState::FREE) continue;
                placement->slots[placement->count] = {
                    worker, local->mix_placed[lane][lane_slot].generation, lane, true, lane_slot
                };
            } else {
                // The AIC subtask needs no handoff: lane 0 is the coordinator
                // itself, so it goes straight into the position it runs from.
                if (local->running.state != SchedulerDispatchSlotState::FREE) continue;
                placement->slots[placement->count] = {worker, local->running.generation, lane, false};
            }
            placement->subtasks[placement->count++] = subtask;
            used_lanes |= 1U << lane;
            found = true;
        }
        // All or nothing: a shape that cannot be seated everywhere is seated
        // nowhere. Nothing above this point mutated shared state, so the caller
        // discarding `placement` is the whole of the undo.
        if (!found) return false;
    }
    return placement->count >= 2;
}

inline __aicore__ bool scheduler_ready_pop_from_inbox(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, TaskType type, uint64_t inbox_index, int64_t *task_ref,
    SchedulerReadyStats *stats, SchedulerMixPlacement *placement = nullptr,
    const SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region = nullptr,
    const uint64_t *first_endpoints_snapshot = nullptr
) {
    if (task_ref == nullptr) return false;
    *task_ref = SCHEDULER_TASK_REF_INVALID;
    __gm__ SchedulerReadyInbox *inbox = scheduler_ready_inbox_at(scheduler_state_base, context, type, inbox_index);
    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        const uint64_t packed = attempt == 0 && first_endpoints_snapshot != nullptr ?
                                    *first_endpoints_snapshot :
                                    scheduler_gm_query(inbox->endpoints);
        if (!scheduler_ready_endpoints_consistent(packed)) {
            scheduler_record_error(
                run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::READY_POP_INVALID_HEAD
            );
            return false;
        }
        const int64_t head = scheduler_ready_endpoints_head(packed);
        const int64_t tail = scheduler_ready_endpoints_tail(packed);
        if (scheduler_ready_endpoints_empty(packed)) return true;
        if (!scheduler_task_ref_valid(graph, head) || !scheduler_task_ref_valid(graph, tail)) {
            scheduler_record_error(
                run_control, head, SchedulerGraphResult::INVALID_TASK_ID, &graph, context,
                SchedulerErrorSite::READY_POP_INVALID_HEAD
            );
            return false;
        }
        if (placement != nullptr &&
            !scheduler_plan_mix_placement(scheduler_state_base, context, head, placement, ssbuf_region))
            return true;
        int64_t next = SCHEDULER_INBOX_EMPTY;
        if (head != tail) {
            // The queue continues past this head, so its link either is already
            // published or a push is storing it right now.
            __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, head);
            for (uint32_t spin = 0; spin < SCHEDULER_READY_LINK_SPIN_LIMIT; ++spin) {
                next = scheduler_observe_next_waiter(control);
                if (next != SCHEDULER_INBOX_EMPTY) break;
            }
            if (next < SCHEDULER_INBOX_EMPTY) {
                scheduler_record_error(
                    run_control, head, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::READY_POP_INVALID_LINK
                );
                return false;
            }
            if (next == SCHEDULER_INBOX_EMPTY) continue;
        }
        const uint64_t next_packed = next == SCHEDULER_INBOX_EMPTY ? SCHEDULER_READY_ENDPOINTS_EMPTY :
                                                                     scheduler_ready_endpoints_pack(next, tail);
        if (scheduler_gm_compare_exchange(inbox->endpoints, packed, next_packed) != packed) {
            if (stats != nullptr) ++stats->cas_retry_count;
            continue;
        }
        if (stats != nullptr) ++stats->pop_count;
        *task_ref = head;
        return true;
    }
    // Bounded contention may defer a non-empty inbox to a later scheduling iteration.
    if (stats != nullptr) ++stats->contention_giveup_count;
    return true;
}

inline __aicore__ uint64_t scheduler_load_ready_directory_shard(
    __gm__ SchedulerReadyDirectory *directory, uint64_t owner_count, TaskType type, uint64_t inbox_index
) {
    uint64_t shard = inbox_index / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t shard_begin = shard * SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t shard_end = shard_begin + SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    if (shard_end > owner_count) shard_end = owner_count;
    uint64_t valid_bits = shard_end > shard_begin ? (UINT64_C(1) << (shard_end - shard_begin)) - 1 : 0;
    return scheduler_gm_query(directory->task_types[scheduler_task_type_index(type)][shard].bits) & valid_bits;
}

inline __aicore__ bool scheduler_steal_ready_from_shard(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, TaskType type, uint64_t shard_begin, uint64_t shard_end, uint64_t start,
    uint64_t bits, SchedulerReadyStats *stats, SchedulerReadyClaim *claim, SchedulerMixPlacement *placement = nullptr,
    const SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region = nullptr
) {
    int64_t task_ref = SCHEDULER_TASK_REF_INVALID;
    bits &= ~(UINT64_C(1) << (context->self_type_rank() - shard_begin));
    for (uint32_t pass = 0; pass < 2; ++pass) {
        uint64_t range_begin = pass == 0 ? start : shard_begin;
        uint64_t range_end = pass == 0 ? shard_end : start;
        if (range_begin == range_end) continue;
        uint32_t lower_bit = static_cast<uint32_t>(range_begin - shard_begin);
        uint32_t upper_bit = static_cast<uint32_t>(range_end - shard_begin);
        uint64_t candidates = bits & (((UINT64_C(1) << upper_bit) - 1) & ~((UINT64_C(1) << lower_bit) - 1));
        while (candidates != 0) {
            uint32_t bit_index = static_cast<uint32_t>(__builtin_ctzll(candidates));
            candidates &= candidates - 1;
            uint64_t victim = shard_begin + bit_index;
            if (!scheduler_ready_pop_from_inbox(
                    graph, scheduler_state_base, context, run_control, type, victim, &task_ref, stats, placement,
                    ssbuf_region
                ))
                return false;
            if (!scheduler_task_ref_is_null(task_ref)) {
                claim->task_ref = task_ref;
                claim->inbox_index = victim;
                claim->source = SchedulerReadySource::STOLEN;
                if (stats != nullptr) ++stats->steal_count;
                return true;
            }
        }
    }
    return true;
}

inline __aicore__ bool scheduler_claim_ready_for_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, uint64_t scheduler_count, TaskType type, uint64_t *victim_cursor,
    SchedulerReadyStats *stats, SchedulerReadyClaim *claim, SchedulerMixPlacement *placement = nullptr,
    bool allow_steal = true, const SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region = nullptr
) {
    if (victim_cursor == nullptr || claim == nullptr || context == nullptr || scheduler_count == 0 ||
        scheduler_count > SCHEDULER_WORKER_CAPACITY || context->config.scheduler_index >= scheduler_count ||
        // The SPMD queue is peeked rather than popped: its head stays put until
        // its last block is claimed, so taking it here would hand one core a
        // task the rest of the machine still has blocks to take.
        type == TaskType::SPMD || !scheduler_ready_queue_holder(type, context->config.self_core_type))
        return false;
    const uint64_t own_index = context->self_type_rank();
    const uint64_t owner_count = scheduler_ready_inbox_count(context, type);
    if (own_index >= owner_count || own_index >= scheduler_ready_inbox_capacity(type)) return false;
    *claim = {};
    uint64_t owner_endpoints_snapshot = SCHEDULER_READY_ENDPOINTS_EMPTY;
    if (!scheduler_refresh_ready_inbox_type(scheduler_state_base, context, type, &owner_endpoints_snapshot))
        return false;
    // Any core may push here, and a thief may change a nonempty head; the pop
    // CAS retries against either change.
    int64_t task_ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_ready_pop_from_inbox(
            graph, scheduler_state_base, context, run_control, type, own_index, &task_ref, stats, placement,
            ssbuf_region, &owner_endpoints_snapshot
        ))
        return false;
    if (!scheduler_task_ref_is_null(task_ref)) {
        claim->task_ref = task_ref;
        claim->inbox_index = own_index;
        return true;
    }

    if (!allow_steal) return true;
    __gm__ SchedulerReadyDirectory *directory = scheduler_ready_directory_at(scheduler_state_base, context);
    uint64_t shard_begin =
        own_index / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD * SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t shard_end = shard_begin + SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    if (shard_end > owner_count) shard_end = owner_count;
    uint64_t start = *victim_cursor;
    if (start < shard_begin || start >= shard_end) start = shard_begin;
    uint64_t bits = scheduler_load_ready_directory_shard(directory, owner_count, type, own_index);
    if (bits != 0 && !scheduler_steal_ready_from_shard(
                         graph, scheduler_state_base, context, run_control, type, shard_begin, shard_end, start, bits,
                         stats, claim, placement, ssbuf_region
                     ))
        return false;
    const uint64_t cursor_base = scheduler_task_ref_is_null(claim->task_ref) ? start : claim->inbox_index;
    *victim_cursor = cursor_base + 1 == shard_end ? shard_begin : cursor_base + 1;
    return true;
}

inline __aicore__ bool scheduler_ready_directory_nonempty(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, uint64_t scheduler_count, TaskType type
) {
    if (scheduler_count == 0 || scheduler_count > SCHEDULER_WORKER_CAPACITY ||
        !scheduler_ready_queue_has_directory(type) ||
        !scheduler_ready_queue_holder(type, context->config.self_core_type))
        return false;
    const uint64_t own_index = context->self_type_rank();
    const uint64_t owner_count = scheduler_ready_inbox_count(context, type);
    if (own_index >= owner_count) return false;
    __gm__ SchedulerReadyDirectory *directory = scheduler_ready_directory_at(scheduler_state_base, context);
    return scheduler_load_ready_directory_shard(directory, owner_count, type, own_index) != 0;
}

inline __aicore__ void scheduler_initialize_free_slot(SchedulerLocalSlotState *local_slot) {
    local_slot->task_ref = SCHEDULER_TASK_REF_INVALID;
    local_slot->state = SchedulerDispatchSlotState::FREE;
    local_slot->mix_tracker = UINT8_MAX;
    local_slot->subtask_slot = UINT8_MAX;
    local_slot->from_mailbox = false;
    local_slot->block_idx = 0;
    local_slot->block_num = 1;
}

inline __aicore__ SchedulerTaskMetadata scheduler_load_dispatch_metadata(
    __gm__ void *base, SchedulerLocalState *scheduler, int64_t task_ref, uint64_t profiling_level
) {
    const auto *source = scheduler_task_metadata_at(base, scheduler, task_ref);
    SchedulerTaskMetadata metadata{};
    metadata.kernel_ids[0] = source->kernel_ids[0];
    metadata.kernel_ids[1] = source->kernel_ids[1];
    metadata.kernel_ids[2] = source->kernel_ids[2];
    metadata.active_mask = source->active_mask;
    metadata.flags = source->flags;
    metadata.timing_slot = source->timing_slot;
    if (scheduler_task_timing_enabled(profiling_level)) metadata.trace_index_base = source->trace_index_base;
    return metadata;
}

inline __aicore__ bool scheduler_prepare_dispatch_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint64_t profiling_level,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, uint8_t subtask_slot, uint64_t *prepare_start_cycles = nullptr,
    const SchedulerTaskPayloadArguments *arguments = nullptr, bool write_arguments = true
) {
    if (ssbuf_region == nullptr || scheduler == nullptr || !scheduler_task_ref_valid(graph, ready_claim.task_ref) ||
        slot_claim.worker_id >= scheduler->config.runtime_worker_count ||
        slot_claim.cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM)
        return false;
    if (prepare_start_cycles != nullptr)
        *prepare_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    if (subtask_slot == UINT8_MAX) subtask_slot = scheduler_metadata_single_subtask_slot(metadata.active_mask);
    __gm__ SchedulerWorkerContext *target =
        scheduler_worker_context_at(scheduler_state_base, scheduler, slot_claim.worker_id);
    if (subtask_slot >= 3 || (metadata.active_mask & (1U << subtask_slot)) == 0 ||
        (target->core_type != static_cast<int32_t>(CoreType::AIC) &&
         target->core_type != static_cast<int32_t>(CoreType::AIV)) ||
        !scheduler_task_is_executable(metadata.flags) ||
        scheduler_metadata_core_type_index(subtask_slot) != scheduler_core_type_index(target->core_type)) {
        scheduler_record_error(
            run_control, ready_claim.task_ref, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, scheduler,
            SchedulerErrorSite::DISPATCH_INVALID_SHAPE
        );
        return false;
    }
    const uint16_t kernel_id = metadata.kernel_ids[subtask_slot];
    if (scheduler->dispatch_sequence == UINT32_MAX) return false;
    auto &reserved = *scheduler->slot_state(slot_claim);
    if (reserved.state != SchedulerDispatchSlotState::FILLING || reserved.generation != slot_claim.generation ||
        scheduler->config.worker_ids[slot_claim.cluster_lane] != slot_claim.worker_id)
        return false;
    __gm__ uint64_t *callable_addresses =
        reinterpret_cast<__gm__ uint64_t *>(scheduler->config.callable_addresses_address);
    const bool inline_task = scheduler_task_is_inline(metadata.flags);
    uint64_t callable_address = UINT64_C(1);
    if (!inline_task && !scheduler_lookup_callable_address(
                            callable_addresses, scheduler->config.callable_addresses_count, kernel_id, &callable_address
                        )) {
        scheduler_record_error(
            run_control, ready_claim.task_ref, SchedulerGraphResult::INVALID_CALLABLE, &graph, scheduler,
            SchedulerErrorSite::DISPATCH_INVALID_CALLABLE
        );
        return false;
    }

    // The claim decides which payload this slot owns, so a Mix slot's payload
    // is not the target's own-slot payload. Deriving it from the claim also
    // saves reading the offset back out of the target's worker context, which
    // bootstrap already checked agrees with this core's own base.
    const uint64_t dispatch_payload_offset = scheduler->slot_payload_offset(slot_claim);

    SchedulerTaskInfo task{
        ready_claim.task_ref,
        static_cast<int32_t>(kernel_id),
        static_cast<int32_t>(subtask_slot),
        subtask_slot == 0 ? CoreType::AIC : CoreType::AIV,
    };
    __gm__ DispatchPayload *payload =
        scheduler_state_at<DispatchPayload>(scheduler_state_base, dispatch_payload_offset);
    SchedulerGraphResult status = SchedulerGraphResult::OK;
    if (inline_task) {
        payload->function_bin_addr = 0;
        payload->src_payload = 0;
    } else {
        status = scheduler_materialize_task_payload_resolved(
            graph, task, callable_address, payload, arguments, write_arguments
        );
        if (status == SchedulerGraphResult::OK && scheduler_task_has_predicate(metadata.flags)) {
            const SchedulerPredicateResult predicate = scheduler_evaluate_task_predicate(graph, ready_claim.task_ref);
            if (predicate == SchedulerPredicateResult::MALFORMED) {
                scheduler_record_error(
                    run_control, ready_claim.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::DISPATCH_INVALID_PREDICATE
                );
                return false;
            }
            if (predicate == SchedulerPredicateResult::FAIL) payload->function_bin_addr = 0;
        }
    }
    if (status != SchedulerGraphResult::OK) {
        scheduler_record_error(
            run_control, ready_claim.task_ref, status, &graph, scheduler,
            SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
        );
        return false;
    }
    payload->global_context.sub_block_id =
        scheduler_task_is_mix(metadata.flags) ? (slot_claim.cluster_lane == 2 ? 1 : 0) : (subtask_slot == 2 ? 1 : 0);
    // Which block of the task this dispatch runs. Both values come from the
    // claim rather than from the task's control line: the claim read them to
    // advance the cursor, and a Mix dispatch reaches here three times for one
    // claim. A task of one block passes 0 and 1, which is what the host
    // prefills in the per-run image.
    payload->local_context.block_idx = static_cast<int32_t>(slot_claim.block_idx);
    payload->local_context.block_num = static_cast<int32_t>(slot_claim.block_num);
    return true;
}

inline __aicore__ void scheduler_stage_dispatch_slot(
    __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, uint8_t subtask_slot
) {
    const uint32_t generation = scheduler->next_slot_generation(slot_claim);
    auto *payload =
        scheduler_state_at<DispatchPayload>(scheduler_state_base, scheduler->slot_payload_offset(slot_claim));
    SchedulerLocalSlotState *local_slot = scheduler->slot_state(slot_claim);
    local_slot->task_ref = ready_claim.task_ref;
    scheduler->set_timing_slot(slot_claim.cluster_lane, metadata.timing_slot);
    local_slot->generation = generation;
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->subtask_slot = subtask_slot;
    local_slot->block_idx = slot_claim.block_idx;
    local_slot->block_num = slot_claim.block_num;
    // Only a Mix slot's holder has to be told what it is running; this core
    // reads its own slots out of the local state written just above.
    if (slot_claim.mix) {
        scheduler_mix_dispatch_control(ssbuf_region, slot_claim)->task_ref = ready_claim.task_ref;
    }
    scheduler_writeback_dispatch_payload(payload);
}

inline __aicore__ void scheduler_commit_dispatch_slot(
    __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint64_t profiling_level,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, uint8_t subtask_slot, uint64_t prepare_start_cycles
) {
    const bool task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t dispatch_start_cycles = prepare_start_cycles;
    const uint32_t generation = scheduler->slot_state(slot_claim)->generation;
    const uint64_t ready_publish_cycles = schedule_timing_enabled ? scheduler_cycles() : 0;
    if (slot_claim.mix) {
        scheduler_ssbuf_store_relaxed(
            &scheduler_mix_dispatch_control(ssbuf_region, slot_claim)->publication,
            scheduler_ssbuf_pack_ready(generation, metadata.timing_slot)
        );
    } else {
        scheduler_local_ready_publish(scheduler);
    }
    if (task_timing_enabled) {
        __gm__ SchedulerTaskTrace *traces =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->profiling->trace_cells_offset);
        __gm__ SchedulerTaskTrace *trace = &traces[scheduler_task_trace_index(
            metadata.trace_index_base, metadata.active_mask, subtask_slot, slot_claim.block_idx
        )];
        trace->worker_id = slot_claim.worker_id;
        trace->task_ref = static_cast<uint64_t>(ready_claim.task_ref);
        if (phase_timing_enabled) {
            trace->ready_source = static_cast<uint64_t>(ready_claim.source);
            trace->publication_mode = static_cast<uint64_t>(ready_claim.publication_mode);
            if (ready_claim.source == SchedulerReadySource::DIRECT_RESOLVE) {
                trace->state_probe_scheduler_worker_id = UINT64_MAX;
                trace->state_probe_start_cycles = 0;
                trace->state_probe_end_cycles = 0;
            } else {
                trace->state_probe_scheduler_worker_id = scheduler->worker_id();
                trace->state_probe_start_cycles = ready_claim.state_probe_start_cycles;
                trace->state_probe_end_cycles = ready_claim.state_probe_end_cycles == 0 ?
                                                    dispatch_start_cycles :
                                                    ready_claim.state_probe_end_cycles;
            }
            trace->dispatch_start_cycles = dispatch_start_cycles;
            trace->dispatch_scheduler_worker_id = scheduler->worker_id();
            trace->dispatch_loop_iter = scheduler->profiling->loop_iter;
        }
        scheduler_publish_cache_line(trace);
        if (schedule_timing_enabled) trace->dispatch_end_cycles = ready_publish_cycles;
        if (schedule_timing_enabled) scheduler_publish_cache_line(&trace->dispatch_start_cycles);
    }
}

inline __aicore__ void scheduler_publish_dispatch_slot(
    __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint64_t profiling_level,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region, uint8_t subtask_slot, uint64_t prepare_start_cycles = 0
) {
    scheduler_stage_dispatch_slot(
        scheduler_state_base, scheduler, slot_claim, ready_claim, metadata, ssbuf_region, subtask_slot
    );
    scheduler_cache_barrier();
    scheduler_commit_dispatch_slot(
        scheduler_state_base, scheduler, slot_claim, ready_claim, metadata, profiling_level, ssbuf_region, subtask_slot,
        prepare_start_cycles
    );
}

inline __aicore__ bool scheduler_fill_dispatch_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, uint64_t profiling_level, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (scheduler == nullptr || slot_claim.cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    auto &slot = *scheduler->slot_state(slot_claim);
    if (slot.state == SchedulerDispatchSlotState::FREE && slot.generation == slot_claim.generation)
        slot.state = SchedulerDispatchSlotState::FILLING;
    const SchedulerTaskMetadata metadata =
        scheduler_load_dispatch_metadata(scheduler_state_base, scheduler, ready_claim.task_ref, profiling_level);
    const uint8_t subtask_slot = scheduler_metadata_single_subtask_slot(metadata.active_mask);
    uint64_t prepare_start_cycles = 0;
    if (!scheduler_prepare_dispatch_slot(
            graph, scheduler_state_base, scheduler, run_control, slot_claim, ready_claim, metadata, profiling_level,
            ssbuf_region, subtask_slot, &prepare_start_cycles
        ))
        return false;
    scheduler_publish_dispatch_slot(
        scheduler_state_base, scheduler, slot_claim, ready_claim, metadata, profiling_level, ssbuf_region, subtask_slot,
        prepare_start_cycles
    );
    return true;
}

inline __aicore__ bool scheduler_resolve_completion(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, SchedulerWakeStats *wake_stats,
    SchedulerReadyStats *ready_stats, SchedulerCompletionStats *completion_stats, uint64_t profiling_level,
    bool validate_done_state, SchedulerReadyClaim *direct_ready, TaskType direct_type,
    SchedulerReadyClaim *direct_mix_ready = nullptr
) {
    if (context == nullptr) return false;
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_ref);
    if (validate_done_state && scheduler_gm_query(control->state) != static_cast<int64_t>(SchedulerTaskState::DONE)) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
            SchedulerErrorSite::COMPLETION_TASK_NOT_DONE
        );
        return false;
    }
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    uint64_t resolve_start = phase_timing_enabled ? scheduler_cycles() : 0;
    if (phase_timing_enabled) {
        scheduler_observe_cache_line(&control->next_fanin_index);
        control->completion_resolve_start_cycles = resolve_start;
        control->scheduler_worker_id = context->worker_id();
        control->completion_resolve_loop_iter = context->profiling->loop_iter;
    }
    int64_t waiter = scheduler_gm_exchange(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    if (waiter == SCHEDULER_WAKE_LIST_CLOSED) {
        scheduler_record_error(
            run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
            SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED
        );
        return false;
    }
    if (wake_stats != nullptr) ++wake_stats->wake_close_count;
    SchedulerReadyBatch batches[SCHEDULER_READY_QUEUE_COUNT]{};
    while (waiter >= 0) {
        if (!scheduler_task_ref_valid(graph, waiter)) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::INVALID_TASK_ID, &graph, context,
                SchedulerErrorSite::COMPLETION_INVALID_WAITER
            );
            return false;
        }
        __gm__ SchedulerTaskControl *waiter_control = scheduler_task_control_at(scheduler_state_base, context, waiter);
        int64_t next = scheduler_observe_next_waiter(waiter_control);
        if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
        __gm__ uint8_t *waiter_payload = scheduler_graph_payload(graph, waiter);
        const int32_t fanin_count =
            *reinterpret_cast<__gm__ int32_t *>(waiter_payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
        const SchedulerRouteResult route =
            fanin_count == 1 ?
                SchedulerRouteResult::READY_TO_ENQUEUE :
                scheduler_route_task(graph, scheduler_state_base, context, run_control, waiter, wake_stats);
        // A fatal scheduler error makes any ready publication in this local batch terminal;
        // an unpublished partial batch is never reused as successful scheduler state.
        if (route == SchedulerRouteResult::ERROR) return false;
        if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
            __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(scheduler_state_base, context, waiter);
            if (!scheduler_task_is_executable(metadata->flags)) {
                scheduler_record_error(
                    run_control, waiter, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::COMPLETION_WAITER_NOT_EXECUTABLE
                );
                return false;
            }
            const uint8_t subtask_slot = scheduler_metadata_single_subtask_slot(metadata->active_mask);
            if (subtask_slot == UINT8_MAX && !scheduler_task_is_mix(metadata->flags)) {
                scheduler_record_error(
                    run_control, waiter, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, context,
                    SchedulerErrorSite::COMPLETION_INVALID_SHAPE
                );
                return false;
            }
            const TaskType type = scheduler_task_ready_queue(metadata->flags, metadata->active_mask);
            const uint32_t type_index = scheduler_task_type_index(type);
            if (type == TaskType::MIX && direct_mix_ready != nullptr &&
                !scheduler_task_ref_is_null(direct_mix_ready->task_ref)) {
                if (!scheduler_ready_batch_append(
                        scheduler_state_base, context, direct_mix_ready->task_ref, &batches[type_index], ready_stats,
                        profiling_level
                    ))
                    return false;
                *direct_mix_ready = {};
            }
            bool direct_mix_available = false;
            if (type == TaskType::MIX && direct_mix_ready != nullptr &&
                batches[type_index].head == SCHEDULER_INBOX_EMPTY) {
                __gm__ SchedulerReadyInbox *inbox =
                    scheduler_ready_inbox_at(scheduler_state_base, context, type, context->self_type_rank());
                direct_mix_available = scheduler_ready_endpoints_empty(scheduler_gm_query(inbox->endpoints)) &&
                                       !scheduler_ready_directory_nonempty(
                                           scheduler_state_base, context, context->config.scheduler_count, type
                                       );
            }
            // A successor goes straight back into the slot that just freed only
            // when this core can run it: the slot belongs to one core type, and
            // the completed subtask says which. An SPMD successor never
            // qualifies -- no core type maps to SPMD -- so it joins the queue
            // the whole machine claims blocks from instead of being dispatched
            // here as a single block.
            SchedulerReadyClaim *direct_claim =
                direct_mix_available ? direct_mix_ready :
                direct_ready != nullptr && scheduler_task_ref_is_null(direct_ready->task_ref) && type == direct_type ?
                                       direct_ready :
                                       nullptr;
            if (direct_claim != nullptr) {
                direct_claim->task_ref = waiter;
                direct_claim->source = SchedulerReadySource::DIRECT_RESOLVE;
                direct_claim->publication_mode = SchedulerPublicationMode::REFILL;
                if (phase_timing_enabled) {
                    __gm__ SchedulerTaskTrace *cells = scheduler_state_at<SchedulerTaskTrace>(
                        scheduler_state_base, context->profiling->trace_cells_offset
                    );
                    const uint64_t transition_cycles = scheduler_cycles();
                    for (uint8_t subtask = 0; subtask < 3; ++subtask) {
                        if ((metadata->active_mask & (1U << subtask)) == 0) continue;
                        auto &trace = cells[scheduler_task_trace_index(
                            metadata->trace_index_base, metadata->active_mask, subtask
                        )];
                        trace.ready_transition_cycles = transition_cycles;
                        scheduler_writeback_cache_line(&trace.ready_transition_cycles);
                    }
                }
            } else if (!scheduler_ready_batch_append(
                           scheduler_state_base, context, waiter, &batches[type_index], ready_stats, profiling_level
                       )) {
                scheduler_record_error(
                    run_control, waiter, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::COMPLETION_READY_APPEND_FAILED
                );
                return false;
            }
        }
        waiter = next;
    }
    for (uint32_t index = 0; index < SCHEDULER_READY_QUEUE_COUNT; ++index) {
        if (!scheduler_ready_batch_push(
                scheduler_state_base, context, static_cast<TaskType>(index), &batches[index], ready_stats
            )) {
            scheduler_record_error(
                run_control, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED
            );
            return false;
        }
    }
    if (phase_timing_enabled) {
        control->completion_resolve_end_cycles = scheduler_cycles();
        scheduler_publish_cache_line(&control->next_fanin_index);
    }
    if (completion_stats != nullptr) ++completion_stats->resolve_count;
    return true;
}
