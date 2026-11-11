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

#include "scheduler_protocol.h"
#include "scheduler_scalar_trace.h"
#include "scheduler_memory.h"
#include "scheduler_graph.h"
#include "scheduler_graph_binding.h"
#include "scheduler_error.h"
#include "scheduler_ready_pool.h"
#include "scheduler_resolve_state.h"

enum class SchedulerPredicateResult : uint8_t {
    PASS = 0,
    FAIL = 1,
    MALFORMED = 2,
};

enum class SchedulerPredicateState : uint8_t { UNEVALUATED, PASS, FAIL };

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
    SchedulerReadySource source{SchedulerReadySource::POOL};
    SchedulerPublicationMode publication_mode{SchedulerPublicationMode::DISPATCH};
    uint64_t state_probe_start_cycles{0};
    uint64_t state_probe_end_cycles{0};
};

struct SchedulerOrderedCandidate {
    SchedulerReadyClaim ready{};
    uint8_t slot_index{UINT8_MAX};
    SchedulerPredicateState predicate{SchedulerPredicateState::UNEVALUATED};
    bool args_observed{false};
};

enum class SchedulerGmMixPhase : uint8_t { IDLE, CLAIMED, PUBLISHED, ARRIVED, RUNNING, RELEASED };

struct SchedulerGmMixLocalState {
    uint64_t mailbox_offset{0};
    uint64_t group_seq{0};
    uint64_t task_last_seq{0};
    uint64_t vector_last_published_seq[2]{};
    uint32_t block_idx{0};
    uint32_t cluster_index{UINT32_MAX};
    uint8_t slot_index{UINT8_MAX};
    uint8_t subtask_slot{UINT8_MAX};
    SchedulerGmMixPhase phase{SchedulerGmMixPhase::IDLE};
};

struct SchedulerAdmissionLocalState {
    uint64_t epoch{0};
    uint64_t control_offset{0};
    uint64_t records_offset{0};
    uint64_t record_stride{0};
    uint64_t log_offset{0};
    uint64_t capacity{0};
    uint64_t member_word_count{0};
    int64_t pending_task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t scan_cursor{0};
    int64_t current_task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t current_block_num{0};
    LaneMask current_active_mask{};
    uint8_t current_flags{0};
    bool initialized{false};
    uint8_t next_candidate{0};
    bool sync_claimed{false};
    bool ticket_exhausted{false};
    SchedulerOrderedCandidate candidates[SCHEDULER_PENDING_SLOT_COUNT]{};
    SchedulerGmMixLocalState gm_mix{};
    // READY freezes topology for this run. This cache contains membership
    // facts only, never a task frontier or an execution permission.
    uint64_t topology_cube_words[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint64_t topology_vector0_words[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint64_t topology_worker_count{0};
};

static_assert(sizeof(SchedulerOrderedCandidate) == 40, "ordered candidate size changed");
static_assert(sizeof(SchedulerGmMixLocalState) == 56, "GM group-local state size changed");
static_assert(sizeof(SchedulerAdmissionLocalState) == 272, "admission-local state size changed");

struct SchedulerFreeSlotClaim {
    uint32_t slot_index{UINT32_MAX};
};

struct SchedulerLocalSlotState {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    SchedulerDispatchSlotState state{SchedulerDispatchSlotState::EMPTY};
    uint8_t subtask_slot{UINT8_MAX};
    bool cooperative{false};
    uint32_t block_idx{0};
    uint8_t candidate_index{UINT8_MAX};
    SchedulerPredicateState predicate{SchedulerPredicateState::UNEVALUATED};
    bool args_observed{false};
    bool gm_mix{false};
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
    uint64_t control_inboxes_offset{0};
    uint64_t ready_pool_offset{0};
    uint64_t graph_summary_offset{0};
    // The callable's registration-owned entry table, named by absolute address
    // because it is not part of the scheduler state these offsets index. The
    // address cannot be narrowed the way the offsets are; the length can, and
    // is validated below.
    uint64_t callable_addresses_address{0};
    // This run's per-subtask PMU staging array. Here rather than in the
    // profiling state because PMU is its own owner: a run can enable it with
    // every swimlane level off, and that state is then absent.
    uint64_t pmu_cells_offset{0};
    uint32_t worker_contexts_offset{0};
    uint32_t callable_addresses_count{0};
    uint32_t task_metadata_offset{0};
    uint32_t dispatch_payloads_offset{0};
    uint16_t worker_ids[PLATFORM_CORES_PER_BLOCKDIM]{};
    uint16_t runtime_worker_count{0};
    uint16_t scheduler_index{UINT16_MAX};
    uint16_t scheduler_count{0};
    // Clusters in this run determine the physical AIC/AIV participant counts.
    uint16_t cluster_count{0};
    uint8_t self_lane{UINT8_MAX};
    // This core's own type, which is also the only type of ready work it can
    // claim for itself. Held here so the dispatch path needs no worker-context
    // read to know which shared ready queue it consumes.
    uint32_t self_core_type{UINT32_MAX};
};

struct SchedulerLocalProfilingState {
    SchedulerScalarTrace scalar;
    uint64_t trace_cells_offset{0};
    uint64_t activity_buffers_offset{0};
    uint64_t protocol_layout_offset{0};
    SchedulerExecutorTaskTrace executor_traces[SCHEDULER_PENDING_SLOT_COUNT]{};
    SchedulerWorkerTraceCache worker_trace{};
    int32_t timing_slots[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint32_t loop_iter{0};
    bool worker_trace_valid{false};

    inline __aicore__ SchedulerLocalProfilingState() {
        for (auto &timing : timing_slots)
            timing = -1;
    }
};

struct SchedulerAsyncLocalState;

struct SchedulerLocalState {
    SchedulerLocalConfig config{};
    SchedulerLocalProfilingState *profiling{nullptr};
    SchedulerAdmissionLocalState *admission{nullptr};
    SchedulerAsyncLocalState *async{nullptr};
    uint64_t pending_completed{0};
    SchedulerResolveCursor control_resolve{};
    uint64_t last_completion_flush_cycles{0};
    uint32_t control_peer_cursor{0};

    inline __aicore__ explicit SchedulerLocalState(SchedulerLocalProfilingState *profile = nullptr) :
        profiling(profile) {}
    // True on the one core that coordinates this cluster's Mix tasks: its AIC.
    inline __aicore__ bool is_mix_coordinator() const { return config.self_lane == 0; }
    // This core's own two dispatch slots -- the only ones it fills or executes.
    inline __aicore__ SchedulerLocalSlotState *own_slots() { return slots; }
    inline __aicore__ const SchedulerLocalSlotState *own_slots() const { return slots; }
    inline __aicore__ uint64_t worker_id() const {
        return config.self_lane < PLATFORM_CORES_PER_BLOCKDIM ? config.worker_ids[config.self_lane] : UINT64_MAX;
    }
    inline __aicore__ uint64_t dispatch_payload_offset(uint32_t lane, uint32_t slot) const {
        return static_cast<uint64_t>(config.dispatch_payloads_offset) +
               (static_cast<uint64_t>(config.worker_ids[lane]) * SCHEDULER_PENDING_SLOT_COUNT + slot) *
                   sizeof(DispatchPayload);
    }
    inline __aicore__ SchedulerLocalSlotState *slot_state(const SchedulerFreeSlotClaim &claim) {
        return &slots[claim.slot_index];
    }
    inline __aicore__ uint64_t slot_payload_offset(const SchedulerFreeSlotClaim &claim) const {
        return dispatch_payload_offset(config.self_lane, claim.slot_index);
    }

    inline __aicore__ int32_t timing_slot(uint32_t slot) const {
        return profiling != nullptr ? profiling->timing_slots[slot] : -1;
    }
    inline __aicore__ void set_timing_slot(uint32_t slot, int32_t timing) {
        if (profiling != nullptr) profiling->timing_slots[slot] = timing;
    }
    inline __aicore__ bool sampled_task_timing(uint32_t slot) const {
        const int32_t timing = timing_slot(slot);
        return timing >= 0 && timing < SCHEDULER_TASK_TIMING_SLOT_COUNT;
    }

    SchedulerLocalSlotState slots[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint8_t local_ready_mask{0};
    uint8_t ready_head{UINT8_MAX};
};

static_assert(SCHEDULER_WORKER_CAPACITY < UINT16_MAX, "worker IDs must fit the local config");
static_assert(PLATFORM_CORES_PER_BLOCKDIM <= 8, "worker trace mask must fit one byte");
static_assert(SCHEDULER_PENDING_SLOT_COUNT == 2, "local ready head requires two slots");
static_assert(alignof(SchedulerLocalSlotState) == alignof(uint64_t));
static_assert(alignof(SchedulerLocalState) == alignof(uint64_t));
static_assert(sizeof(void *) != 8 || sizeof(SchedulerLocalConfig) == 104, "64-bit local config size changed");
static_assert(sizeof(void *) != 8 || sizeof(SchedulerLocalSlotState) == 24, "64-bit local slot size changed");
static_assert(sizeof(void *) != 8 || sizeof(SchedulerLocalState) == 368, "64-bit local state size changed");
static_assert(
    sizeof(void *) != 8 || sizeof(SchedulerLocalProfilingState) == 168 + sizeof(SchedulerScalarTrace),
    "64-bit local profiling size changed"
);

// Called once after READY acquire, before any bootstrap or mailbox operation.
// Each invocation owns a fresh SchedulerLocalState; no cache survives a run.
// Failure leaves local partially initialized; the caller must not use it.
static inline __aicore__ bool scheduler_initialize_local_config(
    __gm__ void *base, __gm__ SchedulerWorkerContext *context, const SchedulerGraphView *graph,
    SchedulerLocalState *local
) {
    if (base == nullptr || context == nullptr || graph == nullptr || local == nullptr) return false;
    for (uint64_t offset = 0; offset < offsetof(SchedulerWorkerContext, bootstrap_task_count); offset += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(context) + offset);
    if (!scheduler_graph_view_valid(*graph) || context->original_task_count != graph->original_task_count ||
        context->graph_view_version != graph->view_version || context->graph_view_bytes != graph->view_bytes ||
        context->native_header_offset != graph->native_header_offset)
        return false;
    SchedulerLocalConfig &config = local->config;
    const uint64_t payload_stride = SCHEDULER_PENDING_SLOT_COUNT * sizeof(DispatchPayload);
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
    config.pmu_cells_offset = context->pmu_cells_offset;
    if (local->profiling != nullptr) {
        local->profiling->trace_cells_offset = context->trace_cells_offset;
        local->profiling->activity_buffers_offset = context->activity_buffers_offset;
    }
    config.worker_contexts_offset = static_cast<uint32_t>(context->worker_contexts_offset);
    config.callable_addresses_address = context->callable_addresses_address;
    config.callable_addresses_count = static_cast<uint32_t>(context->callable_addresses_count);
    config.graph_summary_offset = context->graph_summary_offset;
    auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, config.graph_summary_offset);
    for (uint64_t offset = 0; offset < sizeof(*summary); offset += 64)
        scheduler_observe_cache_line(reinterpret_cast<__gm__ uint8_t *>(summary) + offset);
    if (graph->native_header_offset != 0) {
        const auto *header = scheduler_native_header_checked(
            graph->scheduler_state_address, graph->scheduler_state_bytes, graph->native_header_offset,
            graph->task_count, graph->original_task_count
        );
        if (header == nullptr || header->control_inbox_count != context->runtime_worker_count ||
            header->run_epoch != summary->run_epoch ||
            graph->scheduler_state_address != reinterpret_cast<uint64_t>(base))
            return false;
        config.control_inboxes_offset = header->control_inboxes_offset;
        local->control_peer_cursor = static_cast<uint32_t>(context->worker_index + 1);
    }
    config.ready_pool_offset = summary->ready_pool_offset;
    if (config.ready_pool_offset == 0 || !scheduler_ready_pool_validate(
                                             base, context->scheduler_state_bytes, config.ready_pool_offset,
                                             graph->task_count, static_cast<uint32_t>(context->cluster_count)
                                         ))
        return false;
    const uint64_t runtime_worker_count = context->runtime_worker_count;
    const uint64_t scheduler_count = context->scheduler_count;
    const uint64_t scheduler_index = context->scheduler_index;
    if (runtime_worker_count == 0 || runtime_worker_count > SCHEDULER_WORKER_CAPACITY || scheduler_count == 0 ||
        scheduler_count > SCHEDULER_WORKER_CAPACITY || scheduler_index >= scheduler_count)
        return false;
    config.runtime_worker_count = static_cast<uint16_t>(runtime_worker_count);
    config.scheduler_count = static_cast<uint16_t>(scheduler_count);
    config.scheduler_index = static_cast<uint16_t>(scheduler_index);
    // Every cluster contains one AIC and its AIV execution lanes.
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
        const int32_t rank = target->type_rank;
        const uint64_t type_worker_count = cluster_count * (lane == 0 ? 1U : PLATFORM_AIV_CORES_PER_BLOCKDIM);
        if (rank < 0 || static_cast<uint64_t>(rank) >= type_worker_count) return false;
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

inline __aicore__ void scheduler_local_ready_publish(SchedulerLocalState *local, uint32_t slot_index) {
    if (slot_index >= SCHEDULER_PENDING_SLOT_COUNT) return;
    if (local->local_ready_mask == 0) local->ready_head = static_cast<uint8_t>(slot_index);
    local->local_ready_mask |= static_cast<uint8_t>(1U << slot_index);
}

inline __aicore__ bool scheduler_local_ready_peek(const SchedulerLocalState *local, uint32_t *slot_index) {
    if (slot_index == nullptr || local->local_ready_mask == 0 || local->ready_head >= SCHEDULER_PENDING_SLOT_COUNT ||
        (local->local_ready_mask & (1U << local->ready_head)) == 0)
        return false;
    *slot_index = local->ready_head;
    return true;
}

inline __aicore__ void scheduler_local_ready_remove(SchedulerLocalState *local, uint32_t slot_index) {
    if (slot_index >= SCHEDULER_PENDING_SLOT_COUNT) return;
    local->local_ready_mask &= static_cast<uint8_t>(~(1U << slot_index));
    if (local->local_ready_mask == 0) local->ready_head = UINT8_MAX;
    else if (local->ready_head == slot_index) local->ready_head = static_cast<uint8_t>(slot_index ^ 1U);
}

inline __aicore__ bool scheduler_local_ready_execute(SchedulerLocalState *local, uint32_t slot_index) {
    uint32_t head = UINT32_MAX;
    if (!scheduler_local_ready_peek(local, &head) || head != slot_index ||
        local->slots[slot_index].state != SchedulerDispatchSlotState::READY)
        return false;
    scheduler_local_ready_remove(local, slot_index);
    local->slots[slot_index].state = SchedulerDispatchSlotState::EXECUTING;
    return true;
}

inline __aicore__ uint32_t scheduler_core_type_index(int32_t core_type) {
    return core_type == static_cast<int32_t>(CoreType::AIC) ? 0U : 1U;
}

inline __aicore__ uint32_t scheduler_metadata_core_type_index(uint8_t subtask_slot) {
    return subtask_slot == 0 ? 0U : 1U;
}

inline __aicore__ uint8_t scheduler_metadata_single_subtask_slot(LaneMask active_mask) {
    if (active_mask == LaneMask::aic()) return 0;
    if (active_mask == LaneMask::aiv0()) return 1;
    return UINT8_MAX;
}

static inline __aicore__ SchedulerPredicateResult
scheduler_evaluate_task_predicate(const SchedulerGraphAccess &graph, int64_t task_ref) {
    auto *predicate = scheduler_node_predicate(graph, task_ref);
    if (predicate == nullptr) return SchedulerPredicateResult::MALFORMED;
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

// Queue ownership follows the consuming core type; MIX is claimed by an AIC.
inline __aicore__ bool scheduler_ready_queue_holder(uint32_t queue_class, uint32_t self_core_type) {
    return queue_class < SCHEDULER_READY_QUEUE_COUNT && self_core_type < SCHEDULER_CORE_TYPE_COUNT &&
           queue_class == self_core_type;
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

static inline __aicore__ void scheduler_append_idle_activity(
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
    scheduler_writeback_cache_line(record);
    scheduler_writeback_cache_line(&record->loop_iter);
    scheduler_cache_barrier();
    scheduler_gm_store(buffer->committed, committed + 1);
}

inline __aicore__ SchedulerErrorIdentity scheduler_error_identity(
    const SchedulerLocalState *local, int64_t task_ref, const SchedulerLocalSlotState *slot = nullptr
) {
    SchedulerErrorIdentity identity{};
    if (local == nullptr) return identity;
    if (local->config.self_lane < PLATFORM_CORES_PER_BLOCKDIM &&
        local->worker_id() < local->config.runtime_worker_count) {
        identity.worker_index = local->worker_id();
        identity.cluster_lane = local->config.self_lane;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_WORKER | SCHEDULER_ERROR_HAS_LANE;
    }
    const auto *admission = local->admission;
    if (admission != nullptr && admission->initialized) {
        identity.run_epoch = admission->epoch;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_EPOCH;
        if (!scheduler_task_ref_is_null(task_ref) && task_ref == admission->current_task_ref) {
            identity.order_key = admission->scan_cursor;
            identity.valid_fields |= SCHEDULER_ERROR_HAS_ORDER;
            const auto &group = admission->gm_mix;
            const bool owns_group =
                group.phase >= SchedulerGmMixPhase::CLAIMED && group.phase <= SchedulerGmMixPhase::RELEASED;
            const bool retired_current_group = group.phase == SchedulerGmMixPhase::IDLE && group.task_last_seq != 0 &&
                                               group.task_last_seq == group.group_seq;
            if (scheduler_task_is_mix(admission->current_flags) && group.group_seq != 0 &&
                (owns_group || retired_current_group)) {
                identity.block_idx = group.block_idx;
                identity.group_seq = group.group_seq;
                identity.valid_fields |= SCHEDULER_ERROR_HAS_BLOCK | SCHEDULER_ERROR_HAS_GROUP;
            }
        }
    }
    if (slot != nullptr && !scheduler_task_ref_is_null(task_ref) && slot->task_ref == task_ref &&
        slot->state != SchedulerDispatchSlotState::FREE && slot->state != SchedulerDispatchSlotState::EMPTY) {
        identity.block_idx = slot->block_idx;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_BLOCK;
    }
    return identity;
}

static inline __aicore__ void scheduler_record_error(
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, SchedulerGraphResult status,
    const SchedulerGraphView *graph, const SchedulerLocalState *local,
    SchedulerErrorSite error_site = SchedulerErrorSite::UNKNOWN, const SchedulerLocalSlotState *slot = nullptr
) {
    const auto identity = scheduler_error_identity(local, task_ref, slot);
    scheduler_record_error(
        run_control, task_ref, status, graph, local != nullptr ? local->config.shared_context : nullptr, error_site,
        &identity
    );
}

static inline __aicore__ void scheduler_record_dispatch_error(
    __gm__ SchedulerRunControl *run, int64_t task_ref, SchedulerGraphResult status, const SchedulerGraphView *graph,
    const SchedulerLocalState *local, const SchedulerLocalSlotState *slot, const SchedulerTaskMetadata &metadata,
    SchedulerErrorSite site
) {
    auto identity = scheduler_error_identity(local, task_ref, slot);
    if (slot != nullptr && slot->state == SchedulerDispatchSlotState::FILLING &&
        scheduler_task_ref_is_null(slot->task_ref) && !slot->cooperative && metadata.logical_block_num == 1) {
        identity.block_idx = 0;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_BLOCK;
    }
    scheduler_record_error(run, task_ref, status, graph, local->config.shared_context, site, &identity);
}

static inline __aicore__ void scheduler_publish_waiter_metadata(
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
static inline __aicore__ int64_t scheduler_observe_next_waiter(__gm__ SchedulerTaskControl *control) {
    return scheduler_gm_load(control->next_waiter);
}

// One unit initializes the cursor, observes one fanin, or attempts one wake
// registration. A successful registration hands the waiter to another resolver:
// every later call returns its saved verdict without reading that waiter again.
static inline __aicore__ SchedulerRouteProgress scheduler_route_task_step(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, SchedulerRouteCursor &cursor, uint32_t budget,
    bool validate_current_state = true
) {
    uint32_t used = 0;
    auto fail = [&](SchedulerGraphResult error, SchedulerErrorSite site = SchedulerErrorSite::UNKNOWN) __aicore__ {
        cursor.result = SchedulerRouteResult::ERROR;
        scheduler_record_error(run_control, task_ref, error, &graph, context, site);
        return SchedulerRouteProgress{cursor.result, used};
    };
    if (context == nullptr || !scheduler_task_ref_valid(graph, task_ref))
        return fail(SchedulerGraphResult::INVALID_TASK_ID);
    if (cursor.task_ref != SCHEDULER_TASK_REF_INVALID && cursor.task_ref != task_ref)
        return fail(SchedulerGraphResult::INVALID_ARGUMENTS);
    if (cursor.result != SchedulerRouteResult::YIELD) {
        if (cursor.task_ref == SCHEDULER_TASK_REF_INVALID) return fail(SchedulerGraphResult::INVALID_ARGUMENTS);
        if (cursor.result != SchedulerRouteResult::WAITING && cursor.result != SchedulerRouteResult::READY_TO_ENQUEUE &&
            cursor.result != SchedulerRouteResult::COMPLETED && cursor.result != SchedulerRouteResult::ERROR)
            return fail(SchedulerGraphResult::INVALID_ARGUMENTS);
        return {cursor.result, 0};
    }
    if (budget == 0) return {SchedulerRouteResult::YIELD, 0};
    const bool initialize = cursor.task_ref == SCHEDULER_TASK_REF_INVALID;
    auto *control = scheduler_task_control_at(scheduler_state_base, context, task_ref);
    if (initialize) {
        ++used;
        cursor.task_ref = task_ref;
        if (validate_current_state) {
            const int64_t state = scheduler_gm_query(control->state);
            if (state == static_cast<int64_t>(SchedulerTaskState::DONE)) {
                cursor.result = SchedulerRouteResult::COMPLETED;
                return {cursor.result, used};
            }
            if (state != static_cast<int64_t>(SchedulerTaskState::BLOCKED))
                return fail(SchedulerGraphResult::INVALID_ARGUMENTS, SchedulerErrorSite::ROUTE_INVALID_STATE);
        }
        scheduler_observe_cache_line(&control->next_fanin_index);
        cursor.next_fanin = control->next_fanin_index;
        cursor.producer_ref = SCHEDULER_TASK_REF_INVALID;
    }
    SchedulerFaninView fanin{};
    const auto fanin_status = scheduler_node_fanin_view(graph, task_ref, &fanin);
    if (fanin_status != SchedulerGraphResult::OK) return fail(fanin_status);
    if (initialize) cursor.fanin_count = fanin.count;
    if (cursor.fanin_count != fanin.count || cursor.next_fanin < 0 ||
        static_cast<uint32_t>(cursor.next_fanin) > fanin.count || cursor.producer_ref < SCHEDULER_TASK_REF_INVALID ||
        (cursor.producer_ref >= 0 && (static_cast<uint32_t>(cursor.next_fanin) == fanin.count ||
                                      !scheduler_task_ref_valid(graph, cursor.producer_ref))))
        return fail(SchedulerGraphResult::INVALID_FANIN_ID);

    while (used < budget && static_cast<uint32_t>(cursor.next_fanin) < fanin.count) {
        ++used;
        if (cursor.producer_ref == SCHEDULER_TASK_REF_INVALID) {
            int64_t producer = SCHEDULER_TASK_REF_INVALID;
            if (scheduler_fanin_ref(fanin, task_ref, static_cast<uint32_t>(cursor.next_fanin), &producer) !=
                SchedulerGraphResult::OK)
                return fail(SchedulerGraphResult::INVALID_FANIN_ID);

            auto *producer_control = scheduler_task_control_at(scheduler_state_base, context, producer);
            if (scheduler_gm_query(producer_control->state) == static_cast<int64_t>(SchedulerTaskState::DONE)) {
                ++cursor.next_fanin;
            } else {
                cursor.producer_ref = producer;
            }
            continue;
        }
        auto *producer_control = scheduler_task_control_at(scheduler_state_base, context, cursor.producer_ref);
        const int64_t observed = scheduler_gm_query(producer_control->wake_list_head);
        if (observed == SCHEDULER_WAKE_LIST_CLOSED) {
            ++cursor.next_fanin;
            cursor.producer_ref = SCHEDULER_TASK_REF_INVALID;
            continue;
        }
        if (observed < SCHEDULER_WAKE_LIST_OPEN || (observed >= 0 && !scheduler_task_ref_valid(graph, observed)))
            return fail(SchedulerGraphResult::INVALID_TASK_ID);
        scheduler_publish_waiter_metadata(control, observed, cursor.next_fanin, cursor.producer_ref);
        const int64_t actual = scheduler_scalar_atomic_result(context, [&]() __aicore__ {
            return scheduler_gm_compare_exchange(producer_control->wake_list_head, observed, task_ref);
        });
        if (actual == observed) {
            cursor.result = SchedulerRouteResult::WAITING;
            return {cursor.result, used};
        }

        if (actual == SCHEDULER_WAKE_LIST_CLOSED) {
            ++cursor.next_fanin;
            cursor.producer_ref = SCHEDULER_TASK_REF_INVALID;
        } else if (actual < SCHEDULER_WAKE_LIST_OPEN || (actual >= 0 && !scheduler_task_ref_valid(graph, actual))) {
            return fail(SchedulerGraphResult::INVALID_TASK_ID);
        }
    }
    if (static_cast<uint32_t>(cursor.next_fanin) == fanin.count) cursor.result = SchedulerRouteResult::READY_TO_ENQUEUE;
    return {cursor.result, used};
}

static inline __aicore__ SchedulerRouteResult scheduler_route_task(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, bool validate_current_state = true
) {
    SchedulerRouteCursor cursor;
    do {
        scheduler_route_task_step(
            graph, scheduler_state_base, context, run_control, task_ref, cursor, UINT32_MAX, validate_current_state
        );
    } while (cursor.result == SchedulerRouteResult::YIELD);
    return cursor.result;
}

// The bootstrap partition gives every child one argument writer. Its exact
// cache writeback precedes scan arrival; no ready consumer passes the scan gate
// until all writers finish. This only binds descriptors, never tensor values.
static inline __aicore__ bool scheduler_bootstrap_bind_arguments(
    const SchedulerGraphAccess &graph, int64_t task_ref, SchedulerLocalState *local, __gm__ SchedulerRunControl *run
) {
    const auto error = scheduler_bind_graph_arguments(graph, task_ref);
    if (error == SchedulerGraphBindingError::NONE) return true;
    scheduler_record_error(
        run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local, scheduler_graph_binding_error_site(error)
    );
    return false;
}

// Dependency counters retain their Host-initialized values until the scan gate
// opens. Without a counter table, bootstrap registers intrusive wake links.
// The bootstrap barrier is closed for the whole of every core's registration
// pass: a core announces arrival only after its last registration, and the gate
// opens only once every core has arrived. No resolver can detach a wake list
// here, which is what makes publishing the new head before the waiter's link
// safe on this path and nowhere else. A producer the host completed inline
// carries no metadata flags at all, so the executability filter below is what
// keeps this pass off a wake list the host already closed.
static inline __aicore__ SchedulerRouteResult scheduler_bootstrap_route_task(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref
) {
    if (!scheduler_task_ref_valid(graph, task_ref)) {
        scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_TASK_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }
    const auto *dependencies = scheduler_dependencies(scheduler_state_base, context->config.ready_pool_offset);
    if (dependencies != nullptr) {
        const uint64_t remaining = scheduler_gm_fetch_add(
            scheduler_state_at<uint64_t>(scheduler_state_base, dependencies->remaining_offset)[task_ref], UINT64_C(0)
        );
        if (remaining > static_cast<uint64_t>(task_ref)) {
            scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_FANIN_COUNT, &graph, context);
            return SchedulerRouteResult::ERROR;
        }
        return remaining == 0 ? SchedulerRouteResult::READY_TO_ENQUEUE : SchedulerRouteResult::WAITING;
    }
    SchedulerFaninView fanin{};
    const auto fanin_status = scheduler_node_fanin_view(graph, task_ref, &fanin);
    const uint32_t fanin_count = fanin.count;
    if (fanin_status != SchedulerGraphResult::OK) {
        scheduler_record_error(run_control, task_ref, fanin_status, &graph, context);
        return SchedulerRouteResult::ERROR;
    }

    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_ref);
    for (uint32_t next_fanin = 0; next_fanin < fanin_count; ++next_fanin) {
        int64_t producer = -1;
        if (scheduler_fanin_ref(fanin, task_ref, next_fanin, &producer) != SchedulerGraphResult::OK) {
            scheduler_record_error(run_control, task_ref, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
            return SchedulerRouteResult::ERROR;
        }
        if (graph.native_header_offset != 0) {
            SchedulerNodeKind kind{};
            if (scheduler_node_kind(graph, producer, &kind) != SchedulerGraphResult::OK) {
                scheduler_record_error(run_control, producer, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context);
                return SchedulerRouteResult::ERROR;
            }
            if (kind == SchedulerNodeKind::HOST_DONE) continue;
        } else {
            __gm__ SchedulerTaskMetadata *producer_metadata =
                scheduler_task_metadata_at(scheduler_state_base, context, producer);
            scheduler_observe_cache_line(producer_metadata);
            if (!scheduler_task_is_executable(producer_metadata->flags)) continue;
        }

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

        return SchedulerRouteResult::WAITING;
    }

    return SchedulerRouteResult::READY_TO_ENQUEUE;
}

inline __aicore__ uint32_t scheduler_trace_index_at(
    __gm__ void *base, SchedulerLocalState *local, int64_t task_ref, SubtaskSlot subtask, uint32_t block_idx = 0
) {
    const auto *metadata = scheduler_task_metadata_at(base, local, task_ref);
    return scheduler_task_trace_index(metadata->trace_index_base, scheduler_task_lanes(*metadata), subtask, block_idx);
}

static inline __aicore__ void scheduler_capture_execution_identity(
    __gm__ void *base, SchedulerLocalState *local, uint32_t pending_slot, __gm__ const DispatchPayload *payload
) {
    const auto &slot = local->slots[pending_slot];
    auto *traces = scheduler_state_at<SchedulerTaskTrace>(base, local->profiling->trace_cells_offset);
    auto *trace = &traces[scheduler_trace_index_at(
        base, local, slot.task_ref, static_cast<SubtaskSlot>(slot.subtask_slot), slot.block_idx
    )];
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    scheduler_observe_cache_line(&trace->descriptor_cache_observed_cycles);
    trace->execution_run_epoch = summary->run_epoch;
    trace->execution_order_key = UINT64_MAX;
    trace->execution_group_seq = 0;
    uint32_t flags = SCHEDULER_TRACE_IDENTITY_CAPTURED;
    if (payload->function_bin_addr != 0) flags |= SCHEDULER_TRACE_EXECUTED;
    if (slot.cooperative && local->admission != nullptr) {
        trace->execution_order_key = local->admission->scan_cursor;
        flags |= SCHEDULER_TRACE_ORDERED;
        if (slot.gm_mix) {
            trace->execution_group_seq = local->admission->gm_mix.group_seq;
            flags |= SCHEDULER_TRACE_GROUP;
        }
    }
    trace->execution_block_idx = static_cast<uint32_t>(payload->local_context.block_idx);
    trace->execution_block_num = static_cast<uint32_t>(payload->local_context.block_num);
    trace->execution_subtask_slot = slot.subtask_slot;
    trace->execution_pending_slot = pending_slot;
    trace->execution_cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    trace->execution_cluster_lane = local->config.self_lane;
    trace->execution_identity_flags = flags;
    scheduler_publish_cache_line(&trace->descriptor_cache_observed_cycles);
}

static inline __aicore__ bool scheduler_ready_batch_append(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, int64_t task_ref, SchedulerReadyBatch *batch
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

    ++batch->count;

    return true;
}

// Control inboxes use a linked FIFO. The successful tail update owns the
// old-tail link, which must be published before returning to scheduler work.
static inline __aicore__ bool scheduler_ready_batch_push_queue(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, __gm__ SchedulerReadyInbox *inbox,
    SchedulerReadyBatch *batch
) {
    if (batch == nullptr || context == nullptr || inbox == nullptr) return false;
    if (batch->head == SCHEDULER_INBOX_EMPTY) return true;
    if (batch->tail < 0 || batch->head < 0) return false;
    if (!scheduler_ready_endpoint_fits(batch->head) || !scheduler_ready_endpoint_fits(batch->tail)) return false;

    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        const uint64_t packed = scheduler_gm_query(inbox->endpoints);
        if (!scheduler_ready_endpoints_consistent(packed)) return false;
        const int64_t head = scheduler_ready_endpoints_head(packed);
        const int64_t tail = scheduler_ready_endpoints_tail(packed);
        const bool was_empty = scheduler_ready_endpoints_empty(packed);
        const uint64_t next_packed = scheduler_ready_endpoints_pack(was_empty ? batch->head : head, batch->tail);
        if (scheduler_scalar_atomic_result(context, [&]() __aicore__ {
                return scheduler_gm_compare_exchange(inbox->endpoints, packed, next_packed);
            }) != packed) {
            continue;
        }
        if (!was_empty) {
            __gm__ SchedulerTaskControl *old_tail = scheduler_task_control_at(scheduler_state_base, context, tail);
            scheduler_gm_store(old_tail->next_waiter, batch->head);
        }
        scheduler_cache_barrier();
        scheduler_ready_batch_reset(batch);

        return true;
    }
    // The batch is still linked only to itself, so an exhausted retry budget
    // leaves no partially published chain behind; the caller reports it.
    return false;
}

static inline __aicore__ bool scheduler_ready_pool_push_batch(
    __gm__ void *base, SchedulerLocalState *local, uint32_t type, SchedulerReadyBatch *batch
) {
    const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
    if (batch->count == 0 || batch->count > pool.task_count) return false;
    int64_t current = batch->head;
    uint64_t remaining = batch->count;
    while (remaining != 0) {
        if (!scheduler_task_ref_valid(*local->config.graph, current)) return false;
        int64_t refs[8]{};
        uint32_t count = 0;
        do {
            if (!scheduler_task_ref_valid(*local->config.graph, current)) return false;
            refs[count++] = current;
            const auto *metadata = scheduler_task_metadata_at(base, local, current);
            if (!scheduler_task_is_executable(metadata->flags) ||
                scheduler_task_ready_queue(metadata->flags, scheduler_task_lanes(*metadata)) != type)
                return false;
            auto *control = scheduler_task_control_at(base, local, current);
            const int64_t next = scheduler_observe_next_waiter(control);
            --remaining;
            if (remaining == 0 && (current != batch->tail || next != SCHEDULER_INBOX_EMPTY)) return false;
            current = next;
        } while (remaining != 0 && count < 8 && current >= 0);
        const auto queue = scheduler_ready_pool_queue(base, pool, type);
        SchedulerTicketRange range{};
        if (scheduler_ticket_reserve(queue, count, &range) != SchedulerTicketResult::SUCCESS ||
            scheduler_ticket_publish(queue, range, refs) != SchedulerTicketResult::SUCCESS)
            return false;
    }
    scheduler_ready_batch_reset(batch);
    return true;
}

static inline __aicore__ bool scheduler_ready_batch_push(
    __gm__ void *scheduler_state_base, SchedulerLocalState *context, uint32_t queue_class, SchedulerReadyBatch *batch
) {
    if (scheduler_state_base == nullptr || batch == nullptr || context == nullptr ||
        queue_class >= SCHEDULER_READY_QUEUE_COUNT || context->config.ready_pool_offset == 0)
        return false;
    if (batch->head == SCHEDULER_INBOX_EMPTY) return true;
    return scheduler_ready_pool_push_batch(scheduler_state_base, context, queue_class, batch);
}

static inline __aicore__ bool scheduler_control_enqueue(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t ref
) {
    SchedulerNodeKind kind{};
    if (local == nullptr || local->config.control_inboxes_offset == 0 ||
        local->worker_id() >= local->config.runtime_worker_count ||
        scheduler_node_kind(graph, ref, &kind) != SchedulerGraphResult::OK || !scheduler_node_is_control(kind) ||
        scheduler_read_error(run) != 0)
        return false;
    SchedulerReadyBatch batch{};
    auto *inbox =
        scheduler_state_at<SchedulerReadyInbox>(base, local->config.control_inboxes_offset) + local->worker_id();
    if (!scheduler_ready_batch_append(base, local, ref, &batch) ||
        !scheduler_ready_batch_push_queue(base, local, inbox, &batch)) {
        scheduler_record_error(
            run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
            SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED
        );
        return false;
    }
    return true;
}

static inline __aicore__ bool scheduler_ready_pop_from_queue(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, __gm__ SchedulerReadyInbox *inbox, int64_t *task_ref
) {
    if (task_ref == nullptr) return false;
    *task_ref = SCHEDULER_TASK_REF_INVALID;
    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        const uint64_t packed = scheduler_gm_query(inbox->endpoints);
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
        if (scheduler_scalar_atomic_result(context, [&]() __aicore__ {
                return scheduler_gm_compare_exchange(inbox->endpoints, packed, next_packed);
            }) != packed) {
            continue;
        }

        *task_ref = head;
        return true;
    }
    // Bounded contention may defer a non-empty inbox to a later scheduling iteration.

    return true;
}

static inline __aicore__ bool scheduler_ready_pool_claim(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint32_t queue_class, SchedulerReadyClaim *claim
) {
    if (claim == nullptr) return false;
    *claim = {};
    if (base == nullptr || local == nullptr || run == nullptr || local->config.ready_pool_offset == 0 ||
        !scheduler_ready_queue_holder(queue_class, local->config.self_core_type))
        return false;
    const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
    const auto queue = scheduler_ready_pool_queue(base, pool, queue_class);
    auto fail = [&](int64_t ref) __aicore__ {
        scheduler_record_error(
            run, ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local, SchedulerErrorSite::READY_POP_INVALID_HEAD
        );
        return false;
    };
    if (!scheduler_ticket_view_valid(queue)) return fail(SCHEDULER_TASK_REF_INVALID);
    if (queue.capacity == 0) return true;
    for (uint32_t attempt = 0; attempt < 2; ++attempt) {
        const uint64_t head = scheduler_gm_query(queue.control->head);
        if (head > queue.capacity) return fail(SCHEDULER_TASK_REF_INVALID);
        if (head == queue.capacity) return true;
        auto &entry = queue.entries[head].task_ref;
        if (scheduler_gm_query(entry) == SCHEDULER_TASK_REF_INVALID) return true;
        // Only an atomic observation confirms the atomic-published ref. A stale
        // head cannot move ownership because the CAS below validates it again.
        const int64_t ref = scheduler_gm_load(entry);
        if (ref == SCHEDULER_TASK_REF_INVALID) return true;
        if (!scheduler_task_ref_valid(graph, ref)) return fail(ref);
        const auto &metadata = *scheduler_task_metadata_at(base, local, ref);
        if (!scheduler_task_is_executable(metadata.flags) ||
            scheduler_task_ready_queue(metadata.flags, scheduler_task_lanes(metadata)) != queue_class ||
            (scheduler_task_is_mix(metadata.flags) && !local->is_mix_coordinator()))
            return fail(ref);
        if (scheduler_gm_compare_exchange(queue.control->head, head, head + 1) == head) {
            scheduler_cache_barrier();
            claim->task_ref = ref;

            return true;
        }
    }

    return true;
}

static inline __aicore__ bool
scheduler_ready_pool_may_have_work(__gm__ void *base, const SchedulerLocalState *local, uint32_t queue_class) {
    // Invalid discovery must reach the checked claim path instead of hiding work.
    if (base == nullptr || local == nullptr || local->config.ready_pool_offset == 0 ||
        !scheduler_ready_queue_holder(queue_class, local->config.self_core_type))
        return true;
    const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(base, local->config.ready_pool_offset);
    const auto queue = scheduler_ready_pool_queue(base, pool, queue_class);
    if (!scheduler_ticket_view_valid(queue)) return true;
    if (queue.capacity == 0) return false;
    const uint64_t head = scheduler_gm_query(queue.control->head);
    if (head > queue.capacity) return true;
    return head < queue.capacity && scheduler_gm_query(queue.entries[head].task_ref) != SCHEDULER_TASK_REF_INVALID;
}

static inline __aicore__ bool scheduler_claim_ready_for_slot(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t scheduler_count, uint32_t queue_class, SchedulerReadyClaim *claim
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarReadyClaim, -1);
    if (local == nullptr || claim == nullptr || scheduler_count == 0 || scheduler_count > SCHEDULER_WORKER_CAPACITY ||
        local->config.scheduler_index >= scheduler_count ||
        !scheduler_ready_queue_holder(queue_class, local->config.self_core_type))
        return false;
    return scheduler_ready_pool_claim(graph, base, local, run, queue_class, claim);
}

inline __aicore__ void scheduler_initialize_free_slot(SchedulerLocalSlotState *local_slot) {
    local_slot->task_ref = SCHEDULER_TASK_REF_INVALID;
    local_slot->state = SchedulerDispatchSlotState::FREE;
    local_slot->subtask_slot = UINT8_MAX;
    local_slot->cooperative = false;
    local_slot->gm_mix = false;
    local_slot->block_idx = 0;
    local_slot->candidate_index = UINT8_MAX;
    local_slot->args_observed = false;
    local_slot->predicate = SchedulerPredicateState::UNEVALUATED;
}

static inline __aicore__ SchedulerTaskMetadata scheduler_load_dispatch_metadata(
    __gm__ void *base, SchedulerLocalState *scheduler, int64_t task_ref, uint64_t profiling_level
) {
    SchedulerScalarScope scalar_detail(scheduler, SchedPhaseKind::ScalarDispatchMetadata, task_ref);
    const auto *source = scheduler_task_metadata_at(base, scheduler, task_ref);
    SchedulerTaskMetadata metadata{};
    metadata.kernel_ids[0] = source->kernel_ids[0];
    metadata.kernel_ids[1] = source->kernel_ids[1];
    metadata.kernel_ids[2] = source->kernel_ids[2];
    metadata.active_mask = scheduler_task_lanes(*source);
    metadata.flags = source->flags;
    metadata.timing_slot = source->timing_slot;
    metadata.logical_block_num = source->logical_block_num;
    if (scheduler_task_timing_enabled(profiling_level)) metadata.trace_index_base = source->trace_index_base;
    return metadata;
}

static inline __aicore__ bool scheduler_prepare_dispatch_slot(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint64_t profiling_level,
    uint8_t subtask_slot, uint64_t *prepare_start_cycles = nullptr,
    const SchedulerTaskPayloadArguments *arguments = nullptr, bool write_arguments = true
) {
    SchedulerScalarScope scalar_payload(scheduler, SchedPhaseKind::ScalarPayload, ready_claim.task_ref);
    if (scheduler == nullptr || !scheduler_task_ref_valid(graph, ready_claim.task_ref) ||
        scheduler->worker_id() >= scheduler->config.runtime_worker_count ||
        slot_claim.slot_index >= SCHEDULER_PENDING_SLOT_COUNT ||
        scheduler->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM)
        return false;
    if (prepare_start_cycles != nullptr)
        *prepare_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    const LaneMask lanes = scheduler_task_lanes(metadata);
    if (subtask_slot == UINT8_MAX) subtask_slot = scheduler_metadata_single_subtask_slot(lanes);
    __gm__ SchedulerWorkerContext *target =
        scheduler_worker_context_at(scheduler_state_base, scheduler, scheduler->worker_id());
    if (subtask_slot >= 3 || !lanes.has(static_cast<SubtaskSlot>(subtask_slot)) ||
        (target->core_type != static_cast<int32_t>(CoreType::AIC) &&
         target->core_type != static_cast<int32_t>(CoreType::AIV)) ||
        !scheduler_task_is_executable(metadata.flags) ||
        scheduler_metadata_core_type_index(subtask_slot) != scheduler_core_type_index(target->core_type)) {
        scheduler_record_dispatch_error(
            run_control, ready_claim.task_ref, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, scheduler,
            scheduler->slot_state(slot_claim), metadata, SchedulerErrorSite::DISPATCH_INVALID_SHAPE
        );
        return false;
    }
    const uint16_t kernel_id = metadata.kernel_ids[subtask_slot];
    auto &reserved = *scheduler->slot_state(slot_claim);
    if (reserved.state != SchedulerDispatchSlotState::FILLING) return false;
    __gm__ uint64_t *callable_addresses =
        reinterpret_cast<__gm__ uint64_t *>(scheduler->config.callable_addresses_address);
    const bool inline_task = scheduler_task_is_inline(metadata.flags);
    uint64_t callable_address = UINT64_C(1);
    if (!inline_task && !scheduler_lookup_callable_address(
                            callable_addresses, scheduler->config.callable_addresses_count, kernel_id, &callable_address
                        )) {
        scheduler_record_dispatch_error(
            run_control, ready_claim.task_ref, SchedulerGraphResult::INVALID_CALLABLE, &graph, scheduler, &reserved,
            metadata, SchedulerErrorSite::DISPATCH_INVALID_CALLABLE
        );
        return false;
    }

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
        if (graph.native_header_offset != 0 && !reserved.cooperative && !reserved.args_observed) {
            const auto error = scheduler_observe_graph_arguments(graph, ready_claim.task_ref);
            if (error != SchedulerGraphBindingError::NONE) {
                scheduler_record_dispatch_error(
                    run_control, ready_claim.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    &reserved, metadata, scheduler_graph_binding_error_site(error)
                );
                return false;
            }
            reserved.args_observed = true;
        }
        status = scheduler_materialize_task_payload_resolved(
            graph, task, callable_address, payload, arguments, write_arguments
        );
        if (status == SchedulerGraphResult::OK && scheduler_task_has_predicate(metadata.flags)) {
            if (reserved.predicate == SchedulerPredicateState::UNEVALUATED) {
                const SchedulerPredicateResult predicate =
                    scheduler_evaluate_task_predicate(graph, ready_claim.task_ref);
                if (predicate == SchedulerPredicateResult::MALFORMED) {
                    scheduler_record_dispatch_error(
                        run_control, ready_claim.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                        &reserved, metadata, SchedulerErrorSite::DISPATCH_INVALID_PREDICATE
                    );
                    return false;
                }
                reserved.predicate = predicate == SchedulerPredicateResult::PASS ? SchedulerPredicateState::PASS :
                                                                                   SchedulerPredicateState::FAIL;
            }
            if (reserved.predicate == SchedulerPredicateState::FAIL) payload->function_bin_addr = 0;
        }
    }
    if (status != SchedulerGraphResult::OK) {
        scheduler_record_dispatch_error(
            run_control, ready_claim.task_ref, status, &graph, scheduler, &reserved, metadata,
            SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
        );
        return false;
    }
    payload->local_context.block_idx = reserved.block_idx;
    payload->local_context.block_num = metadata.logical_block_num;
    payload->global_context.sub_block_id = scheduler_task_is_mix(metadata.flags) ?
                                               (scheduler->config.self_lane == 2 ? 1 : 0) :
                                               (subtask_slot == 2 ? 1 : 0);
    return true;
}

static inline __aicore__ void scheduler_stage_dispatch_slot(
    __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint8_t subtask_slot
) {
    auto *payload =
        scheduler_state_at<DispatchPayload>(scheduler_state_base, scheduler->slot_payload_offset(slot_claim));
    SchedulerLocalSlotState *local_slot = scheduler->slot_state(slot_claim);
    local_slot->task_ref = ready_claim.task_ref;
    scheduler->set_timing_slot(slot_claim.slot_index, metadata.timing_slot);
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->subtask_slot = subtask_slot;
    SchedulerScalarScope scalar_cache(scheduler, SchedPhaseKind::ScalarCachePublish, ready_claim.task_ref);
    scheduler_writeback_dispatch_payload(payload);
}

static inline __aicore__ void scheduler_commit_dispatch_slot(
    __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint64_t profiling_level,
    uint8_t subtask_slot, uint64_t prepare_start_cycles
) {
    const bool task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t dispatch_start_cycles = prepare_start_cycles;
    const uint64_t ready_publish_cycles = schedule_timing_enabled ? scheduler_cycles() : 0;
    scheduler_protocol_payload_publication(scheduler_state_base, scheduler, slot_claim, ready_claim, true);
    scheduler_local_ready_publish(scheduler, slot_claim.slot_index);
    scheduler_protocol_payload_publication(scheduler_state_base, scheduler, slot_claim, ready_claim, false);
    if (task_timing_enabled) {
        __gm__ SchedulerTaskTrace *traces =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->profiling->trace_cells_offset);
        __gm__ SchedulerTaskTrace *trace = &traces[scheduler_task_trace_index(
            metadata.trace_index_base, scheduler_task_lanes(metadata), static_cast<SubtaskSlot>(subtask_slot),
            scheduler->slot_state(slot_claim)->block_idx
        )];
        trace->worker_id = scheduler->worker_id();
        trace->task_ref = static_cast<uint64_t>(ready_claim.task_ref);
        if (phase_timing_enabled) {
            trace->ready_source = static_cast<uint64_t>(ready_claim.source);
            trace->publication_mode = static_cast<uint64_t>(ready_claim.publication_mode);
            if (ready_claim.source == SchedulerReadySource::DIRECT_RESOLVE ||
                ready_claim.source == SchedulerReadySource::COOPERATIVE) {
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

static inline __aicore__ void scheduler_publish_dispatch_slot(
    __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint64_t profiling_level,
    uint8_t subtask_slot, uint64_t prepare_start_cycles = 0
) {
    SchedulerScalarScope scalar_detail(scheduler, SchedPhaseKind::ScalarDispatchPublish, ready_claim.task_ref);
    scheduler_stage_dispatch_slot(scheduler_state_base, scheduler, slot_claim, ready_claim, metadata, subtask_slot);
    scheduler_cache_barrier();
    scheduler_commit_dispatch_slot(
        scheduler_state_base, scheduler, slot_claim, ready_claim, metadata, profiling_level, subtask_slot,
        prepare_start_cycles
    );
}

static inline __aicore__ bool scheduler_complete_inline_task(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, uint64_t profiling_level
);

static inline __aicore__ bool scheduler_fill_dispatch_slot(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, uint64_t profiling_level
) {
    SchedulerScalarScope scalar_detail(scheduler, SchedPhaseKind::ScalarDispatchFill, ready_claim.task_ref);
    if (scheduler == nullptr || scheduler->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM ||
        slot_claim.slot_index >= SCHEDULER_PENDING_SLOT_COUNT)
        return false;
    auto &slot = *scheduler->slot_state(slot_claim);
    if (slot.state == SchedulerDispatchSlotState::FREE) slot.state = SchedulerDispatchSlotState::FILLING;
    const SchedulerTaskMetadata metadata =
        scheduler_load_dispatch_metadata(scheduler_state_base, scheduler, ready_claim.task_ref, profiling_level);
    if (scheduler_task_is_inline(metadata.flags)) {
        if (slot.state != SchedulerDispatchSlotState::FILLING) return false;
        scheduler_initialize_free_slot(&slot);
        scheduler->set_timing_slot(slot_claim.slot_index, -1);
        return scheduler_complete_inline_task(
            graph, scheduler_state_base, scheduler, run_control, ready_claim.task_ref, profiling_level
        );
    }
    const uint8_t subtask_slot = scheduler_metadata_single_subtask_slot(scheduler_task_lanes(metadata));
    uint64_t prepare_start_cycles = 0;
    if (!scheduler_prepare_dispatch_slot(
            graph, scheduler_state_base, scheduler, run_control, slot_claim, ready_claim, metadata, profiling_level,
            subtask_slot, &prepare_start_cycles
        ))
        return false;
    scheduler_publish_dispatch_slot(
        scheduler_state_base, scheduler, slot_claim, ready_claim, metadata, profiling_level, subtask_slot,
        prepare_start_cycles
    );
    return true;
}

// These helpers also support direct inclusion of the publication/resolver headers.
inline __aicore__ SchedulerReadyAppendResult scheduler_ready_publication_append(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, int64_t ref,
    SchedulerReadyPublication &publication
);
static inline __aicore__ SchedulerReadyPublishResult scheduler_ready_publication_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local,
    SchedulerReadyPublication &publication
);

// Direct candidates belong only to a synchronous call, never to a saved FINISH cursor.
struct SchedulerCompletionDirect {
    SchedulerReadyClaim *ready{nullptr};
    uint32_t core_type{0};
};

static __aicore__ __attribute__((noinline)) SchedulerResolveProgress scheduler_resolve_completion_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, SchedulerResolveCursor &cursor, uint32_t budget, uint64_t profiling_level,
    SchedulerCompletionDirect *direct, bool validate_done_state
);

static inline __aicore__ bool scheduler_resolve_completion(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, uint64_t profiling_level, bool validate_done_state,
    SchedulerReadyClaim *direct_ready, uint32_t direct_core_type
) {
    SchedulerResolveCursor cursor{};
    SchedulerCompletionDirect direct{direct_ready, direct_core_type};
    for (;;) {
        const auto progress = scheduler_resolve_completion_step(
            graph, scheduler_state_base, context, run_control, task_ref, cursor, UINT32_MAX, profiling_level, &direct,
            validate_done_state
        );
        if (progress.result == SchedulerResolveResult::ERROR) return false;
        if (progress.result == SchedulerResolveResult::DONE) return true;
    }
}

// The ready owner services dependency-only nodes without a payload or kernel
// trace. Successors are first-enqueued, so an inline chain cannot recurse.
static inline __aicore__ bool scheduler_complete_inline_task(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, uint64_t profiling_level
) {
    SchedulerScalarScope scalar_detail(local, SchedPhaseKind::ScalarInlineComplete, task_ref);
    if (local == nullptr || !scheduler_task_ref_valid(graph, task_ref)) return false;
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    if (!scheduler_task_is_executable(metadata.flags) || !scheduler_task_is_inline(metadata.flags) ||
        scheduler_task_is_mix(metadata.flags) || scheduler_task_is_spmd(metadata.flags) ||
        scheduler_task_requires_sync_start(metadata.flags) || scheduler_task_has_predicate(metadata.flags) ||
        metadata.logical_block_num != 1 || metadata.kernel_ids[0] != UINT16_MAX ||
        metadata.kernel_ids[1] != UINT16_MAX || metadata.kernel_ids[2] != UINT16_MAX) {
        scheduler_record_error(
            run, task_ref, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, local,
            SchedulerErrorSite::COMPLETION_INVALID_SHAPE
        );
        return false;
    }
    auto *control = scheduler_task_control_at(base, local, task_ref);
    if (scheduler_gm_load(control->state) != static_cast<int64_t>(SchedulerTaskState::BLOCKED)) {
        scheduler_record_error(
            run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
            SchedulerErrorSite::ROUTE_INVALID_STATE
        );
        return false;
    }
    scheduler_protocol_task_done(base, local, task_ref, true);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    scheduler_protocol_task_done(base, local, task_ref, false);
    if (!scheduler_resolve_completion(
            graph, base, local, run, task_ref, profiling_level, false, nullptr, local->config.self_core_type
        ))
        return false;
    ++local->pending_completed;
    return true;
}

#include "scheduler_completion_resolver.h"
