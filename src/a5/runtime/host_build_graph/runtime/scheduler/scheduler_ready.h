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
#include "scheduler_prepared_execution.h"
#include "scheduler_cluster_signal.h"
#include "scheduler_error.h"
#include "scheduler_task_catalog.h"

enum class SchedulerCompletionObservation : uint8_t { PENDING, DONE, INVALID };

static inline __aicore__ SchedulerCompletionObservation
scheduler_confirm_task_completion(__gm__ SchedulerTaskControl *control) {
    const uint64_t confirmed = scheduler_gm_fetch_add(control->completion_state, UINT64_C(0));
    if (!scheduler_completion_valid(confirmed)) return SchedulerCompletionObservation::INVALID;
    return scheduler_completion_done(confirmed) ? SchedulerCompletionObservation::DONE :
                                                  SchedulerCompletionObservation::PENDING;
}

// A bypass load is only a pending hint. Completion (and malformed hints) must
// be confirmed atomically from the one canonical word before dependency use.
static inline __aicore__ SchedulerCompletionObservation
scheduler_observe_task_completion(__gm__ SchedulerTaskControl *control) {
    const uint64_t hint = scheduler_gm_query(control->completion_state);
    if (scheduler_completion_valid(hint) && !scheduler_completion_done(hint))
        return SchedulerCompletionObservation::PENDING;
    return scheduler_confirm_task_completion(control);
}

enum class SchedulerPredicateResult : uint8_t {
    PASS = 0,
    FAIL = 1,
    MALFORMED = 2,
};

enum class SchedulerPredicateState : uint8_t { UNEVALUATED, PASS, FAIL };

struct SchedulerReadyClaim {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    SchedulerReadySource source{SchedulerReadySource::CATALOG};
    uint64_t state_probe_start_cycles{0};
    uint64_t state_probe_end_cycles{0};
};

enum class SchedulerGmMixPhase : uint8_t { IDLE, CLAIMED, PUBLISHED, ARRIVED, RUNNING, RELEASED };

struct SchedulerGmMixLocalState {
    uint64_t mailbox_offset{0};
    uint64_t group_seq{0};
    uint64_t task_last_seq{0};
    uint64_t prepare_start_cycles{0};
    uint64_t vector_last_published_seq[2]{};
    uint32_t block_idx{0};
    uint32_t cluster_index{UINT32_MAX};
    uint8_t slot_index{UINT8_MAX};
    uint8_t subtask_slot{UINT8_MAX};
    SchedulerGmMixPhase phase{SchedulerGmMixPhase::IDLE};
};

struct SchedulerClusterMixLocalState {
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t anchor_state{0};
    uint64_t prepare_start_cycles{0};
    uint64_t sync_tail{0};
    uint32_t generation{0};
    uint32_t block_idx{0};
    LaneMask active_mask{};
    uint8_t slot_index{UINT8_MAX};
    uint8_t subtask_slot{UINT8_MAX};
    SchedulerGmMixPhase phase{SchedulerGmMixPhase::IDLE};
};
static_assert(sizeof(SchedulerClusterMixLocalState) == 48 && alignof(SchedulerClusterMixLocalState) <= 8);

struct SchedulerClusterMixPending {
    SchedulerPreparedHandle anchor{};
    SchedulerFaninView fanin{};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t block_idx{0};
    uint32_t next_fanin{0};
};
static_assert(alignof(SchedulerClusterMixPending) <= 8);

struct SchedulerAdmissionLocalState {
    uint64_t epoch{0};
    uint64_t control_offset{0};
    uint64_t records_offset{0};
    uint64_t record_stride{0};
    uint64_t log_offset{0};
    uint64_t capacity{0};
    int64_t pending_task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t scan_cursor{0};
    int64_t current_task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t current_block_num{0};
    LaneMask current_active_mask{};
    uint8_t current_flags{0};
    uint64_t current_claim_limit{0};
    uint64_t current_mix_close_seal_offset{0};
    uint32_t current_claim_positions{0};
    uint8_t current_single_subtask{UINT8_MAX};
    LaneMask current_resource_mask{};
    bool current_uses_ticket_claim{false};
    bool initialized{false};
    bool sync_claimed{false};
    bool ticket_exhausted{false};
    SchedulerGmMixLocalState gm_mix{};
    // READY freezes topology for this run. This cache contains membership
    // facts only, never a task frontier or an execution permission.
    uint64_t topology_worker_count{0};
};

static_assert(sizeof(SchedulerGmMixLocalState) == 64, "GM group-local state size changed");
static_assert(sizeof(SchedulerAdmissionLocalState) == 184, "admission-local state size changed");

struct SchedulerFreeSlotClaim {
    uint32_t slot_index{UINT32_MAX};
};

struct SchedulerLocalSlotState {
    SchedulerPreparedHandle prepared{};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    SchedulerDispatchSlotState state{SchedulerDispatchSlotState::EMPTY};
    uint8_t subtask_slot{UINT8_MAX};
    uint32_t block_idx{0};
    SchedulerPredicateState predicate{SchedulerPredicateState::UNEVALUATED};
    bool args_observed{false};
    bool gm_mix{false};
    bool cluster_mix{false};
    bool execution_token{false};
};

// Immutable for one READY publication. Offsets address GM. Configuration and
// Scheduler-private mutable bookkeeping live in core-local storage; shared
// coordination state remains in GM.
struct SchedulerLocalConfig {
    __gm__ const SchedulerPreparedLayout *prepared{nullptr};
    __gm__ const SchedulerWorkerContext *shared_context{nullptr};
    const SchedulerGraphView *graph{nullptr};
    uint64_t task_controls_offset{0};
    uint64_t task_catalog_offset{0};
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
    uint16_t worker_ids[PLATFORM_CORES_PER_BLOCKDIM]{};
    uint16_t runtime_worker_count{0};
    // Clusters in this run determine the physical AIC/AIV participant counts.
    uint16_t cluster_count{0};
    uint8_t self_lane{UINT8_MAX};
    // This core's own type, which is also the only type of ready work it can
    // claim for itself. Held here so the dispatch path needs no worker-context
    // read to select its immutable engine catalog.
    uint32_t self_core_type{UINT32_MAX};
};

struct SchedulerLocalProfilingState {
    SchedulerScalarTrace scalar;
    uint64_t trace_cells_offset{0};
    uint64_t activity_buffers_offset{0};
    uint64_t protocol_layout_offset{0};
    SchedulerExecutorTaskTrace executor_traces[SCHEDULER_PENDING_SLOT_COUNT]{};
    int32_t timing_slots[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint32_t loop_iter{0};

    inline __aicore__ SchedulerLocalProfilingState() {
        for (auto &timing : timing_slots)
            timing = -1;
    }
};

struct SchedulerAsyncLocalState;

inline constexpr uint32_t SCHEDULER_EXECUTION_TOKEN_COUNT = 4;
enum class SchedulerExecutionTokenPhase : uint8_t { EMPTY, TICKETED, ACQUIRING, WAIT_FANIN };

struct SchedulerExecutionToken {
    SchedulerPreparedHandle prepared{};
    SchedulerFaninView fanin{};
    int64_t task_ref{SCHEDULER_TASK_REF_INVALID};
    uint64_t discovery_cycles{0};
    uint64_t acquired_cycles{0};
    uint32_t block_idx{0};
    uint32_t next_fanin{0};
    uint8_t subtask_slot{UINT8_MAX};
    SchedulerExecutionTokenPhase phase{SchedulerExecutionTokenPhase::EMPTY};
    SchedulerPredicateState predicate{SchedulerPredicateState::UNEVALUATED};
};
static_assert(alignof(SchedulerExecutionToken) <= 8);

inline __aicore__ bool scheduler_task_uses_execution_tokens(uint8_t flags) {
    return scheduler_task_is_executable(flags) && !scheduler_task_is_inline(flags) && !scheduler_task_is_mix(flags) &&
           !scheduler_task_requires_admission(flags);
}

struct SchedulerLocalState {
    SchedulerLocalConfig config{};
    SchedulerLocalProfilingState *profiling{nullptr};
    SchedulerAdmissionLocalState *admission{nullptr};
    SchedulerAsyncLocalState *async{nullptr};
    SCHEDULER_CLUSTER_SSBUF SchedulerClusterSignal *cluster_signal{nullptr};
    uint32_t *direct_slot{nullptr};
    uint64_t pending_completed{0};
    uint64_t last_completion_flush_cycles{0};
    uint32_t control_peer_cursor{0};
    uint32_t cluster_prepare_seen{0};
    SchedulerClusterMixLocalState cluster_mix{};
    SchedulerClusterMixPending cluster_pending{};
    SchedulerExecutionToken execution_tokens[SCHEDULER_EXECUTION_TOKEN_COUNT]{};
    uint64_t issue_observed_head{0};
    int64_t issue_task_ref{SCHEDULER_TASK_REF_INVALID};
    uint32_t issue_next_fanin{0};
    uint8_t issue_observe_attempts{0};
    uint32_t execution_check_cursor{0};
    int64_t continuation_task_ref{SCHEDULER_TASK_REF_INVALID};

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
    inline __aicore__ SchedulerLocalSlotState *slot_state(const SchedulerFreeSlotClaim &claim) {
        return &slots[claim.slot_index];
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

inline __aicore__ bool
scheduler_slot_uses_admission(const SchedulerLocalState *local, const SchedulerLocalSlotState &slot) {
    return local->admission != nullptr && !slot.cluster_mix && !slot.execution_token;
}

static_assert(SCHEDULER_WORKER_CAPACITY < UINT16_MAX, "worker IDs must fit the local config");
static_assert(PLATFORM_CORES_PER_BLOCKDIM <= 8, "worker trace mask must fit one byte");
static_assert(SCHEDULER_PENDING_SLOT_COUNT == 2, "local ready head requires two slots");
static_assert(alignof(SchedulerLocalSlotState) == alignof(uint64_t));
static_assert(alignof(SchedulerLocalState) == alignof(uint64_t));
static_assert(sizeof(void *) != 8 || sizeof(SchedulerLocalConfig) == 96, "64-bit local config size changed");
static_assert(sizeof(void *) != 8 || sizeof(SchedulerLocalSlotState) == 40, "64-bit local slot size changed");
static_assert(
    sizeof(void *) != 8 || sizeof(SchedulerLocalProfilingState) == 88 + sizeof(SchedulerScalarTrace),
    "64-bit local profiling size changed"
);

inline __aicore__ bool scheduler_has_direct_slot(const SchedulerLocalState *local) {
    return local->direct_slot != nullptr && *local->direct_slot != UINT32_MAX;
}

class SchedulerDirectSlotScope {
public:
    inline __aicore__ SchedulerDirectSlotScope(SchedulerLocalState *local, uint32_t *slot) :
        local_(local),
        previous_(local->direct_slot) {
        local_->direct_slot = slot;
    }
    inline __aicore__ ~SchedulerDirectSlotScope() { local_->direct_slot = previous_; }
    SchedulerDirectSlotScope(const SchedulerDirectSlotScope &) = delete;
    SchedulerDirectSlotScope &operator=(const SchedulerDirectSlotScope &) = delete;

private:
    SchedulerLocalState *local_;
    uint32_t *previous_;
};

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
    if (context->worker_contexts_offset > UINT32_MAX || context->callable_addresses_count > UINT32_MAX ||
        context->task_metadata_offset > UINT32_MAX || context->runtime_worker_count == 0 ||
        context->runtime_worker_count > SCHEDULER_WORKER_CAPACITY ||
        context->worker_index >= context->runtime_worker_count)
        return false;
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
    if (context->prepared_layout_offset != summary->prepared_layout_offset) return false;
    if (context->prepared_layout_offset != 0) {
        if (!scheduler_prepared_layout_valid(
                base, context->scheduler_state_bytes, context->prepared_layout_offset, graph->task_count,
                summary->run_epoch
            ))
            return false;
        config.prepared = scheduler_state_at<SchedulerPreparedLayout>(base, context->prepared_layout_offset);
    }
    if (graph->native_header_offset != 0) {
        const auto *header = scheduler_native_header_checked(
            graph->scheduler_state_address, graph->scheduler_state_bytes, graph->native_header_offset,
            graph->task_count, graph->original_task_count
        );
        if (header == nullptr || header->run_epoch != summary->run_epoch ||
            graph->scheduler_state_address != reinterpret_cast<uint64_t>(base))
            return false;
        local->control_peer_cursor = static_cast<uint32_t>(context->worker_index + 1);
    }
    config.task_catalog_offset = summary->task_catalog_offset;
    if (config.task_catalog_offset == 0 || !scheduler_task_catalog_validate(
                                               base, context->scheduler_state_bytes, config.task_catalog_offset,
                                               graph->task_count, static_cast<uint32_t>(context->cluster_count)
                                           ))
        return false;
    config.runtime_worker_count = static_cast<uint16_t>(context->runtime_worker_count);
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
        const int32_t rank = target->type_rank;
        const uint64_t type_worker_count = cluster_count * (lane == 0 ? 1U : PLATFORM_AIV_CORES_PER_BLOCKDIM);
        if (rank < 0 || static_cast<uint64_t>(rank) >= type_worker_count) return false;
        config.worker_ids[lane] = static_cast<uint16_t>(worker_id);
        if (worker_id == context->worker_index) {
            config.self_lane = static_cast<uint8_t>(lane);
        }
    }
    if (config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    // The loop above checked that lane 0 is the AIC and the rest are AIVs, so
    // the lane fixes the type and its immutable execution directory.
    config.self_core_type = config.self_lane == 0 ? 0U : 1U;
    return true;
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
        context->worker_id() >= SCHEDULER_WORKER_CAPACITY)
        return nullptr;
    return scheduler_state_at<SchedulerActivityBuffer>(
        scheduler_state_base,
        context->profiling->activity_buffers_offset + context->worker_id() * sizeof(SchedulerActivityBuffer)
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
    if (local->cluster_mix.task_ref == task_ref && local->cluster_mix.phase != SchedulerGmMixPhase::IDLE) {
        identity.run_epoch = local->config.prepared->run_epoch;
        identity.block_idx = local->cluster_mix.block_idx;
        identity.group_seq = local->cluster_mix.generation;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_EPOCH | SCHEDULER_ERROR_HAS_BLOCK | SCHEDULER_ERROR_HAS_GROUP;
    }
    if (slot != nullptr && !scheduler_task_ref_is_null(task_ref) && slot->task_ref == task_ref &&
        slot->state != SchedulerDispatchSlotState::FREE && slot->state != SchedulerDispatchSlotState::EMPTY) {
        identity.block_idx = slot->block_idx;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_BLOCK;
    }
    return identity;
}

static_assert(alignof(SchedulerLocalState) <= 8 && alignof(SchedulerLocalSlotState) <= 8);

static __aicore__ __attribute__((noinline)) void scheduler_record_error(
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
        scheduler_task_ref_is_null(slot->task_ref) && metadata.logical_block_num == 1) {
        identity.block_idx = 0;
        identity.valid_fields |= SCHEDULER_ERROR_HAS_BLOCK;
    }
    scheduler_record_error(run, task_ref, status, graph, local->config.shared_context, site, &identity);
}

inline __aicore__ uint32_t scheduler_trace_index_at(
    __gm__ void *base, SchedulerLocalState *local, int64_t task_ref, SubtaskSlot subtask, uint32_t block_idx = 0
) {
    const auto *metadata = scheduler_task_metadata_at(base, local, task_ref);
    return scheduler_task_trace_index(metadata->trace_index_base, scheduler_task_lanes(*metadata), subtask, block_idx);
}

static inline __aicore__ void scheduler_capture_execution_identity(
    __gm__ void *base, SchedulerLocalState *local, uint32_t pending_slot, uint64_t function_address,
    const __gm__ LocalContext &local_context
) {
    const auto &slot = local->slots[pending_slot];
    auto *traces = scheduler_state_at<SchedulerTaskTrace>(base, local->profiling->trace_cells_offset);
    auto *trace = &traces[scheduler_trace_index_at(
        base, local, slot.task_ref, static_cast<SubtaskSlot>(slot.subtask_slot), slot.block_idx
    )];
    const auto *summary = scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    scheduler_observe_cache_line(&trace->execution_run_epoch);
    trace->execution_run_epoch = summary->run_epoch;
    trace->execution_order_key = UINT64_MAX;
    trace->execution_group_seq = 0;
    uint32_t flags = SCHEDULER_TRACE_IDENTITY_CAPTURED;
    if (function_address != 0) flags |= SCHEDULER_TRACE_EXECUTED;
    if (scheduler_slot_uses_admission(local, slot)) {
        trace->execution_order_key = local->admission->scan_cursor;
        flags |= SCHEDULER_TRACE_ORDERED;
        if (slot.gm_mix) {
            trace->execution_group_seq = local->admission->gm_mix.group_seq;
            flags |= SCHEDULER_TRACE_GROUP;
        }
    }
    if (slot.cluster_mix) {
        trace->execution_group_seq = local->cluster_mix.generation;
        flags |= SCHEDULER_TRACE_GROUP | SCHEDULER_TRACE_CLUSTER_GROUP;
    }
    trace->execution_block_idx = static_cast<uint32_t>(local_context.block_idx);
    trace->execution_block_num = static_cast<uint32_t>(local_context.block_num);
    trace->execution_subtask_slot = slot.subtask_slot;
    trace->execution_pending_slot = pending_slot;
    trace->execution_cluster_index = static_cast<uint32_t>(local->config.shared_context->cluster_index);
    trace->execution_cluster_lane = local->config.self_lane;
    trace->execution_identity_flags = flags;
    scheduler_publish_cache_line(&trace->execution_run_epoch);
}

static inline __aicore__ bool scheduler_prepare_cluster_lane(__gm__ void *base, SchedulerLocalState *local) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarClusterPrepare);
    auto *signal = local->cluster_signal;
    if (signal == nullptr) return true;
    const uint32_t lane = local->config.self_lane;
    const auto *prepared = local->config.prepared;
    if (prepared == nullptr || lane >= PLATFORM_CORES_PER_BLOCKDIM) return false;
    const uint32_t generation = scheduler_cluster_load_acquire(&signal->prepare_generation);
    if (generation == 0 || (generation & 1U) != 0 || generation == local->cluster_prepare_seen) return true;
    const uint64_t epoch = scheduler_cluster_load_field(&signal->run_epoch);
    const int64_t task_ref = scheduler_cluster_load_field(&signal->prepare_task_ref);
    scope.set_task_ref(task_ref);
    const uint32_t block = scheduler_cluster_load_field(&signal->prepare_block);
    const uint32_t mask = scheduler_cluster_load_field(&signal->prepare_mask);
    if (scheduler_cluster_load_acquire(&signal->prepare_generation) != generation) return true;
    if (mask == 0 && task_ref == SCHEDULER_TASK_REF_INVALID) {
        local->cluster_prepare_seen = generation;
        return true;
    }
    if (epoch != prepared->run_epoch || task_ref < 0 || static_cast<uint64_t>(task_ref) >= prepared->task_count)
        return false;
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    if (!scheduler_task_is_mix(metadata.flags) || mask != scheduler_task_lanes(metadata).raw() ||
        block >= metadata.logical_block_num)
        return false;
    if ((mask & (1U << lane)) != 0) {
        uint64_t index = UINT64_MAX;
        if (!scheduler_prepared_execution_index(base, prepared, task_ref, block, static_cast<uint8_t>(lane), &index))
            return false;
        scheduler_observe_prepared_call(scheduler_prepared_call_at(base, prepared, index));
        if (!scheduler_prepared_call_valid(base, prepared, index, task_ref, block, static_cast<uint8_t>(lane)))
            return false;
        if (scheduler_cluster_load_acquire(&signal->prepare_generation) != generation) return true;
        scheduler_cluster_store_release(&signal->prepared[lane], generation);
    }
    local->cluster_prepare_seen = generation;
    return true;
}

// Host retains this zero metadata only for verified, already resolved nodes.
// Live native controls can also have flags==0, but always retain block_num=1.
// This immutable topology certificate grants no permission for device-live work.
static inline __aicore__ bool scheduler_predecessor_is_host_done(
    const SchedulerGraphAccess &graph, int64_t task_ref, const __gm__ SchedulerTaskMetadata &metadata
) {
    if (metadata.flags != 0 || metadata.logical_block_num != 0 || !scheduler_task_lanes(metadata).is_empty())
        return false;
    if (graph.native_header_offset == 0) return true;
    SchedulerNodeKind kind{};
    return scheduler_node_kind(graph, task_ref, &kind) == SchedulerGraphResult::OK &&
           kind == SchedulerNodeKind::HOST_DONE;
}

static inline __aicore__ bool scheduler_observe_fanin_prefix(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t task_ref, const SchedulerFaninView &fanin, uint32_t &next_fanin, bool *ready
) {
    SchedulerScalarScope scope(local, SchedPhaseKind::ScalarFaninObserve, task_ref);
    if (ready == nullptr || next_fanin > fanin.count) {
        scheduler_record_error(
            run, task_ref, SchedulerGraphResult::INVALID_FANIN_COUNT, &graph, local,
            SchedulerErrorSite::ROUTE_INVALID_STATE
        );
        return false;
    }
    *ready = false;
    while (next_fanin < fanin.count) {
        int64_t predecessor = SCHEDULER_TASK_REF_INVALID;
        const auto status = scheduler_fanin_ref(fanin, task_ref, next_fanin, &predecessor);
        if (status != SchedulerGraphResult::OK) {
            scheduler_record_error(run, task_ref, status, &graph, local, SchedulerErrorSite::ROUTE_INVALID_STATE);
            return false;
        }
        const auto &metadata = *scheduler_task_metadata_at(base, local, predecessor);
        if (!scheduler_predecessor_is_host_done(graph, predecessor, metadata)) {
            const auto observed =
                scheduler_observe_task_completion(scheduler_task_control_at(base, local, predecessor));
            if (observed == SchedulerCompletionObservation::INVALID) {
                scheduler_record_error(
                    run, predecessor, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                    SchedulerErrorSite::COMPLETION_TASK_NOT_DONE
                );
                return false;
            }
            if (observed != SchedulerCompletionObservation::DONE) return true;
        }
        ++next_fanin;
        if (next_fanin % 32 == 0 && scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    }
    *ready = true;
    return true;
}

inline __aicore__ void scheduler_initialize_free_slot(SchedulerLocalSlotState *local_slot) {
    local_slot->prepared = {};
    local_slot->task_ref = SCHEDULER_TASK_REF_INVALID;
    local_slot->state = SchedulerDispatchSlotState::FREE;
    local_slot->subtask_slot = UINT8_MAX;
    local_slot->gm_mix = false;
    local_slot->cluster_mix = false;
    local_slot->execution_token = false;
    local_slot->block_idx = 0;
    local_slot->args_observed = false;
    local_slot->predicate = SchedulerPredicateState::UNEVALUATED;
}

inline __aicore__ uint32_t scheduler_execution_token_count(const SchedulerLocalState *local) {
    uint32_t count = 0;
    for (const auto &token : local->execution_tokens)
        count += token.phase != SchedulerExecutionTokenPhase::EMPTY;
    for (const auto &slot : local->slots)
        count += slot.execution_token && slot.state != SchedulerDispatchSlotState::FREE;
    return count;
}

inline __aicore__ bool scheduler_has_execution_tokens(const SchedulerLocalState *local) {
    return scheduler_execution_token_count(local) != 0;
}

inline __aicore__ bool scheduler_execution_token_has_task(const SchedulerLocalState *local, int64_t task_ref) {
    for (const auto &token : local->execution_tokens)
        if (token.phase != SchedulerExecutionTokenPhase::EMPTY && token.task_ref == task_ref) return true;
    for (const auto &slot : local->slots)
        if (slot.state != SchedulerDispatchSlotState::FREE && slot.task_ref == task_ref) return true;
    return false;
}

// The physical slot gives its sole execution handle back to its owner's window.
inline __aicore__ bool scheduler_restore_execution_token(SchedulerLocalState *local, uint32_t index) {
    if (local == nullptr || local->config.graph == nullptr || index >= SCHEDULER_PENDING_SLOT_COUNT) return false;
    auto &slot = local->slots[index];
    if (!slot.execution_token || slot.state != SchedulerDispatchSlotState::READY || slot.prepared.claimed_state == 0)
        return false;
    for (auto &token : local->execution_tokens) {
        if (token.phase != SchedulerExecutionTokenPhase::EMPTY) continue;
        SchedulerFaninView fanin{};
        if (scheduler_node_fanin_view(*local->config.graph, slot.task_ref, &fanin) != SchedulerGraphResult::OK)
            return false;
        token = {};
        token.fanin = fanin;
        token.task_ref = slot.task_ref;
        token.block_idx = slot.block_idx;
        token.subtask_slot = slot.subtask_slot;
        token.prepared = slot.prepared;
        token.phase = SchedulerExecutionTokenPhase::WAIT_FANIN;
        token.next_fanin = fanin.count;
        token.predicate = slot.predicate;
        scheduler_local_ready_remove(local, index);
        if (local->direct_slot != nullptr && *local->direct_slot == index) *local->direct_slot = UINT32_MAX;
        scheduler_initialize_free_slot(&slot);
        local->set_timing_slot(index, -1);
        return true;
    }
    return false;
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

// A task-local ticket reserves one canonical execution index. Its owner keeps
// that reservation across owner-local slot restoration without reserving another block.
static inline __aicore__ PreparedAccessResult scheduler_prepare_dispatch_slot(
    const SchedulerGraphAccess &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint64_t profiling_level,
    uint8_t subtask_slot, uint64_t *prepare_start_cycles = nullptr
) {
    SchedulerScalarScope scalar_payload(scheduler, SchedPhaseKind::ScalarPayload, ready_claim.task_ref);
    if (scheduler == nullptr || !scheduler_task_ref_valid(graph, ready_claim.task_ref) ||
        scheduler->worker_id() >= scheduler->config.runtime_worker_count ||
        slot_claim.slot_index >= SCHEDULER_PENDING_SLOT_COUNT ||
        scheduler->config.self_lane >= PLATFORM_CORES_PER_BLOCKDIM)
        return PreparedAccessResult::ERROR;
    if (prepare_start_cycles != nullptr)
        *prepare_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    const LaneMask lanes = scheduler_task_lanes(metadata);
    if (subtask_slot == UINT8_MAX) subtask_slot = scheduler_metadata_single_subtask_slot(lanes);
    auto &slot = *scheduler->slot_state(slot_claim);
    if (slot.state != SchedulerDispatchSlotState::FILLING) return PreparedAccessResult::ERROR;
    const auto *target = scheduler_worker_context_at(scheduler_state_base, scheduler, scheduler->worker_id());
    if (subtask_slot >= PLATFORM_CORES_PER_BLOCKDIM || !lanes.has(static_cast<SubtaskSlot>(subtask_slot)) ||
        (target->core_type != static_cast<int32_t>(CoreType::AIC) &&
         target->core_type != static_cast<int32_t>(CoreType::AIV)) ||
        !scheduler_task_is_executable(metadata.flags) ||
        scheduler_metadata_core_type_index(subtask_slot) != scheduler->config.self_core_type) {
        scheduler_record_dispatch_error(
            run_control, ready_claim.task_ref, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, scheduler, &slot,
            metadata, SchedulerErrorSite::DISPATCH_INVALID_SHAPE
        );
        return PreparedAccessResult::ERROR;
    }
    slot.task_ref = ready_claim.task_ref;
    slot.subtask_slot = subtask_slot;
    const auto result = scheduler_acquire_execution_token(
        scheduler_state_base, scheduler->config.prepared, scheduler->worker_id(), ready_claim.task_ref, slot.block_idx,
        subtask_slot, &slot.prepared
    );
    if (result == PreparedAccessResult::ERROR) {
        uint64_t function_address = 0;
        const bool callable_valid = scheduler_lookup_callable_address(
            reinterpret_cast<__gm__ uint64_t *>(scheduler->config.callable_addresses_address),
            scheduler->config.callable_addresses_count, metadata.kernel_ids[subtask_slot], &function_address
        );
        scheduler_record_dispatch_error(
            run_control, ready_claim.task_ref,
            callable_valid ? SchedulerGraphResult::INVALID_ARGUMENTS : SchedulerGraphResult::INVALID_CALLABLE, &graph,
            scheduler, &slot, metadata,
            callable_valid ? SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED :
                             SchedulerErrorSite::DISPATCH_INVALID_CALLABLE
        );
    }
    if (result != PreparedAccessResult::READY) return result;
    if (scheduler_task_has_predicate(metadata.flags) && slot.predicate == SchedulerPredicateState::UNEVALUATED) {
        const auto predicate = scheduler_evaluate_task_predicate(graph, ready_claim.task_ref);
        if (predicate == SchedulerPredicateResult::MALFORMED) {
            scheduler_record_dispatch_error(
                run_control, ready_claim.task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler, &slot,
                metadata, SchedulerErrorSite::DISPATCH_INVALID_PREDICATE
            );
            return PreparedAccessResult::ERROR;
        }
        slot.predicate =
            predicate == SchedulerPredicateResult::PASS ? SchedulerPredicateState::PASS : SchedulerPredicateState::FAIL;
    }
    auto *binding =
        scheduler_execution_binding_at(scheduler_state_base, scheduler->config.prepared, slot.prepared.execution_index);
    binding->global_context.sub_block_id = scheduler_task_is_mix(metadata.flags) ?
                                               (scheduler->config.self_lane == 2 ? 1 : 0) :
                                               (subtask_slot == 2 ? 1 : 0);
    return PreparedAccessResult::READY;
}

static inline __aicore__ void scheduler_stage_dispatch_slot(
    __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, const SchedulerTaskMetadata &metadata, uint8_t subtask_slot
) {
    SchedulerLocalSlotState *local_slot = scheduler->slot_state(slot_claim);
    auto *binding = scheduler_execution_binding_at(
        scheduler_state_base, scheduler->config.prepared, local_slot->prepared.execution_index
    );
    local_slot->task_ref = ready_claim.task_ref;
    scheduler->set_timing_slot(slot_claim.slot_index, metadata.timing_slot);
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->subtask_slot = subtask_slot;
    SchedulerScalarScope scalar_cache(scheduler, SchedPhaseKind::ScalarCachePublish, ready_claim.task_ref);
    scheduler_writeback_cache_line(&binding->local_context);
    scheduler_writeback_cache_line(&binding->global_context);
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
    if (scheduler->direct_slot != nullptr && !scheduler_has_direct_slot(scheduler) && scheduler->local_ready_mask == 0)
        *scheduler->direct_slot = slot_claim.slot_index;
    else scheduler_local_ready_publish(scheduler, slot_claim.slot_index);
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
            if (ready_claim.source == SchedulerReadySource::COOPERATIVE) {
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
    if (scheduler_task_is_inline(metadata.flags)) return false;
    const uint8_t subtask_slot = scheduler_metadata_single_subtask_slot(scheduler_task_lanes(metadata));
    uint64_t prepare_start_cycles = 0;
    const auto prepared = scheduler_prepare_dispatch_slot(
        graph, scheduler_state_base, scheduler, run_control, slot_claim, ready_claim, metadata, profiling_level,
        subtask_slot, &prepare_start_cycles
    );
    if (prepared == PreparedAccessResult::ERROR) return false;
    scheduler_publish_dispatch_slot(
        scheduler_state_base, scheduler, slot_claim, ready_claim, metadata, profiling_level, subtask_slot,
        prepare_start_cycles
    );
    return true;
}
