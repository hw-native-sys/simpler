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

#include "scheduler_types.h"
#include "scheduler_memory.h"
#include "scheduler_graph.h"

enum class SchedulerRouteResult : uint64_t {
    READY = 0,
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
    uint64_t ready_to_kernel_cycles{0};
    uint64_t ready_to_kernel_max_cycles{0};
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

inline __aicore__ uint64_t scheduler_ready_pending_pack(int64_t head, int64_t tail) {
    const uint64_t packed_head = static_cast<uint32_t>(static_cast<int32_t>(head));
    const uint64_t packed_tail = static_cast<uint32_t>(static_cast<int32_t>(tail));
    return packed_head | (packed_tail << 32);
}

inline __aicore__ int64_t scheduler_ready_pending_head(uint64_t packed) {
    return static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(packed)));
}

inline __aicore__ int64_t scheduler_ready_pending_tail(uint64_t packed) {
    return static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(packed >> 32)));
}

inline __aicore__ bool scheduler_ready_pending_endpoint_fits(int64_t task_id) {
    return task_id >= SCHEDULER_INBOX_EMPTY && task_id <= INT32_MAX;
}

inline __aicore__ void scheduler_ready_owner_init(__gm__ SchedulerReadyOwnerState *owner_state) {
    if (owner_state == nullptr) return;
    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        scheduler_gm_store(owner_state->queues[type].pending_endpoints, SCHEDULER_READY_PENDING_EMPTY);
        scheduler_gm_store(owner_state->queues[type].advertised, UINT64_C(0));
    }
    scheduler_cache_barrier();
}

struct SchedulerReadyClaim {
    int64_t task_id{SCHEDULER_TASK_ID_INVALID};
    uint64_t inbox_index{UINT64_MAX};
    SchedulerReadySource source{SchedulerReadySource::LOCAL};
    uint64_t claim_start_cycles{0};
    uint64_t claim_end_cycles{0};
};

struct SchedulerFreeSlotClaim {
    uint64_t worker_id{UINT64_MAX};
    uint32_t slot_index{UINT32_MAX};
    uint32_t generation{0};
};

struct SchedulerCompletionServiceTiming {
    uint64_t scan_cycles{0};
    uint64_t consume_cycles{0};
    uint64_t resolve_cycles{0};
    uint64_t ready_publish_cycles{0};
    uint64_t refill_cycles{0};
    uint64_t finalize_cycles{0};
};

struct SchedulerDispatchFillTiming {
    uint64_t prepare_cycles{0};
    uint64_t materialize_cycles{0};
    uint64_t publish_cycles{0};
};

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

inline __aicore__ SchedulerPredicateResult
scheduler_evaluate_task_predicate(const SchedulerGraphView &graph, int64_t task_id) {
    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_id);
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

inline __aicore__ bool
scheduler_lookup_callable_address(__gm__ uint64_t *callable_addresses, uint16_t kernel_id, uint64_t *callable_address) {
    if (callable_addresses == nullptr || callable_address == nullptr || kernel_id >= SCHEDULER_CALLABLE_CAPACITY)
        return false;
    scheduler_observe_cache_line(&callable_addresses[kernel_id]);
    *callable_address = callable_addresses[kernel_id];
    return *callable_address != 0;
}

inline __aicore__ uint64_t
scheduler_completion_id(__gm__ const SchedulerWorkerContext *context, uint64_t local_completion_index) {
    return local_completion_index * context->runtime_worker_count + context->worker_index;
}

inline __aicore__ __gm__ SchedulerTaskControl *scheduler_task_control_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, int64_t task_id
) {
    return scheduler_state_at<SchedulerTaskControl>(
        scheduler_state_base,
        context->task_controls_offset + static_cast<uint64_t>(task_id) * sizeof(SchedulerTaskControl)
    );
}

inline __aicore__ __gm__ SchedulerTaskMetadata *scheduler_task_metadata_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, int64_t task_id
) {
    return scheduler_state_at<SchedulerTaskMetadata>(
        scheduler_state_base,
        context->task_metadata_offset + static_cast<uint64_t>(task_id) * sizeof(SchedulerTaskMetadata)
    );
}

inline __aicore__ __gm__ SchedulerCompletionInbox *scheduler_completion_inbox_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint64_t inbox_index
) {
    return scheduler_state_at<SchedulerCompletionInbox>(
        scheduler_state_base, context->completion_inboxes_offset + inbox_index * sizeof(SchedulerCompletionInbox)
    );
}

inline __aicore__ __gm__ SchedulerReadyInbox *scheduler_ready_inbox_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint32_t core_type_index,
    uint64_t inbox_index
) {
    uint64_t linear = static_cast<uint64_t>(core_type_index) * SCHEDULER_WORKER_CAPACITY + inbox_index;
    return scheduler_state_at<SchedulerReadyInbox>(
        scheduler_state_base, context->ready_inboxes_offset + linear * sizeof(SchedulerReadyInbox)
    );
}

inline __aicore__ __gm__ SchedulerReadyOwnerState *
scheduler_ready_owner_state_at(__gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context) {
    return scheduler_state_at<SchedulerReadyOwnerState>(
        scheduler_state_base,
        context->ready_owner_states_offset + context->inbox_index * sizeof(SchedulerReadyOwnerState)
    );
}

inline __aicore__ __gm__ SchedulerReadyDirectory *
scheduler_ready_directory_at(__gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context) {
    return scheduler_state_at<SchedulerReadyDirectory>(scheduler_state_base, context->ready_directory_offset);
}

inline __aicore__ __gm__ SchedulerGangCoordinator *
scheduler_gang_coordinator_at(__gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context) {
    return scheduler_state_at<SchedulerGangCoordinator>(scheduler_state_base, context->gang_coordinator_offset);
}

inline __aicore__ void scheduler_publish_gang_ready(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, __gm__ SchedulerTaskControl *control,
    uint8_t metadata_flags
) {
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::READY));
    __gm__ SchedulerGangCoordinator *coordinator = scheduler_gang_coordinator_at(scheduler_state_base, context);
    scheduler_gm_fetch_or(coordinator->ready_priority_bits, scheduler_task_priority_bit(metadata_flags));
}

inline __aicore__ __gm__ SchedulerWorkerContext *scheduler_worker_context_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint64_t worker_id
) {
    return scheduler_state_at<SchedulerWorkerContext>(
        scheduler_state_base, context->worker_contexts_offset + worker_id * sizeof(SchedulerWorkerContext)
    );
}

inline __aicore__ __gm__ SchedulerActivityBuffer *
scheduler_activity_buffer_at(__gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context) {
    if (context->activity_buffers_offset == 0 || context->is_scheduler == 0 ||
        context->scheduler_index >= SCHEDULER_CLUSTER_CAPACITY)
        return nullptr;
    return scheduler_state_at<SchedulerActivityBuffer>(
        scheduler_state_base,
        context->activity_buffers_offset + context->scheduler_index * sizeof(SchedulerActivityBuffer)
    );
}

inline __aicore__ void scheduler_append_idle_activity(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, uint64_t start_cycles,
    uint64_t end_cycles
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
    record->loop_iter = static_cast<uint32_t>(context->profiling_loop_iter);
    record->reserved = 0;
    scheduler_writeback_cache_line(record);
    scheduler_writeback_cache_line(&record->reserved);
    scheduler_cache_barrier();
    scheduler_gm_store(buffer->committed, committed + 1);
}

inline __aicore__ __gm__ SchedulerDispatchSlot *scheduler_dispatch_slot_at(
    __gm__ void *scheduler_state_base, __gm__ const SchedulerWorkerContext *context, uint64_t worker_id, uint32_t slot
) {
    return scheduler_state_at<SchedulerDispatchSlot>(
        scheduler_state_base,
        context->dispatch_slots_offset +
            (worker_id * SCHEDULER_PENDING_SLOT_COUNT + static_cast<uint64_t>(slot)) * sizeof(SchedulerDispatchSlot)
    );
}

inline __aicore__ uint64_t scheduler_dispatch_publication(uint32_t generation, SchedulerDispatchSlotState state) {
    return (static_cast<uint64_t>(generation) << 8) | static_cast<uint64_t>(state);
}

inline __aicore__ uint32_t scheduler_dispatch_generation(uint64_t publication) {
    return static_cast<uint32_t>(publication >> 8);
}

inline __aicore__ SchedulerDispatchSlotState scheduler_dispatch_state(uint64_t publication) {
    return static_cast<SchedulerDispatchSlotState>(publication & UINT64_C(0xff));
}

inline __aicore__ void scheduler_record_error(
    __gm__ SchedulerRunControl *run_control, int64_t task_id, SchedulerGraphResult status,
    const SchedulerGraphView *graph = nullptr, __gm__ const SchedulerWorkerContext *context = nullptr,
    SchedulerErrorSite error_site = SchedulerErrorSite::UNKNOWN
) {
    if (scheduler_gm_compare_exchange(run_control->error_claimed, UINT64_C(0), UINT64_C(1)) != 0) return;
    scheduler_gm_store(run_control->error_task_id, static_cast<uint64_t>(task_id));
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
    scheduler_gm_publish(run_control->scheduler_error, static_cast<uint64_t>(status));
}

inline __aicore__ void scheduler_publish_waiter_metadata(
    __gm__ SchedulerTaskControl *control, int64_t next_waiter, int32_t next_fanin_index, int32_t waiting_producer
) {
    control->next_waiter = next_waiter;
    control->next_fanin_index = next_fanin_index;
    control->waiting_producer = waiting_producer;
    scheduler_publish_cache_line(&control->next_waiter);
}

inline __aicore__ int64_t scheduler_observe_next_waiter(__gm__ SchedulerTaskControl *control) {
    scheduler_observe_cache_line(&control->next_waiter);
    return control->next_waiter;
}

inline __aicore__ SchedulerRouteResult scheduler_route_task(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_id, SchedulerWakeStats *stats,
    bool validate_current_state = true
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        scheduler_record_error(run_control, task_id, SchedulerGraphResult::INVALID_TASK_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_id);
    if (validate_current_state) {
        int64_t state = scheduler_gm_query(control->state);
        if (state == static_cast<int64_t>(SchedulerTaskState::DONE)) return SchedulerRouteResult::COMPLETED;
        if (state == static_cast<int64_t>(SchedulerTaskState::READY)) return SchedulerRouteResult::READY;
        if (state != static_cast<int64_t>(SchedulerTaskState::BLOCKED)) {
            scheduler_record_error(
                run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::ROUTE_INVALID_STATE
            );
            return SchedulerRouteResult::ERROR;
        }
    }

    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_id);
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
    int32_t next_fanin = control->next_fanin_index;
    if (fanin_count < 0 || next_fanin < 0 || next_fanin > fanin_count) {
        scheduler_record_error(run_control, task_id, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }

    while (next_fanin < fanin_count) {
        int32_t producer = scheduler_graph_fanin_id(graph, task_id, next_fanin);
        if (producer < 0 || producer >= task_id) {
            scheduler_record_error(run_control, task_id, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
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
            int64_t actual = scheduler_gm_compare_exchange(producer_control->wake_list_head, observed, task_id);
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
    scheduler_publish_waiter_metadata(
        control, SCHEDULER_TASK_ID_INVALID, fanin_count, static_cast<int32_t>(SCHEDULER_TASK_ID_INVALID)
    );
    return SchedulerRouteResult::READY_TO_ENQUEUE;
}

// No task can execute while the bootstrap barrier is closed. Executable
// producers therefore have open wake lists, and non-executable producers are
// the inline-completed tasks initialized by the host. The barrier makes it
// safe to publish the new head before publishing the waiter's link.
inline __aicore__ SchedulerRouteResult scheduler_bootstrap_route_task(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_id, SchedulerWakeStats *stats
) {
    if (task_id < 0 || static_cast<uint64_t>(task_id) >= graph.task_count) {
        scheduler_record_error(run_control, task_id, SchedulerGraphResult::INVALID_TASK_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }
    __gm__ uint8_t *payload = scheduler_graph_payload(graph, task_id);
    int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
    if (fanin_count < 0) {
        scheduler_record_error(run_control, task_id, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
        return SchedulerRouteResult::ERROR;
    }

    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_id);
    for (int32_t next_fanin = 0; next_fanin < fanin_count; ++next_fanin) {
        int32_t producer = scheduler_graph_fanin_id(graph, task_id, next_fanin);
        if (producer < 0 || producer >= task_id) {
            scheduler_record_error(run_control, task_id, SchedulerGraphResult::INVALID_FANIN_ID, &graph, context);
            return SchedulerRouteResult::ERROR;
        }
        __gm__ SchedulerTaskMetadata *producer_metadata =
            scheduler_task_metadata_at(scheduler_state_base, context, producer);
        scheduler_observe_cache_line(producer_metadata);
        if (!scheduler_task_is_executable(producer_metadata->flags)) continue;

        __gm__ SchedulerTaskControl *producer_control =
            scheduler_task_control_at(scheduler_state_base, context, producer);
        int64_t previous = scheduler_gm_exchange(producer_control->wake_list_head, task_id);
        if (previous < SCHEDULER_WAKE_LIST_OPEN) {
            scheduler_record_error(
                run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::BOOTSTRAP_WAKE_INVALID_HEAD
            );
            return SchedulerRouteResult::ERROR;
        }
        control->next_waiter = previous;
        control->next_fanin_index = next_fanin;
        control->waiting_producer = producer;
        scheduler_writeback_cache_line(&control->next_waiter);
        if (stats != nullptr) ++stats->wake_register_count;
        return SchedulerRouteResult::WAITING;
    }

    control->next_fanin_index = fanin_count;
    control->waiting_producer = static_cast<int32_t>(SCHEDULER_TASK_ID_INVALID);
    return SchedulerRouteResult::READY_TO_ENQUEUE;
}

inline __aicore__ bool scheduler_bootstrap_ready_batch_append(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, int64_t task_id,
    SchedulerReadyBatch *batch, SchedulerReadyStats *stats, uint64_t profiling_level = 0
) {
    if (batch == nullptr || task_id < 0 || static_cast<uint64_t>(task_id) >= context->graph_task_count) return false;
    // next_waiter belongs to exactly one wake or Ready chain while the task is live.
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_id);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::READY));
    control->next_waiter = SCHEDULER_INBOX_EMPTY;
    scheduler_writeback_cache_line(&control->next_waiter);
    if (batch->head == SCHEDULER_INBOX_EMPTY) {
        batch->head = task_id;
    } else {
        __gm__ SchedulerTaskControl *tail = scheduler_task_control_at(scheduler_state_base, context, batch->tail);
        tail->next_waiter = task_id;
        scheduler_writeback_cache_line(&tail->next_waiter);
    }
    batch->tail = task_id;
    if (scheduler_phase_timing_enabled(profiling_level)) {
        __gm__ SchedulerTaskTrace *cells =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, context->trace_cells_offset);
        cells[task_id].ready_transition_cycles = scheduler_cycles();
        scheduler_writeback_cache_line(&cells[task_id].ready_transition_cycles);
    }
    ++batch->count;
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ bool scheduler_bootstrap_ready_batch_publish(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, uint32_t core_type_index,
    uint64_t inbox_index, SchedulerReadyBatch *batch, SchedulerReadyStats *stats, uint64_t *ready_types
) {
    if (batch == nullptr || batch->head == SCHEDULER_INBOX_EMPTY) return true;
    if (core_type_index >= SCHEDULER_CORE_TYPE_COUNT || inbox_index >= SCHEDULER_CAPACITY || batch->tail < 0 ||
        ready_types == nullptr)
        return false;
    __gm__ SchedulerReadyInbox *inbox =
        scheduler_ready_inbox_at(scheduler_state_base, context, core_type_index, inbox_index);
    scheduler_cache_barrier();
    scheduler_gm_store(inbox->head, batch->head);
    *ready_types |= UINT64_C(1) << core_type_index;
    if (stats != nullptr) ++stats->batch_count;
    *batch = {};
    return true;
}

inline __aicore__ bool scheduler_bootstrap_ready_directory_publish(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, uint64_t scheduler_count
) {
    if (scheduler_count == 0 || scheduler_count > SCHEDULER_CAPACITY) return false;
    __gm__ SchedulerReadyDirectory *directory = scheduler_ready_directory_at(scheduler_state_base, context);
    for (uint64_t inbox_index = 0; inbox_index < scheduler_count; inbox_index += 8)
        scheduler_invalidate_cache_line(&directory->bootstrap_ready_types[inbox_index]);
    scheduler_cache_barrier();
    uint32_t shard_count = static_cast<uint32_t>(
        (scheduler_count + SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD - 1) / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD
    );
    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        for (uint32_t shard = 0; shard < shard_count; ++shard) {
            uint64_t bits = 0;
            uint64_t shard_begin = static_cast<uint64_t>(shard) * SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
            uint64_t shard_end = shard_begin + SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
            if (shard_end > scheduler_count) shard_end = scheduler_count;
            for (uint64_t inbox_index = shard_begin; inbox_index < shard_end; ++inbox_index) {
                uint64_t ready_types = directory->bootstrap_ready_types[inbox_index];
                if ((ready_types & (UINT64_C(1) << type)) != 0) bits |= UINT64_C(1) << (inbox_index - shard_begin);
            }
            directory->core_types[type][shard].bits = bits;
            scheduler_publish_cache_line(&directory->core_types[type][shard]);
        }
    }
    return true;
}

inline __aicore__ bool scheduler_ready_batch_append(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, int64_t task_id,
    SchedulerReadyBatch *batch, SchedulerReadyStats *stats, uint64_t profiling_level = 0
) {
    if (batch == nullptr || task_id < 0 || static_cast<uint64_t>(task_id) >= context->graph_task_count) return false;
    // next_waiter belongs to exactly one wake or Ready chain while the task is live.
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_id);
    scheduler_gm_store(control->state, static_cast<int64_t>(SchedulerTaskState::READY));
    control->next_waiter = SCHEDULER_INBOX_EMPTY;
    scheduler_publish_cache_line(&control->next_waiter);
    if (batch->head == SCHEDULER_INBOX_EMPTY) {
        batch->head = task_id;
    } else {
        __gm__ SchedulerTaskControl *tail = scheduler_task_control_at(scheduler_state_base, context, batch->tail);
        scheduler_observe_cache_line(&tail->next_waiter);
        tail->next_waiter = task_id;
        scheduler_publish_cache_line(&tail->next_waiter);
    }
    batch->tail = task_id;
    if (scheduler_phase_timing_enabled(profiling_level)) {
        __gm__ SchedulerTaskTrace *cells =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, context->trace_cells_offset);
        __gm__ SchedulerTaskTrace *trace = &cells[task_id];
        scheduler_observe_cache_line(&trace->ready_transition_cycles);
        trace->ready_transition_cycles = scheduler_cycles();
        scheduler_publish_cache_line(&trace->ready_transition_cycles);
    }
    ++batch->count;
    if (stats != nullptr) ++stats->enqueue_count;
    return true;
}

inline __aicore__ void scheduler_ready_directory_set(
    __gm__ SchedulerReadyDirectory *directory, uint32_t core_type_index, uint64_t inbox_index
) {
    uint64_t shard = inbox_index / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t bit = UINT64_C(1) << (inbox_index % SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD);
    scheduler_gm_fetch_or(directory->core_types[core_type_index][shard].bits, bit);
}

inline __aicore__ void scheduler_ready_directory_clear(
    __gm__ SchedulerReadyDirectory *directory, uint32_t core_type_index, uint64_t inbox_index
) {
    uint64_t shard = inbox_index / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t bit = UINT64_C(1) << (inbox_index % SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD);
    scheduler_gm_fetch_and(directory->core_types[core_type_index][shard].bits, ~bit);
}

inline __aicore__ uint64_t scheduler_ready_owner_pending_load(__gm__ SchedulerReadyOwnerQueue *owner_queue) {
    return scheduler_gm_query(owner_queue->pending_endpoints);
}

inline __aicore__ bool
scheduler_ready_owner_pending_store(__gm__ SchedulerReadyOwnerQueue *owner_queue, int64_t head, int64_t tail) {
    if (!scheduler_ready_pending_endpoint_fits(head) || !scheduler_ready_pending_endpoint_fits(tail)) return false;
    if ((head == SCHEDULER_INBOX_EMPTY) != (tail == SCHEDULER_INBOX_EMPTY)) return false;
    scheduler_gm_store(owner_queue->pending_endpoints, scheduler_ready_pending_pack(head, tail));
    scheduler_cache_barrier();
    return true;
}

inline __aicore__ void scheduler_ready_owner_pending_reset(__gm__ SchedulerReadyOwnerQueue *owner_queue) {
    scheduler_gm_store(owner_queue->pending_endpoints, SCHEDULER_READY_PENDING_EMPTY);
    scheduler_cache_barrier();
}

inline __aicore__ bool scheduler_ready_owner_pending_append(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerReadyOwnerQueue *owner_queue, SchedulerReadyBatch *source
) {
    // Only the owner mutates pending links; links reachable from a shared inbox head stay immutable.
    if (owner_queue == nullptr || source == nullptr || source->head == SCHEDULER_INBOX_EMPTY) return true;
    if (!scheduler_ready_pending_endpoint_fits(source->head) || source->head < 0 ||
        !scheduler_ready_pending_endpoint_fits(source->tail) || source->tail < 0)
        return false;
    const uint64_t pending = scheduler_ready_owner_pending_load(owner_queue);
    const int64_t pending_head = scheduler_ready_pending_head(pending);
    const int64_t pending_tail = scheduler_ready_pending_tail(pending);
    if (pending_head == SCHEDULER_INBOX_EMPTY) {
        if (pending_tail != SCHEDULER_INBOX_EMPTY ||
            !scheduler_ready_owner_pending_store(owner_queue, source->head, source->tail))
            return false;
        scheduler_ready_batch_reset(source);
        return true;
    }
    if (pending_head < 0 || pending_tail < 0) return false;
    __gm__ SchedulerTaskControl *tail = scheduler_task_control_at(scheduler_state_base, context, pending_tail);
    scheduler_observe_cache_line(&tail->next_waiter);
    tail->next_waiter = source->head;
    scheduler_publish_cache_line(&tail->next_waiter);
    if (!scheduler_ready_owner_pending_store(owner_queue, pending_head, source->tail)) return false;
    scheduler_ready_batch_reset(source);
    return true;
}

inline __aicore__ bool scheduler_ready_owner_maintain_type(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, uint32_t core_type_index,
    __gm__ SchedulerReadyOwnerState *owner_state
) {
    if (owner_state == nullptr || core_type_index >= SCHEDULER_CORE_TYPE_COUNT ||
        context->inbox_index >= SCHEDULER_CAPACITY)
        return false;
    __gm__ SchedulerReadyOwnerQueue *owner_queue = &owner_state->queues[core_type_index];
    __gm__ SchedulerReadyInbox *inbox =
        scheduler_ready_inbox_at(scheduler_state_base, context, core_type_index, context->inbox_index);
    __gm__ SchedulerReadyDirectory *directory = scheduler_ready_directory_at(scheduler_state_base, context);
    const int64_t head = scheduler_gm_query(inbox->head);
    if (head < SCHEDULER_INBOX_EMPTY) return false;
    if (head != SCHEDULER_INBOX_EMPTY) {
        if (scheduler_gm_query(owner_queue->advertised) == 0) {
            scheduler_ready_directory_set(directory, core_type_index, context->inbox_index);
            scheduler_gm_store(owner_queue->advertised, UINT64_C(1));
        }
        return true;
    }
    const uint64_t pending = scheduler_ready_owner_pending_load(owner_queue);
    const int64_t pending_head = scheduler_ready_pending_head(pending);
    const int64_t pending_tail = scheduler_ready_pending_tail(pending);
    if (pending_head != SCHEDULER_INBOX_EMPTY) {
        if (pending_head < 0 || pending_tail < 0) return false;
        scheduler_cache_barrier();
        scheduler_gm_store(inbox->head, pending_head);
        scheduler_cache_barrier();
        scheduler_ready_owner_pending_reset(owner_queue);
        if (scheduler_gm_query(owner_queue->advertised) == 0) {
            scheduler_ready_directory_set(directory, core_type_index, context->inbox_index);
            scheduler_gm_store(owner_queue->advertised, UINT64_C(1));
        }
        return true;
    } else if (pending_tail != SCHEDULER_INBOX_EMPTY) {
        return false;
    }
    if (scheduler_gm_query(owner_queue->advertised) != 0) {
        scheduler_ready_directory_clear(directory, core_type_index, context->inbox_index);
        scheduler_gm_store(owner_queue->advertised, UINT64_C(0));
    }
    return true;
}

inline __aicore__ bool scheduler_ready_owner_maintain(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerReadyOwnerState *owner_state
) {
    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        if (!scheduler_ready_owner_maintain_type(scheduler_state_base, context, type, owner_state)) return false;
    }
    return true;
}

inline __aicore__ bool scheduler_ready_batch_push(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, uint32_t core_type_index,
    uint64_t inbox_index, SchedulerReadyBatch *batch, SchedulerReadyStats *stats,
    __gm__ SchedulerReadyOwnerState *owner_state
) {
    if (batch == nullptr || owner_state == nullptr || core_type_index >= SCHEDULER_CORE_TYPE_COUNT ||
        inbox_index >= SCHEDULER_CAPACITY || inbox_index != context->inbox_index)
        return false;
    if (batch->head == SCHEDULER_INBOX_EMPTY) return true;
    if (batch->tail < 0) return false;
    __gm__ SchedulerReadyInbox *inbox =
        scheduler_ready_inbox_at(scheduler_state_base, context, core_type_index, inbox_index);
    __gm__ SchedulerReadyOwnerQueue *owner_queue = &owner_state->queues[core_type_index];
    const int64_t head = scheduler_gm_query(inbox->head);
    if (head < SCHEDULER_INBOX_EMPTY) return false;
    const uint64_t pending = scheduler_ready_owner_pending_load(owner_queue);
    const int64_t pending_head = scheduler_ready_pending_head(pending);
    const int64_t pending_tail = scheduler_ready_pending_tail(pending);
    if (head == SCHEDULER_INBOX_EMPTY && pending_head != SCHEDULER_INBOX_EMPTY) {
        if (pending_head < 0 || pending_tail < 0) return false;
        scheduler_cache_barrier();
        scheduler_gm_store(inbox->head, pending_head);
        scheduler_cache_barrier();
        scheduler_ready_owner_pending_reset(owner_queue);
        if (scheduler_gm_query(owner_queue->advertised) == 0) {
            scheduler_ready_directory_set(
                scheduler_ready_directory_at(scheduler_state_base, context), core_type_index, inbox_index
            );
            scheduler_gm_store(owner_queue->advertised, UINT64_C(1));
        }
    } else if (pending_head == SCHEDULER_INBOX_EMPTY && pending_tail != SCHEDULER_INBOX_EMPTY) {
        return false;
    }
    const int64_t published_head = scheduler_gm_query(inbox->head);
    if (published_head == SCHEDULER_INBOX_EMPTY) {
        scheduler_cache_barrier();
        scheduler_gm_store(inbox->head, batch->head);
        if (scheduler_gm_query(owner_queue->advertised) == 0) {
            scheduler_ready_directory_set(
                scheduler_ready_directory_at(scheduler_state_base, context), core_type_index, inbox_index
            );
            scheduler_gm_store(owner_queue->advertised, UINT64_C(1));
        }
        scheduler_ready_batch_reset(batch);
    } else {
        if (published_head < SCHEDULER_INBOX_EMPTY) return false;
        if (scheduler_gm_query(owner_queue->advertised) == 0) {
            scheduler_ready_directory_set(
                scheduler_ready_directory_at(scheduler_state_base, context), core_type_index, inbox_index
            );
            scheduler_gm_store(owner_queue->advertised, UINT64_C(1));
        }
        if (!scheduler_ready_owner_pending_append(scheduler_state_base, context, owner_queue, batch)) return false;
    }
    if (stats != nullptr) ++stats->batch_count;
    return true;
}

inline __aicore__ bool scheduler_ready_pop_from_inbox(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerRunControl *run_control, uint32_t core_type_index, uint64_t inbox_index, int64_t *task_id,
    SchedulerReadyStats *stats
) {
    if (task_id == nullptr) return false;
    *task_id = SCHEDULER_TASK_ID_INVALID;
    __gm__ SchedulerReadyInbox *inbox =
        scheduler_ready_inbox_at(scheduler_state_base, context, core_type_index, inbox_index);
    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        int64_t head = scheduler_gm_query(inbox->head);
        if (head == SCHEDULER_INBOX_EMPTY) return true;
        if (head < 0 || static_cast<uint64_t>(head) >= graph.task_count) {
            scheduler_record_error(
                run_control, head, SchedulerGraphResult::INVALID_TASK_ID, &graph, context,
                SchedulerErrorSite::READY_POP_INVALID_HEAD
            );
            return false;
        }
        __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, head);
        int64_t next = scheduler_observe_next_waiter(control);
        if (next < SCHEDULER_INBOX_EMPTY) {
            scheduler_record_error(
                run_control, head, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::READY_POP_INVALID_LINK
            );
            return false;
        }
        int64_t actual = scheduler_gm_compare_exchange(inbox->head, head, next);
        if (actual != head) {
            if (stats != nullptr) ++stats->cas_retry_count;
            continue;
        }
        if (stats != nullptr) ++stats->pop_count;
        *task_id = head;
        return true;
    }
    // Bounded contention may defer a non-empty inbox to a later scheduling iteration.
    if (stats != nullptr) ++stats->contention_giveup_count;
    return true;
}

inline __aicore__ uint64_t scheduler_load_ready_directory_shard(
    __gm__ SchedulerReadyDirectory *directory, uint64_t scheduler_count, uint32_t core_type_index, uint64_t inbox_index
) {
    uint64_t shard = inbox_index / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t shard_begin = shard * SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t shard_end = shard_begin + SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    if (shard_end > scheduler_count) shard_end = scheduler_count;
    uint64_t valid_bits = shard_end > shard_begin ? (UINT64_C(1) << (shard_end - shard_begin)) - 1 : 0;
    return scheduler_gm_query(directory->core_types[core_type_index][shard].bits) & valid_bits;
}

inline __aicore__ bool scheduler_steal_ready_from_shard(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerRunControl *run_control, uint32_t core_type_index, uint64_t shard_begin, uint64_t shard_end,
    uint64_t start, uint64_t bits, SchedulerReadyStats *stats, SchedulerReadyClaim *claim, uint64_t profiling_level
) {
    int64_t task_id = SCHEDULER_TASK_ID_INVALID;
    bits &= ~(UINT64_C(1) << (context->inbox_index - shard_begin));
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
                    graph, scheduler_state_base, context, run_control, core_type_index, victim, &task_id, stats
                ))
                return false;
            if (task_id >= 0) {
                claim->task_id = task_id;
                claim->inbox_index = victim;
                claim->source = SchedulerReadySource::STOLEN;
                claim->claim_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
                if (stats != nullptr) ++stats->steal_count;
                return true;
            }
        }
    }
    return true;
}

inline __aicore__ bool scheduler_claim_ready_for_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerRunControl *run_control, uint64_t scheduler_count, uint32_t core_type_index,
    uint64_t *victim_cursor, SchedulerReadyStats *stats, SchedulerReadyClaim *claim,
    __gm__ SchedulerReadyOwnerState *owner_state, uint64_t profiling_level = 0
) {
    if (victim_cursor == nullptr || claim == nullptr || owner_state == nullptr || scheduler_count == 0 ||
        scheduler_count > SCHEDULER_CAPACITY || context->inbox_index >= scheduler_count ||
        core_type_index >= SCHEDULER_CORE_TYPE_COUNT)
        return false;
    *claim = {};
    claim->claim_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    if (!scheduler_ready_owner_maintain_type(scheduler_state_base, context, core_type_index, owner_state)) return false;
    int64_t task_id = SCHEDULER_TASK_ID_INVALID;
    if (!scheduler_ready_pop_from_inbox(
            graph, scheduler_state_base, context, run_control, core_type_index, context->inbox_index, &task_id, stats
        ))
        return false;
    if (task_id >= 0) {
        claim->task_id = task_id;
        claim->inbox_index = context->inbox_index;
        claim->claim_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        return true;
    }

    __gm__ SchedulerReadyDirectory *directory = scheduler_ready_directory_at(scheduler_state_base, context);
    uint64_t shard_begin =
        context->inbox_index / SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD * SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    uint64_t shard_end = shard_begin + SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD;
    if (shard_end > scheduler_count) shard_end = scheduler_count;
    uint64_t start = *victim_cursor;
    if (start < shard_begin || start >= shard_end) start = shard_begin;
    uint64_t bits =
        scheduler_load_ready_directory_shard(directory, scheduler_count, core_type_index, context->inbox_index);
    if (bits != 0 && !scheduler_steal_ready_from_shard(
                         graph, scheduler_state_base, context, run_control, core_type_index, shard_begin, shard_end,
                         start, bits, stats, claim, profiling_level
                     ))
        return false;
    const uint64_t cursor_base = claim->task_id >= 0 ? claim->inbox_index : start;
    *victim_cursor = cursor_base + 1 == shard_end ? shard_begin : cursor_base + 1;
    if (claim->task_id >= 0) return true;
    claim->claim_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    return true;
}

inline __aicore__ bool scheduler_ready_directory_nonempty(
    __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context, uint64_t scheduler_count,
    uint32_t core_type_index
) {
    if (scheduler_count == 0 || scheduler_count > SCHEDULER_CAPACITY || context->inbox_index >= scheduler_count ||
        core_type_index >= SCHEDULER_CORE_TYPE_COUNT)
        return false;
    __gm__ SchedulerReadyDirectory *directory = scheduler_ready_directory_at(scheduler_state_base, context);
    return scheduler_load_ready_directory_shard(directory, scheduler_count, core_type_index, context->inbox_index) != 0;
}

inline __aicore__ void scheduler_initialize_free_slot(__gm__ SchedulerDispatchSlot *slot) {
    uint32_t generation = slot->generation + 1;
    if (generation == 0) generation = 1;
    slot->task_id = SCHEDULER_TASK_ID_INVALID;
    slot->generation = generation;
    scheduler_publish_cache_line(slot);
    scheduler_gm_publish(
        slot->publication, scheduler_dispatch_publication(generation, SchedulerDispatchSlotState::FREE)
    );
}

inline __aicore__ bool scheduler_fill_dispatch_slot(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerFreeSlotClaim &slot_claim,
    const SchedulerReadyClaim &ready_claim, uint64_t profiling_level = 0, SchedulerDispatchFillTiming *timing = nullptr
) {
    if (ready_claim.task_id < 0 || static_cast<uint64_t>(ready_claim.task_id) >= graph.task_count ||
        slot_claim.worker_id >= scheduler->runtime_worker_count ||
        slot_claim.slot_index >= SCHEDULER_PENDING_SLOT_COUNT)
        return false;
    const bool record_timeline = timing != nullptr;
    const bool task_timing_enabled = scheduler_task_timing_enabled(profiling_level);
    const bool schedule_timing_enabled = scheduler_schedule_timing_enabled(profiling_level);
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t dispatch_start_cycles = phase_timing_enabled ? scheduler_cycles() : 0;
    uint64_t operation_start = record_timeline ? scheduler_cycles() : 0;
    __gm__ SchedulerTaskMetadata *metadata_source =
        scheduler_task_metadata_at(scheduler_state_base, scheduler, ready_claim.task_id);
    scheduler_observe_cache_line(metadata_source);
    SchedulerTaskMetadata metadata{};
    metadata.kernel_ids[0] = metadata_source->kernel_ids[0];
    metadata.kernel_ids[1] = metadata_source->kernel_ids[1];
    metadata.kernel_ids[2] = metadata_source->kernel_ids[2];
    metadata.active_mask = metadata_source->active_mask;
    metadata.flags = metadata_source->flags;
    metadata.logical_block_num = metadata_source->logical_block_num;
    metadata.total_required_subtasks = metadata_source->total_required_subtasks;
    metadata.timing_slot = metadata_source->timing_slot;
    const uint8_t subtask_slot = scheduler_metadata_single_subtask_slot(metadata.active_mask);
    __gm__ SchedulerWorkerContext *target =
        scheduler_worker_context_at(scheduler_state_base, scheduler, slot_claim.worker_id);
    scheduler_observe_cache_line(target);
    if (subtask_slot == UINT8_MAX ||
        (target->core_type != static_cast<int32_t>(CoreType::AIC) &&
         target->core_type != static_cast<int32_t>(CoreType::AIV)) ||
        !scheduler_task_is_executable(metadata.flags) || scheduler_task_is_gang(metadata.flags) ||
        scheduler_metadata_core_type_index(subtask_slot) != scheduler_core_type_index(target->core_type)) {
        scheduler_record_error(
            run_control, ready_claim.task_id, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, scheduler,
            SchedulerErrorSite::DISPATCH_INVALID_SHAPE
        );
        return false;
    }
    const uint16_t kernel_id = metadata.kernel_ids[subtask_slot];
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, scheduler, slot_claim.worker_id, slot_claim.slot_index);
    uint32_t generation = slot_claim.generation + 1;
    if (generation == 0) generation = 1;
    __gm__ uint64_t *callable_addresses =
        scheduler_state_at<uint64_t>(scheduler_state_base, scheduler->callable_addresses_offset);
    const bool inline_task = scheduler_task_is_inline(metadata.flags);
    uint64_t callable_address = UINT64_C(1);
    if (!inline_task && !scheduler_lookup_callable_address(callable_addresses, kernel_id, &callable_address)) {
        scheduler_record_error(
            run_control, ready_claim.task_id, SchedulerGraphResult::INVALID_CALLABLE, &graph, scheduler,
            SchedulerErrorSite::DISPATCH_INVALID_CALLABLE
        );
        return false;
    }

    slot->task_id = ready_claim.task_id;
    slot->ready_inbox_index = ready_claim.inbox_index;
    slot->claim_start_cycles = ready_claim.claim_start_cycles;
    slot->claim_end_cycles = ready_claim.claim_end_cycles;
    slot->claim_worker_id = scheduler->worker_index;
    slot->kernel_id = kernel_id;
    slot->subtask_slot = subtask_slot;
    slot->has_fanin = scheduler_task_has_fanin(metadata.flags) ? 1 : 0;
    slot->ready_source = static_cast<uint8_t>(ready_claim.source);
    slot->pending_slot = static_cast<uint8_t>(slot_claim.slot_index);
    slot->generation = generation;
    slot->block_idx = 0;
    slot->block_num = 1;
    slot->cohort_generation = 0;
    slot->cohort_index = UINT8_MAX;
    slot->gang = 0;
    scheduler_writeback_cache_line(slot);

    const uint64_t dispatch_payload_offset =
        target->dispatch_payload_offset + static_cast<uint64_t>(slot_claim.slot_index) * sizeof(DispatchPayload);

    uint64_t operation_end = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) timing->prepare_cycles += operation_end - operation_start;

    SchedulerTaskInfo task{
        ready_claim.task_id,
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
        status = scheduler_materialize_task_payload_resolved(graph, task, callable_address, payload);
        if (status == SchedulerGraphResult::OK && scheduler_task_has_predicate(metadata.flags)) {
            const SchedulerPredicateResult predicate = scheduler_evaluate_task_predicate(graph, ready_claim.task_id);
            if (predicate == SchedulerPredicateResult::MALFORMED) {
                scheduler_record_error(
                    run_control, ready_claim.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::DISPATCH_INVALID_PREDICATE
                );
                return false;
            }
            if (predicate == SchedulerPredicateResult::FAIL) payload->function_bin_addr = 0;
        }
    }
    if (status != SchedulerGraphResult::OK) {
        scheduler_record_error(
            run_control, ready_claim.task_id, status, &graph, scheduler, SchedulerErrorSite::DISPATCH_MATERIALIZE_FAILED
        );
        return false;
    }
    uint64_t materialize_end = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) timing->materialize_cycles += materialize_end - operation_end;
    scheduler_publish_dispatch_payload(payload);
    __gm__ SchedulerTaskControl *control =
        scheduler_task_control_at(scheduler_state_base, scheduler, ready_claim.task_id);
    if (phase_timing_enabled) {
        scheduler_observe_cache_line(&control->next_waiter);
        control->ready_publish_cycles = scheduler_cycles();
        scheduler_publish_cache_line(&control->next_waiter);
    }
    uint64_t publish_end = record_timeline ? scheduler_cycles() : 0;
    if (timing != nullptr) timing->publish_cycles += publish_end - materialize_end;
    if (task_timing_enabled) {
        __gm__ SchedulerTaskTrace *traces =
            scheduler_state_at<SchedulerTaskTrace>(scheduler_state_base, scheduler->trace_cells_offset);
        __gm__ SchedulerTaskTrace *trace = &traces[ready_claim.task_id];
        trace->worker_id = slot_claim.worker_id;
        trace->task_id = static_cast<uint64_t>(ready_claim.task_id);
        if (phase_timing_enabled) {
            trace->ready_source = static_cast<uint64_t>(ready_claim.source);
            trace->claim_worker_id = scheduler->worker_index;
            trace->claim_start_cycles = ready_claim.claim_start_cycles;
            trace->claim_end_cycles = ready_claim.claim_end_cycles;
            trace->claim_loop_iter = scheduler->profiling_loop_iter;
            trace->dispatch_start_cycles = dispatch_start_cycles;
            trace->dispatch_scheduler_worker_id = scheduler->worker_index;
            trace->dispatch_loop_iter = scheduler->profiling_loop_iter;
        }
        if (schedule_timing_enabled) trace->dispatch_end_cycles = scheduler_cycles();
        scheduler_publish_cache_line(trace);
        if (schedule_timing_enabled) scheduler_publish_cache_line(&trace->dispatch_start_cycles);
    }
    scheduler_gm_publish(
        slot->publication, scheduler_dispatch_publication(generation, SchedulerDispatchSlotState::READY)
    );
    return true;
}

inline __aicore__ bool scheduler_resolve_completion(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *context,
    __gm__ SchedulerRunControl *run_control, int64_t task_id, SchedulerWakeStats *wake_stats,
    SchedulerReadyStats *ready_stats, SchedulerCompletionStats *completion_stats,
    __gm__ SchedulerReadyOwnerState *owner_state, uint64_t profiling_level = 0, bool validate_done_state = true,
    uint64_t *ready_publish_cycles = nullptr
) {
    if (owner_state == nullptr) return false;
    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(scheduler_state_base, context, task_id);
    if (validate_done_state && scheduler_gm_query(control->state) != static_cast<int64_t>(SchedulerTaskState::DONE)) {
        scheduler_record_error(
            run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
            SchedulerErrorSite::COMPLETION_TASK_NOT_DONE
        );
        return false;
    }
    const bool phase_timing_enabled = scheduler_phase_timing_enabled(profiling_level);
    uint64_t resolve_start = phase_timing_enabled ? scheduler_cycles() : 0;
    if (phase_timing_enabled) {
        scheduler_observe_cache_line(&control->next_waiter);
        control->completion_resolve_start_cycles = resolve_start;
        control->scheduler_worker_id = context->worker_index;
        control->completion_resolve_loop_iter = context->profiling_loop_iter;
    }
    int64_t waiter = scheduler_gm_exchange(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    if (waiter == SCHEDULER_WAKE_LIST_CLOSED) {
        scheduler_record_error(
            run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
            SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED
        );
        return false;
    }
    if (wake_stats != nullptr) ++wake_stats->wake_close_count;
    SchedulerReadyBatch batches[SCHEDULER_CORE_TYPE_COUNT]{};
    while (waiter >= 0) {
        if (static_cast<uint64_t>(waiter) >= graph.task_count) {
            scheduler_record_error(
                run_control, task_id, SchedulerGraphResult::INVALID_TASK_ID, &graph, context,
                SchedulerErrorSite::COMPLETION_INVALID_WAITER
            );
            return false;
        }
        __gm__ SchedulerTaskControl *waiter_control = scheduler_task_control_at(scheduler_state_base, context, waiter);
        int64_t next = scheduler_observe_next_waiter(waiter_control);
        if (wake_stats != nullptr) ++wake_stats->wake_migrate_count;
        SchedulerRouteResult route =
            scheduler_route_task(graph, scheduler_state_base, context, run_control, waiter, wake_stats);
        // A fatal scheduler error makes any READY transition in this local batch terminal;
        // an unpublished partial batch is never reused as successful scheduler state.
        if (route == SchedulerRouteResult::ERROR) return false;
        if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
            __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(scheduler_state_base, context, waiter);
            scheduler_observe_cache_line(metadata);
            if (!scheduler_task_is_executable(metadata->flags)) {
                scheduler_record_error(
                    run_control, waiter, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                    SchedulerErrorSite::COMPLETION_WAITER_NOT_EXECUTABLE
                );
                return false;
            }
            if (scheduler_task_is_gang(metadata->flags)) {
                scheduler_publish_gang_ready(scheduler_state_base, context, waiter_control, metadata->flags);
            } else {
                const uint8_t subtask_slot = scheduler_metadata_single_subtask_slot(metadata->active_mask);
                if (subtask_slot == UINT8_MAX) {
                    scheduler_record_error(
                        run_control, waiter, SchedulerGraphResult::UNSUPPORTED_SHAPE, &graph, context,
                        SchedulerErrorSite::COMPLETION_INVALID_SHAPE
                    );
                    return false;
                }
                if (!scheduler_ready_batch_append(
                        scheduler_state_base, context, waiter,
                        &batches[scheduler_metadata_core_type_index(subtask_slot)], ready_stats, profiling_level
                    )) {
                    scheduler_record_error(
                        run_control, waiter, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                        SchedulerErrorSite::COMPLETION_READY_APPEND_FAILED
                    );
                    return false;
                }
            }
        }
        waiter = next;
    }
    uint64_t ready_publish_start = ready_publish_cycles == nullptr ? 0 : scheduler_cycles();
    for (uint32_t type = 0; type < SCHEDULER_CORE_TYPE_COUNT; ++type) {
        if (!scheduler_ready_batch_push(
                scheduler_state_base, context, type, context->inbox_index, &batches[type], ready_stats, owner_state
            )) {
            scheduler_record_error(
                run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, context,
                SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED
            );
            return false;
        }
    }
    if (ready_publish_cycles != nullptr) *ready_publish_cycles += scheduler_cycles() - ready_publish_start;
    if (phase_timing_enabled) {
        control->completion_resolve_end_cycles = scheduler_cycles();
        scheduler_publish_cache_line(&control->next_waiter);
    }
    if (completion_stats != nullptr) ++completion_stats->resolve_count;
    return true;
}
