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
 * The percounter AICore executor: every lane schedules itself.
 *
 * A lane draws one position at a time from its core type's ticket and keeps
 * unready positions and their prepared GM payloads in a bounded local window.
 * MIX preparation is announced before readiness without gathering the peers.
 * It runs a ready kernel and
 * publishes by adding to its own counter. There is no scheduler lane, no ready
 * queue and no AICPU dispatch; the AICPU only watches the progress count.
 *
 * Startup and retirement are the runtime's: the lane reports through the
 * handshake, waits for the AICPU's READY and the DATA_MAIN_BASE release, and
 * after EXIT answers the COND nonce exactly as the resident executor does.
 *
 * Header-only because aicore_executor.cpp is compiled by name into unit tests.
 */

#pragma once

#include "aicore/aicore.h"
#include "common/platform_config.h"
#include "dispatch_payload.h"
#include "host_build_graph/runtime_status.h"
#include "runtime.h"
#include "scheduler/wire_types.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/image_read.h"
#include "scheduler/gm.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_watchdog.h"
#include "aicore/aicore_profiling_state.h"
#include "aicore/chip_swimlane_collector_aicore.h"

namespace simpler::percounter {

typedef void (*PercounterKernelFunc)(__gm__ int64_t *);

/**
 * Read a counter the way the runtime reads any cross-core word.
 *
 * Not a volatile dereference: on silicon a volatile load never observed a
 * remote atomic across two million polls with the producer confirmed to have
 * published. `ld_dev` reaches a coherent point (~98 cycles at realistic
 * fanout). Not `AtomicAdd(ptr, 0)` either: that is a read-modify-write and
 * would queue every poll at the atomic unit.
 */
inline __aicore__ uint64_t percounter_observe(uint64_t counter_address) {
    __gm__ uint64_t *word = reinterpret_cast<__gm__ uint64_t *>(counter_address);
    return percounter_gm_read(*word);
}

/**
 * How many finished tasks a lane accumulates before folding them into the
 * shared progress count.
 *
 * Not per task: every lane adding to one address would queue at the atomic
 * unit. Not once at the end either: the AICPU progress watchdog would see no
 * progress on a healthy run. It must stay below the tasks a lane runs, or the
 * fold never fires before the end; 8 clears `mixed_fanin32_1024`, the smallest
 * measured shape at about 9 tasks per lane.
 */
inline constexpr uint64_t PERCOUNTER_COMPLETION_BATCH = 8;

/** How often a spinning lane looks at the exit register and the error word. */
inline constexpr uint32_t PERCOUNTER_SPIN_CHECK_INTERVAL = 64;

/**
 * Consecutive missed polls at full rate before a lane spaces them out, and how
 * long it then spins on the cycle counter between polls.
 *
 * `SPIN_WAIT_HINT()` is a no-op on this hardware, so without this every lane
 * waiting on a wide-fanout counter issues back-to-back loads to one line, and
 * the producer's publishing atomic has to win that line against all of them.
 * The pause reads the cycle counter, not the contended line, and is a bounded
 * spin, not a sleep.
 *
 * `AFTER` only decides when spacing starts (16 and 32 measured the same).
 * `CYCLES` is the duty cycle while spaced: 1024 -> 256 took `mixed_chain_1024`
 * from 3182 to 3029 us, while `mixed_fanin32_1024` (52 lanes on one counter)
 * stayed at 295 us. Both were measured on one a5 part; re-validating them
 * elsewhere needs the wide-fanout shape, the only one that shows the failure.
 */
inline constexpr uint64_t PERCOUNTER_SPIN_BACKOFF_AFTER = 16;
inline constexpr uint64_t PERCOUNTER_SPIN_BACKOFF_CYCLES = 256;

struct LaneConfig {
    __gm__ ChipSwimlaneActiveHead *profile_head{nullptr};
    ChipSwimlaneAicoreLocalState *profile{nullptr};
    __gm__ PercounterControl *control;
    SchedulerGraphView graph;
    __gm__ uint64_t *callable_addresses;
    uint32_t callable_count;
    __gm__ DispatchPayload *payloads;
    // This type's tickets, `ticket_group_count` of them; this lane draws from
    // `ticket_group`'s first (see PERCOUNTER_TICKET_CAPACITY).
    __gm__ PercounterTicket *tickets;
    uint32_t ticket_group;
    uint32_t ticket_group_count;
    __gm__ uint64_t *order;
    // Resolved once: the host writes these offsets at bind and never changes
    // them, so re-deriving them per probe would spend device loads on constants.
    __gm__ PercounterFanin *fanin;
    __gm__ uint64_t *fanin_addresses;
    __gm__ PercounterCounter *counters;
    uint64_t task_count;
    // Ids below this are task-table tasks; the rest are Graph body tasks.
    uint64_t table_task_count;
    uint64_t order_count;
    // Whether any task carries a dispatch predicate; a graph without one never
    // reads the predicate words.
    bool has_predicates;

    // This lane's deferred-completion record, which its payload's GlobalContext
    // points a kernel at. The record exists so that a kernel registering a
    // condition leaves a trace: until this runtime honours such conditions, a
    // non-empty record fails the run rather than letting the task publish with
    // its data still in flight. One per lane, as the payload is.
    __gm__ pto2::resident::AsyncGroupData *async_group;

    // This lane's place in its cluster: lane 0 is the AIC, 1 and 2 the AIVs,
    // which is also the subtask slot it runs of a MIX task.
    int32_t cluster_index;
    int32_t cluster_lane;
    // The MIX list. `mix_order_count == 0` means the graph has no MIX task,
    // and the four pointers below are then null.
    uint64_t mix_order_count;
    uint64_t mix_rendezvous_count;
    __gm__ uint64_t *mix_order;
    __gm__ PercounterTicket *mix_tickets;
    __gm__ uint64_t *mix_signal_ready;
    PERCOUNTER_SSBUF PercounterMixSignal *mix_signal;
    // The sync-start cohort queue. `sync_cohort_count == 0` means the graph has
    // no sync-start task, and the two pointers are then null.
    uint64_t sync_cohort_count;
    uint64_t sync_block_total;
    __gm__ PercounterSyncQueue *sync_queue;
    __gm__ uint64_t *sync_pending;
};

/**
 * What this lane knows about the cohort queue.
 *
 * `my_block < 0` is both "holds nothing" and "nothing to serve", the common
 * state. The payload is copied when the block is taken, because the open slot
 * it came from is recycled the moment the cohort is fully claimed.
 */
struct PercounterSyncState {
    // The open cohort this lane has already decided about. The open slot's
    // generation rises by two per cohort, so it names each one, and checking it
    // costs no load beyond the generation itself.
    uint64_t decided_generation;
    uint64_t my_seq;
    int32_t my_block;
    int64_t task_id;
    uint32_t block_num;
    uint32_t active_mask;
    uint64_t taken_cycles;
};

/**
 * The MIX position this cluster has drawn but not announced yet; `task_id` is
 * -1 when it holds none. Only the claimer lane ever holds one.
 */
struct PercounterMixClaim {
    int64_t task_id;
    // A MIX task with `block_num > 1` is one rendezvous per block, so a position
    // names a block and the claim carries it to the arming.
    int32_t block_idx;
    int32_t block_num;
    // Set once a draw comes back past the end of the list, so a cluster with
    // nothing left to claim stops adding to the shared draw point.
    bool list_spent;
    // A position drawn whose entry the AICPU has not published yet, or -1. The
    // claim holds it, as it holds an unready task, until it can read it.
    int64_t position;
    bool prepare_announced{false};
};

/** What a lane carries between its loop and the waits it makes from it. */
struct PercounterLaneState {
    // Finished blocks not yet folded into `completed_block_count`.
    uint64_t local_done;
    // The last MIX generation this lane has looked at, served or not.
    uint32_t last_served;
    PercounterMixClaim claim;
    PercounterSyncState sync;
    uint32_t prepared_mix_generation{0};
    int64_t prepared_mix_task{-1};
    int32_t prepared_mix_block{-1};
};

struct PercounterPending {
    uint64_t position;
    uint32_t payload_slot;
    bool prepared;
};

/** Where a task's producer counters are: `addresses[0 .. count)`. */
struct PercounterFaninView {
    __gm__ uint64_t *addresses;
    int32_t count;
};

/**
 * Locate `task_id`'s producers.
 *
 * A device load, not a dereference. The table reaches the device by DMA, which
 * does not invalidate this core's scalar data cache, so a plain load can return
 * whatever the line held before the image landed -- on silicon a 64-task chain
 * read a producer address of 0x120000000000 for its 60th task. Invisible in
 * simulation, where every load is coherent.
 */
inline __aicore__ PercounterFaninView percounter_fanin_view(const LaneConfig &cfg, int64_t task_id) {
    const uint64_t record = percounter_gm_read_u32_pair(reinterpret_cast<__gm__ uint32_t *>(&cfg.fanin[task_id]));
    PercounterFaninView view{};
    view.count = static_cast<int32_t>(record & 0xffffffffULL);
    view.addresses = cfg.fanin_addresses + static_cast<int32_t>(record >> 32);
    return view;
}

inline __aicore__ void percounter_fail(const LaneConfig &cfg, SchedulerGraphResult result) {
    // First error wins.
    percounter_gm_compare_exchange(cfg.control->lane_error, uint64_t{0}, static_cast<uint64_t>(result));
}

/**
 * A task as a lane runs it: where its storage is, what it waits on, and what
 * its completion publishes.
 *
 * A task-table task is its own storage index and has a fanin row of counter
 * addresses. A Graph body task lives in its invocation's storage at its body
 * index; a root waits on its shell's fanin row, any other on its producers'
 * body indices, and a sink also publishes its shell.
 */
struct PercounterTask {
    int64_t id;     // percounter id: the counter it publishes
    int64_t local;  // index into `graph`
    SchedulerGraphView graph;
    int64_t shell;  // -1 for a task-table task
    bool sink;
    // `fanin` holds producer body indices, to add to `body_base`; otherwise
    // producer counter addresses.
    bool body_fanin;
    int32_t fanin_count;
    __gm__ uint64_t *fanin;
    int64_t body_base;
};

/**
 * Resolve `id` into a task. False, with the lane error set, for an id outside
 * the graph or a body task whose line the AICPU has not written.
 *
 * A body task's line is read once its entry is published, and the AICPU wrote
 * and flushed it before publishing; the invalidate drops whatever this core
 * held of it from an earlier run. Its counter word changes under other cores'
 * atomics, but nothing here reads it.
 */
inline __aicore__ bool percounter_resolve(const LaneConfig &cfg, int64_t id, PercounterTask *task) {
    if (id < 0 || static_cast<uint64_t>(id) >= cfg.task_count) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_ID);
        return false;
    }
    if (static_cast<uint64_t>(id) < cfg.table_task_count) {
        const PercounterFaninView view = percounter_fanin_view(cfg, id);
        *task = PercounterTask{id, id, cfg.graph, -1, false, false, view.count, view.addresses, 0};
        return true;
    }
    __gm__ PercounterCounter *line = &cfg.counters[id];
    percounter_observe_cache_line(line);
    const uint64_t storage = line->storage_address;
    const uint32_t body_index = line->body_index;
    const uint32_t flags = line->flags;
    const uint32_t shell = line->shell_task_id;
    if (storage == 0 || static_cast<uint64_t>(shell) >= cfg.table_task_count || static_cast<int64_t>(body_index) > id) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_ID);
        return false;
    }
    task->id = id;
    task->local = body_index;
    task->graph = scheduler_ordinary_graph_view(storage, body_index + 1, body_index);
    task->shell = shell;
    task->sink = (flags & PERCOUNTER_BODY_SINK) != 0;
    task->body_base = id - static_cast<int64_t>(body_index);
    if ((flags & PERCOUNTER_BODY_ROOT) != 0) {
        const PercounterFaninView view = percounter_fanin_view(cfg, shell);
        task->body_fanin = false;
        task->fanin_count = view.count;
        task->fanin = view.addresses;
    } else {
        task->body_fanin = true;
        task->fanin_count = static_cast<int32_t>(line->fanin_count);
        task->fanin = scheduler_state_at<uint64_t>(cfg.control, line->fanin_offset);
    }
    return true;
}

/** The counter address of `task`'s producer `i`. */
inline __aicore__ uint64_t percounter_producer(const LaneConfig &cfg, const PercounterTask &task, int32_t i) {
    const uint64_t value = percounter_gm_read(task.fanin[i]);
    if (!task.body_fanin) return value;
    return reinterpret_cast<uint64_t>(&cfg.counters[static_cast<uint64_t>(task.body_base) + value].value);
}

/**
 * One pass over `task`'s producers: true when every one has published.
 *
 * Never blocks: a MIX claimer keeps an unready position while its cluster
 * serves other work. The probe stops at the first unpublished producer.
 */
inline __aicore__ bool percounter_fanin_ready(const LaneConfig &cfg, const PercounterTask &task) {
    for (int32_t i = 0; i < task.fanin_count; ++i) {
        if (!percounter_counter_published(percounter_observe(percounter_producer(cfg, task, i)))) return false;
    }
    return true;
}

/** The task's kernel id per subtask slot; negative where the slot is unused. */
inline __aicore__ __gm__ int32_t *percounter_kernel_ids(const PercounterTask &task) {
    return reinterpret_cast<__gm__ int32_t *>(
        scheduler_graph_descriptor(task.graph, task.local) + SCHEDULER_GRAPH_KERNEL_IDS_OFFSET
    );
}

/**
 * Lay down this lane's deferred-completion record and hand back its address.
 *
 * Written once, before any kernel. The record never moves, and its identity
 * fields are the same for every task this lane runs: `task_ref`,
 * `block_index` and `subtask_slot` carry the values `layout_valid` demands
 * rather than the running task's, because nothing reads them back -- a
 * registration is refused whatever task made it.
 *
 * **The writeback is not optional.** The record sits past the region's uploaded
 * prefix, so this store is the only thing that has ever put this run's bytes
 * there, and a kernel that invalidates its data cache on entry would otherwise
 * find whatever an earlier bind left.
 */
inline __aicore__ __gm__ pto2::resident::AsyncGroupData *
percounter_open_async_record(__gm__ PercounterControl *control, int block_idx) {
    using pto2::resident::AsyncGroupData;
    const uint64_t lane = static_cast<uint64_t>(block_idx);
    const uint32_t capacity = static_cast<uint32_t>(scheduler_gm_query(control->async_capacity));
    const uint64_t entry_bytes = capacity * sizeof(DeferredCompletionEntry);
    auto *record = scheduler_state_at<AsyncGroupData>(
        control, scheduler_gm_query(control->async_groups_offset) + lane * sizeof(AsyncGroupData)
    );
    record->run_epoch = scheduler_gm_query(control->async_run_epoch);
    record->task_ref = 0;
    record->group_index = static_cast<uint32_t>(lane);
    record->block_index = 0;
    record->subtask_slot = 0;
    record->capacity = static_cast<uint16_t>(capacity);
    record->reserved = 0;
    record->entries_bytes = static_cast<uint32_t>(entry_bytes);
    record->count = 0;
    record->error_code = SIMPLER_ERROR_NONE;
    record->reservation_serial = 0;
    record->reservation_limit = 0;
    record->reservation_begin = 0;
    record->reservation_written = 0;
    record->state = pto2::resident::AsyncPublicationState::OPEN;
    // SDMA conditions are not served here, and the guard array a submission
    // would take is what their absence is spelled as.
    record->sdma_guards_offset = 0;
    record->sdma_guard_count = 0;
    record->entries_offset = scheduler_gm_query(control->async_entries_offset) + lane * entry_bytes;
    for (uint64_t offset = 0; offset < sizeof(AsyncGroupData); offset += 64)
        dcci((__gm__ uint8_t *)record + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    return record;
}

/**
 * Prepare one block's subtask from published argument metadata. Tensor values
 * and the dispatch predicate may still be produced elsewhere; neither is read
 * here. The slot remains owned by this candidate until it executes.
 */
inline __aicore__ bool percounter_prepare_subtask(
    const LaneConfig &cfg, const PercounterTask &task, int32_t subtask_slot, int32_t kernel_id, int32_t block_idx,
    int32_t block_num, __gm__ DispatchPayload *payload
) {
    percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(payload));
    uint64_t function_address = 0;
    if (!scheduler_lookup_callable_address(
            cfg.callable_addresses, cfg.callable_count, static_cast<uint16_t>(kernel_id), &function_address
        )) {
        return false;
    }
    const SchedulerTaskInfo info{
        task.local, kernel_id, subtask_slot, subtask_slot == 0 ? CoreType::AIC : CoreType::AIV
    };
    if (scheduler_materialize_task_payload_resolved(task.graph, info, function_address, payload) !=
        SchedulerGraphResult::OK) {
        return false;
    }
    if (cfg.profile != nullptr) {
        const TaskId token = task.shell < 0 ? TaskId::make_global(static_cast<int32_t>(task.id)) :
                                              TaskId::make_sub_task(task.shell, task.local);
        TaskId::assign(payload->local_context.async_ctx.task_token, token);
    }
    payload->local_context.block_idx = block_idx;
    payload->local_context.block_num = block_num;
    payload->global_context.sub_block_id = subtask_slot == 2 ? 1 : 0;
    // Other candidates and kernels invalidate the scalar cache. Persist the
    // prepared slot before yielding, even while its producers are still busy.
    percounter_writeback_dispatch_payload(payload);
    return true;
}

/** Execute a prepared slot only after its caller has established readiness. */
inline __aicore__ bool
percounter_run_prepared(const LaneConfig &cfg, const PercounterTask &task, __gm__ DispatchPayload *payload) {
    const uint64_t receive_time = cfg.profile != nullptr ? get_sys_cnt_aicore() : 0;
    percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(payload));
    if (cfg.has_predicates) {
        const SchedulerPredicateResult verdict = scheduler_evaluate_task_predicate(task.graph, task.local);
        if (verdict == SchedulerPredicateResult::MALFORMED) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
        if (verdict == SchedulerPredicateResult::FAIL) return true;
    }
    if (payload->function_bin_addr == 0) return true;

    __gm__ ChipSwimlaneAicoreTaskRecord *record = nullptr;
    if (cfg.profile != nullptr) {
        record = chip_swimlane_aicore_wait_task_record(cfg.profile_head, cfg.profile);
        if (record == nullptr) {
            percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
            return false;
        }
    }

    // The barrier is the publication fence a consumer's spin depends on: the
    // kernel's outputs are in memory before the atomic that announces them.
    PercounterKernelFunc kernel = reinterpret_cast<PercounterKernelFunc>(payload->function_bin_addr);
    const uint64_t start_time = cfg.profile != nullptr ? get_sys_cnt_aicore() : 0;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
    if (cfg.profile != nullptr) {
        const uint64_t end_time = get_sys_cnt_aicore();
        chip_swimlane_aicore_commit_task_record(
            record, payload->local_context.async_ctx.task_token,
            cfg.profile->cached_buf_seq * PLATFORM_AICORE_BUFFER_SIZE + cfg.profile->slot_within_buf, receive_time,
            start_time, end_time
        );
        if (cfg.profile->slot_within_buf == PLATFORM_AICORE_BUFFER_SIZE) {
            chip_swimlane_aicore_publish_task_count(cfg.profile);
        }
    }
    // A kernel that registered a completion condition is asking for its task to
    // be published only once that condition holds, and this runtime does not
    // honour that yet. Publishing anyway would release the consumers while the
    // data is still in flight, so the run fails here instead. Reading the record
    // is this core reading back its own stores: the registration needs no cache
    // operation to be visible, which is the same binding the resident scheduler
    // relies on across a kernel call.
    if (cfg.async_group->count != 0 || cfg.async_group->error_code != SIMPLER_ERROR_NONE) {
        percounter_fail(cfg, SchedulerGraphResult::ASYNC_REGISTRATION_FAILED);
        return false;
    }
    return true;
}

/**
 * Prepare one block drawn from an ordinary list. A dependency-only task has
 * no callable; its zero function address still allows completion publication.
 */
inline __aicore__ bool percounter_prepare_task(
    const LaneConfig &cfg, const PercounterTask &task, int32_t block_idx, int32_t block_num,
    __gm__ DispatchPayload *payload
) {
    percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(payload));
    __gm__ int32_t *kernel_ids = percounter_kernel_ids(task);
    for (int32_t slot = 0; slot < PLATFORM_CORES_PER_BLOCKDIM; ++slot) {
        const int32_t kernel_id = kernel_ids[slot];
        if (kernel_id >= 0)
            return percounter_prepare_subtask(cfg, task, slot, kernel_id, block_idx, block_num, payload);
    }
    payload->function_bin_addr = 0;
    percounter_writeback_dispatch_payload(payload);
    return true;
}

/** Prepare an ordinary candidate without reading its producer completion. */
inline __aicore__ bool percounter_prepare_pending(
    const LaneConfig &cfg, PercounterPending *pending, uint64_t entry, const PercounterTask &task
) {
    if (pending->prepared || percounter_order_is_arming(entry)) return true;
    if (!percounter_prepare_task(
            cfg, task, percounter_order_block_idx(entry), percounter_order_block_num(entry),
            cfg.payloads + pending->payload_slot
        )) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
        return false;
    }
    pending->prepared = true;
    return true;
}

/**
 * Record `count` finished subtasks of `task`; called after their store
 * barriers. A body sink's subtasks also count toward its shell, whose counter
 * is preset against every sink subtask of the body.
 */
inline __aicore__ void
percounter_publish_completion(const LaneConfig &cfg, const PercounterTask &task, uint64_t count) {
    percounter_gm_fetch_add(cfg.counters[task.id].value, count);
    if (task.sink) percounter_gm_fetch_add(cfg.counters[task.shell].value, count);
}

/** True when the lane should stop: EXIT is signalled or some lane has failed. */
inline __aicore__ bool percounter_should_stop(const LaneConfig &cfg) {
    return static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL ||
           percounter_gm_read(cfg.control->lane_error) != 0;
}

/**
 * MIX: a task whose subtasks run on two or three lanes of one cluster at once.
 *
 * Each cluster has one signal line in its SSBUF. The cluster's AIV0, the
 * claimer, draws MIX positions and announces one on the line once it is ready;
 * the lanes it names gather, each runs its own subtask, and the claimer frees
 * the line after all of them have finished. Every lane looks at the line at two
 * checkpoints: once per loop iteration, and inside the fanin spin.
 */
inline constexpr int32_t PERCOUNTER_MIX_CLAIMER_LANE = 1;  // AIV0

/**
 * Put this cluster's MIX line in its starting state, and do not return until it
 * is in that state.
 *
 * SSBUF survives a launch and the bind zeroes only GM, so the line can hold
 * whatever an earlier run left. The claimer clears it and then raises the
 * cluster's ready word, which lives in GM precisely because GM is what the bind
 * zeroes; the other two lanes wait for that word before reading the line.
 *
 * Returns false on EXIT, on another lane's error, or on the watchdog.
 */
inline __aicore__ bool percounter_clear_mix_signal(const LaneConfig &cfg, uint64_t timeout_cycles) {
    __gm__ uint64_t *ready = cfg.mix_signal_ready + cfg.cluster_index;
    if (cfg.cluster_lane == PERCOUNTER_MIX_CLAIMER_LANE) {
        percounter_mix_signal_clear(cfg.mix_signal);
        percounter_gm_store(*ready, UINT64_C(1));
        return true;
    }
    const uint64_t start = get_sys_cnt_aicore();
    uint32_t checks = 0;
    while (percounter_gm_read(*ready) == 0) {
        if (++checks == PERCOUNTER_SPIN_CHECK_INTERVAL) {
            checks = 0;
            if (percounter_should_stop(cfg)) return false;
            if (scheduler_watchdog_expired(start, get_sys_cnt_aicore(), timeout_cycles)) {
                percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
                return false;
            }
        }
        SPIN_WAIT_HINT();
    }
    return true;
}

/**
 * The subtask slots a task occupies, as a bitmask over {AIC, AIV0, AIV1}. Read
 * outside a kernel run, so the descriptor line is invalidated first: a body
 * task's descriptor is written by the AICPU, into a heap this core may have
 * cached in an earlier run.
 */
inline __aicore__ uint32_t percounter_mix_active_mask(const PercounterTask &task) {
    __gm__ int32_t *kernel_ids = percounter_kernel_ids(task);
    percounter_observe_cache_line(kernel_ids);
    uint32_t mask = 0;
    for (uint32_t slot = 0; slot < PLATFORM_CORES_PER_BLOCKDIM; ++slot) {
        if (kernel_ids[slot] >= 0) mask |= uint32_t{1} << slot;
    }
    return mask;
}

inline __aicore__ bool percounter_prepare_mix_payload(
    const LaneConfig &cfg, PercounterLaneState *state, const PercounterTask &task, int32_t block_idx, int32_t block_num
) {
    if (state->prepared_mix_task == task.id && state->prepared_mix_block == block_idx) return true;
    const int32_t kernel_id = percounter_kernel_ids(task)[cfg.cluster_lane];
    if (kernel_id < 0 ||
        !percounter_prepare_subtask(
            cfg, task, cfg.cluster_lane, kernel_id, block_idx, block_num, cfg.payloads + PERCOUNTER_MIX_PAYLOAD_SLOT
        )) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
        return false;
    }
    state->prepared_mix_task = task.id;
    state->prepared_mix_block = block_idx;
    return true;
}

/** An idle lane may prepare its MIX without committing to the rendezvous. */
inline __aicore__ bool percounter_mix_prepare(const LaneConfig &cfg, PercounterLaneState *state) {
    if (cfg.mix_order_count == 0) return true;
    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = percounter_ssbuf_load_acquire(&signal->prepare_generation);
    if (generation == 0 || (generation & 1U) != 0 || generation == state->prepared_mix_generation) return true;
    const int64_t task_id = percounter_ssbuf_load_field(&signal->prepare_task_id);
    const int32_t block_idx = percounter_ssbuf_load_field(&signal->prepare_block_idx);
    const int32_t block_num = percounter_ssbuf_load_field(&signal->prepare_block_num);
    const uint32_t mask = percounter_ssbuf_load_field(&signal->prepare_mask);
    if (percounter_ssbuf_load_acquire(&signal->prepare_generation) != generation) return true;
    if ((mask & (uint32_t{1} << cfg.cluster_lane)) != 0) {
        PercounterTask task{};
        if (!percounter_resolve(cfg, task_id, &task) ||
            !percounter_prepare_mix_payload(cfg, state, task, block_idx, block_num))
            return false;
    }
    state->prepared_mix_generation = generation;
    return true;
}

/**
 * Free the cluster's line once every participant has finished.
 *
 * The claimer's, not the lowest participant's: mask `0b101` omits AIV0, and
 * SSBUF has no atomic to arbitrate a second writer of the generation.
 */
inline __aicore__ void percounter_mix_release(const LaneConfig &cfg) {
    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = percounter_ssbuf_load_acquire(&signal->armed_generation);
    if (!percounter_mix_armed(generation)) return;
    const uint32_t mask = signal->active_mask;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((mask & (uint32_t{1} << lane)) != 0 && signal->finished[lane] != generation) return;
    }
    percounter_ssbuf_store_release(&signal->armed_generation, generation + 1);
    // After the generation, so no lane reads a free line's mask as this
    // rendezvous': a free line carries a mask only while an arming is staging
    // one, which is what holds a block back from its kernel.
    signal->active_mask = 0;
}

/**
 * Sync-start cohorts: a task whose `block_num` blocks must all start together.
 *
 * Such a task holds one arming entry in its list instead of one per block. The
 * lane that runs the entry queues the task; the queue promotes cohorts one at a
 * time into its open slot, and free lanes of the right kind each take one block
 * of the open cohort with a fetch_add. The lane that takes the last block marks
 * the cohort released, and every holder runs its block. The next cohort opens
 * only after every block of this one has completed.
 *
 * **A lane that owes a cohort block and a rendezvous at once serves them in a
 * fixed order.** The two gather different sets of lanes -- a cohort's kernel
 * holds each block at its barrier until every holder is in it, a rendezvous
 * holds each subtask until every participant has arrived -- so a lane that owes
 * both is a lane two sets wait for, and serving them in an order the sets can
 * disagree about is what closes a cycle.
 *
 * **The arming sites keep a holder out of a line it would be the last to
 * reach.** `percounter_mix_arm` refuses a mask naming any lane whose hold it
 * can see; `percounter_sync_serve_block`, which arms the same line for a MIX
 * cohort's own block, refuses one naming a holder of a *strictly older* cohort,
 * since a holder of a later one is exactly the lane this line is waiting to
 * free. Both read the holds twice, the second time with the payload staged and
 * the generation not yet turned, so an arming a take has overtaken is retracted
 * rather than published. A lane likewise does not take a block while the line
 * carries a generation it has not looked at.
 *
 * **What the holds cannot cover, the order does.** Neither read is atomic with
 * the fetch_add that takes a block, so a lane can still end up holding one and
 * named in a line armed while it was clear. `percounter_line_precedes_block`
 * decides that case: an ordinary draw is always served before the block, a
 * cohort rendezvous when its cohort is the older. Every lane the line names
 * resolves it the same way, so the line completes and the blocks follow.
 *
 * A hold is published before its block can be served, and for a single-lane
 * block it is cleared only after that kernel returns, so a lane the arming read
 * clear is not inside one. **A MIX cohort block is the exception**: its claimer
 * clears the hold where it arms the line instead, before the rendezvous runs.
 * Nothing is stranded by that -- the line it just armed is what refuses this
 * cluster's next arming -- but the hold alone does not say where that lane is.
 *
 * Taking a block never blocks, and a holder never takes a second.
 */
inline constexpr uint32_t PERCOUNTER_SYNC_CHECK_INTERVAL = 8;

/**
 * Whether this lane is in the pool that serves a cohort of this shape. A MIX
 * cohort is served by claimers alone: the claimer arms its own cluster's
 * rendezvous rather than running the block, so one block is one commitment.
 */
inline __aicore__ bool percounter_sync_can_serve(const LaneConfig &cfg, uint32_t active_mask) {
    if ((active_mask & (active_mask - 1U)) != 0) return cfg.cluster_lane == PERCOUNTER_MIX_CLAIMER_LANE;
    if (active_mask == 1U) return cfg.cluster_lane == 0;
    return cfg.cluster_lane != 0;
}

/**
 * Order a store on this line against the loads that follow it.
 *
 * Neither SSBUF primitive gives this: the release store puts its barrier
 * *before* the store and the acquire load puts its after, so a store followed
 * by a load of another word has nothing between them. Two lanes that each
 * publish a word and then read the other's are a mutual exclusion that needs
 * exactly that edge, and without it both can read stale and both proceed.
 */
inline __aicore__ void percounter_publish_fence() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/**
 * Order a read of GM against the reads that follow it.
 *
 * **Two GM reads are two samples, not one snapshot.** A lane that gates on
 * one word and then reads what that word vouches for needs this between them,
 * or the second read can return a value older than the first implies. Reading
 * both atomically is not a substitute: measured, that orders a pair sharing a
 * cache line and makes a pair that does not far worse. The measurements, and
 * why no barrier on the writing lane appears here, are in
 * `docs/investigations/2026-11-a5-gm-read-ordering.md`.
 *
 * SSBUF needs no equivalent: `percounter_ssbuf_load_acquire` is a load
 * *followed* by the barrier, so consecutive acquire loads already carry it.
 */
inline __aicore__ void percounter_observe_fence() {
#if defined(__CCE_AICORE__)
    dsb((mem_dsb_t)0);
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/**
 * Read a word a decision gates on, and order it against what follows.
 *
 * The atomic read keeps this one word from being stale; the fence keeps the
 * words read after it from being older. A gate needs both, so they are
 * bundled rather than left for each caller to remember.
 */
inline __aicore__ uint64_t percounter_gate_read(__gm__ volatile uint64_t &word) {
    const uint64_t value = percounter_gm_fetch_add(word, uint64_t{0});
    percounter_observe_fence();
    return value;
}

/**
 * Publish which cohort this lane owes a block of, as `seq + 1`; 0 is none.
 *
 * Only whether it is zero is read. The sequence is kept because it names the
 * cohort in a dump, and because `seq + 1` leaves 0 free to mean none.
 *
 * **The fence is half of a mutual exclusion**, not a publication barrier: an
 * arming stages its mask and then reads these holds, and this lane publishes
 * its hold and then reads that mask. Both edges have to hold for the pair to
 * exclude each other.
 */
inline __aicore__ void percounter_cohort_hold_set(const LaneConfig &cfg, uint64_t seq_plus_one) {
    if (cfg.mix_order_count == 0) return;
    percounter_ssbuf_store_release(&cfg.mix_signal->cohort_hold[cfg.cluster_lane], static_cast<uint32_t>(seq_plus_one));
    percounter_publish_fence();
}

/**
 * Whether any lane `mask` names still owes a cohort block.
 *
 * One cluster's line answers the whole question, because a mask names lanes of
 * one cluster. A graph with no cohort never enters the loop.
 */
inline __aicore__ bool percounter_cohort_held_in(const LaneConfig &cfg, uint32_t mask) {
    if (cfg.sync_cohort_count == 0) return false;
    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((mask & (uint32_t{1} << lane)) != 0 && percounter_ssbuf_load_acquire(&signal->cohort_hold[lane]) != 0) {
            return true;
        }
    }
    return false;
}

/**
 * Whether the cluster's line carries a generation this lane has not looked at.
 *
 * Blind to the mask on purpose: one SSBUF load against a seqlock read of the
 * payload, and a generation that does not name this lane is marked seen at the
 * next checkpoint, so a lane this holds back from a cohort loses a checkpoint
 * rather than the cohort.
 */
inline __aicore__ bool percounter_mix_unseen(const LaneConfig &cfg, const PercounterLaneState *state) {
    if (cfg.mix_order_count == 0) return false;
    const uint32_t generation = percounter_ssbuf_load_acquire(&cfg.mix_signal->armed_generation);
    return percounter_mix_armed(generation) && generation != state->last_served;
}

/**
 * Whether the cluster's line names this lane under a draw from the MIX list,
 * which is served before the block this lane holds.
 *
 * A draw's arming site refuses a mask naming any lane whose hold it can see,
 * so a draw this lane is named in was announced while this lane was clear --
 * it is the take that followed, not the line that followed the take. Serving
 * the block first would put this lane inside a cohort kernel while every
 * participant of that draw waits, and they cannot release it, since the lane
 * it is missing is this one.
 *
 * **A cohort rendezvous on the line is this lane's own cohort**, since only
 * one is outstanding, so it never precedes the block: the two are the same
 * gathering.
 *
 * Read under the same seqlock as discovery: the payload belongs to this
 * generation only while the generation has not moved.
 */
inline __aicore__ bool percounter_line_precedes_block(const LaneConfig &cfg, const PercounterLaneState *state) {
    if (cfg.mix_order_count == 0) return false;
    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = percounter_ssbuf_load_acquire(&signal->armed_generation);
    if (!percounter_mix_armed(generation) || generation == state->last_served) return false;
    const uint32_t mask = signal->active_mask;
    const uint32_t cohort = signal->cohort_block;
    if (percounter_ssbuf_load_acquire(&signal->armed_generation) != generation) return false;
    if ((mask & (uint32_t{1} << cfg.cluster_lane)) == 0) return false;
    return cohort == 0;
}

/**
 * Whether an arming is staged on the cluster's line and names this lane.
 *
 * **A mask under an even generation is an arming in flight.** Both arming sites
 * stage the payload before they turn the generation odd, and both clear the
 * mask on every path that gives up, so a free line carries a mask only between
 * those two points. The lane named there is one the arming has already decided
 * to name: its holds were read clear, and a block taken since is invisible to
 * the store that is about to publish.
 *
 * Serving a block under that mask is what the order cannot repair afterwards --
 * the lane enters a kernel it does not leave until its co-holders arrive, and
 * the line that names it is published a store later, after the last checkpoint
 * it will reach. Waiting costs the stores between the staging and the
 * generation, and the line then decides the order as it does for any other.
 */
inline __aicore__ bool percounter_line_staging_for(const LaneConfig &cfg) {
    if (cfg.mix_order_count == 0) return false;
    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    if (percounter_mix_armed(percounter_ssbuf_load_acquire(&signal->armed_generation))) return false;
    return (signal->active_mask & (uint32_t{1} << cfg.cluster_lane)) != 0;
}

/**
 * Move the next queued cohort into the open slot, if the slot is free.
 *
 * Only an enqueuer and a releaser call this; a general poll would put the
 * whole pool on one compare-exchange. `open_task_id` is written last and is
 * the only field a reader gates on.
 *
 * **The sequence to promote is derived from the generation the CAS checks.**
 * With a separate cursor, a lane could read it, then lose the time in which
 * another lane promoted that cohort and it filled and was released -- and its
 * CAS on the now-free slot would promote the same cohort a second time.
 */
inline __aicore__ void percounter_sync_promote(const LaneConfig &cfg) {
    __gm__ PercounterSyncQueue *q = cfg.sync_queue;
    // The gate for everything below, and the one read nothing downstream can
    // recover from: a lane that reads the previous generation sees the slot
    // still armed and returns here, never reaching the exchange that would
    // have caught it. The fence is what puts `claimed_blocks`, which sits in
    // a cache line of its own, no earlier than this.
    const uint64_t generation = percounter_gate_read(q->open_generation);
    if (percounter_sync_armed(generation)) return;
    const uint64_t seq = generation / 2;
    if (seq >= percounter_gm_fetch_add(q->tail, uint64_t{0})) return;
    // `tail` counts every attempt, including one the enqueue refused for being
    // past the ring, so it alone does not bound this read.
    if (seq >= cfg.sync_cohort_count) return;
    const uint64_t desc = percounter_gm_fetch_add(cfg.sync_pending[seq], uint64_t{0});
    // Reserved but not yet written: its own enqueuer will promote it.
    if (desc == 0) return;
    // **Only one cohort is ever outstanding.** A cohort whose blocks have not
    // all run can still be waiting for a lane to take one, and a second cohort
    // in that state takes lanes the first is waiting for while the first holds
    // lanes the second needs -- each one's kernel is entered a block at a time
    // and cannot be left, so the two hold each other there. Opening the next
    // one only once the last has run is what keeps that pair from existing.
    // Atomic, like every read here, because a spin can afford to be behind
    // and this decides once -- a stale read abandons the cohort for good. The
    // two need no fence between them: they share a cache line, which is what
    // the layout puts them in, and that is enough to order them.
    const uint64_t done = percounter_gm_fetch_add(cfg.sync_queue->completed, uint64_t{0});
    const uint64_t owed = percounter_gm_fetch_add(cfg.sync_queue->claimed_blocks, uint64_t{0});
    if (done != owed) return;
    if (percounter_gm_compare_exchange(q->open_generation, generation, generation + 1) != generation) return;

    percounter_gm_store(q->open_block_num, percounter_sync_desc_blocks(desc));
    percounter_gm_store(q->open_active_mask, percounter_sync_desc_mask(desc));
    percounter_gm_store(q->open_seq, seq);
    percounter_gm_store(q->open_task_id, percounter_sync_desc_task(desc));
}

/**
 * Queue `task_id` for gathering. Never fails and is never retried: the
 * sequence number is taken first, and the trailing promote covers the slot
 * falling free between this lane's two steps.
 */
inline __aicore__ void percounter_sync_enqueue(const LaneConfig &cfg, const PercounterTask &task, int32_t block_num) {
    const uint32_t mask = percounter_mix_active_mask(task);
    const uint64_t seq = percounter_gm_fetch_add(cfg.sync_queue->tail, uint64_t{1});
    // The ring holds one slot per cohort the planner counted, and the claims it
    // hands out live in the half past them. A sequence beyond that would write
    // a descriptor over another cohort's claim count and be found, if at all,
    // as a cohort that admits one block too many -- so it fails here instead.
    if (seq >= cfg.sync_cohort_count) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_ID);
        return;
    }
    percounter_gm_store(cfg.sync_pending[seq], percounter_sync_desc(task.id, static_cast<uint32_t>(block_num), mask));
    percounter_sync_promote(cfg);
}

/**
 * Take a block of cohort `seq`, or -1 if none is left.
 *
 * One fetch_add on that cohort's own claim count, never a retry loop: a
 * compare-exchange loop measured 6-11 attempts per take with the whole pool on
 * one line. An index past the end is discarded, and every real index still
 * went to exactly one lane, however late the add.
 */
inline __aicore__ int32_t percounter_sync_take_block(const LaneConfig &cfg, uint64_t seq, uint32_t block_num) {
    __gm__ uint64_t *claims = percounter_sync_claims(cfg.sync_pending, cfg.sync_cohort_count, seq);
    if (percounter_gm_read(*claims) >= block_num) return -1;
    const uint64_t prev = percounter_gm_fetch_add(*claims, uint64_t{1});
    return prev < block_num ? static_cast<int32_t>(prev) : -1;
}

/**
 * Run the block this lane holds, now that its cohort is released.
 *
 * A MIX cohort block arms this lane's own cluster instead, with `cohort_block`
 * set so the rendezvous reports to the queue rather than to the MIX list. A
 * line already carrying one is left alone and the block kept: the caller comes
 * back at its next checkpoint, and an armed line frees in bounded time, since
 * every participant it names is already gathering. `my_block` stays set until
 * the block is run or armed.
 */
inline __aicore__ bool percounter_sync_serve_block(const LaneConfig &cfg, PercounterLaneState *state) {
    PercounterSyncState *sync = &state->sync;
    if ((sync->active_mask & (sync->active_mask - 1U)) == 0) {
        PercounterTask task{};
        if (!percounter_resolve(cfg, sync->task_id, &task)) return false;
        if (!percounter_run_prepared(cfg, task, cfg.payloads + PERCOUNTER_SYNC_PAYLOAD_SLOT)) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
            return false;
        }
        percounter_publish_completion(cfg, task, uint64_t{1});
        ++state->local_done;
        sync->my_block = -1;
        percounter_cohort_hold_set(cfg, 0);
        percounter_gm_fetch_add(cfg.sync_queue->completed, uint64_t{1});
        // The count this block just raised is what the next cohort waits on,
        // and the lane that raises it is the only one that knows it changed.
        percounter_sync_promote(cfg);
        return true;
    }

    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = percounter_ssbuf_load_acquire(&signal->armed_generation);
    if (percounter_mix_armed(generation)) return true;
    // **No hold to read here, unlike the draw from the MIX list**, and the line
    // itself is why. A lane this mask names is dangerous when it cannot reach a
    // checkpoint -- inside some other gathering's kernel -- and that gathering
    // would be on this very line, which the check above already refuses over.
    // One cohort is outstanding and it is this one, so the only other hold the
    // mask could name is of a block that does not exist.
    //
    // The two arming sites cannot race over the staged payload either: a mask
    // with more than one bit is served by the claimer lane alone
    // (`percounter_sync_can_serve`), so both sites run on that one lane.
    signal->task_id = sync->task_id;
    signal->active_mask = sync->active_mask;
    signal->block_idx = sync->my_block;
    signal->block_num = static_cast<int32_t>(sync->block_num);
    signal->cohort_block = static_cast<uint32_t>(sync->my_seq + 1);
    // The mask is staged before the generation turns, as the draw stages its
    // own, so a lane reading a free line still sees who an arming in flight is
    // about to name.
    percounter_publish_fence();
    percounter_ssbuf_store_release(&signal->armed_generation, generation + 1);
    sync->my_block = -1;
    // The armed line now carries this block, and is itself what holds the
    // claimer's next draw off this lane.
    percounter_cohort_hold_set(cfg, 0);
    return true;
}

/**
 * Serve the block this lane holds if its cohort has been released; `served`
 * reports whether a block was run or armed. Checked at every checkpoint, ahead
 * of MIX discovery, and costs one load only while a block is held.
 *
 * **This is the only way a held block enters its kernel, and the order a lane
 * owing two gatherings serves them in is enforced here rather than in the
 * callers.** Entering the block's kernel puts this lane where no checkpoint
 * reaches it until its co-holders arrive, so it is the step that cannot be
 * taken back, and every reason to wait is checked before it.
 *
 * A rendezvous this lane is named in goes first, and is acyclic for two
 * separate reasons. **An ordinary draw** was announced while this lane was
 * clear -- both arming sites read the holds -- so every lane it names is in the
 * same position and resolves it the same way; the line completes, and the
 * blocks follow. **A cohort rendezvous** goes first only when its cohort is the
 * older, which is a strictly smaller sequence of one global counter.
 *
 * **A mask staged on a free line also waits**, for the one case the holds
 * cannot cover: an arming reads them clear, and a lane takes a block and
 * reaches here before the store that would have told it so.
 *
 * Returns false on an error, including a cohort that has not filled within the
 * watchdog since this lane took its block.
 */
inline __aicore__ bool percounter_sync_serve_released(
    const LaneConfig &cfg, PercounterLaneState *state, uint64_t timeout_cycles, bool *served
) {
    *served = false;
    PercounterSyncState *sync = &state->sync;
    if (sync->my_block < 0) return true;
    // Ahead of the watchdog: the wait is on a rendezvous that carries its own
    // budget, or on the stores that finish one, and neither is what this
    // block's budget bounds.
    if (percounter_line_precedes_block(cfg, state)) return true;
    if (percounter_line_staging_for(cfg)) return true;
    // Cohorts are promoted in sequence order, so this one is fully claimed
    // exactly when the counter passes it.
    if (percounter_gm_read(cfg.sync_queue->released) > sync->my_seq) {
        const bool ok = percounter_sync_serve_block(cfg, state);
        *served = sync->my_block < 0;
        return ok;
    }
    if (scheduler_watchdog_expired(sync->taken_cycles, get_sys_cnt_aicore(), timeout_cycles)) {
        percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
        return false;
    }
    return true;
}

/**
 * Take a block of the open cohort if this lane holds none and has not yet
 * decided about it. Taking the last block releases the cohort and serves at
 * once; promotion waits for the cohort to complete.
 *
 * Never blocks: the lane can serve its ordinary position while holding a
 * cohort block, and its checkpoints serve the block once released.
 */
inline __aicore__ bool
percounter_sync_take(const LaneConfig &cfg, PercounterLaneState *state, uint64_t timeout_cycles, bool *served) {
    *served = false;
    PercounterSyncState *sync = &state->sync;
    if (cfg.sync_cohort_count == 0 || sync->my_block >= 0) return true;
    const uint64_t generation = percounter_gm_read(cfg.sync_queue->open_generation);
    // Before any further load: a lane that has decided reaches here at every
    // checkpoint while the cohort is open, and reading the payload first cost
    // 40% of `sparse_256_b24`.
    if (!percounter_sync_armed(generation) || sync->decided_generation == generation) return true;
    // Ahead of the decision, which is final: a lane that still owes the line a
    // look may be named in it, and a block taken now would leave this lane
    // owing a cohort and a rendezvous at once.
    if (percounter_mix_unseen(cfg, state)) return true;
    // The seqlock's opening edge: a task id older than this generation would
    // be the previous cohort's, and the closing check compares generations,
    // not ids, so it cannot see that. A graph with no MIX list reaches here
    // without the acquire load above, so nothing else carries this edge.
    percounter_observe_fence();
    const int64_t task_id = percounter_gm_read(cfg.sync_queue->open_task_id);
    // Slot taken, payload not yet published: deciding now would read the
    // previous cohort's shape, and a decision is final.
    if (task_id < 0) return true;
    if (task_id >= static_cast<int64_t>(cfg.task_count)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_ID);
        return false;
    }
    // The task id is what vouches for the rest of the payload, so the loads it
    // gates must not be allowed to precede it.
    percounter_observe_fence();
    const uint64_t seq = percounter_gm_read(cfg.sync_queue->open_seq);
    const uint64_t shape = percounter_gm_read_u32_pair(&cfg.sync_queue->open_block_num);
    // A seqlock: the slot may have been released and refilled between these
    // loads, and the task, sequence and shape must all be one cohort's. It is
    // read last and must observe at least what they did -- a generation load
    // that ran ahead of them would return the value this lane already holds
    // and pass a check that exists to fail.
    percounter_observe_fence();
    if (percounter_gm_read(cfg.sync_queue->open_generation) != generation) return true;
    sync->decided_generation = generation;
    sync->task_id = task_id;
    sync->block_num = static_cast<uint32_t>(shape & 0xffffffffULL);
    sync->active_mask = static_cast<uint32_t>(shape >> 32);
    if (!percounter_sync_can_serve(cfg, sync->active_mask)) return true;
    const int32_t block = percounter_sync_take_block(cfg, seq, sync->block_num);
    if (block < 0) return true;
    sync->my_block = block;
    sync->my_seq = seq;
    sync->taken_cycles = get_sys_cnt_aicore();
    percounter_cohort_hold_set(cfg, seq + 1);
    if ((sync->active_mask & (sync->active_mask - 1U)) == 0) {
        PercounterTask task{};
        if (!percounter_resolve(cfg, task_id, &task)) return false;
        if (!percounter_prepare_task(
                cfg, task, block, static_cast<int32_t>(sync->block_num), cfg.payloads + PERCOUNTER_SYNC_PAYLOAD_SLOT
            )) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
            return false;
        }
    }
    if (static_cast<uint32_t>(block) + 1 < sync->block_num) return true;

    // The last block. `released` goes out before the slot is freed: it is what
    // the other holders wait on, and the slot is about to belong to another.
    __gm__ PercounterSyncQueue *q = cfg.sync_queue;
    percounter_gm_store(q->open_task_id, int64_t{-1});
    percounter_gm_store(q->released, seq + 1);
    // Before the slot frees, so a promotion that sees it free also sees what
    // this cohort still owes: the pair is what says whether one is
    // outstanding. The order these two go out in is not what carries that --
    // the promotion fences its own reads instead -- so this lane only has to
    // issue the count before it hands the slot on.
    percounter_gm_fetch_add(q->claimed_blocks, static_cast<uint64_t>(sync->block_num));
    percounter_gm_store(q->open_generation, generation + 1);
    percounter_sync_promote(cfg);
    return percounter_sync_serve_released(cfg, state, timeout_cycles, served);
}

/**
 * Draw the next MIX position for this cluster, and announce the one it holds
 * once that task is runnable.
 *
 * **The position is taken with a fetch_add, not won with a CAS**, so every
 * claimer gets a distinct one and no probe is thrown away. Racing all clusters
 * for the list head lost 83-95% of ready probes on silicon.
 *
 * **Announcing waits for readiness**: a lane is only ever told about a task it
 * can run at once, so a MIX task never stops a lane draining its own list.
 * Holding the position meanwhile lets one cluster per MIX task be waiting at
 * once, instead of every cluster stalling behind an unready list head.
 *
 * Neither the draw nor the probe is rationed. A cluster draws only when its
 * line is free and it holds nothing, so it draws once per MIX task it runs; and
 * on a chain the held position is the critical path, which a probe every few
 * checkpoints stretched from 5 to 82 ms on `chain_1024`.
 */
inline __aicore__ void percounter_mix_arm(const LaneConfig &cfg, PercounterMixClaim *claim) {
    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = percounter_ssbuf_load_acquire(&signal->armed_generation);
    if (percounter_mix_armed(generation)) return;

    if (claim->task_id < 0) {
        if (claim->position < 0) {
            if (claim->list_spent) return;
            const uint64_t draw =
                percounter_gm_fetch_add(cfg.mix_tickets[PERCOUNTER_MIX_TICKET_DRAW].next, uint64_t{1});
            if (draw >= cfg.mix_order_count) {
                claim->list_spent = true;
                return;
            }
            claim->position = static_cast<int64_t>(draw);
        }
        // A Graph body's entry reads zero until the AICPU has materialized it.
        const uint64_t entry = percounter_gm_read(cfg.mix_order[claim->position]);
        if (percounter_order_block_num(entry) == 0) return;
        claim->position = -1;
        claim->task_id = percounter_order_task_id(entry);
        claim->block_idx = percounter_order_block_idx(entry);
        claim->block_num = percounter_order_block_num(entry);
        claim->prepare_announced = false;
    }
    PercounterTask task{};
    if (!percounter_resolve(cfg, claim->task_id, &task)) return;
    if (claim->block_idx != PERCOUNTER_ORDER_BLOCK_IDX_ARM && !claim->prepare_announced) {
        const uint32_t prepare_generation = percounter_ssbuf_load_acquire(&signal->prepare_generation);
        percounter_ssbuf_store_release(&signal->prepare_generation, prepare_generation + 1);
        percounter_ssbuf_store_field(&signal->prepare_task_id, claim->task_id);
        percounter_ssbuf_store_field(&signal->prepare_block_idx, claim->block_idx);
        percounter_ssbuf_store_field(&signal->prepare_block_num, claim->block_num);
        percounter_ssbuf_store_field(&signal->prepare_mask, percounter_mix_active_mask(task));
        percounter_ssbuf_store_release(&signal->prepare_generation, prepare_generation + 2);
        claim->prepare_announced = true;
    }
    if (!percounter_fanin_ready(cfg, task)) return;

    // A sync-start MIX task's single arming entry queues the cohort instead of
    // arming this cluster; its blocks are handed out from the queue.
    if (claim->block_idx == PERCOUNTER_ORDER_BLOCK_IDX_ARM) {
        percounter_sync_enqueue(cfg, task, claim->block_num);
        claim->task_id = -1;
        return;
    }

    const uint32_t mask = percounter_mix_active_mask(task);
    // A lane this mask names that still owes a cohort block would owe two
    // gatherings at once. The claim is kept, and announced once that block is
    // served -- which the holder's own checkpoints reach without this cluster.
    if (percounter_cohort_held_in(cfg, mask)) return;

    // The payload before the generation, and the generation with release
    // ordering: a lane that observes an odd generation observes its payload.
    signal->task_id = claim->task_id;
    signal->active_mask = mask;
    signal->block_idx = claim->block_idx;
    signal->block_num = claim->block_num;
    // Every arming: the previous rendezvous on this line may have been a
    // cohort block.
    signal->cohort_block = 0;
    // The holds again, with the payload staged and nothing published: a lane
    // that took a block since the first read is named here, and the generation
    // is what would tell it so. Returning now retracts the whole arming, since
    // a payload under an even generation is not a rendezvous -- the mask goes
    // with it, because a staged mask is what holds a named lane's block back.
    // The fence is the other half of that exclusion: without it this read may
    // predate the mask store that the lane is meanwhile failing to see.
    percounter_publish_fence();
    if (percounter_cohort_held_in(cfg, mask)) {
        signal->active_mask = 0;
        return;
    }
    percounter_ssbuf_store_release(&signal->armed_generation, generation + 1);
    claim->task_id = -1;
}

/**
 * The claimer's duty at a checkpoint: free the line if its rendezvous is over,
 * then draw or announce.
 *
 * **It runs wherever discovery runs, the fanin spin included.** Without it
 * there, `wide_240` deadlocked in three runs of ten: a claimer blocked on a
 * producer could neither free its line nor announce the next MIX task, and once
 * enough clusters waited on producers that were themselves MIX tasks nothing
 * could advance the MIX list.
 *
 * A claimer holding a cohort block still releases but does not announce: once
 * its cohort is released, peers that started first wait at the kernel's
 * barrier for this block, and a claimer preferring its own MIX list would hold
 * them there.
 */
inline __aicore__ void percounter_mix_tick(const LaneConfig &cfg, PercounterLaneState *state) {
    if (cfg.mix_order_count == 0 || cfg.cluster_lane != PERCOUNTER_MIX_CLAIMER_LANE) return;
    percounter_mix_release(cfg);
    if (state->sync.my_block < 0) percounter_mix_arm(cfg, &state->claim);
}

/**
 * Whether a lane whose own list is spent may stop.
 *
 * A peer that arrives at a rendezvous whose other participant has gone waits
 * inside a kernel, where it polls neither EXIT nor the error word. So the test
 * is the count of MIX rendezvous that have **completed**, not of positions
 * drawn: drawing and announcing are two steps, and a lane reading "list spent,
 * my line free" between them would leave while its claimer is about to name it.
 *
 * Cohorts follow the same rule: every cohort block must have run, not merely
 * have an owner, since a MIX cohort block is run by its cluster's rendezvous
 * rather than by the lane that took it. And a lane that owes a block never
 * leaves: the cohort is released the instant its last block is taken.
 *
 * `progress` is what the caller watches the device by, and it is the whole
 * device: the blocks every lane has resolved, plus the MIX and cohort counts
 * this test already reads. All three are monotone, so the sum advances exactly
 * when one of them does.
 *
 * The resolved count has to be in it. A lane whose own list is spent waits here
 * for work that may be gated on a long run of ordinary tasks elsewhere -- an
 * AIC list that empties while the AIV chain still has a thousand tasks to go
 * before it unlocks the one MIX. The MIX and cohort counts are flat for that
 * whole prefix, so a watchdog on them alone reads a healthy device as a stalled
 * one and fails the run. The cohort count is read before the MIX short-circuit
 * for the same reason: a graph still advancing its cohorts is advancing whether
 * or not its MIX list is done.
 */
inline __aicore__ bool
percounter_lane_may_exit(const LaneConfig &cfg, const PercounterLaneState *state, uint64_t *progress) {
    uint64_t done = percounter_gm_read(cfg.control->completed_block_count);
    bool complete = true;
    if (cfg.mix_order_count != 0) {
        const uint64_t mix_done = percounter_gm_read(cfg.mix_tickets[PERCOUNTER_MIX_TICKET_DONE].next);
        done += mix_done;
        complete = complete && mix_done >= cfg.mix_rendezvous_count;
    }
    if (cfg.sync_cohort_count != 0) {
        const uint64_t cohort_done = percounter_gm_read(cfg.sync_queue->completed);
        done += cohort_done;
        complete = complete && cohort_done >= cfg.sync_block_total;
    }
    *progress = done;
    return complete && state->sync.my_block < 0;
}

/**
 * Spin until every participant's token has reached `generation`.
 *
 * `tokens` is `arrived` on the way in and `finished` on the way out. Returns
 * false on EXIT, on another lane's error, or on the watchdog -- what a lane
 * spinning here can still see and a lane inside a kernel cannot, which is why
 * the gathering is a software spin rather than a channel wait.
 *
 * **Reached, not equal to.** A token holds the generation its lane last served,
 * so it is monotone, and a later one says that lane passed this generation on
 * the way: an entry is written once per generation its lane joins, in order,
 * and the line cannot be rearmed before every participant has written this
 * one. Equality would instead read a token that has moved on as one that never
 * arrived, which is reachable on the way out: the line is freed once the
 * entries are in, while the lowest participant is still gathering them, and the
 * next task on this cluster need not hold the same lanes -- `0b011` followed by
 * `0b110` runs without the lane still gathering and overwrites the entry it
 * waits on. Under equality that lane waits for a value that never returns and
 * its task's counter is never published; under this one it sees the peer is
 * past, which is what it needed to know. The ordering carries with it: the
 * later store is a release behind the earlier one, so observing it observes the
 * subtask outputs of both.
 */
inline __aicore__ bool percounter_mix_wait(
    const LaneConfig &cfg, PERCOUNTER_SSBUF volatile uint32_t *tokens, uint32_t mask, uint32_t generation,
    uint64_t timeout_cycles
) {
    const uint64_t start = get_sys_cnt_aicore();
    uint32_t checks = 0;
    while (true) {
        bool complete = true;
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if ((mask & (uint32_t{1} << lane)) != 0 && percounter_ssbuf_load_acquire(&tokens[lane]) < generation) {
                complete = false;
                break;
            }
        }
        if (complete) return true;
        if (++checks == PERCOUNTER_SPIN_CHECK_INTERVAL) {
            checks = 0;
            if (percounter_should_stop(cfg)) return false;
            if (scheduler_watchdog_expired(start, get_sys_cnt_aicore(), timeout_cycles)) {
                percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
                return false;
            }
        }
        SPIN_WAIT_HINT();
    }
}

/**
 * This lane's part in one rendezvous: arrive, wait for the others, run its own
 * subtask, report finished.
 *
 * **Writing `arrived` is irrevocable.** Once it completes the mask a peer may
 * enter its kernel, and a lane that then withdrew would leave that peer waiting
 * on an end that never comes. Nothing can make a participant want to withdraw:
 * the claimer announced the task only once it was ready, and readiness is
 * monotone. The readiness check below is therefore an assertion.
 *
 * The lowest participant publishes the whole rendezvous; every lane computes
 * the same lowest from the same mask, so this needs no election. It reports a
 * cohort block to the cohort queue and any other block to the MIX list it was
 * drawn from.
 */
inline __aicore__ bool percounter_serve_mix(
    const LaneConfig &cfg, uint32_t generation, int64_t task_id, uint32_t mask, int32_t block_idx, int32_t block_num,
    bool cohort_block, PercounterLaneState *state, uint64_t timeout_cycles
) {
    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t lane = static_cast<uint32_t>(cfg.cluster_lane);
    PercounterTask task{};
    if (!percounter_resolve(cfg, task_id, &task)) return false;
    if (!percounter_fanin_ready(cfg, task)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return false;
    }

    __gm__ DispatchPayload *payload =
        cfg.payloads + (cohort_block ? PERCOUNTER_SYNC_PAYLOAD_SLOT : PERCOUNTER_MIX_PAYLOAD_SLOT);
    if (cohort_block) {
        const int32_t kernel_id = percounter_kernel_ids(task)[cfg.cluster_lane];
        if (kernel_id < 0 ||
            !percounter_prepare_subtask(cfg, task, cfg.cluster_lane, kernel_id, block_idx, block_num, payload)) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
            return false;
        }
    } else if (!percounter_prepare_mix_payload(cfg, state, task, block_idx, block_num)) {
        return false;
    }

    percounter_ssbuf_store_release(&signal->arrived[lane], generation);
    if (!percounter_mix_wait(cfg, signal->arrived, mask, generation, timeout_cycles)) return false;
    // Inside the kernel this lane polls nothing. Every participant has arrived,
    // so a cross-core channel wait in it has a peer to meet.
    if (!percounter_run_prepared(cfg, task, payload)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
        return false;
    }
    // After the kernel's store barrier, so a peer that observes this token has
    // observed the subtask's outputs.
    percounter_ssbuf_store_release(&signal->finished[lane], generation);

    if ((uint32_t{1} << lane) == (mask & (~mask + 1U))) {
        if (!percounter_mix_wait(cfg, signal->finished, mask, generation, timeout_cycles)) return false;
        // The counter was preset against the task's whole subtask total, so
        // this add carries the block's share of it at once.
        percounter_publish_completion(cfg, task, static_cast<uint64_t>(__builtin_popcount(mask)));
        // After the task counter: a lane that sees either count reach its total
        // may stop, and no consumer of the task may still be waiting then.
        if (cohort_block) {
            percounter_gm_fetch_add(cfg.sync_queue->completed, uint64_t{1});
            // As in the single-lane block: whoever raises the count is who can
            // tell the queue that what it was waiting for has happened.
            percounter_sync_promote(cfg);
        } else {
            percounter_gm_fetch_add(cfg.mix_tickets[PERCOUNTER_MIX_TICKET_DONE].next, uint64_t{1});
        }
        ++state->local_done;
    }
    return true;
}

/**
 * Serve the cluster's line if it names this lane under a generation it has not
 * seen yet. `served` reports whether it did.
 *
 * **The payload is read under a seqlock.** Between this lane's loads the claimer
 * may free the line and announce the next task, so re-reading the generation is
 * what proves the payload belongs to it. A generation discarded here is seen
 * again at the next checkpoint, and none can be missed: a line naming this lane
 * stays armed until this lane has reported finished on it.
 */
inline __aicore__ bool
percounter_mix_discover(const LaneConfig &cfg, PercounterLaneState *state, uint64_t timeout_cycles, bool *served) {
    *served = false;
    if (cfg.mix_order_count == 0) return true;
    PERCOUNTER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = percounter_ssbuf_load_acquire(&signal->armed_generation);
    if (!percounter_mix_armed(generation) || generation == state->last_served) return true;
    const int64_t task_id = signal->task_id;
    const uint32_t mask = signal->active_mask;
    const int32_t block_idx = signal->block_idx;
    const int32_t block_num = signal->block_num;
    const bool cohort_block = signal->cohort_block != 0;
    if (percounter_ssbuf_load_acquire(&signal->armed_generation) != generation) return true;

    state->last_served = generation;
    if ((mask & (uint32_t{1} << cfg.cluster_lane)) == 0) return true;
    *served = true;
    return percounter_serve_mix(
        cfg, generation, task_id, mask, block_idx, block_num, cohort_block, state, timeout_cycles
    );
}

/**
 * One checkpoint's service of work other lanes may be waiting on: a released
 * cohort block first, then the cluster's MIX line. `served` reports whether
 * this lane ran or armed anything.
 *
 * **The block goes first by default.** Asking the line first would run another
 * task's kernel while a co-holder sits at this cohort's barrier waiting for the
 * block this lane still owes.
 *
 * **Which cases are exceptions is decided one level down**, in
 * `percounter_sync_serve_released`: it is the only way a held block enters its
 * kernel, so the order holds for the last claimer's direct service too, and
 * this function needs to know nothing about it.
 */
inline __aicore__ bool
percounter_checkpoint_serve(const LaneConfig &cfg, PercounterLaneState *state, uint64_t timeout_cycles, bool *served) {
    if (!percounter_sync_serve_released(cfg, state, timeout_cycles, served)) return false;
    if (*served) return true;
    if (!percounter_mix_discover(cfg, state, timeout_cycles, served)) return false;
    return *served || percounter_mix_prepare(cfg, state);
}

/** Fold this lane's finished blocks into the shared progress count. */
inline __aicore__ void percounter_flush_done(const LaneConfig &cfg, PercounterLaneState *state) {
    if (state->local_done == 0) return;
    percounter_gm_fetch_add(cfg.control->completed_block_count, state->local_done);
    state->local_done = 0;
}

/**
 * Stay reachable after this lane's own list is spent, until every MIX
 * rendezvous and every cohort block is over, serving, taking cohort blocks
 * and -- on the claimer -- drawing as before.
 *
 * The SSBUF line is this cluster's alone, so it is read every iteration. The
 * shared GM words are read by every drained lane on the device, so they are
 * polled once per `PERCOUNTER_SPIN_CHECK_INTERVAL`: polling them every
 * iteration held a drained lane to ~1800 cycles a pass on `multi_root_1024`,
 * which starved the claimer and serialised 342 MIX tasks that could have run
 * 32 at a time. The `released` word a holder polls is the exception, and only
 * a lane holding a block reads it.
 */
inline __aicore__ void
percounter_mix_drain(const LaneConfig &cfg, PercounterLaneState *state, uint64_t timeout_cycles) {
    if (cfg.mix_order_count == 0 && cfg.sync_cohort_count == 0) return;
    // Restarted whenever this lane serves, and whenever the device completes a
    // rendezvous or a cohort block. What this lane waits for is device-wide, so
    // its own idleness is not evidence of a stall: a long task on another
    // cluster leaves every spent lane here with nothing to serve, and a
    // watchdog on that alone fails a run that is still advancing.
    uint64_t start = get_sys_cnt_aicore();
    uint64_t last_progress = ~uint64_t{0};
    uint32_t checks = 0;
    while (true) {
        bool served = false;
        if (!percounter_checkpoint_serve(cfg, state, timeout_cycles, &served)) return;
        percounter_mix_tick(cfg, state);
        if (served) {
            start = get_sys_cnt_aicore();
            checks = 0;
        }
        if (++checks == PERCOUNTER_SPIN_CHECK_INTERVAL) {
            checks = 0;
            percounter_flush_done(cfg, state);
            if (percounter_should_stop(cfg)) return;
            bool took_served = false;
            if (!percounter_sync_take(cfg, state, timeout_cycles, &took_served)) return;
            if (took_served) start = get_sys_cnt_aicore();
            uint64_t progress = 0;
            if (percounter_lane_may_exit(cfg, state, &progress)) return;
            if (progress != last_progress) {
                last_progress = progress;
                start = get_sys_cnt_aicore();
            }
            if (scheduler_watchdog_expired(start, get_sys_cnt_aicore(), timeout_cycles)) {
                percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
                return;
            }
        }
        SPIN_WAIT_HINT();
    }
}

/**
 * A lane draws one position at a time and keeps at most
 * PERCOUNTER_PENDING_CAPACITY candidates. It scans held positions before
 * drawing another, and rechecks every candidate while the window is blocked.
 * Unpublished Graph body entries remain candidates until the AICPU publishes.
 *
 * Every list uses the same descending (b-level, -id) key. The highest-key
 * incomplete task has no incomplete producer. A lane holding its position
 * finds it on the scan; if it is undrawn, every earlier position in that
 * group's share is complete, leaving a lane free to draw it. Graph publication
 * advances on the AICPU without waiting on any lane.
 *
 * A sync-start entry only queues a ready cohort. Its blocks run exclusively
 * through the global cohort protocol, which admits one incomplete cohort.
 */
inline __aicore__ void percounter_lane_loop(const LaneConfig &cfg, uint64_t timeout_cycles) {
    PercounterLaneState state{
        0, 0, PercounterMixClaim{-1, 0, 1, false, -1}, PercounterSyncState{0, 0, -1, -1, 0, 0, 0}
    };
    PercounterPending pending[PERCOUNTER_PENDING_CAPACITY];
    for (uint32_t slot = 0; slot < PERCOUNTER_PENDING_CAPACITY; ++slot)
        pending[slot] = PercounterPending{0, slot, false};
    uint32_t pending_count = 0;
    uint32_t spent_groups = 0;
    // Completion is monotone; all blocks of a task share the same fanin row.
    int64_t proven_task = -1;
    uint64_t wait_start = 0;
    uint64_t misses = 0;
    uint32_t checks = 0;
    while (true) {
        if (percounter_should_stop(cfg)) break;
        bool served = false;
        if (!percounter_checkpoint_serve(cfg, &state, timeout_cycles, &served)) break;
        if (!served && !percounter_sync_take(cfg, &state, timeout_cycles, &served)) break;
        if (served) {
            misses = 0;
            checks = 0;
            continue;
        }
        percounter_mix_tick(cfg, &state);

        uint32_t slot = 0;
        bool found = false;
        bool failed = false;
        uint64_t entry = 0;
        PercounterTask task{};
        for (; slot < pending_count; ++slot) {
            const uint64_t candidate_entry = percounter_gm_read(cfg.order[pending[slot].position]);
            if (percounter_order_block_num(candidate_entry) == 0) continue;
            PercounterTask candidate{};
            if (!percounter_resolve(cfg, percounter_order_task_id(candidate_entry), &candidate)) {
                failed = true;
                break;
            }
            if (candidate.id == proven_task || percounter_fanin_ready(cfg, candidate)) {
                task = candidate;
                entry = candidate_entry;
                found = true;
                break;
            }
        }
        if (failed) break;
        if (!found) {
            // Spend an otherwise idle pass preparing one published candidate.
            // Ready work wins the first scan; preparation never waits on fanin.
            for (uint32_t prepare_slot = 0; prepare_slot < pending_count; ++prepare_slot) {
                if (pending[prepare_slot].prepared) continue;
                const uint64_t prepare_entry = percounter_gm_read(cfg.order[pending[prepare_slot].position]);
                if (percounter_order_block_num(prepare_entry) == 0 || percounter_order_is_arming(prepare_entry))
                    continue;
                PercounterTask candidate{};
                if (!percounter_resolve(cfg, percounter_order_task_id(prepare_entry), &candidate) ||
                    !percounter_prepare_pending(cfg, &pending[prepare_slot], prepare_entry, candidate)) {
                    failed = true;
                }
                break;
            }
            if (failed) break;
            bool claimed = false;
            if (pending_count < PERCOUNTER_PENDING_CAPACITY) {
                while (spent_groups < cfg.ticket_group_count) {
                    const uint32_t group = (cfg.ticket_group + spent_groups) % cfg.ticket_group_count;
                    const uint64_t size = percounter_group_size(cfg.order_count, group, cfg.ticket_group_count);
                    if (size != 0) {
                        const uint64_t index = percounter_gm_fetch_add(cfg.tickets[group].next, uint64_t{1});
                        if (index < size) {
                            pending[pending_count].position =
                                percounter_group_position(index, group, cfg.ticket_group_count);
                            pending[pending_count].prepared = false;
                            ++pending_count;
                            claimed = true;
                            break;
                        }
                    }
                    ++spent_groups;
                }
            }
            if (claimed) continue;
            if (pending_count == 0) {
                percounter_flush_done(cfg, &state);
                percounter_mix_drain(cfg, &state, timeout_cycles);
                break;
            }
            // No candidate is ready and no new position fits or remains.
            // Each retry returns through the MIX/cohort checkpoints and scans
            // the entire window, including producers that became ready later.
            if (misses == 0) wait_start = get_sys_cnt_aicore();
            if (++misses > PERCOUNTER_SPIN_BACKOFF_AFTER) {
                const uint64_t until = get_sys_cnt_aicore() + PERCOUNTER_SPIN_BACKOFF_CYCLES;
                while (get_sys_cnt_aicore() < until) {}
            }
            if (++checks == PERCOUNTER_SPIN_CHECK_INTERVAL) {
                checks = 0;
                if (scheduler_watchdog_expired(wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
                    percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
                    break;
                }
            }
            SPIN_WAIT_HINT();
            continue;
        }
        if (!percounter_prepare_pending(cfg, &pending[slot], entry, task)) break;
        const uint32_t payload_slot = pending[slot].payload_slot;
        for (uint32_t next = slot + 1; next < pending_count; ++next)
            pending[next - 1] = pending[next];
        --pending_count;
        pending[pending_count] = PercounterPending{0, payload_slot, false};
        proven_task = task.id;
        misses = 0;
        checks = 0;
        if (percounter_order_is_arming(entry)) {
            percounter_sync_enqueue(cfg, task, percounter_order_block_num(entry));
            continue;
        }
        if (!percounter_run_prepared(cfg, task, cfg.payloads + payload_slot)) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
            break;
        }
        percounter_publish_completion(cfg, task, uint64_t{1});
        if (++state.local_done >= PERCOUNTER_COMPLETION_BATCH) percounter_flush_done(cfg, &state);
    }
    percounter_flush_done(cfg, &state);
}

/**
 * Resolve a lane's view of the region and run its loop.
 *
 * Every field goes through a device load: the block is DMA-written, and a lane
 * that once read `dispatch_payloads_offset` as 0 placed its payload inside the
 * ticket array.
 */
inline __aicore__ void run_percounter_lane(__gm__ PercounterControl *control, int block_idx, CoreType core_type) {
    const bool is_aic = core_type == CoreType::AIC;
    const uint32_t type_index = is_aic ? 0U : 1U;
    SchedulerClusterCoordinate coordinate{-1, -1};
    const uint64_t cluster_count = percounter_gm_read(control->cluster_count);
    const uint64_t aiv_per_cluster = percounter_gm_read(control->aiv_per_cluster);
    if (!scheduler_cluster_coordinate_from_worker(
            block_idx, is_aic, static_cast<int32_t>(cluster_count), static_cast<int32_t>(aiv_per_cluster), &coordinate
        )) {
        // A lane outside the topology the AICPU published has no payload slot.
        percounter_gm_compare_exchange(
            control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
        );
        return;
    }

    LaneConfig cfg{};
    cfg.control = control;
    const uint64_t storage_count = percounter_gm_read(control->graph_storage_count);
    cfg.graph = scheduler_ordinary_graph_view(
        percounter_gm_read(control->graph_storage_address), static_cast<uint32_t>(storage_count),
        storage_count > 0 ? storage_count - 1 : 0
    );
    cfg.callable_addresses =
        reinterpret_cast<__gm__ uint64_t *>(percounter_gm_read(control->callable_addresses_address));
    cfg.callable_count = static_cast<uint32_t>(percounter_gm_read(control->callable_addresses_count));
    cfg.payloads = scheduler_state_at<DispatchPayload>(
        control, percounter_gm_read(control->dispatch_payloads_offset) +
                     static_cast<uint64_t>(block_idx) * PERCOUNTER_PAYLOAD_SLOT_COUNT * sizeof(DispatchPayload)
    );
    cfg.tickets = scheduler_state_at<PercounterTicket>(
        control, percounter_gm_read(control->tickets_offset) +
                     static_cast<uint64_t>(type_index) * PERCOUNTER_TICKET_CAPACITY * sizeof(PercounterTicket)
    );
    cfg.order = scheduler_state_at<uint64_t>(control, percounter_gm_read(control->orders_offset[type_index]));
    cfg.fanin = scheduler_state_at<PercounterFanin>(control, percounter_gm_read(control->fanin_offset));
    cfg.fanin_addresses = scheduler_state_at<uint64_t>(control, percounter_gm_read(control->fanin_addr_offset));
    cfg.counters = scheduler_state_at<PercounterCounter>(control, percounter_gm_read(control->counters_offset));
    cfg.task_count = percounter_gm_read(control->task_count);
    cfg.table_task_count = percounter_gm_read(control->table_task_count);
    cfg.order_count = percounter_gm_read(control->order_count[type_index]);
    const uint64_t lanes = is_aic ? cluster_count : cluster_count * aiv_per_cluster;
    // Every lane of the type derives the same count from the topology. Grouping
    // by cluster keeps a cluster's two AIVs on one ticket.
    const uint64_t groups = lanes / PERCOUNTER_TICKET_GROUP_LANES;
    cfg.ticket_group_count = static_cast<uint32_t>(
        groups == 0 ? 1 : (groups > PERCOUNTER_TICKET_CAPACITY ? PERCOUNTER_TICKET_CAPACITY : groups)
    );
    cfg.ticket_group = static_cast<uint32_t>(coordinate.cluster_index) % cfg.ticket_group_count;
    cfg.has_predicates = percounter_gm_read(control->has_predicates) != 0;
    cfg.async_group = percounter_open_async_record(control, block_idx);
    cfg.cluster_index = coordinate.cluster_index;
    cfg.cluster_lane = coordinate.cluster_lane;

    const uint64_t timeout_cycles = percounter_gm_read(control->scheduler_timeout_cycles);
    cfg.sync_cohort_count = percounter_gm_read(control->sync_cohort_count);
    if (cfg.sync_cohort_count != 0) {
        cfg.sync_block_total = percounter_gm_read(control->sync_block_total);
        cfg.sync_queue =
            scheduler_state_at<PercounterSyncQueue>(control, percounter_gm_read(control->sync_cohort_offset));
        cfg.sync_pending = scheduler_state_at<uint64_t>(control, percounter_gm_read(control->sync_pending_offset));
    }
    cfg.mix_order_count = percounter_gm_read(control->mix_order_count);
    if (cfg.mix_order_count != 0) {
        cfg.mix_rendezvous_count = percounter_gm_read(control->mix_rendezvous_count);
        cfg.mix_order = scheduler_state_at<uint64_t>(control, percounter_gm_read(control->mix_order_offset));
        cfg.mix_tickets = scheduler_state_at<PercounterTicket>(control, percounter_gm_read(control->mix_ticket_offset));
        cfg.mix_signal_ready =
            scheduler_state_at<uint64_t>(control, percounter_gm_read(control->mix_signal_ready_offset));
        cfg.mix_signal = &percounter_ssbuf_region(platform_ssbuf_hardware_base_address())->mix;
        if (!percounter_clear_mix_signal(cfg, timeout_cycles)) return;
    }

    ChipSwimlaneAicoreLocalState profile{};
    if (SIMPLER_GET_DFX_FLAG(get_aicore_profiling_flag(), SIMPLER_DFX_FLAG_CHIP_SWIMLANE)) {
        cfg.profile_head = get_chip_swimlane_aicore_head();
        if (cfg.profile_head != nullptr) cfg.profile = &profile;
    }
    percounter_lane_loop(cfg, timeout_cycles);
    if (cfg.profile != nullptr) chip_swimlane_aicore_flush_task_count(&profile);
}

/**
 * A percounter AICore, from its report to the COND nonce acknowledgement.
 *
 * The same startup and retirement as the resident executor, minus the
 * scheduler bootstrap: the host already built every table, so a lane goes from
 * READY straight to the register release and into its loop.
 */
inline __aicore__ void percounter_aicore_execute(__gm__ Runtime *runtime, int block_idx, CoreType core_type) {
    __gm__ Handshake *handshake = (__gm__ Handshake *)(&runtime->dev.workers[block_idx]);
    __gm__ PercounterControl *control =
        reinterpret_cast<__gm__ PercounterControl *>(runtime->dev.scheduler_bootstrap.worker_context_base);
    percounter_observe_cache_line(handshake);

    const uint64_t report_epoch = get_aicore_report_epoch();
    handshake->work_start_cycle = 0;
    handshake->work_end_cycle = 0;
    if (report_epoch != 0) {
        aicore_stage_native_report(
            handshake, get_physical_core_id(), core_type, static_cast<uint32_t>(block_idx) + 1, 0
        );
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->report_epoch = report_epoch;
    } else {
        handshake->physical_core_id = get_physical_core_id();
        handshake->core_type = core_type;
        OUT_OF_ORDER_STORE_BARRIER();
        handshake->aicore_done = block_idx + 1;
    }
    dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);

    // The host prefills the timeout, so it bounds this wait before the AICPU
    // has written anything.
    percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
    const uint64_t timeout_cycles = percounter_gm_read(control->scheduler_timeout_cycles);
    bool released = false;
    uint32_t signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
    const uint64_t ready_wait_start = get_sys_cnt_aicore();
    while (handshake->aicpu_ready != SCHEDULER_RUNTIME_MODE_PERCOUNTER && signal != AICORE_EXIT_SIGNAL) {
        percounter_observe_cache_line(handshake);
        if (scheduler_watchdog_expired(ready_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
            percounter_gm_compare_exchange(
                control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)
            );
            signal = AICORE_EXIT_SIGNAL;
            break;
        }
        signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        SPIN_WAIT_HINT();
    }
    if (signal != AICORE_EXIT_SIGNAL) {
        // The AICPU published the topology before READY; re-observe the block.
        percounter_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
        const uint64_t release_wait_start = get_sys_cnt_aicore();
        while (signal != AICPU_IDLE_TASK_ID && signal != AICORE_EXIT_SIGNAL) {
            if (scheduler_watchdog_expired(release_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
                percounter_gm_compare_exchange(
                    control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)
                );
                break;
            }
            signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
            SPIN_WAIT_HINT();
        }
        released = signal == AICPU_IDLE_TASK_ID;
    }
    if (released) {
        const uint64_t work_start = get_sys_cnt_aicore();
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        run_percounter_lane(control, block_idx, core_type);
        const uint64_t work_end = get_sys_cnt_aicore();
        handshake->work_start_cycle = work_start;
        handshake->work_end_cycle = work_end;
        dcci(handshake, SINGLE_CACHE_LINE, CACHELINE_OUT);
        dsb((mem_dsb_t)0);
    }
    // A lane arrives here when its share of the run is done, which can be long
    // before the run ends, or after a local watchdog ended a wait above. The
    // AICPU sends EXIT on every outcome once its supervisor returns, and the
    // supervisor has its own progress watchdog, so this wait needs none. EXIT
    // is published only after the return gate is reset, so wait for it before
    // acknowledging, and complete the preceding GM writes first.
    wait_for_aicpu_exit_signal();
    OUT_OF_ORDER_STORE_BARRIER();
    acknowledge_aicore_cond_nonce_and_wait(&runtime->dev.teardown_gates[block_idx]);
}

}  // namespace simpler::percounter
