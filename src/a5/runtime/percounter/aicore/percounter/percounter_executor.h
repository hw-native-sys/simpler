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
 * A lane draws a range of positions from its core type's ticket, spins on each
 * task's producer counters until they read published, runs the kernel, and
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
#include "runtime.h"
#include "scheduler/percounter/percounter_types.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_layout.h"
#include "scheduler/scheduler_memory.h"
#include "scheduler/scheduler_ready.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_watchdog.h"

namespace simpler::hbg::percounter {

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
    return scheduler_gm_query(*word);
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
    __gm__ PercounterControl *control;
    SchedulerGraphView graph;
    __gm__ uint64_t *callable_addresses;
    uint32_t callable_count;
    __gm__ DispatchPayload *payload;
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
    // The size of this lane's first draw: one on a list short enough that a
    // first range could hold most of a lane's share (see
    // PERCOUNTER_SHORT_LIST_PER_LANE), PERCOUNTER_CLAIM_BATCH otherwise.
    uint64_t first_batch;
    // Whether any task carries a dispatch predicate; a graph without one never
    // reads the predicate words.
    bool has_predicates;

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
    SCHEDULER_SSBUF PercounterMixSignal *mix_signal;
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
};

/** What a lane carries between its loop and the waits it makes from it. */
struct PercounterLaneState {
    // Finished blocks not yet folded into `resolved_task_count`.
    uint64_t local_done;
    // The last MIX generation this lane has looked at, served or not.
    uint32_t last_served;
    PercounterMixClaim claim;
    PercounterSyncState sync;
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
    const uint64_t record = scheduler_gm_query_u32_pair(reinterpret_cast<__gm__ uint32_t *>(&cfg.fanin[task_id]));
    PercounterFaninView view{};
    view.count = static_cast<int32_t>(record & 0xffffffffULL);
    view.addresses = cfg.fanin_addresses + static_cast<int32_t>(record >> 32);
    return view;
}

inline __aicore__ void percounter_fail(const LaneConfig &cfg, SchedulerGraphResult result) {
    // First error wins.
    scheduler_gm_compare_exchange(cfg.control->lane_error, uint64_t{0}, static_cast<uint64_t>(result));
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
    scheduler_observe_cache_line(line);
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
    task->graph = SchedulerGraphView{storage, 0, static_cast<uint64_t>(body_index) + 1, 0};
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
    const uint64_t value = scheduler_gm_query(task.fanin[i]);
    if (!task.body_fanin) return value;
    return reinterpret_cast<uint64_t>(&cfg.counters[static_cast<uint64_t>(task.body_base) + value].value);
}

/**
 * One pass over `task`'s producers: true when every one has published.
 *
 * Never blocks, which is what lets a lane hold a range without serialising on
 * its head. It stops at the first unpublished producer, so an unready task
 * costs as little as one load and a ready one costs one per producer.
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
 * Run `kernel_id` as subtask `subtask_slot` of one block of `task_id`.
 * `kernel_start` receives the timestamp before the kernel call.
 *
 * The caller has already invalidated the data cache: kernels load GM with plain
 * scalar reads and never invalidate first, so a line this core cached during an
 * earlier task would survive another core rewriting it. The legacy executor
 * invalidates the same way on every dispatch.
 *
 * **A predicate decides whether the kernel runs, not whether the task
 * completes.** A false one returns as a task with no kernel does, and the caller
 * publishes either way, so consumers are released all the same. Its operand's
 * producer is in the task's fanin, which has published, so the value is final.
 *
 * Returns false when the predicate is malformed, or when the callable or the
 * payload cannot be resolved.
 */
inline __aicore__ bool percounter_launch_kernel(
    const LaneConfig &cfg, const PercounterTask &task, int32_t subtask_slot, int32_t kernel_id, int32_t block_idx,
    int32_t block_num, uint64_t *kernel_start
) {
    if (cfg.has_predicates) {
        const SchedulerPredicateResult verdict = scheduler_evaluate_task_predicate(task.graph, task.local);
        if (verdict == SchedulerPredicateResult::MALFORMED) {
            // First error wins, so the caller's INVALID_CALLABLE does not replace it.
            percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
            return false;
        }
        if (verdict == SchedulerPredicateResult::FAIL) return true;
    }

    uint64_t function_address = 0;
    if (!scheduler_lookup_callable_address(
            cfg.callable_addresses, cfg.callable_count, static_cast<uint16_t>(kernel_id), &function_address
        )) {
        return false;
    }
    const SchedulerTaskInfo info{
        task.local, kernel_id, subtask_slot, subtask_slot == 0 ? CoreType::AIC : CoreType::AIV
    };
    if (scheduler_materialize_task_payload_resolved(task.graph, info, function_address, cfg.payload) !=
        SchedulerGraphResult::OK) {
        return false;
    }
    // Written for every block, a single-block task included: the payload is
    // this lane's for the whole run, so the previous block's values are still
    // in it. Both fields share the control line with `function_bin_addr`.
    cfg.payload->local_context.block_idx = block_idx;
    cfg.payload->local_context.block_num = block_num;
    cfg.payload->global_context.sub_block_id = subtask_slot == 2 ? 1 : 0;
    scheduler_writeback_dispatch_payload(cfg.payload);

    // The barrier is the publication fence a consumer's spin depends on: the
    // kernel's outputs are in memory before the atomic that announces them.
    PercounterKernelFunc kernel = reinterpret_cast<PercounterKernelFunc>(cfg.payload->function_bin_addr);
    *kernel_start = get_sys_cnt_aicore();
    kernel(reinterpret_cast<__gm__ int64_t *>(cfg.payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
    return true;
}

/**
 * Run one block drawn from an order list. The task's single active subtask is
 * whichever kernel slot is set; a task with none (a dependency-only task) runs
 * nothing, leaves `kernel_start` untouched, and is published like any other.
 */
inline __aicore__ bool percounter_run_task(
    const LaneConfig &cfg, const PercounterTask &task, int32_t block_idx, int32_t block_num, uint64_t *kernel_start
) {
    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(cfg.payload));
    __gm__ int32_t *kernel_ids = percounter_kernel_ids(task);
    for (int32_t slot = 0; slot < PLATFORM_CORES_PER_BLOCKDIM; ++slot) {
        const int32_t kernel_id = kernel_ids[slot];
        if (kernel_id >= 0)
            return percounter_launch_kernel(cfg, task, slot, kernel_id, block_idx, block_num, kernel_start);
    }
    return true;
}

/** Run this lane's own subtask of one block of a MIX task. */
inline __aicore__ bool percounter_run_subtask(
    const LaneConfig &cfg, const PercounterTask &task, int32_t subtask_slot, int32_t block_idx, int32_t block_num,
    uint64_t *kernel_start
) {
    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(cfg.payload));
    const int32_t kernel_id = percounter_kernel_ids(task)[subtask_slot];
    if (kernel_id < 0) return true;
    return percounter_launch_kernel(cfg, task, subtask_slot, kernel_id, block_idx, block_num, kernel_start);
}

/**
 * Record `count` finished subtasks of `task`; called after their store
 * barriers. A body sink's subtasks also count toward its shell, whose counter
 * is preset against every sink subtask of the body.
 */
inline __aicore__ void
percounter_publish_completion(const LaneConfig &cfg, const PercounterTask &task, uint64_t count) {
    scheduler_gm_fetch_add(cfg.counters[task.id].value, count);
    if (task.sink) scheduler_gm_fetch_add(cfg.counters[task.shell].value, count);
}

/** True when the lane should stop: EXIT is signalled or some lane has failed. */
inline __aicore__ bool percounter_should_stop(const LaneConfig &cfg) {
    return static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE)) == AICORE_EXIT_SIGNAL ||
           scheduler_gm_query(cfg.control->lane_error) != 0;
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
        scheduler_gm_publish(*ready, UINT64_C(1));
        return true;
    }
    const uint64_t start = get_sys_cnt_aicore();
    uint32_t checks = 0;
    while (scheduler_gm_query(*ready) == 0) {
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
    scheduler_observe_cache_line(kernel_ids);
    uint32_t mask = 0;
    for (uint32_t slot = 0; slot < PLATFORM_CORES_PER_BLOCKDIM; ++slot) {
        if (kernel_ids[slot] >= 0) mask |= uint32_t{1} << slot;
    }
    return mask;
}

/**
 * Free the cluster's line once every participant has finished.
 *
 * The claimer's, not the lowest participant's: mask `0b101` omits AIV0, and
 * SSBUF has no atomic to arbitrate a second writer of the generation.
 */
inline __aicore__ void percounter_mix_release(const LaneConfig &cfg) {
    SCHEDULER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = scheduler_ssbuf_load_acquire(&signal->armed_generation);
    if (!percounter_mix_armed(generation)) return;
    const uint32_t mask = signal->active_mask;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if ((mask & (uint32_t{1} << lane)) != 0 && signal->finished[lane] != generation) return;
    }
    scheduler_ssbuf_store_release(&signal->armed_generation, generation + 1);
}

/**
 * Sync-start cohorts: a task whose `block_num` blocks must all start together.
 *
 * Such a task holds one arming entry in its list instead of one per block. The
 * lane that runs the entry queues the task; the queue promotes cohorts one at a
 * time into its open slot, and free lanes of the right kind each take one block
 * of the open cohort with a fetch_add. The lane that takes the last block marks
 * the cohort released and promotes the next one, and every holder of a block of
 * a released cohort runs it.
 *
 * **A lane serves a released block before it looks at anything else.** A holder
 * that joined another rendezvous first -- one that needs a lane already inside
 * this cohort's kernel, waiting for this block -- would never come back. Every
 * checkpoint therefore asks `percounter_sync_serve_released` before MIX
 * discovery. Taking a block never blocks, and a holder never takes a second.
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
    const uint64_t generation = scheduler_gm_query(q->open_generation);
    if (percounter_sync_armed(generation)) return;
    const uint64_t seq = generation / 2;
    if (seq >= scheduler_gm_query(q->tail)) return;
    const uint64_t desc = scheduler_gm_query(cfg.sync_pending[seq]);
    // Reserved but not yet written: its own enqueuer will promote it.
    if (desc == 0) return;
    if (scheduler_gm_compare_exchange(q->open_generation, generation, generation + 1) != generation) return;

    scheduler_gm_store(q->open_block_num, percounter_sync_desc_blocks(desc));
    scheduler_gm_store(q->open_active_mask, percounter_sync_desc_mask(desc));
    scheduler_gm_store(q->open_seq, seq);
    scheduler_gm_store(q->open_task_id, percounter_sync_desc_task(desc));
}

/**
 * Queue `task_id` for gathering. Never fails and is never retried: the
 * sequence number is taken first, and the trailing promote covers the slot
 * falling free between this lane's two steps.
 */
inline __aicore__ void percounter_sync_enqueue(const LaneConfig &cfg, const PercounterTask &task, int32_t block_num) {
    const uint32_t mask = percounter_mix_active_mask(task);
    const uint64_t seq = scheduler_gm_fetch_add(cfg.sync_queue->tail, uint64_t{1});
    scheduler_gm_store(cfg.sync_pending[seq], percounter_sync_desc(task.id, static_cast<uint32_t>(block_num), mask));
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
    if (scheduler_gm_query(*claims) >= block_num) return -1;
    const uint64_t prev = scheduler_gm_fetch_add(*claims, uint64_t{1});
    return prev < block_num ? static_cast<int32_t>(prev) : -1;
}

/**
 * Run the block this lane holds, now that its cohort is released.
 *
 * A MIX cohort block arms this lane's own cluster instead, with `cohort_block`
 * set so the rendezvous reports to the queue rather than to the MIX list. The
 * line is acquired by polling: an armed line frees in bounded time, since every
 * participant it names is already gathering. `my_block` stays set until the
 * block is run or armed.
 */
inline __aicore__ bool percounter_sync_serve_block(const LaneConfig &cfg, PercounterLaneState *state) {
    PercounterSyncState *sync = &state->sync;
    if ((sync->active_mask & (sync->active_mask - 1U)) == 0) {
        PercounterTask task{};
        if (!percounter_resolve(cfg, sync->task_id, &task)) return false;
        uint64_t kernel_start = 0;
        if (!percounter_run_task(cfg, task, sync->my_block, static_cast<int32_t>(sync->block_num), &kernel_start)) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
            return false;
        }
        percounter_publish_completion(cfg, task, uint64_t{1});
        ++state->local_done;
        sync->my_block = -1;
        scheduler_gm_fetch_add(cfg.sync_queue->completed, uint64_t{1});
        return true;
    }

    SCHEDULER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = scheduler_ssbuf_load_acquire(&signal->armed_generation);
    if (percounter_mix_armed(generation)) return true;
    signal->task_id = sync->task_id;
    signal->active_mask = sync->active_mask;
    signal->block_idx = sync->my_block;
    signal->block_num = static_cast<int32_t>(sync->block_num);
    signal->cohort_block = 1;
    scheduler_ssbuf_store_release(&signal->armed_generation, generation + 1);
    sync->my_block = -1;
    return true;
}

/**
 * Serve the block this lane holds if its cohort has been released; `served`
 * reports whether a block was run or armed. Checked at every checkpoint, ahead
 * of MIX discovery, and costs one load only while a block is held.
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
    // Cohorts are promoted in sequence order, so this one is fully claimed
    // exactly when the counter passes it.
    if (scheduler_gm_query(cfg.sync_queue->released) > sync->my_seq) {
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
 * decided about it. Taking the last block releases the cohort, promotes the
 * next one, and serves at once.
 *
 * Never blocks: the lane goes back to its own list holding no order position,
 * which keeps cohorts out of the claim range's deadlock argument.
 */
inline __aicore__ bool
percounter_sync_take(const LaneConfig &cfg, PercounterLaneState *state, uint64_t timeout_cycles, bool *served) {
    *served = false;
    PercounterSyncState *sync = &state->sync;
    if (cfg.sync_cohort_count == 0 || sync->my_block >= 0) return true;
    const uint64_t generation = scheduler_gm_query(cfg.sync_queue->open_generation);
    // Before any further load: a lane that has decided reaches here at every
    // checkpoint while the cohort is open, and reading the payload first cost
    // 40% of `sparse_256_b24`.
    if (!percounter_sync_armed(generation) || sync->decided_generation == generation) return true;
    const int64_t task_id = scheduler_gm_query(cfg.sync_queue->open_task_id);
    // Slot taken, payload not yet published: deciding now would read the
    // previous cohort's shape, and a decision is final.
    if (task_id < 0) return true;
    if (task_id >= static_cast<int64_t>(cfg.task_count)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_TASK_ID);
        return false;
    }
    const uint64_t seq = scheduler_gm_query(cfg.sync_queue->open_seq);
    const uint64_t shape = scheduler_gm_query_u32_pair(&cfg.sync_queue->open_block_num);
    // A seqlock: the slot may have been released and refilled between these
    // loads, and the task, sequence and shape must all be one cohort's.
    if (scheduler_gm_query(cfg.sync_queue->open_generation) != generation) return true;
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
    if (static_cast<uint32_t>(block) + 1 < sync->block_num) return true;

    // The last block. `released` goes out before the slot is freed: it is what
    // the other holders wait on, and the slot is about to belong to another.
    __gm__ PercounterSyncQueue *q = cfg.sync_queue;
    scheduler_gm_store(q->open_task_id, int64_t{-1});
    scheduler_gm_store(q->released, seq + 1);
    scheduler_gm_store(q->open_generation, generation + 1);
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
    SCHEDULER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = scheduler_ssbuf_load_acquire(&signal->armed_generation);
    if (percounter_mix_armed(generation)) return;

    if (claim->task_id < 0) {
        if (claim->position < 0) {
            if (claim->list_spent) return;
            const uint64_t draw = scheduler_gm_fetch_add(cfg.mix_tickets[PERCOUNTER_MIX_TICKET_DRAW].next, uint64_t{1});
            if (draw >= cfg.mix_order_count) {
                claim->list_spent = true;
                return;
            }
            claim->position = static_cast<int64_t>(draw);
        }
        // A Graph body's entry reads zero until the AICPU has materialized it.
        const uint64_t entry = scheduler_gm_query(cfg.mix_order[claim->position]);
        if (percounter_order_block_num(entry) == 0) return;
        claim->position = -1;
        claim->task_id = percounter_order_task_id(entry);
        claim->block_idx = percounter_order_block_idx(entry);
        claim->block_num = percounter_order_block_num(entry);
    }
    PercounterTask task{};
    if (!percounter_resolve(cfg, claim->task_id, &task)) return;
    if (!percounter_fanin_ready(cfg, task)) return;

    // A sync-start MIX task's single arming entry queues the cohort instead of
    // arming this cluster; its blocks are handed out from the queue.
    if (claim->block_idx == PERCOUNTER_ORDER_BLOCK_IDX_ARM) {
        percounter_sync_enqueue(cfg, task, claim->block_num);
        claim->task_id = -1;
        return;
    }

    // The payload before the generation, and the generation with release
    // ordering: a lane that observes an odd generation observes its payload.
    signal->task_id = claim->task_id;
    signal->active_mask = percounter_mix_active_mask(task);
    signal->block_idx = claim->block_idx;
    signal->block_num = claim->block_num;
    // Every arming: the previous rendezvous on this line may have been a
    // cohort block.
    signal->cohort_block = 0;
    scheduler_ssbuf_store_release(&signal->armed_generation, generation + 1);
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
 */
inline __aicore__ bool percounter_lane_may_exit(const LaneConfig &cfg, const PercounterLaneState *state) {
    if (state->sync.my_block >= 0) return false;
    if (cfg.mix_order_count != 0 &&
        scheduler_gm_query(cfg.mix_tickets[PERCOUNTER_MIX_TICKET_DONE].next) < cfg.mix_rendezvous_count)
        return false;
    return cfg.sync_cohort_count == 0 || scheduler_gm_query(cfg.sync_queue->completed) >= cfg.sync_block_total;
}

/**
 * Spin until every participant's token has reached `generation`.
 *
 * `tokens` is `arrived` on the way in and `finished` on the way out. Returns
 * false on EXIT, on another lane's error, or on the watchdog -- what a lane
 * spinning here can still see and a lane inside a kernel cannot, which is why
 * the gathering is a software spin rather than a channel wait.
 */
inline __aicore__ bool percounter_mix_wait(
    const LaneConfig &cfg, SCHEDULER_SSBUF volatile uint32_t *tokens, uint32_t mask, uint32_t generation,
    uint64_t timeout_cycles
) {
    const uint64_t start = get_sys_cnt_aicore();
    uint32_t checks = 0;
    while (true) {
        bool complete = true;
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if ((mask & (uint32_t{1} << lane)) != 0 && scheduler_ssbuf_load_acquire(&tokens[lane]) != generation) {
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
    SCHEDULER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t lane = static_cast<uint32_t>(cfg.cluster_lane);
    PercounterTask task{};
    if (!percounter_resolve(cfg, task_id, &task)) return false;
    if (!percounter_fanin_ready(cfg, task)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_ARGUMENTS);
        return false;
    }

    scheduler_ssbuf_store_release(&signal->arrived[lane], generation);
    if (!percounter_mix_wait(cfg, signal->arrived, mask, generation, timeout_cycles)) return false;
    // Inside the kernel this lane polls nothing. Every participant has arrived,
    // so a cross-core channel wait in it has a peer to meet.
    uint64_t kernel_start = 0;
    if (!percounter_run_subtask(cfg, task, cfg.cluster_lane, block_idx, block_num, &kernel_start)) {
        percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
        return false;
    }
    // After the kernel's store barrier, so a peer that observes this token has
    // observed the subtask's outputs.
    scheduler_ssbuf_store_release(&signal->finished[lane], generation);

    if ((uint32_t{1} << lane) == (mask & (~mask + 1U))) {
        if (!percounter_mix_wait(cfg, signal->finished, mask, generation, timeout_cycles)) return false;
        // The counter was preset against the task's whole subtask total, so
        // this add carries the block's share of it at once.
        percounter_publish_completion(cfg, task, static_cast<uint64_t>(__builtin_popcount(mask)));
        // After the task counter: a lane that sees either count reach its total
        // may stop, and no consumer of the task may still be waiting then.
        if (cohort_block) scheduler_gm_fetch_add(cfg.sync_queue->completed, uint64_t{1});
        else scheduler_gm_fetch_add(cfg.mix_tickets[PERCOUNTER_MIX_TICKET_DONE].next, uint64_t{1});
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
    SCHEDULER_SSBUF PercounterMixSignal *signal = cfg.mix_signal;
    const uint32_t generation = scheduler_ssbuf_load_acquire(&signal->armed_generation);
    if (!percounter_mix_armed(generation) || generation == state->last_served) return true;
    const int64_t task_id = signal->task_id;
    const uint32_t mask = signal->active_mask;
    const int32_t block_idx = signal->block_idx;
    const int32_t block_num = signal->block_num;
    const bool cohort_block = signal->cohort_block != 0;
    if (scheduler_ssbuf_load_acquire(&signal->armed_generation) != generation) return true;

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
 */
inline __aicore__ bool
percounter_checkpoint_serve(const LaneConfig &cfg, PercounterLaneState *state, uint64_t timeout_cycles, bool *served) {
    if (!percounter_sync_serve_released(cfg, state, timeout_cycles, served)) return false;
    if (*served) return true;
    return percounter_mix_discover(cfg, state, timeout_cycles, served);
}

/** Fold this lane's finished blocks into the shared progress count. */
inline __aicore__ void percounter_flush_done(const LaneConfig &cfg, PercounterLaneState *state) {
    if (state->local_done == 0) return;
    scheduler_gm_fetch_add(cfg.control->resolved_task_count, state->local_done);
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
    // Restarted whenever the lane serves, so the watchdog measures idleness.
    uint64_t start = get_sys_cnt_aicore();
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
            if (percounter_lane_may_exit(cfg, state)) return;
            if (scheduler_watchdog_expired(start, get_sys_cnt_aicore(), timeout_cycles)) {
                percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
                return;
            }
        }
        SPIN_WAIT_HINT();
    }
}

/**
 * Block until every producer of `task_id` has published.
 *
 * This is also a lane's second MIX checkpoint, and the protocol cannot do
 * without it: a lane blocked here is the one a rendezvous is most likely to be
 * waiting for, and its producer can be a task only that rendezvous runs. The
 * SSBUF read is taken every iteration, since a checkpoint on the interval below
 * would put up to 16384 cycles between a rendezvous and a peer noticing it.
 *
 * Returns false on EXIT, on another lane's error, or when this producer has
 * been late past the watchdog, which it records as this lane's error.
 */
inline __aicore__ bool percounter_wait_for_fanin(
    const LaneConfig &cfg, const PercounterTask &task, PercounterLaneState *state, uint64_t timeout_cycles
) {
    for (int32_t i = 0; i < task.fanin_count; ++i) {
        const uint64_t address = percounter_producer(cfg, task, i);
        uint64_t start = get_sys_cnt_aicore();
        uint32_t checks = 0;
        uint32_t sync_checks = 0;
        uint64_t misses = 0;
        while (!percounter_counter_published(percounter_observe(address))) {
            bool served = false;
            if (!percounter_checkpoint_serve(cfg, state, timeout_cycles, &served)) return false;
            // A cohort fills only as fast as free lanes notice it, and a lane
            // blocked here is often one that has not taken a block yet; at the
            // 64-iteration interval with backoff it would look once per ~16k
            // cycles. Eight keeps the shared read an eighth of every-iteration.
            if (!served && cfg.sync_cohort_count != 0 && ++sync_checks >= PERCOUNTER_SYNC_CHECK_INTERVAL) {
                sync_checks = 0;
                if (!percounter_sync_take(cfg, state, timeout_cycles, &served)) return false;
            }
            if (served) {
                // The watchdog measures how late the producer is, and a
                // rendezvous or a cohort block is time this lane spent
                // running a task.
                start = get_sys_cnt_aicore();
                misses = 0;
                checks = 0;
                continue;
            }
            if (++misses > PERCOUNTER_SPIN_BACKOFF_AFTER) {
                const uint64_t until = get_sys_cnt_aicore() + PERCOUNTER_SPIN_BACKOFF_CYCLES;
                while (get_sys_cnt_aicore() < until) {}
            }
            if (++checks == PERCOUNTER_SPIN_CHECK_INTERVAL) {
                checks = 0;
                if (percounter_should_stop(cfg)) return false;
                percounter_mix_tick(cfg, state);
                if (scheduler_watchdog_expired(start, get_sys_cnt_aicore(), timeout_cycles)) {
                    percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
                    return false;
                }
            }
            SPIN_WAIT_HINT();
        }
    }
    return true;
}

/**
 * Block until the AICPU publishes the entry at `slot`, which is in a Graph
 * body's range, and return it in `entry`.
 *
 * The same checkpoints as the fanin wait, for the same reason: this lane may be
 * what a rendezvous or a cohort is waiting for. The AICPU materializes without
 * waiting on any lane, so the entry comes in bounded time; a failure there
 * reaches this lane as EXIT once the supervisor ends the run.
 */
inline __aicore__ bool percounter_wait_for_entry(
    const LaneConfig &cfg, __gm__ uint64_t *slot, PercounterLaneState *state, uint64_t timeout_cycles, uint64_t *entry
) {
    uint64_t start = get_sys_cnt_aicore();
    uint32_t checks = 0;
    uint64_t misses = 0;
    while (true) {
        const uint64_t value = scheduler_gm_query(*slot);
        if (percounter_order_block_num(value) != 0) {
            *entry = value;
            return true;
        }
        bool served = false;
        if (!percounter_checkpoint_serve(cfg, state, timeout_cycles, &served)) return false;
        if (served) {
            start = get_sys_cnt_aicore();
            misses = 0;
            checks = 0;
            continue;
        }
        if (++misses > PERCOUNTER_SPIN_BACKOFF_AFTER) {
            const uint64_t until = get_sys_cnt_aicore() + PERCOUNTER_SPIN_BACKOFF_CYCLES;
            while (get_sys_cnt_aicore() < until) {}
        }
        if (++checks == PERCOUNTER_SPIN_CHECK_INTERVAL) {
            checks = 0;
            if (percounter_should_stop(cfg)) return false;
            percounter_mix_tick(cfg, state);
            if (!percounter_sync_take(cfg, state, timeout_cycles, &served)) return false;
            if (served) start = get_sys_cnt_aicore();
            if (scheduler_watchdog_expired(start, get_sys_cnt_aicore(), timeout_cycles)) {
                percounter_fail(cfg, SchedulerGraphResult::TIMEOUT);
                return false;
            }
        }
        SPIN_WAIT_HINT();
    }
}

/**
 * The lane loop.
 *
 * A lane draws from its group's ticket, and from the other groups' once its
 * own is spent (see PERCOUNTER_TICKET_CAPACITY); a draw yields a range of
 * consecutive indices of that group's share, which the lane maps to list
 * positions as it reads them. Whichever lane of a group is free takes the
 * group's next range, and nothing is bound to a lane in advance. Deadlock
 * freedom follows from the list order -- see percounter_claim_range.
 *
 * Within its range the lane takes whichever position is ready rather than the
 * next in order, and blocks only when none is. A range is otherwise
 * head-of-line blocked by its own first task: `mixed_random_1024`, whose tasks
 * each wait on up to four producers, lost 53% going from a two-wide range to a
 * four-wide one for exactly that reason.
 *
 * `pending` is which positions are left to run, bit 0 the lowest. Readiness is
 * never cached in it: it belongs to other cores and is re-read every round.
 * With `BATCH = 2`, a range whose first task A waits and whose second B is ready:
 *
 *   pending=0b11  probe A: no, probe B: yes   -> run B, pending=0b01
 *   pending=0b01  probe A: no, scan empty     -> block on A (lowest)
 *   pending=0b01  A published                 -> run A, pending=0b00
 *   pending=0b00                              -> draw the next range
 *
 * **The fallback takes the lowest pending position.** A higher one can have a
 * producer that is a lower position in this same range, which only this lane
 * can run. The lowest cannot: every position before it in the list is complete
 * or held by another lane, so it is the one the deadlock argument names.
 *
 * A position in a Graph body's range may not be published yet. The scan passes
 * over it, and the fallback waits for it: the AICPU publishes it without
 * waiting on any lane, so the argument is unchanged.
 */
inline __aicore__ void percounter_lane_loop(const LaneConfig &cfg, uint64_t timeout_cycles) {
    PercounterLaneState state{
        0, 0, PercounterMixClaim{-1, 0, 1, false, -1}, PercounterSyncState{0, 0, -1, -1, 0, 0, 0}
    };
    uint64_t range_first = 0;
    uint32_t pending = 0;
    // A range withholds its positions from every other lane, which pays with
    // short kernels (fewer ticket atomics) and costs with long ones: on qwen3
    // decode a lane drew ahead into a 110-task fan-out and the wave ran on 24 of
    // 32 AIC lanes. So a lane whose last kernel was long draws one at a time.
    uint64_t batch = cfg.first_batch;
    // The group the current range was drawn from, and how many groups past its
    // own this lane has found spent: it moves on in order and never comes back.
    uint32_t range_group = cfg.ticket_group;
    uint32_t spent_groups = 0;
    // The task this range has already proved ready. An SPMD task puts several
    // of its blocks in one range, and they share a fanin row, so without this
    // the scan re-reads every producer once per block. Exact rather than a
    // heuristic: completion is monotone, so a task proved ready stays ready.
    int64_t proven_task = -1;
    static_assert(
        PERCOUNTER_CLAIM_BATCH <= 32, "the pending set is a uint32_t bitmask, one bit per position in the range"
    );

    while (true) {
        if (percounter_should_stop(cfg)) break;
        bool served = false;
        if (!percounter_checkpoint_serve(cfg, &state, timeout_cycles, &served)) break;
        if (served) continue;
        if (!percounter_sync_take(cfg, &state, timeout_cycles, &served)) break;
        if (served) continue;
        percounter_mix_tick(cfg, &state);

        if (pending == 0) {
            // Own group first, then each other group once its predecessor is
            // spent. Draws keep rising past a share's end, so a spent group
            // stays spent; every lane leaves once all are, through the drain
            // when the graph has MIX or sync-start tasks.
            uint64_t range_end = 0;
            bool claimed = false;
            while (spent_groups < cfg.ticket_group_count) {
                range_group = (cfg.ticket_group + spent_groups) % cfg.ticket_group_count;
                const uint64_t size = percounter_group_size(cfg.order_count, range_group, cfg.ticket_group_count);
                if (size != 0) {
                    const uint64_t base = scheduler_gm_fetch_add(cfg.tickets[range_group].next, batch);
                    if (percounter_claim_range(base, batch, size, &range_first, &range_end)) {
                        claimed = true;
                        break;
                    }
                }
                ++spent_groups;
            }
            if (!claimed) {
                percounter_flush_done(cfg, &state);
                percounter_mix_drain(cfg, &state, timeout_cycles);
                break;
            }
            pending = static_cast<uint32_t>((uint64_t{1} << (range_end - range_first)) - 1);
            proven_task = -1;
        }

        uint32_t slot = 0;
        bool found = false;
        PercounterTask task{};
        uint64_t entry = 0;
        bool failed = false;
        for (uint32_t probe = pending; probe != 0; probe &= probe - 1) {
            const uint32_t candidate = static_cast<uint32_t>(__builtin_ctz(probe));
            const uint64_t candidate_entry = scheduler_gm_query(
                cfg.order[percounter_group_position(range_first + candidate, range_group, cfg.ticket_group_count)]
            );
            // A Graph body's entry reads zero until the AICPU publishes it.
            if (percounter_order_block_num(candidate_entry) == 0) continue;
            PercounterTask candidate_task{};
            if (!percounter_resolve(cfg, percounter_order_task_id(candidate_entry), &candidate_task)) {
                failed = true;
                break;
            }
            if (candidate_task.id == proven_task || percounter_fanin_ready(cfg, candidate_task)) {
                slot = candidate;
                task = candidate_task;
                entry = candidate_entry;
                found = true;
                break;
            }
        }
        if (failed) break;
        if (!found) {
            slot = static_cast<uint32_t>(__builtin_ctz(pending));
            __gm__ uint64_t *slot_entry =
                &cfg.order[percounter_group_position(range_first + slot, range_group, cfg.ticket_group_count)];
            if (!percounter_wait_for_entry(cfg, slot_entry, &state, timeout_cycles, &entry)) break;
            if (!percounter_resolve(cfg, percounter_order_task_id(entry), &task)) break;
        }
        pending &= ~(uint32_t{1} << slot);
        // Only the blocking path waits: the scan above already read a ready
        // task's producers to the end.
        if (!found && !percounter_wait_for_fanin(cfg, task, &state, timeout_cycles)) break;
        proven_task = task.id;

        // A sync-start task's single entry queues the cohort once its fanin is
        // ready. Its blocks are counted where they run, not here.
        if (percounter_order_is_arming(entry)) {
            percounter_sync_enqueue(cfg, task, percounter_order_block_num(entry));
            continue;
        }

        uint64_t kernel_start = 0;
        if (!percounter_run_task(
                cfg, task, percounter_order_block_idx(entry), percounter_order_block_num(entry), &kernel_start
            )) {
            percounter_fail(cfg, SchedulerGraphResult::INVALID_CALLABLE);
            break;
        }
        if (kernel_start != 0) {
            batch = get_sys_cnt_aicore() - kernel_start >= PERCOUNTER_LONG_KERNEL_CYCLES ? 1 : PERCOUNTER_CLAIM_BATCH;
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
    const uint64_t cluster_count = scheduler_gm_query(control->cluster_count);
    const uint64_t aiv_per_cluster = scheduler_gm_query(control->aiv_per_cluster);
    if (!scheduler_cluster_coordinate_from_worker(
            block_idx, is_aic, static_cast<int32_t>(cluster_count), static_cast<int32_t>(aiv_per_cluster), &coordinate
        )) {
        // A lane outside the topology the AICPU published has no payload slot.
        scheduler_gm_compare_exchange(
            control->lane_error, uint64_t{0}, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
        );
        return;
    }

    LaneConfig cfg{};
    cfg.control = control;
    cfg.graph = SchedulerGraphView{
        scheduler_gm_query(control->graph_storage_address), 0, scheduler_gm_query(control->graph_storage_count), 0
    };
    cfg.callable_addresses =
        reinterpret_cast<__gm__ uint64_t *>(scheduler_gm_query(control->callable_addresses_address));
    cfg.callable_count = static_cast<uint32_t>(scheduler_gm_query(control->callable_addresses_count));
    cfg.payload = scheduler_state_at<DispatchPayload>(
        control, scheduler_gm_query(control->dispatch_payloads_offset) +
                     static_cast<uint64_t>(block_idx) * sizeof(DispatchPayload)
    );
    cfg.tickets = scheduler_state_at<PercounterTicket>(
        control, scheduler_gm_query(control->tickets_offset) +
                     static_cast<uint64_t>(type_index) * PERCOUNTER_TICKET_CAPACITY * sizeof(PercounterTicket)
    );
    cfg.order = scheduler_state_at<uint64_t>(control, scheduler_gm_query(control->orders_offset[type_index]));
    cfg.fanin = scheduler_state_at<PercounterFanin>(control, scheduler_gm_query(control->fanin_offset));
    cfg.fanin_addresses = scheduler_state_at<uint64_t>(control, scheduler_gm_query(control->fanin_addr_offset));
    cfg.counters = scheduler_state_at<PercounterCounter>(control, scheduler_gm_query(control->counters_offset));
    cfg.task_count = scheduler_gm_query(control->task_count);
    cfg.table_task_count = scheduler_gm_query(control->table_task_count);
    cfg.order_count = scheduler_gm_query(control->order_count[type_index]);
    const uint64_t lanes = is_aic ? cluster_count : cluster_count * aiv_per_cluster;
    cfg.first_batch = cfg.order_count <= PERCOUNTER_SHORT_LIST_PER_LANE * lanes ? uint64_t{1} : PERCOUNTER_CLAIM_BATCH;
    // Every lane of the type derives the same count from the topology. Grouping
    // by cluster keeps a cluster's two AIVs on one ticket.
    const uint64_t groups = lanes / PERCOUNTER_TICKET_GROUP_LANES;
    cfg.ticket_group_count = static_cast<uint32_t>(
        groups == 0 ? 1 : (groups > PERCOUNTER_TICKET_CAPACITY ? PERCOUNTER_TICKET_CAPACITY : groups)
    );
    cfg.ticket_group = static_cast<uint32_t>(coordinate.cluster_index) % cfg.ticket_group_count;
    cfg.has_predicates = scheduler_gm_query(control->has_predicates) != 0;
    cfg.cluster_index = coordinate.cluster_index;
    cfg.cluster_lane = coordinate.cluster_lane;

    const uint64_t timeout_cycles = scheduler_gm_query(control->scheduler_timeout_cycles);
    cfg.sync_cohort_count = scheduler_gm_query(control->sync_cohort_count);
    if (cfg.sync_cohort_count != 0) {
        cfg.sync_block_total = scheduler_gm_query(control->sync_block_total);
        cfg.sync_queue =
            scheduler_state_at<PercounterSyncQueue>(control, scheduler_gm_query(control->sync_cohort_offset));
        cfg.sync_pending = scheduler_state_at<uint64_t>(control, scheduler_gm_query(control->sync_pending_offset));
    }
    cfg.mix_order_count = scheduler_gm_query(control->mix_order_count);
    if (cfg.mix_order_count != 0) {
        cfg.mix_rendezvous_count = scheduler_gm_query(control->mix_rendezvous_count);
        cfg.mix_order = scheduler_state_at<uint64_t>(control, scheduler_gm_query(control->mix_order_offset));
        cfg.mix_tickets = scheduler_state_at<PercounterTicket>(control, scheduler_gm_query(control->mix_ticket_offset));
        cfg.mix_signal_ready =
            scheduler_state_at<uint64_t>(control, scheduler_gm_query(control->mix_signal_ready_offset));
        cfg.mix_signal = &percounter_ssbuf_region(platform_ssbuf_hardware_base_address())->mix;
        if (!percounter_clear_mix_signal(cfg, timeout_cycles)) return;
    }

    percounter_lane_loop(cfg, timeout_cycles);
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
    scheduler_observe_cache_line(handshake);

    const uint64_t report_epoch = get_aicore_report_epoch();
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
    scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
    const uint64_t timeout_cycles = scheduler_gm_query(control->scheduler_timeout_cycles);
    bool released = false;
    uint32_t signal = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
    const uint64_t ready_wait_start = get_sys_cnt_aicore();
    while (handshake->aicpu_ready != SCHEDULER_RUNTIME_MODE_PERCOUNTER && signal != AICORE_EXIT_SIGNAL) {
        scheduler_observe_cache_line(handshake);
        if (scheduler_watchdog_expired(ready_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
            scheduler_gm_compare_exchange(
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
        scheduler_observe_data_cache(reinterpret_cast<__gm__ void *>(control));
        const uint64_t release_wait_start = get_sys_cnt_aicore();
        while (signal != AICPU_IDLE_TASK_ID && signal != AICORE_EXIT_SIGNAL) {
            if (scheduler_watchdog_expired(release_wait_start, get_sys_cnt_aicore(), timeout_cycles)) {
                scheduler_gm_compare_exchange(
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
        write_reg(RegId::COND, AICORE_IDLE_VALUE);
        run_percounter_lane(control, block_idx, core_type);
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

}  // namespace simpler::hbg::percounter
