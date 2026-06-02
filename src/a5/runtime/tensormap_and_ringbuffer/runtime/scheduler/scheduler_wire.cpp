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
 * Wiring-thread hot path: `drain_wiring_queue` + `wire_task`.
 *
 * Lives here (not in pto_scheduler.h) because after step 4 wiring needs the
 * full PTO2OrchestratorState definition for `orchestrator->tensor_map` and
 * `orchestrator->rings[r].fanin_pool` access. The scheduler header is
 * included by pto_orchestrator.h, so the body can't sit in the header.
 */

#include "scheduler/pto_scheduler.h"

#include "pto_dep_compute.h"
#include "pto_fanin_builder.h"
#include "pto_orchestrator.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "tensor.h"

#include "aicpu/device_time.h"

#if PTO2_WIRING_PROFILING
#define WIRE_CYCLE_START() uint64_t _wt0 = get_sys_cnt_aicpu(), _wt1
#define WIRE_CYCLE_LAP(acc)                              \
    do {                                                 \
        _wt1 = get_sys_cnt_aicpu();                      \
        acc += (_wt1 - _wt0);                            \
        _wt0 = _wt1;                                     \
    } while (0)
#else
#define WIRE_CYCLE_START()
#define WIRE_CYCLE_LAP(acc)
#endif

int PTO2SchedulerState::drain_wiring_queue(bool force_drain) {
    int wired = 0;
#if PTO2_WIRING_PROFILING
    auto &perf = wiring_perf;
    uint64_t _drain_pop_t0 = get_sys_cnt_aicpu();
#endif

    // Refill local batch buffer when exhausted.
    if (wiring.batch_index >= wiring.batch_count) {
        // Backoff: defer pop when queue holds fewer than a full batch,
        // unless force_drain, orch_needs_drain, or backoff limit reached.
        if (!force_drain && wiring.queue.size() < WiringState::BATCH_SIZE) {
            if (!wiring.orch_needs_drain.load(std::memory_order_acquire) &&
                wiring.backoff_counter < WiringState::BACKOFF_LIMIT) {
                wiring.backoff_counter++;
                return 0;
            }
        }
        wiring.backoff_counter = 0;
        wiring.batch_count = wiring.queue.pop_batch(wiring.batch, WiringState::BATCH_SIZE);
        wiring.batch_index = 0;
        if (wiring.batch_count == 0) {
            // Nothing to do — make sure idle flag is sticky-true so
            // wait_for_tensor_ready's spin sees us as quiescent.
            wiring.idle.store(true, std::memory_order_release);
            return 0;
        }
        // We have work — clear idle so wait_for_tensor_ready blocks until we
        // finish this batch.
        wiring.idle.store(false, std::memory_order_release);
        // sync_tensormap is no longer called here. It now runs from
        // wiring_thread_run right after advance_ring_pointers, only when
        // last_task_alive actually advanced — see comment there.
    }

#if PTO2_WIRING_PROFILING
    // Close the drain_pop lap before per-event work begins. drain_pop covers
    // backoff + queue pop + capacity gate (anything in this method outside
    // of sync_tensormap / wire_task / on_scope_end).
    uint64_t _pop_end = get_sys_cnt_aicpu();
    perf.drain_pop_cycle += _pop_end - _drain_pop_t0;
#endif

    // Process events from local buffer in strict FIFO order. FIFO matters
    // for ScopeEnd correctness: every TaskSubmit pushed before a ScopeEnd
    // has already finalized its producer fanout_count++ by the time we
    // call on_scope_end, so release_producer cannot prematurely transition
    // a producer to CONSUMED. See plan
    // `.claude/plans/wiring-wiring-tensormap-submit-task-sub-zazzy-moore.md`.
    while (wiring.batch_index < wiring.batch_count) {
        PTO2WiringEvent &ev = wiring.batch[wiring.batch_index];

        if (ev.kind == PTO2WiringEventKind::TaskSubmit) {
            PTO2TaskSlotState *ws = ev.slot;
            int ring_id = ws->ring_id;
            auto &rss = ring_sched_states[ring_id];

            // dep_pool capacity gate — hard upper bound: every producer
            // discovered in wire_task (explicit_dep prefix + tensormap
            // lookup) takes at most one dep_pool entry. The prefix contributes
            // payload.fanin_actual_count entries; the lookup adds at most
            // PTO2_FANIN_INLINE_CAP more (anything beyond that spills into
            // fanin_spill_pool, not dep_pool). If pool can't fit the worst
            // case, reclaim then break the batch — leave the event in place
            // (do NOT increment batch_index) so the next drain call retries.
            int32_t worst_case_dep_entries = PTO2_FANIN_INLINE_CAP + ws->payload->fanin_actual_count;
            if (rss.dep_pool.available() < worst_case_dep_entries) {
                rss.dep_pool.reclaim(*rss.ring, rss.last_task_alive);
                if (rss.dep_pool.available() < worst_case_dep_entries) {
                    break;
                }
            }

            wiring.batch_index++;
            wire_task(rss, ev);
            wired++;
        } else {
            // PTO2WiringEventKind::ScopeEnd: release_producer on each slot
            // in the scope range. orch never rewinds scope_tasks, so the
            // slot_array pointer is stable.
#if PTO2_WIRING_PROFILING
            uint64_t _se_t0 = get_sys_cnt_aicpu();
#endif
            on_scope_end(ev.slot_array, ev.slot_count);
#if PTO2_WIRING_PROFILING
            perf.scope_end_cycle += get_sys_cnt_aicpu() - _se_t0;
#endif
            wiring.batch_index++;
        }
    }

    // If batch is fully consumed AND queue is empty, advertise idle. Cheap
    // (one release-store) and lets wait_for_tensor_ready stop spinning.
    if (wiring.batch_index >= wiring.batch_count && wiring.queue.size() == 0) {
        wiring.idle.store(true, std::memory_order_release);
    }

    return wired;
}

void PTO2SchedulerState::wire_task(RingSchedState &rss, const PTO2WiringEvent &ev) {
    WIRE_CYCLE_START();
#if PTO2_WIRING_PROFILING
    auto &perf = wiring_perf;
#endif
    PTO2TaskSlotState *ws = ev.slot;
    PTO2TaskPayload *wp = ws->payload;
    PTO2TaskId task_id = ws->task->task_id;
    int ring_id = ws->ring_id;

    PTO2OrchestratorState *orch = orchestrator;
    PTO2TensorMap &tensor_map = orch->tensor_map;
    PTO2FaninPool &spill_pool = *wp->fanin_spill_pool;

    // sync_tensormap is now called once per drain batch (see drain_wiring_queue),
    // not per task — last_task_alive is stable within a drain.

    // 2. Build the DepInputs view that compute_task_fanin / register_task_outputs
    // expect. Tags came in on the wiring event (no payload reach). The loop also
    // kicks off prefetches for tensors[i]'s two cache lines so the tensormap
    // scan finds them in L1/L2 instead of cold-miss latency.
    int32_t tensor_count = wp->tensor_count;
    TensorRef refs_local[MAX_TENSOR_ARGS];
    for (int32_t i = 0; i < tensor_count; i++) {
        refs_local[i].ptr = &wp->tensors[i];
        __builtin_prefetch(&wp->tensors[i], 0, 3);
        __builtin_prefetch(reinterpret_cast<char *>(&wp->tensors[i]) + 64, 0, 3);
    }
    DepInputs inputs{
        tensor_count, refs_local, ev.tags,
        /*explicit_dep_count=*/0, /*explicit_deps=*/nullptr,  // orch already
                                                              // resolved into builder
    };
    bool in_manual_scope = (ev.in_manual_scope != 0);

    // 3. Re-attach the fanin builder onto payload's inline_slots buffer; the
    // orch-side explicit_dep loop left a prefix of `fanin_actual_count` slots
    // already populated. compute_task_fanin's emit lambda extends it.
    PTO2FaninBuilder builder(wp->fanin_inline_slot_states, spill_pool);
    builder.count = wp->fanin_actual_count;
    builder.spill_start = wp->fanin_spill_start;
    WIRE_CYCLE_LAP(perf.blob_cycle);

    PTO2SharedMemoryHeader *sm = sm_header;
    std::atomic<int32_t> *err_code = &sm->orch_error_code;
    auto emit = [&](PTO2TaskId producer_task_id) -> bool {
        PTO2TaskSlotState *prod =
            &sm->rings[producer_task_id.ring()].get_slot_state_by_task_id(producer_task_id.local());
        return append_fanin_or_fail(sm->rings[ring_id], err_code, &builder, prod);
    };
    if (!compute_task_fanin(inputs, tensor_map, in_manual_scope, emit)) {
        return;  // emit flagged fatal
    }
    WIRE_CYCLE_LAP(perf.lookup_cycle);

    // 4. Insert this task's outputs/inouts so subsequent submits' lookups
    // discover them. Order vs (3) matters today (orch ran lookup before
    // insert); same ordering preserved here.
    register_task_outputs(inputs, task_id, tensor_map, in_manual_scope);
    WIRE_CYCLE_LAP(perf.insert_cycle);

    // 5. Publish final fanin metadata back to payload for downstream code
    // (release_fanin_and_check_ready iterates these).
    wp->fanin_actual_count = builder.count;
    wp->fanin_spill_start = builder.spill_start;

    // 6. Wire fanout edges: lock each producer, check completion, prepend to
    // its fanout list (or count as early-finished), bump its fanout_count.
    // fanout_count++ moved here from orch in step 4 — safe because both
    // orch and wiring are single-writer-of-producer.fanout_count and the
    // bump lands FIFO before any matching SCOPE_END's release_producer.
    int32_t wfanin = builder.count;
    ws->fanin_count = wfanin + 1;  // +1 = wiring sentinel

    if (wfanin != 0) {
        int32_t early_finished = 0;
        builder.for_each([&](PTO2TaskSlotState *producer) {
            producer->lock_fanout();
            int32_t pstate = producer->task_state.load(std::memory_order_acquire);
            if (pstate >= PTO2_TASK_COMPLETED) {
                early_finished++;
            } else {
                producer->fanout_head = rss.dep_pool.prepend(producer->fanout_head, ws);
            }
            producer->fanout_count++;
            producer->unlock_fanout();
        });
        WIRE_CYCLE_LAP(perf.fanout_cycle);

        int32_t init_rc = early_finished + 1;
        int32_t new_rc = ws->fanin_refcount.fetch_add(init_rc, std::memory_order_acq_rel) + init_rc;
        if (new_rc >= ws->fanin_count) {
            // All producers already finished at wire time — task is immediately
            // ready. Direct push to global queue (no thread-local buffer here);
            // fanin_zero == enter_global_queue.
#if PTO2_SCHED_PROFILING
            uint64_t now = get_sys_cnt_aicpu();
            ws->fanin_zero_cycles = now;
            ws->enter_global_queue_cycles = now;
#endif
            push_ready_routed(ws);
        }
    } else {
        // No fanin → ready at wire time.
#if PTO2_SCHED_PROFILING
        uint64_t now = get_sys_cnt_aicpu();
        ws->fanin_zero_cycles = now;
        ws->enter_global_queue_cycles = now;
#endif
        ws->fanin_refcount.fetch_add(1, std::memory_order_acq_rel);
        push_ready_routed(ws);
    }

    ws->dep_pool_mark = rss.dep_pool.top;
    WIRE_CYCLE_LAP(perf.ready_cycle);
#if PTO2_WIRING_PROFILING
    perf.task_count++;
#endif
}
