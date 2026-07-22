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

#include "scheduler_context.h"

#include <algorithm>
#include <cinttypes>
#include <limits>

#include "common.h"
#include "callable.h"
#include "aicpu/aicpu_device_config.h"
#include "aicpu/dep_gen_collector_aicpu.h"

static void latch_scheduler_error(PTO2SharedMemoryHeader *header, int32_t thread_idx, int32_t error_code) {
    if (header == nullptr || error_code == PTO2_ERROR_NONE) return;
    int32_t expected = PTO2_ERROR_NONE;
    if (header->sched_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel))
        header->sched_error_thread.store(thread_idx, std::memory_order_release);
    if (thread_idx >= 0 && thread_idx < 32)
        header->sched_error_bitmap.fetch_or(1U << static_cast<uint32_t>(thread_idx), std::memory_order_acq_rel);
}

static void format_core_status(
    char *buf, size_t buf_size, int32_t core_id, bool idle, const CoreExecState *core_state, uint64_t reg_addr_for_cond
) {
    if (idle) {
        snprintf(buf, buf_size, "core%d(idle)", core_id);
        return;
    }
    int32_t kernel = -1;
    int64_t task_id_raw = -1;
    if (core_state && core_state->running_slot_state) {
        int32_t subslot = static_cast<int32_t>(core_state->running_subslot);
        kernel = core_state->running_slot_state->task->kernel_id[subslot];
        task_id_raw = static_cast<int64_t>(core_state->running_slot_state->task->task_id.raw);
    }
    uint64_t cond_reg = read_reg(reg_addr_for_cond, RegId::COND);
    int32_t hw_state = EXTRACT_TASK_STATE(cond_reg);
    const char *cond_reg_state_str = (hw_state == TASK_ACK_STATE) ? "ack" : "fin";
    if (hw_state == TASK_ACK_STATE)
        snprintf(
            buf, buf_size, "core%d(busy kernel=%d task=%" PRId64 " cond_reg_state=%s)", core_id, kernel, task_id_raw,
            cond_reg_state_str
        );
    else
        snprintf(
            buf, buf_size, "core%d(busy kernel=%d task=%" PRId64 " cond_reg_state=%s ANOMALY)", core_id, kernel,
            task_id_raw, cond_reg_state_str
        );
}

int32_t SchedulerContext::pre_handshake_init(
    Runtime *runtime, int32_t aicpu_thread_num, int32_t sched_thread_num, uint64_t regs_base
) {
    always_assert(runtime != nullptr);

    // Zero all per-core execution state before handshake
    memset(core_exec_states_, 0, sizeof(core_exec_states_));

    // Wire thread/transition configuration that handshake/assign need to read.
    aicpu_thread_num_ = aicpu_thread_num;
    sched_thread_num_ = sched_thread_num;
    regs_ = regs_base;

    // Initialize l2-swimlane buffers BEFORE any thread writes aicpu_ready in
    // handshake_partition so the AICore-side rotation table slots are
    // populated when AICore reads them post-handshake. AICore stashes
    // &rotation_table[block_idx] at entry; the slot CONTENTS (the actual
    // record buffer pointer it later dereferences) are written here.
    // aicpu_ready=1 is AICore's signal to proceed past Phase 1 — once it has
    // the green light, it expects the slot to be initialized. This runs on
    // the leader before it publishes hs_setup_done_, so it happens-before
    // every thread's handshake. See the contract comment in
    // aicore/aicore_executor.cpp:105-110 and the parallel call in
    // host_build_graph/aicpu/aicpu_executor.cpp:341. Without this call,
    // --enable-l2-swimlane runs hit AICore-side memory corruption that
    // surfaces as orch FLOW_CONTROL_DEADLOCK (paged_attention C1) or
    // sched SCHEDULER_TIMEOUT (multi_round_paged_attention C1) depending
    // on which AICore op first touches the uninitialized slot.
    if (is_l2_swimlane_enabled()) {
        l2_swimlane_aicpu_init(runtime->dev.worker_count);
    }

    cores_total_num_ = runtime->dev.worker_count;
    if (cores_total_num_ == 0 || cores_total_num_ > RUNTIME_MAX_WORKER) return -1;
    aic_count_ = 0;
    aiv_count_ = 0;
    handshake_failed_.store(false, std::memory_order_release);

    // State the barrier-free init path (handshake_owned_clusters /
    // assign_own_clusters) reads without a leader post_handshake_init:
    // scheduler-thread count and func table. The leader sets these before
    // publishing hs_setup_done_, so they happen-before any thread's
    // self-assignment. The barrier path re-derives them in post_handshake_init,
    // so this is redundant (not harmful) there.
    //
    // payload_per_core_ / deferred_slab_per_core_ are deliberately NOT memset:
    // build_payload() overwrites every dispatched payload field and dispatch
    // resets slab count/error_code before the slab can be read (the same
    // invariant deinit() relies on to skip the ~300 KB zeroing). The global
    // memset here was ~37 us of preamble on the critical path — upstream skips
    // it and that was the bulk of polling's small-kernel preamble gap.
    // sub_block_id is set per owned core in assign_own_clusters and persists
    // across runs, so it survives without the memset.
    active_sched_threads_ = (sched_thread_num > 0) ? sched_thread_num : aicpu_thread_num;
    func_id_to_addr_ = runtime->dev.func_id_to_addr_;
    return 0;
}

void SchedulerContext::handshake_partition(Runtime *runtime, int32_t tidx, int32_t nthreads) {
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->dev.workers);
    const int32_t total = cores_total_num_;
    const int32_t lo = static_cast<int32_t>((static_cast<int64_t>(tidx) * total) / nthreads);
    const int32_t hi = static_cast<int32_t>((static_cast<int64_t>(tidx + 1) * total) / nthreads);

    // Step 1: signal this slice's cores to proceed past Phase 1.
    for (int32_t i = lo; i < hi; i++) {
        all_handshakes[i].task = reinterpret_cast<uint64_t>(&payload_per_core_[i][0]);
        OUT_OF_ORDER_STORE_BARRIER();
        all_handshakes[i].aicpu_ready = 1;
    }
    OUT_OF_ORDER_STORE_BARRIER();

    uint32_t max_physical_cores_count = platform_get_physical_cores_count();

    // Step 2: wait for this slice's cores, then init their registers.
    // Single-round-trip (#1310): the AICore publishes {physical_core_id,
    // core_type, aicore_done} in one write on launch, so aicore_done alone
    // gates discovery — the separate aicore_regs_ready / aicpu_regs_ready
    // round was removed. Registers are opened after aicore_done is observed.
    for (int32_t i = lo; i < hi; i++) {
        Handshake *hank = &all_handshakes[i];

        while (hank->aicore_done == 0)
            SPIN_WAIT_HINT();

        uint32_t physical_core_id = hank->physical_core_id;

        if (physical_core_id >= max_physical_cores_count) {
            handshake_failed_.store(true, std::memory_order_release);
            continue;
        }

        uint64_t *regs = reinterpret_cast<uint64_t *>(regs_);
        uint64_t reg_addr = regs[physical_core_id];

        CoreType type = hank->core_type;

        // Open this core's window after discovery.
        platform_init_aicore_regs(reg_addr);
        OUT_OF_ORDER_STORE_BARRIER();

        core_exec_states_[i].reg_addr = reg_addr;
        core_exec_states_[i].cond_ptr = get_reg_ptr(reg_addr, RegId::COND);

        core_exec_states_[i].worker_id = i;
        core_exec_states_[i].physical_core_id = physical_core_id;
        core_exec_states_[i].core_type = type;
    }
}

int32_t SchedulerContext::post_handshake_init(Runtime *runtime) {
    if (handshake_failed_.load(std::memory_order_acquire)) {
        emergency_shutdown(runtime);
        return -1;
    }

    // Build cluster-ordered AIC/AIV worker-id lists from the discovered
    // cores. Serial and MMIO-free — the expensive per-core handshake already
    // ran in parallel. Core-index order matches the original single-thread
    // handshake so assign_cores_to_threads forms identical clusters.
    for (int32_t i = 0; i < cores_total_num_; i++) {
        if (core_exec_states_[i].core_type == CoreType::AIC) aic_worker_ids_[aic_count_++] = i;
        else aiv_worker_ids_[aiv_count_++] = i;
    }

    if (!assign_cores_to_threads()) return -1;

    // Initialize task counters. Task count comes from PTO2 shared memory.
    if (runtime->get_gm_sm_ptr()) {
        auto *header = static_cast<PTO2SharedMemoryHeader *>(runtime->get_gm_sm_ptr());
        int64_t pto2_count = 0;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            int32_t ring_tasks = header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
            if (ring_tasks > 0 && ring_tasks <= PTO2_SCOPE_TASKS_CAP) pto2_count += ring_tasks;
        }
        total_tasks_ = static_cast<int32_t>(pto2_count);
    } else {
        total_tasks_ = 0;
    }
    completed_tasks_.store(0, std::memory_order_release);

    // Device orchestration: the orchestrator thread flips this when the graph is built.
    orchestrator_done_ = false;

    // Clear per-core dispatch payloads
    memset(payload_per_core_, 0, sizeof(payload_per_core_));
    memset(deferred_slab_per_core_, 0, sizeof(deferred_slab_per_core_));

    // Initialize per-core GlobalContext (sub_block_id) based on cluster position.
    // This is done once at startup and never modified afterwards.
    for (int32_t t = 0; t < sched_thread_num_; t++) {
        CoreTracker &tracker = core_trackers_[t];
        for (int32_t c = 0; c < tracker.get_cluster_count(); c++) {
            int32_t cluster_offset = c * 3;  // Each cluster = 1 AIC + 2 AIV
            auto aiv0_id = tracker.get_core_id_by_offset(tracker.get_aiv0_core_offset(cluster_offset));
            auto aiv1_id = tracker.get_core_id_by_offset(tracker.get_aiv1_core_offset(cluster_offset));
            payload_per_core_[aiv0_id][0].global_context.sub_block_id = 0;
            payload_per_core_[aiv0_id][1].global_context.sub_block_id = 0;
            payload_per_core_[aiv1_id][0].global_context.sub_block_id = 1;
            payload_per_core_[aiv1_id][1].global_context.sub_block_id = 1;
        }
    }

    func_id_to_addr_ = runtime->dev.func_id_to_addr_;

    return 0;
}

void SchedulerContext::deinit() {
    // Reset all per-core execution state
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        core_exec_states_[i] = {};
        core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
        core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // Neither per-core array is zeroed here (mirrors upstream, ~300 KB
    // saved). payload_per_core_: build_payload() overwrites every dispatched
    // field including not_ready=0. deferred_slab_per_core_: dispatch resets
    // count=0/error_code=NONE before the slab can be read, the loop above
    // reset every running/pending_reg_task_id to INVALID so no undispatched
    // slot is ever drained, and the consumer is count-gated so it never
    // reads entries[] past the fresh count.

    // Reset sync-start drain coordination — a previous run that aborted mid-drain
    // would otherwise leave dirty pending/elected/ack state for the next reuse.
    drain_state_.sync_start_pending.store(0, std::memory_order_release);
    drain_state_.drain_worker_elected.store(0, std::memory_order_release);
    drain_state_.drain_ack_mask.store(0, std::memory_order_release);
    drain_state_.pending_task.store(nullptr, std::memory_order_release);

    // Reset task counters and orchestrator state
    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_ = 0;
    orchestrator_done_ = false;
    pto2_init_done_.store(false, std::memory_order_release);
    pto2_init_complete_.store(false, std::memory_order_release);

    completed_.store(false, std::memory_order_release);

    // Reset core discovery and assignment state
    aic_count_ = 0;
    aiv_count_ = 0;
    cores_total_num_ = 0;
    aicpu_thread_num_ = 0;
    sched_thread_num_ = 0;
    active_sched_threads_ = 0;
    for (int32_t t = 0; t < MAX_AICPU_THREADS; t++)
        core_trackers_[t] = CoreTracker{};

    regs_ = 0;
    sched_ = nullptr;
    rt_ = nullptr;
    func_id_to_addr_ = nullptr;
}

int32_t SchedulerContext::shutdown(int32_t thread_idx) {
    const int32_t *cores = core_trackers_[thread_idx].core_ids();
    int32_t core_num = core_trackers_[thread_idx].core_num();
    if (core_num == 0) return 0;

    int32_t rc = 0;
    for (int32_t i = 0; i < core_num; i++) {
        int32_t core_id = cores[i];
        uint64_t reg_addr = core_exec_states_[core_id].reg_addr;
        if (reg_addr != 0) {
            // Timeout means AICore is unresponsive. Log and continue deiniting remaining cores.
            if (platform_deinit_aicore_regs(reg_addr) != 0) rc = -1;
        } else {
        }
    }
    return rc;
}

void SchedulerContext::on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t, int32_t total_tasks) {
    on_orchestration_done(runtime, rt, total_tasks);
}

void SchedulerContext::on_orchestration_done(Runtime *runtime, PTO2Runtime *rt, int32_t total_tasks) {
    total_tasks_ = total_tasks;

    // Fold tasks completed inline during orchestration
    int32_t inline_completed = static_cast<int32_t>(rt->orchestrator.inline_completed_tasks);
    if (inline_completed > 0) completed_tasks_.fetch_add(inline_completed, std::memory_order_relaxed);
    orchestrator_done_ = true;

    // Check for fatal error from orchestration; if so, shut down immediately.
    int32_t orch_err = 0;
    if (sched_->sm_header) orch_err = sched_->sm_header->orch_error_code.load(std::memory_order_relaxed);
    if (orch_err != PTO2_ERROR_NONE) {
        if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
    }
}

void SchedulerContext::bind_runtime(PTO2Runtime *rt) {
    rt_ = rt;
    sched_ = &rt->scheduler;
}

void SchedulerContext::wait_for_orchestration_done_before_dispatch(Runtime *, int32_t thread_idx) {
    while (!orchestrator_done_) {
        if (thread_idx == 0 && sched_ != nullptr) {
            sched_->drain_wiring_queue(false);
        }
        SPIN_WAIT_HINT();
    }
}

bool SchedulerContext::assign_cores_to_threads() {
    // Cluster-aligned round-robin assignment: cluster ci -> sched thread ci % active_sched_threads_.
    // Each cluster = 1 AIC + 2 adjacent AIV; the triple is always kept together.
    active_sched_threads_ = (sched_thread_num_ > 0) ? sched_thread_num_ : aicpu_thread_num_;
    int32_t cluster_count = aic_count_;

    // Max clusters any single sched thread can hold: ceil(cluster_count / active_sched_threads_).
    int32_t max_clusters_per_thread = (cluster_count + active_sched_threads_ - 1) / active_sched_threads_;
    int32_t thread_cores_num = max_clusters_per_thread * 3;

    if (thread_cores_num > CoreTracker::MAX_CORE_PER_THREAD) return false;

    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
        core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // Count clusters per thread first (round-robin may distribute unevenly)
    int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++)
        clusters_per_thread[ci % active_sched_threads_]++;
    for (int32_t i = 0; i < active_sched_threads_; i++)
        core_trackers_[i].init(clusters_per_thread[i]);

    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};

    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % active_sched_threads_;

        int32_t aic_wid = aic_worker_ids_[ci];
        int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
        int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

        core_trackers_[t].set_cluster(cluster_idx_per_thread[t]++, aic_wid, aiv0_wid, aiv1_wid);
    }

    for (int32_t t = 0; t < aicpu_thread_num_; t++) {}

    return true;
}

void SchedulerContext::emergency_shutdown(Runtime *runtime) {
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->dev.workers);
    int32_t timeout_count = 0;
    for (int32_t i = 0; i < cores_total_num_; i++) {
        Handshake *hank = &all_handshakes[i];
        OUT_OF_ORDER_STORE_BARRIER();
        (void)hank;  // single-round-trip: no aicpu_regs_ready round; deinit forces exit
        if (core_exec_states_[i].reg_addr != 0) {
            if (platform_deinit_aicore_regs(core_exec_states_[i].reg_addr) != 0) timeout_count++;
        }
    }
    if (timeout_count > 0) {}
}

LoopAction SchedulerContext::handle_orchestrator_exit(PTO2SharedMemoryHeader *header, Runtime *runtime) {
    if (completed_.load(std::memory_order_acquire)) return LoopAction::BREAK_LOOP;
    int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
    if (orch_err != PTO2_ERROR_NONE) {
        if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
        return LoopAction::BREAK_LOOP;
    }
    int32_t sched_err = header->sched_error_code.load(std::memory_order_acquire);
    if (sched_err != PTO2_ERROR_NONE) {
        if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
        return LoopAction::BREAK_LOOP;
    }

    if (!orchestrator_done_) return LoopAction::NONE;

    if (total_tasks_ > 0 && completed_tasks_.load(std::memory_order_relaxed) >= total_tasks_) {
        completed_.store(true, std::memory_order_release);
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

LoopAction SchedulerContext::check_idle_fatal_error(PTO2SharedMemoryHeader *header, Runtime *runtime) {
    if (completed_.load(std::memory_order_acquire)) return LoopAction::BREAK_LOOP;
    int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
    if (orch_err != PTO2_ERROR_NONE) {
        if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
        return LoopAction::BREAK_LOOP;
    }
    int32_t sched_err = header->sched_error_code.load(std::memory_order_acquire);
    if (sched_err != PTO2_ERROR_NONE) {
        if (!completed_.exchange(true, std::memory_order_acq_rel)) emergency_shutdown(runtime);
        return LoopAction::BREAK_LOOP;
    }
    return LoopAction::NONE;
}

void SchedulerContext::log_stall_diagnostics(int32_t thread_idx) {
    CoreTracker &tracker = core_trackers_[thread_idx];

    // T0 owns the shared-ring scan; printing it from other threads would
    // produce identical TASK lines once per scheduler thread.
    if (thread_idx == 0) {
        int32_t cnt_ready = 0, cnt_waiting = 0, cnt_running = 0;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            PTO2SharedMemoryRingHeader &ring = *sched_->ring_sched_states[r].ring;
            int32_t ring_task_count = ring.fc.current_task_index.load(std::memory_order_relaxed);
            for (int32_t si = 0; si < ring_task_count; si++) {
                PTO2TaskSlotState &slot_state = ring.get_slot_state_by_task_id(si);
                // (m) task_state retired; use completion_flags directly.
                bool fanin_ready = sched_->fanin_satisfied(&slot_state);
                if (ring.completion_flags[si & ring.task_window_mask].load(std::memory_order_relaxed) != 0) continue;
                char running_on[192] = {0};
                int32_t owner = -1;
                int32_t pos = 0;
                bool is_running = false;
                for (int32_t cid = 0; cid < cores_total_num_ && pos + 32 < (int32_t)sizeof(running_on); cid++) {
                    if (core_exec_states_[cid].running_slot_state != &slot_state) continue;
                    is_running = true;
                    if (owner < 0) owner = find_core_owner_thread(cid);
                    const char *sname = subslot_name(core_exec_states_[cid].running_subslot);
                    int32_t written = snprintf(
                        running_on + pos, sizeof(running_on) - pos, "%score=%d(%s)", pos == 0 ? "" : " ", cid, sname
                    );
                    if (written > 0) pos += written;
                }

                if (is_running) {
                    cnt_running++;
                    if (cnt_running > STALL_DUMP_READY_MAX) continue;
                    continue;
                }
                if (fanin_ready) {
                    cnt_ready++;
                    if (cnt_ready > STALL_DUMP_READY_MAX) continue;
                    continue;
                }
                cnt_waiting++;
                if (cnt_waiting > STALL_DUMP_WAIT_MAX) continue;
            }
        }
    }

    for (int32_t cli = 0; cli < tracker.get_cluster_count() && cli < STALL_DUMP_CORE_MAX; cli++) {
        int32_t offset = cli * 3;
        int32_t aic_id = tracker.get_aic_core_id(offset);
        int32_t aiv0_id = tracker.get_aiv0_core_id(offset);
        int32_t aiv1_id = tracker.get_aiv1_core_id(offset);
        bool aic_idle = tracker.is_aic_core_idle(offset);
        bool aiv0_idle = tracker.is_aiv0_core_idle(offset);
        bool aiv1_idle = tracker.is_aiv1_core_idle(offset);
        char aic_buf[128], aiv0_buf[128], aiv1_buf[128];
        format_core_status(
            aic_buf, sizeof(aic_buf), aic_id, aic_idle, &core_exec_states_[aic_id], core_exec_states_[aic_id].reg_addr
        );
        format_core_status(
            aiv0_buf, sizeof(aiv0_buf), aiv0_id, aiv0_idle, &core_exec_states_[aiv0_id],
            core_exec_states_[aiv0_id].reg_addr
        );
        format_core_status(
            aiv1_buf, sizeof(aiv1_buf), aiv1_id, aiv1_idle, &core_exec_states_[aiv1_id],
            core_exec_states_[aiv1_id].reg_addr
        );
    }
}

void SchedulerContext::log_shutdown_stall_snapshot() {
    int32_t thread_count = active_sched_threads_ > 0 ? active_sched_threads_ : aicpu_thread_num_;
    if (thread_count < 0 || thread_count > MAX_AICPU_THREADS) thread_count = thread_count < 0 ? 0 : MAX_AICPU_THREADS;
    for (int32_t t = 0; t < thread_count; t++)
        log_stall_diagnostics(t);
}

int32_t SchedulerContext::find_core_owner_thread(int32_t core_id) const {
    for (int32_t t = 0; t < aicpu_thread_num_; t++) {
        const int32_t *ids = core_trackers_[t].core_ids();
        int32_t n = core_trackers_[t].core_num();
        for (int32_t i = 0; i < n; i++)
            if (ids[i] == core_id) return t;
    }
    return -1;
}

bool SchedulerContext::self_owns_running_task(int32_t thread_idx) const {
    const int32_t *cores = core_trackers_[thread_idx].core_ids();
    int32_t core_num = core_trackers_[thread_idx].core_num();
    for (int32_t i = 0; i < core_num; i++)
        if (core_exec_states_[cores[i]].running_slot_state != nullptr) return true;
    return false;
}

bool SchedulerContext::no_thread_owns_running_task() const {
    for (int32_t t = 0; t < aicpu_thread_num_; t++)
        if (self_owns_running_task(t)) return false;
    return true;
}

int32_t SchedulerContext::handle_timeout_exit(int32_t thread_idx, PTO2SharedMemoryHeader *header, Runtime *runtime) {
    latch_scheduler_error(header, thread_idx, PTO2_ERROR_SCHEDULER_TIMEOUT);
    if (!completed_.exchange(true, std::memory_order_acq_rel)) {
        log_shutdown_stall_snapshot();
        emergency_shutdown(runtime);
    }
    return -PTO2_ERROR_SCHEDULER_TIMEOUT;
}
