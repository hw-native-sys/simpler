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
#include <unistd.h>

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicpu/device_time.h"
#include "aicpu/device_phase_aicpu.h"
#include "callable_protocol.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Performance profiling headers
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "common/unified_log.h"

// Register-based communication
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"
#include "utils/thread_completion_gate.h"

// Core type definitions
#include "common/core_type.h"

// CoreCallable for resolved dispatch address
#include "callable.h"

// Scheduler data structures (CoreExecState, CoreTracker, etc.)
#include "scheduler/scheduler_types.h"

// Scheduler context class
#include "scheduler/scheduler_context.h"

// From orchestration/common.cpp linked into this DSO — updates g_current_runtime
// here (cleared on teardown before runtime_destroy).
extern "C" void framework_bind_runtime(PTO2Runtime *rt);

static int32_t read_pto2_runtime_status(Runtime *runtime) {
    if (runtime == nullptr) {
        return 0;
    }

    void *sm = runtime->get_gm_sm_ptr();
    if (sm == nullptr) {
        return 0;
    }

    auto *header = static_cast<PTO2SharedMemoryHeader *>(sm);
    int32_t orch_error_code = header->orch_error_code.load(std::memory_order_acquire);
    int32_t sched_error_code = header->sched_error_code.load(std::memory_order_acquire);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static PTO2Runtime *rt{nullptr};

struct AicpuExecutor {
    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};

    // Parallel-handshake coordination (see AicpuExecutor::init). hs_setup_done_
    // is published by the leader once the shared pre-handshake setup is visible;
    // hs_arrived_ is the barrier counting threads that finished their core slice.
    // hs_thread_seq_ hands out a distinct [0, nthreads) index when the platform
    // exposes no affinity idx (sim, where platform_aicpu_affinity_thread_idx()
    // is -1 during init) so the threads don't all collapse to leader 0.
    std::atomic<bool> hs_setup_done_{false};
    std::atomic<int32_t> hs_arrived_{0};
    std::atomic<int32_t> hs_thread_seq_{0};

    // Parallel-boot-classify coordination (see AicpuExecutor::run). classify_ready_
    // is published by the boot leader once its leader-only orchestration setup is
    // visible; classify_arrived_ is the barrier counting threads that finished
    // their slice of the initial classify. Both are one-shot per run and reset in
    // deinit().
    std::atomic<bool> classify_ready_{false};
    std::atomic<int32_t> classify_arrived_{0};

    int32_t aicpu_thread_num_{0};

    // ===== Task queue state (managed by scheduler ready queues) =====

    simpler::ThreadCompletionGate completion_gate_;
    std::atomic<bool> runtime_init_ready_{false};

    // Per-Worker arena backing the PTO2Runtime + sm_handle + orch/sched/mailbox
    // sub-regions (created in runtime_create_from_sm, released in runtime_destroy).
    // Default-constructed: libc-backed backend, no ctx.
    DeviceArena runtime_arena_;

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
};

static AicpuExecutor g_aicpu_executor;

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // All AICPU threads enter init. The per-core AICore handshake is the
    // dominant preamble cost (serial MMIO, ~217 µs of ~283 µs for 72 cores), so
    // it is parallelized: the leader (tidx 0) does the shared setup, every
    // thread handshakes a disjoint slice of cores, then the leader finishes init
    // after a barrier. Non-leaders spin on init_done_.
    int32_t nthreads = runtime->aicpu_thread_num;
    if (nthreads == 0) nthreads = 1;
    if (nthreads < 1 || nthreads > MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid aicpu_thread_num: %d", nthreads);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }
    // Each thread needs a distinct index in [0, nthreads) to pick the leader and
    // partition the cores. Onboard the gate filter assigns it (exec_idx); sim's
    // gate does not, so platform_aicpu_affinity_thread_idx() is -1 here for every
    // thread — hand those a distinct index from a counter (mirrors run()'s
    // thread_idx_++ fallback) instead of collapsing them all to leader 0, which
    // would run pre_/post_handshake_init on every thread and race the shared
    // scheduler state. Exactly nthreads threads reach init (the gate drops the
    // rest), so the counter yields a gap-free [0, nthreads).
    int32_t tidx = platform_aicpu_affinity_thread_idx();
    if (tidx < 0) tidx = hs_thread_seq_.fetch_add(1, std::memory_order_acq_rel);
    // A thread whose index still falls outside [0, nthreads) owns no core slice:
    // handshake_partition would compute lo/hi past cores_total_num_ and index
    // all_handshakes[]/core_exec_states_ out of bounds. Reject it here (mirrors
    // the bounds guard already in run()). Fail only this thread and do NOT set
    // init_failed_ — that would make the valid peers abort before their
    // hs_arrived_ increment and hang the leader at the barrier below.
    if (tidx >= nthreads) {
        LOG_ERROR("AICPU affinity thread idx %d out of range [0,%d) in init", tidx, nthreads);
        return -1;
    }
    const bool is_leader = (tidx == 0);

    if (is_leader) {
        LOG_INFO("AicpuExecutor: Initializing");
        // The 0 → 1 fixup already applied above.
        aicpu_thread_num_ = nthreads;

        hs_arrived_.store(0, std::memory_order_relaxed);
        if (sched_ctx_.pre_handshake_init(runtime, aicpu_thread_num_, get_platform_regs()) != 0) {
            init_failed_.store(true, std::memory_order_release);
            hs_setup_done_.store(true, std::memory_order_release);
            return -1;
        }
        hs_setup_done_.store(true, std::memory_order_release);
    } else {
        while (!hs_setup_done_.load(std::memory_order_acquire)) {
            if (init_failed_.load(std::memory_order_acquire)) return -1;
        }
        if (init_failed_.load(std::memory_order_acquire)) return -1;
    }

    // All threads: handshake this thread's slice of cores in parallel.
    sched_ctx_.handshake_partition(runtime, tidx, nthreads);

    // Barrier: leader waits for every slice to finish, then completes init.
    hs_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (is_leader) {
        while (hs_arrived_.load(std::memory_order_acquire) < nthreads) {}
        completion_gate_.reset();
        if (sched_ctx_.post_handshake_init(runtime) != 0) {
            init_failed_.store(true, std::memory_order_release);
            init_done_.store(true, std::memory_order_release);
            return -1;
        }
        init_done_.store(true, std::memory_order_release);
        LOG_INFO("AicpuExecutor: Init complete");
    } else {
        while (!init_done_.load(std::memory_order_acquire)) {
            if (init_failed_.load(std::memory_order_acquire)) return -1;
        }
        if (init_failed_.load(std::memory_order_acquire)) return -1;
    }
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::run(Runtime *runtime) {
    int32_t affinity_exec_idx = platform_aicpu_affinity_thread_idx();
    int32_t thread_idx = (affinity_exec_idx >= 0) ? affinity_exec_idx : (thread_idx_++);
    if (thread_idx < 0 || thread_idx >= aicpu_thread_num_ || thread_idx >= MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Thread index %d out of bounds (active=%d max=%d exec_idx=%d)", thread_idx, aicpu_thread_num_,
            MAX_AICPU_THREADS, affinity_exec_idx
        );
        return -1;
    }
    int32_t run_rc = 0;

    // Boot: the last AICPU thread (aicpu_thread_num_ - 1) performs the one-time
    // host-orch attach. host_build_graph's orchestrator already ran on the host,
    // and every cross-task reference it wrote is an offset from its own block, so
    // the SM/arena this thread sees need no address fixup. This thread attaches
    // the prebuilt arena, points the SM
    // handle's ring-header pointers at the device SM WITHOUT resetting the
    // host-populated data, hands the host-computed task count to the scheduler,
    // and releases the other threads. It then falls through and schedules its own
    // cores like every other thread — host_build_graph has no device-side
    // orchestrator, so there is no orch/sched split.
    if (thread_idx == aicpu_thread_num_ - 1) {
        void *prebuilt_arena = runtime->get_prebuilt_arena_base();
        size_t off_runtime = runtime->get_prebuilt_runtime_offset();

        // A boot failure falls through to the common teardown at the end of
        // run() — it must NOT return early. This thread owns a core slice
        // (handshake_partition assigns [lo, total) to the last thread), so an
        // early return would skip shutdown(thread_idx) — leaving its AICore
        // cores spinning on an unclosed register window — and the completion
        // gate never opens, so the host hangs into the op-execute
        // timeout (507018) instead of seeing the failure. On failure: record it
        // in run_rc, leave rt null so the dispatch block below skips, and still
        // publish runtime_init_ready_ (single point at the block's end) so the
        // peer threads stop spinning.
        bool boot_ok = (prebuilt_arena != nullptr);
        if (!boot_ok) {
            LOG_ERROR("Thread %d: host-orch: prebuilt_arena_base is null", thread_idx);
            rt = nullptr;
            run_rc = -1;
        }

        if (boot_ok) {
            runtime_arena_.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);
            rt = reinterpret_cast<PTO2Runtime *>(static_cast<char *>(prebuilt_arena) + off_runtime);
            runtime_wire_arena_pointers(runtime_arena_, rt->prebuilt_layout, rt);

            void *sm_ptr = runtime->get_gm_sm_ptr();
            // The image the host shipped is pitched to the submitted task count,
            // not to the ring capacity, and the device region holds exactly that
            // image — so its size comes from the same pitch. attach_populated
            // rejects a pitch outside (0, capacity] and a region too small for it.
            const uint64_t live_slots =
                pto2_sm_layout::live_slot_pitch(static_cast<uint64_t>(runtime->host_total_tasks));
            const uint64_t payload_stride = runtime->sm_payload_stride;
            const uint64_t sm_size = pto2_sm_layout::ring_segment_offsets(live_slots, payload_stride).end;
            // sm_handle and the scheduler state are the device-only zone: their
            // bytes never travel, so they start as whatever the pooled arena last
            // held. Zeroing the handle first is what makes attach_populated's
            // assignment of every field checkable here rather than by inspecting
            // attach_populated.
            memset(rt->sm_handle, 0, sizeof(*rt->sm_handle));
            if (!rt->sm_handle->attach_populated(
                    sm_ptr, sm_size, rt->prebuilt_layout.task_capacity, live_slots, payload_stride
                )) {
                LOG_ERROR("Thread %d: host-orch: sm_handle->attach_populated failed", thread_idx);
                rt = nullptr;
                run_rc = -1;
                boot_ok = false;
            } else if (!rt->scheduler->init_data_from_layout(rt->prebuilt_layout.sched, runtime_arena_, sm_ptr)) {
                LOG_ERROR("Thread %d: host-orch: scheduler init_data_from_layout failed", thread_idx);
                rt = nullptr;
                run_rc = -1;
                boot_ok = false;
            } else {
                // Queue headers are set, so the slot arrays can take their ramp,
                // and the mailbox ring gets its cursors and publication gates.
                // All of it precedes runtime_init_ready_, which is what releases
                // the peer threads into the dispatch loop, so neither a push nor a
                // completion message sees an uninitialized region.
                rt->scheduler->seed_queue_slots();
                rt->aicore_mailbox->init_empty();
            }
        }

        if (boot_ok) {
            memset(rt->aicore_mailbox, 0, sizeof(*rt->aicore_mailbox));
            runtime_finalize_after_wire(rt, sched_ctx_.aic_count(), sched_ctx_.aiv_count());
            runtime->set_slot_states_ptr(nullptr);

            sched_ctx_.bind_runtime(rt);

            // Latch the host-built task count (on_orchestration_done sets total_tasks_)
            // BEFORE the runtime_init_ready_ release below — that store is the barrier
            // that unblocks the scheduler threads. Otherwise they would acquire
            // runtime_init_ready_ with total_tasks_=0 and race to an early exit before
            // the host task count is visible (host-orch has no concurrent orchestrator
            // to keep them alive).
            // NOTE: do NOT call rt_orchestration_done(rt) here. The HOST already
            // called it in run_host_orchestration; the orchestrator's own
            // task-allocator pointers name host memory the device never reads, so
            // mark_done()'s active_count() read would dereference it and fault the
            // AICPU. on_orchestration_done only needs total_tasks and the scalar
            // orchestrator.inline_completed_tasks, both already valid.
            sched_ctx_.on_orchestration_done(runtime, rt, thread_idx, runtime->host_total_tasks);
            LOG_INFO("Thread %d: host-orch boot complete (%d tasks)", thread_idx, runtime->host_total_tasks);
        }

        // Publish "leader setup done" (SM attached, task count latched, queues
        // allocated). Every thread then classifies its slice below before any of
        // them may dispatch — the leader holds runtime_init_ready_ until then.
        classify_ready_.store(true, std::memory_order_release);
    }

    // Parallel initial classify. Every AICPU thread waits for the leader's
    // orchestration setup, seeds its disjoint slice of the whole graph's ready
    // set + wake lists, then barriers. Only once all slices are done does the
    // leader publish runtime_init_ready_, so no thread dispatches against a
    // half-seeded graph. This replaces the O(total_tasks) serial classify the
    // leader used to run alone while the others idle-waited.
    while (!classify_ready_.load(std::memory_order_acquire)) {
        SPIN_WAIT_HINT();
    }
    if (!sched_ctx_.is_completed() && rt != nullptr) {
        sched_ctx_.classify_partition(thread_idx, aicpu_thread_num_);
    }
    classify_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (thread_idx == aicpu_thread_num_ - 1) {
        while (classify_arrived_.load(std::memory_order_acquire) < aicpu_thread_num_) {
            SPIN_WAIT_HINT();
        }
        runtime_init_ready_.store(true, std::memory_order_release);
    } else {
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    // Every AICPU thread schedules its assigned cores.
    if (!sched_ctx_.is_completed()) {
        if (rt == nullptr) {
            LOG_ERROR("Thread %d: rt is null after orchestrator error, skipping dispatch", thread_idx);
        } else {
            sched_ctx_.bind_runtime(rt);
            // 3S+1P: the last thread is the core-less resolution (P) thread; the
            // rest are core-owning schedulers (S).
            int32_t completed = (thread_idx == sched_ctx_.p_thread_idx()) ?
                                    sched_ctx_.run_resolution_thread(runtime, thread_idx) :
                                    sched_ctx_.resolve_and_dispatch(runtime, thread_idx);
            if (completed < 0) {
                LOG_ERROR("Thread %d: Scheduler failed with rc=%d", thread_idx, completed);
                run_rc = completed;
            } else {
                LOG_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
            }
        }
    }

    // Always shutdown AICore — even if sched_ctx_.completed_ was already true.
    // platform_deinit_aicore_regs is idempotent.
    int32_t shutdown_rc = sched_ctx_.shutdown(thread_idx);
    if (shutdown_rc != 0 && run_rc == 0) {
        run_rc = shutdown_rc;
    }

    LOG_INFO("Thread %d: Completed", thread_idx);

    completion_gate_.arrive_and_finalize_if_last(aicpu_thread_num_, [&] {
        aicpu_publish_task_timing_tail_usage(aicpu_thread_num_);
        // Destroy the host_build_graph runtime. sm_handle / rt are recreated
        // every run, so always tear them down here.
        if (rt != nullptr) {
            rt->scheduler->print_queues();
            // Clear g_current_runtime in this DSO before destroying rt.
            framework_bind_runtime(nullptr);
            // A Graph's expansion storage is the tail of its outer task's heap
            // allocation, so it retires with that allocation; nothing here owns
            // a separate block to release.
            runtime_destroy(rt, runtime_arena_);
            rt = nullptr;
        }
    });

    return run_rc;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all SchedulerContext-owned state in one place.
    sched_ctx_.deinit();

    completion_gate_.reset();
    runtime_init_ready_.store(false, std::memory_order_release);

    aicpu_thread_num_ = 0;

    // Clear the file-scope runtime pointer (freed by the last scheduler thread before deinit).
    rt = nullptr;

    LOG_INFO("DeInit: Runtime execution state reset");

    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    hs_setup_done_.store(false, std::memory_order_release);
    hs_arrived_.store(0, std::memory_order_release);
    hs_thread_seq_.store(0, std::memory_order_release);
    classify_ready_.store(false, std::memory_order_release);
    classify_arrived_.store(0, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);

    LOG_INFO("DeInit: AicpuExecutor reset complete");
}

// ===== Public Entry Point =====

extern "C" int32_t aicpu_prewarm_callable(Runtime *runtime) {
    // host_build_graph host-orch: the orchestration .so is dlopen'd on the HOST
    // during prepare_callable_impl and the whole task graph is built host-side,
    // so there is no device-side orchestrator .so to pre-load — prewarm is a
    // no-op. The symbol is retained because the platform onboard kernel
    // (src/a2a3/platform/onboard/aicpu/kernel.cpp) links it strongly via
    // simpler_aicpu_prewarm_callable; removing it would break the onboard link.
    (void)runtime;
    return 0;
}

/**
 * aicpu_execute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor: all threads enter init(), which handshakes the cores
 *    in parallel and barriers internally until init is complete (or a thread
 *    failed); its return value is authoritative on every thread.
 * 2. Execute tasks on managed cores
 * 3. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure
 * @return 0 on success, non-zero on error
 */
extern "C" int32_t aicpu_execute(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    LOG_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

    // init() barriers every thread internally until init is complete on the
    // leader (or a thread failed), then returns the status — so a non-zero
    // return is authoritative on all threads and no extra spin is needed.
    if (g_aicpu_executor.init(runtime) != 0) {
        LOG_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
        return -1;
    }

    int32_t rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
    }

    int32_t runtime_rc = read_pto2_runtime_status(runtime);

    // The finalizer publishes cleanup eligibility only after runtime destruction.
    if (g_aicpu_executor.completion_gate_.claim_cleanup()) {
        LOG_INFO("aicpu_execute: All threads finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    if (runtime_rc != 0) {
        LOG_ERROR("aicpu_execute: HBG runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    if (rc != 0) {
        return rc;
    }

    LOG_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
