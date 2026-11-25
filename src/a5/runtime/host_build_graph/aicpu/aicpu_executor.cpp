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
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "aicore_lifecycle.h"
#include "aicpu_executor.h"
#include "aicpu/abort_aicore_startup.h"
#include "args_dump_service.h"
#include "calendar_task_timing.h"
#include "supervisor.h"
#include "aicore_scheduler_error.h"
#include "aicpu/aicpu_device_config.h"
#include "aicpu/cache_maintenance.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/device_run_result_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/device_time.h"
#include "host_build_graph/runtime_status.h"
#include "task_image_status.h"
#include "runtime.h"
#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler_watchdog.h"
#include "spin_hint.h"

#include "common/run_terminal_accumulator.h"
#include "common/unified_log.h"
#include "platform_comm/ccu/ccu_ready.h"

// Register-based communication
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"
#include "common/memory_barrier.h"
#include "utils/thread_completion_gate.h"

static AicpuExecutor g_aicpu_executor;

extern "C" __attribute__((visibility("default"))) void aicpu_reset_execution_entry() {
    g_aicpu_executor.reset_entry_state();
}

// HBG orchestration handles belong to the host. Per-run deinit releases its
// device execution state, so cold close has no persistent callable to destroy.
extern "C" int32_t aicpu_release_callables() {
    return g_aicpu_executor.init_done_.load(std::memory_order_acquire) ? -1 : 0;
}

static uint64_t scheduler_timeout_cycles() {
    const int32_t timeout_ms = get_scheduler_timeout_ms();
    return timeout_ms > 0 ? static_cast<uint64_t>(timeout_ms) * (PLATFORM_PROF_SYS_CNT_FREQ / 1000) :
                            SCHEDULER_TIMEOUT_CYCLES;
}

template <typename Predicate>
static bool wait_with_scheduler_timeout(Predicate predicate) {
    const uint64_t start = get_sys_cnt_aicpu();
    const uint64_t timeout_cycles = scheduler_timeout_cycles();
    while (!predicate()) {
        if (scheduler_watchdog_expired(start, get_sys_cnt_aicpu(), timeout_cycles)) return false;
        SPIN_WAIT_HINT();
    }
    return true;
}

static void record_lifecycle_timeout(Runtime *runtime, const char *site) {
    if (runtime != nullptr && runtime->get_gm_sm_ptr() != nullptr) {
        auto *error_code = task_image_error_code(runtime->get_gm_sm_ptr());
        (void)latch_aicore_scheduler_runtime_error(error_code, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    }
    LOG_ERROR("A5 HBG AICore lifecycle timeout at %s", site);
}

template <typename Predicate>
static void join_initialization_phase(Runtime *runtime, Predicate predicate, std::atomic<bool> &failed) {
    if (wait_with_scheduler_timeout(predicate)) return;
    record_lifecycle_timeout(runtime, "initialization barrier");
    failed.store(true, std::memory_order_release);
    // Phase owners may still write lifecycle state. Retirement and destruction
    // require their completion even when the run has already failed.
    while (!predicate())
        SPIN_WAIT_HINT();
}

static int32_t read_runtime_status(Runtime *runtime) {
    if (runtime == nullptr || runtime->get_gm_sm_ptr() == nullptr) return 0;
    auto *error_code = task_image_error_code(runtime->get_gm_sm_ptr());
    int32_t sched_error_code = error_code->load(std::memory_order_acquire);
    return runtime_status_from_error_code(sched_error_code);
}

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        return -1;
    }

    // All AICPU threads enter init. The per-core AICore handshake is the
    // dominant preamble cost (serial MMIO, ~217 µs of ~283 µs for 72 cores), so
    // it is parallelized: the leader (tidx 0) does the shared setup, every
    // thread handshakes a disjoint slice of cores, then the leader finishes init
    // across two barriers: discovery precedes configuration, and all register
    // releases precede init completion.
    int32_t nthreads = runtime->dev.aicpu_thread_num;
    if (nthreads == 0) nthreads = 1;
    if (nthreads < 1 || nthreads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid aicpu_thread_num: %d", nthreads);
        int32_t callback = platform_aicpu_affinity_thread_idx();
        if (callback < 0) callback = hs_thread_seq_.fetch_add(1, std::memory_order_acq_rel);
        if (callback == 0) {
            if (runtime->get_gm_sm_ptr() != nullptr) {
                auto *error_code = task_image_error_code(runtime->get_gm_sm_ptr());
                (void)latch_aicore_scheduler_runtime_error(
                    error_code, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
                );
            }
            const int32_t shutdown_rc =
                abort_aicore_startup(runtime, get_platform_regs(), get_platform_run_result_epoch());
            terminal_.record_participant(-SIMPLER_ERROR_INVALID_ARGS, shutdown_rc);
            terminal_publisher_.take(run_terminal_select(false, read_runtime_status(runtime), terminal_), nullptr, 0);
            handoff_run_terminal();
        }
        return -SIMPLER_ERROR_INVALID_ARGS;
    }
    // Each thread needs a distinct index in [0, nthreads) to pick the leader and
    // partition the cores. Onboard the gate filter assigns it (exec_idx); sim's
    // gate does not, so platform_aicpu_affinity_thread_idx() is -1 here for every
    // thread — hand those a distinct index from a counter (mirrors run()'s
    // thread_idx_++ fallback) instead of collapsing them all to leader 0, which
    // would run pre_/post_handshake_init on every thread and race the shared
    // lifecycle state. Exactly nthreads threads reach init (the gate drops the
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
        const uint64_t ready_start = get_sys_cnt_aicpu();
        const int ready_rc = simpler::ccu::wait_local_ready(
            runtime->dev.ccu_workspace_addr,
            [ready_start] {
                return get_sys_cnt_aicpu() - ready_start > PLATFORM_PROF_SYS_CNT_FREQ * uint64_t{10};
            },
            [] {
                SPIN_WAIT_HINT();
            }
        );
        if (ready_rc != 0) {
            LOG_ERROR("Local CCU READY initialization failed: %d", ready_rc);
            init_failed_.store(true, std::memory_order_release);
        }
        args_dump_services_allocated_this_init_ = false;
        if (args_dump_services_ != nullptr) init_failed_.store(true, std::memory_order_release);
        // The 0 → 1 fixup already applied above.
        aicpu_thread_num_ = nthreads;

        hs_arrived_.store(0, std::memory_order_relaxed);
        hs_release_arrived_.store(0, std::memory_order_relaxed);
        if (aicore_lifecycle_.pre_handshake_init(runtime, aicpu_thread_num_, get_platform_regs()) != 0) {
            init_failed_.store(true, std::memory_order_release);
        }
        hs_setup_done_.store(true, std::memory_order_release);
    } else {
        join_initialization_phase(
            runtime,
            [this] {
                return hs_setup_done_.load(std::memory_order_acquire);
            },
            init_failed_
        );
    }

    // All threads: handshake this thread's slice of cores in parallel.
    if (hs_setup_done_.load(std::memory_order_acquire) && aicore_lifecycle_.core_count() > 0) {
        aicore_lifecycle_.handshake_partition(runtime, tidx, nthreads);
    }

    // Barrier 1: every thread discovers its core slice before the leader builds
    // and publishes the topology-dependent execution configuration.
    hs_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (is_leader) {
        join_initialization_phase(
            runtime,
            [this, nthreads] {
                return hs_arrived_.load(std::memory_order_acquire) >= nthreads;
            },
            init_failed_
        );
        completion_gate_.reset();
        if (aicore_lifecycle_.validate_discovery() != 0 || !schedules_on_aicore(runtime)) {
            LOG_ERROR("A5 HBG: invalid scheduler bootstrap");
            if (runtime->get_gm_sm_ptr() != nullptr) {
                auto *error_code = task_image_error_code(runtime->get_gm_sm_ptr());
                (void)latch_aicore_scheduler_runtime_error(
                    error_code, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
                );
            }
            init_failed_.store(true, std::memory_order_release);
        }
        if (!init_failed_.load(std::memory_order_acquire) &&
            (post_handshake_init(runtime) != 0 || aicore_lifecycle_.initialize_pmu(runtime) != 0)) {
            init_failed_.store(true, std::memory_order_release);
        }
        if (!init_failed_.load(std::memory_order_acquire)) {
            auto *control = simpler::percounter::percounter_control(runtime);
            if (control != nullptr && control->args_dump_offset != 0) {
                auto *dump_image = simpler::percounter::percounter_region<void>(control, control->args_dump_offset);
                if (!initialize_resident_args_dump_outputs(
                        dump_image, runtime->dev.callable_table_addr_, runtime->dev.callable_table_len_
                    ))
                    init_failed_.store(true, std::memory_order_release);
                args_dump_services_ = new (std::nothrow) ResidentArgsDumpService[nthreads];
                if (args_dump_services_ == nullptr) init_failed_.store(true, std::memory_order_release);
                else {
                    args_dump_service_count_ = nthreads;
                    args_dump_services_allocated_this_init_ = true;
                    for (int32_t thread = 0; thread < nthreads; ++thread)
                        if (!args_dump_services_[thread].initialize(
                                dump_image, runtime->dev.callable_table_addr_, runtime->dev.callable_table_len_, thread,
                                nthreads
                            ))
                            init_failed_.store(true, std::memory_order_release);
                }
            }
        }
        hs_config_done_.store(true, std::memory_order_release);
    } else {
        join_initialization_phase(
            runtime,
            [this] {
                return hs_config_done_.load(std::memory_order_acquire);
            },
            init_failed_
        );
    }

    // The leader publishes START in SoC SRAM after topology, dependency counters and diagnostics are ready.
    if (!init_failed_.load(std::memory_order_acquire)) {
        publish_partition(runtime, tidx, nthreads);
    }

    // Failed initialization signals EXIT and joins the per-core retirement acknowledgements.
    const bool started = !init_failed_.load(std::memory_order_acquire);
    int32_t shutdown_rc = aicore_lifecycle_.release_partition(runtime, tidx, started);
    if (shutdown_rc != 0) {
        init_failed_.store(true, std::memory_order_release);
    }

    // Barrier 2: init cannot return until every successful slice is released,
    // or every failed slice has acknowledged EXIT.
    hs_release_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (is_leader) {
        join_initialization_phase(
            runtime,
            [this, nthreads] {
                return hs_release_arrived_.load(std::memory_order_acquire) >= nthreads;
            },
            init_failed_
        );
        init_done_.store(true, std::memory_order_release);
        if (!init_failed_.load(std::memory_order_acquire)) LOG_INFO("AicpuExecutor: Init complete");
    } else {
        // The leader owns the final failure decision. Participants acquire it
        // without introducing another failure after a peer enters run().
        while (!init_done_.load(std::memory_order_acquire))
            SPIN_WAIT_HINT();
    }
    if (init_failed_.load(std::memory_order_acquire)) {
        if (started) {
            aicore_lifecycle_.signal_shutdown_partition(tidx);
            shutdown_rc = aicore_lifecycle_.finish_shutdown_partition(tidx, runtime);
        }
        return finish_failed_init(runtime, shutdown_rc);
    }
    return 0;
}

int32_t AicpuExecutor::finish_failed_init(Runtime *runtime, int32_t shutdown_rc) {
    if (schedules_on_aicore(runtime)) (void)settle_run_error(runtime);
    if (runtime->get_gm_sm_ptr() != nullptr) {
        auto *error_code = task_image_error_code(runtime->get_gm_sm_ptr());
        (void)latch_aicore_scheduler_runtime_error(
            error_code, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
        );
    }
    const int32_t header_status = read_runtime_status(runtime);
    const int32_t init_rc = header_status != 0 ? header_status : -SIMPLER_ERROR_INVALID_ARGS;
    terminal_.record_participant(init_rc, shutdown_rc);
    const int32_t nthreads = aicpu_thread_num_;
    const bool last = init_failure_arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == nthreads;
    if (last) {
        terminal_publisher_.take(run_terminal_select(false, header_status, terminal_), nullptr, 0);
        if (args_dump_services_allocated_this_init_ && aicore_lifecycle_.cores_retired()) {
            for (int32_t thread = 0; thread < args_dump_service_count_; ++thread)
                args_dump_services_[thread].stop();
            if (args_dump_services_stopped()) {
                delete[] args_dump_services_;
                args_dump_services_ = nullptr;
                args_dump_service_count_ = 0;
                args_dump_services_allocated_this_init_ = false;
            }
        }
        (void)aicore_lifecycle_.settle_run_diagnostics(runtime, false);
        aicore_lifecycle_.deinit();
        completion_gate_.reset();
        shutdown_ready_.store(false, std::memory_order_relaxed);
        shutdown_signaled_.store(0, std::memory_order_relaxed);
        run_status_.store(0, std::memory_order_relaxed);
        init_done_.store(false, std::memory_order_relaxed);
        hs_setup_done_.store(false, std::memory_order_relaxed);
        hs_config_done_.store(false, std::memory_order_relaxed);
        hs_arrived_.store(0, std::memory_order_relaxed);
        hs_release_arrived_.store(0, std::memory_order_relaxed);
        hs_thread_seq_.store(0, std::memory_order_relaxed);
        thread_idx_.store(0, std::memory_order_relaxed);
        aicpu_thread_num_ = 0;
        init_failure_arrived_.store(0, std::memory_order_relaxed);
        init_failed_.store(false, std::memory_order_release);
        handoff_run_terminal();
    } else if (!wait_with_scheduler_timeout([this] {
                   return !init_failed_.load(std::memory_order_acquire);
               })) {
        record_lifecycle_timeout(runtime, "initialization barrier");
    }
    return init_rc;
}

bool AicpuExecutor::service_args_dump(int32_t thread_idx) {
    return args_dump_services_ != nullptr && args_dump_services_[thread_idx].step();
}

int32_t AicpuExecutor::run(Runtime *runtime) {
    int32_t affinity_exec_idx = platform_aicpu_affinity_thread_idx();
    int32_t thread_idx = (affinity_exec_idx >= 0) ? affinity_exec_idx : (thread_idx_++);
    if (thread_idx < 0 || thread_idx >= aicpu_thread_num_ || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Thread index %d out of bounds (active=%d max=%d exec_idx=%d)", thread_idx, aicpu_thread_num_,
            PLATFORM_MAX_AICPU_THREADS, affinity_exec_idx
        );
        return -1;
    }
    if (thread_idx == aicpu_thread_num_ - 1) {
        run_status_.store(supervise(runtime, thread_idx), std::memory_order_release);
        shutdown_ready_.store(true, std::memory_order_release);
    } else {
        await_shutdown(runtime, thread_idx);
    }

    int32_t run_rc = run_status_.load(std::memory_order_acquire);
    if (args_dump_services_ != nullptr) args_dump_services_[thread_idx].stop();
    aicore_lifecycle_.signal_shutdown_partition(thread_idx);
    shutdown_signaled_.fetch_add(1, std::memory_order_acq_rel);
    if (!wait_with_scheduler_timeout([this] {
            return shutdown_signaled_.load(std::memory_order_acquire) >= aicpu_thread_num_;
        })) {
        record_lifecycle_timeout(runtime, "retirement barrier");
        if (run_rc == 0) run_rc = -SIMPLER_ERROR_SCHEDULER_TIMEOUT;
    }
    int32_t shutdown_rc = aicore_lifecycle_.finish_shutdown_partition(thread_idx, runtime);
    // Both outcomes reach the terminal record before this thread's arrival, and
    // with the state that produced each: folding shutdown_rc into run_rc first
    // would publish a teardown failure as an execution one. Every exit above
    // carries a code of its own today, so every arrival claims the audited
    // path — the claim is still counted rather than assumed, so a future exit
    // that reaches here without a code stops the run being publishable as Ok
    // instead of silently keeping it.
    terminal_.record_participant(run_rc, shutdown_rc);
    normal_path_claims_.fetch_add(1, std::memory_order_acq_rel);
    if (shutdown_rc != 0 && run_rc == 0) run_rc = shutdown_rc;
    completion_gate_.arrive_and_finalize_if_last(aicpu_thread_num_, [&] {
        // Every other participant's writes are visible here — the gate's
        // arrival is acq_rel and this is the thread that closed it — so this is
        // the one point where the run's final state can be read. Deciding the
        // record here and committing it after teardown is what keeps a teardown
        // fault from landing behind an already-published success.
        //
        // Two separate questions below, and this arrival answers neither. Did
        // every core stop — which is what licenses reading anything a core
        // wrote — and did the graph complete, which is what says how much data
        // the run owed. A partition that could not retire its cores returns
        // that failure and still arrives here.
        const bool cores_retired = aicore_lifecycle_.cores_retired();
        (void)aicore_lifecycle_.settle_run_diagnostics(
            runtime, run_status_.load(std::memory_order_acquire) == 0 && args_dump_services_stopped()
        );
        // The record is published either way — a run the host cannot read a
        // verdict for is worse than one that reports a failure — but a run
        // whose cores did not prove retirement did not finish the audited path,
        // so it cannot be published as Ok.
        snapshot_run_terminal(runtime, cores_retired);
        if (cores_retired) {
            AicoreWorkWindow work_window{};
            if (aicore_lifecycle_.retired_work_window(work_window, true)) {
                device_phase_set_aicore_work_window(work_window);
            }
            simpler::percounter::publish_calendar_task_timing(
                simpler::percounter::percounter_control(runtime), runtime->get_worker_count()
            );
        } else {
            // Device-owned trace cells, written by the cores. Reading them
            // while one of them may still be writing would publish this run's
            // timing from a half-written cell.
            LOG_ERROR(
                "%s", "A5 HBG AICore Scheduler: task timing is not published for a run whose cores did not prove "
                      "retirement"
            );
        }
    });
    return run_rc;
}

void AicpuExecutor::snapshot_run_terminal(Runtime *runtime, bool cores_retired) {
    // The shared header outranks a participant's own return: the supervisor
    // latches the AICore scheduler's error there, and a peer thread that only
    // waited on the shutdown barrier returns zero for the same run.
    //
    // The header is settled here rather than read as the supervisor left it.
    // The supervisor's last look at `scheduler_error` precedes the AICore's
    // exit wait, whose own watchdog can claim that error and then acknowledge
    // exit normally — leaving the shutdown successful, every participant's rc
    // zero, and the error carried by nothing. This is the boundary where every
    // participant has finished its shutdown partition, so one settle covers
    // every core.
    const int32_t header_status = settle_run_error(runtime);
    // The header is one cache line a core may still be writing when retirement
    // was not proved, so what it returns cannot make such a run Ok. It can
    // still make it worse: `run_terminal_select` reports a header or
    // participant error whatever the audited-path answer is, which is why the
    // read stays and only the verdict is withheld.
    const bool normal_path = cores_retired && normal_path_claims_.load(std::memory_order_acquire) == aicpu_thread_num_;
    terminal_publisher_.take(run_terminal_select(normal_path, header_status, terminal_), nullptr, 0);
}

void AicpuExecutor::handoff_run_terminal() {
    if (!terminal_publisher_.handoff()) {
        // Nothing to publish is the normal shape for a run that never reached
        // its rendezvous; a refused record is a producer defect. Either way the
        // host reads the run as undecided, so say which one happened.
        LOG_INFO("%s", "aicpu_execute: no terminal record handed off for this run");
    }
}

void AicpuExecutor::reset_entry_state() {
    // Only invocation-private coordination is reset. The terminal publisher
    // and resource owners remain valid through teardown and publication.
    completion_gate_.reset();
    terminal_.reset();
    normal_path_claims_.store(0, std::memory_order_release);
    shutdown_ready_.store(false, std::memory_order_release);
    shutdown_signaled_.store(0, std::memory_order_release);
    run_status_.store(0, std::memory_order_release);
    aicpu_thread_num_ = 0;
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    hs_setup_done_.store(false, std::memory_order_release);
    hs_config_done_.store(false, std::memory_order_release);
    hs_arrived_.store(0, std::memory_order_release);
    hs_release_arrived_.store(0, std::memory_order_release);
    hs_thread_seq_.store(0, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    init_failure_arrived_.store(0, std::memory_order_release);
}

void AicpuExecutor::deinit(Runtime * /*runtime*/) {
    if (aicore_lifecycle_.cores_retired() && args_dump_services_stopped()) {
        delete[] args_dump_services_;
        args_dump_services_ = nullptr;
        args_dump_service_count_ = 0;
        args_dump_services_allocated_this_init_ = false;
    }
    aicore_lifecycle_.deinit();

    reset_entry_state();
    LOG_INFO("DeInit: Runtime execution state reset");

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
    const int32_t init_rc = g_aicpu_executor.init(runtime);
    if (init_rc != 0) {
        LOG_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
        const int32_t runtime_rc = g_aicpu_executor.settle_run_error(runtime);
        return runtime_rc != 0 ? runtime_rc : init_rc;
    }

    int32_t rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
    }

    int32_t runtime_rc = read_runtime_status(runtime);

    // The finalizer publishes cleanup eligibility only after runtime destruction.
    if (g_aicpu_executor.completion_gate_.claim_cleanup()) {
        LOG_INFO("aicpu_execute: All threads finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
        g_aicpu_executor.handoff_run_terminal();
    }

    if (runtime_rc != 0) {
        LOG_ERROR("aicpu_execute: runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    if (rc != 0) {
        return rc;
    }

    LOG_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
