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
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "aicpu/orch_so_file.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Performance profiling headers
#include "aicpu/performance_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "common/memory_barrier.h"
#include "common/perf_profiling.h"
#include "common/unified_log.h"

// Register-based communication
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

// Core type definitions
#include "common/core_type.h"

// CoreCallable for resolved dispatch address
#include "callable.h"

// Scheduler data structures (CoreExecState, CoreTracker, etc.)
#include "scheduler/scheduler_types.h"

// Scheduler context class
#include "scheduler/scheduler_context.h"

// Device orchestration function signature (loaded via dlopen).
// The executor binds the current thread's PTO2Runtime into orchestration TLS
// before calling the user entry.
typedef void (*DeviceOrchestrationFunc)(const ChipStorageTaskArgs &orch_args);
typedef void (*DeviceOrchestrationBindRuntimeFunc)(PTO2Runtime *rt);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(const ChipStorageTaskArgs &orch_args);

// From orchestration/common.cpp linked into this DSO — updates g_pto2_current_runtime here (distinct from
// pto2_framework_bind_runtime in the dlopen'd libdevice_orch_*.so).
extern "C" void pto2_framework_bind_runtime(PTO2Runtime *rt);

constexpr const char *DEFAULT_ORCH_ENTRY_SYMBOL = "aicpu_orchestration_entry";
constexpr const char *DEFAULT_ORCH_CONFIG_SYMBOL = "aicpu_orchestration_config";

static int32_t read_pto2_runtime_status(Runtime *runtime) {
    if (runtime == nullptr) {
        return 0;
    }

    void *sm = runtime->get_pto2_gm_sm_ptr();
    if (sm == nullptr) {
        return 0;
    }

    auto *header = static_cast<PTO2SharedMemoryHeader *>(sm);
    int32_t orch_error_code = header->orch_error_code.load(std::memory_order_acquire);
    int32_t sched_error_code = header->sched_error_code.load(std::memory_order_acquire);
    return pto2_runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static PTO2Runtime *rt{nullptr};

struct AicpuExecutor {
    int32_t sched_thread_num_;
    int32_t active_sched_threads_{0};  // Threads currently in dispatch loop (initially sched_thread_num_, becomes
                                       // thread_num_ after orch→sched transition)
    bool orch_to_sched_{false};

    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int32_t thread_num_{0};
    int32_t cores_total_num_{0};
    int32_t thread_cores_num_{0};  // Cores per scheduler thread (0 for orchestrator when thread_num_==4)
    int32_t core_count_per_thread_[MAX_AICPU_THREADS];  // Actual core count per thread
    int32_t core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // Cluster-ordered worker_id lists for core assignment (init-only)
    int32_t aic_worker_ids_[MAX_CORES_PER_THREAD];
    int32_t aiv_worker_ids_[MAX_CORES_PER_THREAD];
    int32_t aic_count_{0};
    int32_t aiv_count_{0};

    // Platform register base address array (set via get_platform_regs())
    uint64_t regs_{0};

    // ===== Task queue state (managed by scheduler ready queues) =====

    // Task execution tracking
    std::atomic<int32_t> completed_tasks_{0};
    int32_t total_tasks_{0};
    std::atomic<int32_t> finished_count_{0};
    // Device orchestration: set by last orchestrator when graph is built; schedulers poll it.
    // volatile prevents the compiler from hoisting the load out of spin loops.
    volatile bool orchestrator_done_{false};
    std::atomic<bool> runtime_init_ready_{false};

    // ===== Dynamic core transition state =====
    std::atomic<bool> transition_requested_{false};
    std::atomic<int32_t> wait_reassign_{0};
    std::atomic<bool> reassigned_{false};
    std::atomic<bool> completed_{false};

    // Orchestration SO handle - defer dlclose until all tasks complete
    void *orch_so_handle_{nullptr};
    char orch_so_path_[256]{};  // Path to orchestration SO file for cleanup

    // Shared orchestration function pointer (loaded by first orch thread, used by all)
    DeviceOrchestrationFunc orch_func_{nullptr};
    DeviceOrchestrationBindRuntimeFunc orch_bind_runtime_{nullptr};
    const ChipStorageTaskArgs *orch_args_cached_{nullptr};

    uint64_t *func_id_to_addr_;
    uint64_t get_function_bin_addr(int func_id) const {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t handshake_all_cores(Runtime *runtime);
    bool assign_cores_to_threads();
    void reassign_cores_for_all_threads();
    int32_t shutdown_aicore(Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
    void emergency_shutdown(Runtime *runtime);
    void diagnose_stuck_state(
        Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num, Handshake *hank
    );
};

static AicpuExecutor g_aicpu_executor;

static void emergency_shutdown_callback(Runtime *runtime) { g_aicpu_executor.emergency_shutdown(runtime); }

// ===== AicpuExecutor Method Implementations =====

/**
 * Handshake with all cores and discover their types
 * Sets up register addresses for fast dispatch.
 */
int32_t AicpuExecutor::handshake_all_cores(Runtime *runtime) {
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
    cores_total_num_ = runtime->worker_count;

    // Validate cores_total_num_ before using as array index
    if (cores_total_num_ == 0 || cores_total_num_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Invalid cores_total_num %d (expected 1-%d)", cores_total_num_, MAX_CORES_PER_THREAD);
        return -1;
    }

    aic_count_ = 0;
    aiv_count_ = 0;

    DEV_INFO("Handshaking with %d cores", cores_total_num_);

    // Step 1: Write per-core payload addresses and send handshake signal
    // OUT_OF_ORDER_STORE_BARRIER() ensures task is globally visible before
    // aicpu_ready=1, so AICore reads the correct payload pointer after waking up.
    for (int32_t i = 0; i < cores_total_num_; i++) {
        all_handshakes[i].task = reinterpret_cast<uint64_t>(&sched_ctx_.payload_per_core_[i][0]);
        OUT_OF_ORDER_STORE_BARRIER();
        all_handshakes[i].aicpu_ready = 1;
    }
    OUT_OF_ORDER_STORE_BARRIER();

    // Get platform physical cores count for validation
    uint32_t max_physical_cores_count = platform_get_physical_cores_count();

    // Step 2: Wait for all cores to respond, collect core type and register addresses
    bool handshake_failed = false;
    for (int32_t i = 0; i < cores_total_num_; i++) {
        Handshake *hank = &all_handshakes[i];

        while (hank->aicore_regs_ready == 0) {}

        uint32_t physical_core_id = hank->physical_core_id;

        // Validate physical_core_id before using as array index
        if (physical_core_id >= max_physical_cores_count) {
            DEV_ERROR(
                "Core %d reported invalid physical_core_id=%u (platform max=%u)", i, physical_core_id,
                max_physical_cores_count
            );
            handshake_failed = true;
            continue;
        }

        // Get register address using physical_core_id
        uint64_t *regs = reinterpret_cast<uint64_t *>(regs_);
        uint64_t reg_addr = regs[physical_core_id];

        // Initialize AICore registers after discovery (first round)
        platform_init_aicore_regs(reg_addr);
        OUT_OF_ORDER_STORE_BARRIER();
        hank->aicpu_regs_ready = 1;

        OUT_OF_ORDER_STORE_BARRIER();

        while (hank->aicore_done == 0) {}

        CoreType type = hank->core_type;

        sched_ctx_.core_exec_states_[i].reg_addr = reg_addr;
#if !PTO2_PROFILING
        sched_ctx_.core_exec_states_[i].worker_id = i;
        sched_ctx_.core_exec_states_[i].physical_core_id = physical_core_id;
        sched_ctx_.core_exec_states_[i].core_type = type;
#endif

        if (type == CoreType::AIC) {
            aic_worker_ids_[aic_count_++] = i;
            DEV_INFO("Core %d: AIC, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        } else {
            aiv_worker_ids_[aiv_count_++] = i;
            DEV_INFO("Core %d: AIV, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        }
    }

    if (handshake_failed) {
        emergency_shutdown(runtime);
        return -1;
    }

    DEV_INFO("Core discovery complete: %d AIC, %d AIV", aic_count_, aiv_count_);
    return 0;
}

/**
 * Assign discovered cores to scheduler threads
 * (Aligned with host_build_graph mechanism)
 */
bool AicpuExecutor::assign_cores_to_threads() {
    // Cluster-aligned round-robin assignment: cluster ci -> sched thread ci % active_sched_threads_.
    // Each cluster = 1 AIC + 2 adjacent AIV; the triple is always kept together.
    active_sched_threads_ = (sched_thread_num_ > 0) ? sched_thread_num_ : thread_num_;
    int32_t cluster_count = aic_count_;

    // Max clusters any single sched thread can hold: ceil(cluster_count / active_sched_threads_).
    int32_t max_clusters_per_thread = (cluster_count + active_sched_threads_ - 1) / active_sched_threads_;
    thread_cores_num_ = max_clusters_per_thread * 3;

    if (thread_cores_num_ > CoreTracker::MAX_CORE_PER_THREAD) {
        DEV_ERROR("Can't assign more then 64 cores in per scheduler");
        return false;
    }

    DEV_INFO(
        "Assigning cores (round-robin): %d clusters across %d sched threads (%d AIC, %d AIV)", cluster_count,
        active_sched_threads_, aic_count_, aiv_count_
    );

    for (int32_t i = 0; i < MAX_CORES_PER_THREAD; i++) {
        sched_ctx_.core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
        sched_ctx_.core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // Count clusters per thread first (round-robin may distribute unevenly)
    int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        clusters_per_thread[ci % active_sched_threads_]++;
    }
    for (int32_t i = 0; i < active_sched_threads_; i++) {
        sched_ctx_.core_trackers_[i].init(clusters_per_thread[i]);
        core_count_per_thread_[i] = 0;
    }

    // Per-sched-thread running core index used while filling core_assignments_.
    int32_t core_idx[MAX_AICPU_THREADS] = {};
    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};

    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % active_sched_threads_;
        int32_t &idx = core_idx[t];

        int32_t aic_wid = aic_worker_ids_[ci];
        int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
        int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

        sched_ctx_.core_trackers_[t].set_cluster(cluster_idx_per_thread[t]++, aic_wid, aiv0_wid, aiv1_wid);

        core_assignments_[t][idx++] = aic_wid;
        core_assignments_[t][idx++] = aiv0_wid;
        core_assignments_[t][idx++] = aiv1_wid;

        DEV_INFO("Thread %d: cluster %d (AIC=%d, AIV0=%d, AIV1=%d)", t, ci, aic_wid, aiv0_wid, aiv1_wid);
    }

    for (int32_t t = 0; t < thread_num_; t++) {
        core_count_per_thread_[t] = core_idx[t];
        DEV_INFO(
            "Thread %d: total %d cores (%d clusters)", t, core_idx[t], sched_ctx_.core_trackers_[t].get_cluster_count()
        );
    }

    return true;
}

/**
 * Reassign all cores evenly across all threads (schedulers + orchestrators).
 * Called by the last orchestrator thread when orchestration completes.
 * Writes into new_core_assignments_ / new_core_count_per_thread_.
 */
void AicpuExecutor::reassign_cores_for_all_threads() {
    DEV_INFO("Reassigning cores (cluster-aligned) for %d threads: %d AIC, %d AIV", thread_num_, aic_count_, aiv_count_);

    // Collect running worker_ids from all current trackers
    bool running_cores[MAX_CORES_PER_THREAD] = {};
    for (int32_t i = 0; i < thread_num_; i++) {
        auto all_running = sched_ctx_.core_trackers_[i].get_all_running_cores();
        int32_t bp;
        while ((bp = all_running.pop_first()) >= 0) {
            running_cores[sched_ctx_.core_trackers_[i].get_core_id_by_offset(bp)] = true;
        }
    }

    // Count clusters per thread (round-robin across all threads)
    int32_t cluster_count = aic_count_;
    int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        clusters_per_thread[ci % thread_num_]++;
    }

    // Re-init all trackers and reset core counts
    for (int32_t i = 0; i < thread_num_; i++) {
        sched_ctx_.core_trackers_[i].init(clusters_per_thread[i]);
        core_count_per_thread_[i] = 0;
    }

    // Assign clusters round-robin and restore running state
    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % thread_num_;

        int32_t aic_wid = aic_worker_ids_[ci];
        int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
        int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

        int32_t cl_idx = cluster_idx_per_thread[t]++;
        sched_ctx_.core_trackers_[t].set_cluster(cl_idx, aic_wid, aiv0_wid, aiv1_wid);

        // init() marks all idle; toggle cores that were running and restore pending_occupied
        if (running_cores[aic_wid]) {
            sched_ctx_.core_trackers_[t].change_core_state(cl_idx * 3);
            sched_ctx_.core_trackers_[t].set_pending_occupied(cl_idx * 3);
        }
        if (running_cores[aiv0_wid]) {
            sched_ctx_.core_trackers_[t].change_core_state(cl_idx * 3 + 1);
            sched_ctx_.core_trackers_[t].set_pending_occupied(cl_idx * 3 + 1);
        }
        if (running_cores[aiv1_wid]) {
            sched_ctx_.core_trackers_[t].change_core_state(cl_idx * 3 + 2);
            sched_ctx_.core_trackers_[t].set_pending_occupied(cl_idx * 3 + 2);
        }

        core_assignments_[t][core_count_per_thread_[t]++] = aic_wid;
        core_assignments_[t][core_count_per_thread_[t]++] = aiv0_wid;
        core_assignments_[t][core_count_per_thread_[t]++] = aiv1_wid;
    }

    // Log final distribution
    DEV_INFO("Core reassignment complete:");
    for (int32_t t = 0; t < thread_num_; t++) {
        int32_t aic_running = sched_ctx_.core_trackers_[t].get_running_count<CoreType::AIC>();
        int32_t aiv_running = sched_ctx_.core_trackers_[t].get_running_count<CoreType::AIV>();
        DEV_INFO(
            "  Thread %d: %d cores, %d clusters (AIC running=%d, AIV running=%d)", t, core_count_per_thread_[t],
            sched_ctx_.core_trackers_[t].get_cluster_count(), aic_running, aiv_running
        );
    }
    active_sched_threads_ = thread_num_;
    sched_ctx_.active_sched_threads_ = thread_num_;
}

int32_t AicpuExecutor::init(Runtime *runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO("AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        DEV_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    func_id_to_addr_ = runtime->func_id_to_addr_;

    // Read execution parameters from runtime
    thread_num_ = runtime->sche_cpu_num;
    sched_thread_num_ = thread_num_ - 1;
    orch_to_sched_ = runtime->orch_to_sched;
    if (thread_num_ == 0) thread_num_ = 1;

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Zero all per-core execution state before handshake
    memset(sched_ctx_.core_exec_states_, 0, sizeof(sched_ctx_.core_exec_states_));

    // Use handshake mechanism to discover cores (aligned with host_build_graph)
    int32_t rc = handshake_all_cores(runtime);
    if (rc != 0) {
        DEV_ERROR("handshake_all_cores failed");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Dynamically assign cores to threads
    if (!assign_cores_to_threads()) {
        return -1;
    }

    DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    // Initialize runtime execution state
    // Task count comes from PTO2 shared memory
    if (runtime->get_pto2_gm_sm_ptr()) {
        auto *header = static_cast<PTO2SharedMemoryHeader *>(runtime->get_pto2_gm_sm_ptr());
        int32_t pto2_count = 0;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            pto2_count += header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
        }
        total_tasks_ = pto2_count > 0 ? pto2_count : 0;
    } else {
        total_tasks_ = 0;
    }
    completed_tasks_.store(0, std::memory_order_release);
    // Host orchestration: graph already built, no wait needed. Device orch: Thread 3 will set this.
    bool orch_on_host = runtime->get_orch_built_on_host();
    DEV_INFO("Init: orch_built_on_host=%d", orch_on_host ? 1 : 0);
    orchestrator_done_ = orch_on_host;

    // Initial ready tasks will be populated via scheduler ready queues

    // Clear per-core dispatch payloads
    memset(sched_ctx_.payload_per_core_, 0, sizeof(sched_ctx_.payload_per_core_));

    // Initialize per-core GlobalContext (sub_block_id) based on cluster position.
    // This is done once at startup and never modified afterwards.
    for (int32_t t = 0; t < sched_thread_num_; t++) {
        CoreTracker &tracker = sched_ctx_.core_trackers_[t];
        for (int32_t c = 0; c < tracker.get_cluster_count(); c++) {
            int32_t cluster_offset = c * 3;  // Each cluster = 1 AIC + 2 AIV
            auto aiv0_id = tracker.get_core_id_by_offset(tracker.get_aiv0_core_offset(cluster_offset));
            auto aiv1_id = tracker.get_core_id_by_offset(tracker.get_aiv1_core_offset(cluster_offset));
            sched_ctx_.payload_per_core_[aiv0_id][0].global_context.sub_block_id = 0;
            sched_ctx_.payload_per_core_[aiv0_id][1].global_context.sub_block_id = 0;
            sched_ctx_.payload_per_core_[aiv1_id][0].global_context.sub_block_id = 1;
            sched_ctx_.payload_per_core_[aiv1_id][1].global_context.sub_block_id = 1;
        }
    }

    DEV_INFO("Init: PTO2 mode, task count from shared memory");

    finished_count_.store(0, std::memory_order_release);

    // Initialize SchedulerContext: wire pointers to shared AicpuExecutor state
    // Note: sched_ is set later in run() after rt is created (device orch) or below (host orch)
    if (rt) {
        sched_ctx_.sched_ = &rt->scheduler;
    }
    sched_ctx_.completed_tasks_ptr_ = &completed_tasks_;
    sched_ctx_.total_tasks_ptr_ = &total_tasks_;
    sched_ctx_.orchestrator_done_ptr_ = &orchestrator_done_;
    sched_ctx_.completed_ptr_ = &completed_;
    sched_ctx_.func_id_to_addr_ = func_id_to_addr_;
    sched_ctx_.transition_requested_ptr_ = &transition_requested_;
    sched_ctx_.wait_reassign_ptr_ = &wait_reassign_;
    sched_ctx_.reassigned_ptr_ = &reassigned_;
    sched_ctx_.active_sched_threads_ = active_sched_threads_;
    sched_ctx_.sched_thread_num_ = sched_thread_num_;
    sched_ctx_.orch_to_sched_ = orch_to_sched_;
    sched_ctx_.thread_num_ = thread_num_;
    sched_ctx_.core_count_per_thread_ = core_count_per_thread_;
    sched_ctx_.core_assignments_ = core_assignments_;
    sched_ctx_.emergency_shutdown_fn_ = emergency_shutdown_callback;

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::shutdown_aicore(
    Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num
) {
    (void)runtime;
    if (core_num == 0) return 0;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, core_num);

    for (int32_t i = 0; i < core_num; i++) {
        int32_t core_id = cur_thread_cores[i];
        uint64_t reg_addr = sched_ctx_.core_exec_states_[core_id].reg_addr;
        if (reg_addr != 0) {
            platform_deinit_aicore_regs(reg_addr);
        } else {
            DEV_ERROR("Thread %d: Core %d has invalid register address", thread_idx, core_id);
        }
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

int32_t AicpuExecutor::run(Runtime *runtime) {
    int32_t thread_idx = thread_idx_++;
    DEV_INFO("Thread %d: Start", thread_idx);

    // Orchestrator check
    if (thread_idx >= sched_thread_num_) {
#if PTO2_PROFILING
        uint64_t orch_cycle_start = 0;
        int32_t pto2_submitted_tasks = -1;
#endif
        if (runtime->get_orch_built_on_host()) {
            DEV_INFO("Thread %d: Host orchestration mode, no-op", thread_idx);
        } else {
            DEV_INFO("Thread %d: Orchestrator, loading SO via dlopen", thread_idx);

            const void *so_data = runtime->get_device_orch_so_data();
            size_t so_size = runtime->get_device_orch_so_size();

            if (so_data == nullptr || so_size == 0) {
                DEV_ERROR("Thread %d: Device orchestration SO not set", thread_idx);
                return -1;
            }

            // Try multiple paths that may allow execution on AICPU
            char so_path[256];
            bool file_created = false;
            const char *candidate_dirs[] = {
                "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device", "/usr/lib64", "/lib64", "/var/tmp", "/tmp"
            };
            const int32_t num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

            for (int32_t i = 0; i < num_candidates && !file_created; i++) {
                int32_t fd = create_orch_so_file(candidate_dirs[i], so_path, sizeof(so_path));
                if (fd < 0) {
                    DEV_INFO(
                        "Thread %d: Cannot create SO at %s (errno=%d), trying next path", thread_idx, so_path, errno
                    );
                    continue;
                }
                ssize_t written = write(fd, so_data, so_size);
                close(fd);
                if (written != static_cast<ssize_t>(so_size)) {
                    DEV_INFO(
                        "Thread %d: Cannot write SO to %s (errno=%d), trying next path", thread_idx, so_path, errno
                    );
                    unlink(so_path);
                    continue;
                }
                file_created = true;
                DEV_INFO("Thread %d: Created SO file at %s (%zu bytes)", thread_idx, so_path, so_size);
            }

            if (!file_created) {
                DEV_ERROR("Thread %d: Failed to create SO file in any candidate path", thread_idx);
                return -1;
            }

            dlerror();
            void *handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
            const char *dlopen_err = dlerror();
            if (handle == nullptr) {
                DEV_ERROR("Thread %d: dlopen failed: %s", thread_idx, dlopen_err ? dlopen_err : "unknown");
                unlink(so_path);
                return -1;
            }
            DEV_INFO("Thread %d: dlopen succeeded, handle=%p", thread_idx, handle);

            const char *entry_symbol = runtime->get_device_orch_func_name();
            if (entry_symbol == nullptr || entry_symbol[0] == '\0') {
                entry_symbol = DEFAULT_ORCH_ENTRY_SYMBOL;
            }
            const char *config_symbol = runtime->get_device_orch_config_name();
            if (config_symbol == nullptr || config_symbol[0] == '\0') {
                config_symbol = DEFAULT_ORCH_CONFIG_SYMBOL;
            }

            dlerror();
            DeviceOrchestrationFunc orch_func = reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, entry_symbol));
            const char *entry_dlsym_error = dlerror();
            if (entry_dlsym_error != nullptr) {
                DEV_ERROR(
                    "Thread %d: dlsym failed for entry symbol '%s': %s", thread_idx, entry_symbol, entry_dlsym_error
                );
                dlclose(handle);
                unlink(so_path);
                return -1;
            }
            if (orch_func == nullptr) {
                DEV_ERROR("Thread %d: dlsym returned NULL for entry symbol '%s'", thread_idx, entry_symbol);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            dlerror();
            auto config_func = reinterpret_cast<DeviceOrchestrationConfigFunc>(dlsym(handle, config_symbol));
            const char *config_dlsym_error = dlerror();
            if (config_dlsym_error != nullptr || config_func == nullptr) {
                DEV_ERROR(
                    "Thread %d: dlsym failed for config symbol '%s': %s", thread_idx, config_symbol,
                    config_dlsym_error ? config_dlsym_error : "NULL function pointer"
                );
                config_func = nullptr;
            }

            dlerror();
            auto bind_runtime_func =
                reinterpret_cast<DeviceOrchestrationBindRuntimeFunc>(dlsym(handle, "pto2_framework_bind_runtime"));
            const char *bind_runtime_error = dlerror();
            if (bind_runtime_error != nullptr) {
                DEV_ERROR(
                    "Thread %d: dlsym failed for pto2_framework_bind_runtime: %s", thread_idx, bind_runtime_error
                );
                bind_runtime_func = nullptr;
            }

            const ChipStorageTaskArgs &args = runtime->get_orch_args();
            int32_t arg_count = args.tensor_count() + args.scalar_count();
            DEV_INFO("Thread %d: sm_ptr=%p, arg_count=%d", thread_idx, runtime->get_pto2_gm_sm_ptr(), arg_count);
            for (int32_t i = 0; i < args.tensor_count() && i < 20; i++) {
                const ContinuousTensor &t = args.tensor(i);
                DEV_INFO(
                    "Thread %d: orch_args[%d] = TENSOR(data=0x%lx, ndims=%u, dtype=%u)", thread_idx, i,
                    static_cast<uint64_t>(t.data), t.ndims, static_cast<unsigned>(t.dtype)
                );
            }
            for (int32_t i = 0; i < args.scalar_count() && (args.tensor_count() + i) < 20; i++) {
                DEV_INFO(
                    "Thread %d: orch_args[%d] = SCALAR(0x%lx)", thread_idx, args.tensor_count() + i,
                    static_cast<uint64_t>(args.scalar(i))
                );
            }

            uint64_t task_window_size = PTO2_TASK_WINDOW_SIZE;
            uint64_t heap_size = PTO2_HEAP_SIZE;
            int32_t expected_arg_count = 0;
            if (config_func) {
                PTO2OrchestrationConfig cfg = config_func(args);
                expected_arg_count = cfg.expected_arg_count;
                DEV_INFO("Thread %d: Config: expected_args=%d", thread_idx, expected_arg_count);
            } else {
                DEV_INFO("Thread %d: No config function, using defaults", thread_idx);
            }

            if (expected_arg_count > 0 && arg_count < expected_arg_count) {
                DEV_ERROR("Thread %d: arg_count %d < expected %d", thread_idx, arg_count, expected_arg_count);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            if (runtime->pto2_task_window_size > 0) {
                task_window_size = runtime->pto2_task_window_size;
            }
            if (runtime->pto2_heap_size > 0) {
                heap_size = runtime->pto2_heap_size;
            }
            int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE;
            if (runtime->pto2_dep_pool_size > 0) {
                dep_pool_capacity = static_cast<int32_t>(runtime->pto2_dep_pool_size);
            }
            DEV_INFO(
                "Thread %d: Ring sizes: task_window=%lu, heap=%lu, dep_pool=%d", thread_idx,
                static_cast<uint64_t>(task_window_size), static_cast<uint64_t>(heap_size), dep_pool_capacity
            );

            void *sm_ptr = runtime->get_pto2_gm_sm_ptr();
            void *gm_heap = runtime->get_pto2_gm_heap_ptr();

            uint64_t sm_size = pto2_sm_calculate_size(task_window_size);
            PTO2SharedMemoryHandle *sm_handle =
                pto2_sm_create_from_buffer(sm_ptr, sm_size, task_window_size, heap_size);
            if (!sm_handle) {
                DEV_ERROR("Thread %d: Failed to create shared memory handle", thread_idx);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            rt = pto2_runtime_create_from_sm(PTO2_MODE_EXECUTE, sm_handle, gm_heap, heap_size, dep_pool_capacity);
            if (!rt) {
                DEV_ERROR("Thread %d: Failed to create PTO2Runtime", thread_idx);
                pto2_sm_destroy(sm_handle);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

#if PTO2_PROFILING
            rt->orchestrator.enable_profiling = runtime->enable_profiling;
#endif

            // Total core counts = aic_count_ / aiv_count_ (set once at runtime init).
            rt->orchestrator.total_cluster_count = aic_count_;
            rt->orchestrator.total_aiv_count = aiv_count_;

            // With multi-ring, slot_states are per-ring inside the scheduler.
            runtime->set_pto2_slot_states_ptr(nullptr);

            orch_func_ = orch_func;
            orch_bind_runtime_ = bind_runtime_func;
            orch_args_cached_ = &args;
            orch_so_handle_ = handle;
            snprintf(orch_so_path_, sizeof(orch_so_path_), "%s", so_path);

            // Wire scheduler context to the newly created PTO2Runtime
            sched_ctx_.sched_ = &rt->scheduler;

            runtime_init_ready_.store(true, std::memory_order_release);

            // Wait for scheduler's one-time init to complete
            while (!sched_ctx_.pto2_init_complete_.load(std::memory_order_acquire)) {
                SPIN_WAIT_HINT();
            }

#if PTO2_PROFILING
            if (runtime->enable_profiling) {
                perf_aicpu_set_orch_thread_idx(thread_idx);
            }
#endif

#if PTO2_PROFILING
            orch_cycle_start = get_sys_cnt_aicpu();
#endif
            pto2_framework_bind_runtime(rt);
            if (orch_bind_runtime_ != nullptr) {
                orch_bind_runtime_(rt);
            }
            pto2_rt_scope_begin(rt);
            orch_func_(*orch_args_cached_);
            pto2_rt_scope_end(rt);
#if PTO2_PROFILING
            uint64_t orch_cycle_end = get_sys_cnt_aicpu();
            (void)orch_cycle_end;
#endif

            // Print orchestrator profiling data
#if PTO2_ORCH_PROFILING
            PTO2OrchProfilingData p = pto2_orchestrator_get_profiling();
            uint64_t total =
                p.sync_cycle + p.alloc_cycle + p.args_cycle + p.lookup_cycle + p.insert_cycle + p.fanin_cycle;
            if (total == 0) total = 1;  // avoid div-by-zero
            DEV_ALWAYS(
                "Thread %d: === Orchestrator Profiling: %" PRId64 " tasks, total=%.3fus ===", thread_idx,
                static_cast<int64_t>(p.submit_count), cycles_to_us(total)
            );
            DEV_ALWAYS(
                "Thread %d:   task+heap_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
                cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
                static_cast<uint64_t>(p.alloc_atomic_count)
            );
            DEV_ALWAYS(
                "Thread %d:   sync_tensormap : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.sync_cycle),
                p.sync_cycle * 100.0 / total
            );
            DEV_ALWAYS(
                "Thread %d:   lookup+dep     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.lookup_cycle),
                p.lookup_cycle * 100.0 / total
            );
            DEV_ALWAYS(
                "Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.insert_cycle),
                p.insert_cycle * 100.0 / total
            );
            DEV_ALWAYS(
                "Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
                cycles_to_us(p.args_cycle), p.args_cycle * 100.0 / total, static_cast<uint64_t>(p.args_atomic_count)
            );
            DEV_ALWAYS(
                "Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total,
                cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle),
                static_cast<uint64_t>(p.fanin_atomic_count)
            );
            DEV_ALWAYS(
                "Thread %d:   avg/task       : %.3fus", thread_idx,
                p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0
            );

#if PTO2_TENSORMAP_PROFILING
            PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
            DEV_ALWAYS("Thread %d: === TensorMap Lookup Stats ===", thread_idx);
            DEV_ALWAYS(
                "Thread %d:   lookups        : %" PRIu64 ", inserts: %" PRIu64 "", thread_idx,
                static_cast<uint64_t>(tp.lookup_count), static_cast<uint64_t>(tp.insert_count)
            );
            DEV_ALWAYS(
                "Thread %d:   chain walked   : total=%" PRIu64 ", avg=%.1f, max=%d", thread_idx,
                static_cast<uint64_t>(tp.lookup_chain_total),
                tp.lookup_count > 0 ? static_cast<double>(tp.lookup_chain_total) / tp.lookup_count : 0.0,
                tp.lookup_chain_max
            );
            DEV_ALWAYS(
                "Thread %d:   overlap checks : %" PRIu64 ", hits=%" PRIu64 " (%.1f%%)", thread_idx,
                static_cast<uint64_t>(tp.overlap_checks), static_cast<uint64_t>(tp.overlap_hits),
                tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0
            );
#endif

#if PTO2_PROFILING
            // Write orchestrator summary to shared memory for host-side export (only if profiling enabled)
            if (runtime->enable_profiling) {
                AicpuOrchSummary orch_summary = {};
                orch_summary.start_time = orch_cycle_start;
                orch_summary.end_time = orch_cycle_end;
                orch_summary.sync_cycle = p.sync_cycle;
                orch_summary.alloc_cycle = p.alloc_cycle;
                orch_summary.args_cycle = p.args_cycle;
                orch_summary.lookup_cycle = p.lookup_cycle;
                orch_summary.heap_cycle = 0;  // Now included in alloc_cycle
                orch_summary.insert_cycle = p.insert_cycle;
                orch_summary.fanin_cycle = p.fanin_cycle;
                orch_summary.scope_end_cycle = p.scope_end_cycle;
                orch_summary.submit_count = p.submit_count;
                perf_aicpu_write_orch_summary(&orch_summary);
            }
#endif
#endif

#if PTO2_PROFILING
            // Write core-to-thread mapping (one-time, after orchestration)
            if (runtime->enable_profiling) {
                perf_aicpu_write_core_assignments(
                    core_assignments_, core_count_per_thread_, sched_thread_num_, cores_total_num_
                );
            }
#endif

            // Signal completion and trigger core transition
            pto2_rt_orchestration_done(rt);

            void *sm = runtime->get_pto2_gm_sm_ptr();
            PTO2SharedMemoryHeader *sm_header = static_cast<PTO2SharedMemoryHeader *>(sm);
            int32_t pto2_task_count = 0;
            if (sm_header) {
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                    pto2_task_count += sm_header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
                }
            }
#if PTO2_PROFILING
            pto2_submitted_tasks = pto2_task_count;
#endif
            total_tasks_ = pto2_task_count;
            if (runtime->enable_profiling && pto2_task_count > 0) {
                perf_aicpu_update_total_tasks(runtime, static_cast<uint32_t>(pto2_task_count));
            }
            int32_t inline_completed = static_cast<int32_t>(rt->orchestrator.inline_completed_tasks);
            if (inline_completed > 0) {
                completed_tasks_.fetch_add(inline_completed, std::memory_order_relaxed);
#if PTO2_SCHED_PROFILING
                rt->scheduler.tasks_completed.fetch_add(inline_completed, std::memory_order_relaxed);
#endif
            }
            orchestrator_done_ = true;
            {
                int32_t orch_err = 0;
                void *sm = runtime->get_pto2_gm_sm_ptr();
                if (sm) {
                    orch_err =
                        static_cast<PTO2SharedMemoryHeader *>(sm)->orch_error_code.load(std::memory_order_relaxed);
                }

                // Fatal error: shutdown AICore immediately before core transition.
                if (orch_err != PTO2_ERROR_NONE) {
                    emergency_shutdown(runtime);
                    completed_.store(true, std::memory_order_release);
                }
            }

#if PTO2_ORCH_PROFILING
            uint64_t reassign_cycle_start = get_sys_cnt_aicpu();
#endif

            // Skip core transition on fatal error — cores already shut down above
            if (completed_.load(std::memory_order_acquire)) {
                // Signal transition to unblock scheduler threads waiting at core transition
                transition_requested_.store(true, std::memory_order_release);
                reassigned_.store(true, std::memory_order_release);
            } else if (orch_to_sched_) {
                // Compute new core assignments for all threads and initialize donated slots
                DEV_INFO("Thread %d: Set orchestrator_done=true, requesting core transition", thread_idx);
#if PTO2_PROFILING
                uint64_t orch_stage_end_ts = get_sys_cnt_aicpu();
#endif
                transition_requested_.store(true, std::memory_order_release);
#if PTO2_PROFILING
                DEV_ALWAYS(
                    "Thread %d: orch_stage_end=%" PRIu64 "", thread_idx, static_cast<uint64_t>(orch_stage_end_ts)
                );
#endif

                // Wait for scheduler threads to acknowledge transition request
                while (wait_reassign_.load(std::memory_order_acquire) != sched_thread_num_) {
                    if (completed_.load(std::memory_order_acquire)) {
                        break;
                    }
                    SPIN_WAIT_HINT();
                }
                if (!completed_.load(std::memory_order_acquire)) {
                    reassign_cores_for_all_threads();
                    reassigned_.store(true, std::memory_order_release);
                }
            }

#if PTO2_ORCH_PROFILING
            uint64_t reassign_cycle_end = get_sys_cnt_aicpu();
            DEV_ALWAYS(
                "Thread %d: reassign, cost %.3fus", thread_idx, cycles_to_us(reassign_cycle_end - reassign_cycle_start)
            );
#endif
        }
#if PTO2_PROFILING
        uint64_t orch_end_ts = get_sys_cnt_aicpu();
        DEV_ALWAYS(
            "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", thread_idx,
            static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
            cycles_to_us(orch_end_ts - orch_cycle_start)
        );
        if (pto2_submitted_tasks >= 0) {
            DEV_ALWAYS(
                "PTO2 total submitted tasks = %d, already executed %d tasks", pto2_submitted_tasks,
                completed_tasks_.load(std::memory_order_acquire)
            );
        }
#endif
        DEV_INFO("Thread %d: Orchestrator completed", thread_idx);
    }

    // Scheduler thread (orchestrator threads skip dispatch when orch_to_sched_ is false)
    if (!completed_.load(std::memory_order_acquire) && (thread_idx < sched_thread_num_ || orch_to_sched_)) {
        // Device orchestration: wait for primary orchestrator to initialize SM header
        if (!runtime->get_orch_built_on_host()) {
            while (!runtime_init_ready_.load(std::memory_order_acquire)) {
                SPIN_WAIT_HINT();
            }
        }
        always_assert(rt != nullptr);
        sched_ctx_.sched_ = &rt->scheduler;
        int32_t completed = sched_ctx_.resolve_and_dispatch(runtime, thread_idx);
        DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
    }

    // Always shutdown AICore — even if completed_ was already true.
    // platform_deinit_aicore_regs is idempotent; orchestrator threads have
    // core_count_per_thread_ == 0 so they skip the loop harmlessly.
    {
        const int32_t *shutdown_cores = core_assignments_[thread_idx];
        int32_t shutdown_count = core_count_per_thread_[thread_idx];
        if (shutdown_count > 0) {
            auto rc = shutdown_aicore(runtime, thread_idx, shutdown_cores, shutdown_count);
            if (rc != 0) {
                return rc;
            }
        }
    }

    DEV_INFO("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int32_t prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        // Destroy PTO2 runtime and close orchestration SO (moved from orchestrator path)
        if (!runtime->get_orch_built_on_host() && orch_so_handle_ != nullptr) {
            // Clear g_pto2_current_runtime in this DSO and in the orchestration SO before destroying rt.
            pto2_framework_bind_runtime(nullptr);
            if (orch_bind_runtime_ != nullptr) {
                orch_bind_runtime_(nullptr);
            }
            pto2_runtime_destroy(rt);
        }
    }

    return 0;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all per-core execution state
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        sched_ctx_.core_exec_states_[i] = {};
        sched_ctx_.core_exec_states_[i].running_reg_task_id = AICPU_TASK_INVALID;
        sched_ctx_.core_exec_states_[i].pending_reg_task_id = AICPU_TASK_INVALID;
    }

    // Clear per-core dispatch payloads
    memset(sched_ctx_.payload_per_core_, 0, sizeof(sched_ctx_.payload_per_core_));

    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_ = 0;
    finished_count_.store(0, std::memory_order_release);
    orchestrator_done_ = false;
    sched_ctx_.pto2_init_done_.store(false, std::memory_order_release);
    sched_ctx_.pto2_init_complete_.store(false, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);

    // Reset core transition state
    transition_requested_.store(false, std::memory_order_release);
    wait_reassign_.store(0, std::memory_order_release);
    reassigned_.store(false, std::memory_order_release);
    completed_.store(false, std::memory_order_release);

    // Reset core discovery and assignment state
    aic_count_ = 0;
    aiv_count_ = 0;
    cores_total_num_ = 0;
    thread_num_ = 0;
    sched_thread_num_ = 0;
    thread_cores_num_ = 0;
    orch_to_sched_ = false;
    active_sched_threads_ = 0;
    memset(sched_ctx_.core_trackers_, 0, sizeof(sched_ctx_.core_trackers_));
    memset(core_assignments_, 0, sizeof(core_assignments_));
    memset(core_count_per_thread_, 0, sizeof(core_count_per_thread_));

    regs_ = 0;
    orch_func_ = nullptr;
    orch_bind_runtime_ = nullptr;
    orch_args_cached_ = nullptr;
    if (orch_so_handle_ != nullptr) {
        dlclose(orch_so_handle_);
    }
    if (orch_so_path_[0] != '\0') {
        unlink(orch_so_path_);
    }
    orch_so_handle_ = nullptr;
    orch_so_path_[0] = '\0';

    // Clear file-scope PTO2Runtime pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    DEV_INFO("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: AicpuExecutor reset complete");
}

void AicpuExecutor::emergency_shutdown(Runtime *runtime) {
    DEV_WARN("Emergency shutdown: sending exit signal to all initialized cores");
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
    for (int32_t i = 0; i < cores_total_num_; i++) {
        Handshake *hank = &all_handshakes[i];
        OUT_OF_ORDER_STORE_BARRIER();
        hank->aicpu_regs_ready = 1;
        if (sched_ctx_.core_exec_states_[i].reg_addr != 0) {
            platform_deinit_aicore_regs(sched_ctx_.core_exec_states_[i].reg_addr);
        }
    }

    DEV_WARN("Emergency shutdown complete");
}

void AicpuExecutor::diagnose_stuck_state(
    Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num, Handshake *hank
) {
    (void)runtime;
    PTO2SchedulerState *sched = &rt->scheduler;
    DEV_ALWAYS("========== DIAGNOSTIC REPORT: Thread %d ==========", thread_idx);

    int32_t completed = completed_tasks_.load(std::memory_order_acquire);
    int32_t total = total_tasks_;
    DEV_ALWAYS("Progress: %d/%d tasks (%.1f%%)", completed, total, total > 0 ? completed * 100.0 / total : 0.0);

    uint64_t aic_ready = 0, aiv_ready = 0, mix_ready = 0;
    if (rt) {
        aic_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIC)].size();
        aiv_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIV)].size();
        mix_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::MIX)].size();
    }
    DEV_ALWAYS("Ready Queues: AIC=%lu, AIV=%lu, MIX=%lu", aic_ready, aiv_ready, mix_ready);

    int32_t busy_cores = 0;
    int32_t idle_cores = 0;

    DEV_ALWAYS("Core Status:");
    for (int32_t i = 0; i < core_num; i++) {
        int32_t core_id = cur_thread_cores[i];
        Handshake *h = &hank[core_id];
        const char *core_type_str = core_type_to_string(h->core_type);

        uint64_t reg_addr = sched_ctx_.core_exec_states_[core_id].reg_addr;
        uint64_t reg_val = read_reg(reg_addr, RegId::COND);
        int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
        int32_t reg_state = EXTRACT_TASK_STATE(reg_val);
        int32_t task_id = sched_ctx_.core_exec_states_[core_id].running_reg_task_id;

        if (reg_state != TASK_FIN_STATE || task_id >= 0) {
            busy_cores++;
            if (task_id >= 0) {
                int32_t kernel_id = -1;
                if (rt && rt->sm_handle && sched_ctx_.core_exec_states_[core_id].running_slot_state) {
                    int32_t diag_slot = static_cast<int32_t>(sched_ctx_.core_exec_states_[core_id].running_subslot);
                    kernel_id = sched_ctx_.core_exec_states_[core_id].running_slot_state->task->kernel_id[diag_slot];
                }
                DEV_ALWAYS(
                    "  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s), running_reg_task_id=%d, "
                    "kernel_id=%d",
                    core_id, core_type_str, reg_val, reg_task_id, reg_state == TASK_FIN_STATE ? "FIN" : "ACK", task_id,
                    kernel_id
                );
            } else {
                DEV_ALWAYS(
                    "  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s) but task_id not tracked", core_id,
                    core_type_str, reg_val, reg_task_id, reg_state == TASK_FIN_STATE ? "FIN" : "ACK"
                );
            }
        } else {
            idle_cores++;
        }
    }

    DEV_ALWAYS("Summary: %d busy, %d idle", busy_cores, idle_cores);

    // Diagnose deadlock vs livelock
    if (busy_cores == 0 && aic_ready == 0 && aiv_ready == 0 && completed < total) {
        DEV_ALWAYS("*** DEADLOCK DETECTED ***");
        DEV_ALWAYS("All cores idle, no ready tasks, but %d tasks incomplete", total - completed);
        DEV_ALWAYS("Check PTO2 shared memory for task dependency state");
    } else if (busy_cores > 0) {
        DEV_ALWAYS("*** LIVELOCK / HUNG TASK ***");
        DEV_ALWAYS("%d cores executing but no progress", busy_cores);
    }

    DEV_ALWAYS("========== END DIAGNOSTIC ==========");
}

// ===== Public Entry Point =====

/**
 * aicpu_execute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor (thread-safe, first thread only)
 * 2. Wait for initialization to complete
 * 3. Execute tasks on managed cores
 * 4. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure
 * @return 0 on success, non-zero on error
 */
extern "C" int32_t aicpu_execute(Runtime *runtime) {
    if (runtime == nullptr) {
        DEV_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    DEV_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

    // Get platform register addresses from platform-level global
    g_aicpu_executor.regs_ = get_platform_regs();

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int32_t rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        DEV_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    int32_t runtime_rc = read_pto2_runtime_status(runtime);

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    if (runtime_rc != 0) {
        DEV_ERROR("aicpu_execute: PTO2 runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    DEV_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
