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

#include "aicpu/device_time.h"
#include "aicpu/kernel_invocation_consumer.h"
#include "aicpu/device_log.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/orch_so_file.h"
#include "callable_protocol.h"
#include "common/kernel_args.h"
#include "dispatch_payload.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers (full struct definition for create/destroy + SIMPLER_SCOPE)
#include "runtime_core.h"
#include "runtime_types.h"
#include "shared_memory.h"

// Performance profiling headers
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/args_dump_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/chip_swimlane_profiling.h"
#include "common/unified_log.h"

// Register-based communication
#include "aicpu/aicpu_device_config.h"
#include "aicpu/platform_aicpu_affinity.h"
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
#include "tensormap_and_ringbuffer/kernel_execution_inputs.h"
#include "tensormap_and_ringbuffer/kernel_execution.h"
#include "tensormap_and_ringbuffer/kernel_execution_round.h"
#include "tensormap_and_ringbuffer/kernel_registration.h"
#include "utils/thread_completion_gate.h"

using simpler::tmr::ExecutionInputs;

// Device orchestration function signature (loaded via dlopen).
// The executor binds the current thread's RuntimeContext into orchestration TLS
// before calling the user entry.
typedef void (*DeviceOrchestrationFunc)(const ChipTaskArgs &orch_args);
typedef void (*DeviceOrchestrationBindRuntimeFunc)(RuntimeContext *rt);

// Config function exported by orchestration .so
typedef OrchestrationConfig (*DeviceOrchestrationConfigFunc)(const ChipTaskArgs &orch_args);

// From orchestration/common.cpp linked into this DSO — updates g_current_runtime here (distinct from
// framework_bind_runtime in the dlopen'd libdevice_orch_*.so).
extern "C" void framework_bind_runtime(RuntimeContext *rt);

constexpr const char *DEFAULT_ORCH_ENTRY_SYMBOL = "aicpu_orchestration_entry";
constexpr const char *DEFAULT_ORCH_CONFIG_SYMBOL = "aicpu_orchestration_config";

static int32_t read_runtime_status(void *sm) {
    if (sm == nullptr) {
        return 0;
    }

    auto *header = static_cast<SharedMemoryHeader *>(sm);
    int32_t orch_error_code = header->orch_error_code.load(std::memory_order_acquire);
    int32_t sched_error_code = header->sched_error_code.load(std::memory_order_acquire);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static RuntimeContext *rt{nullptr};

// Per-callable_id orchestration SO table. The executor dispatches
// `orch_so_table_[active_callable_id_]` (created on first sighting of
// that callable_id, kept warm across runs).
// MAX_REGISTERED_CALLABLE_IDS is the protocol hard cap on callable_id values
// (mailbox uint32 callable_id, register() returns small ints) and is shared
// with the host bounds check in DeviceRunner::register_callable —
// see src/common/task_interface/callable_protocol.h.

struct OrchSoEntry {
    simpler::tmr::PreparedKernelCallable kernel;
    bool kernel_owned{false};
    // Set when a residency is recorded, cleared once the round leader has
    // loaded that image. A handle left over from a previous residency at this
    // id is stale, so a non-null handle alone does not mean loaded.
    bool needs_load{false};
    bool in_use{false};
    void *handle{nullptr};
    char path[256]{};
    DeviceOrchestrationFunc func{nullptr};
    DeviceOrchestrationBindRuntimeFunc bind{nullptr};
    DeviceOrchestrationConfigFunc config_func{nullptr};
};

struct AicpuExecutor {
    int32_t sched_thread_num_;
    bool serial_orch_sched_{false};

    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    simpler::ThreadCompletionGate completion_gate_;
    std::atomic<int32_t> execution_error_{0};

    // Parallel-handshake coordination (see AicpuExecutor::init). hs_setup_done_
    // is published by the leader once the shared pre-handshake setup is visible;
    // hs_arrived_ is the barrier counting threads that finished their core slice.
    // hs_thread_seq_ hands out a distinct [0, nthreads) index when the platform
    // exposes no affinity idx (sim, where platform_aicpu_affinity_thread_idx()
    // is -1 during init) so the threads don't all collapse to leader 0.
    std::atomic<bool> hs_setup_done_{false};
    std::atomic<int32_t> hs_arrived_{0};
    std::atomic<int32_t> hs_thread_seq_{0};

    int32_t aicpu_thread_num_{0};

    // ===== Task queue state (managed by scheduler ready queues) =====

    std::atomic<bool> runtime_init_ready_{false};

    // Per-Worker arena backing the RuntimeContext + sm_handle + orch/sched/mailbox
    // sub-regions (created in runtime_create_from_sm, released in runtime_destroy).
    // Default-constructed: libc-backed backend, no ctx.
    DeviceArena runtime_arena_;

    // Entry-arg ChipTaskArgs built (via create_from_entry_storage) from get_orch_args()
    // before scheduler init; consumed by the (*p_func)(orch_args_cached_) below.
    ChipTaskArgs orch_args_cached_;
    simpler::tmr::KernelInvocationState kernel_invocation_;
    simpler::tmr::KernelRoundGate kernel_gate_;
    simpler::tmr::KernelRoundStorage kernel_storage_;
    bool kernel_storage_attached_{false};
    simpler::tmr::PreparedKernelContext kernel_context_;
    bool kernel_context_ready_{false};

    static bool kernel_arch_argument(const KernelArgs &args, uint64_t *out) noexcept {
        if (args.pmu_reg_addrs != 0) return false;
        *out = args.ffts_base_addr;
        return true;
    }

    // Per-callable_id table. Single orch thread today, so first-write/read
    // race is not possible; if multiple orch threads are ever introduced,
    // guard the in_use=false→true transition with a mutex.
    OrchSoEntry orch_so_table_[MAX_REGISTERED_CALLABLE_IDS];

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime, const ExecutionInputs &inputs);
    // (Re)load a callable's orchestration SO into orch_so_table_[callable_id].
    // Register-only: the register_callable entry calls this to dlopen and
    // populate the slot. The run path never loads — it consumes an already
    // registered slot (see run()), so loading is solely a registration step.
    int32_t load_orch_so(
        int32_t callable_id, uint64_t dev_orch_so_addr, uint64_t dev_orch_so_size, const char *entry_symbol,
        const char *config_symbol, int32_t thread_idx
    );
    int32_t
    run(Runtime *runtime, const ExecutionInputs &inputs, const simpler::tmr::KernelThreadView *kernel_thread = nullptr);
    int32_t prepare_kernel_round(const simpler::tmr::KernelExecutionRequest &request);
    int32_t prepare_execution(Runtime *runtime, const ExecutionInputs &inputs);
    int32_t
    execute(Runtime *runtime, const ExecutionInputs &inputs, const simpler::tmr::KernelThreadView *thread = nullptr);
    int32_t finalize_execution(const ExecutionInputs &inputs);
    void clear_kernel_round() noexcept;
    void cancel_kernel_round() noexcept;
    int32_t kernel_status() const {
        if (init_failed_.load(std::memory_order_acquire)) return -1;
        const int32_t error = execution_error_.load(std::memory_order_acquire);
        return error != 0 ? error :
                            (kernel_invocation_.active() ? read_runtime_status(kernel_invocation_.inputs().sm) : 0);
    }
    void deinit(Runtime *runtime, bool invalidate_host_image);

    ~AicpuExecutor() {
        // Process-wide teardown (the single static instance dies here). Every
        // in-use callable_id slot is dlclose()'d here; each is otherwise kept
        // alive across runs for cache-hit reuse.
        for (auto &e : orch_so_table_) {
            if (!e.in_use) continue;
            if (e.handle != nullptr) dlclose(e.handle);
            if (e.path[0] != '\0') unlink(e.path);
            e = OrchSoEntry{};
        }
    }
};

static AicpuExecutor g_aicpu_executor;

// The register_callable payload mirrors the runtime's orch symbol-name
// capacity; keep the two in lockstep so a name that fits in Runtime also fits
// in RegisterCallableArgs.
static_assert(
    INIT_ARGS_MAX_ORCH_SYMBOL_NAME == RUNTIME_MAX_ORCH_SYMBOL_NAME,
    "RegisterCallableArgs orch-symbol capacity must match RUNTIME_MAX_ORCH_SYMBOL_NAME"
);

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime, const ExecutionInputs &inputs) {
    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Both entry adapters supply the same execution inputs. Program's scheduler
    // leader prepares them here; kernel admission already published that setup.
    int32_t nthreads = runtime->dev.aicpu_thread_num;
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
    platform_aicpu_affinity_set_thread_idx(tidx);
    const bool is_leader = (tidx == 0);

    if (is_leader && !hs_setup_done_.load(std::memory_order_acquire)) {
        if (prepare_execution(runtime, inputs) != 0) return -1;
    } else {
        while (!hs_setup_done_.load(std::memory_order_acquire)) {}
        if (init_done_.load(std::memory_order_acquire) && init_failed_.load(std::memory_order_acquire)) return -1;
    }

    // The orchestrator (top thread, tidx == nthreads-1) does not dispatch to
    // cores, so it skips the handshake entirely and returns to
    // run() immediately to build the graph — overlapping the schedulers' handshake.
    // Core counts it needs are derived from cores_total_num_ (fixed 1:2 ratio) in
    // run(). The remaining (nthreads-1) threads re-partition ALL cores among
    // themselves, so every core still gets its register window opened.
    const bool decouple_orch = (nthreads > 1) && !serial_orch_sched_;
    const bool is_orchestrator = (tidx == nthreads - 1);
    if (decouple_orch && is_orchestrator) {
        return 0;  // do NOT touch the handshake or hs_arrived_ barrier
    }
    const int32_t hs_nthreads = decouple_orch ? (nthreads - 1) : nthreads;

    // Barrier-free scheduler init (the decoupled default). Each scheduler thread
    // handshakes exactly the clusters it will dispatch to (blocked-layout
    // ownership: cluster ci = {ci, N/3+2ci, N/3+2ci+1}, owned by ci % hs_nthreads)
    // and self-assigns them, then returns straight to run(). With no all-thread
    // barrier a thread starts dispatching to its own cores as soon as they come
    // up, independent of peers still handshaking. hs_nthreads == active_sched_threads_
    // in this branch, so handshake ownership matches assign_own_clusters'.
    if (decouple_orch) {
        sched_ctx_.handshake_owned_clusters(runtime, tidx, hs_nthreads);
        if (!sched_ctx_.handshake_failed() && !sched_ctx_.initialization_aborted())
            sched_ctx_.assign_own_clusters(tidx);
    } else {
        sched_ctx_.handshake_partition(runtime, tidx, hs_nthreads);
    }
    if (sched_ctx_.handshake_failed()) {
        init_failed_.store(true, std::memory_order_release);
        sched_ctx_.abort_and_shutdown(runtime);
    }
    const auto initialization_status = [&] {
        if (init_failed_.load(std::memory_order_acquire)) return int32_t{-1};
        if (!sched_ctx_.initialization_aborted()) return int32_t{0};
        const int32_t error = read_runtime_status(inputs.sm);
        return error != 0 ? error : int32_t{-1};
    };
    // Without capture, a scheduler's own clusters are sufficient to dispatch.
    // Profiling and the serial layout require every initializer to finish.
    if (decouple_orch && !sched_ctx_.requires_profiling_init_barrier()) return initialization_status();
    hs_arrived_.fetch_add(1, std::memory_order_acq_rel);
    if (is_leader) {
        while (hs_arrived_.load(std::memory_order_acquire) < hs_nthreads) {}
        if (initialization_status() == 0) {
            if (decouple_orch) sched_ctx_.post_handshake_profiling_init();
            else if (sched_ctx_.post_handshake_init(runtime, inputs.functions) != 0)
                init_failed_.store(true, std::memory_order_release);
        }
        init_done_.store(true, std::memory_order_release);
    } else {
        while (!init_done_.load(std::memory_order_acquire)) {}
    }
    return initialization_status();
}

int32_t AicpuExecutor::prepare_execution(Runtime *runtime, const ExecutionInputs &inputs) {
    aicpu_thread_num_ = runtime->dev.aicpu_thread_num == 0 ? 1 : runtime->dev.aicpu_thread_num;
    sched_thread_num_ = aicpu_thread_num_ - 1;
    serial_orch_sched_ = runtime->dev.serial_orch_sched;
    const int32_t status = sched_ctx_.pre_handshake_init(
        runtime, aicpu_thread_num_, sched_thread_num_, get_platform_regs(), inputs.functions, inputs.sm
    );
    if (status != 0) {
        init_failed_.store(true, std::memory_order_release);
        init_done_.store(true, std::memory_order_release);
    }
    hs_setup_done_.store(true, std::memory_order_release);
    return status;
}

int32_t
AicpuExecutor::execute(Runtime *runtime, const ExecutionInputs &inputs, const simpler::tmr::KernelThreadView *thread) {
    if (thread != nullptr) platform_aicpu_affinity_set_thread_idx(thread->execution_index);
    int32_t status;
    {
        AicpuPhaseScope preamble(AicpuPhase::Preamble);
        status = init(runtime, inputs);
    }
    if (status == 0) {
        AicpuPhaseScope graph_build(AicpuPhase::GraphBuild);
        try {
            status = run(runtime, inputs, thread);
        } catch (...) {
            status = -1;
        }
    }
    const int32_t index = platform_aicpu_affinity_thread_idx();
    if (status != 0) {
        int32_t expected = 0;
        execution_error_.compare_exchange_strong(expected, status, std::memory_order_acq_rel);
        if (index == aicpu_thread_num_ - 1) runtime_init_ready_.store(true, std::memory_order_release);
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {}
        sched_ctx_.abort_and_shutdown(runtime);
    }
    if (index >= 0 && index < aicpu_thread_num_) {
        const int32_t shutdown = sched_ctx_.shutdown(index, runtime);
        if (status == 0) status = shutdown;
    }
    if (status == 0) status = read_runtime_status(inputs.sm);
    if (status != 0) {
        int32_t expected = 0;
        execution_error_.compare_exchange_strong(expected, status, std::memory_order_acq_rel);
    }
    completion_gate_.arrive_and_finalize_if_last(aicpu_thread_num_, [&] {
        aicpu_publish_task_timing_tail_usage(aicpu_thread_num_);
        if (execution_error_.load(std::memory_order_acquire) == 0) finalize_execution(inputs);
    });
    return status;
}

int32_t AicpuExecutor::load_orch_so(
    int32_t callable_id, uint64_t dev_orch_so_addr, uint64_t dev_orch_so_size, const char *entry_symbol_in,
    const char *config_symbol_in, int32_t thread_idx
) {
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR("Thread %d: invalid callable_id %d (limit=%d)", thread_idx, callable_id, MAX_REGISTERED_CALLABLE_IDS);
        return -1;
    }

    OrchSoEntry &entry = orch_so_table_[callable_id];

    // Loading always (re)loads: the slot may have been reused after an
    // unregister, so dlclose any stale handle before dlopen'ing the new SO.
    // No AicpuPhase::SoLoad stamp here: this has no phase buffer.
    LOG_INFO("Thread %d: New orch SO detected (callable_id=%d), (re)loading", thread_idx, callable_id);
    if (entry.handle != nullptr) {
        dlclose(entry.handle);
    }
    if (entry.path[0] != '\0') {
        // Unlink the old file so the new open() lands on a fresh inode.
        unlink(entry.path);
    }
    // Only the dlopen state resets. A kernel-mode residency records the image
    // span and the child function table, and a dispatch admitted before this
    // load holds a view into that table — resetting it would dangle the view.
    auto resident = std::move(entry.kernel);
    const bool kernel_owned = entry.kernel_owned;
    const bool needs_load = entry.needs_load;
    entry = OrchSoEntry{};
    entry.kernel = std::move(resident);
    entry.kernel_owned = kernel_owned;
    entry.needs_load = needs_load;

    const void *so_data = reinterpret_cast<const void *>(dev_orch_so_addr);
    size_t so_size = dev_orch_so_size;
    if (so_data == nullptr || so_size == 0) {
        LOG_ERROR("Thread %d: Device orchestration SO not set", thread_idx);
        return -1;
    }

    char so_path[256];
    bool file_created = false;
    const char *candidate_dirs[] = {
        "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device", "/usr/lib64", "/lib64", "/var/tmp", "/tmp"
    };
    const int32_t num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

    for (int32_t i = 0; i < num_candidates && !file_created; i++) {
        int32_t fd =
            create_orch_so_file(candidate_dirs[i], callable_id, get_orch_device_id(), so_path, sizeof(so_path));
        if (fd < 0) {
            LOG_INFO("Thread %d: Cannot create SO at %s (errno=%d), trying next path", thread_idx, so_path, errno);
            continue;
        }
        ssize_t written = write(fd, so_data, so_size);
        close(fd);
        if (written != static_cast<ssize_t>(so_size)) {
            LOG_INFO("Thread %d: Cannot write SO to %s (errno=%d), trying next path", thread_idx, so_path, errno);
            unlink(so_path);
            continue;
        }
        file_created = true;
        LOG_DEBUG("Thread %d: Created SO file at %s (%zu bytes)", thread_idx, so_path, so_size);
    }

    if (!file_created) {
        LOG_ERROR("Thread %d: Failed to create SO file in any candidate path", thread_idx);
        return -1;
    }

    dlerror();
    void *handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    const char *dlopen_err = dlerror();
    if (handle == nullptr) {
        LOG_ERROR("Thread %d: dlopen failed: %s", thread_idx, dlopen_err ? dlopen_err : "unknown");
        unlink(so_path);
        return -1;
    }
    LOG_DEBUG("Thread %d: dlopen succeeded, handle=%p", thread_idx, handle);

    const char *bind_log_error = nullptr;
    if (bind_orchestration_host_log_state(handle, &bind_log_error) != 0) {
        LOG_ERROR(
            "Thread %d: failed to bind orchestration host-log state: %s", thread_idx,
            bind_log_error ? bind_log_error : "invalid shared state"
        );
        dlclose(handle);
        unlink(so_path);
        return -1;
    }

    // The image is mmap'd after dlopen; keeping only the handle avoids stale
    // libdevice_orch_<pid>_<cid>.so files when worker children exit via os._exit.
    unlink(so_path);

    const char *entry_symbol = entry_symbol_in;
    if (entry_symbol == nullptr || entry_symbol[0] == '\0') {
        entry_symbol = DEFAULT_ORCH_ENTRY_SYMBOL;
    }
    const char *config_symbol = config_symbol_in;
    if (config_symbol == nullptr || config_symbol[0] == '\0') {
        config_symbol = DEFAULT_ORCH_CONFIG_SYMBOL;
    }

    dlerror();
    DeviceOrchestrationFunc orch_func = reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, entry_symbol));
    const char *entry_dlsym_error = dlerror();
    if (entry_dlsym_error != nullptr) {
        LOG_ERROR("Thread %d: dlsym failed for entry symbol '%s': %s", thread_idx, entry_symbol, entry_dlsym_error);
        dlclose(handle);
        unlink(so_path);
        return -1;
    }
    if (orch_func == nullptr) {
        LOG_ERROR("Thread %d: dlsym returned NULL for entry symbol '%s'", thread_idx, entry_symbol);
        dlclose(handle);
        unlink(so_path);
        return -1;
    }

    dlerror();
    auto config_func = reinterpret_cast<DeviceOrchestrationConfigFunc>(dlsym(handle, config_symbol));
    const char *config_dlsym_error = dlerror();
    if (config_dlsym_error != nullptr || config_func == nullptr) {
        LOG_ERROR(
            "Thread %d: dlsym failed for config symbol '%s': %s", thread_idx, config_symbol,
            config_dlsym_error ? config_dlsym_error : "NULL function pointer"
        );
        config_func = nullptr;
    }

    dlerror();
    auto bind_runtime_func =
        reinterpret_cast<DeviceOrchestrationBindRuntimeFunc>(dlsym(handle, "framework_bind_runtime"));
    const char *bind_runtime_error = dlerror();
    if (bind_runtime_error != nullptr) {
        LOG_ERROR("Thread %d: dlsym failed for framework_bind_runtime: %s", thread_idx, bind_runtime_error);
        bind_runtime_func = nullptr;
    }

    entry.handle = handle;
    entry.func = orch_func;
    entry.bind = bind_runtime_func;
    entry.config_func = config_func;
    snprintf(entry.path, sizeof(entry.path), "%s", so_path);
    entry.in_use = true;
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::run(
    Runtime *runtime, const ExecutionInputs &inputs, const simpler::tmr::KernelThreadView *kernel_thread
) {
    int32_t affinity_exec_idx = platform_aicpu_affinity_thread_idx();
    int32_t thread_idx = kernel_thread != nullptr ? kernel_thread->execution_index :
                                                    ((affinity_exec_idx >= 0) ? affinity_exec_idx : (thread_idx_++));
    if (thread_idx < 0 || thread_idx >= aicpu_thread_num_ || thread_idx >= MAX_AICPU_THREADS) {
        LOG_ERROR(
            "Thread index %d out of bounds (active=%d max=%d exec_idx=%d)", thread_idx, aicpu_thread_num_,
            MAX_AICPU_THREADS, affinity_exec_idx
        );
        // Reachable before the orchestrator split: this thread may be the
        // would-be orchestrator, so release the scheduler threads waiting on
        // runtime_init_ready_ (the orchestrator block is the only other publisher).
        runtime_init_ready_.store(true, std::memory_order_release);
        return -1;
    }
    int32_t run_rc = 0;
    // Publish the resolved index so per-thread readers in this `.so` (notably
    // the AICPU phase-record slot) agree with the executor. On sim the basic
    // affinity gate leaves the index unset (-1); without this the sub-phase
    // stamps below would have no valid slot and silently drop. Idempotent
    // onboard, where the filter gate already set this same value.
    platform_aicpu_affinity_set_thread_idx(thread_idx);

    // Orchestrator check
    if (thread_idx >= sched_thread_num_) {
#if SIMPLER_DFX
        uint64_t orch_cycle_start = 0;
#endif
#if SIMPLER_ORCH_PROFILING
        int32_t submitted_tasks = -1;
#endif
        // Orchestrator thread: load + run the device orchestration SO. The braces
        // scope the per-callable dlopen / SO-table locals to this block.
        {
            // Per-callable_id dispatch: the orch SO state lives in
            // `orch_so_table_[callable_id]`, loaded once by the
            // register_callable entry. The run path only consumes it — it never
            // loads. A missing handle means run was reached without a prior
            // successful registration, which is a caller/scheduling bug.
            const int32_t callable_id = inputs.callable_id;
            if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
                LOG_ERROR(
                    "Thread %d: invalid callable_id %d (limit=%d)", thread_idx, callable_id, MAX_REGISTERED_CALLABLE_IDS
                );
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }
            if (orch_so_table_[callable_id].handle == nullptr || orch_so_table_[callable_id].func == nullptr) {
                LOG_ERROR(
                    "Thread %d: callable_id=%d not registered (no orch SO loaded); register before run", thread_idx,
                    callable_id
                );
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }
            // graph_build front-matter phases (orch thread only); the scheduler
            // threads spin-wait on runtime_init_ready_ across this whole region.
            // Each sub-phase gets its own `{}` scope so the boundaries are
            // visible and an early `return` still records the end via the guard
            // dtor. The few values used past their phase (p_func / p_bind for the
            // orch call below; rt / sm_ptr across phases) are declared out here.
            DeviceOrchestrationFunc *p_func = nullptr;
            DeviceOrchestrationBindRuntimeFunc *p_bind = nullptr;
            void *sm_ptr = nullptr;
            uint64_t sm_size = 0;
            {
                AicpuPhaseScope config_validate(AicpuPhase::ConfigValidate);
                OrchSoEntry &entry = orch_so_table_[callable_id];
                p_func = &entry.func;
                p_bind = &entry.bind;
                DeviceOrchestrationConfigFunc *p_config_func = &entry.config_func;

                if (kernel_thread == nullptr &&
                    !simpler::tmr::configure_orchestration_args(inputs, orch_args_cached_, *p_config_func)) {
                    LOG_ERROR("Thread %d: invocation argument count does not match callable", thread_idx);
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                // sm_handle / rt are bound to *this* run's memory and must be
                // (re)created every run, regardless of whether the SO itself was
                // reused above.
                sm_ptr = inputs.sm;
            }

            // Prebuilt-arena fast path. Host uploads the runtime arena image
            // on cache miss; cache hits reuse the resident device arena. AICPU
            // re-wires arena-internal pointers to device addresses below.
            {
                AicpuPhaseScope arena_wire(AicpuPhase::ArenaWire);
                void *prebuilt_arena = inputs.arena;
                size_t off_runtime = inputs.runtime_offset;
                if (prebuilt_arena == nullptr) {
                    LOG_ERROR("Thread %d: prebuilt_arena_base is null", thread_idx);
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
                runtime_arena_.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);
                rt = reinterpret_cast<RuntimeContext *>(static_cast<char *>(prebuilt_arena) + off_runtime);

                // Wire every arena-internal pointer field (host wrote host-mirror
                // addresses; we overwrite them with device addresses).
                runtime_wire_arena_pointers(runtime_arena_, rt->prebuilt_layout, rt);
                sm_size = SharedMemoryHandle::calculate_size_per_ring(rt->prebuilt_layout.sizing.task_window_sizes);
            }

            // Reset SM state. setup_pointers + init_header_per_ring restore
            // ring flow-control counters, layout metadata, and error flags.
            {
                AicpuPhaseScope sm_reset(AicpuPhase::SmReset);
                memset(rt->sm_handle, 0, sizeof(*rt->sm_handle));
                if (!rt->sm_handle->init_per_ring(
                        sm_ptr, sm_size, rt->prebuilt_layout.sizing.task_window_sizes,
                        rt->prebuilt_layout.sizing.heap_sizes
                    )) {
                    LOG_ERROR("Thread %d: sm_handle->init_per_ring failed", thread_idx);
                    rt = nullptr;
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
                if (!runtime_reset_for_reuse(runtime_arena_, rt->prebuilt_layout, rt)) {
                    LOG_ERROR("Thread %d: runtime_reset_for_reuse failed", thread_idx);
                    rt = nullptr;
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                // AICore completion mailbox lives in the pooled arena, so its
                // head/tail/seq survive across runs and stay monotonic. We do
                // NOT zero entries[] (256 KB): try_pop only reads a slot whose
                // seq matches the exact current ticket, and a producer writes
                // all payload before release-storing seq, so a prior run's stale
                // seq can never false-match a fresh ticket. The only per-boot
                // need is to discard any messages an error-aborted prior run
                // left undrained (head > tail) so the new consumer starts empty;
                // single-threaded here (no producers yet), tail := head does it.
                rt->aicore_mailbox->tail.store(
                    rt->aicore_mailbox->head.load(std::memory_order_acquire), std::memory_order_release
                );

                // Fill ops / core counts (host can't resolve s_runtime_ops's
                // device address nor know the SchedulerContext's core fan-out).
                // aic_count()/aiv_count() carry the handshake-derived cluster
                // count: cores_total_num_/3 on the fixed 1:2 blocked layout, or
                // the core-type-classified count post_handshake_init produces on
                // the serial path.
                runtime_finalize_after_wire(rt, sched_ctx_.aic_count(), sched_ctx_.aiv_count());
#if SIMPLER_DFX
                rt->orchestrator.chip_swimlane_level = get_chip_swimlane_level();
                {
                    auto &orch = rt->orchestrator;
                    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
                        auto &alloc = orch.rings[r].task_allocator;
                        scope_stats_set_ring_capacity(
                            r, alloc.window_size(), alloc.heap_capacity(),
                            rt->prebuilt_layout.sizing.dep_pool_capacities[r]
                        );
                    }
                    scope_stats_set_tensormap_capacity(orch.tensor_map.pool_capacity());
                }
#endif

                // Wire scheduler context to the newly created RuntimeContext before
                // releasing scheduler threads from runtime_init_ready_.
                sched_ctx_.bind_runtime(rt);
            }

            runtime_init_ready_.store(true, std::memory_order_release);

#if SIMPLER_DFX
            if (get_chip_swimlane_level() >= ChipSwimlaneLevel::ORCH_PHASES) {
                chip_swimlane_aicpu_set_orch_thread_idx(thread_idx);
            }
#endif

#if SIMPLER_DFX
            // dep_gen plugs into the orchestrator thread (single-instance subsystem):
            // resolve its buffer state and record the per-thread ready_queue index
            // before any submit_task fires inside orch_func_. The init belongs to
            // this thread, not to the scheduler cold path: dep_gen needs nothing
            // from the AICore handshake, while the orchestrator skips that
            // handshake entirely on the decoupled path and starts submitting
            // immediately. Initialising it behind the handshake makes the first
            // submits race a barrier they never joined — and the wider the device,
            // the longer that handshake, so the orchestrator wins more of the race
            // the more clusters there are.
            //
            // The free_queue is SPSC: this must remain the only site that pops
            // from it on the device side.
            if (is_dep_gen_enabled()) {
                dep_gen_aicpu_init();
                dep_gen_aicpu_set_orch_thread_idx(thread_idx);
            }

            // scope_stats streams scope_end records off the orchestrator thread:
            // record the per-thread ready_queue index. No-op (writer shared
            // state null) when scope_stats is disabled; the current buffer is
            // popped lazily on the first scope_end append.
            scope_stats_aicpu_set_orch_thread_idx(thread_idx);
#endif

#if SIMPLER_DFX
            orch_cycle_start = get_sys_cnt_aicpu();
#endif
            framework_bind_runtime(rt);
            if (*p_bind != nullptr) {
                (*p_bind)(rt);
            }
            rt_scope_begin(rt);
            (*p_func)(orch_args_cached_);
            rt_scope_end(rt);

#if SIMPLER_DFX
            // Flush the (potentially partially-filled) DepGenBuffer so the host
            // collector can pick it up before this orchestrator thread joins.
            if (is_dep_gen_enabled()) {
                dep_gen_aicpu_flush();
            }
            // Push the partially-filled scope_stats buffer so the host gets the
            // final scope_end records. Idempotent / no-op when disabled.
            scope_stats_aicpu_flush_buffers();
#endif
#if SIMPLER_DFX
            uint64_t orch_cycle_end = get_sys_cnt_aicpu();
            (void)orch_cycle_end;
#endif

            // Print orchestrator profiling data
#if SIMPLER_ORCH_PROFILING
            OrchProfilingData p = orchestrator_get_profiling();
            uint64_t total =
                p.sync_cycle + p.alloc_cycle + p.args_cycle + p.lookup_cycle + p.insert_cycle + p.fanin_cycle;
            if (total == 0) total = 1;  // avoid div-by-zero
            LOG_INFO(
                "Thread %d: === Orchestrator Profiling: %" PRId64 " tasks, total=%.3fus ===", thread_idx,
                static_cast<int64_t>(p.submit_count), cycles_to_us(total)
            );
            LOG_INFO(
                "Thread %d:   task+heap_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
                cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
                static_cast<uint64_t>(p.alloc_atomic_count)
            );
            LOG_INFO(
                "Thread %d:   sync_tensormap : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.sync_cycle),
                p.sync_cycle * 100.0 / total
            );
            LOG_INFO(
                "Thread %d:   lookup+dep     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.lookup_cycle),
                p.lookup_cycle * 100.0 / total
            );
            LOG_INFO(
                "Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.insert_cycle),
                p.insert_cycle * 100.0 / total
            );
            LOG_INFO(
                "Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
                cycles_to_us(p.args_cycle), p.args_cycle * 100.0 / total, static_cast<uint64_t>(p.args_atomic_count)
            );
            LOG_INFO(
                "Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus", thread_idx,
                cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total,
                cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle)
            );
            LOG_INFO(
                "Thread %d:   avg/task       : %.3fus", thread_idx,
                p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0
            );

#if SIMPLER_TENSORMAP_PROFILING
            ChipTensorMapProfilingData tp = chip_tensormap_get_profiling();
            LOG_INFO("Thread %d: === TensorMap Lookup Stats ===", thread_idx);
            LOG_INFO(
                "Thread %d:   lookups        : %" PRIu64 ", inserts: %" PRIu64 "", thread_idx,
                static_cast<uint64_t>(tp.lookup_count), static_cast<uint64_t>(tp.insert_count)
            );
            LOG_INFO(
                "Thread %d:   chain walked   : total=%" PRIu64 ", avg=%.1f, max=%d", thread_idx,
                static_cast<uint64_t>(tp.lookup_chain_total),
                tp.lookup_count > 0 ? static_cast<double>(tp.lookup_chain_total) / tp.lookup_count : 0.0,
                tp.lookup_chain_max
            );
            LOG_INFO(
                "Thread %d:   overlap checks : %" PRIu64 ", hits=%" PRIu64 " (%.1f%%)", thread_idx,
                static_cast<uint64_t>(tp.overlap_checks), static_cast<uint64_t>(tp.overlap_hits),
                tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0
            );
#endif
#endif  // SIMPLER_ORCH_PROFILING

            // Core-assignment capture reads every scheduler's initialized tracker.
            if (sched_ctx_.requires_profiling_init_barrier()) {
                if (read_runtime_status(inputs.sm) != 0) sched_ctx_.abort_and_shutdown(runtime);
                while (!init_done_.load(std::memory_order_acquire)) {}
            }

            // Latch task count from shared memory to hand off to the
            // scheduler. The orchestrator's run window (start_time / end_time /
            // submit_count) is no longer published to shared memory — the
            // device LOG_INFO "orch_start=… orch_end=… orch_cost=…" line
            // below carries the same envelope info for debugging, and
            // host-side swimlane derives per-phase timing from the per-event
            // ChipSwimlaneAicpuSchedPhaseRecord[] + ChipSwimlaneAicpuOrchPhaseRecord[]
            // streams that already cover everything inside submit_task().
            int32_t total_tasks = 0;
            if (rt->orchestrator.sm_header) {
                for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
                    total_tasks +=
                        rt->orchestrator.sm_header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
                }
            }

#if SIMPLER_ORCH_PROFILING
            submitted_tasks = total_tasks;
#endif

            // Signal completion to the orchestrator state machine
            rt_orchestration_done(rt);

            sched_ctx_.on_orchestration_done(runtime, rt, thread_idx, total_tasks);
        }
#if SIMPLER_DFX
        uint64_t orch_end_ts = get_sys_cnt_aicpu();
        // Ride the orch window home to the host phase buffer so the host emits
        // it as an `Orch` [STRACE] marker (the everyday path). The verbose
        // per-thread device-log line below is now opt-in deep-dive.
        aicpu_phase_set_window(AicpuPhase::OrchWindow, static_cast<uint64_t>(orch_cycle_start), orch_end_ts);
#if SIMPLER_ORCH_PROFILING
        LOG_INFO(
            "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", thread_idx,
            static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
            cycles_to_us(orch_end_ts - orch_cycle_start)
        );
        if (submitted_tasks >= 0) {
            LOG_INFO(
                "total submitted tasks = %d, already executed %d tasks", submitted_tasks,
                sched_ctx_.completed_tasks_count()
            );
        }
#endif  // SIMPLER_ORCH_PROFILING
#endif  // SIMPLER_DFX
        LOG_INFO("Thread %d: Orchestrator completed", thread_idx);
    }

    // Scheduler thread (orchestrator thread skips dispatch and exits after orchestration)
    if (!sched_ctx_.is_completed() && thread_idx < sched_thread_num_) {
        // Device orchestration: wait for the primary orchestrator to initialize the SM header
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
        if (rt == nullptr) {
            LOG_ERROR("Thread %d: rt is null after orchestrator error, skipping dispatch", thread_idx);
        } else {
            if (serial_orch_sched_) {
                sched_ctx_.wait_for_orchestration_done_before_dispatch(runtime, thread_idx);
            }
            int32_t completed = sched_ctx_.resolve_and_dispatch(runtime, thread_idx);
            if (completed < 0) {
                LOG_ERROR("Thread %d: Scheduler failed with rc=%d", thread_idx, completed);
                run_rc = completed;
            } else {
                LOG_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
            }
        }
    }

    return run_rc;
}

void AicpuExecutor::deinit(Runtime *runtime, bool invalidate_host_image) {
    // 1. Invalidate AICPU cache for the device-copied Runtime range (`dev`).
    //    Next round's Host DMA (rtMemcpy) writes fresh bytes to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from
    //    HBM. Only `dev` is uploaded, so only `dev` needs invalidation.
    if (invalidate_host_image) cache_invalidate_range(runtime, sizeof(runtime->dev));

    // Reset all SchedulerContext-owned state in one place.
    sched_ctx_.deinit();

    completion_gate_.reset();
    execution_error_.store(0, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);

    aicpu_thread_num_ = 0;
    sched_thread_num_ = 0;
    serial_orch_sched_ = false;

    orch_args_cached_.reset();
    // orch_so_table_ entries are intentionally preserved across deinit: they
    // are loaded once by register_callable and consumed by every subsequent
    // run. The destructor releases them at process teardown.

    // Clear file-scope RuntimeContext pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    // Clear dep_gen file-local bookkeeping. No-op when dep_gen is disabled.
    dep_gen_aicpu_finalize();

    LOG_INFO("DeInit: Runtime execution state reset");

    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    hs_setup_done_.store(false, std::memory_order_release);
    hs_arrived_.store(0, std::memory_order_release);
    hs_thread_seq_.store(0, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);

    LOG_INFO("DeInit: AicpuExecutor reset complete");
}

int32_t AicpuExecutor::prepare_kernel_round(const simpler::tmr::KernelExecutionRequest &request) {
    using namespace simpler::tmr;
    refresh_kernel_dfx(kernel_context_);
    if (request.execution_threads > MAX_AICPU_THREADS ||
        validate_execution_binding(request.binding) != InvocationStatus::Ok ||
        request.binding.resident->dev.aicpu_thread_num != request.execution_threads ||
        request.binding.resident->dev.worker_count != request.handshake.worker_count)
        return static_cast<int32_t>(KernelDispatchStatus::InvalidBinding);
    // Residency and the SO load both happen here rather than in preparation:
    // this runs on the round leader alone, so the shared table is written once
    // per round, and the load lands in the task that needs the orchestration
    // instead of depending on host-side stream order a replayed graph lacks.
    if (!ensure_kernel_residency(
            *this, request.callable_id, request.image_address, request.image_bytes,
            kernel_context_.descriptor.context_generation
        ))
        return static_cast<int32_t>(KernelDispatchStatus::InvalidBinding);
    const auto callable = orch_so_table_[request.callable_id].kernel.view();
    const auto admitted = kernel_invocation_.admit(request.packet, callable, request.binding);
    if (admitted != InvocationStatus::Ok) return invocation_dispatch_status(admitted);
    const auto &inputs = kernel_invocation_.inputs();
    const int32_t cid = inputs.callable_id;
    if (cid < 0 || cid >= MAX_REGISTERED_CALLABLE_IDS) return -1;
    // The orchestration SO is loaded here, on the round leader, because dlopen
    // can only run in this process and a host-side load would be unordered
    // against a launch replayed from a graph. The gate admits one leader per
    // round, so the load happens once and every other thread observes its
    // verdict through the admission this function's status publishes.
    if (orch_so_table_[cid].needs_load || orch_so_table_[cid].handle == nullptr ||
        orch_so_table_[cid].func == nullptr) {
        const auto &resident = orch_so_table_[cid].kernel;
        if (resident.device_address == 0) return -1;
        const auto *image = reinterpret_cast<const ChipCallable *>(resident.device_address);
        if (load_orch_so(
                cid, reinterpret_cast<uint64_t>(image->binary_data()), image->binary_size(), image->func_name(),
                image->config_name(), 0
            ) != 0)
            return -1;
        orch_so_table_[cid].needs_load = false;
    }
    if (!orch_so_table_[cid].in_use || orch_so_table_[cid].handle == nullptr || orch_so_table_[cid].func == nullptr)
        return -1;
    if (!configure_orchestration_args(inputs, orch_args_cached_, orch_so_table_[cid].config_func))
        return static_cast<int32_t>(KernelDispatchStatus::InvalidArgs);
    Runtime *resident = kernel_invocation_.resident();
    sched_ctx_.bind_handshakes(kernel_storage_.reports());
    const int32_t status = prepare_execution(resident, inputs);
    if (status != 0) return status;
    chip_swimlane_aicpu_record_run_boundary();
    return 0;
}

void AicpuExecutor::cancel_kernel_round() noexcept {
    if (kernel_invocation_.active()) {
        auto *header = static_cast<SharedMemoryHeader *>(kernel_invocation_.inputs().sm);
        int32_t expected = SIMPLER_ERROR_NONE;
        header->sched_error_code.compare_exchange_strong(
            expected, SIMPLER_ERROR_INVALID_ARGS, std::memory_order_acq_rel
        );
        runtime_init_ready_.store(true, std::memory_order_release);
        sched_ctx_.abort_and_shutdown(kernel_invocation_.resident());
    }
}

int32_t AicpuExecutor::finalize_execution(const ExecutionInputs &inputs) {
    if (rt == nullptr) return 0;
    framework_bind_runtime(nullptr);
    const int32_t cid = inputs.callable_id;
    if (cid >= 0 && cid < MAX_REGISTERED_CALLABLE_IDS && orch_so_table_[cid].bind != nullptr)
        orch_so_table_[cid].bind(nullptr);
    runtime_destroy(rt, runtime_arena_);
    rt = nullptr;
    return 0;
}

void AicpuExecutor::clear_kernel_round() noexcept {
    deinit(kernel_invocation_.resident(), false);
    kernel_invocation_.clear();
    kernel_storage_attached_ = false;
}

namespace simpler::tmr {

int32_t
execute_kernel_round(const KernelExecutionRequest &request, int32_t reported_cpu, KernelFinalStatus *out) noexcept {
    return execute_kernel_round_impl(g_aicpu_executor, request, reported_cpu, out);
}

int32_t kernel_execution_status() {
    const auto &state = g_aicpu_executor.kernel_invocation_;
    return state.active() ? read_runtime_status(state.inputs().sm) : -1;
}

}  // namespace simpler::tmr

// ===== Public Entry Point =====

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_prepare_tmr_context(void *arg) {
    return simpler::tmr::to_aicpu_native_status(simpler::tmr::register_kernel_context(g_aicpu_executor, arg));
}
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_register_tmr_kernel_callable(void *arg) {
    return simpler::tmr::to_aicpu_native_status(simpler::tmr::register_kernel_callable(g_aicpu_executor, arg));
}
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_revoke_tmr_context(void *arg) {
    return simpler::tmr::to_aicpu_native_status(simpler::tmr::revoke_kernel_context(g_aicpu_executor, arg));
}
int simpler::tmr::execute_kernel_task(void *arg) noexcept {
    prepare_kernel_aicpu_thread();
    return simpler::tmr::dispatch_prepared_kernel_task(g_aicpu_executor, arg, platform_aicpu_current_cpu());
}

// Device orchestration SO registration entry. Exported directly by the runtime
// (not via a platform forwarding shell): registration is a TMARB-only ability,
// so the symbol lives where the capability does. host_build_graph does not
// export it at all (host-side orchestration has nothing to register).
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_register_callable(void *arg) {
    if (g_aicpu_executor.kernel_context_ready_) return -1;
    if (arg == nullptr) {
        LOG_ERROR("%s", "simpler_aicpu_register_callable: null RegisterCallableArgs pointer");
        return -1;
    }
    const RegisterCallableArgs *args = reinterpret_cast<const RegisterCallableArgs *>(arg);
    // `arg` is the launch-arg payload CANN copies into the AICPU arg space
    // (same coherent channel exec reads KernelArgs fields from) — no HBM deref,
    // so unlike the old prewarm path there is no Runtime to cache-invalidate.
    int32_t rc = g_aicpu_executor.load_orch_so(
        args->active_callable_id, args->dev_orch_so_addr, args->dev_orch_so_size, args->device_orch_func_name,
        args->device_orch_config_name, /*thread_idx=*/0
    );
    if (rc != 0) {
        LOG_ERROR("simpler_aicpu_register_callable: SO load failed with rc=%d", rc);
        return rc;
    }
    LOG_INFO("simpler_aicpu_register_callable: completed for callable_id=%d", args->active_callable_id);
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
    if (g_aicpu_executor.kernel_context_ready_) return -1;
    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    LOG_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

    const int32_t rc = g_aicpu_executor.execute(runtime, simpler::tmr::program_execution_inputs(*runtime));
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
    }

    // PostOrch measures only the real teardown (deinit), and only on the last
    // thread to finish. Stamping it on every thread would let an orchestrator
    // thread that finished early (it submits then returns while the scheduler
    // threads are still draining) open the window at its early exit, so the
    // cross-thread max(end)-min(start) reduction would absorb the orch-waits-for-
    // sched overlap into post_orch — inflating it well past the actual teardown.
    // read_runtime_status is two atomic loads every thread needs, so it
    // stays outside the scope.
    if (g_aicpu_executor.completion_gate_.claim_cleanup()) {
        AicpuPhaseScope post_orch(AicpuPhase::PostOrch);
        LOG_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime, true);
    }

    if (rc != 0) {
        return rc;
    }

    LOG_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
