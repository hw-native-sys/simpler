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
#include <math.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicpu/device_time.h"
#include "aicpu/orch_so_file.h"
#include "callable_protocol.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Performance profiling headers
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/l2_perf_profiling.h"
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

// From orchestration/common.cpp linked into this DSO — updates g_current_runtime here (distinct from
// framework_bind_runtime in the dlopen'd libdevice_orch_*.so).
extern "C" void framework_bind_runtime(PTO2Runtime *rt);

constexpr const char *DEFAULT_ORCH_ENTRY_SYMBOL = "aicpu_orchestration_entry";
constexpr const char *DEFAULT_ORCH_CONFIG_SYMBOL = "aicpu_orchestration_config";

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

// Per-callable_id orchestration SO table. The executor dispatches
// `orch_so_table_[active_callable_id_]` (created on first sighting of
// that callable_id, kept warm across runs).
// MAX_REGISTERED_CALLABLE_IDS is the protocol hard cap on callable_id values
// (mailbox uint32 callable_id, register() returns small ints) and is shared
// with the host bounds check in DeviceRunner::register_prepared_callable —
// see src/common/task_interface/callable_protocol.h.

struct OrchSoEntry {
    bool in_use{false};
    void *handle{nullptr};
    char path[256]{};
    DeviceOrchestrationFunc func{nullptr};
    DeviceOrchestrationBindRuntimeFunc bind{nullptr};
    DeviceOrchestrationConfigFunc config_func{nullptr};
};

struct AicpuExecutor {
    int32_t sched_thread_num_;
    bool orch_to_sched_{false};

    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_accumulator{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int32_t thread_num_{0};

    // ====== Barrier counters for synchronization (timing) across threads
    std::atomic<uint64_t> barrier_counter_in_{0};
    std::atomic<uint64_t> barrier_counter_out_{0};

    // ===== Task queue state (managed by scheduler ready queues) =====
    std::atomic<int32_t> finished_count_{0};
    std::atomic<bool> runtime_init_ready_{false};

    // Cached orch args pointer set by the orchestration thread before scheduler
    // init; consumed by the (*p_func_)(*orch_args_cached_) invocation below.
    const ChipStorageTaskArgs *orch_args_cached_{nullptr};

    // Per-callable_id table. Single orch thread today, so first-write/read
    // race is not possible; if multiple orch threads are ever introduced,
    // guard the in_use=false→true transition with a mutex.
    OrchSoEntry orch_so_table_[MAX_REGISTERED_CALLABLE_IDS];

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Orchestrator dynamically loaded structures
    void **p_handle_{nullptr};
    char *p_path_{nullptr};
    DeviceOrchestrationFunc *p_func_{nullptr};
    DeviceOrchestrationBindRuntimeFunc *p_bind_{nullptr};
    DeviceOrchestrationConfigFunc *p_config_func_{nullptr};

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t runScheduling(Runtime *runtime);
    int32_t runOrchestration(Runtime* runtime);
    int32_t shutdown(Runtime *runtime);
    int32_t loadOrchestrator(Runtime* runtime);
    int32_t performTimingRuns(Runtime *runtime);
    void deinit(Runtime *runtime);
    int32_t getThreadId() { return thread_idx_accumulator.fetch_add(1); }

    // Barrier function to synchronize threads
    inline void barrier()
    {
        // Two phase barrier (in_out), guarantees that all threads are retained in the rendezvous point, until all of them have arrived
       barrier_counter_in_.fetch_add(1);
       while (barrier_counter_in_.load(std::memory_order_acquire) % thread_num_ != 0);
       
       barrier_counter_out_.fetch_add(1);
       while (barrier_counter_out_.load(std::memory_order_acquire) % thread_num_ != 0);
    }


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

// Thread-local identifier
thread_local int32_t my_thread_idx_;

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    LOG_INFO_V0("AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Read execution parameters from runtime
    thread_num_ = runtime->sche_cpu_num;
    sched_thread_num_ = thread_num_ - 1;
    orch_to_sched_ = runtime->orch_to_sched;
    if (thread_num_ == 0) thread_num_ = 1;

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (sched_ctx_.init(runtime, thread_num_, sched_thread_num_, orch_to_sched_, get_platform_regs()) != 0) {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    finished_count_.store(0, std::memory_order_release);

    sched_ctx_.initializePerfCounters();

    // Loading orchestrator
    int32_t load_orch_rc = loadOrchestrator(runtime);
    if (load_orch_rc != 0)
    {
        LOG_ERROR("Thread %d: Failed to load orchestrator", my_thread_idx_);
        return load_orch_rc;
    }

    init_done_.store(true, std::memory_order_release);
    LOG_INFO_V0("AicpuExecutor: Init complete");
    return 0;
}

int32_t AicpuExecutor::loadOrchestrator(Runtime* runtime)
{
    LOG_INFO_V0("Thread %d: Orchestrator Loading", my_thread_idx_);

    // Per-callable_id dispatch: the orch SO state lives in
    // `orch_so_table_[callable_id]` keyed by registration order;
    // reload is governed by `register_new_callable_id_`.
    const int32_t callable_id = runtime->get_active_callable_id();
    if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
        LOG_ERROR(
            "Thread %d: invalid callable_id %d (limit=%d)", my_thread_idx_, callable_id, MAX_REGISTERED_CALLABLE_IDS
        );
        runtime_init_ready_.store(true, std::memory_order_release);
        return -1;
    }
    p_handle_ = &orch_so_table_[callable_id].handle;
    p_path_ = orch_so_table_[callable_id].path;
    p_func_ = &orch_so_table_[callable_id].func;
    p_bind_ = &orch_so_table_[callable_id].bind;
    p_config_func_ = &orch_so_table_[callable_id].config_func;
    const bool reload_so = runtime->register_new_callable_id();

    if (reload_so) {
        LOG_INFO_V0("Thread %d: New orch SO detected (callable_id=%d), (re)loading", my_thread_idx_, callable_id);
        
        if (*p_handle_ != nullptr) {
            dlclose(*p_handle_);
            *p_handle_ = nullptr;
            *p_func_ = nullptr;
            *p_bind_ = nullptr;
            if (p_path_[0] != '\0') {
                // Unlink the old file so the new open() lands on a
                // fresh inode — protects against SIGBUS / ETXTBSY when
                // the kernel still has the old mapping pinned.
                unlink(p_path_);
                p_path_[0] = '\0';
            }
        }

        const void *so_data = reinterpret_cast<const void *>(runtime->get_dev_orch_so_addr());
        size_t so_size = runtime->get_dev_orch_so_size();

        if (so_data == nullptr || so_size == 0) {
            LOG_ERROR("Thread %d: Device orchestration SO not set", my_thread_idx_);
            // Unblock scheduler threads before returning so they don't spin forever.
            runtime_init_ready_.store(true, std::memory_order_release);
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
            int32_t fd = create_orch_so_file(candidate_dirs[i], callable_id, so_path, sizeof(so_path));
            if (fd < 0) {
                LOG_INFO_V0(
                    "Thread %d: Cannot create SO at %s (errno=%d), trying next path", my_thread_idx_, so_path, errno
                );
                continue;
            }
            ssize_t written = write(fd, so_data, so_size);
            close(fd);
            if (written != static_cast<ssize_t>(so_size)) {
                LOG_INFO_V0(
                    "Thread %d: Cannot write SO to %s (errno=%d), trying next path", my_thread_idx_, so_path, errno
                );
                unlink(so_path);
                continue;
            }
            file_created = true;
            LOG_INFO_V0("Thread %d: Created SO file at %s (%zu bytes)", my_thread_idx_, so_path, so_size);
        }

        if (!file_created) {
            LOG_ERROR("Thread %d: Failed to create SO file in any candidate path", my_thread_idx_);
            // Unblock scheduler threads before returning so they don't spin forever.
            runtime_init_ready_.store(true, std::memory_order_release);
            return -1;
        }

        dlerror();
        void *handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
        const char *dlopen_err = dlerror();
        if (handle == nullptr) {
            LOG_ERROR("Thread %d: dlopen failed: %s", my_thread_idx_, dlopen_err ? dlopen_err : "unknown");
            unlink(so_path);
            // Unblock scheduler threads before returning so they don't spin forever.
            runtime_init_ready_.store(true, std::memory_order_release);
            return -1;
        }
        LOG_INFO_V0("Thread %d: dlopen succeeded, handle=%p", my_thread_idx_, handle);

        // Unlink the on-disk SO immediately: dlopen has already mmap'd
        // the image, so the kernel keeps the inode alive until the
        // matching dlclose / process exit. This prevents stale
        // libdevice_orch_<pid>_<cid>.so files from accumulating in
        // /tmp when child processes exit via os._exit(0), which skips
        // ~AicpuExecutor (worker.py: _sub/_chip/_child loops).
        unlink(so_path);

        const char *entry_symbol = runtime->get_device_orch_func_name();
        if (entry_symbol == nullptr || entry_symbol[0] == '\0') {
            entry_symbol = DEFAULT_ORCH_ENTRY_SYMBOL;
        }
        const char *config_symbol = runtime->get_device_orch_config_name();
        if (config_symbol == nullptr || config_symbol[0] == '\0') {
            config_symbol = DEFAULT_ORCH_CONFIG_SYMBOL;
        }

        dlerror();
        DeviceOrchestrationFunc orch_func =
            reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, entry_symbol));
        const char *entry_dlsym_error = dlerror();
        if (entry_dlsym_error != nullptr) {
            LOG_ERROR(
                "Thread %d: dlsym failed for entry symbol '%s': %s", my_thread_idx_, entry_symbol, entry_dlsym_error
            );
            dlclose(handle);
            unlink(so_path);
            // Unblock scheduler threads before returning so they don't spin forever.
            runtime_init_ready_.store(true, std::memory_order_release);
            return -1;
        }
        if (orch_func == nullptr) {
            LOG_ERROR("Thread %d: dlsym returned NULL for entry symbol '%s'", my_thread_idx_, entry_symbol);
            dlclose(handle);
            unlink(so_path);
            // Unblock scheduler threads before returning so they don't spin forever.
            runtime_init_ready_.store(true, std::memory_order_release);
            return -1;
        }

        dlerror();
        auto config_func = reinterpret_cast<DeviceOrchestrationConfigFunc>(dlsym(handle, config_symbol));
        const char *config_dlsym_error = dlerror();
        if (config_dlsym_error != nullptr || config_func == nullptr) {
            LOG_ERROR(
                "Thread %d: dlsym failed for config symbol '%s': %s", my_thread_idx_, config_symbol,
                config_dlsym_error ? config_dlsym_error : "NULL function pointer"
            );
            config_func = nullptr;
        }

        dlerror();
        auto bind_runtime_func =
            reinterpret_cast<DeviceOrchestrationBindRuntimeFunc>(dlsym(handle, "framework_bind_runtime"));
        const char *bind_runtime_error = dlerror();
        if (bind_runtime_error != nullptr) {
            LOG_ERROR("Thread %d: dlsym failed for framework_bind_runtime: %s", my_thread_idx_, bind_runtime_error);
            bind_runtime_func = nullptr;
        }

        *p_handle_ = handle;
        *p_func_ = orch_func;
        *p_bind_ = bind_runtime_func;
        *p_config_func_ = config_func;
        snprintf(p_path_, 256, "%s", so_path);
    } else {
        LOG_INFO_V0(
            "Thread %d: Reusing cached orch SO handle=%p (callable_id=%d)", my_thread_idx_, *p_handle_, callable_id
        );
        if (*p_handle_ == nullptr || *p_func_ == nullptr) {
            LOG_ERROR(
                "Thread %d: reload=false but no cached SO handle/func for callable_id=%d", my_thread_idx_,
                callable_id
            );
            // Unblock scheduler threads before returning so they don't spin forever.
            runtime_init_ready_.store(true, std::memory_order_release);
            return -1;
        }
    }

    // Validate arg count on every run (reload or cache hit).
    if (*p_config_func_ != nullptr) {
        PTO2OrchestrationConfig cfg = (*p_config_func_)(runtime->get_orch_args());
        LOG_INFO_V0("Thread %d: Config: expected_args=%d", my_thread_idx_, cfg.expected_arg_count);
        if (cfg.expected_arg_count > 0) {
            const ChipStorageTaskArgs &args_validate = runtime->get_orch_args();
            int32_t actual_arg_count = args_validate.tensor_count() + args_validate.scalar_count();
            if (actual_arg_count < cfg.expected_arg_count) {
                LOG_ERROR(
                    "Thread %d: arg_count %d < expected %d", my_thread_idx_, actual_arg_count,
                    cfg.expected_arg_count
                );
                // Clean up cached state so a subsequent run does a full reload.
                if (*p_handle_ != nullptr) {
                    dlclose(*p_handle_);
                    *p_handle_ = nullptr;
                }
                if (p_path_[0] != '\0') {
                    unlink(p_path_);
                    p_path_[0] = '\0';
                }
                *p_func_ = nullptr;
                *p_bind_ = nullptr;
                *p_config_func_ = nullptr;
                // Unblock scheduler threads before returning so they don't spin forever.
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }
        }
    } else {
        LOG_INFO_V0("Thread %d: No config function, using defaults", my_thread_idx_);
    }

    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::runScheduling(Runtime *runtime) {
    int32_t run_rc = 0;
    LOG_INFO_V0("Thread %d: at AicpuExecutor::runScheduling", my_thread_idx_);

    // Scheduler thread (orchestrator threads skip dispatch when orch_to_sched_ is false)
    if (!sched_ctx_.is_completed() && (my_thread_idx_ < sched_thread_num_ || orch_to_sched_)) {
        // Device orchestration: wait for the primary orchestrator to initialize the SM header
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
        if (rt == nullptr) {
            LOG_ERROR("Thread %d: rt is null after orchestrator error, skipping dispatch", my_thread_idx_);
        } else {
            sched_ctx_.bind_runtime(rt);
            int32_t completed = sched_ctx_.resolve_and_dispatch(runtime, my_thread_idx_);
            if (completed < 0) {
                LOG_ERROR("Thread %d: Scheduler failed with rc=%d", my_thread_idx_, completed);
                run_rc = completed;
            } else {
                LOG_INFO_V0("Thread %d: Executed %d tasks from runtime", my_thread_idx_, completed);
            }
        }
    }

    LOG_INFO_V0("Thread %d: Scheduling Completed", my_thread_idx_);

    return run_rc;
}

int32_t AicpuExecutor::runOrchestration(Runtime* runtime)
{
    // Only the orchestrator thread runs this
    if (my_thread_idx_ < sched_thread_num_) return 0;

#if PTO2_PROFILING
    uint64_t orch_cycle_start = 0;
    int32_t pto2_submitted_tasks = -1;
#endif

    // Orchestrator thread: load + run the device orchestration SO. 
    LOG_INFO_V0("Thread %d: Orchestrator Running", my_thread_idx_);

    // sm_handle / rt are bound to *this* run's memory and must be
    // (re)created every run, regardless of whether the SO itself was
    // reused above.
    const ChipStorageTaskArgs &args = runtime->get_orch_args();
    int32_t arg_count = args.tensor_count() + args.scalar_count();
    LOG_INFO_V0("Thread %d: sm_ptr=%p, arg_count=%d", my_thread_idx_, runtime->get_gm_sm_ptr(), arg_count);
    for (int32_t i = 0; i < args.tensor_count() && i < 20; i++) {
        const ContinuousTensor &t = args.tensor(i);
        LOG_INFO_V0(
            "Thread %d: orch_args[%d] = TENSOR(data=0x%lx, ndims=%u, dtype=%u)", my_thread_idx_, i,
            static_cast<uint64_t>(t.data), t.ndims, static_cast<unsigned>(t.dtype)
        );
    }
    for (int32_t i = 0; i < args.scalar_count() && (args.tensor_count() + i) < 20; i++) {
        LOG_INFO_V0(
            "Thread %d: orch_args[%d] = SCALAR(0x%lx)", my_thread_idx_, args.tensor_count() + i,
            static_cast<uint64_t>(args.scalar(i))
        );
    }

    uint64_t task_window_size = PTO2_TASK_WINDOW_SIZE;
    uint64_t heap_size = PTO2_HEAP_SIZE;

    if (runtime->task_window_size > 0) {
        task_window_size = runtime->task_window_size;
    }
    if (runtime->heap_size > 0) {
        heap_size = runtime->heap_size;
    }
    int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE;
    if (runtime->dep_pool_size > 0) {
        dep_pool_capacity = static_cast<int32_t>(runtime->dep_pool_size);
    }
    LOG_INFO_V0(
        "Thread %d: Ring sizes: task_window=%lu, heap=%lu, dep_pool=%d", my_thread_idx_,
        static_cast<uint64_t>(task_window_size), static_cast<uint64_t>(heap_size), dep_pool_capacity
    );

    void *sm_ptr = runtime->get_gm_sm_ptr();
    void *gm_heap = runtime->get_gm_heap_ptr();

    uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(task_window_size);
    PTO2SharedMemoryHandle *sm_handle =
        PTO2SharedMemoryHandle::create_from_buffer(sm_ptr, sm_size, task_window_size, heap_size);
    if (!sm_handle) {
        LOG_ERROR("Thread %d: Failed to create shared memory handle", my_thread_idx_);
        // Unblock scheduler threads before returning so they don't spin forever.
        runtime_init_ready_.store(true, std::memory_order_release);
        return -1;
    }

    rt = runtime_create_from_sm(PTO2_MODE_EXECUTE, sm_handle, gm_heap, heap_size, dep_pool_capacity);
    if (!rt) {
        LOG_ERROR("Thread %d: Failed to create PTO2Runtime", my_thread_idx_);
        sm_handle->destroy();
        // Unblock scheduler threads before returning so they don't spin forever.
        runtime_init_ready_.store(true, std::memory_order_release);
        return -1;
    }

#if PTO2_PROFILING
    rt->orchestrator.l2_perf_level = get_l2_perf_level();
#endif

    // Total core counts = aic_count_ / aiv_count_ (set once at runtime init).
    rt->orchestrator.total_cluster_count = sched_ctx_.aic_count();
    rt->orchestrator.total_aiv_count = sched_ctx_.aiv_count();

    // With multi-ring, slot_states are per-ring inside the scheduler.
    runtime->set_slot_states_ptr(nullptr);

    orch_args_cached_ = &args;

    // Wire scheduler context to the newly created PTO2Runtime before
    // releasing scheduler threads from runtime_init_ready_.
    sched_ctx_.bind_runtime(rt);

    runtime_init_ready_.store(true, std::memory_order_release);

    // Wait for scheduler's one-time init to complete
    // sched_ctx_.wait_pto2_init_complete();

#if PTO2_PROFILING
    if (get_l2_perf_level() >= L2PerfLevel::ORCH_PHASES) {
        l2_perf_aicpu_set_orch_thread_idx(my_thread_idx_);
    }
#endif

    // dep_gen plugs into the orchestrator thread (single-instance subsystem):
    // set the per-thread queue index and pop the initial buffer before any
    // submit_task can fire inside orch_func_.
    if (is_dep_gen_enabled()) {
        dep_gen_aicpu_set_orch_thread_idx(my_thread_idx_);
        dep_gen_aicpu_init();
    }

#if PTO2_PROFILING
    orch_cycle_start = get_sys_cnt_aicpu();
#endif
    framework_bind_runtime(rt);
    if (*p_bind_ != nullptr) {
        (*p_bind_)(rt);
    }
    rt_scope_begin(rt);
    (*p_func_)(*orch_args_cached_);
    rt_scope_end(rt);

    // Flush the (potentially partially-filled) DepGenBuffer so the host
    // collector can pick it up before this orchestrator thread joins.
    if (is_dep_gen_enabled()) {
        dep_gen_aicpu_flush();
    }
#if PTO2_PROFILING
    uint64_t orch_cycle_end = get_sys_cnt_aicpu();
    (void)orch_cycle_end;
#endif

    // Print orchestrator profiling data
#if PTO2_ORCH_PROFILING
    PTO2OrchProfilingData p = orchestrator_get_profiling();
    uint64_t total =
        p.sync_cycle + p.alloc_cycle + p.args_cycle + p.lookup_cycle + p.insert_cycle + p.fanin_cycle;
    if (total == 0) total = 1;  // avoid div-by-zero
    LOG_INFO_V9(
        "Thread %d: === Orchestrator Profiling: %" PRId64 " tasks, total=%.3fus ===", my_thread_idx_,
        static_cast<int64_t>(p.submit_count), cycles_to_us(total)
    );
    LOG_INFO_V9(
        "Thread %d:   task+heap_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
        my_thread_idx_, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
        cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
        static_cast<uint64_t>(p.alloc_atomic_count)
    );
    LOG_INFO_V9(
        "Thread %d:   sync_tensormap : %.3fus (%.1f%%)", my_thread_idx_, cycles_to_us(p.sync_cycle),
        p.sync_cycle * 100.0 / total
    );
    LOG_INFO_V9(
        "Thread %d:   lookup+dep     : %.3fus (%.1f%%)", my_thread_idx_, cycles_to_us(p.lookup_cycle),
        p.lookup_cycle * 100.0 / total
    );
    LOG_INFO_V9(
        "Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", my_thread_idx_, cycles_to_us(p.insert_cycle),
        p.insert_cycle * 100.0 / total
    );
    LOG_INFO_V9(
        "Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", my_thread_idx_,
        cycles_to_us(p.args_cycle), p.args_cycle * 100.0 / total, static_cast<uint64_t>(p.args_atomic_count)
    );
    LOG_INFO_V9(
        "Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus", my_thread_idx_,
        cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total,
        cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle)
    );
    LOG_INFO_V9(
        "Thread %d:   avg/task       : %.3fus", my_thread_idx_,
        p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0
    );

#if PTO2_TENSORMAP_PROFILING
    PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
    LOG_INFO_V9("Thread %d: === TensorMap Lookup Stats ===", my_thread_idx_);
    LOG_INFO_V9(
        "Thread %d:   lookups        : %" PRIu64 ", inserts: %" PRIu64 "", my_thread_idx_,
        static_cast<uint64_t>(tp.lookup_count), static_cast<uint64_t>(tp.insert_count)
    );
    LOG_INFO_V9(
        "Thread %d:   chain walked   : total=%" PRIu64 ", avg=%.1f, max=%d", my_thread_idx_,
        static_cast<uint64_t>(tp.lookup_chain_total),
        tp.lookup_count > 0 ? static_cast<double>(tp.lookup_chain_total) / tp.lookup_count : 0.0,
        tp.lookup_chain_max
    );
    LOG_INFO_V9(
        "Thread %d:   overlap checks : %" PRIu64 ", hits=%" PRIu64 " (%.1f%%)", my_thread_idx_,
        static_cast<uint64_t>(tp.overlap_checks), static_cast<uint64_t>(tp.overlap_hits),
        tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0
    );
#endif
#endif  // PTO2_ORCH_PROFILING

    // Latch task count from PTO2 shared memory to hand off to the
    // scheduler. The orchestrator's run window (start_time / end_time /
    // submit_count) is no longer published to shared memory — the
    // device LOG_INFO_V9 "orch_start=… orch_end=… orch_cost=…" line
    // below carries the same envelope info for debugging, and
    // host-side swimlane derives per-phase timing from the per-event
    // AicpuPhaseRecord[] stream that already covers everything inside
    // submit_task().
    int32_t total_tasks = 0;
    if (rt->orchestrator.sm_header) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            total_tasks +=
                rt->orchestrator.sm_header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
        }
    }

#if PTO2_PROFILING
    pto2_submitted_tasks = total_tasks;
#endif

    // Signal completion to the orchestrator state machine
    rt_orchestration_done(rt);

    sched_ctx_.on_orchestration_done(runtime, rt, my_thread_idx_, total_tasks);
#if PTO2_PROFILING
    uint64_t orch_end_ts = get_sys_cnt_aicpu();
    LOG_INFO_V9(
        "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", my_thread_idx_,
        static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
        cycles_to_us(orch_end_ts - orch_cycle_start)
    );
    if (pto2_submitted_tasks >= 0) {
        LOG_INFO_V9(
            "PTO2 total submitted tasks = %d, already executed %d tasks", pto2_submitted_tasks,
            sched_ctx_.completed_tasks_count()
        );
    }
#endif
    LOG_INFO_V0("Thread %d: Orchestrator completed", my_thread_idx_);

    return 0;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all SchedulerContext-owned state in one place.
    sched_ctx_.deinit();

    finished_count_.store(0, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);

    thread_num_ = 0;
    sched_thread_num_ = 0;
    orch_to_sched_ = false;

    orch_args_cached_ = nullptr;
    // orch_so_table_ entries are intentionally preserved across deinit: the
    // next run reuses cached handles when register_new_callable_id() returns
    // false. The destructor releases them at process teardown.

    // Clear file-scope PTO2Runtime pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    // Clear dep_gen file-local bookkeeping. No-op when dep_gen is disabled.
    dep_gen_aicpu_finalize();

    LOG_INFO_V0("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_accumulator.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    LOG_INFO_V0("DeInit: AicpuExecutor reset complete");
}

int32_t AicpuExecutor::shutdown(Runtime *runtime)
{
    // Always shutdown AICore — even if sched_ctx_.completed_ was already true.
    // platform_deinit_aicore_regs is idempotent; orchestrator threads have
    // core_trackers_[my_thread_idx_].core_num() == 0 so they skip the loop harmlessly.
    int32_t shutdown_rc = sched_ctx_.shutdown(my_thread_idx_);

    // Check if this is the last thread to finish
    int32_t prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if ((prev_finished + 1) % thread_num_ == 0) {
        finished_.store(true, std::memory_order_release);
        // Destroy PTO2 runtime. sm_handle / rt are recreated every run so we
        // always tear them down here, but we keep the per-cid orch SO entries
        // alive for the next run's cache-hit reuse (see run() reload_so branch).
        if (rt != nullptr) {
            // Clear g_current_runtime in this DSO and in the orchestration SO before destroying rt.
            const int32_t callable_id = runtime->get_active_callable_id();
            framework_bind_runtime(nullptr);
            if (callable_id >= 0 && callable_id < MAX_REGISTERED_CALLABLE_IDS) {
                DeviceOrchestrationBindRuntimeFunc bind = orch_so_table_[callable_id].bind;
                if (bind != nullptr) {
                    bind(nullptr);
                }
            }
            runtime_destroy(rt);
        }
    }

    return shutdown_rc;
}

int32_t AicpuExecutor::performTimingRuns(Runtime *runtime)
{
    const auto warmupIterationCount = runtime->get_warmup_iteration_count(); 
    const auto timingIterationCount = runtime->get_timing_iteration_count();

    // Return code for running the kernel. It must be zero (no error) for all runs
    int32_t rc = 0;

    // First, perform warmup to disregard any cold cache effects and thread initialization times
    for (int32_t i = 0; i < warmupIterationCount; i++)
    {
        barrier();
        uint64_t t0 = get_sys_cnt_aicpu();
        rc |= runOrchestration(runtime);
        rc |= runScheduling(runtime);
        barrier();
        uint64_t tf = get_sys_cnt_aicpu();
        LOG_INFO_V0("Thread %d: Warmup %d/%d Time: %luns", my_thread_idx_, i, warmupIterationCount, tf - t0);

        // Resetting execution back to start
        if (my_thread_idx_ == 0) {
            deinit(runtime);
            init(runtime);
        }
    } 

    // Second, perform timed runs (the ones that count)
    std::vector<uint64_t> runTimes;
    for (int32_t i = 0; i < timingIterationCount; i++)
    {
        // Waiting for threads to arrive for before-timing.
        barrier();
        uint64_t t0 = get_sys_cnt_aicpu();
        rc |= runOrchestration(runtime);
        rc |= runScheduling(runtime);
        barrier();
        uint64_t tf = get_sys_cnt_aicpu();

        // Calculating time segments and adding them to the timing vector
        const uint64_t runTime   = tf - t0;
        runTimes.push_back(runTime);

        LOG_INFO_V0("Thread %d: Timing %d/%d Total Time: %luns", my_thread_idx_, i, timingIterationCount, runTime);

        // Resetting execution back to start before the next run
        if (my_thread_idx_ == 0) {
            deinit(runtime);
            init(runtime);
        }

        // Synchronizing before actual run
        barrier();
    }   

    if (rc != 0)
    {
        LOG_ERROR("Thread %d - Timed runs failed with rc=%d", my_thread_idx_, rc);
        return rc;
    } 

    // The orchestrator thread now calculates and reports timing
    if (my_thread_idx_ == sched_thread_num_)
    {
        // Calculating timing sum over all runs
        uint64_t runSum = 0; 
        for (const auto t : runTimes) runSum += t;

        // Calculating average
        const auto runCount = runTimes.size();
        double runAvg   = (double)runSum   / (double)runCount;

        // Calculating stddev
        double runDiff   = 0; 
        for (const auto t : runTimes)   runDiff   += ((double)t - runAvg)   * ((double)t - runAvg)  ;
        double runStdDev    = runCount == 1 ? 0.0 : std::sqrt(runDiff    / (double)(runCount-1));

        // Printing
        LOG_INFO_V9("Thread %d: [Timing] Average Runtime over %d timed runs: %10.0fns +- %6.0fns", my_thread_idx_, timingIterationCount, runAvg,   runStdDev);
    }

    return 0;
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
        LOG_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    my_thread_idx_ = g_aicpu_executor.getThreadId();

    LOG_INFO_V0("%s", "aicpu_execute: Starting AICPU kernel execution");

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
            LOG_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
            return -1;
        }
    }

    // Performing timing evaluation, exclusively for performance evaluation:
    if (runtime->get_timing_enabled() == true)
    {
        int32_t timing_rc = g_aicpu_executor.performTimingRuns(runtime);
        if (timing_rc != 0) {
            LOG_ERROR("aicpu_execute: timing run failed with rc=%d", timing_rc);
            return timing_rc;
        }
    }

    // Perform orchestration step
    int32_t orch_rc = g_aicpu_executor.runOrchestration(runtime);
    if (orch_rc != 0) {
        LOG_ERROR("aicpu_execute: Orhestration execution failed with rc=%d", orch_rc);
        return orch_rc;
    }

    // Perform scheduling step
    int32_t sched_rc = g_aicpu_executor.runScheduling(runtime);
    if (sched_rc != 0) {
        LOG_ERROR("aicpu_execute: Scheduling execution failed with rc=%d", sched_rc);
        return sched_rc;
    }

    // Reading PTO2 runtime status
    int32_t runtime_rc = read_pto2_runtime_status(runtime);
    if (runtime_rc != 0) {
        LOG_ERROR("aicpu_execute: PTO2 runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    // Shutting down
    int32_t shutdown_rc = g_aicpu_executor.shutdown(runtime);
    if (shutdown_rc != 0) {
        LOG_ERROR("aicpu_execute: shutdown failed with rc=%d", shutdown_rc);
        return shutdown_rc;
    }
    
    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        LOG_INFO_V0("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    LOG_INFO_V0("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
