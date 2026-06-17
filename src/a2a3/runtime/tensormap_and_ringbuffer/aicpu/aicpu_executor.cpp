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
#include "aicpu/l2_swimlane_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/l2_swimlane_profiling.h"

// Register-based communication
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

// Core type definitions
#include "common/core_type.h"

// CoreCallable for resolved dispatch address
#include "callable.h"

// Scheduler data structures (CoreExecState, CoreTracker, etc.)
#include "scheduler_types.h"

// Scheduler context class
#include "scheduler_context.h"

typedef void (*DeviceOrchestrationFunc)(const ChipStorageTaskArgs &orch_args);
typedef void (*DeviceOrchestrationBindRuntimeFunc)(PTO2Runtime *rt);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(const ChipStorageTaskArgs &orch_args);

// From orchestration/common.cpp linked into this DSO — updates g_current_runtime here (distinct from
// framework_bind_runtime in the dlopen'd libdevice_orch_*.so).
extern "C" void framework_bind_runtime(PTO2Runtime *rt);

constexpr const char *DEFAULT_ORCH_ENTRY_SYMBOL = "aicpu_orchestration_entry";
constexpr const char *DEFAULT_ORCH_CONFIG_SYMBOL = "aicpu_orchestration_config";

static int32_t read_pto2_runtime_status(Runtime *runtime)
{
    if (runtime == nullptr) return 0;

    void *sm = runtime->get_gm_sm_ptr();
    if (sm == nullptr) return 0;

    auto *header = static_cast<PTO2SharedMemoryHeader *>(sm);
    int32_t orch_error_code = header->orch_error_code.load(std::memory_order_acquire);
    int32_t sched_error_code = header->sched_error_code.load(std::memory_order_acquire);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static PTO2Runtime *rt{nullptr};

struct OrchSoEntry
{
    bool in_use{false};
    void *handle{nullptr};
    char path[256]{};
    DeviceOrchestrationFunc func{nullptr};
    DeviceOrchestrationBindRuntimeFunc bind{nullptr};
    DeviceOrchestrationConfigFunc config_func{nullptr};
};

struct AicpuExecutor
{
    int32_t sched_thread_num_;
    bool orch_to_sched_{false};

    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int32_t aicpu_thread_num_{0};

    // ===== Task queue state (managed by scheduler ready queues) =====

    std::atomic<int32_t> finished_count_{0};
    std::atomic<bool> runtime_init_ready_{false};

    DeviceArena runtime_arena_;

    // Cached orch args pointer set by the orchestration thread before scheduler
    // init; consumed by the (*p_func)(*orch_args_cached_) invocation below.
    const ChipStorageTaskArgs *orch_args_cached_{nullptr};

    OrchSoEntry orch_so_table_[MAX_REGISTERED_CALLABLE_IDS];

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);

    ~AicpuExecutor()
    {
        for (auto &e : orch_so_table_)
        {
            if (!e.in_use) continue;
            if (e.handle != nullptr) dlclose(e.handle);
            if (e.path[0] != '\0') unlink(e.path);
            e = OrchSoEntry{};
        }
    }
};

static AicpuExecutor g_aicpu_executor;

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime)
{
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) return 0;

    if (runtime == nullptr)
    {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    aicpu_thread_num_ = runtime->aicpu_thread_num;
    if (aicpu_thread_num_ == 0) aicpu_thread_num_ = 1;
    sched_thread_num_ = aicpu_thread_num_ - 1;
    orch_to_sched_ = runtime->orch_to_sched;

    if (aicpu_thread_num_ < 1 || aicpu_thread_num_ > MAX_AICPU_THREADS)
    {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (sched_ctx_.init(runtime, aicpu_thread_num_, sched_thread_num_, orch_to_sched_, get_platform_regs()) != 0)
    {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    return 0;
}

int32_t AicpuExecutor::run(Runtime *runtime)
{
    int32_t thread_idx = thread_idx_++;
    int32_t run_rc = 0;

    // Orchestrator check
    if (thread_idx >= sched_thread_num_)
    {
        // Orchestrator thread: load + run the device orchestration SO. The braces
        // scope the per-callable dlopen / SO-table locals to this block.
        {
            const int32_t callable_id = runtime->get_active_callable_id();
            if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS)
            {
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }
            void **p_handle = &orch_so_table_[callable_id].handle;
            char *p_path = orch_so_table_[callable_id].path;
            DeviceOrchestrationFunc *p_func = &orch_so_table_[callable_id].func;
            DeviceOrchestrationBindRuntimeFunc *p_bind = &orch_so_table_[callable_id].bind;
            DeviceOrchestrationConfigFunc *p_config_func = &orch_so_table_[callable_id].config_func;
            const bool reload_so = runtime->register_new_callable_id();

            if (reload_so)
            {
                if (*p_handle != nullptr)
                {
                    dlclose(*p_handle);
                    *p_handle = nullptr;
                    *p_func = nullptr;
                    *p_bind = nullptr;
                    if (p_path[0] != '\0')
                    {
                        unlink(p_path);
                        p_path[0] = '\0';
                    }
                }

                const void *so_data = reinterpret_cast<const void *>(runtime->get_dev_orch_so_addr());
                size_t so_size = runtime->get_dev_orch_so_size();

                if (so_data == nullptr || so_size == 0)
                {
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                // Try multiple paths that may allow execution on AICPU.
                char so_path[256];
                bool file_created = false;
                const char *candidate_dirs[] = {"/usr/lib64/aicpu_kernels/0/aicpu_kernels_device", "/usr/lib64", "/lib64", "/var/tmp", "/tmp"};
                const int32_t num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

                for (int32_t i = 0; i < num_candidates && !file_created; i++)
                {
                    int32_t fd = create_orch_so_file(candidate_dirs[i], callable_id, get_orch_device_id(), so_path, sizeof(so_path));
                    if (fd < 0) continue;
                    ssize_t written = write(fd, so_data, so_size);
                    close(fd);
                    if (written != static_cast<ssize_t>(so_size))
                    {
                        unlink(so_path);
                        continue;
                    }
                    file_created = true;
                }

                if (!file_created)
                {
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                dlerror();
                void *handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
                if (handle == nullptr)
                {
                    unlink(so_path);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                unlink(so_path);

                const char *entry_symbol = runtime->get_device_orch_func_name();
                if (entry_symbol == nullptr || entry_symbol[0] == '\0') entry_symbol = DEFAULT_ORCH_ENTRY_SYMBOL;
                const char *config_symbol = runtime->get_device_orch_config_name();
                if (config_symbol == nullptr || config_symbol[0] == '\0') config_symbol = DEFAULT_ORCH_CONFIG_SYMBOL;

                dlerror();
                DeviceOrchestrationFunc orch_func = reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, entry_symbol));
                const char *entry_dlsym_error = dlerror();
                if (entry_dlsym_error != nullptr)
                {
                    dlclose(handle);
                    unlink(so_path);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
                if (orch_func == nullptr)
                {
                    dlclose(handle);
                    unlink(so_path);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                dlerror();
                auto config_func = reinterpret_cast<DeviceOrchestrationConfigFunc>(dlsym(handle, config_symbol));
                const char *config_dlsym_error = dlerror();
                if (config_dlsym_error != nullptr || config_func == nullptr) config_func = nullptr;

                dlerror();
                auto bind_runtime_func = reinterpret_cast<DeviceOrchestrationBindRuntimeFunc>(dlsym(handle, "framework_bind_runtime"));
                const char *bind_runtime_error = dlerror();
                if (bind_runtime_error != nullptr) bind_runtime_func = nullptr;

                *p_handle = handle;
                *p_func = orch_func;
                *p_bind = bind_runtime_func;
                *p_config_func = config_func;
                snprintf(p_path, 256, "%s", so_path);
                orch_so_table_[callable_id].in_use = true;
            }
            else if (*p_handle == nullptr || *p_func == nullptr)
            {
                // Unblock scheduler threads before returning so they don't spin forever.
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }

            // Validate arg count on every run (reload or cache hit).
            if (*p_config_func != nullptr)
            {
                PTO2OrchestrationConfig cfg = (*p_config_func)(runtime->get_orch_args());
                if (cfg.expected_arg_count > 0)
                {
                    const ChipStorageTaskArgs &args_validate = runtime->get_orch_args();
                    int32_t actual_arg_count = args_validate.tensor_count() + args_validate.scalar_count();
                    if (actual_arg_count < cfg.expected_arg_count)
                    {
                        // Clean up cached state so a subsequent run does a full reload.
                        if (*p_handle != nullptr)
                        {
                            dlclose(*p_handle);
                            *p_handle = nullptr;
                        }
                        if (p_path[0] != '\0')
                        {
                            unlink(p_path);
                            p_path[0] = '\0';
                        }
                        *p_func = nullptr;
                        *p_bind = nullptr;
                        *p_config_func = nullptr;
                        orch_so_table_[callable_id].in_use = false;
                        // Unblock scheduler threads before returning so they don't spin forever.
                        runtime_init_ready_.store(true, std::memory_order_release);
                        return -1;
                    }
                }
            }
            else
            {}

            const ChipStorageTaskArgs &args = runtime->get_orch_args();
            uint64_t task_window_size = PTO2_TASK_WINDOW_SIZE;
            uint64_t heap_size = PTO2_HEAP_SIZE;

            if (runtime->task_window_size > 0) task_window_size = runtime->task_window_size;
            if (runtime->heap_size > 0) heap_size = runtime->heap_size;
            int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE;
            if (runtime->dep_pool_size > 0) dep_pool_capacity = static_cast<int32_t>(runtime->dep_pool_size);

            (void)dep_pool_capacity;

            void *sm_ptr = runtime->get_gm_sm_ptr();
            uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(task_window_size);

            void *prebuilt_arena = runtime->get_prebuilt_arena_base();
            size_t off_runtime = runtime->get_prebuilt_runtime_offset();
            if (prebuilt_arena == nullptr)
            {
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }
            runtime_arena_.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);
            rt = reinterpret_cast<PTO2Runtime *>(static_cast<char *>(prebuilt_arena) + off_runtime);

            // Wire every arena-internal pointer field (host wrote host-mirror
            // addresses; we overwrite them with device addresses).
            runtime_wire_arena_pointers(runtime_arena_, rt->prebuilt_layout, rt);

            memset(rt->sm_handle, 0, sizeof(*rt->sm_handle));
            if (!rt->sm_handle->init(sm_ptr, sm_size, task_window_size, heap_size))
            {
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }

            memset(rt->aicore_mailbox, 0, sizeof(*rt->aicore_mailbox));

            // Fill ops / core counts (host can't resolve s_runtime_ops's
            // device address nor know the SchedulerContext's core fan-out).
            runtime_finalize_after_wire(rt, sched_ctx_.aic_count(), sched_ctx_.aiv_count());

            // With multi-ring, slot_states are per-ring inside the scheduler.
            runtime->set_slot_states_ptr(nullptr);

            orch_args_cached_ = &args;

            // Wire scheduler context to the newly created PTO2Runtime before
            // releasing scheduler threads from runtime_init_ready_.
            sched_ctx_.bind_runtime(rt);

            runtime_init_ready_.store(true, std::memory_order_release);

            // Wait for scheduler's one-time init to complete
            sched_ctx_.wait_pto2_init_complete();

            if (is_dep_gen_enabled())
            {
                dep_gen_aicpu_set_orch_thread_idx(thread_idx);
                dep_gen_aicpu_init();
            }

            framework_bind_runtime(rt);
            if (*p_bind != nullptr) (*p_bind)(rt);
            rt_scope_begin(rt);
            (*p_func)(*orch_args_cached_);
            rt_scope_end(rt);

            // Flush the (potentially partially-filled) DepGenBuffer so the host
            // collector can pick it up before this orchestrator thread joins.
            if (is_dep_gen_enabled()) dep_gen_aicpu_flush();

            // Print orchestrator profiling data

            int32_t total_tasks = 0;
            if (rt->orchestrator.sm_header)
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) total_tasks += rt->orchestrator.sm_header->rings[r].fc.current_task_index.load(std::memory_order_acquire);

            // Signal completion to the orchestrator state machine
            rt_orchestration_done(rt);

            sched_ctx_.on_orchestration_done(runtime, rt, total_tasks);
        }
    }

    // Scheduler thread (orchestrator threads skip dispatch when orch_to_sched_ is false)
    if (!sched_ctx_.is_completed() && (thread_idx < sched_thread_num_ || orch_to_sched_))
    {
        // Device orchestration: wait for the primary orchestrator to initialize the SM header
        while (!runtime_init_ready_.load(std::memory_order_acquire)) SPIN_WAIT_HINT();
        if (rt == nullptr)
        {}
        else
        {
            sched_ctx_.bind_runtime(rt);
            int32_t completed = sched_ctx_.resolve_and_dispatch(runtime, thread_idx);
            if (completed < 0)
            {
                run_rc = completed;
            }
            else
            {}
        }
    }

    int32_t shutdown_rc = sched_ctx_.shutdown(thread_idx);
    if (shutdown_rc != 0 && run_rc == 0) run_rc = shutdown_rc;

    // Check if this is the last thread to finish
    int32_t prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == aicpu_thread_num_)
    {
        finished_.store(true, std::memory_order_release);
        if (rt != nullptr)
        {
            // Clear g_current_runtime in this DSO and in the orchestration SO before destroying rt.
            const int32_t callable_id = runtime->get_active_callable_id();
            framework_bind_runtime(nullptr);
            if (callable_id >= 0 && callable_id < MAX_REGISTERED_CALLABLE_IDS)
            {
                DeviceOrchestrationBindRuntimeFunc bind = orch_so_table_[callable_id].bind;
                if (bind != nullptr) bind(nullptr);
            }
            runtime_destroy(rt);
            rt = nullptr;
        }
    }

    return run_rc;
}

void AicpuExecutor::deinit(Runtime *runtime)
{
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all SchedulerContext-owned state in one place.
    sched_ctx_.deinit();

    finished_count_.store(0, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);

    aicpu_thread_num_ = 0;
    sched_thread_num_ = 0;
    orch_to_sched_ = false;

    orch_args_cached_ = nullptr;

    // Clear file-scope PTO2Runtime pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    // Clear dep_gen file-local bookkeeping. No-op when dep_gen is disabled.
    dep_gen_aicpu_finalize();

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);
}

// ===== Public Entry Point =====

extern "C" int32_t aicpu_execute(Runtime *runtime)
{
    if (runtime == nullptr) return -1;

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire))
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) return -1;

    int32_t rc = g_aicpu_executor.run(runtime);
    if (rc != 0)
    {}

    int32_t runtime_rc = read_pto2_runtime_status(runtime);

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) g_aicpu_executor.deinit(runtime);

    if (runtime_rc != 0) return runtime_rc;

    if (rc != 0) return rc;

    return 0;
}
