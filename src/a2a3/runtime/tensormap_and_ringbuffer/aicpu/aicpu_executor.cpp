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
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
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

struct AicpuExecutor {
    int32_t sched_thread_num_;
    bool orch_to_sched_{false};

    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int32_t thread_num_{0};

    // ===== Task queue state (managed by scheduler ready queues) =====

    std::atomic<int32_t> finished_count_{0};
    std::atomic<bool> runtime_init_ready_{false};

    // Orchestration SO handle - defer dlclose until all tasks complete
    void *orch_so_handle_{nullptr};
    char orch_so_path_[256]{};  // Path to orchestration SO file for cleanup

    // Shared orchestration function pointer (loaded by first orch thread, used by all)
    DeviceOrchestrationFunc orch_func_{nullptr};
    DeviceOrchestrationBindRuntimeFunc orch_bind_runtime_{nullptr};
    DeviceOrchestrationConfigFunc orch_config_func_{nullptr};
    const ChipStorageTaskArgs *orch_args_cached_{nullptr};

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);

    ~AicpuExecutor() {
        // Process-wide teardown (the single static instance dies here). The
        // handle is otherwise kept alive across runs for cache-hit reuse.
        if (orch_so_handle_ != nullptr) {
            dlclose(orch_so_handle_);
            orch_so_handle_ = nullptr;
        }
        if (orch_so_path_[0] != '\0') {
            unlink(orch_so_path_);
            orch_so_path_[0] = '\0';
        }
    }
};

static AicpuExecutor g_aicpu_executor;

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO_V(0, "AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        DEV_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

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

    if (sched_ctx_.init(runtime, thread_num_, sched_thread_num_, orch_to_sched_, get_platform_regs()) != 0) {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO_V(0, "AicpuExecutor: Init complete");
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::run(Runtime *runtime) {
    int32_t thread_idx = thread_idx_++;
    DEV_INFO_V(0, "Thread %d: Start", thread_idx);

    // Orchestrator check
    if (thread_idx >= sched_thread_num_) {
#if PTO2_PROFILING
        uint64_t orch_cycle_start = 0;
        int32_t pto2_submitted_tasks = -1;
#endif
        if (runtime->get_orch_built_on_host()) {
            DEV_INFO_V(0, "Thread %d: Host orchestration mode, no-op", thread_idx);
        } else {
            // Two paths:
            //   1) has_new_orch_so == true → host believes the SO identity
            //      changed, so we drop the cached handle (if any), write the
            //      new bytes to disk, and dlopen + dlsym a fresh handle.
            //   2) has_new_orch_so == false → host detected a cache hit, so
            //      we reuse `orch_so_handle_` / `orch_func_` / `orch_bind_runtime_`
            //      from the previous run untouched. sm_handle / rt below are
            //      always recreated because they bind this run's memory.
            const bool reload_so = runtime->has_new_orch_so();

            if (reload_so) {
                DEV_INFO_V(0, "Thread %d: New orch SO detected, (re)loading", thread_idx);
                if (orch_so_handle_ != nullptr) {
                    dlclose(orch_so_handle_);
                    orch_so_handle_ = nullptr;
                    orch_func_ = nullptr;
                    orch_bind_runtime_ = nullptr;
                    if (orch_so_path_[0] != '\0') {
                        // Unlink the old file so the new open() lands on a
                        // fresh inode — protects against SIGBUS / ETXTBSY when
                        // the kernel still has the old mapping pinned.
                        unlink(orch_so_path_);
                        orch_so_path_[0] = '\0';
                    }
                }

                const void *so_data = reinterpret_cast<const void *>(runtime->get_dev_orch_so_addr());
                size_t so_size = runtime->get_dev_orch_so_size();

                if (so_data == nullptr || so_size == 0) {
                    DEV_ERROR("Thread %d: Device orchestration SO not set", thread_idx);
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
                    int32_t fd = create_orch_so_file(candidate_dirs[i], so_path, sizeof(so_path));
                    if (fd < 0) {
                        DEV_INFO_V(
                            0, "Thread %d: Cannot create SO at %s (errno=%d), trying next path", thread_idx, so_path,
                            errno
                        );
                        continue;
                    }
                    ssize_t written = write(fd, so_data, so_size);
                    close(fd);
                    if (written != static_cast<ssize_t>(so_size)) {
                        DEV_INFO_V(
                            0, "Thread %d: Cannot write SO to %s (errno=%d), trying next path", thread_idx, so_path,
                            errno
                        );
                        unlink(so_path);
                        continue;
                    }
                    file_created = true;
                    DEV_INFO_V(0, "Thread %d: Created SO file at %s (%zu bytes)", thread_idx, so_path, so_size);
                }

                if (!file_created) {
                    DEV_ERROR("Thread %d: Failed to create SO file in any candidate path", thread_idx);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                dlerror();
                void *handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
                const char *dlopen_err = dlerror();
                if (handle == nullptr) {
                    DEV_ERROR("Thread %d: dlopen failed: %s", thread_idx, dlopen_err ? dlopen_err : "unknown");
                    unlink(so_path);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
                DEV_INFO_V(0, "Thread %d: dlopen succeeded, handle=%p", thread_idx, handle);

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
                    DEV_ERROR(
                        "Thread %d: dlsym failed for entry symbol '%s': %s", thread_idx, entry_symbol, entry_dlsym_error
                    );
                    dlclose(handle);
                    unlink(so_path);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
                if (orch_func == nullptr) {
                    DEV_ERROR("Thread %d: dlsym returned NULL for entry symbol '%s'", thread_idx, entry_symbol);
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
                    DEV_ERROR(
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
                    DEV_ERROR("Thread %d: dlsym failed for framework_bind_runtime: %s", thread_idx, bind_runtime_error);
                    bind_runtime_func = nullptr;
                }

                orch_so_handle_ = handle;
                orch_func_ = orch_func;
                orch_bind_runtime_ = bind_runtime_func;
                orch_config_func_ = config_func;
                snprintf(orch_so_path_, sizeof(orch_so_path_), "%s", so_path);
            } else {
                DEV_INFO_V(0, "Thread %d: Reusing cached orch SO handle=%p", thread_idx, orch_so_handle_);
                if (orch_so_handle_ == nullptr || orch_func_ == nullptr) {
                    DEV_ERROR("Thread %d: has_new_orch_so=false but no cached SO handle/func", thread_idx);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
            }

            // Validate arg count on every run (reload or cache hit).
            if (orch_config_func_ != nullptr) {
                PTO2OrchestrationConfig cfg = orch_config_func_(runtime->get_orch_args());
                DEV_INFO_V(0, "Thread %d: Config: expected_args=%d", thread_idx, cfg.expected_arg_count);
                if (cfg.expected_arg_count > 0) {
                    const ChipStorageTaskArgs &args_validate = runtime->get_orch_args();
                    int32_t actual_arg_count = args_validate.tensor_count() + args_validate.scalar_count();
                    if (actual_arg_count < cfg.expected_arg_count) {
                        DEV_ERROR(
                            "Thread %d: arg_count %d < expected %d", thread_idx, actual_arg_count,
                            cfg.expected_arg_count
                        );
                        // Clean up cached state so a subsequent run does a full reload.
                        if (orch_so_handle_ != nullptr) {
                            dlclose(orch_so_handle_);
                            orch_so_handle_ = nullptr;
                        }
                        if (orch_so_path_[0] != '\0') {
                            unlink(orch_so_path_);
                            orch_so_path_[0] = '\0';
                        }
                        orch_func_ = nullptr;
                        orch_bind_runtime_ = nullptr;
                        orch_config_func_ = nullptr;
                        // Unblock scheduler threads before returning so they don't spin forever.
                        runtime_init_ready_.store(true, std::memory_order_release);
                        return -1;
                    }
                }
            } else {
                DEV_INFO_V(0, "Thread %d: No config function, using defaults", thread_idx);
            }

            // sm_handle / rt are bound to *this* run's memory and must be
            // (re)created every run, regardless of whether the SO itself was
            // reused above.
            const ChipStorageTaskArgs &args = runtime->get_orch_args();
            int32_t arg_count = args.tensor_count() + args.scalar_count();
            DEV_INFO_V(0, "Thread %d: sm_ptr=%p, arg_count=%d", thread_idx, runtime->get_gm_sm_ptr(), arg_count);
            for (int32_t i = 0; i < args.tensor_count() && i < 20; i++) {
                const ContinuousTensor &t = args.tensor(i);
                DEV_INFO_V(
                    0, "Thread %d: orch_args[%d] = TENSOR(data=0x%lx, ndims=%u, dtype=%u)", thread_idx, i,
                    static_cast<uint64_t>(t.data), t.ndims, static_cast<unsigned>(t.dtype)
                );
            }
            for (int32_t i = 0; i < args.scalar_count() && (args.tensor_count() + i) < 20; i++) {
                DEV_INFO_V(
                    0, "Thread %d: orch_args[%d] = SCALAR(0x%lx)", thread_idx, args.tensor_count() + i,
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
            DEV_INFO_V(
                0, "Thread %d: Ring sizes: task_window=%lu, heap=%lu, dep_pool=%d", thread_idx,
                static_cast<uint64_t>(task_window_size), static_cast<uint64_t>(heap_size), dep_pool_capacity
            );

            void *sm_ptr = runtime->get_gm_sm_ptr();
            void *gm_heap = runtime->get_gm_heap_ptr();

            uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(task_window_size);
            PTO2SharedMemoryHandle *sm_handle =
                PTO2SharedMemoryHandle::create_from_buffer(sm_ptr, sm_size, task_window_size, heap_size);
            if (!sm_handle) {
                DEV_ERROR("Thread %d: Failed to create shared memory handle", thread_idx);
                // Unblock scheduler threads before returning so they don't spin forever.
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }

            rt = runtime_create_from_sm(PTO2_MODE_EXECUTE, sm_handle, gm_heap, heap_size, dep_pool_capacity);
            if (!rt) {
                DEV_ERROR("Thread %d: Failed to create PTO2Runtime", thread_idx);
                sm_handle->destroy();
                // Unblock scheduler threads before returning so they don't spin forever.
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }

#if PTO2_PROFILING
            rt->orchestrator.enable_l2_swimlane = is_l2_swimlane_enabled();
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
            sched_ctx_.wait_pto2_init_complete();

#if PTO2_PROFILING
            if (is_l2_swimlane_enabled()) {
                l2_perf_aicpu_set_orch_thread_idx(thread_idx);
            }
#endif

#if PTO2_PROFILING
            orch_cycle_start = get_sys_cnt_aicpu();
#endif
            framework_bind_runtime(rt);
            if (orch_bind_runtime_ != nullptr) {
                orch_bind_runtime_(rt);
            }
            rt_scope_begin(rt);
            orch_func_(*orch_args_cached_);
            rt_scope_end(rt);
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
            DEV_INFO_V(
                9, "Thread %d: === Orchestrator Profiling: %" PRId64 " tasks, total=%.3fus ===", thread_idx,
                static_cast<int64_t>(p.submit_count), cycles_to_us(total)
            );
            DEV_INFO_V(
                9, "Thread %d:   task+heap_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
                cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
                static_cast<uint64_t>(p.alloc_atomic_count)
            );
            DEV_INFO_V(
                9, "Thread %d:   sync_tensormap : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.sync_cycle),
                p.sync_cycle * 100.0 / total
            );
            DEV_INFO_V(
                9, "Thread %d:   lookup+dep     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.lookup_cycle),
                p.lookup_cycle * 100.0 / total
            );
            DEV_INFO_V(
                9, "Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.insert_cycle),
                p.insert_cycle * 100.0 / total
            );
            DEV_INFO_V(
                9, "Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
                cycles_to_us(p.args_cycle), p.args_cycle * 100.0 / total, static_cast<uint64_t>(p.args_atomic_count)
            );
            DEV_INFO_V(
                9, "Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total,
                cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle),
                static_cast<uint64_t>(p.fanin_atomic_count)
            );
            DEV_INFO_V(
                9, "Thread %d:   avg/task       : %.3fus", thread_idx,
                p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0
            );

#if PTO2_TENSORMAP_PROFILING
            PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
            DEV_INFO_V(9, "Thread %d: === TensorMap Lookup Stats ===", thread_idx);
            DEV_INFO_V(
                9, "Thread %d:   lookups        : %" PRIu64 ", inserts: %" PRIu64 "", thread_idx,
                static_cast<uint64_t>(tp.lookup_count), static_cast<uint64_t>(tp.insert_count)
            );
            DEV_INFO_V(
                9, "Thread %d:   chain walked   : total=%" PRIu64 ", avg=%.1f, max=%d", thread_idx,
                static_cast<uint64_t>(tp.lookup_chain_total),
                tp.lookup_count > 0 ? static_cast<double>(tp.lookup_chain_total) / tp.lookup_count : 0.0,
                tp.lookup_chain_max
            );
            DEV_INFO_V(
                9, "Thread %d:   overlap checks : %" PRIu64 ", hits=%" PRIu64 " (%.1f%%)", thread_idx,
                static_cast<uint64_t>(tp.overlap_checks), static_cast<uint64_t>(tp.overlap_hits),
                tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0
            );
#endif

#if PTO2_PROFILING
            // Write orchestrator summary to shared memory for host-side export (only if profiling enabled)
            if (is_l2_swimlane_enabled()) {
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
                l2_perf_aicpu_write_orch_summary(&orch_summary);
            }
#endif
#endif

            // Signal completion to the orchestrator state machine
            rt_orchestration_done(rt);

            // Latch task count from PTO2 shared memory
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

            if (is_l2_swimlane_enabled() && total_tasks > 0) {
                l2_perf_aicpu_update_total_tasks(static_cast<uint32_t>(total_tasks));
            }

            sched_ctx_.on_orchestration_done(runtime, rt, thread_idx, total_tasks);
        }
#if PTO2_PROFILING
        uint64_t orch_end_ts = get_sys_cnt_aicpu();
        DEV_INFO_V(
            9, "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", thread_idx,
            static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
            cycles_to_us(orch_end_ts - orch_cycle_start)
        );
        if (pto2_submitted_tasks >= 0) {
            DEV_INFO_V(
                9, "PTO2 total submitted tasks = %d, already executed %d tasks", pto2_submitted_tasks,
                sched_ctx_.completed_tasks_count()
            );
        }
#endif
        DEV_INFO_V(0, "Thread %d: Orchestrator completed", thread_idx);
    }

    // Scheduler thread (orchestrator threads skip dispatch when orch_to_sched_ is false)
    if (!sched_ctx_.is_completed() && (thread_idx < sched_thread_num_ || orch_to_sched_)) {
        // Device orchestration: wait for primary orchestrator to initialize SM header
        if (!runtime->get_orch_built_on_host()) {
            while (!runtime_init_ready_.load(std::memory_order_acquire)) {
                SPIN_WAIT_HINT();
            }
        }
        if (rt == nullptr) {
            DEV_ERROR("Thread %d: rt is null after orchestrator error, skipping dispatch", thread_idx);
        } else {
            sched_ctx_.bind_runtime(rt);
            int32_t completed = sched_ctx_.resolve_and_dispatch(runtime, thread_idx);
            DEV_INFO_V(0, "Thread %d: Executed %d tasks from runtime", thread_idx, completed);
        }
    }

    // Always shutdown AICore — even if sched_ctx_.completed_ was already true.
    // platform_deinit_aicore_regs is idempotent; orchestrator threads have
    // core_trackers_[thread_idx].core_num() == 0 so they skip the loop harmlessly.
    auto rc = sched_ctx_.shutdown(thread_idx);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO_V(0, "Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int32_t prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        // Destroy PTO2 runtime. sm_handle / rt are recreated every run so we
        // always tear them down here, but we keep orch_so_handle_ alive for
        // the next run's cache-hit reuse (see run() reload_so branch).
        if (!runtime->get_orch_built_on_host() && rt != nullptr) {
            // Clear g_current_runtime in this DSO and in the orchestration SO before destroying rt.
            framework_bind_runtime(nullptr);
            if (orch_bind_runtime_ != nullptr) {
                orch_bind_runtime_(nullptr);
            }
            runtime_destroy(rt);
        }
    }

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
    // orch_so_handle_ / orch_func_ / orch_bind_runtime_ / orch_config_func_ / orch_so_path_ are
    // intentionally preserved across deinit: the next run reuses them when
    // has_new_orch_so() == false. The destructor releases them at process
    // teardown.

    // Clear file-scope PTO2Runtime pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    DEV_INFO_V(0, "DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO_V(0, "DeInit: AicpuExecutor reset complete");
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

    DEV_INFO_V(0, "%s", "aicpu_execute: Starting AICPU kernel execution");

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
        DEV_INFO_V(0, "aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    if (runtime_rc != 0) {
        DEV_ERROR("aicpu_execute: PTO2 runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    DEV_INFO_V(0, "%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
