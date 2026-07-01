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
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/orch_so_file.h"
#include "callable_protocol.h"
#include "common/kernel_args.h"
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
#include "pipeline_strategy.h"

// Scheduler data structures (CoreExecState, CoreTracker, etc.)
#include "scheduler/scheduler_types.h"

// Scheduler context class
#include "scheduler/scheduler_context.h"

// Device orchestration function signature (loaded via dlopen).
// The executor binds the current thread's PTO2Runtime into orchestration TLS
// before calling the user entry.
typedef void (*DeviceOrchestrationFunc)(const L2TaskArgs &orch_args);
typedef void (*DeviceOrchestrationBindRuntimeFunc)(PTO2Runtime *rt);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(const L2TaskArgs &orch_args);

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
// with the host bounds check in DeviceRunner::register_callable —
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
    int32_t orch_thread_num_;
    int32_t primary_orch_thread_idx_;
    PipelineLayout pipeline_layout_{};
    int32_t scheduler_index_by_thread_[MAX_AICPU_THREADS];
    int32_t orchestrator_stage_by_thread_[MAX_AICPU_THREADS];
    bool serial_orch_sched_{false};

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

    // Per-Worker arena backing the PTO2Runtime + sm_handle + orch/sched/mailbox
    // sub-regions (created in runtime_create_from_sm, released in runtime_destroy).
    // Default-constructed: libc-backed backend, no ctx.
    DeviceArena runtime_arena_;

    // Entry-arg L2TaskArgs built (via create_from_chip_args) from get_orch_args()
    // before scheduler init; consumed by the (*p_func)(orch_args_cached_) below.
    L2TaskArgs orch_args_cached_;

    // Per-callable_id table. Single orch thread today, so first-write/read
    // race is not possible; if multiple orch threads are ever introduced,
    // guard the in_use=false→true transition with a mutex.
    OrchSoEntry orch_so_table_[MAX_REGISTERED_CALLABLE_IDS];

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    // (Re)load a callable's orchestration SO into orch_so_table_[callable_id].
    // Register-only: the register_callable entry calls this to dlopen and
    // populate the slot. The run path never loads — it consumes an already
    // registered slot (see run()), so loading is solely a registration step.
    int32_t load_orch_so(
        int32_t callable_id, uint64_t dev_orch_so_addr, uint64_t dev_orch_so_size, const char *entry_symbol,
        const char *config_symbol, int32_t thread_idx
    );
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
    void resolve_thread_layout(Runtime *runtime);
    int32_t scheduler_idx_for_thread(int32_t thread_idx) const;
    int32_t orch_stage_idx_for_thread(int32_t thread_idx) const;
    int32_t scheduler_context_idx_for_thread(int32_t thread_idx) const;

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

void AicpuExecutor::resolve_thread_layout(Runtime *runtime) {
    aicpu_thread_num_ = runtime->dev.aicpu_thread_num;
    if (aicpu_thread_num_ == 0) aicpu_thread_num_ = 1;

    for (int32_t i = 0; i < MAX_AICPU_THREADS; ++i) {
        scheduler_index_by_thread_[i] = -1;
        orchestrator_stage_by_thread_[i] = -1;
    }

    const int32_t raw_strategy = runtime->dev.pipeline_strategy;
    bool use_baseline = raw_strategy < 0;

    if (!use_baseline) {
        if (raw_strategy != static_cast<int32_t>(PipelineStrategy::S2_O2_SPLIT_CTRL_STRATEGY2)) {
            LOG_WARN("Pipeline strategy %d is not enabled; falling back to baseline layout", raw_strategy);
            use_baseline = true;
        } else {
            pipeline_layout_ = resolve_pipeline_layout(raw_strategy);
            int32_t requested_total = pipeline_layout_.scheduler_threads + pipeline_layout_.orchestrator_threads;
            if (aicpu_thread_num_ < requested_total) {
                LOG_WARN(
                    "Pipeline strategy %s needs %d AICPU threads, got %d; falling back to baseline layout",
                    pipeline_layout_.name, requested_total, aicpu_thread_num_
                );
                use_baseline = true;
            }
        }
    }

    if (use_baseline) {
        pipeline_layout_ = resolve_pipeline_layout(PIPELINE_STRATEGY_UNSET_BASELINE);
        orch_thread_num_ = 1;
        sched_thread_num_ = aicpu_thread_num_ - orch_thread_num_;
        for (int32_t i = 0; i < sched_thread_num_ && i < MAX_AICPU_THREADS; ++i) {
            scheduler_index_by_thread_[i] = i;
        }
        if (sched_thread_num_ >= 0 && sched_thread_num_ < MAX_AICPU_THREADS) {
            orchestrator_stage_by_thread_[sched_thread_num_] = 0;
        }
        runtime->dev.pipeline_strategy = PIPELINE_STRATEGY_UNSET_BASELINE;
    } else {
        sched_thread_num_ = pipeline_layout_.scheduler_threads;
        orch_thread_num_ = pipeline_layout_.orchestrator_threads;
        for (int32_t i = 0; i < aicpu_thread_num_ && i < PIPELINE_LAYOUT_MAX_THREADS; ++i) {
            scheduler_index_by_thread_[i] = pipeline_layout_.scheduler_index_by_thread[i];
            orchestrator_stage_by_thread_[i] = pipeline_layout_.orchestrator_stage_by_thread[i];
        }
        runtime->dev.pipeline_strategy = raw_strategy;
    }

    if (sched_thread_num_ < 0) sched_thread_num_ = 0;
    primary_orch_thread_idx_ = -1;
    for (int32_t i = 0; i < aicpu_thread_num_ && i < MAX_AICPU_THREADS; ++i) {
        if (orchestrator_stage_by_thread_[i] == 0) {
            primary_orch_thread_idx_ = i;
            break;
        }
    }
    if (primary_orch_thread_idx_ < 0) {
        primary_orch_thread_idx_ = sched_thread_num_;
    }
}

int32_t AicpuExecutor::scheduler_idx_for_thread(int32_t thread_idx) const {
    if (thread_idx < 0 || thread_idx >= MAX_AICPU_THREADS) {
        return -1;
    }
    return scheduler_index_by_thread_[thread_idx];
}

int32_t AicpuExecutor::orch_stage_idx_for_thread(int32_t thread_idx) const {
    if (thread_idx < 0 || thread_idx >= MAX_AICPU_THREADS) {
        return -1;
    }
    return orchestrator_stage_by_thread_[thread_idx];
}

int32_t AicpuExecutor::scheduler_context_idx_for_thread(int32_t thread_idx) const {
    int32_t sched_idx = scheduler_idx_for_thread(thread_idx);
    if (sched_idx >= 0) {
        return sched_idx;
    }
    int32_t orch_stage_idx = orch_stage_idx_for_thread(thread_idx);
    if (orch_stage_idx >= 0) {
        return sched_thread_num_ + orch_stage_idx;
    }
    return thread_idx;
}

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

    resolve_thread_layout(runtime);
    serial_orch_sched_ = runtime->dev.serial_orch_sched;

    if (aicpu_thread_num_ < 1 || aicpu_thread_num_ > MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid aicpu_thread_num: %d", aicpu_thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }
    LOG_INFO_V0(
        "AicpuExecutor layout: strategy=%s threads=%d sched=%d orch=%d primary_orch=%d clusters=%s",
        pipeline_layout_.name, aicpu_thread_num_, sched_thread_num_, orch_thread_num_, primary_orch_thread_idx_,
        pipeline_layout_.cluster_layout
    );

    if (sched_ctx_.init(runtime, aicpu_thread_num_, sched_thread_num_, get_platform_regs()) != 0) {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    LOG_INFO_V0("AicpuExecutor: Init complete");
    return 0;
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

    // Registration always (re)loads: the slot may have been reused after an
    // unregister, so dlclose any stale handle before dlopen'ing the new SO.
    // No AicpuPhase::SoLoad stamp here: that phase times the dlopen within a
    // simpler_run launch, but loading now happens in the separate
    // register_callable launch which has no phase buffer (the run-path SoLoad
    // slot is simply 0 now that run never loads).
    LOG_INFO_V0("Thread %d: New orch SO detected (callable_id=%d), (re)loading", thread_idx, callable_id);
    if (entry.handle != nullptr) {
        dlclose(entry.handle);
    }
    if (entry.path[0] != '\0') {
        // Unlink the old file so the new open() lands on a fresh inode.
        unlink(entry.path);
    }
    entry = OrchSoEntry{};

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
            LOG_INFO_V0("Thread %d: Cannot create SO at %s (errno=%d), trying next path", thread_idx, so_path, errno);
            continue;
        }
        ssize_t written = write(fd, so_data, so_size);
        close(fd);
        if (written != static_cast<ssize_t>(so_size)) {
            LOG_INFO_V0("Thread %d: Cannot write SO to %s (errno=%d), trying next path", thread_idx, so_path, errno);
            unlink(so_path);
            continue;
        }
        file_created = true;
        LOG_INFO_V0("Thread %d: Created SO file at %s (%zu bytes)", thread_idx, so_path, so_size);
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
    LOG_INFO_V0("Thread %d: dlopen succeeded, handle=%p", thread_idx, handle);

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
    // Publish the resolved index so per-thread readers in this `.so` (notably
    // the AICPU phase-record slot) agree with the executor. On sim the basic
    // affinity gate leaves the index unset (-1); without this the sub-phase
    // stamps below would have no valid slot and silently drop. Idempotent
    // onboard, where the filter gate already set this same value.
    platform_aicpu_affinity_set_thread_idx(thread_idx);
    LOG_INFO_V0("Thread %d: Start (exec_idx=%d)", thread_idx, affinity_exec_idx);
    int32_t scheduler_idx = scheduler_idx_for_thread(thread_idx);
    int32_t orch_stage_idx = orch_stage_idx_for_thread(thread_idx);

    // Orchestrator check
    if (orch_stage_idx == 0) {
#if PTO2_PROFILING
        uint64_t orch_cycle_start = 0;
#endif
#if PTO2_ORCH_PROFILING
        int32_t pto2_submitted_tasks = -1;
#endif
        // Orchestrator thread: load + run the device orchestration SO. The braces
        // scope the per-callable dlopen / SO-table locals to this block.
        {
            // Per-callable_id dispatch: the orch SO state lives in
            // `orch_so_table_[callable_id]`, loaded once by the
            // register_callable entry. The run path only consumes it — it never
            // loads. A missing handle means run was reached without a prior
            // successful registration, which is a caller/scheduling bug.
            const int32_t callable_id = runtime->get_active_callable_id();
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

                // Build the entry-arg once per run; both the config call below and
                // the orchestration entry (consumed at orch_args_cached_) use it.
                orch_args_cached_.create_from_chip_args(runtime->get_orch_args());

                // Validate arg count on every run against the registered SO.
                if (*p_config_func != nullptr) {
                    PTO2OrchestrationConfig cfg = (*p_config_func)(orch_args_cached_);
                    LOG_INFO_V0("Thread %d: Config: expected_args=%d", thread_idx, cfg.expected_arg_count);
                    if (cfg.expected_arg_count > 0) {
                        const ChipStorageTaskArgs &args_validate = runtime->get_orch_args();
                        int32_t actual_arg_count = args_validate.tensor_count() + args_validate.scalar_count();
                        if (actual_arg_count < cfg.expected_arg_count) {
                            LOG_ERROR(
                                "Thread %d: arg_count %d < expected %d", thread_idx, actual_arg_count,
                                cfg.expected_arg_count
                            );
                            // The registered SO is fine — these run args are
                            // incompatible with it. Run only consumes the slot
                            // (no reload), so leave the table intact and just
                            // fail this run; unblock scheduler threads first so
                            // they don't spin forever.
                            runtime_init_ready_.store(true, std::memory_order_release);
                            return -1;
                        }
                    }
                } else {
                    LOG_INFO_V0("Thread %d: No config function, using defaults", thread_idx);
                }

                // sm_handle / rt are bound to *this* run's memory and must be
                // (re)created every run, regardless of whether the SO itself was
                // reused above.
                const ChipStorageTaskArgs &args = runtime->get_orch_args();
                int32_t arg_count = args.tensor_count() + args.scalar_count();
                LOG_INFO_V0("Thread %d: sm_ptr=%p, arg_count=%d", thread_idx, runtime->get_gm_sm_ptr(), arg_count);
                for (int32_t i = 0; i < args.tensor_count() && i < 20; i++) {
                    const Tensor &t = args.tensor(i);
                    LOG_INFO_V0(
                        "Thread %d: orch_args[%d] = TENSOR(data=0x%lx, ndims=%u, dtype=%u)", thread_idx, i,
                        static_cast<uint64_t>(t.buffer.addr), t.ndims, static_cast<unsigned>(t.dtype)
                    );
                }
                for (int32_t i = 0; i < args.scalar_count() && (args.tensor_count() + i) < 20; i++) {
                    LOG_INFO_V0(
                        "Thread %d: orch_args[%d] = SCALAR(0x%lx)", thread_idx, args.tensor_count() + i,
                        static_cast<uint64_t>(args.scalar(i))
                    );
                }
                sm_ptr = runtime->get_gm_sm_ptr();
            }

            // Prebuilt-arena fast path. Host uploads the runtime arena image
            // on cache miss; cache hits reuse the resident device arena. AICPU
            // re-wires arena-internal pointers to device addresses below.
            {
                AicpuPhaseScope arena_wire(AicpuPhase::ArenaWire);
                void *prebuilt_arena = runtime->get_prebuilt_arena_base();
                size_t off_runtime = runtime->get_prebuilt_runtime_offset();
                if (prebuilt_arena == nullptr) {
                    LOG_ERROR("Thread %d: prebuilt_arena_base is null", thread_idx);
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
                runtime_arena_.attach(prebuilt_arena, DeviceArena::kDefaultBaseAlign);
                rt = reinterpret_cast<PTO2Runtime *>(static_cast<char *>(prebuilt_arena) + off_runtime);

                // Wire every arena-internal pointer field (host wrote host-mirror
                // addresses; we overwrite them with device addresses).
                runtime_wire_arena_pointers(runtime_arena_, rt->prebuilt_layout, rt);
                sm_size = PTO2SharedMemoryHandle::calculate_size_per_ring(rt->prebuilt_layout.sizing.task_window_sizes);
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; ++r) {
                    LOG_INFO_V0(
                        "Thread %d: Ring %d sizes: task_window=%" PRIu64 " heap=%" PRIu64 " dep_pool=%d", thread_idx, r,
                        rt->prebuilt_layout.sizing.task_window_sizes[r], rt->prebuilt_layout.sizing.heap_sizes[r],
                        rt->prebuilt_layout.sizing.dep_pool_capacities[r]
                    );
                }
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
                runtime_finalize_after_wire(rt, sched_ctx_.aic_count(), sched_ctx_.aiv_count());
                bool use_strategy2_submit_ctrl =
                    pipeline_layout_.strategy == PipelineStrategy::S2_O2_SPLIT_CTRL_STRATEGY2;
                bool use_deferred_submit = use_strategy2_submit_ctrl && sched_thread_num_ == 2 && orch_thread_num_ == 2;
                if (runtime->dev.pipeline_defer_submit_disabled) {
                    use_deferred_submit = false;
                }
                rt->orchestrator.enable_submit_pipeline(
                    orch_thread_num_, false, use_deferred_submit, use_deferred_submit, use_deferred_submit
                );
#if PTO2_PROFILING
                rt->orchestrator.l2_swimlane_level = get_l2_swimlane_level();
                {
                    auto &orch = rt->orchestrator;
                    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                        auto &alloc = orch.rings[r].task_allocator;
                        scope_stats_set_ring_capacity(
                            r, alloc.window_size(), alloc.heap_capacity(),
                            rt->prebuilt_layout.sizing.dep_pool_capacities[r]
                        );
                    }
                    scope_stats_set_tensormap_capacity(orch.tensor_map.pool_capacity());
                }
#endif

                // Wire scheduler context to the newly created PTO2Runtime before
                // releasing scheduler threads from runtime_init_ready_.
                sched_ctx_.bind_runtime(rt);
            }

            runtime_init_ready_.store(true, std::memory_order_release);

#if PTO2_PROFILING
            if (get_l2_swimlane_level() >= L2SwimlaneLevel::ORCH_PHASES) {
                l2_swimlane_aicpu_set_orch_thread_idx(thread_idx);
            }
#endif

#if PTO2_PROFILING
            // dep_gen plugs into the orchestrator thread (single-instance subsystem):
            // record the per-thread ready_queue index before any submit_task fires
            // inside orch_func_.
            if (is_dep_gen_enabled()) {
                dep_gen_aicpu_set_orch_thread_idx(thread_idx);
            }

            // scope_stats streams scope_end records off the orchestrator thread:
            // record the per-thread ready_queue index. No-op (writer shared
            // state null) when scope_stats is disabled; the current buffer is
            // popped lazily on the first scope_end append.
            scope_stats_aicpu_set_orch_thread_idx(thread_idx);
#endif

#if PTO2_PROFILING
            orch_cycle_start = get_sys_cnt_aicpu();
#endif
            framework_bind_runtime(rt);
            if (*p_bind != nullptr) {
                (*p_bind)(rt);
            }
            rt_scope_begin(rt);
            (*p_func)(orch_args_cached_);
            rt_scope_end(rt);
            rt->orchestrator.stop_submit_pipeline();

#if PTO2_PROFILING
            // Flush the (potentially partially-filled) DepGenBuffer so the host
            // collector can pick it up before this orchestrator thread joins.
            if (is_dep_gen_enabled()) {
                dep_gen_aicpu_flush();
            }
            // Push the partially-filled scope_stats buffer so the host gets the
            // final scope_end records. Idempotent / no-op when disabled.
            scope_stats_aicpu_flush_buffers();
#endif
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
                "Thread %d: === Orchestrator Profiling: %" PRId64 " tasks, total=%.3fus ===", thread_idx,
                static_cast<int64_t>(p.submit_count), cycles_to_us(total)
            );
            LOG_INFO_V9(
                "Thread %d:   task+heap_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
                cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
                static_cast<uint64_t>(p.alloc_atomic_count)
            );
            LOG_INFO_V9(
                "Thread %d:   sync_tensormap : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.sync_cycle),
                p.sync_cycle * 100.0 / total
            );
            LOG_INFO_V9(
                "Thread %d:   lookup+dep     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.lookup_cycle),
                p.lookup_cycle * 100.0 / total
            );
            LOG_INFO_V9(
                "Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.insert_cycle),
                p.insert_cycle * 100.0 / total
            );
            LOG_INFO_V9(
                "Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
                cycles_to_us(p.args_cycle), p.args_cycle * 100.0 / total, static_cast<uint64_t>(p.args_atomic_count)
            );
            LOG_INFO_V9(
                "Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus", thread_idx,
                cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total,
                cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle)
            );
            LOG_INFO_V9(
                "Thread %d:   avg/task       : %.3fus", thread_idx,
                p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0
            );

#if PTO2_TENSORMAP_PROFILING
            PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
            LOG_INFO_V9("Thread %d: === TensorMap Lookup Stats ===", thread_idx);
            LOG_INFO_V9(
                "Thread %d:   lookups        : %" PRIu64 ", inserts: %" PRIu64 "", thread_idx,
                static_cast<uint64_t>(tp.lookup_count), static_cast<uint64_t>(tp.insert_count)
            );
            LOG_INFO_V9(
                "Thread %d:   chain walked   : total=%" PRIu64 ", avg=%.1f, max=%d", thread_idx,
                static_cast<uint64_t>(tp.lookup_chain_total),
                tp.lookup_count > 0 ? static_cast<double>(tp.lookup_chain_total) / tp.lookup_count : 0.0,
                tp.lookup_chain_max
            );
            LOG_INFO_V9(
                "Thread %d:   overlap checks : %" PRIu64 ", hits=%" PRIu64 " (%.1f%%)", thread_idx,
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
            // L2SwimlaneAicpuSchedPhaseRecord[] + L2SwimlaneAicpuOrchPhaseRecord[]
            // streams that already cover everything inside submit_task().
            int32_t total_tasks = 0;
            if (rt->orchestrator.sm_header) {
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                    total_tasks +=
                        rt->orchestrator.sm_header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
                }
            }

#if PTO2_ORCH_PROFILING
            pto2_submitted_tasks = total_tasks;
#endif

            // Signal completion to the orchestrator state machine
            rt_orchestration_done(rt);

            sched_ctx_.on_orchestration_done(runtime, rt, thread_idx, total_tasks);
        }
#if PTO2_PROFILING
        uint64_t orch_end_ts = get_sys_cnt_aicpu();
        // Ride the orch window home to the host phase buffer so the host emits
        // it as an `Orch` [STRACE] marker (the everyday path). The verbose
        // per-thread device-log line below is now opt-in deep-dive.
        aicpu_phase_set_window(AicpuPhase::OrchWindow, static_cast<uint64_t>(orch_cycle_start), orch_end_ts);
#if PTO2_ORCH_PROFILING
        LOG_INFO_V9(
            "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", thread_idx,
            static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
            cycles_to_us(orch_end_ts - orch_cycle_start)
        );
        if (pto2_submitted_tasks >= 0) {
            LOG_INFO_V9(
                "PTO2 total submitted tasks = %d, already executed %d tasks", pto2_submitted_tasks,
                sched_ctx_.completed_tasks_count()
            );
        }
#endif  // PTO2_ORCH_PROFILING
#endif  // PTO2_PROFILING
        LOG_INFO_V0("Thread %d: Orchestrator completed", thread_idx);
    } else if (orch_stage_idx > 0) {
        LOG_INFO_V0(
            "Thread %d: Orchestrator pipeline stage %d waiting for primary orchestrator", thread_idx, orch_stage_idx
        );
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
        if (rt != nullptr) {
#if PTO2_PROFILING
            if (get_l2_swimlane_level() >= L2SwimlaneLevel::ORCH_PHASES) {
                l2_swimlane_aicpu_set_orch_thread_idx(thread_idx);
            }
            uint64_t orch_stage_cycle_start = get_sys_cnt_aicpu();
            uint64_t orch_stage_active_cycles = rt->orchestrator.run_submit_pipeline_worker(orch_stage_idx, thread_idx);
            uint64_t orch_stage_cycle_end = get_sys_cnt_aicpu();
            if (rt->orchestrator.submit_pipeline_work_available.load(std::memory_order_acquire) ||
                orch_stage_active_cycles > 0) {
                LOG_INFO_V9(
                    "Thread %d: orch_stage_start=%" PRIu64 " orch_stage_end=%" PRIu64
                    " orch_stage=%d orch_stage_cost=%.3fus orch_stage_active_cost=%.3fus",
                    thread_idx, static_cast<uint64_t>(orch_stage_cycle_start),
                    static_cast<uint64_t>(orch_stage_cycle_end), orch_stage_idx,
                    cycles_to_us(orch_stage_cycle_end - orch_stage_cycle_start), cycles_to_us(orch_stage_active_cycles)
                );
            }
#else
            (void)rt->orchestrator.run_submit_pipeline_worker(orch_stage_idx, thread_idx);
#endif
        } else {
            LOG_ERROR("Thread %d: rt is null, skipping orchestrator pipeline stage %d", thread_idx, orch_stage_idx);
        }
        LOG_INFO_V0("Thread %d: Orchestrator pipeline stage %d completed", thread_idx, orch_stage_idx);
    }

    // Scheduler thread (orchestrator thread skips dispatch and exits after orchestration)
    if (!sched_ctx_.is_completed() && scheduler_idx >= 0) {
        // Device orchestration: wait for the primary orchestrator to initialize the SM header
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
        if (rt == nullptr) {
            LOG_ERROR("Thread %d: rt is null after orchestrator error, skipping dispatch", thread_idx);
        } else {
            sched_ctx_.bind_runtime(rt);
            if (serial_orch_sched_) {
                sched_ctx_.wait_for_orchestration_done_before_dispatch(runtime, thread_idx);
            }
            int32_t completed = sched_ctx_.resolve_and_dispatch(runtime, scheduler_idx);
            if (completed < 0) {
                LOG_ERROR("Thread %d: Scheduler failed with rc=%d", thread_idx, completed);
                run_rc = completed;
            } else {
                LOG_INFO_V0("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
            }
        }
    }

    // Always shutdown AICore — even if sched_ctx_.completed_ was already true.
    // platform_deinit_aicore_regs is idempotent; orchestrator threads have
    // core_trackers_[thread_idx].core_num() == 0 so they skip the loop harmlessly.
    int32_t shutdown_rc = sched_ctx_.shutdown(thread_idx);
    if (shutdown_rc != 0 && run_rc == 0) {
        run_rc = shutdown_rc;
    }

    LOG_INFO_V0("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int32_t prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == aicpu_thread_num_) {
        finished_.store(true, std::memory_order_release);
        // Destroy PTO2 runtime. sm_handle / rt are recreated every run so we
        // always tear them down here, but we keep the per-cid orch SO entries
        // alive — they are loaded once by register_callable and consumed by
        // every subsequent run.
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
            runtime_destroy(rt, runtime_arena_);
            rt = nullptr;
        }
    }

    return run_rc;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for the device-copied Runtime range (`dev`).
    //    Next round's Host DMA (rtMemcpy) writes fresh bytes to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from
    //    HBM. Only `dev` is uploaded, so only `dev` needs invalidation.
    cache_invalidate_range(runtime, sizeof(runtime->dev));

    // Reset all SchedulerContext-owned state in one place.
    sched_ctx_.deinit();

    finished_count_.store(0, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);

    aicpu_thread_num_ = 0;
    sched_thread_num_ = 0;
    orch_thread_num_ = 0;
    primary_orch_thread_idx_ = -1;
    pipeline_layout_ = {};
    serial_orch_sched_ = false;
    for (int32_t i = 0; i < MAX_AICPU_THREADS; ++i) {
        scheduler_index_by_thread_[i] = -1;
        orchestrator_stage_by_thread_[i] = -1;
    }

    orch_args_cached_.reset();
    // orch_so_table_ entries are intentionally preserved across deinit: they
    // are loaded once by register_callable and consumed by every subsequent
    // run. The destructor releases them at process teardown.

    // Clear file-scope PTO2Runtime pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    // Clear dep_gen file-local bookkeeping. No-op when dep_gen is disabled.
    dep_gen_aicpu_finalize();

    LOG_INFO_V0("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    LOG_INFO_V0("DeInit: AicpuExecutor reset complete");
}

// ===== Public Entry Point =====

// Device orchestration SO registration entry. Exported directly by the runtime
// (not via a platform forwarding shell): registration is a TMARB-only ability,
// so the symbol lives where the capability does. host_build_graph does not
// export it at all (host-side orchestration has nothing to register).
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_register_callable(void *arg) {
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
    LOG_INFO_V0("simpler_aicpu_register_callable: completed for callable_id=%d", args->active_callable_id);
    return 0;
}

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

    LOG_INFO_V0("%s", "aicpu_execute: Starting AICPU kernel execution");

    // Each phase is bracketed by its own scope so the start/end boundaries are
    // visible and an early `return` still records the end via the guard dtor.
    // rc / runtime_rc are declared out here because they outlive their phase.
    {
        AicpuPhaseScope preamble(AicpuPhase::Preamble);
        g_aicpu_executor.init(runtime);
        while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
            if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
                LOG_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
                return -1;
            }
        }
    }

    int32_t rc = 0;
    {
        AicpuPhaseScope graph_build(AicpuPhase::GraphBuild);
        rc = g_aicpu_executor.run(runtime);
    }
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
    }

    // PostOrch measures only the real teardown (deinit), and only on the last
    // thread to finish. Stamping it on every thread would let an orchestrator
    // thread that finished early (it submits then returns while the scheduler
    // threads are still draining) open the window at its early exit, so the
    // cross-thread max(end)-min(start) reduction would absorb the orch-waits-for-
    // sched overlap into post_orch — inflating it well past the actual teardown.
    // read_pto2_runtime_status is two atomic loads every thread needs, so it
    // stays outside the scope.
    int32_t runtime_rc = read_pto2_runtime_status(runtime);
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        AicpuPhaseScope post_orch(AicpuPhase::PostOrch);
        LOG_INFO_V0("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    if (runtime_rc != 0) {
        LOG_ERROR("aicpu_execute: PTO2 runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    if (rc != 0) {
        return rc;
    }

    LOG_INFO_V0("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
