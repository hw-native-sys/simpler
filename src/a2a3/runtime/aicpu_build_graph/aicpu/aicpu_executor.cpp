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

#if PTO2_PROFILING
// Accumulated nanoseconds per sub-step
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#endif

// Device orchestration function signature (loaded via dlopen).
// The orchestration .so receives a PTO2Runtime* (with ops table populated)
// instead of a raw shared-memory pointer.
typedef void (*DeviceOrchestrationFunc)(PTO2Runtime *rt, const ChipStorageTaskArgs &orch_args);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(const ChipStorageTaskArgs &orch_args);

constexpr int32_t MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int32_t MAX_AIC_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD;
constexpr int32_t MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;

constexpr int32_t MAX_IDLE_ITERATIONS = 800000;       // ~20s idle then scheduler gives up (avoid long hang)
constexpr int32_t STALL_LOG_INTERVAL = 50000;         // DEV_ALWAYS every N idle iters to debug hang
constexpr int32_t FATAL_ERROR_CHECK_INTERVAL = 1024;  // Check orchestrator error every N idle iters
constexpr int32_t STALL_DUMP_READY_MAX = 8;
constexpr int32_t STALL_DUMP_WAIT_MAX = 4;
constexpr int32_t STALL_DUMP_CORE_MAX = 8;
constexpr int32_t PROGRESS_VERBOSE_THRESHOLD = 10;  // log every completion for the first N tasks
constexpr int32_t PROGRESS_LOG_INTERVAL = 250;      // log every N completions after threshold

static PTO2Runtime *rt{nullptr};

// Per-core dispatch payload storage (one per physical core)
static PTO2DispatchPayload s_pto2_payload_per_core[RUNTIME_MAX_WORKER];

// Core information for discovery (with register address for fast dispatch)
struct CoreInfo {
    int32_t worker_id;          // Index in runtime.workers[]
    uint32_t physical_core_id;  // Hardware physical core ID (from AICore)
    uint64_t reg_addr;          // Cached register address for fast access
    CoreType core_type;
};

struct CoreTypeTracker {
    int32_t idle_count;
    int32_t running_count;
    int32_t idle[MAX_CORES_PER_THREAD];
    int32_t running[MAX_CORES_PER_THREAD];

    void move_idle_to_running(int32_t idx) {
        running[running_count++] = idle[idx];
        idle[idx] = idle[--idle_count];
    }

    void move_running_to_idle(int32_t idx) {
        idle[idle_count++] = running[idx];
        running[idx] = running[--running_count];
    }

    int32_t find_idle_index(int32_t core_id) {
        for (int32_t i = 0; i < idle_count; i++) {
            if (idle[i] == core_id) return i;
        }
        return -1;
    }
};

struct Cluster {
    int32_t aic_core_id;
    int32_t aiv_core_ids[2];
};

struct CoreStateTracker {
    CoreTypeTracker by_type[2];  // indexed by static_cast<int32_t>(CoreType)
    Cluster clusters[MAX_AIC_PER_THREAD];
    int32_t cluster_count;

    CoreTypeTracker &aic() { return by_type[0]; }
    CoreTypeTracker &aiv() { return by_type[1]; }

    template <CoreType CT>
    CoreTypeTracker &get() {
        return by_type[static_cast<int32_t>(CT)];
    }

    int32_t find_cluster_for_shape(PTO2ResourceShape shape, bool *core_idle) {
        for (int32_t i = 0; i < cluster_count; i++) {
            Cluster &c = clusters[i];
            switch (shape) {
            case PTO2ResourceShape::AIC_ONLY:
                if (core_idle[c.aic_core_id]) return i;
                break;
            case PTO2ResourceShape::AIV_X1:
                if (core_idle[c.aiv_core_ids[0]] || core_idle[c.aiv_core_ids[1]]) return i;
                break;
            case PTO2ResourceShape::AIV_X2:
                if (core_idle[c.aiv_core_ids[0]] && core_idle[c.aiv_core_ids[1]]) return i;
                break;
            case PTO2ResourceShape::AIC_AIV_X1:
                if (core_idle[c.aic_core_id] && (core_idle[c.aiv_core_ids[0]] || core_idle[c.aiv_core_ids[1]]))
                    return i;
                break;
            case PTO2ResourceShape::AIC_AIV_X2:
                if (core_idle[c.aic_core_id] && core_idle[c.aiv_core_ids[0]] && core_idle[c.aiv_core_ids[1]]) return i;
                break;
            }
        }
        return -1;
    }
};

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
    int32_t cores_total_num_{0};
    int32_t thread_cores_num_{0};  // Cores per scheduler thread (0 for orchestrator when thread_num_==4)
    int32_t core_count_per_thread_[MAX_AICPU_THREADS];  // Actual core count per thread
    int32_t core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // Core discovery arrays (with register addresses)
    CoreInfo aic_cores_[MAX_CORES_PER_THREAD];
    CoreInfo aiv_cores_[MAX_CORES_PER_THREAD];
    int32_t aic_count_{0};
    int32_t aiv_count_{0};

    // Fast lookup: core_id -> reg_addr (for register-based dispatch)
    uint64_t core_id_to_reg_addr_[MAX_CORES_PER_THREAD];

    // Per-core monotonic dispatch counter for register protocol uniqueness.
    // Multi-ring task_ids can collide in the lower 32 bits (e.g., ring 0 local 0
    // and ring 1 local 0 both truncate to 0), breaking the AICore's last_reg_val
    // duplicate detection and causing false-positive COND completion. A per-core
    // counter guarantees each dispatch writes a unique DATA_MAIN_BASE value.
    uint32_t dispatch_seq_by_core_[RUNTIME_MAX_WORKER]{};

    // Per-core subtask slot tracking (which PTO2SubtaskSlot is running on each core)
    PTO2SubtaskSlot executing_subslot_by_core_[RUNTIME_MAX_WORKER]{};

    // Per-core slot state tracking (PTO2TaskSlotState* for the running task on each core)
    PTO2TaskSlotState *executing_slot_state_by_core_[RUNTIME_MAX_WORKER]{};

    // Platform register base address array (set via get_platform_regs())
    uint64_t regs_{0};

    // Track executing register task_id per core (AICPU_TASK_INVALID = idle).
    // NOTE: this is NOT the task_id; it is the per-core dispatch id used by the
    // register protocol (derived from dispatch_seq_by_core_ and masked by TASK_ID_MASK).
    int32_t executing_reg_task_ids_[MAX_CORES_PER_THREAD];
    CoreStateTracker trackers_[MAX_AICPU_THREADS];
    bool core_idle_[MAX_CORES_PER_THREAD];

    // ===== Task queue state (managed by scheduler ready queues) =====

    // Task execution tracking
    std::atomic<int32_t> completed_tasks_{0};
    int32_t total_tasks_{0};
    std::atomic<int32_t> finished_count_{0};
    // Device orchestration: set by last orchestrator when graph is built; schedulers poll it.
    // volatile prevents the compiler from hoisting the load out of spin loops.
    volatile bool orchestrator_done_{false};
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> runtime_init_ready_{false};
    std::atomic<bool> pto2_init_complete_{false};  // init block finished; others wait for this

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
    const ChipStorageTaskArgs *orch_args_cached_{nullptr};

    // ===== Performance profiling state =====
    uint64_t dispatch_timestamps_[RUNTIME_MAX_WORKER];  // Per-core AICPU dispatch timestamp
    uint32_t
        core_dispatch_counts_[RUNTIME_MAX_WORKER];  // Per-core total dispatched task counter (for buffer management)

    uint64_t *func_id_to_addr_;
    uint64_t get_function_bin_addr(int func_id) const {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t handshake_all_cores(Runtime *runtime);
    void assign_cores_to_threads();
    void reassign_cores_for_all_threads();
    int32_t resolve_and_dispatch_pto2(Runtime *runtime, int32_t thread_idx);
    int32_t shutdown_aicore(Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
    void emergency_shutdown(Runtime *runtime);
    void diagnose_stuck_state(
        Runtime *runtime, int32_t thread_idx, const int32_t *cur_thread_cores, int32_t core_num, Handshake *hank
    );

    // Build slim PTO2DispatchPayload: only function_bin_addr + args.
    // Metadata (task_id, subslot, kernel_id, core_type) stays in TaskDescriptor.
    // Dispatch order: tensor args first, then scalar args.
    void build_pto2_payload(PTO2DispatchPayload &out, int32_t kernel_id, PTO2TaskPayload &task_pl) {
        uint64_t callable_addr = get_function_bin_addr(kernel_id);
        const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
        out.function_bin_addr = callable->resolved_addr();
        int32_t n = 0;
        for (int32_t i = 0; i < task_pl.tensor_count; i++) {
            task_pl.tensors[i].update_start_offset();
            out.args[n++] = reinterpret_cast<uint64_t>(&task_pl.tensors[i]);
        }
        for (int32_t i = 0; i < task_pl.scalar_count; i++) {
            out.args[n++] = task_pl.scalars[i];
        }
    }

    // Template methods for Phase 1 and Phase 2
    template <CoreType CT>
    void check_running_cores_for_completion(
        int32_t thread_idx, CoreTypeTracker &ct, Handshake *hank, int32_t &completed_this_turn,
        int32_t &cur_thread_completed, bool &made_progress, PTO2TaskSlotState *deferred_release_slot_states[],
        int32_t &deferred_release_count, PTO2LocalReadyBuffer *local_bufs
#if PTO2_PROFILING
        ,
        bool profiling_enabled, uint32_t &phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
        ,
        uint64_t &complete_probe_count, uint64_t &complete_hit_count, uint64_t &notify_edges_total,
        int32_t &notify_max_degree, uint64_t &notify_tasks_enqueued, uint64_t &fanin_edges_total,
        int32_t &fanin_max_degree, uint64_t &sched_complete_perf_cycle
#endif
    ) {
        for (int32_t i = ct.running_count - 1; i >= 0; i--) {
            int32_t core_id = ct.running[i];
            uint64_t reg_addr = core_id_to_reg_addr_[core_id];

            int32_t expected_reg_task_id = executing_reg_task_ids_[core_id];
            uint64_t reg_val = read_reg(reg_addr, RegId::COND);
            int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
            int32_t reg_state = EXTRACT_TASK_STATE(reg_val);
            bool done = reg_task_id == expected_reg_task_id && reg_state == TASK_FIN_STATE;
#if PTO2_SCHED_PROFILING
            if (profiling_enabled) {
                complete_probe_count++;
                if (done) {
                    complete_hit_count++;
                }
            }
#endif

            if (done) {
                executing_reg_task_ids_[core_id] = AICPU_TASK_INVALID;
                PTO2SubtaskSlot subslot = executing_subslot_by_core_[core_id];
                PTO2TaskSlotState &slot_state = *executing_slot_state_by_core_[core_id];

                // Two-stage completion: mark subtask done, then handle mixed-task completion
                bool mixed_complete = rt->scheduler.on_subtask_complete(slot_state, subslot);
                if (mixed_complete) {
#if PTO2_SCHED_PROFILING
                    PTO2CompletionStats cstats =
                        rt->scheduler.on_mixed_task_complete(slot_state, thread_idx, local_bufs);
                    notify_edges_total += cstats.fanout_edges;
                    if (cstats.fanout_edges > notify_max_degree) notify_max_degree = cstats.fanout_edges;
                    notify_tasks_enqueued += cstats.tasks_enqueued;
                    phase_complete_count++;
#else
                    rt->scheduler.on_mixed_task_complete(slot_state, local_bufs);
#if PTO2_PROFILING
                    phase_complete_count++;
#endif
#endif
                    if (deferred_release_count < 256) {
                        deferred_release_slot_states[deferred_release_count++] = &slot_state;
                    } else {
                        DEV_ALWAYS("Thread %d: release", thread_idx);
                        while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                            int32_t fe = rt->scheduler.on_task_release(
                                *deferred_release_slot_states[--deferred_release_count], thread_idx
                            );
#else
                            int32_t fe =
                                rt->scheduler.on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
                            (void)fe;
#if PTO2_SCHED_PROFILING
                            fanin_edges_total += fe;
                            if (fe > fanin_max_degree) fanin_max_degree = fe;
#endif
                        }
                        deferred_release_slot_states[deferred_release_count++] = &slot_state;
                    }
                }
                ct.move_running_to_idle(i);
                core_idle_[core_id] = true;
#if PTO2_PROFILING
                if (profiling_enabled) {
#if PTO2_SCHED_PROFILING
                    uint64_t t_perf_start = get_sys_cnt_aicpu();
#endif
                    Handshake *h = &hank[core_id];
                    uint64_t finish_ts = get_sys_cnt_aicpu();
                    PerfBuffer *perf_buf = reinterpret_cast<PerfBuffer *>(h->perf_records_addr);

                    // Pre-extract fanout (platform layer cannot depend on PTO2DepListEntry)
                    uint64_t fanout_arr[RUNTIME_MAX_FANOUT];
                    int32_t fanout_n = 0;
                    PTO2DepListEntry *cur = slot_state.fanout_head;
                    while (cur != nullptr && fanout_n < RUNTIME_MAX_FANOUT) {
                        fanout_arr[fanout_n++] = cur->slot_state->task->task_id.raw;
                        cur = cur->next;
                    }

                    int32_t perf_slot_idx = static_cast<int32_t>(executing_subslot_by_core_[core_id]);
                    if (perf_aicpu_complete_record(
                            perf_buf, static_cast<uint32_t>(expected_reg_task_id), slot_state.task->task_id.raw,
                            slot_state.task->kernel_id[perf_slot_idx], CT, dispatch_timestamps_[core_id], finish_ts,
                            fanout_arr, fanout_n
                        ) != 0) {
                        DEV_ERROR(
                            "Core %d: perf_aicpu_complete_record failed for task 0x%" PRIx64, core_id,
                            static_cast<uint64_t>(slot_state.task->task_id.raw)
                        );
                    }
#if PTO2_SCHED_PROFILING
                    sched_complete_perf_cycle += (get_sys_cnt_aicpu() - t_perf_start);
#endif
                }
#endif

                DEV_DEBUG(
                    "Thread %d: %s core %d completed PTO2 task %d (mixed_complete=%d)", thread_idx,
                    CT == CoreType::AIC ? "AIC" : "AIV", core_id, expected_reg_task_id, mixed_complete ? 1 : 0
                );
                cur_thread_completed++;
                if (mixed_complete) {
#if PTO2_PROFILING
                    if (get_enable_dump_tensor()) {
                        dump_tensors_for_task<PTO2_SUBTASK_SLOT_COUNT>(
                            thread_idx, slot_state, TensorDumpStage::AFTER_COMPLETION,
                            [](uint8_t active_mask, uint8_t raw_subtask_id) {
                                return pto2_subtask_active(active_mask, static_cast<PTO2SubtaskSlot>(raw_subtask_id));
                            },
                            [this](int32_t func_id) {
                                return get_function_bin_addr(func_id);
                            }
                        );
                    }
#endif
                    completed_this_turn++;
                }
                made_progress = true;
            }
        }
    }

    static const char *shape_name(PTO2ResourceShape shape) {
        switch (shape) {
        case PTO2ResourceShape::AIC_ONLY:
            return "AIC_ONLY";
        case PTO2ResourceShape::AIV_X1:
            return "AIV_X1";
        case PTO2ResourceShape::AIV_X2:
            return "AIV_X2";
        case PTO2ResourceShape::AIC_AIV_X1:
            return "AIC_AIV_X1";
        case PTO2ResourceShape::AIC_AIV_X2:
            return "AIC_AIV_X2";
        }
        return "UNKNOWN";
    }

    struct ResourceCount {
        int32_t aic;
        int32_t aiv;
    };

    static constexpr ResourceCount shape_resource_count(PTO2ResourceShape shape) {
        constexpr ResourceCount kTable[PTO2_NUM_RESOURCE_SHAPES] = {
            {1, 0},  // AIC_ONLY    = 0
            {0, 1},  // AIV_X1      = 1
            {0, 2},  // AIV_X2      = 2
            {1, 1},  // AIC_AIV_X1  = 3
            {1, 2},  // AIC_AIV_X2  = 4
        };
        return kTable[static_cast<int>(shape)];
    }

    /**
     * Returns the dispatch probe order for a given scheduler thread.
     * Widest shapes first to avoid consuming cluster resources with narrow tasks.
     * Even/odd threads use different fallback orders (AIC-first vs AIV-first)
     * to reduce contention on the same ready queue across adjacent threads.
     */
    static const PTO2ResourceShape *get_dispatch_order(int32_t thread_idx) {
        // Even threads: AIC-first fallback after widest
        static constexpr PTO2ResourceShape kEvenOrder[PTO2_NUM_RESOURCE_SHAPES] = {
            PTO2ResourceShape::AIC_AIV_X2, PTO2ResourceShape::AIC_AIV_X1, PTO2ResourceShape::AIC_ONLY,
            PTO2ResourceShape::AIV_X2,     PTO2ResourceShape::AIV_X1,
        };
        // Odd threads: AIV-first fallback after widest
        static constexpr PTO2ResourceShape kOddOrder[PTO2_NUM_RESOURCE_SHAPES] = {
            PTO2ResourceShape::AIC_AIV_X2, PTO2ResourceShape::AIV_X2,   PTO2ResourceShape::AIC_AIV_X1,
            PTO2ResourceShape::AIV_X1,     PTO2ResourceShape::AIC_ONLY,
        };
        return (thread_idx % 2 == 0) ? kEvenOrder : kOddOrder;
    }

    PTO2TaskSlotState *pop_ready_task(
        PTO2ResourceShape shape, int32_t thread_idx
#if PTO2_SCHED_PROFILING
        ,
        uint64_t &pop_hit, uint64_t &pop_miss, uint64_t &sched_dispatch_pop_cycle
#endif
    ) {
        (void)thread_idx;
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[];
        uint64_t t_pop_start = get_sys_cnt_aicpu();
        PTO2TaskSlotState *slot_state = rt->scheduler.get_ready_task(
            shape, g_sched_pop_atomic_count[thread_idx], g_sched_pop_wait_cycle[thread_idx]
        );
        sched_dispatch_pop_cycle += (get_sys_cnt_aicpu() - t_pop_start);
#else
        PTO2TaskSlotState *slot_state = rt->scheduler.get_ready_task(shape);
#endif
        if (slot_state) {
#if PTO2_SCHED_PROFILING
            pop_hit++;
#endif
        } else {
#if PTO2_SCHED_PROFILING
            pop_miss++;
#endif
        }
        return slot_state;
    }

    void dispatch_subtask_to_core(
        Runtime *runtime, CoreStateTracker &tracker, int32_t core_id, CoreType core_type, PTO2TaskSlotState &slot_state,
        PTO2SubtaskSlot subslot
#if PTO2_PROFILING
        ,
        bool profiling_enabled
#endif
#if PTO2_PROFILING
        ,
        int32_t thread_idx
#endif
    ) {
#if !PTO2_PROFILING
        (void)runtime;  // NOLINT(readability/casting)
#endif
        PTO2DispatchPayload &payload = s_pto2_payload_per_core[core_id];
        PTO2TaskDescriptor &task = *slot_state.task;
        int32_t slot_idx = static_cast<int32_t>(subslot);
        build_pto2_payload(payload, task.kernel_id[slot_idx], *slot_state.payload);
        executing_subslot_by_core_[core_id] = subslot;
        executing_slot_state_by_core_[core_id] = &slot_state;
#if PTO2_PROFILING
        if (profiling_enabled) {
            dispatch_timestamps_[core_id] = get_sys_cnt_aicpu();
            if (core_dispatch_counts_[core_id] >= PLATFORM_PROF_BUFFER_SIZE) {
                perf_aicpu_switch_buffer(runtime, core_id, thread_idx);
                core_dispatch_counts_[core_id] = 0;
            }
            core_dispatch_counts_[core_id]++;
        }
#endif

        // Per-core monotonic counter for register protocol uniqueness.
        // PTO2 task_id encodes (ring_id << 32 | local_id); truncation to uint32 loses ring_id,
        // so tasks from different rings with the same local_id would write identical DATA_MAIN_BASE
        // values. The AICore uses last_reg_val to detect new dispatches and would skip the
        // duplicate, while the stale COND register from the previous task (same local_id) would
        // cause a false-positive completion.
        dispatch_seq_by_core_[core_id]++;
        uint32_t reg_task_id = dispatch_seq_by_core_[core_id] & TASK_ID_MASK;
        // Skip reserved sentinel range [AICORE_EXIT_SIGNAL, 0x7FFFFFFF]: jump directly to 0.
        if (reg_task_id >= AICORE_EXIT_SIGNAL) {
            dispatch_seq_by_core_[core_id] += (TASK_ID_MASK - reg_task_id + 1);
            reg_task_id = dispatch_seq_by_core_[core_id] & TASK_ID_MASK;
        }
        write_reg(core_id_to_reg_addr_[core_id], RegId::DATA_MAIN_BASE, static_cast<uint64_t>(reg_task_id));

        CoreTypeTracker &ct = tracker.by_type[static_cast<int32_t>(core_type)];
        int32_t idle_idx = ct.find_idle_index(core_id);
        ct.move_idle_to_running(idle_idx);
        core_idle_[core_id] = false;
        executing_reg_task_ids_[core_id] = reg_task_id;
    }
};

static AicpuExecutor g_aicpu_executor;

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
        all_handshakes[i].task = reinterpret_cast<uint64_t>(&s_pto2_payload_per_core[i]);
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

        if (type == CoreType::AIC) {
            aic_cores_[aic_count_].worker_id = i;
            aic_cores_[aic_count_].physical_core_id = physical_core_id;
            aic_cores_[aic_count_].reg_addr = reg_addr;
            aic_cores_[aic_count_].core_type = type;
            aic_count_++;
            DEV_INFO("Core %d: AIC, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        } else {
            aiv_cores_[aiv_count_].worker_id = i;
            aiv_cores_[aiv_count_].physical_core_id = physical_core_id;
            aiv_cores_[aiv_count_].reg_addr = reg_addr;
            aiv_cores_[aiv_count_].core_type = type;
            aiv_count_++;
            DEV_INFO("Core %d: AIV, physical_id=%u, reg_addr=0x%lx", i, physical_core_id, reg_addr);
        }

        core_id_to_reg_addr_[i] = reg_addr;
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
void AicpuExecutor::assign_cores_to_threads() {
    // Cluster-aligned round-robin assignment: cluster ci -> sched thread ci % divisor.
    // Each cluster = 1 AIC + 2 adjacent AIV; the triple is always kept together.
    int32_t divisor = (sched_thread_num_ > 0) ? sched_thread_num_ : thread_num_;
    int32_t cluster_count = aic_count_;

    DEV_INFO(
        "Assigning cores (round-robin): %d clusters across %d sched threads (%d AIC, %d AIV)", cluster_count, divisor,
        aic_count_, aiv_count_
    );

    memset(core_idle_, true, sizeof(core_idle_));
    for (int32_t i = 0; i < MAX_CORES_PER_THREAD; i++) {
        executing_reg_task_ids_[i] = AICPU_TASK_INVALID;
    }
    for (int32_t i = 0; i < thread_num_; i++) {
        trackers_[i].aic().running_count = 0;
        trackers_[i].aiv().running_count = 0;
        trackers_[i].aic().idle_count = 0;
        trackers_[i].aiv().idle_count = 0;
        trackers_[i].cluster_count = 0;
        core_count_per_thread_[i] = 0;
    }

    // Per-sched-thread running core index used while filling core_assignments_.
    int32_t core_idx[MAX_AICPU_THREADS] = {};

    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % divisor;
        CoreStateTracker &tracker = trackers_[t];
        int32_t &idx = core_idx[t];

        int32_t aic_wid = aic_cores_[ci].worker_id;
        int32_t aiv0_wid = aiv_cores_[2 * ci].worker_id;
        int32_t aiv1_wid = aiv_cores_[2 * ci + 1].worker_id;

        tracker.clusters[tracker.cluster_count++] = {aic_wid, {aiv0_wid, aiv1_wid}};

        core_assignments_[t][idx++] = aic_wid;
        tracker.aic().idle[tracker.aic().idle_count++] = aic_wid;

        core_assignments_[t][idx++] = aiv0_wid;
        core_assignments_[t][idx++] = aiv1_wid;
        tracker.aiv().idle[tracker.aiv().idle_count++] = aiv0_wid;
        tracker.aiv().idle[tracker.aiv().idle_count++] = aiv1_wid;

        DEV_INFO("Thread %d: cluster %d (AIC=%d, AIV0=%d, AIV1=%d)", t, ci, aic_wid, aiv0_wid, aiv1_wid);
    }

    for (int32_t t = 0; t < divisor; t++) {
        core_count_per_thread_[t] = core_idx[t];
        DEV_INFO("Thread %d: total %d cores (%d clusters)", t, core_idx[t], trackers_[t].cluster_count);
    }

    // Max clusters any single sched thread can hold: ceil(cluster_count / divisor).
    int32_t max_clusters_per_thread = (cluster_count + divisor - 1) / divisor;
    thread_cores_num_ = max_clusters_per_thread * 3;
}

/**
 * Reassign all cores evenly across all threads (schedulers + orchestrators).
 * Called by the last orchestrator thread when orchestration completes.
 * Writes into new_core_assignments_ / new_core_count_per_thread_.
 */
void AicpuExecutor::reassign_cores_for_all_threads() {
    DEV_INFO("Reassigning cores (cluster-aligned) for %d threads: %d AIC, %d AIV", thread_num_, aic_count_, aiv_count_);

    // Collect running/idle state from all threads before reassignment
    bool running_cores[MAX_CORES_PER_THREAD];
    memset(running_cores, 0, sizeof(running_cores));

    for (int32_t i = 0; i < thread_num_; i++) {
        for (int32_t j = 0; j < trackers_[i].aic().running_count; j++) {
            int32_t core_id = trackers_[i].aic().running[j];
            running_cores[core_id] = true;
        }
        for (int32_t j = 0; j < trackers_[i].aiv().running_count; j++) {
            int32_t core_id = trackers_[i].aiv().running[j];
            running_cores[core_id] = true;
        }
    }

    // Reset all trackers
    for (int32_t i = 0; i < thread_num_; i++) {
        core_count_per_thread_[i] = 0;
        trackers_[i].aic().running_count = 0;
        trackers_[i].aic().idle_count = 0;
        trackers_[i].aiv().running_count = 0;
        trackers_[i].aiv().idle_count = 0;
        trackers_[i].cluster_count = 0;
    }

    // Restore a single core's running/idle state into its new thread's tracker
    auto reassign_core = [&](int32_t worker_id, CoreTypeTracker &type_tracker, int32_t thread_idx) {
        core_assignments_[thread_idx][core_count_per_thread_[thread_idx]++] = worker_id;
        if (running_cores[worker_id]) {
            type_tracker.running[type_tracker.running_count++] = worker_id;
        } else {
            type_tracker.idle[type_tracker.idle_count++] = worker_id;
        }
    };

    // Assign whole clusters round-robin across all threads
    for (int32_t ci = 0; ci < aic_count_; ci++) {
        int32_t t = ci % thread_num_;
        CoreStateTracker &tracker = trackers_[t];

        int32_t aic_wid = aic_cores_[ci].worker_id;
        int32_t aiv0_wid = aiv_cores_[2 * ci].worker_id;
        int32_t aiv1_wid = aiv_cores_[2 * ci + 1].worker_id;

        tracker.clusters[tracker.cluster_count++] = {aic_wid, {aiv0_wid, aiv1_wid}};

        reassign_core(aic_wid, tracker.aic(), t);
        reassign_core(aiv0_wid, tracker.aiv(), t);
        reassign_core(aiv1_wid, tracker.aiv(), t);
    }

    // Log final distribution for verification
    DEV_INFO("Core reassignment complete:");
    for (int32_t t = 0; t < thread_num_; t++) {
        DEV_INFO(
            "  Thread %d: %d cores, %d clusters (AIC: running=%d idle=%d, AIV: running=%d idle=%d)", t,
            core_count_per_thread_[t], trackers_[t].cluster_count, trackers_[t].aic().running_count,
            trackers_[t].aic().idle_count, trackers_[t].aiv().running_count, trackers_[t].aiv().idle_count
        );
    }
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
    if (thread_num_ == 0) thread_num_ = 1;
    sched_thread_num_ = thread_num_ - 1;
    orch_to_sched_ = runtime->orch_to_sched;

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Initialize core_id_to_reg_addr_ array to 0 before handshake
    for (int32_t i = 0; i < MAX_CORES_PER_THREAD; i++) {
        core_id_to_reg_addr_[i] = 0;
    }

    // Use handshake mechanism to discover cores (aligned with host_build_graph)
    int32_t rc = handshake_all_cores(runtime);
    if (rc != 0) {
        DEV_ERROR("handshake_all_cores failed");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Dynamically assign cores to threads
    assign_cores_to_threads();

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

    // Reset per-core dispatch timestamps and task counters
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        dispatch_timestamps_[i] = 0;
        core_dispatch_counts_[i] = 0;
    }

    // Clear per-core dispatch payloads and subslot tracking
    memset(s_pto2_payload_per_core, 0, sizeof(s_pto2_payload_per_core));
    memset(dispatch_seq_by_core_, 0, sizeof(dispatch_seq_by_core_));
    memset(executing_subslot_by_core_, 0, sizeof(executing_subslot_by_core_));
    memset(executing_slot_state_by_core_, 0, sizeof(executing_slot_state_by_core_));

    DEV_INFO("Init: PTO2 mode, task count from shared memory");

    finished_count_.store(0, std::memory_order_release);

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
        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
        if (reg_addr != 0) {
            platform_deinit_aicore_regs(reg_addr);
        } else {
            DEV_ERROR("Thread %d: Core %d has invalid register address", thread_idx, core_id);
        }
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

int32_t AicpuExecutor::resolve_and_dispatch_pto2(Runtime *runtime, int32_t thread_idx) {
    int32_t &core_num = core_count_per_thread_[thread_idx];
    CoreStateTracker &tracker = trackers_[thread_idx];
    DEV_INFO("Thread %d: resolve_and_dispatch_pto2 entry", thread_idx);

    void *sm_base = runtime->get_pto2_gm_sm_ptr();
    if (!sm_base) {
        DEV_ERROR("PTO2 dispatch: sm_base is null");
        return -1;
    }
    DEV_INFO("Thread %d: sm_base=%p", thread_idx, sm_base);

    PTO2SharedMemoryHeader *header = static_cast<PTO2SharedMemoryHeader *>(sm_base);
    DEV_INFO(
        "Thread %d: header=%p, task_desc_offset[0]=%lu, window_size=%lu", thread_idx, static_cast<void *>(header),
        static_cast<uint64_t>(header->rings[0].task_descriptors_offset),
        static_cast<uint64_t>(header->rings[0].task_window_size)
    );

    Handshake *hank = static_cast<Handshake *>(runtime->workers);
    DEV_INFO(
        "Thread %d: hank=%p, window_size=%lu", thread_idx, static_cast<void *>(hank),
        static_cast<uint64_t>(header->rings[0].task_window_size)
    );

    // One-time init: assign perf buffers (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) {
        DEV_INFO("Thread %d: doing one-time init", thread_idx);

#if PTO2_PROFILING
        // Assign perf buffers to cores early so profiling captures all tasks
        // (total_tasks written to header later when orchestrator completes)
        if (runtime->enable_profiling) {
            perf_aicpu_init_profiling(runtime);
            // Initialize phase profiling for scheduler threads + orchestrator threads
            perf_aicpu_init_phase_profiling(runtime, sched_thread_num_);
            perf_aicpu_set_orch_thread_idx(sched_thread_num_);
        }
#endif
#if PTO2_PROFILING
        if (get_enable_dump_tensor()) {
            dump_tensor_init(orch_to_sched_ ? thread_num_ : sched_thread_num_);
        }
#endif

        DEV_INFO("Thread %d: one-time init done", thread_idx);
        pto2_init_complete_.store(true, std::memory_order_release);
    } else {
        while (!pto2_init_complete_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    DEV_INFO("Thread %d: PTO2 dispatch starting with %d cores", thread_idx, core_num);
    int32_t cur_thread_completed = 0;
    int32_t idle_iterations = 0;
    int32_t last_progress_count = 0;
#if PTO2_PROFILING
    bool profiling_enabled = runtime->enable_profiling;
#endif

    // Scheduler profiling counters
#if PTO2_PROFILING
    uint64_t sched_scan_cycle = 0;
    uint64_t sched_complete_cycle = 0;
    uint64_t sched_dispatch_cycle = 0;
    uint64_t sched_idle_cycle = 0;
    uint64_t sched_loop_count = 0;
    uint32_t phase_complete_count = 0;
    uint32_t phase_dispatch_count = 0;
#if PTO2_SCHED_PROFILING
    uint64_t complete_probe_count = 0;
    uint64_t complete_hit_count = 0;
    uint64_t notify_edges_total = 0;
    int32_t notify_max_degree = 0;
    uint64_t notify_tasks_enqueued = 0;
    uint64_t fanin_edges_total = 0;
    int32_t fanin_max_degree = 0;
    uint64_t pop_hit = 0;
    uint64_t pop_miss = 0;
    uint64_t local_dispatch_count = 0;
    uint64_t local_overflow_count = 0;
    uint64_t sched_complete_perf_cycle = 0;
    uint64_t sched_dispatch_pop_cycle = 0;
    uint64_t sched_dispatch_setup_cycle = 0;
#endif
#endif

    // Local-first dispatch buffers (stack-allocated, one per CoreType per scheduling thread).
    // Initialized once; must be empty at the start of each iteration.
    constexpr int LOCAL_READY_CAP_PER_TYPE = 256;
    PTO2TaskSlotState *local_aic_ptrs[LOCAL_READY_CAP_PER_TYPE];
    PTO2TaskSlotState *local_aiv_ptrs[LOCAL_READY_CAP_PER_TYPE];
    PTO2LocalReadyBuffer local_bufs[PTO2_LOCAL_DISPATCH_TYPE_NUM];  // [0]=AIC, [1]=AIV
    local_bufs[0].reset(local_aic_ptrs, LOCAL_READY_CAP_PER_TYPE);
    local_bufs[1].reset(local_aiv_ptrs, LOCAL_READY_CAP_PER_TYPE);
    PTO2TaskSlotState *deferred_release_slot_states[256];
    int32_t deferred_release_count = 0;

    bool cores_released = false;

    while (true) {
        bool made_progress = false;
#if PTO2_PROFILING
        CYCLE_COUNT_START();
        sched_loop_count++;
        uint64_t _t0_phase = _t0;
#endif
        int32_t task_count = 0;
        if (tracker.aic().running_count == 0 && tracker.aiv().running_count == 0) {
            bool orch_done = orchestrator_done_;
            if (orch_done) {
                // Check for orchestrator fatal error — exit immediately
                int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
                if (orch_err != PTO2_ERROR_NONE) {
                    DEV_ERROR(
                        "Thread %d: Fatal error (code=%d), sending EXIT_SIGNAL to all cores. "
                        "completed_tasks=%d, total_tasks=%d",
                        thread_idx, orch_err, completed_tasks_.load(std::memory_order_relaxed), total_tasks_
                    );
                    emergency_shutdown(runtime);
                    completed_.store(true, std::memory_order_release);
                    break;
                }

                // Normal exit: all tasks complete
                task_count = total_tasks_;
                if (task_count > 0 && completed_tasks_.load(std::memory_order_relaxed) >= task_count) {
                    completed_.store(true, std::memory_order_release);
                    DEV_INFO(
                        "Thread %d: PTO2 completed tasks %d/%d", thread_idx,
                        completed_tasks_.load(std::memory_order_relaxed), task_count
                    );
                    break;
                }
            }
        }

        // Check for core transition request (execute once per thread)
        if (!cores_released && orch_to_sched_ && transition_requested_.load(std::memory_order_acquire)) {
            if (!reassigned_.load(std::memory_order_acquire)) {
                wait_reassign_.fetch_add(1, std::memory_order_release);
                while (!reassigned_.load(std::memory_order_acquire)) {
                    if (completed_.load(std::memory_order_acquire)) {
                        break;
                    }
                    SPIN_WAIT_HINT();
                }
                if (completed_.load(std::memory_order_acquire)) {
                    break;
                }
            }
            cores_released = true;
        }

#if PTO2_PROFILING
        CYCLE_COUNT_LAP(sched_idle_cycle);
#endif

        // Process completed and dispatch FIRST to minimize Sched (dispatch→finish) latency.
        // Sched time = finish_ts - dispatch_ts; recording finish_ts here at loop start reduces
        // tail overhead (time from AICore done to AICPU recording finish).

        // Phase 1: Check running cores for completion, process and move to idle
        int32_t completed_this_turn = 0;

        // Check AIC running cores
        bool try_completed = false;
        always_assert(
            local_bufs[0].count == 0 && local_bufs[1].count == 0
        );  // Invariant: previous iteration fully consumed
        if (tracker.aic().running_count > 0) {
            try_completed = true;
            check_running_cores_for_completion<CoreType::AIC>(
                thread_idx, tracker.aic(), hank, completed_this_turn, cur_thread_completed, made_progress,
                deferred_release_slot_states, deferred_release_count, local_bufs
#if PTO2_PROFILING
                ,
                profiling_enabled, phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
                ,
                complete_probe_count, complete_hit_count, notify_edges_total, notify_max_degree, notify_tasks_enqueued,
                fanin_edges_total, fanin_max_degree, sched_complete_perf_cycle
#endif
            );
        }

        // Check AIV running cores
        if (tracker.aiv().running_count > 0) {
            try_completed = true;
            check_running_cores_for_completion<CoreType::AIV>(
                thread_idx, tracker.aiv(), hank, completed_this_turn, cur_thread_completed, made_progress,
                deferred_release_slot_states, deferred_release_count, local_bufs
#if PTO2_PROFILING
                ,
                profiling_enabled, phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
                ,
                complete_probe_count, complete_hit_count, notify_edges_total, notify_max_degree, notify_tasks_enqueued,
                fanin_edges_total, fanin_max_degree, sched_complete_perf_cycle
#endif
            );
        }
        if (completed_this_turn > 0) {
#if PTO2_SCHED_PROFILING
            rt->scheduler.tasks_completed.fetch_add(completed_this_turn, std::memory_order_relaxed);
#endif
            int32_t prev = completed_tasks_.fetch_add(completed_this_turn, std::memory_order_relaxed);
            int32_t new_total = prev + completed_this_turn;
            last_progress_count = new_total;
            if (thread_idx == 0 && task_count > 0) {
                if (new_total <= PROGRESS_VERBOSE_THRESHOLD ||
                    new_total / PROGRESS_LOG_INTERVAL != prev / PROGRESS_LOG_INTERVAL || new_total >= task_count) {
                    DEV_ALWAYS(
                        "PTO2 progress: completed=%d total=%d (%.1f%%)", new_total, task_count,
                        100.0 * new_total / task_count
                    );
                }
            }
        }

#if PTO2_PROFILING
        if (!try_completed) {
            CYCLE_COUNT_LAP(sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(sched_complete_cycle);
            if (profiling_enabled && phase_complete_count > 0) {
                perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_COMPLETE, _t0_phase, _t1, sched_loop_count, phase_complete_count
                );
                _t0_phase = _t1;
                phase_complete_count = 0;
            }
        }
#endif

        // Phase 2: Local dispatch — drain local_bufs, match to idle clusters (zero MPMC operations)
        // Phase 3: Global queue — push overflow to readyQ + fill remaining idle cores from readyQ
        bool try_pushed = false;

        // Local dispatch: drain both per-CoreType local_bufs, match to idle clusters by shape
        PTO2TaskSlotState *overflow_ptrs[LOCAL_READY_CAP_PER_TYPE * PTO2_LOCAL_DISPATCH_TYPE_NUM];
        int overflow_count = 0;
        for (int bi = 0; bi < PTO2_LOCAL_DISPATCH_TYPE_NUM; bi++) {
            while (local_bufs[bi].count > 0) {
                PTO2TaskSlotState *slot_state = local_bufs[bi].pop();
                PTO2ResourceShape shape = pto2_active_mask_to_shape(slot_state->active_mask);
                int32_t ci = tracker.find_cluster_for_shape(shape, core_idle_);

                if (ci >= 0) {
                    try_pushed = true;
                    Cluster &c = tracker.clusters[ci];
#if PTO2_SCHED_PROFILING
                    uint64_t t_setup_start = get_sys_cnt_aicpu();
#endif
                    ResourceCount rc = shape_resource_count(shape);
#if PTO2_PROFILING
                    if (get_enable_dump_tensor()) {
                        dump_tensors_for_task<PTO2_SUBTASK_SLOT_COUNT>(
                            thread_idx, *slot_state, TensorDumpStage::BEFORE_DISPATCH,
                            [](uint8_t active_mask, uint8_t raw_subtask_id) {
                                return pto2_subtask_active(active_mask, static_cast<PTO2SubtaskSlot>(raw_subtask_id));
                            },
                            [this](int32_t func_id) {
                                return get_function_bin_addr(func_id);
                            }
                        );
                    }
#endif
                    if (rc.aic) {
                        dispatch_subtask_to_core(
                            runtime, tracker, c.aic_core_id, CoreType::AIC, *slot_state, PTO2SubtaskSlot::AIC
#if PTO2_PROFILING
                            ,
                            profiling_enabled
#endif
#if PTO2_PROFILING
                            ,
                            thread_idx
#endif
                        );
                    }
                    if (rc.aiv >= 1) {
                        int32_t aiv0 = core_idle_[c.aiv_core_ids[0]] ? c.aiv_core_ids[0] : c.aiv_core_ids[1];
                        dispatch_subtask_to_core(
                            runtime, tracker, aiv0, CoreType::AIV, *slot_state, PTO2SubtaskSlot::AIV0
#if PTO2_PROFILING
                            ,
                            profiling_enabled
#endif
#if PTO2_PROFILING
                            ,
                            thread_idx
#endif
                        );
                    }
                    if (rc.aiv >= 2) {
                        dispatch_subtask_to_core(
                            runtime, tracker, c.aiv_core_ids[1], CoreType::AIV, *slot_state, PTO2SubtaskSlot::AIV1
#if PTO2_PROFILING
                            ,
                            profiling_enabled
#endif
#if PTO2_PROFILING
                            ,
                            thread_idx
#endif
                        );
                    }
#if PTO2_PROFILING
                    phase_dispatch_count++;
#endif
#if PTO2_SCHED_PROFILING
                    pop_hit++;
                    local_dispatch_count++;
                    sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif
                    made_progress = true;
                    DEV_DEBUG(
                        "Thread %d: Dispatching %s task %" PRId64 " to cluster %d (local)", thread_idx,
                        shape_name(shape), static_cast<int64_t>(slot_state->task->task_id.raw), ci
                    );
                } else {
                    overflow_ptrs[overflow_count++] = slot_state;
#if PTO2_SCHED_PROFILING
                    local_overflow_count++;
#endif
                }
            }
        }

        // Push overflow to global readyQ (shape-based)
        for (int i = 0; i < overflow_count; i++) {
            rt->scheduler.requeue_ready_task(*overflow_ptrs[i]);
        }

        // Phase 3: Global dispatch — fill remaining idle cores from global readyQ (cluster-based)
        const PTO2ResourceShape *dispatch_order = get_dispatch_order(thread_idx);

        for (int32_t si = 0; si < PTO2_NUM_RESOURCE_SHAPES; si++) {
            PTO2ResourceShape shape = dispatch_order[si];
            if (rt->scheduler.ready_queues[static_cast<int32_t>(shape)].size() == 0) continue;

            while (true) {
                int32_t ci = tracker.find_cluster_for_shape(shape, core_idle_);
                if (ci < 0) break;

                PTO2TaskSlotState *slot_state = pop_ready_task(
                    shape, thread_idx
#if PTO2_SCHED_PROFILING
                    ,
                    pop_hit, pop_miss, sched_dispatch_pop_cycle
#endif
                );
                if (!slot_state) break;

                try_pushed = true;
#if PTO2_PROFILING
                phase_dispatch_count++;
#endif
#if PTO2_SCHED_PROFILING
                uint64_t t_setup_start = get_sys_cnt_aicpu();
#endif
                Cluster &c = tracker.clusters[ci];
                ResourceCount rc = shape_resource_count(shape);
#if PTO2_PROFILING
                if (get_enable_dump_tensor()) {
                    dump_tensors_for_task<PTO2_SUBTASK_SLOT_COUNT>(
                        thread_idx, *slot_state, TensorDumpStage::BEFORE_DISPATCH,
                        [](uint8_t active_mask, uint8_t raw_subtask_id) {
                            return pto2_subtask_active(active_mask, static_cast<PTO2SubtaskSlot>(raw_subtask_id));
                        },
                        [this](int32_t func_id) {
                            return get_function_bin_addr(func_id);
                        }
                    );
                }
#endif
                if (rc.aic) {
                    dispatch_subtask_to_core(
                        runtime, tracker, c.aic_core_id, CoreType::AIC, *slot_state, PTO2SubtaskSlot::AIC
#if PTO2_PROFILING
                        ,
                        profiling_enabled
#endif
#if PTO2_PROFILING
                        ,
                        thread_idx
#endif
                    );
                }
                if (rc.aiv >= 1) {
                    int32_t aiv_id = core_idle_[c.aiv_core_ids[0]] ? c.aiv_core_ids[0] : c.aiv_core_ids[1];
                    dispatch_subtask_to_core(
                        runtime, tracker, aiv_id, CoreType::AIV, *slot_state, PTO2SubtaskSlot::AIV0
#if PTO2_PROFILING
                        ,
                        profiling_enabled
#endif
#if PTO2_PROFILING
                        ,
                        thread_idx
#endif
                    );
                }
                if (rc.aiv >= 2) {
                    dispatch_subtask_to_core(
                        runtime, tracker, c.aiv_core_ids[1], CoreType::AIV, *slot_state, PTO2SubtaskSlot::AIV1
#if PTO2_PROFILING
                        ,
                        profiling_enabled
#endif
#if PTO2_PROFILING
                        ,
                        thread_idx
#endif
                    );
                }
                made_progress = true;
#if PTO2_SCHED_PROFILING
                sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif
                DEV_DEBUG(
                    "Thread %d: Dispatching %s task %" PRId64 " to cluster %d", thread_idx, shape_name(shape),
                    static_cast<int64_t>(slot_state->task->task_id.raw), ci
                );
            }
        }

#if PTO2_PROFILING
        if (!try_pushed) {
            CYCLE_COUNT_LAP(sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(sched_dispatch_cycle);
            if (profiling_enabled && phase_dispatch_count > 0) {
                perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_DISPATCH, _t0_phase, _t1, sched_loop_count, phase_dispatch_count
                );
                _t0_phase = _t1;
                phase_dispatch_count = 0;
            }
#endif
        }

        if (made_progress) {
            idle_iterations = 0;
        } else {
            // Batch deferred fanin releases during idle.
            // Processing all pending releases at once advances the ring faster,
            // freeing heap space for the orchestrator without blocking completion polling.
            while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                int32_t fe =
                    rt->scheduler.on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
            int32_t fe = rt->scheduler.on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
                (void)fe;
#if PTO2_SCHED_PROFILING
                fanin_edges_total += fe;
                if (fe > fanin_max_degree) fanin_max_degree = fe;
#endif
            }
            idle_iterations++;

            // Check for orchestrator fatal error during idle (every 1024 iterations)
            // orch_error_code is set in shared memory by the orchestrator's spin loop
            // BEFORE orchestrator_done_ is set, so this catches errors earlier.
            if (idle_iterations % FATAL_ERROR_CHECK_INTERVAL == 0) {
                int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
                if (orch_err != PTO2_ERROR_NONE) {
                    DEV_ERROR(
                        "Thread %d: Fatal error detected (code=%d), sending EXIT_SIGNAL to all cores", thread_idx,
                        orch_err
                    );
                    emergency_shutdown(runtime);
                    completed_.store(true, std::memory_order_release);
                    break;
                }
            }

            if (thread_idx == 0 && task_count > 0 && idle_iterations % STALL_LOG_INTERVAL == 0) {
                int32_t c = completed_tasks_.load(std::memory_order_relaxed);
                DEV_ALWAYS(
                    "PTO2 stall: no progress for %d iterations, completed=%d total=%d (last progress at %d)",
                    idle_iterations, c, task_count, last_progress_count
                );
                // Scan all task slots to find truly stuck tasks using scheduler state
                PTO2SchedulerState *sched = &rt->scheduler;
                PTO2SharedMemoryHeader *sm_header_diag = static_cast<PTO2SharedMemoryHeader *>(sm_base);
                int32_t cnt_ready = 0, cnt_waiting = 0, cnt_inflight = 0;
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                    int32_t ring_task_count =
                        sm_header_diag->rings[r].fc.current_task_index.load(std::memory_order_relaxed);
                    for (int32_t si = 0; si < ring_task_count; si++) {
                        PTO2TaskSlotState &slot_state = sched->get_slot_state(r, si);
                        PTO2TaskState st = slot_state.task_state.load(std::memory_order_relaxed);
                        int32_t rc = slot_state.fanin_refcount.load(std::memory_order_relaxed);
                        int32_t fi = slot_state.fanin_count;
                        int32_t kid = slot_state.task->kernel_id[0];
                        if (st >= PTO2_TASK_COMPLETED) continue;  // Already done
                        if (st == PTO2_TASK_READY || st == PTO2_TASK_RUNNING) {
                            cnt_inflight++;
                            continue;
                        }
                        // PENDING
                        if (rc >= fi) {
                            // Ready (all deps satisfied) but not enqueued — this is the real bug
                            cnt_ready++;
                            if (cnt_ready <= STALL_DUMP_READY_MAX) {
                                DEV_ALWAYS(
                                    "  STUCK-READY  ring=%d task_id=%" PRId64
                                    " kernel_id=%d refcount=%d fanin=%d state=%d",
                                    r, static_cast<int64_t>(slot_state.task->task_id.raw), kid, rc, fi,
                                    static_cast<int32_t>(st)
                                );
                            }
                        } else {
                            cnt_waiting++;
                            if (cnt_waiting <= STALL_DUMP_WAIT_MAX) {
                                DEV_ALWAYS(
                                    "  STUCK-WAIT   ring=%d task_id=%" PRId64
                                    " kernel_id=%d refcount=%d fanin=%d state=%d",
                                    r, static_cast<int64_t>(slot_state.task->task_id.raw), kid, rc, fi,
                                    static_cast<int32_t>(st)
                                );
                            }
                        }
                    }
                }
                DEV_ALWAYS(
                    "  scan result: stuck_ready=%d stuck_waiting=%d in_flight=%d", cnt_ready, cnt_waiting, cnt_inflight
                );
                // Log this thread's dispatch state
                int32_t total_idle = tracker.aic().idle_count + tracker.aiv().idle_count;
                int32_t total_running = tracker.aic().running_count + tracker.aiv().running_count;
                DEV_ALWAYS(
                    "  thread=%d idle_cores=%d (AIC=%d AIV=%d) running_cores=%d (AIC=%d AIV=%d) core_num=%d",
                    thread_idx, total_idle, tracker.aic().idle_count, tracker.aiv().idle_count, total_running,
                    tracker.aic().running_count, tracker.aiv().running_count, core_num
                );
                // Dump AIC running cores
                for (int32_t ci = 0; ci < tracker.aic().running_count && ci < STALL_DUMP_CORE_MAX; ci++) {
                    int32_t cid = tracker.aic().running[ci];
                    int32_t sw_tid = executing_reg_task_ids_[cid];
                    int32_t hw_kernel = -1;
                    if (sw_tid >= 0 && executing_slot_state_by_core_[cid]) {
                        int32_t diag_slot = static_cast<int32_t>(executing_subslot_by_core_[cid]);
                        hw_kernel = executing_slot_state_by_core_[cid]->task->kernel_id[diag_slot];
                    }
                    uint64_t cond_reg = read_reg(core_id_to_reg_addr_[cid], RegId::COND);
                    DEV_ALWAYS(
                        "    core=%d cond=0x%x(state=%d,id=%d) exec_id=%d kernel=%d", cid,
                        static_cast<unsigned>(cond_reg), EXTRACT_TASK_STATE(cond_reg), EXTRACT_TASK_ID(cond_reg),
                        sw_tid, hw_kernel
                    );
                }
                // Dump AIV running cores
                for (int32_t ci = 0; ci < tracker.aiv().running_count && ci < STALL_DUMP_CORE_MAX; ci++) {
                    int32_t cid = tracker.aiv().running[ci];
                    int32_t sw_tid = executing_reg_task_ids_[cid];
                    int32_t hw_kernel = -1;
                    if (sw_tid >= 0 && executing_slot_state_by_core_[cid]) {
                        int32_t diag_slot = static_cast<int32_t>(executing_subslot_by_core_[cid]);
                        hw_kernel = executing_slot_state_by_core_[cid]->task->kernel_id[diag_slot];
                    }
                    uint64_t cond_reg = read_reg(core_id_to_reg_addr_[cid], RegId::COND);
                    DEV_ALWAYS(
                        "    core=%d cond=0x%x(state=%d,id=%d) exec_id=%d kernel=%d", cid,
                        static_cast<unsigned>(cond_reg), EXTRACT_TASK_STATE(cond_reg), EXTRACT_TASK_ID(cond_reg),
                        sw_tid, hw_kernel
                    );
                }
                // Dump cluster state
                for (int32_t cli = 0; cli < tracker.cluster_count && cli < STALL_DUMP_CORE_MAX; cli++) {
                    Cluster &cl = tracker.clusters[cli];
                    DEV_ALWAYS(
                        "    cluster[%d] aic=%d(%s) aiv0=%d(%s) aiv1=%d(%s)", cli, cl.aic_core_id,
                        core_idle_[cl.aic_core_id] ? "idle" : "busy", cl.aiv_core_ids[0],
                        core_idle_[cl.aiv_core_ids[0]] ? "idle" : "busy", cl.aiv_core_ids[1],
                        core_idle_[cl.aiv_core_ids[1]] ? "idle" : "busy"
                    );
                }
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("Thread %d: PTO2 timeout after %d idle iterations", thread_idx, idle_iterations);
                return -1;
            } else {
                SPIN_WAIT_HINT();
            }
#if PTO2_PROFILING
            CYCLE_COUNT_LAP(sched_idle_cycle);
            if (profiling_enabled) {
                perf_aicpu_record_phase(thread_idx, AicpuPhaseId::SCHED_IDLE_WAIT, _t0_phase, _t1, sched_loop_count, 0);
                _t0_phase = _t1;
            }
#endif
        }
    }

#if PTO2_PROFILING
    // Scheduler summary logging (always print when PTO2_PROFILING=1)
    uint64_t sched_total = sched_complete_cycle + sched_scan_cycle + sched_dispatch_cycle + sched_idle_cycle;
    if (sched_total == 0) sched_total = 1;  // avoid div-by-zero

#if PTO2_SCHED_PROFILING
    // Two-level tree display: sub-phase breakdown within complete and dispatch
    {
        PTO2SchedProfilingData sp = pto2_scheduler_get_profiling(thread_idx);
        uint64_t otc_total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
        uint64_t complete_poll = (sched_complete_cycle > otc_total + sched_complete_perf_cycle) ?
                                     (sched_complete_cycle - otc_total - sched_complete_perf_cycle) :
                                     0;
        uint64_t dispatch_poll = (sched_dispatch_cycle > sched_dispatch_pop_cycle + sched_dispatch_setup_cycle) ?
                                     (sched_dispatch_cycle - sched_dispatch_pop_cycle - sched_dispatch_setup_cycle) :
                                     0;

        DEV_ALWAYS(
            "Thread %d: === Scheduler Phase Breakdown: total=%.3fus, %d tasks ===", thread_idx,
            cycles_to_us(sched_total), cur_thread_completed
        );

        // Level 1: complete
        double notify_avg =
            cur_thread_completed > 0 ? static_cast<double>(notify_edges_total) / cur_thread_completed : 0.0;
        double fanin_avg =
            cur_thread_completed > 0 ? static_cast<double>(fanin_edges_total) / cur_thread_completed : 0.0;
        DEV_ALWAYS(
            "Thread %d:   complete       : %.3fus (%.1f%%)  [fanout: edges=%" PRIu64
            ", max_degree=%d, avg=%.1f]  [fanin: "
            "edges=%" PRIu64 ", max_degree=%d, avg=%.1f]",
            thread_idx, cycles_to_us(sched_complete_cycle), sched_complete_cycle * 100.0 / sched_total,
            static_cast<uint64_t>(notify_edges_total), notify_max_degree, notify_avg,
            static_cast<uint64_t>(fanin_edges_total), fanin_max_degree, fanin_avg
        );

        // Level 2: complete sub-phases (percentage relative to complete)
        uint64_t c_parent = sched_complete_cycle > 0 ? sched_complete_cycle : 1;
        uint64_t complete_miss_count =
            (complete_probe_count > complete_hit_count) ? (complete_probe_count - complete_hit_count) : 0;
        double complete_hit_rate = complete_probe_count > 0 ? complete_hit_count * 100.0 / complete_probe_count : 0.0;
        DEV_ALWAYS(
            "Thread %d:     poll         : %.3fus (%.1f%%)  hit=%" PRIu64 ", miss=%" PRIu64 ", hit_rate=%.1f%%",
            thread_idx, cycles_to_us(complete_poll), complete_poll * 100.0 / c_parent,
            static_cast<uint64_t>(complete_hit_count), static_cast<uint64_t>(complete_miss_count), complete_hit_rate
        );
        DEV_ALWAYS(
            "Thread %d:     otc_lock     : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.lock_cycle), sp.lock_cycle * 100.0 / c_parent,
            cycles_to_us(sp.lock_cycle - sp.lock_wait_cycle), cycles_to_us(sp.lock_wait_cycle),
            static_cast<uint64_t>(sp.lock_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     otc_fanout   : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.fanout_cycle), sp.fanout_cycle * 100.0 / c_parent,
            cycles_to_us(sp.fanout_cycle - sp.push_wait_cycle), cycles_to_us(sp.push_wait_cycle),
            static_cast<uint64_t>(sp.fanout_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     otc_fanin    : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.fanin_cycle), sp.fanin_cycle * 100.0 / c_parent,
            static_cast<uint64_t>(sp.fanin_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     otc_self     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sp.self_consumed_cycle), sp.self_consumed_cycle * 100.0 / c_parent,
            static_cast<uint64_t>(sp.self_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     perf         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_complete_perf_cycle),
            sched_complete_perf_cycle * 100.0 / c_parent
        );

        // Level 1: dispatch
        uint64_t pop_total = pop_hit + pop_miss;
        double pop_hit_rate = pop_total > 0 ? pop_hit * 100.0 / pop_total : 0.0;
        DEV_ALWAYS(
            "Thread %d:   dispatch       : %.3fus (%.1f%%)  [pop: hit=%" PRIu64 ", miss=%" PRIu64 ", hit_rate=%.1f%%]",
            thread_idx, cycles_to_us(sched_dispatch_cycle), sched_dispatch_cycle * 100.0 / sched_total,
            static_cast<uint64_t>(pop_hit), static_cast<uint64_t>(pop_miss), pop_hit_rate
        );
        uint64_t global_dispatch_count = pop_hit - local_dispatch_count;
        uint64_t total_dispatched = local_dispatch_count + global_dispatch_count;
        double local_hit_rate = total_dispatched > 0 ? local_dispatch_count * 100.0 / total_dispatched : 0.0;
        DEV_ALWAYS(
            "Thread %d:     local_disp   : local=%" PRIu64 ", global=%" PRIu64 ", overflow=%" PRIu64
            ", local_rate=%.1f%%",
            thread_idx, static_cast<uint64_t>(local_dispatch_count), static_cast<uint64_t>(global_dispatch_count),
            static_cast<uint64_t>(local_overflow_count), local_hit_rate
        );

        // Level 2: dispatch sub-phases (percentage relative to dispatch)
        uint64_t d_parent = sched_dispatch_cycle > 0 ? sched_dispatch_cycle : 1;
        DEV_ALWAYS(
            "Thread %d:     poll         : %.3fus (%.1f%%)", thread_idx, cycles_to_us(dispatch_poll),
            dispatch_poll * 100.0 / d_parent
        );
        DEV_ALWAYS(
            "Thread %d:     pop          : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "", thread_idx,
            cycles_to_us(sched_dispatch_pop_cycle), sched_dispatch_pop_cycle * 100.0 / d_parent,
            cycles_to_us(sched_dispatch_pop_cycle - sp.pop_wait_cycle), cycles_to_us(sp.pop_wait_cycle),
            static_cast<uint64_t>(sp.pop_atomic_count)
        );
        DEV_ALWAYS(
            "Thread %d:     setup        : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_dispatch_setup_cycle),
            sched_dispatch_setup_cycle * 100.0 / d_parent
        );

        // Level 1: scan
        DEV_ALWAYS(
            "Thread %d:   scan           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_scan_cycle),
            sched_scan_cycle * 100.0 / sched_total
        );

        // Level 1: idle
        DEV_ALWAYS(
            "Thread %d:   idle           : %.3fus (%.1f%%)", thread_idx, cycles_to_us(sched_idle_cycle),
            sched_idle_cycle * 100.0 / sched_total
        );

        // Average per completion
        if (cur_thread_completed > 0) {
            DEV_ALWAYS(
                "Thread %d:   avg/complete   : %.3fus", thread_idx,
                cycles_to_us(sched_complete_cycle) / cur_thread_completed
            );
        }
    }
#endif
    // Summary line (always print when PTO2_PROFILING=1)
    DEV_ALWAYS(
        "Thread %d: Scheduler summary: total_time=%.3fus, loops=%" PRIu64 ", tasks_scheduled=%d", thread_idx,
        cycles_to_us(sched_total), static_cast<uint64_t>(sched_loop_count), cur_thread_completed
    );
#endif

#if PTO2_PROFILING
    // Flush performance buffers for cores managed by this thread
    if (profiling_enabled) {
        perf_aicpu_flush_buffers(runtime, thread_idx, core_assignments_[thread_idx], core_num);
        perf_aicpu_flush_phase_buffers(thread_idx);
    }
#endif
#if PTO2_PROFILING
    if (get_enable_dump_tensor()) {
        dump_tensor_flush(thread_idx);
    }
#endif

    return cur_thread_completed;
}

int32_t AicpuExecutor::run(Runtime *runtime) {
    int32_t thread_idx = thread_idx_++;

    DEV_ALWAYS("Thread %d: Start", thread_idx);

    // Orchestrator check
    if (thread_idx >= sched_thread_num_) {
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

            dlerror();
            auto config_func =
                reinterpret_cast<DeviceOrchestrationConfigFunc>(dlsym(handle, "aicpu_orchestration_config"));

            dlerror();
            DeviceOrchestrationFunc orch_func =
                reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, "aicpu_orchestration_entry"));
            const char *dlsym_error = dlerror();
            if (dlsym_error != nullptr) {
                DEV_ERROR("Thread %d: dlsym failed: %s", thread_idx, dlsym_error);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }
            if (orch_func == nullptr) {
                DEV_ERROR("Thread %d: dlsym returned NULL for aicpu_orchestration_entry", thread_idx);
                dlclose(handle);
                unlink(so_path);
                return -1;
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

            // With multi-ring, slot_states are per-ring inside the scheduler.
            runtime->set_pto2_slot_states_ptr(nullptr);

            // Store shared state for orchestrator thread
            orch_func_ = orch_func;
            orch_args_cached_ = &args;
            orch_so_handle_ = handle;
            snprintf(orch_so_path_, sizeof(orch_so_path_), "%s", so_path);

            runtime_init_ready_.store(true, std::memory_order_release);

            // Wait for scheduler's one-time init to complete
            while (!pto2_init_complete_.load(std::memory_order_acquire)) {
                SPIN_WAIT_HINT();
            }

#if PTO2_PROFILING
            // Each orchestrator thread sets its own phase buffer index (thread-local)
            if (runtime->enable_profiling) {
                perf_aicpu_set_orch_thread_idx(thread_idx);
            }
#endif

            // Call orchestration function wrapped in an outer scope
            DEV_ALWAYS("Thread %d: Calling aicpu_orchestration_entry from SO", thread_idx);
#if PTO2_PROFILING
            uint64_t orch_cycle_start = get_sys_cnt_aicpu();
#endif
            PTO2_SCOPE(rt) { orch_func_(rt, *orch_args_cached_); }
#if PTO2_PROFILING
            uint64_t orch_cycle_end = get_sys_cnt_aicpu();
            DEV_ALWAYS(
                "Thread %d: orch_start=%" PRIu64 " orch_func_cost=%.3fus", thread_idx,
                static_cast<uint64_t>(orch_cycle_start), cycles_to_us(orch_cycle_end - orch_cycle_start)
            );
#endif

            // Print orchestrator profiling data
#if PTO2_ORCH_PROFILING
            PTO2OrchProfilingData p = pto2_orchestrator_get_profiling();
            uint64_t total = p.alloc_cycle + p.args_cycle + p.heap_cycle + p.fanin_cycle;
            if (total == 0) total = 1;  // avoid div-by-zero
            DEV_ALWAYS(
                "Thread %d: === Orchestrator Profiling: %" PRId64 " tasks, total=%.3fus ===", thread_idx,
                static_cast<int64_t>(p.submit_count), cycles_to_us(total)
            );
            DEV_ALWAYS(
                "Thread %d:   task_ring_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
                cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
                static_cast<uint64_t>(p.alloc_atomic_count)
            );
            DEV_ALWAYS(
                "Thread %d:   heap_alloc     : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                thread_idx, cycles_to_us(p.heap_cycle), p.heap_cycle * 100.0 / total,
                cycles_to_us(p.heap_cycle - p.heap_wait_cycle), cycles_to_us(p.heap_wait_cycle),
                static_cast<uint64_t>(p.heap_atomic_count)
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

#if PTO2_PROFILING
            // Write orchestrator summary to shared memory for host-side export (only if profiling enabled)
            if (runtime->enable_profiling) {
                AicpuOrchSummary orch_summary = {};
                orch_summary.start_time = orch_cycle_start;
                orch_summary.end_time = orch_cycle_end;
                orch_summary.sync_cycle = 0;
                orch_summary.alloc_cycle = p.alloc_cycle;
                orch_summary.args_cycle = p.args_cycle;
                orch_summary.lookup_cycle = 0;
                orch_summary.heap_cycle = p.heap_cycle;
                orch_summary.insert_cycle = 0;
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
                // Flush orchestrator's phase record buffer
                perf_aicpu_flush_phase_buffers(thread_idx);
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
            DEV_ALWAYS(
                "PTO2 total submitted tasks = %d, already executed %d tasks", pto2_task_count,
                completed_tasks_.load(std::memory_order_acquire)
            );
#endif
            total_tasks_ = pto2_task_count;
            if (runtime->enable_profiling && pto2_task_count > 0) {
                perf_aicpu_update_total_tasks(runtime, static_cast<uint32_t>(pto2_task_count));
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
                if (sched_thread_num_ > 0) {
                    while (wait_reassign_.load(std::memory_order_acquire) != sched_thread_num_) {
                        if (completed_.load(std::memory_order_acquire)) {
                            break;
                        }
                        SPIN_WAIT_HINT();
                    }
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
        DEV_INFO("Thread %d: Orchestrator completed", thread_idx);
    }

    // Scheduler thread (orchestrator threads skip dispatch when orch_to_sched_ is false)
    if (!completed_.load(std::memory_order_acquire) && (thread_idx < sched_thread_num_ || orch_to_sched_)) {
        DEV_ALWAYS("Thread %d: Starting PTO2 dispatch", thread_idx);
        // Device orchestration: wait for primary orchestrator to initialize SM header
        if (!runtime->get_orch_built_on_host()) {
            while (!runtime_init_ready_.load(std::memory_order_acquire)) {
                SPIN_WAIT_HINT();
            }
        }
        always_assert(rt != nullptr);
        int32_t completed = resolve_and_dispatch_pto2(runtime, thread_idx);
        DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
    }

    // Always shutdown AICore — even if completed_ was already true.
    // platform_deinit_aicore_regs is idempotent; orchestrator threads have
    // core_count_per_thread_ == 0 so they skip the loop harmlessly.
    {
        const int32_t *shutdown_cores = core_assignments_[thread_idx];
        int32_t shutdown_count = core_count_per_thread_[thread_idx];
#if PTO2_PROFILING
        if (shutdown_count > 0) {
            uint64_t sched_end_ts = get_sys_cnt_aicpu();
            DEV_ALWAYS("Thread %d: sched_end=%" PRIu64 "", thread_idx, static_cast<uint64_t>(sched_end_ts));
        }
#endif
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
            pto2_runtime_destroy(rt);
        }
        DEV_ALWAYS("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return 0;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset per-core dispatch timestamps and task counters
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        dispatch_timestamps_[i] = 0;
        core_dispatch_counts_[i] = 0;
    }

    // Clear per-core dispatch payloads and subslot tracking
    memset(s_pto2_payload_per_core, 0, sizeof(s_pto2_payload_per_core));
    memset(dispatch_seq_by_core_, 0, sizeof(dispatch_seq_by_core_));
    memset(executing_subslot_by_core_, 0, sizeof(executing_subslot_by_core_));
    memset(executing_slot_state_by_core_, 0, sizeof(executing_slot_state_by_core_));

    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_ = 0;
    finished_count_.store(0, std::memory_order_release);
    orchestrator_done_ = false;
    pto2_init_done_.store(false, std::memory_order_release);
    pto2_init_complete_.store(false, std::memory_order_release);
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
    memset(trackers_, 0, sizeof(trackers_));
    memset(core_idle_, 0, sizeof(core_idle_));
    memset(core_assignments_, 0, sizeof(core_assignments_));
    memset(core_count_per_thread_, 0, sizeof(core_count_per_thread_));

    // Reset orchestration SO state (handle freed by last thread before deinit)
    orch_func_ = nullptr;
    orch_args_cached_ = nullptr;
    if (orch_so_handle_ != nullptr) {
        dlclose(orch_so_handle_);
    }
    if (orch_so_path_[0] != '\0') {
        unlink(orch_so_path_);
    }
    orch_so_handle_ = nullptr;
    orch_so_path_[0] = '\0';

    // Reset register-related state
    for (int32_t i = 0; i < MAX_CORES_PER_THREAD; i++) {
        core_id_to_reg_addr_[i] = 0;
        executing_reg_task_ids_[i] = AICPU_TASK_INVALID;
    }
    regs_ = 0;

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
        if (core_id_to_reg_addr_[i] != 0) {
            platform_deinit_aicore_regs(core_id_to_reg_addr_[i]);
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

    uint64_t aic_ready = 0, aiv_ready = 0, aiv_x2_ready = 0, mixed_x1_ready = 0, mixed_x2_ready = 0;
    if (rt) {
        aic_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIC_ONLY)].size();
        aiv_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIV_X1)].size();
        aiv_x2_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIV_X2)].size();
        mixed_x1_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIC_AIV_X1)].size();
        mixed_x2_ready = sched->ready_queues[static_cast<int32_t>(PTO2ResourceShape::AIC_AIV_X2)].size();
    }
    DEV_ALWAYS(
        "Ready Queues: AIC=%lu, AIV=%lu, AIV_X2=%lu, AIC_AIV_X1=%lu, AIC_AIV_X2=%lu", aic_ready, aiv_ready,
        aiv_x2_ready, mixed_x1_ready, mixed_x2_ready
    );

    int32_t busy_cores = 0;
    int32_t idle_cores = 0;

    DEV_ALWAYS("Core Status:");
    for (int32_t i = 0; i < core_num; i++) {
        int32_t core_id = cur_thread_cores[i];
        Handshake *h = &hank[core_id];
        const char *core_type_str = core_type_to_string(h->core_type);

        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
        uint64_t reg_val = read_reg(reg_addr, RegId::COND);
        int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
        int32_t reg_state = EXTRACT_TASK_STATE(reg_val);
        int32_t task_id = executing_reg_task_ids_[core_id];

        if (reg_state != TASK_FIN_STATE || task_id >= 0) {
            busy_cores++;
            if (task_id >= 0) {
                int32_t kernel_id = -1;
                if (rt && rt->sm_handle && executing_slot_state_by_core_[core_id]) {
                    int32_t diag_slot = static_cast<int32_t>(executing_subslot_by_core_[core_id]);
                    kernel_id = executing_slot_state_by_core_[core_id]->task->kernel_id[diag_slot];
                }
                DEV_ALWAYS(
                    "  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s), executing_reg_task_id=%d, "
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

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    DEV_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
