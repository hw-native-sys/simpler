#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "spin_hint.h"
#include "runtime.h"
#include "pto2_dispatch_payload.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "pto_runtime2_types.h"

// Performance profiling headers
#include "aicpu/performance_collector_aicpu.h"
#include "common/perf_profiling.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"

// Register-based communication
#include "common/platform_config.h"
#include "aicpu/platform_regs.h"

// Core type definitions
#include "common/core_type.h"

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
// The executor binds the current thread's PTO2Runtime into orchestration TLS
// before calling the user entry.
typedef void (*DeviceOrchestrationFunc)(TaskArg* orch_args, int32_t orch_thread_num, int32_t orch_thread_index);
typedef void (*DeviceOrchestrationBindRuntimeFunc)(PTO2Runtime* rt);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(TaskArg* orch_args);

constexpr int32_t MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int32_t MAX_AIC_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD;
constexpr int32_t MAX_AIV_PER_THREAD = PLATFORM_MAX_AIV_PER_THREAD;
constexpr int32_t MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;

constexpr int32_t MAX_IDLE_ITERATIONS = 800000;       // ~20s idle then scheduler gives up (avoid long hang)
constexpr int32_t STALL_LOG_INTERVAL = 50000;         // DEV_ALWAYS every N idle iters to debug hang
constexpr int32_t FATAL_ERROR_CHECK_INTERVAL = 1024;  // Check orchestrator error every N idle iters
constexpr int32_t STALL_DUMP_READY_MAX = 8;
constexpr int32_t STALL_DUMP_WAIT_MAX = 4;
constexpr int32_t STALL_DUMP_CORE_MAX = 8;
constexpr int32_t PROGRESS_VERBOSE_THRESHOLD = 10;  // log every completion for the first N tasks
constexpr int32_t PROGRESS_LOG_INTERVAL = 250;      // log every N completions after threshold

static PTO2Runtime* rt{nullptr};

// Per-core dispatch payload storage (one aligned cache line per physical core)
static PTO2DispatchPayload s_pto2_payload_per_core[RUNTIME_MAX_WORKER];

// Per-core state: one cache line per core to eliminate false sharing
// and co-locate all hot-path fields for minimal cache misses.
struct alignas(64) CoreExecState {
    // --- Hot fields (completion + dispatch, every iteration) ---
    uint64_t reg_addr;                        // offset  0: register address (set once in handshake)
    PTO2TaskSlotState* executing_slot_state;  // offset  8: slot state for running task
    int32_t executing_reg_task_id;            // offset 16: register task ID (AICPU_TASK_INVALID = idle)
    uint32_t dispatch_seq;                    // offset 20: monotonic dispatch counter
    PTO2SubtaskSlot executing_subslot;        // offset 24: which subtask slot is running
    uint8_t pad_[3];                          // offset 25: alignment padding
#if PTO2_PROFILING
    // --- Profiling fields (dispatch path, compile-time gated) ---
    uint32_t dispatch_count;      // offset 28: dispatched task count (buffer mgmt)
    uint64_t dispatch_timestamp;  // offset 32: AICPU dispatch timestamp
#endif
    // --- Cold fields (init/diagnostics only, never in hot path) ---
    int32_t worker_id;          // index in runtime.workers[]
    uint32_t physical_core_id;  // hardware physical core ID
    CoreType core_type;         // AIC or AIV
};
static_assert(sizeof(CoreExecState) == 64, "CoreExecState must occupy exactly one cache line");

// core_states_ encodes per-cluster core idle/running in 3 bits per cluster:
//   bit i*3   = AIC of cluster i   (1 = idle, 0 = running)
//   bit i*3+1 = AIV0 of cluster i
//   bit i*3+2 = AIV1 of cluster i
// Max 21 clusters per tracker (63 bits in uint64_t).
class alignas(64) CoreTracker {
public:
    static inline int32_t MAX_CORE_PER_THREAD = 63;
    static constexpr int32_t MAX_CLUSTERS = 63 / 3;

public:
    CoreTracker() = default;

    class BitStates {
    public:
        BitStates() = default;

        explicit BitStates(uint64_t states) : states_(states) {}
        void init() { states_ = 0; }

        BitStates operator~() const { return BitStates(~states_); }
        BitStates operator&(const BitStates& other) const { return BitStates(states_ & other.states_); }
        BitStates operator|(const BitStates& other) const { return BitStates(states_ | other.states_); }
        BitStates operator^(const BitStates& other) const { return BitStates(states_ ^ other.states_); }
        BitStates operator>>(int32_t offset) const { return BitStates(states_ >> offset); }
        BitStates operator<<(int32_t offset) const { return BitStates(states_ << offset); }
        void operator&=(const BitStates& other) { states_ &= other.states_; }
        void operator|=(const BitStates& other) { states_ |= other.states_; }
        void operator^=(const BitStates& other) { states_ ^= other.states_; }

        bool has_value() const { return states_ > 0; }
        int32_t count() const { return __builtin_popcountll(states_); }

        // Extract the lowest set bit from mask, clear it, and return its position.
        // Returns -1 if mask is empty.
        int32_t pop_first() {
            if (states_ == 0) return -1;
            int32_t pos = __builtin_ctzll(states_);
            states_ &= states_ - 1;
            return pos;
        }

    private:
        uint64_t states_{0};
    };

public:
    void init(int32_t cluster_count) {
        cluster_count_ = cluster_count;
        aic_mask_.init();
        aiv_mask_.init();
        for (int32_t i = 0; i < cluster_count; i++) {
            aic_mask_ |= BitStates(1ULL << (i * 3));
            aiv_mask_ |= BitStates(6ULL << (i * 3));
        }
        core_states_ = aic_mask_ | aiv_mask_;
    }

    void set_cluster(int32_t cluster_idx, int32_t aic_wid, int32_t aiv0_wid, int32_t aiv1_wid) {
        core_id_map_[cluster_idx * 3] = aic_wid;
        core_id_map_[cluster_idx * 3 + 1] = aiv0_wid;
        core_id_map_[cluster_idx * 3 + 2] = aiv1_wid;
    }

    int32_t get_cluster_count() const { return cluster_count_; }

    // --- Running core queries ---

    template <CoreType CT>
    bool has_running_cores() const {
        if constexpr (CT == CoreType::AIC) {
            return ((~core_states_) & aic_mask_).has_value();
        } else {
            return ((~core_states_) & aiv_mask_).has_value();
        }
    }

    bool has_any_running_cores() const { return ((~core_states_) & (aic_mask_ | aiv_mask_)).has_value(); }

    template <CoreType CT>
    int32_t get_running_count() const {
        if constexpr (CT == CoreType::AIC) {
            return ((~core_states_) & aic_mask_).count();
        } else {
            return ((~core_states_) & aiv_mask_).count();
        }
    }

    // Return an opaque bitmask for iterating running cores of a given type.
    // Use pop_first() to extract core bit offsets one at a time.
    template <CoreType CT>
    BitStates get_running_cores() const {
        if constexpr (CT == CoreType::AIC) {
            return (~core_states_) & aic_mask_;
        } else {
            return (~core_states_) & aiv_mask_;
        }
    }

    BitStates get_all_running_cores() const { return (~core_states_) & (aic_mask_ | aiv_mask_); }

    // --- Cluster matching ---

    BitStates get_valid_cluster_offset_states(PTO2ResourceShape shape) const {
        switch (shape) {
            case PTO2ResourceShape::AIC:
                return core_states_ & aic_mask_;
            case PTO2ResourceShape::AIV:
                return ((core_states_ >> 1) | (core_states_ >> 2)) & aic_mask_;
            case PTO2ResourceShape::MIX:
                return (core_states_ >> 1) & (core_states_ >> 2) & core_states_ & aic_mask_;
        }
        return BitStates(0ULL);
    }

    int32_t get_aic_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset]; }
    int32_t get_aiv0_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset + 1]; }
    int32_t get_aiv1_core_id(int32_t cluster_offset) const { return core_id_map_[cluster_offset + 2]; }

    int32_t get_aic_core_offset(int32_t cluster_offset) const { return cluster_offset; }
    int32_t get_aiv0_core_offset(int32_t cluster_offset) const { return cluster_offset + 1; }
    int32_t get_aiv1_core_offset(int32_t cluster_offset) const { return cluster_offset + 2; }

    bool is_aic_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> cluster_offset) & BitStates(1ULL)).has_value();
    }
    bool is_aiv0_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> (cluster_offset + 1)) & BitStates(1ULL)).has_value();
    }
    bool is_aiv1_core_idle(int32_t cluster_offset) const {
        return ((core_states_ >> (cluster_offset + 2)) & BitStates(1ULL)).has_value();
    }

    // --- State mutation ---

    // Toggle bit at the given bit offset (running <-> idle)
    void change_core_state(int32_t bit_offset) { core_states_ ^= BitStates(1ULL << bit_offset); }

    // --- Bit offset <-> worker_id mapping ---

    int32_t get_core_id_by_offset(int32_t offset) const { return core_id_map_[offset]; }

private:
    int32_t cluster_count_;
    BitStates aic_mask_;
    BitStates aiv_mask_;
    BitStates core_states_;
    int32_t core_id_map_[63];  // bit_position -> worker_id, max 21 clusters * 3
};

struct AicpuExecutor {
    int32_t orch_thread_num_;
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

    // Per-core execution state, indexed by core_id (= worker_id)
    CoreExecState core_exec_states_[RUNTIME_MAX_WORKER];

    // Cluster-ordered worker_id lists for core assignment (init-only)
    int32_t aic_worker_ids_[MAX_CORES_PER_THREAD];
    int32_t aiv_worker_ids_[MAX_CORES_PER_THREAD];
    int32_t aic_count_{0};
    int32_t aiv_count_{0};

    // Platform register base address array (set via get_platform_regs())
    uint64_t regs_{0};

    CoreTracker core_trackers_[MAX_AICPU_THREADS];

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
    std::atomic<int32_t> orch_finished_count_{0};  // Number of orchestrator threads that have finished

    // ===== Dynamic core transition state =====
    std::atomic<bool> transition_requested_{false};
    std::atomic<int32_t> wait_reassign_{0};
    std::atomic<bool> reassigned_{false};
    std::atomic<bool> completed_{false};

    // Orchestration SO handle - defer dlclose until all tasks complete
    void* orch_so_handle_{nullptr};
    char orch_so_path_[256]{};  // Path to orchestration SO file for cleanup

    // Shared orchestration function pointer (loaded by first orch thread, used by all)
    DeviceOrchestrationFunc orch_func_{nullptr};
    DeviceOrchestrationBindRuntimeFunc orch_bind_runtime_{nullptr};
    TaskArg* orch_args_cached_{nullptr};

    uint64_t* func_id_to_addr_;
    uint64_t get_function_bin_addr(int func_id) const {
        if (func_id < 0 || func_id >= RUNTIME_MAX_FUNC_ID) return 0;
        return func_id_to_addr_[func_id];
    }

    // ===== Methods =====
    int32_t init(Runtime* runtime);
    int32_t handshake_all_cores(Runtime* runtime);
    bool assign_cores_to_threads();
    void reassign_cores_for_all_threads();
    int32_t resolve_and_dispatch_pto2(Runtime* runtime, int32_t thread_idx);
    int32_t shutdown_aicore(Runtime* runtime, int32_t thread_idx, const int32_t* cur_thread_cores, int32_t core_num);
    int32_t run(Runtime* runtime);
    void deinit(Runtime* runtime);
    void emergency_shutdown(Runtime* runtime);
    void diagnose_stuck_state(
        Runtime* runtime, int32_t thread_idx, const int32_t* cur_thread_cores, int32_t core_num, Handshake* hank);

    template <CoreType CT>
    void check_running_cores_for_completion(int32_t thread_idx,
        Handshake* hank,
        int32_t& completed_this_turn,
        int32_t& cur_thread_completed,
        bool& made_progress,
        PTO2TaskSlotState* deferred_release_slot_states[],
        int32_t& deferred_release_count,
        PTO2LocalReadyBuffer* local_bufs
#if PTO2_PROFILING
        ,
        bool profiling_enabled,
        uint32_t& phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
        ,
        uint64_t& complete_probe_count,
        uint64_t& complete_hit_count,
        uint64_t& notify_edges_total,
        int32_t& notify_max_degree,
        uint64_t& notify_tasks_enqueued,
        uint64_t& fanin_edges_total,
        int32_t& fanin_max_degree,
        uint64_t& sched_complete_perf_cycle
#endif
    ) {
#if !PTO2_PROFILING
        (void)hank;
#endif
        CoreTracker& tracker = core_trackers_[thread_idx];
        auto running_core_states = tracker.get_running_cores<CT>();
        while (running_core_states.has_value()) {
            int32_t bit_pos = running_core_states.pop_first();
            int32_t core_id = tracker.get_core_id_by_offset(bit_pos);
            CoreExecState& core_exec_state = core_exec_states_[core_id];
            uint64_t reg_addr = core_exec_state.reg_addr;

            int32_t expected_reg_task_id = core_exec_state.executing_reg_task_id;
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
                core_exec_state.executing_reg_task_id = AICPU_TASK_INVALID;
                PTO2SubtaskSlot subslot = core_exec_state.executing_subslot;
                PTO2TaskSlotState& slot_state = *core_exec_state.executing_slot_state;

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
                                *deferred_release_slot_states[--deferred_release_count], thread_idx);
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
                tracker.change_core_state(bit_pos);
#if PTO2_PROFILING
                if (profiling_enabled) {
#if PTO2_SCHED_PROFILING
                    uint64_t t_perf_start = get_sys_cnt_aicpu();
#endif
                    Handshake* h = &hank[core_id];
                    uint64_t finish_ts = get_sys_cnt_aicpu();
                    PerfBuffer* perf_buf = (PerfBuffer*)h->perf_records_addr;

                    // Pre-extract fanout (platform layer cannot depend on PTO2DepListEntry)
                    uint64_t fanout_arr[RUNTIME_MAX_FANOUT];
                    int32_t fanout_n = 0;
                    PTO2DepListEntry* cur = slot_state.fanout_head;
                    while (cur != nullptr && fanout_n < RUNTIME_MAX_FANOUT) {
                        fanout_arr[fanout_n++] = cur->slot_state->task->task_id.raw;
                        cur = cur->next;
                    }

                    int32_t perf_slot_idx = static_cast<int32_t>(core_exec_state.executing_subslot);
                    if (perf_aicpu_complete_record(perf_buf,
                            static_cast<uint32_t>(expected_reg_task_id),
                            slot_state.task->task_id.raw,
                            slot_state.task->kernel_id[perf_slot_idx],
                            CT,
                            core_exec_state.dispatch_timestamp,
                            finish_ts,
                            fanout_arr,
                            fanout_n) != 0) {
                        DEV_ERROR("Core %d: perf_aicpu_complete_record failed for task 0x%llx",
                            core_id,
                            (unsigned long long)slot_state.task->task_id.raw);
                    }
#if PTO2_SCHED_PROFILING
                    sched_complete_perf_cycle += (get_sys_cnt_aicpu() - t_perf_start);
#endif
                }
#endif

                DEV_DEBUG("Thread %d: %s core %d completed PTO2 task %d (mixed_complete=%d)",
                    thread_idx,
                    CT == CoreType::AIC ? "AIC" : "AIV",
                    core_id,
                    expected_reg_task_id,
                    mixed_complete ? 1 : 0);
                cur_thread_completed++;
                if (mixed_complete) {
                    completed_this_turn++;
                }
                made_progress = true;
            }
        }
    }

    static const char* shape_name(PTO2ResourceShape shape) {
        switch (shape) {
            case PTO2ResourceShape::AIC:
                return "AIC";
            case PTO2ResourceShape::AIV:
                return "AIV";
            case PTO2ResourceShape::MIX:
                return "MIX";
        }
        return "UNKNOWN";
    }

    /**
     * Returns the dispatch probe order for a given scheduler thread.
     * Widest shapes first to avoid consuming cluster resources with narrow tasks.
     * Even/odd threads use different fallback orders (AIC-first vs AIV-first)
     * to reduce contention on the same ready queue across adjacent threads.
     */
    static const PTO2ResourceShape* get_dispatch_order(int32_t thread_idx) {
        // Even threads: AIC-first fallback after widest
        static constexpr PTO2ResourceShape kEvenOrder[PTO2_NUM_RESOURCE_SHAPES] = {
            PTO2ResourceShape::MIX,
            PTO2ResourceShape::AIC,
            PTO2ResourceShape::AIV,
        };
        // Odd threads: AIV-first fallback after widest
        static constexpr PTO2ResourceShape kOddOrder[PTO2_NUM_RESOURCE_SHAPES] = {
            PTO2ResourceShape::MIX,
            PTO2ResourceShape::AIV,
            PTO2ResourceShape::AIC,
        };
        return (thread_idx % 2 == 0) ? kEvenOrder : kOddOrder;
    }

    int pop_ready_tasks_batch(PTO2ResourceShape shape,
        int32_t thread_idx,
        PTO2LocalReadyBuffer& local_buf,
        PTO2TaskSlotState** out,
        int max_count
#if PTO2_SCHED_PROFILING
        ,
        uint64_t& pop_hit,
        uint64_t& pop_miss,
        uint64_t& local_dispatch_count,
        uint64_t& sched_dispatch_pop_cycle
#endif
    ) {
        (void)thread_idx;
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[];
        uint64_t t_pop_start = get_sys_cnt_aicpu();
        int count = rt->scheduler.get_ready_tasks_batch(shape,
            local_buf,
            out,
            max_count,
            g_sched_pop_atomic_count[thread_idx],
            g_sched_pop_wait_cycle[thread_idx],
            local_dispatch_count);
        sched_dispatch_pop_cycle += (get_sys_cnt_aicpu() - t_pop_start);
        if (count > 0) {
            pop_hit += count;
        } else {
            pop_miss++;
        }
#else
        int count = rt->scheduler.get_ready_tasks_batch(shape, local_buf, out, max_count);
#endif
        return count;
    }

    void dispatch_subtask_to_core(Runtime* runtime,
        int32_t thread_idx,
        int32_t core_offset,
        PTO2TaskSlotState& slot_state,
        PTO2SubtaskSlot subslot
#if PTO2_PROFILING
        ,
        bool profiling_enabled
#endif
    ) {
        CoreTracker& tracker = core_trackers_[thread_idx];
        auto core_id = tracker.get_core_id_by_offset(core_offset);
#if !PTO2_PROFILING
        (void)runtime;
#endif
        CoreExecState& core_exec_state = core_exec_states_[core_id];
        PTO2DispatchPayload& payload = s_pto2_payload_per_core[core_id];
        int32_t slot_idx = static_cast<int32_t>(subslot);
        payload.function_bin_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
        payload.args = slot_state.payload->dispatch_args;
        core_exec_state.executing_subslot = subslot;
        core_exec_state.executing_slot_state = &slot_state;
#if PTO2_PROFILING
        if (profiling_enabled) {
            core_exec_state.dispatch_timestamp = get_sys_cnt_aicpu();
            if (core_exec_state.dispatch_count >= PLATFORM_PROF_BUFFER_SIZE) {
                perf_aicpu_switch_buffer(runtime, core_id, thread_idx);
                core_exec_state.dispatch_count = 0;
            }
            core_exec_state.dispatch_count++;
        }
#endif
        // Per-core monotonic counter for register protocol uniqueness (32-bit).
        // PTO2 task_id encodes (ring_id << 32 | local_id); truncation to uint32 loses ring_id,
        // so tasks from different rings with the same local_id would write identical DATA_MAIN_BASE
        // values. The AICore uses last_reg_val to detect new dispatches and would skip the
        // duplicate, while the stale COND register from the previous task (same local_id) would
        // cause a false-positive completion.
        // PerfRecord.task_id: register token (low 32) until AICPU overwrites with full (ring_id << 32 | local_id).
        core_exec_state.dispatch_seq++;
        uint32_t reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
        // Skip reserved sentinel range [AICORE_EXIT_SIGNAL, 0x7FFFFFFF]
        if (reg_task_id >= AICORE_EXIT_SIGNAL) {
            core_exec_state.dispatch_seq += (TASK_ID_MASK - reg_task_id + 1);
            reg_task_id = core_exec_state.dispatch_seq & TASK_ID_MASK;
        }
        write_reg(core_exec_state.reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(reg_task_id));

        tracker.change_core_state(core_offset);
        core_exec_state.executing_reg_task_id = reg_task_id;
    }
};

static AicpuExecutor g_aicpu_executor;

// ===== AicpuExecutor Method Implementations =====

/**
 * Handshake with all cores and discover their types
 * Sets up register addresses for fast dispatch.
 */
int32_t AicpuExecutor::handshake_all_cores(Runtime* runtime) {
    Handshake* all_handshakes = (Handshake*)runtime->workers;
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
    // task must be written BEFORE aicpu_ready so AICore sees it after waking up
    for (int32_t i = 0; i < cores_total_num_; i++) {
        all_handshakes[i].task = reinterpret_cast<uint64_t>(&s_pto2_payload_per_core[i]);
        all_handshakes[i].aicpu_ready = 1;
    }

    // Get platform physical cores count for validation
    uint32_t max_physical_cores_count = platform_get_physical_cores_count();

    // Step 2: Wait for all cores to respond, collect core type and register addresses
    bool handshake_failed = false;
    for (int32_t i = 0; i < cores_total_num_; i++) {
        Handshake* hank = &all_handshakes[i];

        while (hank->aicore_regs_ready == 0) {
        }

        uint32_t physical_core_id = hank->physical_core_id;

        // Validate physical_core_id before using as array index
        if (physical_core_id >= max_physical_cores_count) {
            DEV_ERROR("Core %d reported invalid physical_core_id=%u (platform max=%u)",
                i,
                physical_core_id,
                max_physical_cores_count);
            handshake_failed = true;
            continue;
        }

        // Get register address using physical_core_id
        uint64_t* regs = reinterpret_cast<uint64_t*>(regs_);
        uint64_t reg_addr = regs[physical_core_id];

        // Initialize AICore registers after discovery (first round)
        platform_init_aicore_regs(reg_addr);
        hank->aicpu_regs_ready = 1;

        while (hank->aicore_done == 0) {
        }

        CoreType type = hank->core_type;

        core_exec_states_[i].reg_addr = reg_addr;
        core_exec_states_[i].worker_id = i;
        core_exec_states_[i].physical_core_id = physical_core_id;
        core_exec_states_[i].core_type = type;

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
    // Cluster-aligned round-robin assignment: cluster ci -> sched thread ci % divisor.
    // Each cluster = 1 AIC + 2 adjacent AIV; the triple is always kept together.
    int32_t divisor = (sched_thread_num_ > 0) ? sched_thread_num_ : thread_num_;
    int32_t cluster_count = aic_count_;

    // Max clusters any single sched thread can hold: ceil(cluster_count / divisor).
    int32_t max_clusters_per_thread = (cluster_count + divisor - 1) / divisor;
    thread_cores_num_ = max_clusters_per_thread * 3;

    if (thread_cores_num_ > CoreTracker::MAX_CORE_PER_THREAD) {
        DEV_ERROR("Can't assign more then 64 cores in per scheduler");
        return false;
    }

    DEV_INFO("Assigning cores (round-robin): %d clusters across %d sched threads (%d AIC, %d AIV)",
        cluster_count,
        divisor,
        aic_count_,
        aiv_count_);

    for (int32_t i = 0; i < MAX_CORES_PER_THREAD; i++) {
        core_exec_states_[i].executing_reg_task_id = AICPU_TASK_INVALID;
    }

    // Count clusters per thread first (round-robin may distribute unevenly)
    int32_t clusters_per_thread[MAX_AICPU_THREADS] = {};
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        clusters_per_thread[ci % divisor]++;
    }
    for (int32_t i = 0; i < divisor; i++) {
        core_trackers_[i].init(clusters_per_thread[i]);
        core_count_per_thread_[i] = 0;
    }

    // Mark orchestrator threads explicitly (no cores).
    for (int32_t t = divisor; t < thread_num_; t++) {
        DEV_INFO("Thread %d: orchestrator (0 cores)", t);
    }

    // Per-sched-thread running core index used while filling core_assignments_.
    int32_t core_idx[MAX_AICPU_THREADS] = {};
    int32_t cluster_idx_per_thread[MAX_AICPU_THREADS] = {};

    for (int32_t ci = 0; ci < cluster_count; ci++) {
        int32_t t = ci % divisor;
        int32_t& idx = core_idx[t];

        int32_t aic_wid = aic_worker_ids_[ci];
        int32_t aiv0_wid = aiv_worker_ids_[2 * ci];
        int32_t aiv1_wid = aiv_worker_ids_[2 * ci + 1];

        core_trackers_[t].set_cluster(cluster_idx_per_thread[t]++, aic_wid, aiv0_wid, aiv1_wid);

        core_assignments_[t][idx++] = aic_wid;
        core_assignments_[t][idx++] = aiv0_wid;
        core_assignments_[t][idx++] = aiv1_wid;

        DEV_INFO("Thread %d: cluster %d (AIC=%d, AIV0=%d, AIV1=%d)", t, ci, aic_wid, aiv0_wid, aiv1_wid);
    }

    for (int32_t t = 0; t < divisor; t++) {
        core_count_per_thread_[t] = core_idx[t];
        DEV_INFO("Thread %d: total %d cores (%d clusters)", t, core_idx[t], core_trackers_[t].get_cluster_count());
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
        auto all_running = core_trackers_[i].get_all_running_cores();
        int32_t bp;
        while ((bp = all_running.pop_first()) >= 0) {
            running_cores[core_trackers_[i].get_core_id_by_offset(bp)] = true;
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
        core_trackers_[i].init(clusters_per_thread[i]);
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
        core_trackers_[t].set_cluster(cl_idx, aic_wid, aiv0_wid, aiv1_wid);

        // init() marks all idle; toggle cores that were running
        if (running_cores[aic_wid]) {
            core_trackers_[t].change_core_state(cl_idx * 3);
        }
        if (running_cores[aiv0_wid]) {
            core_trackers_[t].change_core_state(cl_idx * 3 + 1);
        }
        if (running_cores[aiv1_wid]) {
            core_trackers_[t].change_core_state(cl_idx * 3 + 2);
        }

        core_assignments_[t][core_count_per_thread_[t]++] = aic_wid;
        core_assignments_[t][core_count_per_thread_[t]++] = aiv0_wid;
        core_assignments_[t][core_count_per_thread_[t]++] = aiv1_wid;
    }

    // Log final distribution
    DEV_INFO("Core reassignment complete:");
    for (int32_t t = 0; t < thread_num_; t++) {
        int32_t aic_running = core_trackers_[t].get_running_count<CoreType::AIC>();
        int32_t aiv_running = core_trackers_[t].get_running_count<CoreType::AIV>();
        DEV_INFO("  Thread %d: %d cores, %d clusters (AIC running=%d, AIV running=%d)",
            t,
            core_count_per_thread_[t],
            core_trackers_[t].get_cluster_count(),
            aic_running,
            aiv_running);
    }
}

int32_t AicpuExecutor::init(Runtime* runtime) {
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
    orch_thread_num_ = runtime->orch_thread_num;
    sched_thread_num_ = thread_num_ - orch_thread_num_;
    orch_to_sched_ = runtime->orch_to_sched;
    if (thread_num_ == 0) thread_num_ = 1;

    if (!orch_to_sched_ && sched_thread_num_ == 0) {
        DEV_ERROR(
            "no scheduler and orch not trans to schedulers when finished, maybe you need set env PTO2_ORCH_TO_SCHED=1 "
            "or scale down orch number.");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Zero all per-core execution state before handshake
    memset(core_exec_states_, 0, sizeof(core_exec_states_));

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
        auto* header = static_cast<PTO2SharedMemoryHeader*>(runtime->get_pto2_gm_sm_ptr());
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
    memset(s_pto2_payload_per_core, 0, sizeof(s_pto2_payload_per_core));

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
    Runtime* runtime, int32_t thread_idx, const int32_t* cur_thread_cores, int32_t core_num) {
    (void)runtime;
    if (core_num == 0) return 0;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, core_num);

    for (int32_t i = 0; i < core_num; i++) {
        int32_t core_id = cur_thread_cores[i];
        uint64_t reg_addr = core_exec_states_[core_id].reg_addr;
        if (reg_addr != 0) {
            platform_deinit_aicore_regs(reg_addr);
        } else {
            DEV_ERROR("Thread %d: Core %d has invalid register address", thread_idx, core_id);
        }
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

int32_t AicpuExecutor::resolve_and_dispatch_pto2(Runtime* runtime, int32_t thread_idx) {
    int32_t& core_num = core_count_per_thread_[thread_idx];
    CoreTracker& tracker = core_trackers_[thread_idx];
    DEV_INFO("Thread %d: resolve_and_dispatch_pto2 entry", thread_idx);

    void* sm_base = runtime->get_pto2_gm_sm_ptr();
    if (!sm_base) {
        DEV_ERROR("PTO2 dispatch: sm_base is null");
        return -1;
    }
    DEV_INFO("Thread %d: sm_base=%p", thread_idx, sm_base);

    PTO2SharedMemoryHeader* header = static_cast<PTO2SharedMemoryHeader*>(sm_base);
    DEV_INFO("Thread %d: header=%p, task_desc_offset[0]=%lu, window_size=%lu",
        thread_idx,
        (void*)header,
        (unsigned long)header->rings[0].task_descriptors_offset,
        (unsigned long)header->rings[0].task_window_size);

    Handshake* hank = static_cast<Handshake*>(runtime->workers);
    DEV_INFO("Thread %d: hank=%p, window_size=%lu",
        thread_idx,
        (void*)hank,
        (unsigned long)header->rings[0].task_window_size);

    // One-time init: assign perf buffers (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) {
        DEV_INFO("Thread %d: doing one-time init", thread_idx);

#if PTO2_PROFILING
        // Assign perf buffers to cores early so profiling captures all tasks
        // (total_tasks written to header later when orchestrator completes)
        if (runtime->enable_profiling) {
            perf_aicpu_init_profiling(runtime);
            // Initialize phase profiling for scheduler threads + orchestrator threads
            perf_aicpu_init_phase_profiling(runtime, sched_thread_num_, orch_thread_num_);
            perf_aicpu_set_orch_thread_idx(sched_thread_num_);
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
    constexpr int LOCAL_READY_CAP_PER_TYPE = 64;
    PTO2TaskSlotState* local_ptrs[PTO2_NUM_RESOURCE_SHAPES][LOCAL_READY_CAP_PER_TYPE];
    PTO2LocalReadyBuffer local_bufs[PTO2_NUM_RESOURCE_SHAPES];
    for (int32_t i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        local_bufs[i].reset(local_ptrs[i], LOCAL_READY_CAP_PER_TYPE);
    }
    PTO2TaskSlotState* deferred_release_slot_states[256];
    int32_t deferred_release_count = 0;

    bool cores_released = false;

#if PTO2_PROFILING
    uint64_t sched_start_ts = get_sys_cnt_aicpu();
#endif

    while (true) {
        bool made_progress = false;
#if PTO2_PROFILING
        CYCLE_COUNT_START();
        sched_loop_count++;
        uint64_t _t0_phase = _t0;
#endif
        int32_t task_count = 0;
        if (!tracker.has_any_running_cores()) {
            bool orch_done = orchestrator_done_;
            if (orch_done) {
                // Check for orchestrator fatal error — exit immediately
                int32_t orch_err = header->orch_error_code.load(std::memory_order_acquire);
                if (orch_err != PTO2_ERROR_NONE) {
                    DEV_ERROR(
                        "Thread %d: Fatal error (code=%d), sending EXIT_SIGNAL to all cores. "
                        "completed_tasks=%d, total_tasks=%d",
                        thread_idx,
                        orch_err,
                        completed_tasks_.load(std::memory_order_relaxed),
                        total_tasks_);
                    emergency_shutdown(runtime);
                    completed_.store(true, std::memory_order_release);
                    break;
                }

                // Normal exit: all tasks complete
                task_count = total_tasks_;
                if (task_count > 0 && completed_tasks_.load(std::memory_order_relaxed) >= task_count) {
                    completed_.store(true, std::memory_order_release);
                    DEV_INFO("Thread %d: PTO2 completed tasks %d/%d",
                        thread_idx,
                        completed_tasks_.load(std::memory_order_relaxed),
                        task_count);
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
        if (tracker.has_running_cores<CoreType::AIC>()) {
            try_completed = true;
            check_running_cores_for_completion<CoreType::AIC>(thread_idx,
                hank,
                completed_this_turn,
                cur_thread_completed,
                made_progress,
                deferred_release_slot_states,
                deferred_release_count,
                local_bufs
#if PTO2_PROFILING
                ,
                profiling_enabled,
                phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
                ,
                complete_probe_count,
                complete_hit_count,
                notify_edges_total,
                notify_max_degree,
                notify_tasks_enqueued,
                fanin_edges_total,
                fanin_max_degree,
                sched_complete_perf_cycle
#endif
            );
        }

        // Check AIV running cores
        if (tracker.has_running_cores<CoreType::AIV>()) {
            try_completed = true;
            check_running_cores_for_completion<CoreType::AIV>(thread_idx,
                hank,
                completed_this_turn,
                cur_thread_completed,
                made_progress,
                deferred_release_slot_states,
                deferred_release_count,
                local_bufs
#if PTO2_PROFILING
                ,
                profiling_enabled,
                phase_complete_count
#endif
#if PTO2_SCHED_PROFILING
                ,
                complete_probe_count,
                complete_hit_count,
                notify_edges_total,
                notify_max_degree,
                notify_tasks_enqueued,
                fanin_edges_total,
                fanin_max_degree,
                sched_complete_perf_cycle
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
                    DEV_ALWAYS("PTO2 progress: completed=%d total=%d (%.1f%%)",
                        new_total,
                        task_count,
                        100.0 * new_total / task_count);
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
                    thread_idx, AicpuPhaseId::SCHED_COMPLETE, _t0_phase, _t1, sched_loop_count, phase_complete_count);
                _t0_phase = _t1;
                phase_complete_count = 0;
            }
        }
#endif

        bool try_pushed = false;
        const PTO2ResourceShape* dispatch_order = get_dispatch_order(thread_idx);
        for (int32_t si = 0; si < PTO2_NUM_RESOURCE_SHAPES; si++) {
            PTO2ResourceShape shape = dispatch_order[si];
            auto valid_cluster_states = tracker.get_valid_cluster_offset_states(shape);
            if (!valid_cluster_states.has_value()) {
                continue;
            }
            auto& local_buf = local_bufs[static_cast<int32_t>(shape)];

            while (valid_cluster_states.has_value()) {
                int want = valid_cluster_states.count();
                PTO2TaskSlotState* batch[CoreTracker::MAX_CLUSTERS];
                int got = pop_ready_tasks_batch(shape,
                    thread_idx,
                    local_buf,
                    batch,
                    want
#if PTO2_SCHED_PROFILING
                    ,
                    pop_hit,
                    pop_miss,
                    local_dispatch_count,
                    sched_dispatch_pop_cycle
#endif
                );
                if (got == 0) break;

                for (int bi = 0; bi < got; bi++) {
                    PTO2TaskSlotState* slot_state = batch[bi];
                    try_pushed = true;
#if PTO2_PROFILING
                    phase_dispatch_count++;
#endif
#if PTO2_SCHED_PROFILING
                    uint64_t t_setup_start = get_sys_cnt_aicpu();
#endif
                    auto current_valid_cluster_offset = valid_cluster_states.pop_first();
                    if (shape == PTO2ResourceShape::MIX) {
                        // Full-cluster scheduling: dispatch only to cores indicated by active_mask.
                        // Unused cores in the cluster remain idle for single-core tasks.
                        uint8_t mask = slot_state->active_mask;
                        if (mask & PTO2_SUBTASK_MASK_AIC) {
                            auto core_offset = tracker.get_aic_core_offset(current_valid_cluster_offset);
                            dispatch_subtask_to_core(runtime,
                                thread_idx,
                                core_offset,
                                *slot_state,
                                PTO2SubtaskSlot::AIC
#if PTO2_PROFILING
                                ,
                                profiling_enabled
#endif
                            );
                        }
                        if (mask & PTO2_SUBTASK_MASK_AIV0) {
                            auto core_offset = tracker.get_aiv0_core_offset(current_valid_cluster_offset);
                            dispatch_subtask_to_core(runtime,
                                thread_idx,
                                core_offset,
                                *slot_state,
                                PTO2SubtaskSlot::AIV0
#if PTO2_PROFILING
                                ,
                                profiling_enabled
#endif
                            );
                        }
                        if (mask & PTO2_SUBTASK_MASK_AIV1) {
                            auto core_offset = tracker.get_aiv1_core_offset(current_valid_cluster_offset);
                            dispatch_subtask_to_core(runtime,
                                thread_idx,
                                core_offset,
                                *slot_state,
                                PTO2SubtaskSlot::AIV1
#if PTO2_PROFILING
                                ,
                                profiling_enabled
#endif
                            );
                        }
                    } else if (shape == PTO2ResourceShape::AIC) {
                        auto core_offset = tracker.get_aic_core_offset(current_valid_cluster_offset);
                        dispatch_subtask_to_core(runtime,
                            thread_idx,
                            core_offset,
                            *slot_state,
                            PTO2SubtaskSlot::AIC
#if PTO2_PROFILING
                            ,
                            profiling_enabled
#endif
                        );
                    } else {  // shape == PTO2ResourceShape::AIV
                        auto core_offset = tracker.is_aiv0_core_idle(current_valid_cluster_offset)
                                               ? tracker.get_aiv0_core_offset(current_valid_cluster_offset)
                                               : tracker.get_aiv1_core_offset(current_valid_cluster_offset);
                        dispatch_subtask_to_core(runtime,
                            thread_idx,
                            core_offset,
                            *slot_state,
                            PTO2SubtaskSlot::AIV0
#if PTO2_PROFILING
                            ,
                            profiling_enabled
#endif
                        );
                    }
                    made_progress = true;
#if PTO2_SCHED_PROFILING
                    sched_dispatch_setup_cycle += (get_sys_cnt_aicpu() - t_setup_start);
#endif
                    DEV_DEBUG("Thread %d: Dispatching %s task %lld to cluster_offset %d",
                        thread_idx,
                        shape_name(shape),
                        (long long)slot_state->task->task_id.raw,
                        current_valid_cluster_offset);
                }

                // lazy update valid_cluster_states
                if (!valid_cluster_states.has_value()) {
                    valid_cluster_states = tracker.get_valid_cluster_offset_states(shape);
                }
            }
        }

        // requeue in global ready queue
        for (int32_t si = 0; si < PTO2_NUM_RESOURCE_SHAPES; si++) {
            PTO2ResourceShape shape = dispatch_order[si];
            auto& local_buf = local_bufs[static_cast<int32_t>(shape)];
            auto& ready_queue = rt->scheduler.ready_queues[static_cast<int32_t>(shape)];
#if PTO2_SCHED_PROFILING
            local_overflow_count += local_buf.count;
#endif
            if (local_buf.count > 0) {
                ready_queue.push_batch(local_buf.slot_states, local_buf.count);
                local_buf.count = 0;
            }
        }

#if PTO2_PROFILING
        if (!try_pushed) {
            CYCLE_COUNT_LAP(sched_idle_cycle);
        } else {
            CYCLE_COUNT_LAP(sched_dispatch_cycle);
            if (profiling_enabled && phase_dispatch_count > 0) {
                perf_aicpu_record_phase(
                    thread_idx, AicpuPhaseId::SCHED_DISPATCH, _t0_phase, _t1, sched_loop_count, phase_dispatch_count);
                _t0_phase = _t1;
                phase_dispatch_count = 0;
            }
        }
#endif

#if !PTO2_PROFILING
        (void)try_completed;
        (void)try_pushed;
#endif

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
                    DEV_ERROR("Thread %d: Fatal error detected (code=%d), sending EXIT_SIGNAL to all cores",
                        thread_idx,
                        orch_err);
                    emergency_shutdown(runtime);
                    completed_.store(true, std::memory_order_release);
                    break;
                }
            }

            if (thread_idx == 0 && task_count > 0 && idle_iterations % STALL_LOG_INTERVAL == 0) {
                int32_t c = completed_tasks_.load(std::memory_order_relaxed);
                DEV_ALWAYS("PTO2 stall: no progress for %d iterations, completed=%d total=%d (last progress at %d)",
                    idle_iterations,
                    c,
                    task_count,
                    last_progress_count);
                // Scan all task slots to find truly stuck tasks using scheduler state
                PTO2SchedulerState* sched = &rt->scheduler;
                PTO2SharedMemoryHeader* sm_header_diag = static_cast<PTO2SharedMemoryHeader*>(sm_base);
                int32_t cnt_ready = 0, cnt_waiting = 0, cnt_inflight = 0;
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                    int32_t ring_task_count =
                        sm_header_diag->rings[r].fc.current_task_index.load(std::memory_order_relaxed);
                    for (int32_t si = 0; si < ring_task_count; si++) {
                        PTO2TaskSlotState& slot_state = sched->get_slot_state(r, si);
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
                                    "  STUCK-READY  ring=%d task_id=%lld kernel_id=%d refcount=%d fanin=%d state=%d",
                                    r,
                                    (long long)slot_state.task->task_id.raw,
                                    kid,
                                    rc,
                                    fi,
                                    (int32_t)st);
                            }
                        } else {
                            cnt_waiting++;
                            if (cnt_waiting <= STALL_DUMP_WAIT_MAX) {
                                DEV_ALWAYS(
                                    "  STUCK-WAIT   ring=%d task_id=%lld kernel_id=%d refcount=%d fanin=%d state=%d",
                                    r,
                                    (long long)slot_state.task->task_id.raw,
                                    kid,
                                    rc,
                                    fi,
                                    (int32_t)st);
                            }
                        }
                    }
                }
                DEV_ALWAYS("  scan result: stuck_ready=%d stuck_waiting=%d in_flight=%d",
                    cnt_ready,
                    cnt_waiting,
                    cnt_inflight);
                // Log this thread's dispatch state
                int32_t aic_running = tracker.get_running_count<CoreType::AIC>();
                int32_t aiv_running = tracker.get_running_count<CoreType::AIV>();
                int32_t total_running = aic_running + aiv_running;
                DEV_ALWAYS("  thread=%d running_cores=%d (AIC=%d AIV=%d) core_num=%d",
                    thread_idx,
                    total_running,
                    aic_running,
                    aiv_running,
                    core_num);
                // Dump running cores
                auto all_running = tracker.get_all_running_cores();
                int32_t dump_count = 0;
                int32_t bp;
                while (dump_count < STALL_DUMP_CORE_MAX && (bp = all_running.pop_first()) >= 0) {
                    dump_count++;
                    int32_t cid = tracker.get_core_id_by_offset(bp);
                    int32_t sw_tid = core_exec_states_[cid].executing_reg_task_id;
                    int32_t hw_kernel = -1;
                    if (sw_tid >= 0 && core_exec_states_[cid].executing_slot_state) {
                        int32_t diag_slot = static_cast<int32_t>(core_exec_states_[cid].executing_subslot);
                        hw_kernel = core_exec_states_[cid].executing_slot_state->task->kernel_id[diag_slot];
                    }
                    uint64_t cond_reg = read_reg(core_exec_states_[cid].reg_addr, RegId::COND);
                    DEV_ALWAYS("    core=%d cond=0x%x(state=%d,id=%d) exec_id=%d kernel=%d",
                        cid,
                        (unsigned)cond_reg,
                        EXTRACT_TASK_STATE(cond_reg),
                        EXTRACT_TASK_ID(cond_reg),
                        sw_tid,
                        hw_kernel);
                }
                // Dump cluster state
                for (int32_t cli = 0; cli < tracker.get_cluster_count() && cli < STALL_DUMP_CORE_MAX; cli++) {
                    int32_t offset = cli * 3;
                    DEV_ALWAYS("    cluster[%d] aic=%d(%s) aiv0=%d(%s) aiv1=%d(%s)",
                        cli,
                        tracker.get_aic_core_id(offset),
                        tracker.is_aic_core_idle(offset) ? "idle" : "busy",
                        tracker.get_aiv0_core_id(offset),
                        tracker.is_aiv0_core_idle(offset) ? "idle" : "busy",
                        tracker.get_aiv1_core_id(offset),
                        tracker.is_aiv1_core_idle(offset) ? "idle" : "busy");
                }
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("Thread %d: PTO2 timeout after %d idle iterations", thread_idx, idle_iterations);
#if PTO2_PROFILING
                // Benchmark: scheduler lifetime end timestamp on timeout path
                uint64_t sched_timeout_ts = get_sys_cnt_aicpu();
                DEV_ALWAYS("Thread %d: sched_start=%llu sched_end(timeout)=%llu sched_cost=%.3fus",
                    thread_idx,
                    (unsigned long long)sched_start_ts,
                    (unsigned long long)sched_timeout_ts,
                    cycles_to_us(sched_timeout_ts - sched_start_ts));
#endif
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
    // Record sched_end before any DEV_ALWAYS to avoid init cost contamination
    uint64_t sched_end_ts = get_sys_cnt_aicpu();
    DEV_ALWAYS("Thread %d: sched_start=%llu sched_end=%llu sched_cost=%.3fus",
        thread_idx,
        (unsigned long long)sched_start_ts,
        (unsigned long long)sched_end_ts,
        cycles_to_us(sched_end_ts - sched_start_ts));

    // Scheduler summary logging (always print when PTO2_PROFILING=1)
    uint64_t sched_total = sched_complete_cycle + sched_scan_cycle + sched_dispatch_cycle + sched_idle_cycle;
    if (sched_total == 0) sched_total = 1;  // avoid div-by-zero

#if PTO2_SCHED_PROFILING
    // Two-level tree display: sub-phase breakdown within complete and dispatch
    {
        PTO2SchedProfilingData sp = pto2_scheduler_get_profiling(thread_idx);
        uint64_t otc_total = sp.lock_cycle + sp.fanout_cycle + sp.fanin_cycle + sp.self_consumed_cycle;
        uint64_t complete_poll = (sched_complete_cycle > otc_total + sched_complete_perf_cycle)
                                     ? (sched_complete_cycle - otc_total - sched_complete_perf_cycle)
                                     : 0;
        uint64_t dispatch_poll = (sched_dispatch_cycle > sched_dispatch_pop_cycle + sched_dispatch_setup_cycle)
                                     ? (sched_dispatch_cycle - sched_dispatch_pop_cycle - sched_dispatch_setup_cycle)
                                     : 0;

        DEV_ALWAYS("Thread %d: === Scheduler Phase Breakdown: total=%.3fus, %d tasks ===",
            thread_idx,
            cycles_to_us(sched_total),
            cur_thread_completed);

        // Level 1: complete
        double notify_avg = cur_thread_completed > 0 ? (double)notify_edges_total / cur_thread_completed : 0.0;
        double fanin_avg = cur_thread_completed > 0 ? (double)fanin_edges_total / cur_thread_completed : 0.0;
        DEV_ALWAYS(
            "Thread %d:   complete       : %.3fus (%.1f%%)  [fanout: edges=%llu, max_degree=%d, avg=%.1f]  [fanin: "
            "edges=%llu, max_degree=%d, avg=%.1f]",
            thread_idx,
            cycles_to_us(sched_complete_cycle),
            sched_complete_cycle * 100.0 / sched_total,
            (unsigned long long)notify_edges_total,
            notify_max_degree,
            notify_avg,
            (unsigned long long)fanin_edges_total,
            fanin_max_degree,
            fanin_avg);

        // Level 2: complete sub-phases (percentage relative to complete)
        uint64_t c_parent = sched_complete_cycle > 0 ? sched_complete_cycle : 1;
        uint64_t complete_miss_count =
            (complete_probe_count > complete_hit_count) ? (complete_probe_count - complete_hit_count) : 0;
        double complete_hit_rate = complete_probe_count > 0 ? complete_hit_count * 100.0 / complete_probe_count : 0.0;
        DEV_ALWAYS("Thread %d:     poll         : %.3fus (%.1f%%)  hit=%llu, miss=%llu, hit_rate=%.1f%%",
            thread_idx,
            cycles_to_us(complete_poll),
            complete_poll * 100.0 / c_parent,
            (unsigned long long)complete_hit_count,
            (unsigned long long)complete_miss_count,
            complete_hit_rate);
        DEV_ALWAYS("Thread %d:     otc_lock     : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu",
            thread_idx,
            cycles_to_us(sp.lock_cycle),
            sp.lock_cycle * 100.0 / c_parent,
            cycles_to_us(sp.lock_cycle - sp.lock_wait_cycle),
            cycles_to_us(sp.lock_wait_cycle),
            (unsigned long long)sp.lock_atomic_count);
        DEV_ALWAYS("Thread %d:     otc_fanout   : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu",
            thread_idx,
            cycles_to_us(sp.fanout_cycle),
            sp.fanout_cycle * 100.0 / c_parent,
            cycles_to_us(sp.fanout_cycle - sp.push_wait_cycle),
            cycles_to_us(sp.push_wait_cycle),
            (unsigned long long)sp.fanout_atomic_count);
        DEV_ALWAYS("Thread %d:     otc_fanin    : %.3fus (%.1f%%)  atomics=%llu",
            thread_idx,
            cycles_to_us(sp.fanin_cycle),
            sp.fanin_cycle * 100.0 / c_parent,
            (unsigned long long)sp.fanin_atomic_count);
        DEV_ALWAYS("Thread %d:     otc_self     : %.3fus (%.1f%%)  atomics=%llu",
            thread_idx,
            cycles_to_us(sp.self_consumed_cycle),
            sp.self_consumed_cycle * 100.0 / c_parent,
            (unsigned long long)sp.self_atomic_count);
        DEV_ALWAYS("Thread %d:     perf         : %.3fus (%.1f%%)",
            thread_idx,
            cycles_to_us(sched_complete_perf_cycle),
            sched_complete_perf_cycle * 100.0 / c_parent);

        // Level 1: dispatch
        uint64_t pop_total = pop_hit + pop_miss;
        double pop_hit_rate = pop_total > 0 ? pop_hit * 100.0 / pop_total : 0.0;
        DEV_ALWAYS("Thread %d:   dispatch       : %.3fus (%.1f%%)  [pop: hit=%llu, miss=%llu, hit_rate=%.1f%%]",
            thread_idx,
            cycles_to_us(sched_dispatch_cycle),
            sched_dispatch_cycle * 100.0 / sched_total,
            (unsigned long long)pop_hit,
            (unsigned long long)pop_miss,
            pop_hit_rate);
        uint64_t global_dispatch_count = pop_hit - local_dispatch_count;
        uint64_t total_dispatched = local_dispatch_count + global_dispatch_count;
        double local_hit_rate = total_dispatched > 0 ? local_dispatch_count * 100.0 / total_dispatched : 0.0;
        DEV_ALWAYS("Thread %d:     local_disp   : local=%llu, global=%llu, overflow=%llu, local_rate=%.1f%%",
            thread_idx,
            (unsigned long long)local_dispatch_count,
            (unsigned long long)global_dispatch_count,
            (unsigned long long)local_overflow_count,
            local_hit_rate);

        // Level 2: dispatch sub-phases (percentage relative to dispatch)
        uint64_t d_parent = sched_dispatch_cycle > 0 ? sched_dispatch_cycle : 1;
        DEV_ALWAYS("Thread %d:     poll         : %.3fus (%.1f%%)",
            thread_idx,
            cycles_to_us(dispatch_poll),
            dispatch_poll * 100.0 / d_parent);
        DEV_ALWAYS("Thread %d:     pop          : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu",
            thread_idx,
            cycles_to_us(sched_dispatch_pop_cycle),
            sched_dispatch_pop_cycle * 100.0 / d_parent,
            cycles_to_us(sched_dispatch_pop_cycle - sp.pop_wait_cycle),
            cycles_to_us(sp.pop_wait_cycle),
            (unsigned long long)sp.pop_atomic_count);
        DEV_ALWAYS("Thread %d:     setup        : %.3fus (%.1f%%)",
            thread_idx,
            cycles_to_us(sched_dispatch_setup_cycle),
            sched_dispatch_setup_cycle * 100.0 / d_parent);

        // Level 1: scan
        DEV_ALWAYS("Thread %d:   scan           : %.3fus (%.1f%%)",
            thread_idx,
            cycles_to_us(sched_scan_cycle),
            sched_scan_cycle * 100.0 / sched_total);

        // Level 1: idle
        DEV_ALWAYS("Thread %d:   idle           : %.3fus (%.1f%%)",
            thread_idx,
            cycles_to_us(sched_idle_cycle),
            sched_idle_cycle * 100.0 / sched_total);

        // Average per completion
        if (cur_thread_completed > 0) {
            DEV_ALWAYS("Thread %d:   avg/complete   : %.3fus",
                thread_idx,
                cycles_to_us(sched_complete_cycle) / cur_thread_completed);
        }
    }
#endif
    // Summary line (always print when PTO2_PROFILING=1)
    DEV_ALWAYS("Thread %d: Scheduler summary: total_time=%.3fus, loops=%llu, tasks_scheduled=%d",
        thread_idx,
        cycles_to_us(sched_total),
        (unsigned long long)sched_loop_count,
        cur_thread_completed);
#endif

#if PTO2_PROFILING
    // Flush performance buffers for cores managed by this thread
    if (profiling_enabled) {
        perf_aicpu_flush_buffers(runtime, thread_idx, core_assignments_[thread_idx], core_num);
        perf_aicpu_flush_phase_buffers(thread_idx);
    }
#endif

    return cur_thread_completed;
}

int32_t AicpuExecutor::run(Runtime* runtime) {
    int32_t thread_idx = thread_idx_++;
    DEV_INFO("Thread %d: Start", thread_idx);

    // Orchestrator check
    if (thread_idx >= sched_thread_num_) {
#if PTO2_PROFILING
        uint64_t orch_cycle_start = 0;
        int32_t pto2_submitted_tasks = -1;
#endif
        int32_t orch_idx = thread_idx - sched_thread_num_;
        if (runtime->get_orch_built_on_host()) {
            DEV_INFO("Thread %d: Host orchestration mode, no-op (orch_idx=%d)", thread_idx, orch_idx);
        } else {
            // First orchestrator thread (orch_idx == 0): load SO, create runtime
            if (orch_idx == 0) {
                DEV_INFO("Thread %d: Primary orchestrator, loading SO via dlopen", thread_idx);

                const void* so_data = runtime->get_device_orch_so_data();
                size_t so_size = runtime->get_device_orch_so_size();

                if (so_data == nullptr || so_size == 0) {
                    DEV_ERROR("Thread %d: Device orchestration SO not set", thread_idx);
                    return -1;
                }

                // Try multiple paths that may allow execution on AICPU
                char so_path[256];
                bool file_created = false;
                const char* candidate_dirs[] = {
                    "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device", "/usr/lib64", "/lib64", "/var/tmp", "/tmp"};
                const int32_t num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

                for (int32_t i = 0; i < num_candidates && !file_created; i++) {
                    snprintf(so_path, sizeof(so_path), "%s/libdevice_orch_%d.so", candidate_dirs[i], getpid());
                    int32_t fd = open(so_path, O_WRONLY | O_CREAT | O_TRUNC, 0755);
                    if (fd < 0) {
                        DEV_INFO("Thread %d: Cannot create SO at %s (errno=%d), trying next path",
                            thread_idx,
                            so_path,
                            errno);
                        continue;
                    }
                    ssize_t written = write(fd, so_data, so_size);
                    close(fd);
                    if (written != static_cast<ssize_t>(so_size)) {
                        DEV_INFO("Thread %d: Cannot write SO to %s (errno=%d), trying next path",
                            thread_idx,
                            so_path,
                            errno);
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
                void* handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
                const char* dlopen_err = dlerror();
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
                const char* dlsym_error = dlerror();
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

                dlerror();
                auto bind_runtime_func =
                    reinterpret_cast<DeviceOrchestrationBindRuntimeFunc>(dlsym(handle, "pto2_framework_bind_runtime"));
                const char* bind_runtime_error = dlerror();
                if (bind_runtime_error != nullptr) {
                    DEV_INFO("Thread %d: Optional TLS runtime binder not found: %s", thread_idx, bind_runtime_error);
                    bind_runtime_func = nullptr;
                }

                TaskArg* args = runtime->get_orch_args();
                int32_t arg_count = runtime->get_orch_arg_count();
                DEV_INFO("Thread %d: sm_ptr=%p, arg_count=%d", thread_idx, runtime->get_pto2_gm_sm_ptr(), arg_count);
                for (int32_t i = 0; i < arg_count && i < 20; i++) {
                    if (args[i].kind == TaskArgKind::TENSOR) {
                        DEV_INFO("Thread %d: orch_args[%d] = TENSOR(data=0x%lx, ndims=%u, dtype=%u)",
                            thread_idx,
                            i,
                            (unsigned long)args[i].tensor.data,
                            args[i].tensor.ndims,
                            (unsigned)args[i].tensor.dtype);
                    } else {
                        DEV_INFO(
                            "Thread %d: orch_args[%d] = SCALAR(0x%lx)", thread_idx, i, (unsigned long)args[i].scalar);
                    }
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
                DEV_INFO("Thread %d: Ring sizes: task_window=%lu, heap=%lu, dep_pool=%d",
                    thread_idx,
                    (unsigned long)task_window_size,
                    (unsigned long)heap_size,
                    dep_pool_capacity);

                void* sm_ptr = runtime->get_pto2_gm_sm_ptr();
                void* gm_heap = runtime->get_pto2_gm_heap_ptr();

                uint64_t sm_size = pto2_sm_calculate_size(task_window_size);
                PTO2SharedMemoryHandle* sm_handle =
                    pto2_sm_create_from_buffer(sm_ptr, sm_size, task_window_size, heap_size);
                if (!sm_handle) {
                    DEV_ERROR("Thread %d: Failed to create shared memory handle", thread_idx);
                    dlclose(handle);
                    unlink(so_path);
                    return -1;
                }

                rt = pto2_runtime_create_from_sm(
                    PTO2_MODE_EXECUTE, sm_handle, gm_heap, heap_size, orch_thread_num_, dep_pool_capacity);
                if (!rt) {
                    DEV_ERROR("Thread %d: Failed to create PTO2Runtime", thread_idx);
                    pto2_sm_destroy(sm_handle);
                    dlclose(handle);
                    unlink(so_path);
                    return -1;
                }

#if PTO2_PROFILING
                for (int i = 0; i < orch_thread_num_; i++) {
                    rt->orchestrators[i].enable_profiling = runtime->enable_profiling;
                }
#endif

                // With multi-ring, slot_states are per-ring inside the scheduler.
                runtime->set_pto2_slot_states_ptr(nullptr);

                // Store shared state for other orchestrator threads
                orch_func_ = orch_func;
                orch_bind_runtime_ = bind_runtime_func;
                orch_args_cached_ = args;
                orch_so_handle_ = handle;
                snprintf(orch_so_path_, sizeof(orch_so_path_), "%s", so_path);

                // All-orchestrator mode: primary orchestrator does one-time init
                if (sched_thread_num_ == 0) {
                    DEV_INFO("Thread %d: All-orchestrator mode, doing one-time init", thread_idx);
                    if (runtime->enable_profiling) {
                        perf_aicpu_init_profiling(runtime);
                        // After transition, all threads become schedulers
                        perf_aicpu_init_phase_profiling(runtime, thread_num_, orch_thread_num_);
                        perf_aicpu_set_orch_thread_idx(0);
                    }
                    pto2_init_done_.store(true, std::memory_order_release);
                    pto2_init_complete_.store(true, std::memory_order_release);
                    DEV_INFO("Thread %d: One-time init done", thread_idx);
                }

                runtime_init_ready_.store(true, std::memory_order_release);
            } else {
                // Non-primary orchestrator: wait for primary to finish setup
                while (!runtime_init_ready_.load(std::memory_order_acquire)) {
                    SPIN_WAIT_HINT();
                }
            }

            // Wait for scheduler's one-time init to complete
            // (or primary orchestrator's init in all-orchestrator mode)
            while (!pto2_init_complete_.load(std::memory_order_acquire)) {
                SPIN_WAIT_HINT();
            }

            pto2_set_orch_thread_idx(orch_idx);

#if PTO2_PROFILING
            // Each orchestrator thread sets its own phase buffer index (thread-local)
            if (runtime->enable_profiling) {
                perf_aicpu_set_orch_thread_idx(thread_idx);
            }
#endif

#if PTO2_PROFILING
            orch_cycle_start = get_sys_cnt_aicpu();
#endif
            if (orch_bind_runtime_ != nullptr) {
                orch_bind_runtime_(rt);
            }
            pto2_rt_scope_begin(rt);
            orch_func_(orch_args_cached_, orch_thread_num_, orch_idx);
            pto2_rt_scope_end(rt);
#if PTO2_PROFILING
            uint64_t orch_cycle_end = get_sys_cnt_aicpu();
            (void)orch_cycle_end;
#endif

            // Print orchestrator profiling data
#if PTO2_ORCH_PROFILING
            PTO2OrchProfilingData p = pto2_orchestrator_get_profiling();
            uint64_t total =
                p.sync_cycle + p.alloc_cycle + p.params_cycle + p.lookup_cycle + p.insert_cycle + p.fanin_cycle;
            if (total == 0) total = 1;  // avoid div-by-zero
            DEV_ALWAYS("Thread %d: === Orchestrator Profiling: %lld tasks, total=%.3fus ===",
                thread_idx,
                (long long)p.submit_count,
                cycles_to_us(total));
            DEV_ALWAYS("Thread %d:   task+heap_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu",
                thread_idx,
                cycles_to_us(p.alloc_cycle),
                p.alloc_cycle * 100.0 / total,
                cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle),
                cycles_to_us(p.alloc_wait_cycle),
                (unsigned long long)p.alloc_atomic_count);
            DEV_ALWAYS("Thread %d:   sync_tensormap : %.3fus (%.1f%%)",
                thread_idx,
                cycles_to_us(p.sync_cycle),
                p.sync_cycle * 100.0 / total);
            DEV_ALWAYS("Thread %d:   lookup+dep     : %.3fus (%.1f%%)",
                thread_idx,
                cycles_to_us(p.lookup_cycle),
                p.lookup_cycle * 100.0 / total);
            DEV_ALWAYS("Thread %d:   tensormap_ins  : %.3fus (%.1f%%)",
                thread_idx,
                cycles_to_us(p.insert_cycle),
                p.insert_cycle * 100.0 / total);
            DEV_ALWAYS("Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%llu",
                thread_idx,
                cycles_to_us(p.params_cycle),
                p.params_cycle * 100.0 / total,
                (unsigned long long)p.params_atomic_count);
            DEV_ALWAYS("Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%llu",
                thread_idx,
                cycles_to_us(p.fanin_cycle),
                p.fanin_cycle * 100.0 / total,
                cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle),
                cycles_to_us(p.fanin_wait_cycle),
                (unsigned long long)p.fanin_atomic_count);
            DEV_ALWAYS("Thread %d:   avg/task       : %.3fus",
                thread_idx,
                p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0);

#if PTO2_TENSORMAP_PROFILING
            PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
            DEV_ALWAYS("Thread %d: === TensorMap Lookup Stats ===", thread_idx);
            DEV_ALWAYS("Thread %d:   lookups        : %llu, inserts: %llu",
                thread_idx,
                (unsigned long long)tp.lookup_count,
                (unsigned long long)tp.insert_count);
            DEV_ALWAYS("Thread %d:   chain walked   : total=%llu, avg=%.1f, max=%d",
                thread_idx,
                (unsigned long long)tp.lookup_chain_total,
                tp.lookup_count > 0 ? (double)tp.lookup_chain_total / tp.lookup_count : 0.0,
                tp.lookup_chain_max);
            DEV_ALWAYS("Thread %d:   overlap checks : %llu, hits=%llu (%.1f%%)",
                thread_idx,
                (unsigned long long)tp.overlap_checks,
                (unsigned long long)tp.overlap_hits,
                tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0);
#endif

#if PTO2_PROFILING
            // Write orchestrator summary to shared memory for host-side export (only if profiling enabled)
            if (runtime->enable_profiling) {
                AicpuOrchSummary orch_summary = {};
                orch_summary.start_time = orch_cycle_start;
                orch_summary.end_time = orch_cycle_end;
                orch_summary.sync_cycle = p.sync_cycle;
                orch_summary.alloc_cycle = p.alloc_cycle;
                orch_summary.params_cycle = p.params_cycle;
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
                    core_assignments_, core_count_per_thread_, sched_thread_num_, cores_total_num_);
                // Flush orchestrator's phase record buffer
                perf_aicpu_flush_phase_buffers(thread_idx);
            }
#endif

            // Coordinate orchestrator completion
            int32_t finished = orch_finished_count_.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (finished == orch_thread_num_) {
                // Last orchestrator: signal completion and trigger core transition
                pto2_rt_orchestration_done(rt);

                void* sm = runtime->get_pto2_gm_sm_ptr();
                PTO2SharedMemoryHeader* sm_header = static_cast<PTO2SharedMemoryHeader*>(sm);
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
                orchestrator_done_ = true;
                {
                    int32_t orch_err = 0;
                    void* sm = runtime->get_pto2_gm_sm_ptr();
                    if (sm) {
                        orch_err =
                            static_cast<PTO2SharedMemoryHeader*>(sm)->orch_error_code.load(std::memory_order_relaxed);
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
                    DEV_ALWAYS("Thread %d: orch_stage_end=%llu", thread_idx, (unsigned long long)orch_stage_end_ts);
#endif

                    // Wait for scheduler threads to acknowledge transition request
                    // All-orchestrator mode (sched_thread_num_ == 0): skip the wait
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
                DEV_ALWAYS("Thread %d: reassign, cost %.3fus (orch_idx=%d)",
                    thread_idx,
                    cycles_to_us(reassign_cycle_end - reassign_cycle_start),
                    orch_idx);
#endif
            } else {
                // Non-last orchestrator: wait for last orchestrator to finish setup
                if (orch_to_sched_) {
                    while (!transition_requested_.load(std::memory_order_acquire)) {
                        SPIN_WAIT_HINT();
                    }
                    while (!reassigned_.load(std::memory_order_acquire)) {
                        if (completed_.load(std::memory_order_acquire)) {
                            break;
                        }
                        SPIN_WAIT_HINT();
                    }
                }
            }
        }
#if PTO2_PROFILING
        uint64_t orch_end_ts = get_sys_cnt_aicpu();
        DEV_ALWAYS("Thread %d: orch_start=%llu orch_end=%llu orch_cost=%.3fus",
            thread_idx,
            (unsigned long long)orch_cycle_start,
            (unsigned long long)orch_end_ts,
            cycles_to_us(orch_end_ts - orch_cycle_start));
        if (pto2_submitted_tasks >= 0) {
            DEV_ALWAYS("PTO2 total submitted tasks = %d, already executed %d tasks",
                pto2_submitted_tasks,
                completed_tasks_.load(std::memory_order_acquire));
        }
#endif
        DEV_INFO("Thread %d: Orchestrator completed (orch_idx=%d)", thread_idx, orch_idx);
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
        int32_t completed = resolve_and_dispatch_pto2(runtime, thread_idx);
        DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
    }

    // Always shutdown AICore — even if completed_ was already true.
    // platform_deinit_aicore_regs is idempotent; orchestrator threads have
    // core_count_per_thread_ == 0 so they skip the loop harmlessly.
    {
        const int32_t* shutdown_cores = core_assignments_[thread_idx];
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
            // Clear the borrowed pointer in the orchestration SO before destroying
            // rt, so g_pto2_current_runtime never points to freed memory.
            if (orch_bind_runtime_ != nullptr) {
                orch_bind_runtime_(nullptr);
            }
            pto2_runtime_destroy(rt);
            dlclose(orch_so_handle_);
            unlink(orch_so_path_);
        }
    }

    return 0;
}

void AicpuExecutor::deinit(Runtime* runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all per-core execution state
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) {
        core_exec_states_[i] = {};
        core_exec_states_[i].executing_reg_task_id = AICPU_TASK_INVALID;
    }

    // Clear per-core dispatch payloads
    memset(s_pto2_payload_per_core, 0, sizeof(s_pto2_payload_per_core));

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
    orch_finished_count_.store(0, std::memory_order_release);

    // Reset core discovery state
    aic_count_ = 0;
    aiv_count_ = 0;

    regs_ = 0;
    orch_func_ = nullptr;
    orch_bind_runtime_ = nullptr;
    orch_args_cached_ = nullptr;
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

void AicpuExecutor::emergency_shutdown(Runtime* runtime) {
    DEV_WARN("Emergency shutdown: sending exit signal to all initialized cores");
    Handshake* all_handshakes = (Handshake*)runtime->workers;
    for (int32_t i = 0; i < cores_total_num_; i++) {
        Handshake* hank = &all_handshakes[i];
        hank->aicpu_regs_ready = 1;
        if (core_exec_states_[i].reg_addr != 0) {
            platform_deinit_aicore_regs(core_exec_states_[i].reg_addr);
        }
    }

    DEV_WARN("Emergency shutdown complete");
}

void AicpuExecutor::diagnose_stuck_state(
    Runtime* runtime, int32_t thread_idx, const int32_t* cur_thread_cores, int32_t core_num, Handshake* hank) {
    (void)runtime;
    PTO2SchedulerState* sched = &rt->scheduler;
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
        Handshake* h = &hank[core_id];
        const char* core_type_str = core_type_to_string(h->core_type);

        uint64_t reg_addr = core_exec_states_[core_id].reg_addr;
        uint64_t reg_val = read_reg(reg_addr, RegId::COND);
        int32_t reg_task_id = EXTRACT_TASK_ID(reg_val);
        int32_t reg_state = EXTRACT_TASK_STATE(reg_val);
        int32_t task_id = core_exec_states_[core_id].executing_reg_task_id;

        if (reg_state != TASK_FIN_STATE || task_id >= 0) {
            busy_cores++;
            if (task_id >= 0) {
                int32_t kernel_id = -1;
                if (rt && rt->sm_handle && core_exec_states_[core_id].executing_slot_state) {
                    int32_t diag_slot = static_cast<int32_t>(core_exec_states_[core_id].executing_subslot);
                    kernel_id = core_exec_states_[core_id].executing_slot_state->task->kernel_id[diag_slot];
                }
                DEV_ALWAYS(
                    "  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s), executing_reg_task_id=%d, "
                    "kernel_id=%d",
                    core_id,
                    core_type_str,
                    reg_val,
                    reg_task_id,
                    reg_state == TASK_FIN_STATE ? "FIN" : "ACK",
                    task_id,
                    kernel_id);
            } else {
                DEV_ALWAYS("  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s) but task_id not tracked",
                    core_id,
                    core_type_str,
                    reg_val,
                    reg_task_id,
                    reg_state == TASK_FIN_STATE ? "FIN" : "ACK");
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
extern "C" int32_t aicpu_execute(Runtime* runtime) {
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
