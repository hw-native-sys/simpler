#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
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

// Scheduler profiling helper
#ifndef PTO2_ORCH_PROFILING
#define PTO2_ORCH_PROFILING 1
#endif

#if PTO2_ORCH_PROFILING
// Accumulated nanoseconds per sub-step
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc) do { _t1 = get_sys_cnt_aicpu(); acc += (_t1 - _t0); _t0 = _t1; } while(0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#endif

// Device orchestration function signature (loaded via dlopen).
// The orchestration .so receives a PTO2Runtime* (with ops table populated)
// instead of a raw shared-memory pointer.
typedef void (*DeviceOrchestrationFunc)(PTO2Runtime* rt, uint64_t* args, int arg_count);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(uint64_t* args, int arg_count);

constexpr int MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int MAX_AIC_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD;
constexpr int MAX_AIV_PER_THREAD = PLATFORM_MAX_AIV_PER_THREAD;
constexpr int MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;

// Maximum tasks for ready queue (PTO2 mode uses shared memory task count)
constexpr int AICPU_MAX_READY_TASKS = 16384;
constexpr int AICPU_READY_MASK = AICPU_MAX_READY_TASKS - 1;
// One shard per scheduler thread: push to own shard (thread_idx % shards), pop own first + work stealing
constexpr int PTO2_READY_QUEUE_SHARDS = MAX_AICPU_THREADS - 1;

// Lightweight spinlock (avoids futex syscall overhead of std::mutex)
struct SpinLock {
    std::atomic<int> flag{0};
    void lock() { while (flag.exchange(1, std::memory_order_acquire) != 0) { PTO2_SPIN_PAUSE_LIGHT(); } }
    void unlock() { flag.store(0, std::memory_order_release); }
};

// Core information for discovery (with register address for fast dispatch)
struct CoreInfo {
    int worker_id;              // Index in runtime.workers[]
    uint32_t physical_core_id;  // Hardware physical core ID (from AICore)
    uint64_t reg_addr;          // Cached register address for fast access
    CoreType core_type;
};

struct AicpuExecutor {
    // ===== Thread management state =====
    std::atomic<int> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int thread_num_{0};
    int cores_total_num_{0};
    int thread_cores_num_{0};  // Cores per scheduler thread (0 for orchestrator when thread_num_==4)
    int core_count_per_thread_[MAX_AICPU_THREADS];  // Actual core count per thread
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // Core discovery arrays (with register addresses)
    CoreInfo aic_cores_[MAX_CORES_PER_THREAD];
    CoreInfo aiv_cores_[MAX_CORES_PER_THREAD];
    int aic_count_{0};
    int aiv_count_{0};

    // Fast lookup: core_id -> reg_addr (for register-based dispatch)
    uint64_t core_id_to_reg_addr_[MAX_CORES_PER_THREAD];

    // Platform register base address array (set via get_platform_regs())
    uint64_t regs_{0};

    // Track executing task_id per core (AICPU_TASK_INVALID = idle)
    int executing_task_ids_[MAX_CORES_PER_THREAD];

    // ===== 3 shards per type: push to own shard (thread_idx % 3), pop own first + work stealing =====
    SpinLock ready_queue_aic_lock_[PTO2_READY_QUEUE_SHARDS];
    int ready_queue_aic_[PTO2_READY_QUEUE_SHARDS][AICPU_MAX_READY_TASKS];
    int ready_queue_aic_head_[PTO2_READY_QUEUE_SHARDS]{0};
    int ready_queue_aic_tail_[PTO2_READY_QUEUE_SHARDS]{0};

    SpinLock ready_queue_aiv_lock_[PTO2_READY_QUEUE_SHARDS];
    int ready_queue_aiv_[PTO2_READY_QUEUE_SHARDS][AICPU_MAX_READY_TASKS];
    int ready_queue_aiv_head_[PTO2_READY_QUEUE_SHARDS]{0};
    int ready_queue_aiv_tail_[PTO2_READY_QUEUE_SHARDS]{0};

    // Task execution tracking
    std::atomic<int> completed_tasks_{0};
    std::atomic<int> total_tasks_{0};
    std::atomic<int> finished_count_{0};
    // Device orchestration: set by Thread 3 when graph is built; workers wait for it
    std::atomic<bool> orchestrator_done_{false};
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> pto2_init_complete_{false};  // init block finished; others wait for this
    std::atomic<int> next_scan_index_{0};
    std::atomic<bool> sm_header_ready_{false};  // Thread 3 sets after SM header init

    // Orchestrator ready queue pointers (set by Thread 3, read by scheduler threads)
    volatile int32_t* orch_ready_queue_{nullptr};
    volatile int32_t* orch_ready_tail_{nullptr};
    volatile int32_t* orch_ready_head_{nullptr};
    int32_t orch_ready_capacity_{0};

    // Orchestration SO handle - defer dlclose until all tasks complete
    void* orch_so_handle_{nullptr};
    char orch_so_path_[256]{};  // Path to orchestration SO file for cleanup

    // ===== Performance profiling state =====
    uint64_t dispatch_timestamps_[RUNTIME_MAX_WORKER];  // Per-core AICPU dispatch timestamp
    uint32_t core_dispatch_counts_[RUNTIME_MAX_WORKER]; // Per-core total dispatched task counter (for buffer management)

    // ===== Methods =====
    int init(Runtime* runtime);
    int handshake_all_cores(Runtime* runtime);
    void assign_cores_to_threads();
    int resolve_and_dispatch_pto2(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int run(Runtime* runtime);
    void deinit();
    void diagnose_stuck_state(Runtime* runtime, int thread_idx, const int* cur_thread_cores,
                              int core_num, Handshake* hank);

private:
    // Helper: enqueue a ready task to the appropriate shard with profiling
    inline void enqueue_ready_task_with_profiling(
        int32_t task_id,
        int32_t worker_type,
        int thread_idx
#if PTO2_ORCH_PROFILING
        , uint64_t& wait_counter,
        uint64_t& hold_counter
#endif
    );
};

static AicpuExecutor g_aicpu_executor;

// PTO2 device-mode state (shared memory view + per-task fanin refcount)
static constexpr int PTO2_MAX_SLOTS = PTO2_TASK_WINDOW_SIZE;
static int s_pto2_fanin_refcount[PTO2_MAX_SLOTS];
static volatile int32_t s_pto2_task_completed[PTO2_MAX_SLOTS];
static PTO2DispatchPayload s_pto2_payload_per_core[RUNTIME_MAX_WORKER];

// ===== AicpuExecutor Method Implementations =====

// Helper: enqueue a ready task to the appropriate shard with profiling
inline void AicpuExecutor::enqueue_ready_task_with_profiling(
    int32_t task_id,
    int32_t worker_type,
    int thread_idx
#if PTO2_ORCH_PROFILING
    , uint64_t& wait_counter,
    uint64_t& hold_counter
#endif
) {
    int my_shard = thread_idx % PTO2_READY_QUEUE_SHARDS;
#if PTO2_ORCH_PROFILING
    uint64_t _l0 = get_sys_cnt_aicpu(), _l1, _l2;
#endif

    if (worker_type == PTO2_WORKER_CUBE) {
        ready_queue_aic_lock_[my_shard].lock();
#if PTO2_ORCH_PROFILING
        _l1 = get_sys_cnt_aicpu();
#endif
        ready_queue_aic_[my_shard][ready_queue_aic_tail_[my_shard]++ & AICPU_READY_MASK] = task_id;
        ready_queue_aic_lock_[my_shard].unlock();
    } else {
        ready_queue_aiv_lock_[my_shard].lock();
#if PTO2_ORCH_PROFILING
        _l1 = get_sys_cnt_aicpu();
#endif
        ready_queue_aiv_[my_shard][ready_queue_aiv_tail_[my_shard]++ & AICPU_READY_MASK] = task_id;
        ready_queue_aiv_lock_[my_shard].unlock();
    }

#if PTO2_ORCH_PROFILING
    _l2 = get_sys_cnt_aicpu();
    wait_counter += (_l1 - _l0);
    hold_counter += (_l2 - _l1);
#endif
}

/**
 * Handshake with all cores and discover their types
 * Sets up register addresses for fast dispatch.
 */
int AicpuExecutor::handshake_all_cores(Runtime* runtime) {
    Handshake* all_hanks = (Handshake*)runtime->workers;
    cores_total_num_ = runtime->worker_count;

    aic_count_ = 0;
    aiv_count_ = 0;

    DEV_INFO("Handshaking with %d cores", cores_total_num_);

    // Step 1: Write per-core payload addresses and send handshake signal
    // task must be written BEFORE aicpu_ready so AICore sees it after waking up
    for (int i = 0; i < cores_total_num_; i++) {
        all_hanks[i].task = reinterpret_cast<uint64_t>(&s_pto2_payload_per_core[i]);
        all_hanks[i].aicpu_ready = 1;
    }

    // Step 2: Wait for all cores to respond, collect core type and register addresses
    for (int i = 0; i < cores_total_num_; i++) {
        Handshake* hank = &all_hanks[i];
        while (hank->aicore_done == 0) {
            // Spin wait for core to respond
        }

        CoreType type = hank->core_type;
        uint32_t physical_core_id = hank->physical_core_id;

        // Get register address using physical_core_id
        uint64_t* regs = reinterpret_cast<uint64_t*>(regs_);
        uint64_t reg_addr = regs[physical_core_id];

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

        // Initialize fast path registers
        if (reg_addr != 0) {
            write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
            write_reg(reg_addr, RegId::DATA_MAIN_BASE, 0);
        }
    }

    DEV_INFO("Core discovery complete: %d AIC, %d AIV", aic_count_, aiv_count_);
    return 0;
}

/**
 * Assign discovered cores to scheduler threads
 * (Aligned with host_build_graph mechanism)
 */
void AicpuExecutor::assign_cores_to_threads() {
    // When thread_num_ == 4: 3 schedulers + 1 orchestrator
    int scheduler_thread_num = (thread_num_ == 4) ? 3 : thread_num_;

    int aic_per_thread = aic_count_ / scheduler_thread_num;
    int aiv_per_thread = aiv_count_ / scheduler_thread_num;

    DEV_INFO("Assigning cores: %d AIC per thread, %d AIV per thread", aic_per_thread, aiv_per_thread);

    for (int t = 0; t < thread_num_; t++) {
        if (t >= scheduler_thread_num) {
            // Orchestrator thread: no cores
            core_count_per_thread_[t] = 0;
            DEV_INFO("Thread %d: orchestrator (0 cores)", t);
            continue;
        }

        int core_idx = 0;

        // Assign AIC cores
        int aic_start = t * aic_per_thread;
        for (int i = 0; i < aic_per_thread; i++) {
            int worker_id = aic_cores_[aic_start + i].worker_id;
            core_assignments_[t][core_idx++] = worker_id;
            DEV_INFO("Thread %d: assigned AIC worker_id=%d", t, worker_id);
        }

        // Assign AIV cores
        int aiv_start = t * aiv_per_thread;
        for (int i = 0; i < aiv_per_thread; i++) {
            int worker_id = aiv_cores_[aiv_start + i].worker_id;
            core_assignments_[t][core_idx++] = worker_id;
            DEV_INFO("Thread %d: assigned AIV worker_id=%d", t, worker_id);
        }

        core_count_per_thread_[t] = core_idx;

        DEV_INFO("Thread %d: total %d cores", t, core_idx);
    }

    thread_cores_num_ = aic_per_thread + aiv_per_thread;
}

int AicpuExecutor::init(Runtime* runtime) {
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

    // Read execution parameters from runtime
    thread_num_ = runtime->sche_cpu_num;
    if (thread_num_ == 0) thread_num_ = 1;

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Use handshake mechanism to discover cores (aligned with host_build_graph)
    int rc = handshake_all_cores(runtime);
    if (rc != 0) {
        DEV_ERROR("handshake_all_cores failed");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Dynamically assign cores to threads
    assign_cores_to_threads();

    if (cores_total_num_ > MAX_CORES_PER_THREAD * MAX_AICPU_THREADS) {
        DEV_ERROR("Total cores %d exceeds maximum", cores_total_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Initialize executing_task_ids_ to AICPU_TASK_INVALID (idle)
    for (int i = 0; i < MAX_CORES_PER_THREAD; i++) {
        executing_task_ids_[i] = AICPU_TASK_INVALID;
    }

    DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    // Initialize runtime execution state
    // Task count comes from PTO2 shared memory
    if (runtime->get_pto2_gm_sm_ptr()) {
        int32_t pto2_count = *static_cast<const volatile int32_t*>(runtime->get_pto2_gm_sm_ptr());
        total_tasks_.store(pto2_count > 0 ? pto2_count : 0, std::memory_order_release);
    } else {
        total_tasks_.store(0, std::memory_order_release);
    }
    completed_tasks_.store(0, std::memory_order_release);
    // Host orchestration: graph already built, no wait needed. Device orch: Thread 3 will set this.
    bool orch_on_host = runtime->get_orch_built_on_host();
    DEV_INFO("Init: orch_built_on_host=%d", orch_on_host ? 1 : 0);
    orchestrator_done_.store(orch_on_host, std::memory_order_release);

    // Initial ready tasks will be populated from PTO2 shared memory in resolve_and_dispatch_pto2
    for (int s = 0; s < PTO2_READY_QUEUE_SHARDS; s++) {
        ready_queue_aic_head_[s] = 0;
        ready_queue_aic_tail_[s] = 0;
        ready_queue_aiv_head_[s] = 0;
        ready_queue_aiv_tail_[s] = 0;
    }

    // Reset per-core dispatch timestamps and task counters
    for (int i = 0; i < RUNTIME_MAX_WORKER; i++) {
        dispatch_timestamps_[i] = 0;
        core_dispatch_counts_[i] = 0;
    }

    DEV_INFO("Init: PTO2 mode, task count from shared memory");

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int AicpuExecutor::shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
    (void)runtime;
    if (core_num == 0) return 0;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, core_num);

    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
        if (reg_addr != 0) {
            write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
            write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        } else {
            DEV_ERROR("Thread %d: Core %d has invalid register address", thread_idx, core_id);
        }
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

// Build PTO2DispatchPayload from PTO2TaskDescriptor.
static void build_pto2_payload(PTO2DispatchPayload* out, Runtime* runtime,
                               PTO2TaskDescriptor* task, PTO2TaskDescriptor* task_descriptors,
                               PTO2DepListEntry* dep_list_pool, int32_t window_size) {
    (void)task_descriptors;
    (void)dep_list_pool;
    (void)window_size;
    out->task_id = task->task_id;
    out->kernel_id = task->kernel_id;
    out->core_type = (task->worker_type == PTO2_WORKER_CUBE) ? CoreType::AIC : CoreType::AIV;
    out->function_bin_addr = runtime->get_function_bin_addr(task->kernel_id);
    int n = 0;

    for (int i = 0; i < task->param_count; i++) {
        if (task->params[i].type == PTOParamType::SCALAR) {
            out->args[n++] = task->params[i].scalar_value;
        } else {
            // Pass pointer to the Tensor (in task-owned storage), not the raw buffer address.
            // Kernels expect args[i] to be a Tensor* from which they read buffer.addr.
            task->params[i].tensor.data().update_start_offset();
            out->args[n++] = reinterpret_cast<uint64_t>(&task->params[i].tensor.data());
        }
    }

    out->num_args = n;
}

int AicpuExecutor::resolve_and_dispatch_pto2(Runtime* runtime, int thread_idx,
                                              const int* cur_thread_cores, int core_num) {
    DEV_INFO("Thread %d: resolve_and_dispatch_pto2 entry", thread_idx);

    void* sm_base = runtime->get_pto2_gm_sm_ptr();
    if (!sm_base) {
        DEV_ERROR("PTO2 dispatch: sm_base is null");
        return -1;
    }
    DEV_INFO("Thread %d: sm_base=%p", thread_idx, sm_base);

    // Device orchestration: wait for Thread 3 to initialize SM header
    if (thread_num_ == 4 && !runtime->get_orch_built_on_host()) {
        while (!sm_header_ready_.load(std::memory_order_acquire)) {
        }
    }

    PTO2SharedMemoryHeader* header = static_cast<PTO2SharedMemoryHeader*>(sm_base);
    DEV_INFO("Thread %d: header=%p, task_desc_offset=%d, dep_pool_offset=%d, window_size=%d",
             thread_idx, (void*)header, header->task_descriptors_offset,
             header->dep_list_pool_offset, header->task_window_size);

    PTO2TaskDescriptor* task_descriptors = reinterpret_cast<PTO2TaskDescriptor*>(
        static_cast<char*>(sm_base) + header->task_descriptors_offset);
    PTO2DepListEntry* dep_list_pool = reinterpret_cast<PTO2DepListEntry*>(
        static_cast<char*>(sm_base) + header->dep_list_pool_offset);
    DEV_INFO("Thread %d: task_descriptors=%p, dep_list_pool=%p",
             thread_idx, (void*)task_descriptors, (void*)dep_list_pool);

    int32_t window_size = header->task_window_size;
    if (window_size <= 0 || window_size > PTO2_MAX_SLOTS) window_size = PTO2_MAX_SLOTS;
    int32_t window_mask = window_size - 1;

    Handshake* hank = static_cast<Handshake*>(runtime->workers);
    DEV_INFO("Thread %d: hank=%p, window_size=%d",
             thread_idx, (void*)hank, window_size);

    // One-time init: clear refcount and completed arrays (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) {
        DEV_INFO("Thread %d: doing one-time init", thread_idx);
        std::memset(s_pto2_fanin_refcount, 0, sizeof(s_pto2_fanin_refcount));
        std::memset((void*)s_pto2_task_completed, 0, sizeof(s_pto2_task_completed));

        // Assign perf buffers to cores early so profiling captures all tasks
        // (total_tasks written to header later when orchestrator completes)
        if (runtime->enable_profiling) {
            perf_aicpu_init_profiling(runtime);
        }

        DEV_INFO("Thread %d: one-time init done", thread_idx);
        pto2_init_complete_.store(true, std::memory_order_release);
    } else {
        while (!pto2_init_complete_.load(std::memory_order_acquire)) {
        }
    }

    DEV_INFO("Thread %d: PTO2 dispatch starting with %d cores", thread_idx, core_num);
    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int idle_iterations = 0;
    const int MAX_IDLE_ITERATIONS = 800000;  // ~20s idle then scheduler gives up (avoid long hang)
    const int STALL_LOG_INTERVAL = 50000;  // DEV_ALWAYS every N idle iters to debug hang
    const int STALL_DUMP_READY_MAX = 8;
    const int STALL_DUMP_WAIT_MAX = 4;
    const int STALL_DUMP_CORE_MAX = 8;
    const int PROGRESS_VERBOSE_THRESHOLD = 10;  // log every completion for the first N tasks
    const int PROGRESS_LOG_INTERVAL = 25;       // log every N completions after threshold
    bool profiling_enabled = runtime->enable_profiling;
    int32_t last_reported_task_count = 0;

    // Scheduler profiling counters
#if PTO2_ORCH_PROFILING
    uint64_t sched_scan_cycle = 0;
    uint64_t sched_early_ready_cycle = 0;
    uint64_t sched_complete_cycle = 0;
    uint64_t sched_dispatch_cycle = 0;
    uint64_t sched_loop_count = 0;
    uint64_t sched_scan_ready_wait = 0, sched_scan_ready_hold = 0;
    uint64_t sched_early_ready_wait = 0, sched_early_ready_hold = 0;
    uint64_t sched_complete_ready_wait = 0, sched_complete_ready_hold = 0;
    uint64_t sched_dispatch_hit_wait = 0, sched_dispatch_hit_hold = 0;
    uint64_t sched_dispatch_miss_wait = 0, sched_dispatch_miss_hold = 0;
    uint64_t ready_pop_own = 0, ready_pop_steal = 0;
#endif
    // Fanout traversal statistics: how many downstream deps were notified after task completions
    uint64_t fanout_edges_notified = 0;
    int32_t fanout_max_degree = 0;

    while (true) {
#if PTO2_ORCH_PROFILING
        sched_loop_count++;
#endif
        CYCLE_COUNT_START();
        // Dynamic task_count (Thread 3 sets total_tasks_ when orchestration completes)
        int32_t task_count = total_tasks_.load(std::memory_order_acquire);
        bool orch_done = orchestrator_done_.load(std::memory_order_acquire);

        if (orch_done && task_count == 0) break;  // Empty graph
        if (task_count > 0 && completed_tasks_.load(std::memory_order_acquire) >= task_count) {
            bool all_cores_idle = true;
            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                uint64_t reg_addr = core_id_to_reg_addr_[core_id];
                uint64_t reg_val = read_reg(reg_addr, RegId::COND);
                int reg_state = EXTRACT_TASK_STATE(reg_val);

                if (reg_state != TASK_FIN_STATE || executing_task_ids_[core_id] != AICPU_TASK_INVALID) {
                    all_cores_idle = false; break;
                }
            }
            if (all_cores_idle && orch_done) break;
        }

        bool made_progress = false;

        // Process completed and dispatch FIRST to minimize Sched (dispatch→finish) latency.
        // Sched time = finish_ts - dispatch_ts; recording finish_ts here at loop start reduces
        // tail overhead (time from AICore done to AICPU recording finish).

        // Phase 1: Process completed tasks (register-based completion detection)
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            uint64_t reg_addr = core_id_to_reg_addr_[core_id];

            // Read task_id and state from COND register
            uint64_t reg_val = read_reg(reg_addr, RegId::COND);
            int reg_task_id = EXTRACT_TASK_ID(reg_val);
            int reg_state = EXTRACT_TASK_STATE(reg_val);

            // Only accept FIN state with matching task_id
            if (executing_task_ids_[core_id] != AICPU_TASK_INVALID &&
                reg_task_id == executing_task_ids_[core_id] &&
                reg_state == TASK_FIN_STATE) {

                PTO2DispatchPayload* payload = &s_pto2_payload_per_core[core_id];
                int32_t task_id = executing_task_ids_[core_id];
                executing_task_ids_[core_id] = AICPU_TASK_INVALID;

                // Write AICPU dispatch/finish timestamps into the PerfRecord
                if (profiling_enabled) {
                    Handshake* h = &hank[core_id];
                    uint64_t finish_ts = get_sys_cnt_aicpu();
                    PerfBuffer* perf_buf = (PerfBuffer*)h->perf_records_addr;
                    rmb();
                    uint32_t count = perf_buf->count;
                    if (count > 0) {
                        PerfRecord* record = &perf_buf->records[count - 1];
                        if (record->task_id == static_cast<uint32_t>(payload->task_id)) {
                            perf_aicpu_record_dispatch_and_finish_time(record,
                                                                        dispatch_timestamps_[core_id],
                                                                        finish_ts);
                        }
                    }
                }

                PTO2TaskDescriptor* pto2_task = &task_descriptors[task_id & window_mask];

                DEV_DEBUG("Thread %d: Core %d completed PTO2 task %d", thread_idx, core_id, task_id);

                // Mark completed (state=2), then snapshot fanout_head under the per-task spinlock.
                //
                // WHY THE LOCK IS REQUIRED (device orchestration / AICPU parallel mode):
                // The orchestrator (Thread 3) runs concurrently with the scheduler threads and
                // may still be adding consumers to this task's fanout list via
                // pto2_add_consumer_to_producer().  That function holds fanout_lock while it
                // (a) checks the completion state and (b) prepends to fanout_head.
                //
                // Without the lock here we have a TOCTOU race:
                //   1. Orch acquires lock, checks state=0 (task still running), plans insert.
                //   2. Task finishes; we store state=2 (RELEASE) but haven't acquired the lock.
                //   3. Orch inserts consumer X into fanout_head, releases lock.
                //   4. We read the OLD fanout_head (before X was inserted) → X is never woken.
                //
                // By acquiring the lock AFTER storing state=2 we guarantee mutual exclusion:
                //   • If Orch holds the lock first  → it writes fanout_head → we read it with X.
                //   • If we acquire the lock first  → Orch's subsequent lock-acquire sees state=2
                //     via the release/acquire pair and takes the early-return path, directly
                //     incrementing X's fanin_refcount instead of touching fanout_head.
                // Either way every consumer is accounted for exactly once.
                __atomic_store_n(&s_pto2_task_completed[task_id & window_mask], 2, __ATOMIC_RELEASE);
                pto2_fanout_lock(pto2_task);
                int32_t fanout_head = (int32_t)pto2_task->fanout_head;
                pto2_fanout_unlock(pto2_task);

                // Traverse fanout (no lock)
                //
                // SEQ_CST on the refcount increment and fanin_count load breaks the IRIW
                // (Independent Reads of Independent Writes) hazard with the orchestrator's
                // Step 5 / Step 5b:
                //
                //   Thread 0 (here):           Thread 3 (orchestrator Step 5/5b):
                //     fetch_add(refcount, SEQ_CST)   store(fanin_count=N, SEQ_CST)
                //     load(fanin_count,  SEQ_CST)    load(refcount,       SEQ_CST)
                //
                // On ARM (IRIW is architecturally allowed with ACQ/REL), both threads could
                // simultaneously read stale values — this thread sees fanin_count=0 and Step 5b
                // sees refcount<N — leaving the consumer stuck forever.
                //
                // With SEQ_CST, C++ guarantees a single total order over all SEQ_CST ops.
                // In any ordering the two writes fall, one of the two reads will observe the
                // other thread's write, ensuring the consumer is enqueued exactly once.
                int32_t fanout_len = 0;
                int32_t current = fanout_head;
                while (current > 0) {
                    fanout_len++;
                    PTO2DepListEntry* entry = &dep_list_pool[current];
                    int32_t consumer_id = entry->task_id;
                    int32_t consumer_slot = consumer_id & window_mask;
                    int prev = __atomic_fetch_add(&s_pto2_fanin_refcount[consumer_slot], 1, __ATOMIC_SEQ_CST);
                    PTO2TaskDescriptor* consumer_desc = &task_descriptors[consumer_slot];
                    int32_t fanin_count = __atomic_load_n(&consumer_desc->fanin_count, __ATOMIC_SEQ_CST);
                    if (prev + 1 == fanin_count) {
                        __atomic_store_n(&s_pto2_task_completed[consumer_slot], 1, __ATOMIC_RELEASE);
                        enqueue_ready_task_with_profiling(
                            consumer_id, consumer_desc->worker_type, thread_idx
#if PTO2_ORCH_PROFILING
                            , sched_complete_ready_wait, sched_complete_ready_hold
#endif
                        );
                    }
                    current = entry->next_offset;
                }
                fanout_edges_notified += fanout_len;
                if (fanout_len > fanout_max_degree) fanout_max_degree = fanout_len;

                cur_thread_tasks_in_flight--;
                cur_thread_completed++;
                made_progress = true;
                completed_tasks_.fetch_add(1, std::memory_order_release);
                // Debug: periodic progress (thread 0 only) to find which task hangs
                if (thread_idx == 0 && task_count > 0) {
                    int32_t c = completed_tasks_.load(std::memory_order_relaxed);
                    if (c <= PROGRESS_VERBOSE_THRESHOLD || c % PROGRESS_LOG_INTERVAL == 0 || c == task_count) {
                        DEV_ALWAYS("PTO2 progress: completed=%d total=%d last_task_id=%d (%.1f%%)",
                                  c, task_count, task_id, task_count > 0 ? 100.0 * c / task_count : 0.0);
                    }
                }
            }
        }
        CYCLE_COUNT_LAP(sched_complete_cycle);

        // Phase 2: Dispatch ready tasks to idle cores (register-based dispatch)
        if (cur_thread_tasks_in_flight < core_num) {
            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                uint64_t reg_addr = core_id_to_reg_addr_[core_id];
                uint64_t reg_val = read_reg(reg_addr, RegId::COND);
                int reg_state = EXTRACT_TASK_STATE(reg_val);
                if (reg_state == TASK_FIN_STATE && executing_task_ids_[core_id] == AICPU_TASK_INVALID) {
                    Handshake* h = &hank[core_id];
                    int32_t task_id = AICPU_TASK_INVALID;
#if PTO2_ORCH_PROFILING
                    bool found_task = false;
                    bool is_stolen = false;
#endif
                    int my_shard = thread_idx % PTO2_READY_QUEUE_SHARDS;
                    if (h->core_type == CoreType::AIC) {
                        for (int k = 0; k < PTO2_READY_QUEUE_SHARDS && task_id < 0; k++) {
                            int shard = (my_shard + k) % PTO2_READY_QUEUE_SHARDS;
#if PTO2_ORCH_PROFILING
                            uint64_t _l0 = get_sys_cnt_aicpu();
#endif
                            ready_queue_aic_lock_[shard].lock();
#if PTO2_ORCH_PROFILING
                            uint64_t _l1 = get_sys_cnt_aicpu();
#endif
                            if (ready_queue_aic_head_[shard] < ready_queue_aic_tail_[shard]) {
                                task_id = ready_queue_aic_[shard][ready_queue_aic_head_[shard]++ & AICPU_READY_MASK];
                                ready_queue_aic_lock_[shard].unlock();
#if PTO2_ORCH_PROFILING
                                uint64_t _l2 = get_sys_cnt_aicpu();
                                sched_dispatch_hit_wait += (_l1 - _l0);
                                sched_dispatch_hit_hold += (_l2 - _l1);
                                found_task = true;
                                is_stolen = (k != 0);
#endif
                                break;
                            }
                            ready_queue_aic_lock_[shard].unlock();
#if PTO2_ORCH_PROFILING
                            uint64_t _l2 = get_sys_cnt_aicpu();
                            sched_dispatch_miss_wait += (_l1 - _l0);
                            sched_dispatch_miss_hold += (_l2 - _l1);
#endif
                        }
                    } else {
                        for (int k = 0; k < PTO2_READY_QUEUE_SHARDS && task_id < 0; k++) {
                            int shard = (my_shard + k) % PTO2_READY_QUEUE_SHARDS;
#if PTO2_ORCH_PROFILING
                            uint64_t _l0 = get_sys_cnt_aicpu();
#endif
                            ready_queue_aiv_lock_[shard].lock();
#if PTO2_ORCH_PROFILING
                            uint64_t _l1 = get_sys_cnt_aicpu();
#endif
                            if (ready_queue_aiv_head_[shard] < ready_queue_aiv_tail_[shard]) {
                                task_id = ready_queue_aiv_[shard][ready_queue_aiv_head_[shard]++ & AICPU_READY_MASK];
                                ready_queue_aiv_lock_[shard].unlock();
#if PTO2_ORCH_PROFILING
                                uint64_t _l2 = get_sys_cnt_aicpu();
                                sched_dispatch_hit_wait += (_l1 - _l0);
                                sched_dispatch_hit_hold += (_l2 - _l1);
                                found_task = true;
                                is_stolen = (k != 0);
#endif
                                break;
                            }
                            ready_queue_aiv_lock_[shard].unlock();
#if PTO2_ORCH_PROFILING
                            uint64_t _l2 = get_sys_cnt_aicpu();
                            sched_dispatch_miss_wait += (_l1 - _l0);
                            sched_dispatch_miss_hold += (_l2 - _l1);
#endif
                        }
                    }
#if PTO2_ORCH_PROFILING
                    if (found_task) {
                        if (is_stolen) ready_pop_steal++; else ready_pop_own++;
                    }
#endif
                    if (task_id >= 0) {
                        PTO2TaskDescriptor* task = &task_descriptors[task_id & window_mask];
                        PTO2DispatchPayload* payload = &s_pto2_payload_per_core[core_id];
                        build_pto2_payload(payload, runtime, task, task_descriptors, dep_list_pool, window_size);
                        // Performance profiling: check if buffer needs switching
                        if (profiling_enabled) {
                            dispatch_timestamps_[core_id] = get_sys_cnt_aicpu();
                            if (core_dispatch_counts_[core_id] >= PLATFORM_PROF_BUFFER_SIZE) {
                                perf_aicpu_switch_buffer(runtime, core_id, thread_idx);
                                core_dispatch_counts_[core_id] = 0;
                            }
                            core_dispatch_counts_[core_id]++;
                        }

                        write_reg(reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(task_id + 1));
                        executing_task_ids_[core_id] = task_id;
                        cur_thread_tasks_in_flight++;
                        made_progress = true;
                        DEV_DEBUG("Thread %d: Dispatching PTO2 task %d to core %d", thread_idx, task_id, core_id);
                    }
                }
            }
        }
        CYCLE_COUNT_LAP(sched_dispatch_cycle);

        // Incremental scan: discover root tasks (fanin_count == 0)
        {
            int32_t visible = __atomic_load_n(&header->current_task_index, __ATOMIC_ACQUIRE);

            // Update perf header total_tasks if visible tasks have changed
            if (profiling_enabled && visible > 0 && visible != last_reported_task_count) {
                perf_aicpu_update_total_tasks(runtime, static_cast<uint32_t>(visible));

                DEV_INFO("Thread %d: Updated perf total_tasks to %d%s",
                            thread_idx, visible, orch_done ? " (final)" : "");

                last_reported_task_count = visible;
            }

            while (true) {
                int32_t idx = next_scan_index_.load(std::memory_order_acquire);
                if (idx >= visible) break;
                if (!next_scan_index_.compare_exchange_weak(idx, idx + 1,
                        std::memory_order_acq_rel, std::memory_order_acquire)) continue;

                int32_t slot = idx & window_mask;

                PTO2TaskDescriptor* t = &task_descriptors[slot];
                int32_t fanin_count = __atomic_load_n(&t->fanin_count, __ATOMIC_ACQUIRE);
                if (fanin_count == 0) {
                    // Mark as enqueued (state=1) to prevent double-enqueue
                    __atomic_store_n(&s_pto2_task_completed[slot], 1, __ATOMIC_RELEASE);
                    enqueue_ready_task_with_profiling(
                        idx, t->worker_type, thread_idx
#if PTO2_ORCH_PROFILING
                        , sched_scan_ready_wait, sched_scan_ready_hold
#endif
                    );
                    made_progress = true;
                }
            }
        }
        CYCLE_COUNT_LAP(sched_scan_cycle);

        // Early-ready drain: tasks whose deps were already met at submit time
        // (orchestrator detected all producers completed → pushed to orch_ready_queue_)
        if (orch_ready_queue_ != nullptr) {
            while (true) {
                int32_t head = __atomic_load_n(orch_ready_head_, __ATOMIC_ACQUIRE);
                int32_t tail = __atomic_load_n(orch_ready_tail_, __ATOMIC_ACQUIRE);
                if (head == tail) break;  // queue empty

                // CAS to claim this slot (multiple scheduler threads compete)
                if (!__atomic_compare_exchange_n(orch_ready_head_, &head, head + 1,
                        false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;

                int32_t task_id = orch_ready_queue_[head & (orch_ready_capacity_ - 1)];
                int32_t slot = task_id & window_mask;

                // CAS from 0 → 1 to claim enqueue rights (may already be enqueued by fanout path)
                int32_t expected = 0;
                if (!__atomic_compare_exchange_n(&s_pto2_task_completed[slot], &expected, 1,
                        false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;

                PTO2TaskDescriptor* t = &task_descriptors[slot];
                enqueue_ready_task_with_profiling(
                    task_id, t->worker_type, thread_idx
#if PTO2_ORCH_PROFILING
                    , sched_early_ready_wait, sched_early_ready_hold
#endif
                );
                made_progress = true;
            }
        }
        CYCLE_COUNT_LAP(sched_early_ready_cycle);

        if (!made_progress) {
            idle_iterations++;
            if (thread_idx == 0 && task_count > 0 && idle_iterations % STALL_LOG_INTERVAL == 0) {
                int32_t c = completed_tasks_.load(std::memory_order_relaxed);
                DEV_ALWAYS("PTO2 stall: no progress for %d iterations, completed=%d total=%d",
                           idle_iterations, c, task_count);
                // Scan all task slots to find truly stuck tasks
                // state=0: not yet completed (may be waiting for deps or ready but not enqueued)
                // state=1: enqueued in ready queue or dispatched to hardware
                // state=2: completed by Phase 1
                int cnt_ready = 0, cnt_waiting = 0, cnt_inflight = 0;
                for (int si = 0; si < task_count; si++) {
                    int32_t st  = __atomic_load_n(&s_pto2_task_completed[si], __ATOMIC_RELAXED);
                    int32_t rc  = __atomic_load_n(&s_pto2_fanin_refcount[si],  __ATOMIC_RELAXED);
                    int32_t fi  = __atomic_load_n(&task_descriptors[si].fanin_count, __ATOMIC_RELAXED);
                    int32_t kid = task_descriptors[si].kernel_id;
                    if (st == 2) continue; // Already done
                    if (st == 1) { cnt_inflight++; continue; }
                    // st == 0
                    if (rc >= fi) {
                        // Ready (all deps satisfied) but not enqueued — this is the real bug
                        cnt_ready++;
                        if (cnt_ready <= STALL_DUMP_READY_MAX) {
                            DEV_ALWAYS("  STUCK-READY  slot=%d kernel_id=%d refcount=%d fanin=%d",
                                       si, kid, rc, fi);
                        }
                    } else {
                        cnt_waiting++;
                        if (cnt_waiting <= STALL_DUMP_WAIT_MAX) {
                            DEV_ALWAYS("  STUCK-WAIT   slot=%d kernel_id=%d refcount=%d fanin=%d",
                                       si, kid, rc, fi);
                        }
                    }
                }
                DEV_ALWAYS("  scan result: stuck_ready=%d stuck_waiting=%d in_flight=%d",
                           cnt_ready, cnt_waiting, cnt_inflight);
                // Log this thread's dispatch state
                DEV_ALWAYS("  thread=%d cur_in_flight=%d core_num=%d",
                           thread_idx, cur_thread_tasks_in_flight, core_num);
                for (int ci = 0; ci < core_num && ci < STALL_DUMP_CORE_MAX; ci++) {
                    int cid = cur_thread_cores[ci];
                    Handshake* hh = &hank[cid];
                    int32_t hw_task_id = -1;
                    int32_t hw_kernel = -1;
                    if (hh->task != 0) {
                        const PTO2DispatchPayload* pl = reinterpret_cast<const PTO2DispatchPayload*>((uintptr_t)hh->task);
                        hw_task_id = pl->task_id;
                        hw_kernel  = pl->kernel_id;
                    }
                    DEV_ALWAYS("    core=%d status=%d task_id=%d kernel_id=%d",
                               cid, (int)hh->task_status, hw_task_id, hw_kernel);
                }
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("Thread %d: PTO2 timeout after %d idle iterations", thread_idx, idle_iterations);
                return -1;
            }
        } else {
            idle_iterations = 0;
        }
    }

#if PTO2_ORCH_PROFILING
    uint64_t sched_total =
        sched_scan_cycle + sched_early_ready_cycle + sched_complete_cycle + sched_dispatch_cycle;
    if (sched_total == 0) sched_total = 1;  // avoid div-by-zero
    double tasks_per_loop = sched_loop_count > 0 ? (double)cur_thread_completed / sched_loop_count : 0.0;

    // === Summary ===
    DEV_ALWAYS("Thread %d: === PTO2 Scheduler Summary ===", thread_idx);
    DEV_ALWAYS("Thread %d: completed=%d tasks in %.0fus (%llu loops, %.1f tasks/loop)",
        thread_idx, cur_thread_completed, cycles_to_us(sched_total),
        (unsigned long long)sched_loop_count, tasks_per_loop);

    // --- Phase Breakdown (execution order) ---
    DEV_ALWAYS("Thread %d: --- Phase Breakdown (execution order) ---", thread_idx);
    DEV_ALWAYS("Thread %d:   scan:        %8.0fus (%4.1f%%)",
        thread_idx, cycles_to_us(sched_scan_cycle), sched_scan_cycle * 100.0 / sched_total);
    DEV_ALWAYS("Thread %d:   early_ready: %8.0fus (%4.1f%%)  (deps already met at submit time)",
        thread_idx, cycles_to_us(sched_early_ready_cycle), sched_early_ready_cycle * 100.0 / sched_total);
    DEV_ALWAYS("Thread %d:   complete:    %8.0fus (%4.1f%%)  [fanout: edges=%llu, max_degree=%d, avg=%.1f]",
        thread_idx, cycles_to_us(sched_complete_cycle), sched_complete_cycle * 100.0 / sched_total,
        (unsigned long long)fanout_edges_notified, fanout_max_degree,
        cur_thread_completed > 0 ? (double)fanout_edges_notified / cur_thread_completed : 0.0);
    DEV_ALWAYS("Thread %d:   dispatch:    %8.0fus (%4.1f%%)  [steal: own=%llu, steal=%llu, pct=%.1f%%]",
        thread_idx, cycles_to_us(sched_dispatch_cycle), sched_dispatch_cycle * 100.0 / sched_total,
        (unsigned long long)ready_pop_own, (unsigned long long)ready_pop_steal,
        (ready_pop_own + ready_pop_steal) > 0 ? 100.0 * (double)ready_pop_steal / (double)(ready_pop_own + ready_pop_steal) : 0.0);

    // --- Lock Contention (ready_q) ---
    DEV_ALWAYS("Thread %d: --- Lock Contention (ready_q) ---", thread_idx);
    DEV_ALWAYS("Thread %d:   total:         wait=%5.0fus hold=%5.0fus",
        thread_idx,
        (double)cycles_to_us(sched_scan_ready_wait + sched_early_ready_wait + sched_complete_ready_wait + sched_dispatch_hit_wait + sched_dispatch_miss_wait),
        (double)cycles_to_us(sched_scan_ready_hold + sched_early_ready_hold + sched_complete_ready_hold + sched_dispatch_hit_hold + sched_dispatch_miss_hold));
    DEV_ALWAYS("Thread %d:   scan:          wait=%5.0fus hold=%5.0fus",
        thread_idx,
        (double)cycles_to_us(sched_scan_ready_wait), (double)cycles_to_us(sched_scan_ready_hold));
    DEV_ALWAYS("Thread %d:   early_ready:   wait=%5.0fus hold=%5.0fus",
        thread_idx,
        (double)cycles_to_us(sched_early_ready_wait), (double)cycles_to_us(sched_early_ready_hold));
    DEV_ALWAYS("Thread %d:   complete:      wait=%5.0fus hold=%5.0fus",
        thread_idx,
        (double)cycles_to_us(sched_complete_ready_wait), (double)cycles_to_us(sched_complete_ready_hold));
    DEV_ALWAYS("Thread %d:   dispatch:      wait=%5.0fus hold=%5.0fus",
        thread_idx,
        (double)cycles_to_us(sched_dispatch_hit_wait + sched_dispatch_miss_wait),
        (double)cycles_to_us(sched_dispatch_hit_hold + sched_dispatch_miss_hold));
    DEV_ALWAYS("Thread %d:     hit:         wait=%5.0fus hold=%5.0fus (dequeued task)",
        thread_idx,
        (double)cycles_to_us(sched_dispatch_hit_wait), (double)cycles_to_us(sched_dispatch_hit_hold));
    DEV_ALWAYS("Thread %d:     miss:        wait=%5.0fus hold=%5.0fus (empty queue)",
        thread_idx,
        (double)cycles_to_us(sched_dispatch_miss_wait), (double)cycles_to_us(sched_dispatch_miss_hold));
#endif

    // Flush performance buffers for cores managed by this thread
    if (profiling_enabled) {
        perf_aicpu_flush_buffers(runtime, thread_idx, cur_thread_cores, core_num);
    }

    return cur_thread_completed;
}

int AicpuExecutor::run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

    DEV_INFO("Thread %d: Start", thread_idx);

    const int* cur_thread_cores = core_assignments_[thread_idx];
    int my_cores = core_count_per_thread_[thread_idx];

    // Thread 3 when 4 AICPU threads: orchestrator (no cores)
    if (thread_num_ == 4 && thread_idx == 3) {
        if (runtime->get_orch_built_on_host()) {
            DEV_INFO("Thread 3: Host orchestration mode, no-op");
        } else {
            DEV_INFO("Thread 3: Device orchestration, loading SO via dlopen");

            // Get SO binary from runtime
            const void* so_data = runtime->get_device_orch_so_data();
            size_t so_size = runtime->get_device_orch_so_size();

            if (so_data == nullptr || so_size == 0) {
                DEV_ERROR("Thread 3: Device orchestration SO not set");
                return -1;
            }

            // /dev/shm, /tmp, and memfd are mounted noexec on real hardware
            // Try multiple paths that may allow execution on AICPU
            char so_path[256];
            bool file_created = false;

            // List of candidate paths to try (in order of preference)
            const char* candidate_dirs[] = {
                "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device",
                "/usr/lib64",
                "/lib64",
                "/var/tmp",
                "/tmp"  // Fallback, may not work on some AICPU configurations
            };
            const int num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

            for (int i = 0; i < num_candidates && !file_created; i++) {
                snprintf(so_path, sizeof(so_path), "%s/libdevice_orch_%d.so",
                         candidate_dirs[i], getpid());

                int fd = open(so_path, O_WRONLY | O_CREAT | O_TRUNC, 0755);
                if (fd < 0) {
                    DEV_INFO("Thread 3: Cannot create SO at %s (errno=%d), trying next path",
                             so_path, errno);
                    continue;
                }
                ssize_t written = write(fd, so_data, so_size);
                close(fd);
                if (written != static_cast<ssize_t>(so_size)) {
                    DEV_INFO("Thread 3: Cannot write SO to %s (errno=%d), trying next path",
                             so_path, errno);
                    unlink(so_path);
                    continue;
                }
                file_created = true;
                DEV_INFO("Thread 3: Created SO file at %s (%zu bytes)", so_path, so_size);
            }

            if (!file_created) {
                DEV_ERROR("Thread 3: Failed to create SO file in any candidate path");
                return -1;
            }

            // dlopen the SO
            dlerror();  // Clear any existing error before dlopen
            void* handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
            const char* dlopen_err = dlerror();
            if (handle == nullptr) {
                DEV_ERROR("Thread 3: dlopen failed: %s", dlopen_err ? dlopen_err : "unknown");
                unlink(so_path);
                return -1;
            }
            DEV_INFO("Thread 3: dlopen succeeded, handle=%p", handle);

            // Get the config function to read orchestration parameters
            dlerror();
            auto config_func = reinterpret_cast<DeviceOrchestrationConfigFunc>(
                dlsym(handle, "aicpu_orchestration_config"));

            // Get the orchestration entry function
            dlerror();
            DeviceOrchestrationFunc orch_func =
                reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, "aicpu_orchestration_entry"));
            const char* dlsym_error = dlerror();
            if (dlsym_error != nullptr) {
                DEV_ERROR("Thread 3: dlsym failed: %s", dlsym_error);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }
            if (orch_func == nullptr) {
                DEV_ERROR("Thread 3: dlsym returned NULL for aicpu_orchestration_entry");
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            uint64_t* args = runtime->get_orch_args();
            int arg_count = runtime->get_orch_arg_count();
            DEV_INFO("Thread 3: sm_ptr=%p, arg_count=%d", runtime->get_pto2_gm_sm_ptr(), arg_count);
            for (int i = 0; i < arg_count && i < 20; i++) {
                DEV_INFO("Thread 3: args[%d] = 0x%lx", i, args[i]);
            }

            // Read config from orchestration SO (or use defaults)
            uint64_t task_window_size = PTO2_TASK_WINDOW_SIZE;
            uint64_t dep_list_pool_size = PTO2_DEP_LIST_POOL_SIZE;
            uint64_t heap_size = PTO2_HEAP_SIZE;
            int expected_arg_count = 0;
            if (config_func) {
                PTO2OrchestrationConfig cfg = config_func(args, arg_count);
                expected_arg_count = cfg.expected_arg_count;
                DEV_INFO("Thread 3: Config: expected_args=%d", expected_arg_count);
            } else {
                DEV_INFO("Thread 3: No config function, using defaults");
            }

            if (expected_arg_count > 0 && arg_count < expected_arg_count) {
                DEV_ERROR("Thread 3: arg_count %d < expected %d", arg_count, expected_arg_count);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            // Get GM heap from runtime (dedicated field)
            void* sm_ptr = runtime->get_pto2_gm_sm_ptr();
            PTO2SharedMemoryHeader* header = static_cast<PTO2SharedMemoryHeader*>(sm_ptr);
            void* gm_heap = runtime->get_pto2_gm_heap_ptr();

            // Create shared memory handle and runtime (ops table populated inside)
            uint64_t sm_size = pto2_sm_calculate_size(task_window_size, dep_list_pool_size);
            PTO2SharedMemoryHandle* sm_handle =
                pto2_sm_create_from_buffer(sm_ptr, sm_size, task_window_size,
                                            heap_size, dep_list_pool_size);
            if (!sm_handle) {
                DEV_ERROR("Thread 3: Failed to create shared memory handle");
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            // Signal scheduler threads that SM header is initialized
            sm_header_ready_.store(true, std::memory_order_release);

            PTO2Runtime* rt = pto2_runtime_create_from_sm(PTO2_MODE_EXECUTE,
                                                            sm_handle, gm_heap, heap_size);
            if (!rt) {
                DEV_ERROR("Thread 3: Failed to create PTO2Runtime");
                pto2_sm_destroy(sm_handle);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            // Wait for scheduler's one-time init to complete (ensures memset has executed)
            while (!pto2_init_complete_.load(std::memory_order_acquire)) {
            }

            // Set orchestrator's aicpu parallel mode pointers
            uint64_t ws = header->task_window_size;
            if (ws == 0 || ws > PTO2_MAX_SLOTS) ws = PTO2_MAX_SLOTS;
            rt->orchestrator.aicpu_fanin_refcount = s_pto2_fanin_refcount;
            rt->orchestrator.aicpu_task_completed = s_pto2_task_completed;
            rt->orchestrator.aicpu_window_mask = ws - 1;

            // Expose orchestrator ready queue to scheduler threads
            orch_ready_queue_ = rt->orchestrator.orch_ready_queue;
            orch_ready_tail_ = &rt->orchestrator.orch_ready_tail;
            orch_ready_head_ = &rt->orchestrator.orch_ready_head;
            orch_ready_capacity_ = PTO2OrchestratorState::ORCH_READY_QUEUE_SIZE;

            // Call orchestration wrapped in outer scope (matches old PTO2_ORCHESTRATION behavior)
            DEV_ALWAYS("Thread 3: Calling aicpu_orchestration_entry from SO");
            uint64_t orch_cycle_start = get_sys_cnt_aicpu();
            PTO2_SCOPE(rt) { orch_func(rt, args, arg_count); }
            uint64_t orch_cycle_end = get_sys_cnt_aicpu();
            DEV_ALWAYS("Thread 3: aicpu_orchestration_entry returned, cost %.3fus",
                cycles_to_us(orch_cycle_end - orch_cycle_start));

            // Print orchestrator profiling data
#if PTO2_ORCH_PROFILING
            {
                PTO2OrchProfilingData p = pto2_orchestrator_get_profiling();
                uint64_t total = p.sync_cycle + p.alloc_cycle + p.params_cycle +
                                 p.lookup_cycle + p.heap_cycle + p.insert_cycle +
                                 p.fanin_cycle + p.finalize_cycle;
                DEV_ALWAYS("=== Orchestrator Profiling: %lld tasks, total=%.3fus ===",
                         (long long)p.submit_count, cycles_to_us(total));
                DEV_ALWAYS("  sync_tensormap : %.3fus (%.1f%%)", cycles_to_us(p.sync_cycle), p.sync_cycle * 100.0 / total);
                DEV_ALWAYS("  task_ring_alloc: %.3fus (%.1f%%)", cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total);
                DEV_ALWAYS("  param_copy     : %.3fus (%.1f%%)", cycles_to_us(p.params_cycle), p.params_cycle * 100.0 / total);
                DEV_ALWAYS("  lookup+dep     : %.3fus (%.1f%%)", cycles_to_us(p.lookup_cycle), p.lookup_cycle * 100.0 / total);
                DEV_ALWAYS("  heap_alloc     : %.3fus (%.1f%%)", cycles_to_us(p.heap_cycle), p.heap_cycle * 100.0 / total);
                DEV_ALWAYS("  tensormap_ins  : %.3fus (%.1f%%)", cycles_to_us(p.insert_cycle), p.insert_cycle * 100.0 / total);
                DEV_ALWAYS("  fanin+ready    : %.3fus (%.1f%%)", cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total);
                DEV_ALWAYS("  finalize+SM    : %.3fus (%.1f%%)", cycles_to_us(p.finalize_cycle), p.finalize_cycle * 100.0 / total);
                DEV_ALWAYS("  scope_end      : %.3fus", cycles_to_us(p.scope_end_cycle));
                DEV_ALWAYS("  avg/task       : %.3fus", cycles_to_us(total) / p.submit_count);

                PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
                DEV_ALWAYS("=== TensorMap Lookup Stats ===");
                DEV_ALWAYS("  lookups        : %llu, inserts: %llu",
                    (unsigned long long)tp.lookup_count, (unsigned long long)tp.insert_count);
                DEV_ALWAYS("  chain walked   : total=%llu, avg=%.1f, max=%d",
                    (unsigned long long)tp.lookup_chain_total,
                    tp.lookup_count > 0 ? (double)tp.lookup_chain_total / tp.lookup_count : 0.0,
                    tp.lookup_chain_max);
                DEV_ALWAYS("  overlap checks : %llu, hits=%llu (%.1f%%)",
                    (unsigned long long)tp.overlap_checks, (unsigned long long)tp.overlap_hits,
                    tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0);
            }
#endif

            // Signal orchestration complete in SM header (needs runtime alive)
            pto2_rt_orchestration_done(rt);

            // The orchestration .so no longer contains static output buffers
            // (heap is managed by the executor), so we can close immediately
            dlclose(handle);
            unlink(so_path);

            // Device mode: task count lives in PTO2 shared memory
            void* sm = runtime->get_pto2_gm_sm_ptr();
            PTO2SharedMemoryHeader* sm_header = static_cast<PTO2SharedMemoryHeader*>(sm);
            int32_t pto2_task_count = sm_header ? sm_header->current_task_index : 0;
            DEV_ALWAYS("PTO2 total submitted tasks = %d", pto2_task_count);
            total_tasks_.store(pto2_task_count, std::memory_order_release);
            orchestrator_done_.store(true, std::memory_order_release);
            DEV_INFO("Thread 3: Set orchestrator_done=true, waiting for scheduler threads");

            // Wait for all scheduler threads (0, 1, 2) to finish before destroying
            // runtime. Scheduler threads access TensorPool via orch_ready_queue_
            // and tensor.data() in build_pto2_payload — freeing early is use-after-free.
            while (finished_count_.load(std::memory_order_acquire) < thread_num_ - 1) {
                std::this_thread::yield();
            }
            DEV_INFO("Thread 3: All scheduler threads finished, destroying runtime");

            // Safe to destroy — no scheduler thread accesses runtime data anymore
            pto2_runtime_destroy(rt);
        }
        DEV_INFO("Thread 3: Orchestrator completed");
    } else {
        // Note: Handshake already completed in init() via handshake_all_cores()

        DEV_INFO("Thread %d: Starting PTO2 dispatch", thread_idx);
        int completed = resolve_and_dispatch_pto2(runtime, thread_idx, cur_thread_cores, my_cores);
        DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);

        auto rc = shutdown_aicore(runtime, thread_idx, cur_thread_cores, my_cores);
        if (rc != 0) {
            return rc;
        }

        DEV_INFO("Thread %d: Completed", thread_idx);
    }

    // Check if this is the last thread to finish
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        DEV_INFO("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return 0;
}

void AicpuExecutor::deinit() {
    // Cleanup runtime execution state
    for (int s = 0; s < PTO2_READY_QUEUE_SHARDS; s++) {
        ready_queue_aic_head_[s] = 0;
        ready_queue_aic_tail_[s] = 0;
        ready_queue_aiv_head_[s] = 0;
        ready_queue_aiv_tail_[s] = 0;
    }

    // Reset per-core dispatch timestamps and task counters
    for (int i = 0; i < RUNTIME_MAX_WORKER; i++) {
        dispatch_timestamps_[i] = 0;
        core_dispatch_counts_[i] = 0;
    }

    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);
    orchestrator_done_.store(false, std::memory_order_release);
    pto2_init_done_.store(false, std::memory_order_release);
    pto2_init_complete_.store(false, std::memory_order_release);
    next_scan_index_.store(0, std::memory_order_release);
    sm_header_ready_.store(false, std::memory_order_release);

    // Reset core discovery state
    aic_count_ = 0;
    aiv_count_ = 0;

    // Reset register-related state
    for (int i = 0; i < MAX_CORES_PER_THREAD; i++) {
        executing_task_ids_[i] = AICPU_TASK_INVALID;
        core_id_to_reg_addr_[i] = 0;
    }
    regs_ = 0;

    DEV_INFO("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: AicpuExecutor reset complete");
}

void AicpuExecutor::diagnose_stuck_state(Runtime* runtime, int thread_idx,
                                         const int* cur_thread_cores, int core_num,
                                         Handshake* hank) {
    (void)runtime;
    DEV_ALWAYS("========== DIAGNOSTIC REPORT: Thread %d ==========", thread_idx);

    int completed = completed_tasks_.load(std::memory_order_acquire);
    int total = total_tasks_.load(std::memory_order_acquire);
    DEV_ALWAYS("Progress: %d/%d tasks (%.1f%%)",
             completed, total, total > 0 ? completed * 100.0 / total : 0.0);

    int aic_ready = 0, aiv_ready = 0;
    for (int s = 0; s < PTO2_READY_QUEUE_SHARDS; s++) {
        aic_ready += ready_queue_aic_tail_[s] - ready_queue_aic_head_[s];
        aiv_ready += ready_queue_aiv_tail_[s] - ready_queue_aiv_head_[s];
    }
    DEV_ALWAYS("Ready Queues (3 shards, per-thread push + work-steal pop): AIC=%d, AIV=%d", aic_ready, aiv_ready);

    int busy_cores = 0;
    int idle_cores = 0;

    DEV_ALWAYS("Core Status:");
    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* h = &hank[core_id];
        const char* core_type_str = core_type_to_string(h->core_type);

        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
        uint64_t reg_val = read_reg(reg_addr, RegId::COND);
        int reg_task_id = EXTRACT_TASK_ID(reg_val);
        int reg_state = EXTRACT_TASK_STATE(reg_val);
        int task_id = executing_task_ids_[core_id];

        if (reg_state != TASK_FIN_STATE || task_id >= 0) {
            busy_cores++;
            if (task_id >= 0) {
                PTO2DispatchPayload* payload = &s_pto2_payload_per_core[core_id];
                DEV_ALWAYS("  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s), executing_task_id=%d, kernel_id=%d",
                        core_id, core_type_str, reg_val, reg_task_id,
                        reg_state == TASK_FIN_STATE ? "FIN" : "ACK",
                        payload->task_id, payload->kernel_id);
            } else {
                DEV_ALWAYS("  Core %d [%s, BUSY]: COND=0x%lx (reg_task_id=%d, reg_state=%s) but task_id not tracked",
                        core_id, core_type_str, reg_val, reg_task_id,
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
extern "C" int aicpu_execute(Runtime* runtime) {
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

    int rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        DEV_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit();
    }

    DEV_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
