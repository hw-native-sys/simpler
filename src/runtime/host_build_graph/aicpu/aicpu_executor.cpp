#include <atomic>
#include <cstdint>
#include <mutex>

#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "aicpu/performance_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"
#include "common/perf_profiling.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "runtime.h"

constexpr int MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;
constexpr int MAX_CORES = PLATFORM_MAX_CORES;

// Core information for discovery
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
    int thread_cores_num_{0};
    int aic_per_thread_{0};  // Fixed number of AIC cores per thread
    int aiv_per_thread_{0};  // Fixed number of AIV cores per thread
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // Core discovery arrays (space-time tradeoff: avoid sorting)
    CoreInfo aic_cores_[MAX_CORES_PER_THREAD];
    CoreInfo aiv_cores_[MAX_CORES_PER_THREAD];
    int aic_count_{0};
    int aiv_count_{0};

    // Fast lookup: core_id -> reg_addr
    uint64_t core_id_to_reg_addr_[MAX_CORES_PER_THREAD];

    // Platform register base address array (set via get_platform_regs())
    uint64_t regs_{0};

    // volatile required to prevent compiler from caching in registers during polling loops
    volatile int pending_task_ids_[MAX_CORES];  // Task waiting for ACK
    volatile int running_task_ids_[MAX_CORES];  // Task executing after ACK

    bool core_first_dispatch_[MAX_CORES];

    // Per-thread local ready queues
    int cur_ready_queue_aic_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];
    int cur_ready_queue_aiv_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];
    int cur_ready_queue_aic_head_[MAX_AICPU_THREADS];
    int cur_ready_queue_aic_tail_[MAX_AICPU_THREADS];
    int cur_ready_queue_aiv_head_[MAX_AICPU_THREADS];
    int cur_ready_queue_aiv_tail_[MAX_AICPU_THREADS];

    // ===== Task queue state =====
    std::mutex ready_queue_aic_mutex_;
    int ready_queue_aic_[RUNTIME_MAX_TASKS];
    std::atomic<int> ready_count_aic_{0};
    int ready_queue_aic_head_{0};  // Circular queue: read position (front)
    int ready_queue_aic_tail_{0};  // Circular queue: write position (back)

    std::mutex ready_queue_aiv_mutex_;
    int ready_queue_aiv_[RUNTIME_MAX_TASKS];
    std::atomic<int> ready_count_aiv_{0};
    int ready_queue_aiv_head_{0};  // Circular queue: read position (front)
    int ready_queue_aiv_tail_{0};  // Circular queue: write position (back)

    // Task execution tracking
    std::atomic<int> completed_tasks_{0};
    std::atomic<int> total_tasks_{0};
    std::atomic<int> finished_count_{0};

    // ===== Performance profiling state =====
    uint64_t dispatch_timestamps_[RUNTIME_MAX_WORKER];  // Per-core AICPU dispatch timestamp
    uint32_t core_dispatch_counts_[RUNTIME_MAX_WORKER]; // Per-core total dispatched task counter

    // ===== Methods =====
    int init(Runtime* runtime);
    int handshake_all_cores(Runtime* runtime);
    void assign_cores_to_threads();
    void classify_and_distribute_initial_tasks(Runtime* runtime);
    int resolve_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int run(Runtime* runtime);
    void deinit();
    void diagnose_stuck_state(
        Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num, Handshake* hank);

    // Helper functions (inline to avoid linker issues, not always_inline to preserve barriers)
    inline void resolve_task_dependencies(Task* task,
        Runtime& runtime,
        int* cur_ready_queue_aic,
        int& cur_aic_tail,
        int& cur_aic_ready_count,
        int* cur_ready_queue_aiv,
        int& cur_aiv_tail,
        int& cur_aiv_ready_count);

    inline bool try_dispatch_task(int core_id,
        uint64_t reg_addr,
        CoreType core_type,
        int thread_idx,
        int* local_queue,
        int& head,
        int& ready_count,
        bool profiling_enabled,
        Runtime& runtime);
};

static AicpuExecutor g_aicpu_executor;

// ===== Helper Function Implementations =====

// Resolve dependencies: decrement fanin and enqueue newly ready tasks
inline void AicpuExecutor::resolve_task_dependencies(Task* task,
    Runtime& runtime,
    int* cur_ready_queue_aic,
    int& cur_aic_tail,
    int& cur_aic_ready_count,
    int* cur_ready_queue_aiv,
    int& cur_aiv_tail,
    int& cur_aiv_ready_count) {
    for (int j = 0; j < task->fanout_count; j++) {
        int dep_id = task->fanout[j];
        Task* dep = runtime.get_task(dep_id);
        int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);

        if (prev_fanin == 1) {
            if (dep->core_type == CoreType::AIC) {
                if (cur_aic_ready_count < aic_per_thread_) {
                    cur_ready_queue_aic[cur_aic_tail] = dep_id;
                    cur_aic_tail = (cur_aic_tail + 1) % MAX_CORES_PER_THREAD;
                    cur_aic_ready_count++;
                } else {
                    std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                    ready_queue_aic_[ready_queue_aic_tail_] = dep_id;
                    ready_queue_aic_tail_ = (ready_queue_aic_tail_ + 1) % RUNTIME_MAX_TASKS;
                    ready_count_aic_.fetch_add(1, std::memory_order_release);
                }
            } else {
                if (cur_aiv_ready_count < aiv_per_thread_) {
                    cur_ready_queue_aiv[cur_aiv_tail] = dep_id;
                    cur_aiv_tail = (cur_aiv_tail + 1) % MAX_CORES_PER_THREAD;
                    cur_aiv_ready_count++;
                } else {
                    std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                    ready_queue_aiv_[ready_queue_aiv_tail_] = dep_id;
                    ready_queue_aiv_tail_ = (ready_queue_aiv_tail_ + 1) % RUNTIME_MAX_TASKS;
                    ready_count_aiv_.fetch_add(1, std::memory_order_release);
                }
            }
        }
    }
}

// Try to dispatch a task from thread-local queue to a core
inline bool AicpuExecutor::try_dispatch_task(int core_id,
    uint64_t reg_addr,
    CoreType core_type,
    int thread_idx,
    int* local_queue,
    int& head,
    int& ready_count,
    bool profiling_enabled,
    Runtime& runtime) {
    if (ready_count <= 0) {
        return false;
    }

    // Dequeue task from thread-local queue
    int task_id = local_queue[head];
    head = (head + 1) % MAX_CORES_PER_THREAD;
    ready_count--;

    // Profiling: buffer switch check
    if (profiling_enabled) {
        core_dispatch_counts_[core_id]++;
        if (core_dispatch_counts_[core_id] >= PLATFORM_PROF_BUFFER_SIZE - 1) {
            perf_aicpu_switch_buffer(&runtime, core_id, thread_idx);
            core_dispatch_counts_[core_id] = 0;
        }
    }

    const char* core_type_str = (core_type == CoreType::AIC) ? "AIC" : "AIV";
    LOG_INFO("Thread %d: Dispatching %s task %d to core %d (running_id=%d)",
        thread_idx,
        core_type_str,
        task_id,
        core_id,
        running_task_ids_[core_id]);

    // Set state before writing register to avoid race with AICore ACK
    pending_task_ids_[core_id] = task_id;

    write_reg(reg_addr, RegId::DATA_MAIN_BASE, static_cast<uint64_t>(task_id + 1));

    return true;
}

// ===== AicpuExecutor Method Implementations =====

int AicpuExecutor::init(Runtime* runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    LOG_INFO("AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Read execution parameters from runtime
    thread_num_ = runtime->sche_cpu_num;

    // Simplified defensive check
    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid thread_num: %d (valid range: 1-%d)", thread_num_, MAX_AICPU_THREADS);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Perform core discovery: handshake with all cores and collect core type information
    int rc = handshake_all_cores(runtime);
    if (rc != 0) {
        LOG_ERROR("Core discovery failed");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Calculate cores per thread
    thread_cores_num_ = cores_total_num_ / thread_num_;

    LOG_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    for (int i = 0; i < cores_total_num_; i++) {
        pending_task_ids_[i] = AICPU_TASK_INVALID;
        running_task_ids_[i] = AICPU_TASK_INVALID;
        core_first_dispatch_[i] = true;
    }

    assign_cores_to_threads();
    classify_and_distribute_initial_tasks(runtime);

    total_tasks_.store(runtime->get_task_count(), std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);

    for (int i = 0; i < RUNTIME_MAX_WORKER; i++) {
        dispatch_timestamps_[i] = 0;
        core_dispatch_counts_[i] = 0;
    }
    if (runtime->enable_profiling) {
        perf_aicpu_init_profiling(runtime);
    }

    init_done_.store(true, std::memory_order_release);
    LOG_INFO("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Handshake with all AICore workers and discover core types
 *
 * This function performs centralized handshaking with all cores and collects
 * their type information. By doing this in a single thread, we avoid redundant
 * handshakes and enable dynamic core assignment.
 *
 * Protocol:
 * 1. Send aicpu_ready=1 to all cores
 * 2. Wait for each core's aicore_done response
 * 3. Read core_type reported by each core
 * 4. Classify cores into aic_cores_[] and aiv_cores_[] arrays
 *
 * @param runtime Runtime pointer
 * @return 0 on success, -1 on failure
 */
int AicpuExecutor::handshake_all_cores(Runtime* runtime) {
    Handshake* all_hanks = (Handshake*)runtime->workers;
    cores_total_num_ = runtime->worker_count;

    if (cores_total_num_ == 0) {
        LOG_ERROR("worker_count is 0, no cores to handshake");
        return -1;
    }

    // Simplified defensive check
    if (cores_total_num_ > MAX_CORES_PER_THREAD) {
        LOG_ERROR("Total cores %d exceeds maximum %d", cores_total_num_, MAX_CORES_PER_THREAD);
        return -1;
    }

    aic_count_ = 0;
    aiv_count_ = 0;

    LOG_INFO("Core Discovery: Handshaking with %d cores", cores_total_num_);

    // Step 1: Send handshake signal to all cores
    for (int i = 0; i < cores_total_num_; i++) {
        all_hanks[i].aicpu_ready = 1;
    }

    // Step 2: Wait for all cores to respond and collect core type information
    for (int i = 0; i < cores_total_num_; i++) {
        Handshake* hank = &all_hanks[i];

        // Wait for aicore_done signal
        while (hank->aicore_done == 0) {
            // Busy wait for core response
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
        } else if (type == CoreType::AIV) {
            aiv_cores_[aiv_count_].worker_id = i;
            aiv_cores_[aiv_count_].physical_core_id = physical_core_id;
            aiv_cores_[aiv_count_].reg_addr = reg_addr;
            aiv_cores_[aiv_count_].core_type = type;
            aiv_count_++;
        } else {
            LOG_ERROR("Unknown core type from core %d", i);
            return -1;
        }

        core_id_to_reg_addr_[i] = reg_addr;

        LOG_INFO("  Core %d: type=%s, physical_id=%u, reg_addr=0x%lx",
            i,
            core_type_to_string(type),
            physical_core_id,
            reg_addr);

        if (reg_addr != 0) {
            write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_OPEN);
            write_reg(reg_addr, RegId::DATA_MAIN_BASE, 0);
        }
    }

    LOG_INFO("Discovery complete: AIC=%d, AIV=%d, Total=%d", aic_count_, aiv_count_, cores_total_num_);
    return 0;
}

// Assign discovered cores to threads (requires even distribution)
void AicpuExecutor::assign_cores_to_threads() {
    if (aic_count_ % thread_num_ != 0) {
        LOG_ERROR("AIC cores (%d) cannot be evenly distributed to %d threads", aic_count_, thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return;
    }

    if (aiv_count_ % thread_num_ != 0) {
        LOG_ERROR("AIV cores (%d) cannot be evenly distributed to %d threads", aiv_count_, thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return;
    }

    aic_per_thread_ = aic_count_ / thread_num_;
    aiv_per_thread_ = aiv_count_ / thread_num_;

    LOG_INFO("Core Assignment: %d AIC/thread, %d AIV/thread", aic_per_thread_, aiv_per_thread_);

    for (int t = 0; t < thread_num_; t++) {
        int core_idx = 0;

        // Assign AIC cores to this thread
        int aic_start = t * aic_per_thread_;
        int aic_end = (t + 1) * aic_per_thread_;
        for (int i = aic_start; i < aic_end; i++) {
            core_assignments_[t][core_idx++] = aic_cores_[i].worker_id;
        }

        // Assign AIV cores to this thread
        int aiv_start = t * aiv_per_thread_;
        int aiv_end = (t + 1) * aiv_per_thread_;
        for (int i = aiv_start; i < aiv_end; i++) {
            core_assignments_[t][core_idx++] = aiv_cores_[i].worker_id;
        }

        char log_buffer[256];
        int offset = 0;

        offset += snprintf(
            log_buffer + offset, sizeof(log_buffer) - offset, "Thread %d: assigned %d cores - AIC[", t, core_idx);

        for (int i = 0; i < aic_per_thread_; i++) {
            if (i > 0) offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ",");
            offset +=
                snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "%d", aic_cores_[aic_start + i].worker_id);
        }

        offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "] AIV[");

        for (int i = 0; i < aiv_per_thread_; i++) {
            if (i > 0) offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ",");
            offset +=
                snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "%d", aiv_cores_[aiv_start + i].worker_id);
        }

        offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "]");

        LOG_INFO("%s", log_buffer);
    }
}

// Classify and distribute initial ready tasks to thread-local and shared queues
void AicpuExecutor::classify_and_distribute_initial_tasks(Runtime* runtime) {
    ready_queue_aic_head_ = 0;
    ready_queue_aic_tail_ = 0;
    ready_queue_aiv_head_ = 0;
    ready_queue_aiv_tail_ = 0;
    int initial_ready[RUNTIME_MAX_TASKS];
    int initial_count = runtime->get_initial_ready_tasks(initial_ready);

    LOG_INFO("Init: Found %d initially ready tasks", initial_count);

    // Classify initial ready tasks by type
    int initial_aic[RUNTIME_MAX_TASKS];
    int initial_aiv[RUNTIME_MAX_TASKS];
    int initial_aic_count = 0;
    int initial_aiv_count = 0;

    for (int i = 0; i < initial_count; i++) {
        Task* task = runtime->get_task(initial_ready[i]);
        if (task->core_type == CoreType::AIC) {
            initial_aic[initial_aic_count++] = initial_ready[i];
        } else {
            initial_aiv[initial_aiv_count++] = initial_ready[i];
        }
    }

    LOG_INFO("Init: Initial ready tasks by type: AIC=%d, AIV=%d", initial_aic_count, initial_aiv_count);

    for (int t = 0; t < MAX_AICPU_THREADS; t++) {
        cur_ready_queue_aic_head_[t] = 0;
        cur_ready_queue_aic_tail_[t] = 0;
        cur_ready_queue_aiv_head_[t] = 0;
        cur_ready_queue_aiv_tail_[t] = 0;
    }

    int aic_shared_count = 0;
    int thread_idx = 0;
    for (int i = 0; i < initial_aic_count; i++) {
        int task_id = initial_aic[i];

        int head = cur_ready_queue_aic_head_[thread_idx];
        int tail = cur_ready_queue_aic_tail_[thread_idx];
        int cur_size = (tail - head + MAX_CORES_PER_THREAD) % MAX_CORES_PER_THREAD;

        if (cur_size < aic_per_thread_) {
            cur_ready_queue_aic_[thread_idx][tail] = task_id;
            cur_ready_queue_aic_tail_[thread_idx] = (tail + 1) % MAX_CORES_PER_THREAD;
            LOG_INFO("Init: AIC task %d -> Thread %d local queue (size=%d)", task_id, thread_idx, cur_size + 1);
        } else {
            ready_queue_aic_[ready_queue_aic_tail_] = task_id;
            ready_queue_aic_tail_ = (ready_queue_aic_tail_ + 1) % RUNTIME_MAX_TASKS;
            aic_shared_count++;
        }

        thread_idx = (thread_idx + 1) % thread_num_;
    }
    ready_count_aic_.store(aic_shared_count, std::memory_order_release);

    int aiv_shared_count = 0;
    thread_idx = 0;
    for (int i = 0; i < initial_aiv_count; i++) {
        int task_id = initial_aiv[i];

        int head = cur_ready_queue_aiv_head_[thread_idx];
        int tail = cur_ready_queue_aiv_tail_[thread_idx];
        int cur_size = (tail - head + MAX_CORES_PER_THREAD) % MAX_CORES_PER_THREAD;

        if (cur_size < aiv_per_thread_) {
            cur_ready_queue_aiv_[thread_idx][tail] = task_id;
            cur_ready_queue_aiv_tail_[thread_idx] = (tail + 1) % MAX_CORES_PER_THREAD;
            LOG_INFO("Init: AIV task %d -> Thread %d local queue (size=%d)", task_id, thread_idx, cur_size + 1);
        } else {
            ready_queue_aiv_[ready_queue_aiv_tail_] = task_id;
            ready_queue_aiv_tail_ = (ready_queue_aiv_tail_ + 1) % RUNTIME_MAX_TASKS;
            aiv_shared_count++;
        }

        thread_idx = (thread_idx + 1) % thread_num_;
    }
    ready_count_aiv_.store(aiv_shared_count, std::memory_order_release);

    LOG_INFO("Init: Task distribution complete - AIC: %d in local queues, %d in shared queue",
        initial_aic_count - aic_shared_count,
        aic_shared_count);
    LOG_INFO("Init: Task distribution complete - AIV: %d in local queues, %d in shared queue",
        initial_aiv_count - aiv_shared_count,
        aiv_shared_count);

    for (int t = 0; t < thread_num_; t++) {
        int aic_size =
            (cur_ready_queue_aic_tail_[t] - cur_ready_queue_aic_head_[t] + MAX_CORES_PER_THREAD) % MAX_CORES_PER_THREAD;
        int aiv_size =
            (cur_ready_queue_aiv_tail_[t] - cur_ready_queue_aiv_head_[t] + MAX_CORES_PER_THREAD) % MAX_CORES_PER_THREAD;
        LOG_INFO("Init: Thread %d local queues - AIC: %d tasks, AIV: %d tasks", t, aic_size, aiv_size);
    }
}

/**
 * Shutdown AICore - Send quit signal to all AICore kernels
 */
int AicpuExecutor::shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    LOG_INFO("Thread %d: Shutting down %d cores", thread_idx, thread_cores_num_);

    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        LOG_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);

        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
        if (reg_addr != 0) {
            write_reg(reg_addr, RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
            write_reg(reg_addr, RegId::FAST_PATH_ENABLE, REG_SPR_FAST_PATH_CLOSE);
        } else {
            LOG_ERROR("Thread %d: Core %d has invalid register address", thread_idx, core_id);
        }
    }
    LOG_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

/**
 * Resolve dependencies and dispatch tasks using fast-path scheduling
 */
int AicpuExecutor::resolve_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
    Handshake* hank = (Handshake*)runtime.workers;

    LOG_INFO("Thread %d: Starting execution with %d cores", thread_idx, core_num);

    int cur_thread_completed = 0;
    int task_count = total_tasks_.load(std::memory_order_acquire);

    // Timeout detection using idle iteration counting
    int idle_iterations = 0;
    const int MAX_IDLE_ITERATIONS = 50000000;
    const int WARN_INTERVAL = 1000000;
    bool made_progress = false;

    int verification_warning_count = 0;
    const int MAX_VERIFICATION_WARNINGS = 10;
    bool profiling_enabled = runtime.enable_profiling;

    // Extract array pointers as local variables for better readability and performance
    int* cur_ready_queue_aic = cur_ready_queue_aic_[thread_idx];
    int* cur_ready_queue_aiv = cur_ready_queue_aiv_[thread_idx];

    // Initialize local circular queue pointers from member variables (set by init())
    // After this point, only use local variables for lock-free performance
    int cur_aic_head = cur_ready_queue_aic_head_[thread_idx];
    int cur_aic_tail = cur_ready_queue_aic_tail_[thread_idx];
    int cur_aiv_head = cur_ready_queue_aiv_head_[thread_idx];
    int cur_aiv_tail = cur_ready_queue_aiv_tail_[thread_idx];

    // Calculate initial queue sizes
    int cur_aic_ready_count = (cur_aic_tail - cur_aic_head + MAX_CORES_PER_THREAD) % MAX_CORES_PER_THREAD;
    int cur_aiv_ready_count = (cur_aiv_tail - cur_aiv_head + MAX_CORES_PER_THREAD) % MAX_CORES_PER_THREAD;

    LOG_INFO(
        "Thread %d: Initial state - local queue: %d AIC, %d AIV", thread_idx, cur_aic_ready_count, cur_aiv_ready_count);

    // Initialize dispatch timestamps for all cores
    uint64_t dispatch_start_time = get_sys_cnt_aicpu();
    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        dispatch_timestamps_[core_id] = dispatch_start_time;
    }

    // Main execution loop with unified scheduling
    while (true) {
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            uint64_t reg_addr = core_id_to_reg_addr_[core_id];
            Handshake* h = &hank[core_id];

            uint64_t reg_val = read_reg(reg_addr, RegId::COND);
            int reg_task_id = EXTRACT_TASK_ID(reg_val);
            int reg_state = EXTRACT_TASK_STATE(reg_val);

            // Case 1: Pending task finished directly
            if (reg_task_id == pending_task_ids_[core_id] &&
                reg_state == TASK_FIN_STATE) {

                LOG_INFO("Thread %d: Core %d completed task %d (running_id=%d)",
                         thread_idx, core_id, pending_task_ids_[core_id], running_task_ids_[core_id]);
                

                int completed_task_id = pending_task_ids_[core_id];

                // Profiling
                if (profiling_enabled) {
                    uint64_t finish_ts = get_sys_cnt_aicpu();
                    PerfBuffer* perf_buf = (PerfBuffer*)h->perf_records_addr;
                    rmb();
                    uint32_t count = perf_buf->count;
                    if (count > 0) {
                        PerfRecord* record = &perf_buf->records[count - 1];
                        if (record->task_id == static_cast<uint32_t>(completed_task_id)) {
                            perf_aicpu_record_dispatch_and_finish_time(
                                record, dispatch_timestamps_[core_id], finish_ts);
                        }
                    }
                    dispatch_timestamps_[core_id] = get_sys_cnt_aicpu();
                }

                cur_thread_completed++;
                completed_tasks_.fetch_add(1, std::memory_order_release);

                int prev_running_id = running_task_ids_[core_id];
                pending_task_ids_[core_id] = AICPU_TASK_INVALID;
                running_task_ids_[core_id] = AICPU_TASK_INVALID;

                // Try dispatch BEFORE resolve_dependencies
                // This allows the core to start next task immediately
                bool dispatched = false;
                if (h->core_type == CoreType::AIC && cur_aic_ready_count > 0) {
                    dispatched = try_dispatch_task(core_id,
                        reg_addr,
                        CoreType::AIC,
                        thread_idx,
                        cur_ready_queue_aic,
                        cur_aic_head,
                        cur_aic_ready_count,
                        profiling_enabled,
                        runtime);
                } else if (h->core_type == CoreType::AIV && cur_aiv_ready_count > 0) {
                    dispatched = try_dispatch_task(core_id,
                        reg_addr,
                        CoreType::AIV,
                        thread_idx,
                        cur_ready_queue_aiv,
                        cur_aiv_head,
                        cur_aiv_ready_count,
                        profiling_enabled,
                        runtime);
                }

                // Resolve old running task dependencies (if exists)
                // When pending task FINs directly, the running task was implicitly
                // completed (AICore overwrote COND before we could read its FIN).
                // Count it here to avoid losing completion.
                if (prev_running_id != AICPU_TASK_INVALID) {
                    cur_thread_completed++;
                    completed_tasks_.fetch_add(1, std::memory_order_release);

                    Task* prev_running_task = runtime.get_task(prev_running_id);
                    resolve_task_dependencies(prev_running_task,
                        runtime,
                        cur_ready_queue_aic,
                        cur_aic_tail,
                        cur_aic_ready_count,
                        cur_ready_queue_aiv,
                        cur_aiv_tail,
                        cur_aiv_ready_count);

                    LOG_INFO("Thread %d: Core %d resolved old running task %d",
                             thread_idx, core_id, prev_running_id);
                }

                Task* task = runtime.get_task(completed_task_id);
                resolve_task_dependencies(task,
                    runtime,
                    cur_ready_queue_aic,
                    cur_aic_tail,
                    cur_aic_ready_count,
                    cur_ready_queue_aiv,
                    cur_aiv_tail,
                    cur_aiv_ready_count);

                made_progress = true;

                // Update timestamp if didn't dispatch (try_dispatch_task updates it if dispatched)
                if (!dispatched && profiling_enabled) {
                    dispatch_timestamps_[core_id] = get_sys_cnt_aicpu();
                }
            }

            // Case 2: Pending task received ACK
            else if (reg_task_id == pending_task_ids_[core_id] &&
                     reg_state == TASK_ACK_STATE) {

                LOG_INFO("Thread %d: Core %d ACKed task %d (running_id=%d)",
                         thread_idx, core_id, pending_task_ids_[core_id], running_task_ids_[core_id]);

                int prev_running_id = running_task_ids_[core_id];

                // Move pending to running
                running_task_ids_[core_id] = pending_task_ids_[core_id];
                pending_task_ids_[core_id] = AICPU_TASK_INVALID;
                made_progress = true;

                // When pending task ACKs, the old running task was implicitly
                // completed (AICore overwrote COND before we could read its FIN).
                // Count it here to avoid losing completion.
                if (prev_running_id != AICPU_TASK_INVALID) {
                    cur_thread_completed++;
                    completed_tasks_.fetch_add(1, std::memory_order_release);

                    Task* prev_running_task = runtime.get_task(prev_running_id);
                    resolve_task_dependencies(prev_running_task,
                        runtime,
                        cur_ready_queue_aic,
                        cur_aic_tail,
                        cur_aic_ready_count,
                        cur_ready_queue_aiv,
                        cur_aiv_tail,
                        cur_aiv_ready_count);

                    LOG_INFO("Thread %d: Core %d resolved old running task %d",
                             thread_idx, core_id, prev_running_id);
                }

                // Core can accept new task now (pipeline!)
                // Continue to Case 4 to dispatch next task
            }

            // Case 3: Running task finished
            else if (reg_task_id == running_task_ids_[core_id] &&
                     reg_state == TASK_FIN_STATE) {

                LOG_INFO("Thread %d: Core %d completed task %d (pending_id=%d)",
                         thread_idx, core_id, running_task_ids_[core_id], pending_task_ids_[core_id]);

                int completed_task_id = running_task_ids_[core_id];

                if (profiling_enabled) {
                    uint64_t finish_ts = get_sys_cnt_aicpu();
                    PerfBuffer* perf_buf = (PerfBuffer*)h->perf_records_addr;
                    rmb();
                    uint32_t count = perf_buf->count;
                    if (count > 0) {
                        PerfRecord* record = &perf_buf->records[count - 1];
                        if (record->task_id == static_cast<uint32_t>(completed_task_id)) {
                            perf_aicpu_record_dispatch_and_finish_time(
                                record, dispatch_timestamps_[core_id], finish_ts);
                        }
                    }
                    dispatch_timestamps_[core_id] = get_sys_cnt_aicpu();
                }

                cur_thread_completed++;
                completed_tasks_.fetch_add(1, std::memory_order_release);

                running_task_ids_[core_id] = AICPU_TASK_INVALID;

                bool dispatched = false;
                if (pending_task_ids_[core_id] == AICPU_TASK_INVALID) {
                    if (h->core_type == CoreType::AIC && cur_aic_ready_count > 0) {
                        dispatched = try_dispatch_task(core_id,
                            reg_addr,
                            CoreType::AIC,
                            thread_idx,
                            cur_ready_queue_aic,
                            cur_aic_head,
                            cur_aic_ready_count,
                            profiling_enabled,
                            runtime);
                    } else if (h->core_type == CoreType::AIV && cur_aiv_ready_count > 0) {
                        dispatched = try_dispatch_task(core_id,
                            reg_addr,
                            CoreType::AIV,
                            thread_idx,
                            cur_ready_queue_aiv,
                            cur_aiv_head,
                            cur_aiv_ready_count,
                            profiling_enabled,
                            runtime);
                    }
                }

                Task* task = runtime.get_task(completed_task_id);
                resolve_task_dependencies(task,
                    runtime,
                    cur_ready_queue_aic,
                    cur_aic_tail,
                    cur_aic_ready_count,
                    cur_ready_queue_aiv,
                    cur_aiv_tail,
                    cur_aiv_ready_count);

                made_progress = true;

                // Update timestamp if didn't dispatch (try_dispatch_task updates it if dispatched)
                if (!dispatched && profiling_enabled) {
                    dispatch_timestamps_[core_id] = get_sys_cnt_aicpu();
                }
            }

            // Case 4: Dispatch new task if pending slot is available
            if (pending_task_ids_[core_id] == AICPU_TASK_INVALID) {
                if (h->core_type == CoreType::AIC && cur_aic_ready_count > 0) {
                    if (try_dispatch_task(core_id,
                            reg_addr,
                            CoreType::AIC,
                            thread_idx,
                            cur_ready_queue_aic,
                            cur_aic_head,
                            cur_aic_ready_count,
                            profiling_enabled,
                            runtime)) {
                        made_progress = true;
                    }
                } else if (h->core_type == CoreType::AIV && cur_aiv_ready_count > 0) {
                    if (try_dispatch_task(core_id,
                            reg_addr,
                            CoreType::AIV,
                            thread_idx,
                            cur_ready_queue_aiv,
                            cur_aiv_head,
                            cur_aiv_ready_count,
                            profiling_enabled,
                            runtime)) {
                        made_progress = true;
                    }
                }
            }
        }

        // Refill local queues from shared queues
        if (cur_aic_ready_count == 0) {
            if (ready_count_aic_.load(std::memory_order_acquire) > 0) {
                std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                int available = ready_count_aic_.load(std::memory_order_relaxed);
                int to_grab = (available < aic_per_thread_) ? available : aic_per_thread_;

                for (int i = 0; i < to_grab; i++) {
                    int task_id = ready_queue_aic_[ready_queue_aic_head_];
                    ready_queue_aic_head_ = (ready_queue_aic_head_ + 1) % RUNTIME_MAX_TASKS;
                    cur_ready_queue_aic[cur_aic_tail] = task_id;
                    cur_aic_tail = (cur_aic_tail + 1) % MAX_CORES_PER_THREAD;
                }
                ready_count_aic_.fetch_sub(to_grab, std::memory_order_release);
                cur_aic_ready_count += to_grab;

                LOG_INFO(
                    "Thread %d: Grabbed %d AIC tasks from shared queue (available=%d)", thread_idx, to_grab, available);
            }
        }

        if (cur_aiv_ready_count == 0) {
            if (ready_count_aiv_.load(std::memory_order_acquire) > 0) {
                std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                int available = ready_count_aiv_.load(std::memory_order_relaxed);
                int to_grab = (available < aiv_per_thread_) ? available : aiv_per_thread_;

                for (int i = 0; i < to_grab; i++) {
                    int task_id = ready_queue_aiv_[ready_queue_aiv_head_];
                    ready_queue_aiv_head_ = (ready_queue_aiv_head_ + 1) % RUNTIME_MAX_TASKS;
                    cur_ready_queue_aiv[cur_aiv_tail] = task_id;
                    cur_aiv_tail = (cur_aiv_tail + 1) % MAX_CORES_PER_THREAD;
                }
                ready_count_aiv_.fetch_sub(to_grab, std::memory_order_release);
                cur_aiv_ready_count += to_grab;

                LOG_INFO(
                    "Thread %d: Grabbed %d AIV tasks from shared queue (available=%d)", thread_idx, to_grab, available);
            }
        }

        // Check completion
        if (completed_tasks_.load(std::memory_order_acquire) >= task_count) {
            bool all_cores_idle = true;

            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                if (pending_task_ids_[core_id] != AICPU_TASK_INVALID || running_task_ids_[core_id] != AICPU_TASK_INVALID) {
                    all_cores_idle = false;

                    if (verification_warning_count == 0) {
                        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
                        uint64_t reg_val = read_reg(reg_addr, RegId::COND);
                        LOG_WARN(
                            "Thread %d: Counter reached %d/%d but core %d still has work (COND=0x%lx, pending_id=%d, "
                            "running_id=%d)",
                            thread_idx,
                            completed_tasks_.load(std::memory_order_acquire),
                            task_count,
                            core_id,
                            reg_val,
                            pending_task_ids_[core_id],
                            running_task_ids_[core_id]);
                    }
                    break;
                }
            }

            if (all_cores_idle) {
                // Truly complete: counter reached and all cores idle
                int aic_remaining = ready_count_aic_.load(std::memory_order_acquire);
                int aiv_remaining = ready_count_aiv_.load(std::memory_order_acquire);
                if (aic_remaining > 0 || aiv_remaining > 0) {
                    LOG_WARN("Thread %d: Queues not empty after completion! AIC=%d, AIV=%d",
                        thread_idx,
                        aic_remaining,
                        aiv_remaining);
                }
                break;  // Exit main loop
            }

            verification_warning_count++;
            if (verification_warning_count > MAX_VERIFICATION_WARNINGS) {
                LOG_ERROR("Thread %d: Counter reached but cores still working after %d checks!",
                    thread_idx,
                    verification_warning_count);
                diagnose_stuck_state(runtime, thread_idx, cur_thread_cores, core_num, hank);
                return -1;
            }
        }

        // Timeout detection
        if (!made_progress) {
            idle_iterations++;
            if (idle_iterations % WARN_INTERVAL == 0) {
                int current = completed_tasks_.load(std::memory_order_acquire);
                LOG_WARN("Thread %d: %d idle iterations, progress %d/%d tasks",
                    thread_idx,
                    idle_iterations,
                    current,
                    task_count);
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                LOG_ERROR("Thread %d: Timeout after %d idle iterations!", thread_idx, idle_iterations);
                diagnose_stuck_state(runtime, thread_idx, cur_thread_cores, core_num, hank);
                return -1;
            }
        } else {
            idle_iterations = 0;
        }
        made_progress = false;
    }

    LOG_INFO("Thread %d: Execution complete, completed %d tasks", thread_idx, cur_thread_completed);
    return cur_thread_completed;
}

int AicpuExecutor::run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

    LOG_INFO("Thread %d: Start", thread_idx);

    const int* cur_thread_cores = core_assignments_[thread_idx];

    LOG_INFO("Thread %d: Runtime has %d tasks", thread_idx, runtime->get_task_count());
    int completed = resolve_and_dispatch(*runtime, thread_idx, cur_thread_cores, thread_cores_num_);
    LOG_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);

    int rc = shutdown_aicore(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    // Flush performance buffers for cores managed by this thread
    if (runtime->enable_profiling) {
        perf_aicpu_flush_buffers(runtime, thread_idx, cur_thread_cores, thread_cores_num_);
    }

    LOG_INFO("Thread %d: Completed", thread_idx);

    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        LOG_INFO("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return 0;
}

void AicpuExecutor::deinit() {
    ready_count_aic_.store(0, std::memory_order_release);
    ready_count_aiv_.store(0, std::memory_order_release);

    ready_queue_aic_head_ = 0;
    ready_queue_aic_tail_ = 0;
    ready_queue_aiv_head_ = 0;
    ready_queue_aiv_tail_ = 0;

    for (int i = 0; i < RUNTIME_MAX_WORKER; i++) {
        dispatch_timestamps_[i] = 0;
        core_dispatch_counts_[i] = 0;
        pending_task_ids_[i] = AICPU_TASK_INVALID;
        running_task_ids_[i] = AICPU_TASK_INVALID;
        core_first_dispatch_[i] = true;
    }

    for (int t = 0; t < MAX_AICPU_THREADS; t++) {
        cur_ready_queue_aic_head_[t] = 0;
        cur_ready_queue_aic_tail_[t] = 0;
        cur_ready_queue_aiv_head_[t] = 0;
        cur_ready_queue_aiv_tail_[t] = 0;
    }

    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    LOG_INFO("DeInit: AicpuExecutor reset complete");
}

void AicpuExecutor::diagnose_stuck_state(
    Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num, Handshake* hank) {
    LOG_ERROR("========== DIAGNOSTIC REPORT: Thread %d ==========", thread_idx);

    int completed = completed_tasks_.load(std::memory_order_acquire);
    int total = total_tasks_.load(std::memory_order_acquire);
    LOG_ERROR("Progress: %d/%d tasks (%.1f%%)", completed, total, total > 0 ? completed * 100.0 / total : 0.0);

    int aic_ready = ready_count_aic_.load(std::memory_order_acquire);
    int aiv_ready = ready_count_aiv_.load(std::memory_order_acquire);
    LOG_ERROR("Ready Queues: AIC=%d, AIV=%d", aic_ready, aiv_ready);

    int busy_cores = 0;
    int idle_cores = 0;

    LOG_ERROR("Core Status:");
    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* h = &hank[core_id];

        const char* core_type_str = core_type_to_string(h->core_type);

        uint64_t reg_addr = core_id_to_reg_addr_[core_id];
        uint64_t reg_val = read_reg(reg_addr, RegId::COND);
        int reg_task_id = EXTRACT_TASK_ID(reg_val);
        int reg_state = EXTRACT_TASK_STATE(reg_val);

        int pending_id = pending_task_ids_[core_id];
        int running_id = running_task_ids_[core_id];

        if (pending_id != AICPU_TASK_INVALID || running_id != AICPU_TASK_INVALID) {
            busy_cores++;

            if (pending_id != AICPU_TASK_INVALID) {
                Task* task = runtime.get_task(pending_id);
                LOG_ERROR(
                    "  Core %d [%s, PENDING]: COND=0x%lx (reg_task_id=%d, reg_state=%d), pending_id=%d, func_id=%d, "
                    "fanin=%d, fanout=%d",
                    core_id,
                    core_type_str,
                    reg_val,
                    reg_task_id,
                    reg_state,
                    task->task_id,
                    task->func_id,
                    task->fanin.load(std::memory_order_acquire),
                    task->fanout_count);
            }
            if (running_id != AICPU_TASK_INVALID) {
                Task* task = runtime.get_task(running_id);
                LOG_ERROR(
                    "  Core %d [%s, RUNNING]: COND=0x%lx (reg_task_id=%d, reg_state=%d), running_id=%d, func_id=%d, "
                    "fanin=%d, fanout=%d",
                    core_id,
                    core_type_str,
                    reg_val,
                    reg_task_id,
                    reg_state,
                    task->task_id,
                    task->func_id,
                    task->fanin.load(std::memory_order_acquire),
                    task->fanout_count);
            }
        } else {
            idle_cores++;
        }
    }

    LOG_ERROR("Summary: %d busy, %d idle", busy_cores, idle_cores);

    // Diagnose deadlock vs livelock
    if (busy_cores == 0 && aic_ready == 0 && aiv_ready == 0 && completed < total) {
        LOG_ERROR("*** DEADLOCK DETECTED ***");
        LOG_ERROR("All cores idle, no ready tasks, but %d tasks incomplete", total - completed);

        LOG_ERROR("Tasks with fanin > 0:");
        int stuck_count = 0;
        for (int tid = 0; tid < total && stuck_count < 10; tid++) {
            Task* t = runtime.get_task(tid);
            int fanin = t->fanin.load(std::memory_order_acquire);
            if (fanin > 0) {
                LOG_ERROR("  Task %d: fanin=%d (waiting for dependencies)", tid, fanin);
                stuck_count++;
            }
        }
        if (stuck_count == 0) {
            LOG_ERROR("  No tasks waiting! Possible counter corruption.");
        }
    } else if (busy_cores > 0) {
        LOG_ERROR("*** LIVELOCK / HUNG TASK ***");
        LOG_ERROR("%d cores executing but no progress", busy_cores);
    }

    LOG_ERROR("========== END DIAGNOSTIC ==========");
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
    // Initialize log switches (only once, thread-safe)
    static std::once_flag log_init_flag;
    std::call_once(log_init_flag, []() { init_log_switch(); });

    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    LOG_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

    // Get platform register addresses from platform-level global
    g_aicpu_executor.regs_ = get_platform_regs();

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
            LOG_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        LOG_INFO("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit();
    }

    LOG_INFO("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
