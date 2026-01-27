#include <cstdint>
#include <atomic>
#include <mutex>
#include "device_log.h"
#include "runtime.h"

constexpr int MAX_AICPU_THREADS = 4;
constexpr int MAX_AIC_PER_THREAD = 24;
constexpr int MAX_AIV_PER_THREAD = 48;
constexpr int MAX_CORES_PER_THREAD = MAX_AIC_PER_THREAD + MAX_AIV_PER_THREAD;

struct RuntimeExecutor {
    // ===== Thread management state =====
    std::atomic<int> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int thread_num_{0};
    int total_cores_{0};
    int coresPerBlockdim_{3};
    int cores_per_thread_{0};
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // ===== Task queue state =====
    std::mutex ready_queue_aic_mutex_;
    int ready_queue_aic_[RUNTIME_MAX_TASKS];
    std::atomic<int> ready_count_aic_{0};

    std::mutex ready_queue_aiv_mutex_;
    int ready_queue_aiv_[RUNTIME_MAX_TASKS];
    std::atomic<int> ready_count_aiv_{0};

    // Task execution tracking
    std::atomic<int> completed_tasks_{0};
    std::atomic<int> total_tasks_{0};
    std::atomic<int> finished_count_{0};

    // ===== Methods =====
    int Init(Runtime* runtime);
    int HankAiCore(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int Execute(Runtime& r, Handshake* hank, int thread_idx,
                const int* cur_thread_cores, int core_num);
    int ShutdownAiCore(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int Run(Runtime* runtime);
    void DeInit();
};

static RuntimeExecutor g_executor;

// ===== RuntimeExecutor Method Implementations =====

int RuntimeExecutor::Init(Runtime* runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    DEV_INFO("RuntimeExecutor: Initializing");

    if (runtime == nullptr) {
        DEV_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Read execution parameters from runtime
    thread_num_ = runtime->scheCpuNum;
    if (thread_num_ == 0) thread_num_ = 1;

    total_cores_ = runtime->block_dim * coresPerBlockdim_;
    cores_per_thread_ = total_cores_ / thread_num_;

    DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d",
             thread_num_, total_cores_, cores_per_thread_);

    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (total_cores_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Total cores %d exceeds maximum %d", total_cores_, MAX_CORES_PER_THREAD);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Pre-compute core assignments for each thread
    // Each thread manages blocks_per_thread blocks
    // For each block b: AIC is core b, AIVs are cores (nrAic + b*2) and (nrAic + b*2 + 1)
    int num_aic = runtime->block_dim;  // Total AIC cores (= block_dim)
    int blocks_per_thread = runtime->block_dim / thread_num_;

    // Validate block distribution
    if (runtime->block_dim % thread_num_ != 0) {
        DEV_ERROR("block_dim (%d) must be divisible by thread_num (%d)",
                  runtime->block_dim, thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    DEV_INFO("Block assignment: %d blocks, %d threads, %d blocks per thread",
             runtime->block_dim, thread_num_, blocks_per_thread);

    for (int t = 0; t < thread_num_; t++) {
        int start_block = t * blocks_per_thread;
        int end_block = (t + 1) * blocks_per_thread;
        int core_idx = 0;

        // Assign AIC cores for all blocks managed by this thread
        for (int b = start_block; b < end_block; b++) {
            core_assignments_[t][core_idx++] = b;  // AIC core ID = block ID
        }

        // Assign AIV cores for all blocks managed by this thread
        for (int b = start_block; b < end_block; b++) {
            int aiv_base = num_aic;  // AIV cores start after all AIC cores
            core_assignments_[t][core_idx++] = aiv_base + b * 2;      // First AIV of block b
            core_assignments_[t][core_idx++] = aiv_base + b * 2 + 1;  // Second AIV of block b
        }

        DEV_INFO("Thread %d: manages blockDims [%d-%d], cores: AIC[%d-%d] AIV[%d-%d]",
                 t, start_block, end_block - 1,
                 start_block, end_block - 1,
                 num_aic + start_block * 2, num_aic + (end_block - 1) * 2 + 1);
    }

    // Initialize runtime execution state
    total_tasks_.store(runtime->get_task_count(), std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);

    int initial_ready[RUNTIME_MAX_TASKS];
    int initial_count = runtime->get_initial_ready_tasks(initial_ready);

    DEV_INFO("Init: Found %d initially ready tasks", initial_count);

    int aic_count = 0;
    int aiv_count = 0;
    for (int i = 0; i < initial_count; i++) {
        Task* task = runtime->get_task(initial_ready[i]);
        if (task->core_type == 0) {  // AIC
            ready_queue_aic_[aic_count++] = initial_ready[i];
        } else {  // AIV
            ready_queue_aiv_[aiv_count++] = initial_ready[i];
        }
    }
    ready_count_aic_.store(aic_count, std::memory_order_release);
    ready_count_aiv_.store(aiv_count, std::memory_order_release);

    DEV_INFO("Init: Initial ready tasks: AIC=%d, AIV=%d", aic_count, aiv_count);

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("RuntimeExecutor: Init complete");
    return 0;
}

/**
 * Handshake AICore - Initialize and synchronize with AICore kernels
 */
int RuntimeExecutor::HankAiCore(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("Thread %d: Handshaking with %d cores", thread_idx, cores_per_thread_);

    for (int i = 0; i < cores_per_thread_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
        hank->aicpu_ready = 1;
    }

    for (int i = 0; i < cores_per_thread_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        while (hank->aicore_done == 0) { }
        DEV_INFO("Thread %d: success hank->aicore_done = %u", thread_idx, (uint64_t)hank->aicore_done);
    }
    return 0;
}

/**
 * Shutdown AICore - Send quit signal to all AICore kernels
 */
int RuntimeExecutor::ShutdownAiCore(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, cores_per_thread_);

    for (int i = 0; i < cores_per_thread_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
        hank->control = 1;
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

/**
 * Execute task runtime using polling-based dispatch to AICore
 */
int RuntimeExecutor::Execute(Runtime& r, Handshake* hank, int thread_idx,
                           const int* cur_thread_cores, int core_num) {

    DEV_INFO("Thread %d: Starting execution with %d cores", thread_idx, core_num);

    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int task_count = total_tasks_.load(std::memory_order_acquire);

    // Execute tasks using polling-based dispatch
    while (completed_tasks_.load(std::memory_order_acquire) < task_count) {

        // Phase 1: Process completed tasks on my managed cores
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Core finished a task (idle + task not null)
            if (h->task_status == 0 && h->task != 0) {
                // Get completed task
                Task* task = reinterpret_cast<Task*>(h->task);
                int task_id = task->task_id;

                DEV_INFO("Thread %d: Core %d completed task %d", thread_idx, core_id, task_id);

                // Update fanin of successors atomically and add to appropriate shared ready queue
                for (int j = 0; j < task->fanout_count; j++) {
                    int dep_id = task->fanout[j];
                    Task* dep = r.get_task(dep_id);

                    // Atomic decrement fanin
                    int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);

                    // Dependency resolved, add to appropriate shared ready queue
                    if (prev_fanin == 1) {
                        if (dep->core_type == 0) {  // AIC task
                            std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                            int idx = ready_count_aic_.load(std::memory_order_relaxed);
                            ready_queue_aic_[idx] = dep_id;
                            ready_count_aic_.fetch_add(1, std::memory_order_release);
                            DEV_INFO("Thread %d: Task %d became ready -> AIC queue", thread_idx, dep_id);
                        } else {  // AIV task
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            int idx = ready_count_aiv_.load(std::memory_order_relaxed);
                            ready_queue_aiv_[idx] = dep_id;
                            ready_count_aiv_.fetch_add(1, std::memory_order_release);
                            DEV_INFO("Thread %d: Task %d became ready -> AIV queue", thread_idx, dep_id);
                        }
                    }
                }

                // Clear task pointer and update counters
                h->task = 0;
                cur_thread_tasks_in_flight--;
                completed_tasks_.fetch_add(1, std::memory_order_release);
                cur_thread_completed++;
            }
        }

        // Load balancing: Skip dispatch if all my cores are busy
        if (cur_thread_tasks_in_flight >= core_num) {
            continue;
        }

        // Phase 2: Dispatch new tasks from matching ready queue to idle cores
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Core is idle and available (idle + task is null)
            if (h->task_status == 0 && h->task == 0) {
                // Dispatch from matching queue based on core type
                if (h->core_type == 0) {  // AIC core
                    if (ready_count_aic_.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                        int count = ready_count_aic_.load(std::memory_order_relaxed);
                        if (count > 0) {
                            ready_count_aic_.fetch_sub(1, std::memory_order_release);
                            int task_id = ready_queue_aic_[count - 1];
                            Task* task = r.get_task(task_id);

                            DEV_INFO("Thread %d: Dispatching AIC task %d to core %d",
                                    thread_idx, task_id, core_id);

                            h->task = reinterpret_cast<uint64_t>(task);
                            h->task_status = 1;  // Mark as busy
                            cur_thread_tasks_in_flight++;
                        }
                    }
                } else if (h->core_type == 1) {  // AIV core
                    if (ready_count_aiv_.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                        int count = ready_count_aiv_.load(std::memory_order_relaxed);
                        if (count > 0) {
                            ready_count_aiv_.fetch_sub(1, std::memory_order_release);
                            int task_id = ready_queue_aiv_[count - 1];
                            Task* task = r.get_task(task_id);

                            DEV_INFO("Thread %d: Dispatching AIV task %d to core %d",
                                    thread_idx, task_id, core_id);

                            h->task = reinterpret_cast<uint64_t>(task);
                            h->task_status = 1;  // Mark as busy
                            cur_thread_tasks_in_flight++;
                        }
                    }
                }
            }
        }
    }

    DEV_INFO("Thread %d: Execution complete, completed %d tasks", thread_idx, cur_thread_completed);
    return cur_thread_completed;
}

int RuntimeExecutor::Run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

    DEV_INFO("Thread %d: Start", thread_idx);

    const int* cur_thread_cores = core_assignments_[thread_idx];

    auto rc = HankAiCore(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    Handshake* hank = (Handshake*)runtime->workers;
    DEV_INFO("Thread %d: Runtime has %d tasks", thread_idx, runtime->get_task_count());
    int completed = Execute(*runtime, hank, thread_idx, cur_thread_cores, cores_per_thread_);
    DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);

    rc = ShutdownAiCore(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        DEV_INFO("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return 0;
}

void RuntimeExecutor::DeInit() {
    // Cleanup runtime execution state
    ready_count_aic_.store(0, std::memory_order_release);
    ready_count_aiv_.store(0, std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);

    DEV_INFO("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: RuntimeExecutor reset complete");
}

// ===== Public Entry Point =====

/**
 * AicpuExecute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor (thread-safe, first thread only)
 * 2. Wait for initialization to complete
 * 3. Execute tasks on managed cores
 * 4. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure containing:
 *                - workers[]: handshake buffers for AICPU-AICore communication
 *                - block_dim, scheCpuNum: execution parameters
 *                - tasks[]: task runtime to execute
 * @return 0 on success, non-zero on error
 */
extern "C" int AicpuExecute(Runtime* runtime) {
    if (runtime == nullptr) {
        DEV_ERROR("%s", "Invalid runtime argument: null pointer");
        return -1;
    }

    DEV_INFO("%s", "AicpuExecute: Starting AICPU kernel execution");

    g_executor.Init(runtime);

    while (!g_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_executor.init_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("%s", "AicpuExecute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int rc = g_executor.Run(runtime);
    if (rc != 0) {
        DEV_ERROR("AicpuExecute: Thread execution failed with rc=%d", rc);
        return rc;
    }

    // Last thread cleans up
    if (g_executor.finished_.load(std::memory_order_acquire)) {
        DEV_INFO("AicpuExecute: Last thread finished, cleaning up");
        g_executor.DeInit();
    }

    DEV_INFO("%s", "AicpuExecute: Kernel execution completed successfully");
    return 0;
}
