#include <atomic>
#include <cstdint>
#include <mutex>

#include "device_log.h"
#include "runtime.h"

// Platform-specific constants are provided via CMake compile definitions
// (PLATFORM_MAX_AICPU_THREADS, PLATFORM_MAX_AIC_PER_THREAD, etc.)
// defined in src/platform/*/host/CMakeLists.txt
constexpr int MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int MAX_AIC_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD;
constexpr int MAX_AIV_PER_THREAD = PLATFORM_MAX_AIV_PER_THREAD;
constexpr int MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;

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
    int init(Runtime* runtime);
    int hank_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int resolve_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores);
    int run(Runtime* runtime);
    void deinit();
    void diagnose_stuck_state(Runtime& runtime, int thread_idx, const int* cur_thread_cores,
                              int core_num, Handshake* hank);
};

static AicpuExecutor g_aicpu_executor;

// ===== AicpuExecutor Method Implementations =====

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

    // Verify core topology has been initialized by platform
    if (!runtime->core_topology.initialized) {
        DEV_ERROR("Core topology not initialized by platform");
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

    // Use core topology instead of hardcoded calculation
    cores_total_num_ = runtime->core_topology.total_cores;
    thread_cores_num_ = cores_total_num_ / thread_num_;

    if (cores_total_num_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Total cores %d exceeds maximum %d", cores_total_num_, MAX_CORES_PER_THREAD);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    DEV_INFO("Config: threads=%d, cores=%d, cores_per_thread=%d", thread_num_, cores_total_num_, thread_cores_num_);

    // Infer block information from core topology
    // Find the maximum block_idx to determine number of blocks
    int num_blocks = 0;
    for (int i = 0; i < cores_total_num_; i++) {
        const CoreInfo* info = runtime->get_core_info(i);
        if (info && info->block_idx + 1 > num_blocks) {
            num_blocks = info->block_idx + 1;
        }
    }

    // Calculate blocks per thread
    int blocks_per_thread = num_blocks / thread_num_;
    if (num_blocks % thread_num_ != 0) {
        DEV_ERROR("Number of blocks (%d) must be divisible by thread_num (%d)", num_blocks, thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    DEV_INFO("Block assignment: %d blocks, %d threads, %d blocks per thread",
        num_blocks,
        thread_num_,
        blocks_per_thread);

    // Assign cores to threads using platform-provided topology
    // Maintain original allocation order: group by blocks, AIC first then AIV
    for (int t = 0; t < thread_num_; t++) {
        int start_block = t * blocks_per_thread;
        int end_block = (t + 1) * blocks_per_thread;
        int core_idx = 0;

        int aic_start = -1, aic_end = -1;
        int aiv_start = -1, aiv_end = -1;

        // First, assign all AIC cores for this thread's blocks
        for (int i = 0; i < cores_total_num_; i++) {
            const CoreInfo* info = runtime->get_core_info(i);
            if (info && info->core_type == 0 &&
                info->block_idx >= start_block && info->block_idx < end_block) {
                if (aic_start == -1) aic_start = info->core_id;
                aic_end = info->core_id;
                core_assignments_[t][core_idx++] = info->core_id;
            }
        }

        // Then, assign all AIV cores for this thread's blocks
        for (int i = 0; i < cores_total_num_; i++) {
            const CoreInfo* info = runtime->get_core_info(i);
            if (info && info->core_type == 1 &&
                info->block_idx >= start_block && info->block_idx < end_block) {
                if (aiv_start == -1) aiv_start = info->core_id;
                aiv_end = info->core_id;
                core_assignments_[t][core_idx++] = info->core_id;
            }
        }

        DEV_INFO(
            "Thread %d: manages cores: AIC[%d-%d] AIV[%d-%d], total %d cores",
            t,
            aic_start,
            aic_end,
            aiv_start,
            aiv_end,
            core_idx);
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
    DEV_INFO("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Handshake AICore - Initialize and synchronize with AICore kernels
 */
int AicpuExecutor::hank_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("Thread %d: Handshaking with %d cores", thread_idx, thread_cores_num_);

    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
        hank->aicpu_ready = 1;
    }

    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        while (hank->aicore_done == 0) {
        }
        DEV_INFO("Thread %d: success hank->aicore_done = %u", thread_idx, hank->aicore_done);
    }
    return 0;
}

/**
 * Shutdown AICore - Send quit signal to all AICore kernels
 */
int AicpuExecutor::shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores) {
    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, thread_cores_num_);

    for (int i = 0; i < thread_cores_num_; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
        hank->control = 1;
    }
    DEV_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

/**
 * Resolve dependencies and dispatch tasks using polling-based dispatch to
 * AICore
 */
int AicpuExecutor::resolve_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
    Handshake* hank = (Handshake*)runtime.workers;

    DEV_INFO("Thread %d: Starting execution with %d cores", thread_idx, core_num);

    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int task_count = total_tasks_.load(std::memory_order_acquire);

    // Timeout detection using idle iteration counting
    int idle_iterations = 0;
    const int MAX_IDLE_ITERATIONS = 1000000;
    const int WARN_INTERVAL = 100000;
    bool made_progress = false;

    int verification_warning_count = 0;
    const int MAX_VERIFICATION_WARNINGS = 10;

    // Execute tasks using polling-based dispatch with integrated verification
    while (true) {
        // Double verification: check counter reached AND all cores truly idle
        if (completed_tasks_.load(std::memory_order_acquire) >= task_count) {
            bool all_cores_idle = true;

            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                Handshake* h = &hank[core_id];

                if (h->task_status != 0 || h->task != 0) {
                    all_cores_idle = false;

                    if (verification_warning_count == 0) {
                        DEV_WARN("Thread %d: Counter reached %d/%d but core %d still has work (status=%d, task=%p)",
                                thread_idx, completed_tasks_.load(std::memory_order_acquire), task_count,
                                core_id, h->task_status, (void*)h->task);
                    }
                    break;
                }
            }

            if (all_cores_idle) {
                // Truly complete: counter reached and all cores idle
                int aic_remaining = ready_count_aic_.load(std::memory_order_acquire);
                int aiv_remaining = ready_count_aiv_.load(std::memory_order_acquire);
                if (aic_remaining > 0 || aiv_remaining > 0) {
                    DEV_WARN("Thread %d: Queues not empty after completion! AIC=%d, AIV=%d",
                            thread_idx, aic_remaining, aiv_remaining);
                }
                break;  // Exit main loop
            }

            // Counter reached but cores still working, continue main loop to process them
            verification_warning_count++;
            if (verification_warning_count > MAX_VERIFICATION_WARNINGS) {
                DEV_ERROR("Thread %d: Counter reached but cores still working after %d checks!",
                         thread_idx, verification_warning_count);
                diagnose_stuck_state(runtime, thread_idx, cur_thread_cores, core_num, hank);
                return -1;
            }
        }

        made_progress = false;

        // Phase 1: Process completed tasks on my managed cores
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            // Core finished a task (idle + task not null)
            if (h->task_status == 0 && h->task != 0) {
                // Get completed task and immediately clear the pointer to prevent duplicate detection
                Task* task = reinterpret_cast<Task*>(h->task);
                h->task = 0;  // Clear immediately to minimize race condition window

                int task_id = task->task_id;

                DEV_INFO("Thread %d: Core %d completed task %d", thread_idx, core_id, task_id);

                // Update fanin of successors atomically and add to appropriate
                // shared ready queue
                for (int j = 0; j < task->fanout_count; j++) {
                    int dep_id = task->fanout[j];
                    Task* dep = runtime.get_task(dep_id);

                    // Atomic decrement fanin
                    int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);

                    // Dependency resolved, add to appropriate shared ready
                    // queue
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

                // Update counters
                cur_thread_tasks_in_flight--;
                cur_thread_completed++;
                made_progress = true;
                completed_tasks_.fetch_add(1, std::memory_order_release);
            }
        }

        // Load balancing: Skip dispatch if all my cores are busy
        if (cur_thread_tasks_in_flight < core_num) {
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
                                Task* task = runtime.get_task(task_id);

                                DEV_INFO("Thread %d: Dispatching AIC task %d to core %d", thread_idx, task_id, core_id);

                                h->task = reinterpret_cast<uint64_t>(task);
                                h->task_status = 1;  // Mark as busy
                                cur_thread_tasks_in_flight++;
                                made_progress = true;
                            }
                        }
                    } else if (h->core_type == 1) {  // AIV core
                        if (ready_count_aiv_.load(std::memory_order_acquire) > 0) {
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            int count = ready_count_aiv_.load(std::memory_order_relaxed);
                            if (count > 0) {
                                ready_count_aiv_.fetch_sub(1, std::memory_order_release);
                                int task_id = ready_queue_aiv_[count - 1];
                                Task* task = runtime.get_task(task_id);

                                DEV_INFO("Thread %d: Dispatching AIV task %d to core %d", thread_idx, task_id, core_id);

                                h->task = reinterpret_cast<uint64_t>(task);
                                h->task_status = 1;  // Mark as busy
                                cur_thread_tasks_in_flight++;
                                made_progress = true;
                            }
                        }
                    }
                }
            }
        }

        // Timeout detection: track idle iterations when no progress
        if (!made_progress) {
            idle_iterations++;
            if (idle_iterations % WARN_INTERVAL == 0) {
                int current = completed_tasks_.load(std::memory_order_acquire);
                DEV_WARN("Thread %d: %d idle iterations, progress %d/%d tasks",
                        thread_idx, idle_iterations, current, task_count);
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("Thread %d: Timeout after %d idle iterations!", thread_idx, idle_iterations);
                diagnose_stuck_state(runtime, thread_idx, cur_thread_cores, core_num, hank);
                return -1;
            }
        } else {
            idle_iterations = 0;
        }
    }

    DEV_INFO("Thread %d: Execution complete, completed %d tasks", thread_idx, cur_thread_completed);
    return cur_thread_completed;
}

int AicpuExecutor::run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

    DEV_INFO("Thread %d: Start", thread_idx);

    const int* cur_thread_cores = core_assignments_[thread_idx];

    auto rc = hank_aicore(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    DEV_INFO("Thread %d: Runtime has %d tasks", thread_idx, runtime->get_task_count());
    int completed = resolve_and_dispatch(*runtime, thread_idx, cur_thread_cores, thread_cores_num_);
    DEV_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);

    rc = shutdown_aicore(runtime, thread_idx, cur_thread_cores);
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

void AicpuExecutor::deinit() {
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

    DEV_INFO("DeInit: AicpuExecutor reset complete");
}

void AicpuExecutor::diagnose_stuck_state(Runtime& runtime, int thread_idx,
                                         const int* cur_thread_cores, int core_num,
                                         Handshake* hank) {
    DEV_ERROR("========== DIAGNOSTIC REPORT: Thread %d ==========", thread_idx);

    int completed = completed_tasks_.load(std::memory_order_acquire);
    int total = total_tasks_.load(std::memory_order_acquire);
    DEV_ERROR("Progress: %d/%d tasks (%.1f%%)",
             completed, total, total > 0 ? completed * 100.0 / total : 0.0);

    int aic_ready = ready_count_aic_.load(std::memory_order_acquire);
    int aiv_ready = ready_count_aiv_.load(std::memory_order_acquire);
    DEV_ERROR("Ready Queues: AIC=%d, AIV=%d", aic_ready, aiv_ready);

    int busy_cores = 0;
    int idle_cores = 0;
    int anomaly_cores = 0;

    DEV_ERROR("Core Status:");
    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* h = &hank[core_id];

        const char* core_type_str = (h->core_type == 0) ? "AIC" : "AIV";

        if (h->task != 0) {
            Task* task = reinterpret_cast<Task*>(h->task);
            busy_cores++;

            DEV_ERROR("  Core %d [%s, BUSY]: task_id=%d, func_id=%d, fanin=%d, fanout=%d",
                     core_id, core_type_str,
                     task->task_id, task->func_id,
                     task->fanin.load(std::memory_order_acquire),
                     task->fanout_count);
        } else if (h->task_status != 0) {
            anomaly_cores++;
            DEV_ERROR("  Core %d [%s, ANOMALY]: status=BUSY but task=NULL", core_id, core_type_str);
        } else {
            idle_cores++;
        }
    }

    DEV_ERROR("Summary: %d busy, %d idle, %d anomaly", busy_cores, idle_cores, anomaly_cores);

    // Diagnose deadlock vs livelock
    if (busy_cores == 0 && aic_ready == 0 && aiv_ready == 0 && completed < total) {
        DEV_ERROR("*** DEADLOCK DETECTED ***");
        DEV_ERROR("All cores idle, no ready tasks, but %d tasks incomplete", total - completed);

        DEV_ERROR("Tasks with fanin > 0:");
        int stuck_count = 0;
        for (int tid = 0; tid < total && stuck_count < 10; tid++) {
            Task* t = runtime.get_task(tid);
            int fanin = t->fanin.load(std::memory_order_acquire);
            if (fanin > 0) {
                DEV_ERROR("  Task %d: fanin=%d (waiting for dependencies)", tid, fanin);
                stuck_count++;
            }
        }
        if (stuck_count == 0) {
            DEV_ERROR("  No tasks waiting! Possible counter corruption.");
        }
    } else if (busy_cores > 0) {
        DEV_ERROR("*** LIVELOCK / HUNG TASK ***");
        DEV_ERROR("%d cores executing but no progress", busy_cores);
    }

    DEV_ERROR("========== END DIAGNOSTIC ==========");
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
 * @param runtime Pointer to Runtime structure containing:
 *                - workers[]: handshake buffers for AICPU-AICore communication
 *                - block_dim, sche_cpu_num: execution parameters
 *                - tasks[]: task runtime to execute
 * @return 0 on success, non-zero on error
 */
extern "C" int aicpu_execute(Runtime* runtime) {
    if (runtime == nullptr) {
        DEV_ERROR("%s", "Invalid runtime argument: null pointer");
        return -1;
    }

    DEV_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

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
