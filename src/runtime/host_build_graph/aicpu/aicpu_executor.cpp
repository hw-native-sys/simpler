#include <atomic>
#include <cstdint>
#include <mutex>

#include "common/unified_log.h"
#include "common/platform_config.h"
#include "runtime.h"
#include "aicpu/device_log.h"

constexpr int MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int MAX_AIC_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD;
constexpr int MAX_AIV_PER_THREAD = PLATFORM_MAX_AIV_PER_THREAD;
constexpr int MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;

// Core information for discovery
struct CoreInfo {
    int worker_id;     // Index in runtime.workers[]
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
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // Core discovery arrays (space-time tradeoff: avoid sorting)
    CoreInfo aic_cores_[MAX_CORES_PER_THREAD];
    CoreInfo aiv_cores_[MAX_CORES_PER_THREAD];
    int aic_count_{0};
    int aiv_count_{0};

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

    // ===== Methods =====
    int init(Runtime* runtime);
    int handshake_all_cores(Runtime* runtime);
    void assign_cores_to_threads();
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

    // Assign discovered cores to threads
    assign_cores_to_threads();

    // Initialize runtime execution state
    total_tasks_.store(runtime->get_task_count(), std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);

    int initial_ready[RUNTIME_MAX_TASKS];
    int initial_count = runtime->get_initial_ready_tasks(initial_ready);

    LOG_INFO("Init: Found %d initially ready tasks", initial_count);

    // Reset circular queue indices
    ready_queue_aic_head_ = 0;
    ready_queue_aic_tail_ = 0;
    ready_queue_aiv_head_ = 0;
    ready_queue_aiv_tail_ = 0;

    int aic_count = 0;
    int aiv_count = 0;
    for (int i = 0; i < initial_count; i++) {
        Task* task = runtime->get_task(initial_ready[i]);
        if (task->core_type == CoreType::AIC) {  // AIC
            // Enqueue to tail position (circular)
            ready_queue_aic_[ready_queue_aic_tail_] = initial_ready[i];
            ready_queue_aic_tail_ = (ready_queue_aic_tail_ + 1) % RUNTIME_MAX_TASKS;
            aic_count++;
        } else {  // AIV
            // Enqueue to tail position (circular)
            ready_queue_aiv_[ready_queue_aiv_tail_] = initial_ready[i];
            ready_queue_aiv_tail_ = (ready_queue_aiv_tail_ + 1) % RUNTIME_MAX_TASKS;
            aiv_count++;
        }
    }
    ready_count_aic_.store(aic_count, std::memory_order_release);
    ready_count_aiv_.store(aiv_count, std::memory_order_release);

    LOG_INFO("Init: Initial ready tasks: AIC=%d, AIV=%d", aic_count, aiv_count);

    finished_count_.store(0, std::memory_order_release);

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

        // Read core type (written by AICore during handshake)
        CoreType type = hank->core_type;

        // Classify and store core information
        if (type == CoreType::AIC) {
            aic_cores_[aic_count_].worker_id = i;
            aic_cores_[aic_count_].core_type = type;
            aic_count_++;
        } else if (type == CoreType::AIV) {
            aiv_cores_[aiv_count_].worker_id = i;
            aiv_cores_[aiv_count_].core_type = type;
            aiv_count_++;
        } else {
            LOG_ERROR("Unknown core type %d for core %d", static_cast<int>(type), i);
            return -1;
        }

        LOG_INFO("  Core %d: type=%s", i, core_type_to_string(type));
    }

    LOG_INFO("Discovery complete: AIC=%d, AIV=%d, Total=%d", aic_count_, aiv_count_, cores_total_num_);
    return 0;
}

/**
 * Assign discovered cores to threads using even distribution strategy
 *
 * Each thread receives an equal number of AIC and AIV cores.
 * Cores are assigned in the order they were discovered.
 *
 * Strategy: Evenly distribute each core type across all threads
 * - Thread assignment formula: cores_of_type[thread_idx * cores_per_thread : (thread_idx+1) * cores_per_thread]
 *
 * Requirements (strict mode):
 * - aic_count_ % thread_num_ == 0
 * - aiv_count_ % thread_num_ == 0
 */
void AicpuExecutor::assign_cores_to_threads() {
    // Validate even distribution (strict mode)
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

    int aic_per_thread = aic_count_ / thread_num_;
    int aiv_per_thread = aiv_count_ / thread_num_;

    LOG_INFO("Core Assignment: %d AIC/thread, %d AIV/thread", aic_per_thread, aiv_per_thread);

    for (int t = 0; t < thread_num_; t++) {
        int core_idx = 0;

        // Assign AIC cores to this thread
        int aic_start = t * aic_per_thread;
        int aic_end = (t + 1) * aic_per_thread;
        for (int i = aic_start; i < aic_end; i++) {
            core_assignments_[t][core_idx++] = aic_cores_[i].worker_id;
        }

        // Assign AIV cores to this thread
        int aiv_start = t * aiv_per_thread;
        int aiv_end = (t + 1) * aiv_per_thread;
        for (int i = aiv_start; i < aiv_end; i++) {
            core_assignments_[t][core_idx++] = aiv_cores_[i].worker_id;
        }

        // Build detailed log message with specific core IDs
        char log_buffer[256];
        int offset = 0;

        offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset,
                          "Thread %d: assigned %d cores - AIC[", t, core_idx);

        for (int i = 0; i < aic_per_thread; i++) {
            if (i > 0) offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ",");
            offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset,
                             "%d", aic_cores_[aic_start + i].worker_id);
        }

        offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "] AIV[");

        for (int i = 0; i < aiv_per_thread; i++) {
            if (i > 0) offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ",");
            offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset,
                             "%d", aiv_cores_[aiv_start + i].worker_id);
        }

        offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "]");

        LOG_INFO("%s", log_buffer);
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
        hank->control = 1;
    }
    LOG_INFO("Thread %d: Shutdown complete", thread_idx);
    return 0;
}

/**
 * Resolve dependencies and dispatch tasks using polling-based dispatch to
 * AICore
 */
int AicpuExecutor::resolve_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
    Handshake* hank = (Handshake*)runtime.workers;

    LOG_INFO("Thread %d: Starting execution with %d cores", thread_idx, core_num);

    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int task_count = total_tasks_.load(std::memory_order_acquire);

    // Timeout detection using idle iteration counting
    int idle_iterations = 0;
    const int MAX_IDLE_ITERATIONS = 50000000;
    const int WARN_INTERVAL = 1000000;
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
                        LOG_WARN("Thread %d: Counter reached %d/%d but core %d still has work (status=%d, task=%p)",
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
                    LOG_WARN("Thread %d: Queues not empty after completion! AIC=%d, AIV=%d",
                            thread_idx, aic_remaining, aiv_remaining);
                }
                break;  // Exit main loop
            }

            // Counter reached but cores still working, continue main loop to process them
            verification_warning_count++;
            if (verification_warning_count > MAX_VERIFICATION_WARNINGS) {
                LOG_ERROR("Thread %d: Counter reached but cores still working after %d checks!",
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

                LOG_INFO("Thread %d: Core %d completed task %d", thread_idx, core_id, task_id);

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
                        if (dep->core_type == CoreType::AIC) {  // AIC task
                            std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                            // Enqueue to tail position (circular)
                            ready_queue_aic_[ready_queue_aic_tail_] = dep_id;
                            ready_queue_aic_tail_ = (ready_queue_aic_tail_ + 1) % RUNTIME_MAX_TASKS;
                            ready_count_aic_.fetch_add(1, std::memory_order_release);
                            LOG_INFO("Thread %d: Task %d became ready -> AIC queue (tail=%d)",
                                     thread_idx, dep_id, ready_queue_aic_tail_);
                        } else {  // AIV task
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            // Enqueue to tail position (circular)
                            ready_queue_aiv_[ready_queue_aiv_tail_] = dep_id;
                            ready_queue_aiv_tail_ = (ready_queue_aiv_tail_ + 1) % RUNTIME_MAX_TASKS;
                            ready_count_aiv_.fetch_add(1, std::memory_order_release);
                            LOG_INFO("Thread %d: Task %d became ready -> AIV queue (tail=%d)",
                                     thread_idx, dep_id, ready_queue_aiv_tail_);
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
                    if (h->core_type == CoreType::AIC) {  // AIC core
                        if (ready_count_aic_.load(std::memory_order_acquire) > 0) {
                            std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                            int count = ready_count_aic_.load(std::memory_order_relaxed);
                            if (count > 0) {
                                // Dequeue from head position (circular) - FIFO order
                                int task_id = ready_queue_aic_[ready_queue_aic_head_];
                                ready_queue_aic_head_ = (ready_queue_aic_head_ + 1) % RUNTIME_MAX_TASKS;
                                ready_count_aic_.fetch_sub(1, std::memory_order_release);
                                Task* task = runtime.get_task(task_id);

                                LOG_INFO("Thread %d: Dispatching AIC task %d to core %d (head=%d)",
                                         thread_idx, task_id, core_id, ready_queue_aic_head_);

                                h->task = reinterpret_cast<uint64_t>(task);
                                h->task_status = 1;  // Mark as busy
                                cur_thread_tasks_in_flight++;
                                made_progress = true;
                            }
                        }
                    } else if (h->core_type == CoreType::AIV) {  // AIV core
                        if (ready_count_aiv_.load(std::memory_order_acquire) > 0) {
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            int count = ready_count_aiv_.load(std::memory_order_relaxed);
                            if (count > 0) {
                                // Dequeue from head position (circular) - FIFO order
                                int task_id = ready_queue_aiv_[ready_queue_aiv_head_];
                                ready_queue_aiv_head_ = (ready_queue_aiv_head_ + 1) % RUNTIME_MAX_TASKS;
                                ready_count_aiv_.fetch_sub(1, std::memory_order_release);
                                Task* task = runtime.get_task(task_id);

                                LOG_INFO("Thread %d: Dispatching AIV task %d to core %d (head=%d)",
                                         thread_idx, task_id, core_id, ready_queue_aiv_head_);

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
                LOG_WARN("Thread %d: %d idle iterations, progress %d/%d tasks",
                        thread_idx, idle_iterations, current, task_count);
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                LOG_ERROR("Thread %d: Timeout after %d idle iterations!", thread_idx, idle_iterations);
                diagnose_stuck_state(runtime, thread_idx, cur_thread_cores, core_num, hank);
                return -1;
            }
        } else {
            idle_iterations = 0;
        }
    }

    LOG_INFO("Thread %d: Execution complete, completed %d tasks", thread_idx, cur_thread_completed);
    return cur_thread_completed;
}

int AicpuExecutor::run(Runtime* runtime) {
    int thread_idx = thread_idx_++;

    LOG_INFO("Thread %d: Start", thread_idx);

    const int* cur_thread_cores = core_assignments_[thread_idx];

    // Handshaking is already done in init() - no per-thread handshake needed
    LOG_INFO("Thread %d: Runtime has %d tasks", thread_idx, runtime->get_task_count());
    int completed = resolve_and_dispatch(*runtime, thread_idx, cur_thread_cores, thread_cores_num_);
    LOG_INFO("Thread %d: Executed %d tasks from runtime", thread_idx, completed);

    int rc = shutdown_aicore(runtime, thread_idx, cur_thread_cores);
    if (rc != 0) {
        return rc;
    }

    LOG_INFO("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        LOG_INFO("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return 0;
}

void AicpuExecutor::deinit() {
    // Cleanup runtime execution state
    ready_count_aic_.store(0, std::memory_order_release);
    ready_count_aiv_.store(0, std::memory_order_release);

    // Reset circular queue indices
    ready_queue_aic_head_ = 0;
    ready_queue_aic_tail_ = 0;
    ready_queue_aiv_head_ = 0;
    ready_queue_aiv_tail_ = 0;

    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);

    LOG_INFO("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    LOG_INFO("DeInit: AicpuExecutor reset complete");
}

void AicpuExecutor::diagnose_stuck_state(Runtime& runtime, int thread_idx,
                                         const int* cur_thread_cores, int core_num,
                                         Handshake* hank) {
    LOG_ERROR("========== DIAGNOSTIC REPORT: Thread %d ==========", thread_idx);

    int completed = completed_tasks_.load(std::memory_order_acquire);
    int total = total_tasks_.load(std::memory_order_acquire);
    LOG_ERROR("Progress: %d/%d tasks (%.1f%%)",
             completed, total, total > 0 ? completed * 100.0 / total : 0.0);

    int aic_ready = ready_count_aic_.load(std::memory_order_acquire);
    int aiv_ready = ready_count_aiv_.load(std::memory_order_acquire);
    LOG_ERROR("Ready Queues: AIC=%d, AIV=%d", aic_ready, aiv_ready);

    int busy_cores = 0;
    int idle_cores = 0;
    int anomaly_cores = 0;

    LOG_ERROR("Core Status:");
    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* h = &hank[core_id];

        const char* core_type_str = core_type_to_string(h->core_type);

        if (h->task != 0) {
            Task* task = reinterpret_cast<Task*>(h->task);
            busy_cores++;

            LOG_ERROR("  Core %d [%s, BUSY]: task_id=%d, func_id=%d, fanin=%d, fanout=%d",
                     core_id, core_type_str,
                     task->task_id, task->func_id,
                     task->fanin.load(std::memory_order_acquire),
                     task->fanout_count);
        } else if (h->task_status != 0) {
            anomaly_cores++;
            LOG_ERROR("  Core %d [%s, ANOMALY]: status=BUSY but task=NULL", core_id, core_type_str);
        } else {
            idle_cores++;
        }
    }

    LOG_ERROR("Summary: %d busy, %d idle, %d anomaly", busy_cores, idle_cores, anomaly_cores);

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
 * @param runtime Pointer to Runtime structure containing:
 *                - workers[]: handshake buffers for AICPU-AICore communication
 *                - block_dim, sche_cpu_num: execution parameters
 *                - tasks[]: task runtime to execute
 * @return 0 on success, non-zero on error
 */
extern "C" int aicpu_execute(Runtime* runtime) {
    // Initialize log switches (only once, thread-safe)
    static std::once_flag log_init_flag;
    std::call_once(log_init_flag, []() {
        init_log_switch();
    });

    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid runtime argument: null pointer");
        return -1;
    }

    LOG_INFO("%s", "aicpu_execute: Starting AICPU kernel execution");

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
