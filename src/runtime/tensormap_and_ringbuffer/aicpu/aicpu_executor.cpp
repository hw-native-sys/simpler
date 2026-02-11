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
#include "runtime.h"
#include "pto2_dispatch_payload.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_shared_memory.h"

// Performance profiling headers
#include "common/perf_profiling.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"

// Device orchestration function signature (loaded via dlopen).
// The orchestration .so receives a PTO2Runtime* (with ops table populated)
// instead of a raw shared-memory pointer.
typedef void (*DeviceOrchestrationFunc)(PTO2Runtime* rt, uint64_t* args, int arg_count);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(uint64_t* args, int arg_count);

constexpr int MAX_AICPU_THREADS = 4;
constexpr int MAX_AIC_PER_THREAD = 24;
constexpr int MAX_AIV_PER_THREAD = 48;
constexpr int MAX_CORES_PER_THREAD = MAX_AIC_PER_THREAD + MAX_AIV_PER_THREAD;

// Maximum tasks for ready queue (PTO2 mode uses shared memory task count)
constexpr int AICPU_MAX_READY_TASKS = 16384;

// Core information for discovery (aligned with host_build_graph)
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
    int thread_cores_num_{0};  // Cores per scheduler thread (0 for orchestrator when thread_num_==4)
    int core_count_per_thread_[MAX_AICPU_THREADS];  // Actual core count per thread
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];

    // Core discovery arrays (aligned with host_build_graph)
    CoreInfo aic_cores_[MAX_CORES_PER_THREAD];
    CoreInfo aiv_cores_[MAX_CORES_PER_THREAD];
    int aic_count_{0};
    int aiv_count_{0};

    // ===== Task queue state (FIFO circular queue, aligned with host_build_graph) =====
    std::mutex ready_queue_aic_mutex_;
    int ready_queue_aic_[AICPU_MAX_READY_TASKS];
    std::atomic<int> ready_count_aic_{0};
    int ready_queue_aic_head_{0};  // Circular queue: read position (front)
    int ready_queue_aic_tail_{0};  // Circular queue: write position (back)

    std::mutex ready_queue_aiv_mutex_;
    int ready_queue_aiv_[AICPU_MAX_READY_TASKS];
    std::atomic<int> ready_count_aiv_{0};
    int ready_queue_aiv_head_{0};  // Circular queue: read position (front)
    int ready_queue_aiv_tail_{0};  // Circular queue: write position (back)

    // Task execution tracking
    std::atomic<int> completed_tasks_{0};
    std::atomic<int> total_tasks_{0};
    std::atomic<int> finished_count_{0};
    // Device orchestration: set by Thread 3 when graph is built; workers wait for it
    std::atomic<bool> orchestrator_done_{false};
    std::atomic<bool> pto2_init_done_{false};
    std::atomic<bool> pto2_init_complete_{false};  // init block finished; others wait for this

    // Orchestration SO handle - defer dlclose until all tasks complete
    void* orch_so_handle_{nullptr};
    char orch_so_path_[256]{};  // Path to orchestration SO file for cleanup

    // ===== Performance profiling state =====
    std::mutex perf_ready_queue_mutex_;  // Protects enqueue_ready_buffer operations

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

    // Performance profiling methods
    void init_performance_profiling(Runtime* runtime);
    void complete_perf_records(Runtime* runtime, PerfBuffer* perf_buf,
                               PTO2TaskDescriptor* task_descriptors,
                               PTO2DepListEntry* dep_list_pool,
                               int32_t window_mask);
    void switch_perf_buffer(Runtime* runtime, int core_id, int thread_idx,
                           PTO2TaskDescriptor* task_descriptors,
                           PTO2DepListEntry* dep_list_pool,
                           int32_t window_mask);
    int enqueue_ready_buffer(PerfDataHeader* header, uint32_t core_index, uint32_t buffer_id);
    void flush_performance_buffers(Runtime* runtime, int thread_idx,
                                  const int* cur_thread_cores, int core_num,
                                  PTO2TaskDescriptor* task_descriptors,
                                  PTO2DepListEntry* dep_list_pool,
                                  int32_t window_mask);
};

static AicpuExecutor g_aicpu_executor;

// PTO2 device-mode state (shared memory view + per-task fanin refcount)
static constexpr int PTO2_MAX_SLOTS = 16384;
static int s_pto2_fanin_refcount[PTO2_MAX_SLOTS];
static PTO2DispatchPayload s_pto2_payload_per_core[RUNTIME_MAX_WORKER];

// ===== AicpuExecutor Method Implementations =====

/**
 * Handshake with all cores and discover their types
 * (Aligned with host_build_graph mechanism)
 */
int AicpuExecutor::handshake_all_cores(Runtime* runtime) {
    Handshake* all_hanks = (Handshake*)runtime->workers;
    cores_total_num_ = runtime->worker_count;

    aic_count_ = 0;
    aiv_count_ = 0;

    DEV_INFO("Handshaking with %d cores", cores_total_num_);

    // Step 1: Send handshake signal to all cores
    for (int i = 0; i < cores_total_num_; i++) {
        all_hanks[i].aicpu_ready = 1;
    }

    // Step 2: Wait for all cores to respond and collect core type
    for (int i = 0; i < cores_total_num_; i++) {
        Handshake* hank = &all_hanks[i];
        while (hank->aicore_done == 0) {
            // Spin wait for core to respond
        }

        CoreType type = hank->core_type;
        if (type == CoreType::AIC) {
            aic_cores_[aic_count_].worker_id = i;
            aic_cores_[aic_count_].core_type = type;
            aic_count_++;
            DEV_INFO("Core %d: AIC", i);
        } else {
            aiv_cores_[aiv_count_].worker_id = i;
            aiv_cores_[aiv_count_].core_type = type;
            aiv_count_++;
            DEV_INFO("Core %d: AIV", i);
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
    ready_count_aic_.store(0, std::memory_order_release);
    ready_count_aiv_.store(0, std::memory_order_release);
    ready_queue_aic_head_ = 0;
    ready_queue_aic_tail_ = 0;
    ready_queue_aiv_head_ = 0;
    ready_queue_aiv_tail_ = 0;

    DEV_INFO("Init: PTO2 mode, task count from shared memory");

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Shutdown AICore - Send quit signal to all AICore kernels
 */
int AicpuExecutor::shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
    if (core_num == 0) return 0;

    Handshake* all_hanks = (Handshake*)runtime->workers;

    DEV_INFO("Thread %d: Shutting down %d cores", thread_idx, core_num);

    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* hank = &all_hanks[core_id];
        DEV_INFO("Thread %d: AICPU hank addr = 0x%lx", thread_idx, (uint64_t)hank);
        hank->control = 1;
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
            out->args[n++] = reinterpret_cast<uint64_t>(task->params[i].tensor);
        }
    }

    out->num_args = n;
    DEV_INFO("build_pto2_payload ok");
    for (int i = 0; i < task->param_count; i++) {
        if (task->params[i].type == PTOParamType::SCALAR) {
            DEV_INFO("build_pto2_payload param %d scalar: %d", i, out->args[i]);
        } else {
            DEV_INFO("build_pto2_payload param %d addr: %x", i, out->args[i]);
        }
    }
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
    int32_t task_count = total_tasks_.load(std::memory_order_acquire);
    int32_t window_mask = window_size - 1;

    Handshake* hank = static_cast<Handshake*>(runtime->workers);
    DEV_INFO("Thread %d: hank=%p, task_count=%d, window_size=%d",
             thread_idx, (void*)hank, task_count, window_size);

    // One-time init: fanin_refcount and initial ready queue (one thread does it; others wait)
    if (!pto2_init_done_.exchange(true, std::memory_order_acq_rel)) {
        DEV_INFO("Thread %d: doing one-time init", thread_idx);
        std::memset(s_pto2_fanin_refcount, 0, sizeof(s_pto2_fanin_refcount));
        for (int32_t i = 0; i < task_count; i++) {
            PTO2TaskDescriptor* t = &task_descriptors[i & window_mask];
            int32_t fanin_count = __atomic_load_n(&t->fanin_count, __ATOMIC_ACQUIRE);
            if (fanin_count == 0) {
                int32_t wt = t->worker_type;
                if (wt == PTO2_WORKER_CUBE) {
                    std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                    // FIFO: enqueue to tail
                    ready_queue_aic_[ready_queue_aic_tail_] = i;
                    ready_queue_aic_tail_ = (ready_queue_aic_tail_ + 1) % AICPU_MAX_READY_TASKS;
                    ready_count_aic_.fetch_add(1, std::memory_order_release);
                } else {
                    std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                    // FIFO: enqueue to tail
                    ready_queue_aiv_[ready_queue_aiv_tail_] = i;
                    ready_queue_aiv_tail_ = (ready_queue_aiv_tail_ + 1) % AICPU_MAX_READY_TASKS;
                    ready_count_aiv_.fetch_add(1, std::memory_order_release);
                }
            }
        }
        // Performance profiling initialization (one-time, after all cores discovered)
        if (runtime->enable_profiling) {
            init_performance_profiling(runtime);
        }

        DEV_INFO("Thread %d: one-time init done", thread_idx);
        pto2_init_complete_.store(true, std::memory_order_release);
    } else {
        while (!pto2_init_complete_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    DEV_INFO("Thread %d: PTO2 dispatch starting with %d tasks, %d cores", thread_idx, task_count, core_num);
    int cur_thread_completed = 0;
    int cur_thread_tasks_in_flight = 0;
    int idle_iterations = 0;
    const int MAX_IDLE_ITERATIONS = 50000000;
    const int WARN_INTERVAL = 1000000;
    bool profiling_enabled = runtime->enable_profiling;

    while (true) {
        if (completed_tasks_.load(std::memory_order_acquire) >= task_count) {
            bool all_cores_idle = true;
            for (int i = 0; i < core_num; i++) {
                Handshake* h = &hank[cur_thread_cores[i]];
                if (h->task_status != 0 || h->task != 0) { all_cores_idle = false; break; }
            }
            if (all_cores_idle && orchestrator_done_.load(std::memory_order_acquire)) {
                int aic = ready_count_aic_.load(std::memory_order_acquire);
                int aiv = ready_count_aiv_.load(std::memory_order_acquire);
                if (aic > 0 || aiv > 0) {
                    DEV_WARN("Thread %d: Queues not empty at exit AIC=%d AIV=%d", thread_idx, aic, aiv);
                }
                break;
            }
        }

        bool made_progress = false;

        // Phase 1: Process completed tasks (Handshake.task = PTO2DispatchPayload*)
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];
            if (h->task_status == 0 && h->task != 0) {
                PTO2DispatchPayload* payload = reinterpret_cast<PTO2DispatchPayload*>(h->task);
                h->task = 0;

                // Check and switch performance buffer if needed
                if (profiling_enabled && h->perf_buffer_status == 1) {
                    switch_perf_buffer(runtime, core_id, thread_idx,
                                     task_descriptors, dep_list_pool, window_mask);
                }

                int32_t task_id = payload->task_id;
                PTO2TaskDescriptor* pto2_task = &task_descriptors[task_id & window_mask];

                DEV_INFO("Thread %d: Core %d completed PTO2 task %d", thread_idx, core_id, task_id);

                int32_t fanout_head = __atomic_load_n(&pto2_task->fanout_head, __ATOMIC_ACQUIRE);
                int32_t current = fanout_head;
                while (current > 0) {
                    PTO2DepListEntry* entry = &dep_list_pool[current];
                    int32_t consumer_id = entry->task_id;
                    int32_t consumer_slot = consumer_id & window_mask;
                    int prev = __atomic_fetch_add(&s_pto2_fanin_refcount[consumer_slot], 1, __ATOMIC_ACQ_REL);
                    PTO2TaskDescriptor* consumer_desc = &task_descriptors[consumer_slot];
                    int32_t fanin_count = __atomic_load_n(&consumer_desc->fanin_count, __ATOMIC_ACQUIRE);
                    if (prev + 1 == fanin_count) {
                        int32_t wt = consumer_desc->worker_type;
                        if (wt == PTO2_WORKER_CUBE) {
                            std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                            // FIFO: enqueue to tail
                            ready_queue_aic_[ready_queue_aic_tail_] = consumer_id;
                            ready_queue_aic_tail_ = (ready_queue_aic_tail_ + 1) % AICPU_MAX_READY_TASKS;
                            ready_count_aic_.fetch_add(1, std::memory_order_release);
                        } else {
                            std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                            // FIFO: enqueue to tail
                            ready_queue_aiv_[ready_queue_aiv_tail_] = consumer_id;
                            ready_queue_aiv_tail_ = (ready_queue_aiv_tail_ + 1) % AICPU_MAX_READY_TASKS;
                            ready_count_aiv_.fetch_add(1, std::memory_order_release);
                        }
                    }
                    current = entry->next_offset;
                }

                cur_thread_tasks_in_flight--;
                cur_thread_completed++;
                made_progress = true;
                completed_tasks_.fetch_add(1, std::memory_order_release);
            }
        }

        // Phase 2: Dispatch ready tasks to idle cores (build PTO2DispatchPayload)
        if (cur_thread_tasks_in_flight < core_num) {
            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                Handshake* h = &hank[core_id];
                if (h->task_status == 0 && h->task == 0) {
                    bool dispatched = false;
                    if (h->core_type == CoreType::AIC && ready_count_aic_.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lock(ready_queue_aic_mutex_);
                        int count = ready_count_aic_.load(std::memory_order_relaxed);
                        if (count > 0) {
                            // FIFO: dequeue from head
                            int32_t task_id = ready_queue_aic_[ready_queue_aic_head_];
                            ready_queue_aic_head_ = (ready_queue_aic_head_ + 1) % AICPU_MAX_READY_TASKS;
                            ready_count_aic_.fetch_sub(1, std::memory_order_release);
                            PTO2TaskDescriptor* task = &task_descriptors[task_id & window_mask];
                            PTO2DispatchPayload* payload = &s_pto2_payload_per_core[core_id];
                            build_pto2_payload(payload, runtime, task, task_descriptors, dep_list_pool, window_size);
                            h->task = reinterpret_cast<uint64_t>(payload);
                            h->task_status = 1;
                            cur_thread_tasks_in_flight++;
                            made_progress = true;
                            dispatched = true;
                            DEV_INFO("Thread %d: Dispatching PTO2 AIC task %d to core %d", thread_idx, task_id, core_id);
                        }
                    }
                    if (!dispatched && h->core_type == CoreType::AIV && ready_count_aiv_.load(std::memory_order_acquire) > 0) {
                        std::lock_guard<std::mutex> lock(ready_queue_aiv_mutex_);
                        int count = ready_count_aiv_.load(std::memory_order_relaxed);
                        if (count > 0) {
                            // FIFO: dequeue from head
                            int32_t task_id = ready_queue_aiv_[ready_queue_aiv_head_];
                            ready_queue_aiv_head_ = (ready_queue_aiv_head_ + 1) % AICPU_MAX_READY_TASKS;
                            ready_count_aiv_.fetch_sub(1, std::memory_order_release);
                            PTO2TaskDescriptor* task = &task_descriptors[task_id & window_mask];
                            PTO2DispatchPayload* payload = &s_pto2_payload_per_core[core_id];
                            build_pto2_payload(payload, runtime, task, task_descriptors, dep_list_pool, window_size);
                            h->task = reinterpret_cast<uint64_t>(payload);
                            h->task_status = 1;
                            cur_thread_tasks_in_flight++;
                            made_progress = true;
                            DEV_INFO("Thread %d: Dispatching PTO2 AIV task %d to core %d", thread_idx, task_id, core_id);
                        }
                    }
                }
            }
        }

        if (!made_progress) {
            idle_iterations++;
            if (idle_iterations % WARN_INTERVAL == 0) {
                DEV_WARN("Thread %d: PTO2 %d idle iterations, %d/%d completed",
                        thread_idx, idle_iterations, completed_tasks_.load(std::memory_order_acquire), task_count);
            }
            if (idle_iterations > MAX_IDLE_ITERATIONS) {
                DEV_ERROR("Thread %d: PTO2 timeout after %d idle iterations", thread_idx, idle_iterations);
                return -1;
            }
            std::this_thread::yield();
        } else {
            idle_iterations = 0;
        }
    }

    DEV_INFO("Thread %d: PTO2 execution complete, completed %d tasks", thread_idx, cur_thread_completed);

    // Flush performance buffers for cores managed by this thread
    if (profiling_enabled) {
        flush_performance_buffers(runtime, thread_idx, cur_thread_cores, core_num,
                                 task_descriptors, dep_list_pool, window_mask);
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
            int32_t task_window_size = PTO2_TASK_WINDOW_SIZE;
            int32_t dep_list_pool_size = PTO2_DEP_LIST_POOL_SIZE;
            int32_t heap_size = PTO2_HEAP_SIZE;
            int expected_arg_count = 0;
            if (config_func) {
                PTO2OrchestrationConfig cfg = config_func(args, arg_count);
                task_window_size = cfg.task_window_size;
                dep_list_pool_size = cfg.dep_list_pool_size;
                heap_size = cfg.heap_size;
                expected_arg_count = cfg.expected_arg_count;
                DEV_INFO("Thread 3: Config: window=%d, dep_pool=%d, heap=%d, expected_args=%d",
                         task_window_size, dep_list_pool_size, heap_size, expected_arg_count);
            } else {
                DEV_INFO("Thread 3: No config function, using defaults");
            }

            if (expected_arg_count > 0 && arg_count < expected_arg_count) {
                DEV_ERROR("Thread 3: arg_count %d < expected %d", arg_count, expected_arg_count);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            // Extract GM heap from args (always last 2: gm_heap ptr, heap_size)
            void* sm_ptr = runtime->get_pto2_gm_sm_ptr();
            PTO2SharedMemoryHeader* header = static_cast<PTO2SharedMemoryHeader*>(sm_ptr);
            void* gm_heap = nullptr;
            int32_t gm_heap_size = heap_size;
            if (arg_count >= 2) {
                uint64_t heap_arg = args[arg_count - 2];
                uint64_t size_arg = args[arg_count - 1];
                if (heap_arg != 0 && size_arg != 0) {
                    gm_heap = reinterpret_cast<void*>(static_cast<uintptr_t>(heap_arg));
                    gm_heap_size = static_cast<int32_t>(size_arg & 0x7FFFFFFF);
                }
            }

            // Create shared memory handle and runtime (ops table populated inside)
            int32_t sm_size = pto2_sm_calculate_size(task_window_size, dep_list_pool_size);
            PTO2SharedMemoryHandle* sm_handle =
                pto2_sm_create_from_buffer(sm_ptr, sm_size, task_window_size,
                                            gm_heap_size, dep_list_pool_size);
            if (!sm_handle) {
                DEV_ERROR("Thread 3: Failed to create shared memory handle");
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            PTO2Runtime* rt = pto2_runtime_create_from_sm(PTO2_MODE_EXECUTE,
                                                            sm_handle, gm_heap, gm_heap_size);
            if (!rt) {
                DEV_ERROR("Thread 3: Failed to create PTO2Runtime");
                pto2_sm_destroy(sm_handle);
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            // Call orchestration wrapped in outer scope (matches old PTO2_ORCHESTRATION behavior)
            DEV_INFO("Thread 3: Calling aicpu_orchestration_entry from SO");
            PTO2_SCOPE(rt) {
                orch_func(rt, args, arg_count);
            }
            DEV_INFO("Thread 3: aicpu_orchestration_entry returned");

            // Teardown runtime
            pto2_rt_orchestration_done(rt);
            pto2_runtime_destroy(rt);
            header->orchestrator_done = 1;

            // The orchestration .so no longer contains static output buffers
            // (heap is managed by the executor), so we can close immediately
            dlclose(handle);
            unlink(so_path);

            // Device mode: task count lives in PTO2 shared memory (current_task_index at offset 0)
            void* sm = runtime->get_pto2_gm_sm_ptr();
            int32_t pto2_task_count = sm ? *(volatile int32_t*)sm : 0;
            DEV_INFO("Thread 3: PTO2 task count = %d", pto2_task_count);
            total_tasks_.store(pto2_task_count, std::memory_order_release);
            pto2_init_done_.store(false, std::memory_order_release);
            pto2_init_complete_.store(false, std::memory_order_release);
            orchestrator_done_.store(true, std::memory_order_release);
            DEV_INFO("Thread 3: Set orchestrator_done=true");
        }
        DEV_INFO("Thread 3: Orchestrator completed");
    } else {
        // Device orchestration: wait until Thread 3 has built the graph
        if (thread_num_ == 4 && !runtime->get_orch_built_on_host()) {
            while (!orchestrator_done_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
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
    ready_count_aic_.store(0, std::memory_order_release);
    ready_count_aiv_.store(0, std::memory_order_release);
    ready_queue_aic_head_ = 0;
    ready_queue_aic_tail_ = 0;
    ready_queue_aiv_head_ = 0;
    ready_queue_aiv_tail_ = 0;
    completed_tasks_.store(0, std::memory_order_release);
    total_tasks_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);
    orchestrator_done_.store(false, std::memory_order_release);
    pto2_init_done_.store(false, std::memory_order_release);
    pto2_init_complete_.store(false, std::memory_order_release);

    // Reset core discovery state
    aic_count_ = 0;
    aiv_count_ = 0;

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
    (void)runtime;  // Reserved for future use
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

        const char* core_type_str = core_type_to_string(h->core_type);

        if (h->task != 0) {
            PTO2DispatchPayload* payload = reinterpret_cast<PTO2DispatchPayload*>(h->task);
            busy_cores++;

            DEV_ERROR("  Core %d [%s, BUSY]: task_id=%d, kernel_id=%d",
                     core_id, core_type_str,
                     payload->task_id, payload->kernel_id);
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
        DEV_ERROR("Check PTO2 shared memory for task dependency state");
    } else if (busy_cores > 0) {
        DEV_ERROR("*** LIVELOCK / HUNG TASK ***");
        DEV_ERROR("%d cores executing but no progress", busy_cores);
    }

    DEV_ERROR("========== END DIAGNOSTIC ==========");
}

// =============================================================================
// Performance Profiling Methods
// =============================================================================

/**
 * Initialize performance profiling for all cores
 *
 * Called once in resolve_and_dispatch_pto2() after one-time init.
 * Assigns buffer1 to each core and sets initial states.
 * Also writes total_tasks to PerfDataHeader for Host access.
 */
void AicpuExecutor::init_performance_profiling(Runtime* runtime) {
    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        LOG_ERROR("perf_data_base is NULL, cannot initialize profiling");
        return;
    }

    PerfDataHeader* header = get_perf_header(perf_base);
    DoubleBuffer* buffers = get_double_buffers(perf_base);

    // Write total_tasks to shared memory header for Host access
    // This is necessary because Runtime object is not copied back from Device to Host
    int32_t task_count = total_tasks_.load(std::memory_order_acquire);
    header->total_tasks = static_cast<uint32_t>(task_count);
    wmb();  // Ensure total_tasks is visible to Host

    LOG_INFO("Initializing performance profiling for %d cores, total_tasks=%d", runtime->worker_count, task_count);

    // Assign initial buffer (buffer1) to each AICore
    for (int i = 0; i < runtime->worker_count; i++) {
        Handshake* h = &runtime->workers[i];
        DoubleBuffer* db = &buffers[i];

        // Read memory barrier before checking buffer1 status
        rmb();
        if (db->buffer1_status != BufferStatus::IDLE) {
            LOG_WARN("Core %d: buffer1 not idle (status=%u)", i, static_cast<uint32_t>(db->buffer1_status));
        }

        // Assign buffer1 to AICore
        h->perf_records_addr = (uint64_t)&db->buffer1;
        h->perf_buffer_status = 0;  // 0 = can write

        // Write barrier: ensure writes visible to AICore before changing status
        wmb();
        db->buffer1_status = BufferStatus::WRITING;

        LOG_INFO("Core %d: assigned buffer1 (addr=0x%lx)", i, h->perf_records_addr);
    }

    LOG_INFO("Performance profiling initialized for %d cores", runtime->worker_count);
}

/**
 * Complete performance records by filling fanout information
 *
 * This function is called by AICPU to fill in fanout information that
 * was not recorded by AICore. Duration is NOT calculated here - it will
 * be calculated by Host when printing/processing the data.
 *
 * Key difference from host_build_graph: fanout is stored as a linked list
 * in PTO2 shared memory, not as an array in Task structure.
 *
 * Called in two places:
 * 1. switch_perf_buffer() - when switching buffers during normal operation
 * 2. flush_performance_buffers() - when flushing buffers during shutdown
 *
 * @param runtime Runtime instance (unused but kept for API consistency)
 * @param perf_buf PerfBuffer to be completed with fanout data
 * @param task_descriptors Pointer to task descriptor array in shared memory
 * @param dep_list_pool Pointer to dependency list pool in shared memory
 * @param window_mask Mask for computing task slot (window_size - 1)
 */
void AicpuExecutor::complete_perf_records(Runtime* runtime, PerfBuffer* perf_buf,
                                          PTO2TaskDescriptor* task_descriptors,
                                          PTO2DepListEntry* dep_list_pool,
                                          int32_t window_mask) {
    (void)runtime;  // Unused parameter
    uint32_t count = perf_buf->count;

    for (uint32_t i = 0; i < count; i++) {
        PerfRecord* record = &perf_buf->records[i];
        int32_t task_id = record->task_id;

        // Get TaskDescriptor from PTO2 shared memory
        int32_t slot = task_id & window_mask;
        PTO2TaskDescriptor* task = &task_descriptors[slot];

        // Fill fanout information by traversing the linked list
        record->fanout_count = 0;
        int32_t fanout_offset = task->fanout_head;

        while (fanout_offset != 0 && record->fanout_count < RUNTIME_MAX_FANOUT) {
            PTO2DepListEntry* entry = &dep_list_pool[fanout_offset];
            record->fanout[record->fanout_count++] = entry->task_id;
            fanout_offset = entry->next_offset;
        }
    }

    // Write memory barrier: ensure fanout data is visible to Host
    wmb();
}

/**
 * Switch performance buffer for a core
 *
 * Called when perf_buffer_status == 1 (buffer full).
 * Determines which buffer is full by address comparison,
 * then switches to the alternate buffer if available.
 *
 * @param runtime Runtime instance
 * @param core_id AICore ID
 * @param thread_idx AICPU thread ID (for logging)
 * @param task_descriptors Pointer to task descriptor array in shared memory
 * @param dep_list_pool Pointer to dependency list pool in shared memory
 * @param window_mask Mask for computing task slot (window_size - 1)
 */
void AicpuExecutor::switch_perf_buffer(Runtime* runtime, int core_id, int thread_idx,
                                       PTO2TaskDescriptor* task_descriptors,
                                       PTO2DepListEntry* dep_list_pool,
                                       int32_t window_mask) {
    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        return;
    }

    Handshake* h = &runtime->workers[core_id];
    PerfDataHeader* header = get_perf_header(perf_base);
    DoubleBuffer* db = get_core_double_buffer(perf_base, core_id);

    // Determine if current buffer is buffer1 or buffer2 by address comparison
    uint64_t current_addr = h->perf_records_addr;
    uint64_t buffer1_addr = (uint64_t)&db->buffer1;
    uint64_t buffer2_addr = (uint64_t)&db->buffer2;

    uint32_t full_buffer_id = 0;
    PerfBuffer* full_buf = nullptr;
    volatile BufferStatus* full_status_ptr = nullptr;
    PerfBuffer* alternate_buf = nullptr;
    volatile BufferStatus* alternate_status_ptr = nullptr;
    uint32_t alternate_buffer_id = 0;

    if (current_addr == buffer1_addr) {
        // Current buffer is buffer1, it's full
        full_buffer_id = 1;
        full_buf = &db->buffer1;
        full_status_ptr = &db->buffer1_status;
        alternate_buf = &db->buffer2;
        alternate_status_ptr = &db->buffer2_status;
        alternate_buffer_id = 2;
    } else if (current_addr == buffer2_addr) {
        // Current buffer is buffer2, it's full
        full_buffer_id = 2;
        full_buf = &db->buffer2;
        full_status_ptr = &db->buffer2_status;
        alternate_buf = &db->buffer1;
        alternate_status_ptr = &db->buffer1_status;
        alternate_buffer_id = 1;
    } else {
        LOG_ERROR("Thread %d: Core %d has invalid perf_records_addr=0x%lx",
                  thread_idx, core_id, current_addr);
        return;
    }

    LOG_INFO("Thread %d: Core %d buffer%u is full (count=%u)",
             thread_idx, core_id, full_buffer_id, full_buf->count);

    // Complete performance records by filling fanout information
    // (called before checking alternate buffer status to make data ready earlier)
    complete_perf_records(runtime, full_buf, task_descriptors, dep_list_pool, window_mask);

    // Read alternate buffer status (rmb needed, since status is modified by Host)
    rmb();

    BufferStatus alternate_status = *alternate_status_ptr;

    // If alternate buffer is not idle, spin wait for Host to finish reading
    if (alternate_status != BufferStatus::IDLE) {
        LOG_WARN("Thread %d: Core %d cannot switch, buffer%u status=%u, spinning until Host reads it",
                 thread_idx, core_id, alternate_buffer_id, static_cast<uint32_t>(alternate_status));

        // Spin wait: continuously check alternate buffer status until Host sets it to IDLE
        while (true) {
            rmb();  // Read barrier: ensure reading latest status modified by Host
            alternate_status = *alternate_status_ptr;

            if (alternate_status == BufferStatus::IDLE) {
                LOG_INFO("Thread %d: Core %d buffer%u now idle, proceeding with switch",
                         thread_idx, core_id, alternate_buffer_id);
                break;
            }
        }
    }

    // Alternate buffer is idle, can switch

    // Step 1: Enqueue full buffer to ready queue
    int enqueue_result = enqueue_ready_buffer(header, core_id, full_buffer_id);
    if (enqueue_result != 0) {
        LOG_WARN("Thread %d: Core %d failed to enqueue buffer%u (queue full)",
                 thread_idx, core_id, full_buffer_id);
        return;
    }

    // Step 2: Change full buffer status to READY (visible to Host)
    *full_status_ptr = BufferStatus::READY;
    // Step 3: Change alternate buffer status to WRITING (visible to Host)
    *alternate_status_ptr = BufferStatus::WRITING;
    wmb();  // Write barrier: ensure status changes visible to Host

    LOG_INFO("Thread %d: Core %d enqueued buffer%u", thread_idx, core_id, full_buffer_id);

    // Step 4: Switch perf_records_addr to alternate buffer (visible to AICore)
    h->perf_records_addr = (uint64_t)alternate_buf;

    // Step 5: Reset perf_buffer_status = 0 (notify AICore can continue writing)
    h->perf_buffer_status = 0;

    LOG_INFO("Thread %d: Core %d switched to buffer%u (status=0)",
             thread_idx, core_id, alternate_buffer_id);
}

/**
 * Enqueue a ready buffer to the queue
 *
 * Thread-safe: Uses mutex to protect queue operations since multiple
 * AICPU threads may enqueue concurrently.
 *
 * @return 0=success, -1=queue full
 */
int AicpuExecutor::enqueue_ready_buffer(PerfDataHeader* header, uint32_t core_index, uint32_t buffer_id) {
    std::lock_guard<std::mutex> lock(perf_ready_queue_mutex_);

    uint32_t capacity = PLATFORM_PROF_READYQUEUE_SIZE;

    // Read barrier: ensure reading latest tail value
    rmb();

    uint32_t current_tail = header->queue_tail;
    uint32_t current_head = header->queue_head;

    // Check if queue is full
    uint32_t next_tail = (current_tail + 1) % capacity;
    if (next_tail == current_head) {
        return -1;  // Queue full
    }

    // Enqueue entry
    header->queue[current_tail].core_index = core_index;
    header->queue[current_tail].buffer_id = buffer_id;
    header->queue_tail = next_tail;

    // Write memory barrier: ensure data written before updating tail, visible to Host
    wmb();

    return 0;
}

/**
 * Flush performance buffers for cores managed by this thread
 *
 * Called after shutdown_aicore to ensure all buffers with data
 * (even if not full) are enqueued for Host collection.
 *
 * For each core managed by this thread:
 * - Check which buffer is currently assigned (via perf_records_addr)
 * - If buffer has data (count > 0), mark it as READY and enqueue
 *
 * @param runtime Runtime instance
 * @param thread_idx Current thread index
 * @param cur_thread_cores Array of core IDs managed by this thread
 * @param core_num Number of cores managed by this thread
 * @param task_descriptors Pointer to task descriptor array in shared memory
 * @param dep_list_pool Pointer to dependency list pool in shared memory
 * @param window_mask Mask for computing task slot (window_size - 1)
 */
void AicpuExecutor::flush_performance_buffers(Runtime* runtime, int thread_idx,
                                              const int* cur_thread_cores, int core_num,
                                              PTO2TaskDescriptor* task_descriptors,
                                              PTO2DepListEntry* dep_list_pool,
                                              int32_t window_mask) {
    if (!runtime->enable_profiling) {
        return;
    }

    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        return;
    }

    PerfDataHeader* header = get_perf_header(perf_base);
    DoubleBuffer* buffers = get_double_buffers(perf_base);

    LOG_INFO("Thread %d: Flushing performance buffers for %d cores", thread_idx, core_num);

    int flushed_count = 0;

    // Only process cores managed by this thread
    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* h = &runtime->workers[core_id];
        DoubleBuffer* db = &buffers[core_id];

        // Read current buffer address
        uint64_t current_addr = h->perf_records_addr;
        if (current_addr == 0) {
            continue;  // No buffer assigned
        }

        // Determine which buffer is current
        uint64_t buf1_addr = (uint64_t)&db->buffer1;
        uint64_t buf2_addr = (uint64_t)&db->buffer2;

        PerfBuffer* current_buf = nullptr;
        volatile BufferStatus* current_status = nullptr;
        uint32_t buffer_id = 0;

        if (current_addr == buf1_addr) {
            current_buf = &db->buffer1;
            current_status = &db->buffer1_status;
            buffer_id = 1;
        } else if (current_addr == buf2_addr) {
            current_buf = &db->buffer2;
            current_status = &db->buffer2_status;
            buffer_id = 2;
        } else {
            LOG_WARN("Thread %d: Core %d perf_records_addr=0x%lx doesn't match buffer1=0x%lx or buffer2=0x%lx",
                     thread_idx, core_id, current_addr, buf1_addr, buf2_addr);
            continue;
        }

        // Read buffer count with memory barrier
        rmb();
        uint32_t count = current_buf->count;

        // If buffer has data, enqueue it
        if (count > 0) {
            // Complete performance records by filling fanout information before flush
            complete_perf_records(runtime, current_buf, task_descriptors, dep_list_pool, window_mask);

            // Mark buffer as READY
            *current_status = BufferStatus::READY;
            wmb();

            // Enqueue to ready queue
            int rc = enqueue_ready_buffer(header, core_id, buffer_id);
            if (rc == 0) {
                LOG_INFO("Thread %d: Core %d flushed buffer%d with %u records", thread_idx, core_id, buffer_id, count);
                flushed_count++;
            } else {
                LOG_WARN("Thread %d: Core %d failed to enqueue buffer%d (queue full)", thread_idx, core_id, buffer_id);
            }
        }
    }

    LOG_INFO("Thread %d: Performance buffer flush complete, %d buffers flushed", thread_idx, flushed_count);
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
 *                - worker_count, sche_cpu_num: execution parameters
 *                - PTO2 shared memory for task graph
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
