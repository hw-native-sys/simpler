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

// Include C headers - they have their own extern "C" guards
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Device orchestration function signature (loaded via dlopen)
typedef void (*DeviceOrchestrationFunc)(void* sm_ptr, uint64_t* args, int arg_count);

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
            // Pass pointer to the TensorDescriptor (in task-owned storage), not the raw buffer address.
            // Kernels expect args[i] to be a TensorDescriptor* from which they read buffer.addr.
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

            // Get the orchestration entry function
            dlerror();  // Clear any existing error before dlsym
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
                DEV_ERROR("Thread 3: dlsym returned NULL (no error)");
                dlclose(handle);
                unlink(so_path);
                return -1;
            }

            DEV_INFO("Thread 3: Calling aicpu_orchestration_entry from SO");
            DEV_INFO("Thread 3: sm_ptr=%p, arg_count=%d", runtime->get_pto2_gm_sm_ptr(), runtime->get_orch_arg_count());
            uint64_t* args = runtime->get_orch_args();
            int arg_count = runtime->get_orch_arg_count();
            for (int i = 0; i < arg_count && i < 20; i++) {
                DEV_INFO("Thread 3: args[%d] = 0x%lx", i, args[i]);
            }

            orch_func(runtime->get_pto2_gm_sm_ptr(),
                      args,
                      arg_count);
            DEV_INFO("Thread 3: aicpu_orchestration_entry returned");

            // Store the SO handle and path - defer dlclose and unlink until all tasks complete
            // The orchestration SO contains static buffers (s_gm_heap_stub) that are
            // used by tasks as output buffers. Closing the SO prematurely would
            // invalidate those buffers and cause segfaults.
            orch_so_handle_ = handle;
            strncpy(orch_so_path_, so_path, sizeof(orch_so_path_) - 1);
            orch_so_path_[sizeof(orch_so_path_) - 1] = '\0';

            // Device mode: task count lives in PTO2 shared memory (current_task_index at offset 0)
            void* sm = runtime->get_pto2_gm_sm_ptr();
            int32_t pto2_task_count = sm ? *(volatile int32_t*)sm : 0;
            DEV_INFO("Thread 3: PTO2 task count = %d", pto2_task_count);
            total_tasks_.store(pto2_task_count, std::memory_order_release);
            pto2_init_done_.store(false, std::memory_order_release);
            pto2_init_complete_.store(false, std::memory_order_release);  // so workers re-init and wait this run
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
    // NOTE: Do NOT close the orchestration SO handle here!
    // The SO contains static buffers (s_gm_heap_stub) that are used as task
    // output buffers. These buffers are referenced by graph_output_ptr in
    // the PTO2 shared memory header, which is read during validate_runtime_impl
    // (copy-back phase) that runs AFTER all AICPU threads finish.
    //
    // Closing the SO here would unmap s_gm_heap_stub, causing a segfault when
    // validate_runtime_impl tries to copy from graph_output_ptr.
    //
    // In simulation mode, we let the SO "leak" - the OS will clean it up when
    // the process exits. For production (a2a3), the SO runs in device memory
    // which has a different lifecycle.
    if (orch_so_handle_ != nullptr) {
        DEV_INFO("DeInit: Keeping orchestration SO handle open for copy-back phase");
        // Do NOT call dlclose here - the SO must remain loaded until
        // validate_runtime_impl completes
        // TODO: Add a separate cleanup phase after validate_runtime_impl that calls:
        //   dlclose(orch_so_handle_);
        //   if (orch_so_path_[0] != '\0') unlink(orch_so_path_);
    }

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
