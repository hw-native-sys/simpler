#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "aicpu/device_log.h"
#include "common/platform_config.h"
#include "runtime.h"

constexpr int MAX_AICPU_THREADS = PLATFORM_MAX_AICPU_THREADS;
constexpr int MAX_AIC_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD;
constexpr int MAX_AIV_PER_THREAD = PLATFORM_MAX_AIV_PER_THREAD;
constexpr int MAX_CORES_PER_THREAD = PLATFORM_MAX_CORES_PER_THREAD;
constexpr int BUILDER_THREAD_NUM = 1;

// Best-effort per-thread context for logging (helps attribute builder activity).
static thread_local int tl_thread_idx = -1;
static thread_local const char* tl_thread_role = "unknown";

// Core information for discovery
struct CoreInfo {
    int worker_id;  // Index in runtime.workers[]
    CoreType core_type;
};

extern "C" int aicpu_runtime_add_task(
    Runtime* runtime, uint64_t* args, int num_args, int func_id, CoreType core_type, uint64_t function_bin_addr);
extern "C" void aicpu_runtime_add_successor_conditional(Runtime* runtime, int from_task, int to_task);
extern "C" void aicpu_runtime_publish_task(Runtime* runtime, int task_id);

namespace {
using AicpuBuilderFunc = int (*)(Runtime*);

int write_bytes_to_file(const char* path, const uint8_t* data, size_t size) {
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0755);
    if (fd < 0) {
        return -1;
    }
    size_t off = 0;
    while (off < size) {
        ssize_t n = ::write(fd, data + off, size - off);
        if (n <= 0) {
            ::close(fd);
            return -1;
        }
        off += static_cast<size_t>(n);
    }
    ::close(fd);
    return 0;
}

void ensure_current_so_is_global(int thread_idx) {
    static std::once_flag once;
    std::call_once(once, [&]() {
        Dl_info info;
        if (::dladdr(reinterpret_cast<void*>(&ensure_current_so_is_global), &info) == 0 || info.dli_fname == nullptr ||
            info.dli_fname[0] == '\0') {
            DEV_WARN("Thread %d: dladdr failed; cannot promote current AICPU runtime .so to RTLD_GLOBAL", thread_idx);
            return;
        }
        void* h = ::dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL);
        if (h == nullptr) {
            DEV_WARN("Thread %d: dlopen(self, RTLD_GLOBAL) failed: %s", thread_idx, ::dlerror());
            return;
        }
        DEV_INFO("Thread %d: Promoted current AICPU runtime .so to RTLD_GLOBAL: %s", thread_idx, info.dli_fname);
        // Intentionally leak `h` for process lifetime.
    });
}

int build_graph_via_aicpu_plugin(Runtime* runtime, int thread_idx) {
    if (runtime == nullptr) {
        return -1;
    }

    const void* so_data_v = runtime->get_aicpu_orch_so_data();
    size_t so_size = runtime->get_aicpu_orch_so_size();
    if (so_data_v == nullptr || so_size == 0) {
        DEV_ERROR("Thread %d: AICPU orch plugin not embedded (size=0). Host orchestration must embed plugin bytes.",
            thread_idx);
        return -1;
    }

    const char* sym = (runtime->aicpu_orch_func_name[0] != '\0') ? runtime->aicpu_orch_func_name : "build_graph_aicpu";
    const uint8_t* so_data = reinterpret_cast<const uint8_t*>(so_data_v);

    // On some real AICPU configurations, /dev/shm, /tmp, and memfd may be mounted `noexec`,
    // so we try multiple candidate directories that may allow dlopen() execution.
    const char* candidate_dirs[] = {
        "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device",
        "/usr/lib64",
        "/lib64",
        "/var/tmp",
        "/tmp",
    };
    constexpr int num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

    // Ensure the current runtime .so's symbols are visible to the dynamic loader.
    // This helps the plugin resolve `aicpu_runtime_*` symbols at dlopen time even
    // if the initial loader didn't use RTLD_GLOBAL.
    ensure_current_so_is_global(thread_idx);

    void* handle = nullptr;
    const char* last_err = nullptr;
    char so_path[256]{};
    for (int i = 0; i < num_candidates; ++i) {
        snprintf(so_path, sizeof(so_path), "%s/libaicpu_orch_%p_%d.so", candidate_dirs[i], (void*)runtime, thread_idx);

        DEV_INFO("Thread %d: Trying AICPU orch plugin path %s (bytes=%lu, sym=%s)",
            thread_idx,
            so_path,
            static_cast<uint64_t>(so_size),
            sym);

        if (write_bytes_to_file(so_path, so_data, so_size) != 0) {
            DEV_INFO("Thread %d: Cannot create/write plugin at %s (errno=%d), trying next", thread_idx, so_path, errno);
            continue;
        }

        handle = ::dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
        last_err = ::dlerror();
        ::unlink(so_path);
        if (handle != nullptr) {
            break;
        }
        DEV_INFO("Thread %d: dlopen failed for %s: %s", thread_idx, so_path, last_err ? last_err : "<null>");
    }

    if (handle == nullptr) {
        DEV_ERROR("Thread %d: dlopen failed for AICPU orch plugin in all candidate dirs: %s",
            thread_idx,
            last_err ? last_err : "<null>");
        return -1;
    }

    ::dlerror();  // clear
    AicpuBuilderFunc func = reinterpret_cast<AicpuBuilderFunc>(::dlsym(handle, sym));
    const char* err = ::dlerror();
    if (err != nullptr || func == nullptr) {
        DEV_ERROR("Thread %d: dlsym failed for '%s': %s", thread_idx, sym, err ? err : "<null>");
        ::dlclose(handle);
        return -1;
    }

    int rc = func(runtime);
    ::dlclose(handle);
    return rc;
}
}  // namespace

struct AicpuExecutor {
    // ===== Thread management state =====
    std::atomic<int> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int thread_num_{0};
    int cores_total_num_{0};
    int core_assignments_[MAX_AICPU_THREADS][MAX_CORES_PER_THREAD];
    int thread_core_counts_[MAX_AICPU_THREADS];
    int schedule_thread_num_{0};

    // Core discovery arrays (space-time tradeoff: avoid sorting)
    CoreInfo aic_cores_[MAX_CORES_PER_THREAD];
    CoreInfo aiv_cores_[MAX_CORES_PER_THREAD];
    int aic_count_{0};
    int aiv_count_{0};

    // Protects graph mutation + queue ops for concurrent build||schedule.
    std::mutex graph_mutex_;

    // ===== Task queue state =====
    std::mutex ready_queue_aic_mutex_;
    int ready_queue_aic_[RUNTIME_MAX_TASKS];
    std::atomic<int> ready_count_aic_{0};

    std::mutex ready_queue_aiv_mutex_;
    int ready_queue_aiv_[RUNTIME_MAX_TASKS];
    std::atomic<int> ready_count_aiv_{0};

    // Task execution tracking
    std::atomic<int> completed_tasks_{0};
    std::atomic<int> published_tasks_{0};
    std::atomic<bool> build_done_{false};
    std::atomic<bool> build_failed_{false};
    std::atomic<int> finished_count_{0};

    // ===== Methods =====
    int init(Runtime* runtime);
    int handshake_all_cores(Runtime* runtime);
    void assign_cores_to_threads();
    bool pop_ready_task(Runtime& runtime, CoreType want, int* task_id_out);
    void push_ready_task(Runtime& runtime, int task_id);
    int resolve_and_dispatch(Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num);
    int run(Runtime* runtime);
    void deinit();
    void diagnose_stuck_state(
        Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num, Handshake* hank);
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

    // Read execution parameters from runtime
    thread_num_ = runtime->sche_cpu_num;
    schedule_thread_num_ = thread_num_ - BUILDER_THREAD_NUM;

    // Simplified defensive check
    if (thread_num_ < 2 || thread_num_ > MAX_AICPU_THREADS) {
        DEV_ERROR("Invalid thread_num: %d (valid range: 2-%d)", thread_num_, MAX_AICPU_THREADS);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (schedule_thread_num_ < 1) {
        DEV_ERROR("Invalid schedule_thread_num: %d (thread_num=%d, builder=%d)",
            schedule_thread_num_,
            thread_num_,
            BUILDER_THREAD_NUM);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Perform core discovery: handshake with all cores and collect core type information
    int rc = handshake_all_cores(runtime);
    if (rc != 0) {
        DEV_ERROR("Core discovery failed");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    for (int t = 0; t < MAX_AICPU_THREADS; t++) {
        thread_core_counts_[t] = 0;
    }

    DEV_INFO("Config: aicpu_threads=%d (builder=%d scheduler=%d), cores=%d",
        thread_num_,
        BUILDER_THREAD_NUM,
        schedule_thread_num_,
        cores_total_num_);

    // Assign discovered cores to threads
    assign_cores_to_threads();

    // Initialize AICPU graph-build API table for dlopen'd orchestration plugins.
    // This avoids requiring the plugin to resolve `aicpu_runtime_*` symbols via
    // the dynamic loader at plugin load time.
    runtime->aicpu_build_api.add_task = &aicpu_runtime_add_task;
    runtime->aicpu_build_api.add_successor_conditional = &aicpu_runtime_add_successor_conditional;
    runtime->aicpu_build_api.publish_task = &aicpu_runtime_publish_task;

    // Hard error: scheduler threads must have at least one assigned core.
    // Otherwise, they will spin in resolve_and_dispatch() and eventually timeout.
    for (int t = BUILDER_THREAD_NUM; t < thread_num_; ++t) {
        if (thread_core_counts_[t] <= 0) {
            DEV_ERROR(
                "Invalid core assignment: scheduler thread %d has core_num=%d (aic=%d aiv=%d total=%d "
                "sched_threads=%d)",
                t,
                thread_core_counts_[t],
                aic_count_,
                aiv_count_,
                cores_total_num_,
                schedule_thread_num_);
            init_failed_.store(true, std::memory_order_release);
            return -1;
        }
    }

    // Initialize runtime execution state
    completed_tasks_.store(0, std::memory_order_release);
    published_tasks_.store(0, std::memory_order_release);
    build_done_.store(false, std::memory_order_release);
    build_failed_.store(false, std::memory_order_release);
    ready_count_aic_.store(0, std::memory_order_release);
    ready_count_aiv_.store(0, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    DEV_INFO("AicpuExecutor: Init complete");
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
        DEV_ERROR("worker_count is 0, no cores to handshake");
        return -1;
    }

    // Simplified defensive check
    if (cores_total_num_ > MAX_CORES_PER_THREAD) {
        DEV_ERROR("Total cores %d exceeds maximum %d", cores_total_num_, MAX_CORES_PER_THREAD);
        return -1;
    }

    aic_count_ = 0;
    aiv_count_ = 0;

    DEV_INFO("Core Discovery: Handshaking with %d cores", cores_total_num_);

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
            DEV_ERROR("Unknown core type %d for core %d", static_cast<int>(type), i);
            return -1;
        }

        DEV_INFO("  Core %d: type=%s", i, core_type_to_string(type));
    }

    DEV_INFO("Discovery complete: AIC=%d, AIV=%d, Total=%d", aic_count_, aiv_count_, cores_total_num_);
    return 0;
}

/**
 * Assign discovered cores to scheduler threads.
 *
 * Thread 0 is the AICPU builder (no AICore workers assigned).
 * Scheduler threads receive a best-effort balanced distribution of AIC and AIV
 * cores (no strict divisibility requirement).
 */
void AicpuExecutor::assign_cores_to_threads() {
    for (int t = 0; t < thread_num_; t++) {
        thread_core_counts_[t] = 0;
    }

    // Builder threads get no AICore workers.
    for (int t = 0; t < BUILDER_THREAD_NUM && t < thread_num_; t++) {
        thread_core_counts_[t] = 0;
    }

    int sched = schedule_thread_num_;
    if (sched <= 0) {
        DEV_ERROR("assign_cores_to_threads: schedule_thread_num=%d", sched);
        init_failed_.store(true, std::memory_order_release);
        return;
    }

    int aic_base = aic_count_ / sched;
    int aic_rem = aic_count_ % sched;
    int aiv_base = aiv_count_ / sched;
    int aiv_rem = aiv_count_ % sched;

    DEV_INFO("Core Assignment (scheduler only): AIC=%d, AIV=%d, sched_threads=%d", aic_count_, aiv_count_, sched);

    int aic_idx = 0;
    int aiv_idx = 0;

    for (int s = 0; s < sched; s++) {
        int t = s + BUILDER_THREAD_NUM;
        if (t >= thread_num_) {
            break;
        }

        int core_idx = 0;
        int aic_take = aic_base + (s < aic_rem ? 1 : 0);
        int aiv_take = aiv_base + (s < aiv_rem ? 1 : 0);

        for (int i = 0; i < aic_take; i++) {
            core_assignments_[t][core_idx++] = aic_cores_[aic_idx++].worker_id;
        }
        for (int i = 0; i < aiv_take; i++) {
            core_assignments_[t][core_idx++] = aiv_cores_[aiv_idx++].worker_id;
        }

        thread_core_counts_[t] = core_idx;

        char log_buffer[256];
        int offset = 0;
        offset += snprintf(
            log_buffer + offset, sizeof(log_buffer) - offset, "Thread %d: assigned %d cores - AIC[", t, core_idx);

        for (int i = 0; i < aic_take; i++) {
            if (i > 0) offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ",");
            offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "%d", core_assignments_[t][i]);
        }
        offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "] AIV[");

        for (int i = 0; i < aiv_take; i++) {
            if (i > 0) offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ",");
            offset +=
                snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "%d", core_assignments_[t][aic_take + i]);
        }
        offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "]");

        DEV_INFO("%s", log_buffer);
    }

    if (aic_idx != aic_count_ || aiv_idx != aiv_count_) {
        DEV_ERROR("Core Assignment mismatch: assigned AIC=%d/%d AIV=%d/%d", aic_idx, aic_count_, aiv_idx, aiv_count_);
        init_failed_.store(true, std::memory_order_release);
    }
}

/**
 * Shutdown AICore - Send quit signal to all AICore kernels
 */
int AicpuExecutor::shutdown_aicore(Runtime* runtime, int thread_idx, const int* cur_thread_cores, int core_num) {
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

bool AicpuExecutor::pop_ready_task(Runtime& runtime, CoreType want, int* task_id_out) {
    if (task_id_out == nullptr) {
        return false;
    }

    std::mutex& q_mutex = (want == CoreType::AIC) ? ready_queue_aic_mutex_ : ready_queue_aiv_mutex_;
    int* q = (want == CoreType::AIC) ? ready_queue_aic_ : ready_queue_aiv_;
    std::atomic<int>& q_count = (want == CoreType::AIC) ? ready_count_aic_ : ready_count_aiv_;

    std::scoped_lock lock(graph_mutex_, q_mutex);

    int cur = q_count.load(std::memory_order_acquire);
    while (cur > 0) {
        int task_id = q[cur - 1];
        cur--;

        Task* task = runtime.get_task(task_id);
        if (task == nullptr) {
            continue;
        }

        if (task->published.load(std::memory_order_acquire) == 0) {
            continue;
        }
        if (task->completed.load(std::memory_order_acquire) != 0) {
            continue;
        }
        if (task->fanin.load(std::memory_order_acquire) != 0) {
            continue;
        }

        q_count.store(cur, std::memory_order_release);
        *task_id_out = task_id;
        return true;
    }

    q_count.store(cur, std::memory_order_release);
    return false;
}

void AicpuExecutor::push_ready_task(Runtime& runtime, int task_id) {
    Task* task = runtime.get_task(task_id);
    if (task == nullptr) {
        return;
    }

    CoreType want = task->core_type;
    std::mutex& q_mutex = (want == CoreType::AIC) ? ready_queue_aic_mutex_ : ready_queue_aiv_mutex_;
    int* q = (want == CoreType::AIC) ? ready_queue_aic_ : ready_queue_aiv_;
    std::atomic<int>& q_count = (want == CoreType::AIC) ? ready_count_aic_ : ready_count_aiv_;

    std::scoped_lock lock(q_mutex);

    int cur = q_count.load(std::memory_order_acquire);
    if (cur >= RUNTIME_MAX_TASKS) {
        DEV_ERROR("Ready queue overflow for %s (task_id=%d)", core_type_to_string(want), task_id);
        return;
    }
    q[cur] = task_id;
    q_count.store(cur + 1, std::memory_order_release);
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

    // Timeout detection using idle iteration counting
    int idle_iterations = 0;
    const int MAX_IDLE_ITERATIONS = 1000000;
    const int WARN_INTERVAL = 100000;
    bool made_progress = false;

    int verification_warning_count = 0;
    const int MAX_VERIFICATION_WARNINGS = 10;

    int last_seen_published = -1;

    while (true) {
        if (build_failed_.load(std::memory_order_acquire)) {
            DEV_ERROR("Thread %d: build_failed set, aborting scheduling loop", thread_idx);
            return -1;
        }

        int published = published_tasks_.load(std::memory_order_acquire);
        int completed = completed_tasks_.load(std::memory_order_acquire);
        bool build_done = build_done_.load(std::memory_order_acquire);

        if (!build_done && published != last_seen_published) {
            DEV_INFO("Thread %d: Observed published=%d (completed=%d, build_done=%d)",
                thread_idx,
                published,
                completed,
                build_done ? 1 : 0);
            last_seen_published = published;
        }

        if (build_done && completed >= published) {
            bool all_cores_idle = true;
            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                Handshake* h = &hank[core_id];
                if (h->task_status != 0 || h->task != 0) {
                    all_cores_idle = false;
                    if (verification_warning_count == 0) {
                        DEV_WARN("Thread %d: Counter reached %d/%d but core %d still has work (status=%d, task=%p)",
                            thread_idx,
                            completed,
                            published,
                            core_id,
                            h->task_status,
                            (void*)h->task);
                    }
                    break;
                }
            }

            if (all_cores_idle) {
                int aic_remaining = ready_count_aic_.load(std::memory_order_acquire);
                int aiv_remaining = ready_count_aiv_.load(std::memory_order_acquire);
                if (aic_remaining == 0 && aiv_remaining == 0) {
                    break;
                }
            }

            verification_warning_count++;
            if (verification_warning_count > MAX_VERIFICATION_WARNINGS) {
                DEV_ERROR("Thread %d: Counter reached but cores still working after %d checks!",
                    thread_idx,
                    verification_warning_count);
                diagnose_stuck_state(runtime, thread_idx, cur_thread_cores, core_num, hank);
                return -1;
            }
        }

        made_progress = false;

        // Phase 1: process completed tasks.
        for (int i = 0; i < core_num; i++) {
            int core_id = cur_thread_cores[i];
            Handshake* h = &hank[core_id];

            if (h->task_status == 0 && h->task != 0) {
                Task* task = reinterpret_cast<Task*>(h->task);
                h->task = 0;

                int task_id = task->task_id;
                DEV_INFO("Thread %d: Core %d completed task %d", thread_idx, core_id, task_id);

                {
                    std::scoped_lock lock(graph_mutex_);
                    task->completed.store(1, std::memory_order_release);

                    for (int j = 0; j < task->fanout_count; j++) {
                        int dep_id = task->fanout[j];
                        Task* dep = runtime.get_task(dep_id);
                        if (dep == nullptr) {
                            continue;
                        }

                        int prev_fanin = dep->fanin.fetch_sub(1, std::memory_order_acq_rel);
                        if (prev_fanin == 1) {
                            if (dep->published.load(std::memory_order_acquire) != 0) {
                                push_ready_task(runtime, dep_id);
                            }
                            DEV_INFO("Thread %d: Task %d becomes ready (published=%d)",
                                thread_idx,
                                dep_id,
                                dep->published.load(std::memory_order_acquire));
                        } else if (prev_fanin <= 0) {
                            DEV_WARN("Thread %d: Task %d fanin underflow (prev=%d)", thread_idx, dep_id, prev_fanin);
                        }
                    }
                }

                cur_thread_tasks_in_flight--;
                cur_thread_completed++;
                made_progress = true;
                completed_tasks_.fetch_add(1, std::memory_order_release);
            }
        }

        // Phase 2: dispatch ready tasks to idle cores.
        if (cur_thread_tasks_in_flight < core_num) {
            for (int i = 0; i < core_num; i++) {
                int core_id = cur_thread_cores[i];
                Handshake* h = &hank[core_id];

                if (h->task_status == 0 && h->task == 0) {
                    int task_id = -1;
                    if (!pop_ready_task(runtime, h->core_type, &task_id)) {
                        continue;
                    }

                    Task* task = runtime.get_task(task_id);
                    if (task == nullptr) {
                        continue;
                    }
                    if (task->function_bin_addr == 0) {
                        DEV_ERROR(
                            "Thread %d: Task %d has function_bin_addr==0, refusing to dispatch", thread_idx, task_id);
                        return -1;
                    }

                    DEV_INFO("Thread %d: Dispatching %s task %d to core %d",
                        thread_idx,
                        core_type_to_string(h->core_type),
                        task_id,
                        core_id);
                    h->task = reinterpret_cast<uint64_t>(task);
                    h->task_status = 1;
                    cur_thread_tasks_in_flight++;
                    made_progress = true;
                }
            }
        }

        if (!made_progress) {
            idle_iterations++;
            if (idle_iterations % WARN_INTERVAL == 0) {
                int current_completed = completed_tasks_.load(std::memory_order_acquire);
                int current_published = published_tasks_.load(std::memory_order_acquire);
                DEV_WARN("Thread %d: %d idle iterations, progress %d/%d tasks (build_done=%d)",
                    thread_idx,
                    idle_iterations,
                    current_completed,
                    current_published,
                    build_done_.load(std::memory_order_acquire) ? 1 : 0);
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
    int final_rc = 0;

    tl_thread_idx = thread_idx;
    tl_thread_role = (thread_idx < BUILDER_THREAD_NUM) ? "builder" : "scheduler";

    DEV_INFO("Thread %d: Start", thread_idx);

    if (thread_idx < BUILDER_THREAD_NUM) {
        DEV_INFO("Thread %d: Builder starting (build_mode=%d)", thread_idx, runtime ? runtime->build_mode : -1);

        int rc = build_graph_via_aicpu_plugin(runtime, thread_idx);
        if (rc != 0) {
            DEV_ERROR("Thread %d: build_graph_aicpu failed rc=%d", thread_idx, rc);
            build_failed_.store(true, std::memory_order_release);
        }
        build_done_.store(true, std::memory_order_release);
        DEV_INFO("Thread %d: Builder done (rc=%d)", thread_idx, rc);
        final_rc = (rc == 0) ? 0 : -1;
    } else {
        DEV_INFO("Thread %d: Scheduler thread (build_mode=%d, build_done=%d, published=%d)",
            thread_idx,
            runtime ? runtime->build_mode : -1,
            build_done_.load(std::memory_order_acquire) ? 1 : 0,
            published_tasks_.load(std::memory_order_acquire));

        if (runtime->build_mode == 0) {
            DEV_INFO("Thread %d: Sequential mode: waiting for builder barrier", thread_idx);
            while (!build_done_.load(std::memory_order_acquire)) {
                if (build_failed_.load(std::memory_order_acquire)) {
                    DEV_ERROR("Thread %d: build_failed while waiting for sequential barrier", thread_idx);
                    break;
                }
            }
        } else {
            DEV_INFO("Thread %d: Concurrent mode: not waiting for builder barrier", thread_idx);
        }

        const int* cur_thread_cores = core_assignments_[thread_idx];
        int core_num = thread_core_counts_[thread_idx];

        if (core_num <= 0) {
            DEV_ERROR("Thread %d: Scheduler has core_num=%d, aborting", thread_idx, core_num);
            return -1;
        }

        // Handshaking is already done in init() - no per-thread handshake needed
        DEV_INFO("Thread %d: Scheduler starting (core_num=%d)", thread_idx, core_num);
        int sched_rc = resolve_and_dispatch(*runtime, thread_idx, cur_thread_cores, core_num);
        DEV_INFO("Thread %d: Scheduler finished (rc=%d)", thread_idx, sched_rc);

        int rc = shutdown_aicore(runtime, thread_idx, cur_thread_cores, core_num);
        if (rc != 0) {
            return rc;
        }

        DEV_INFO("Thread %d: Scheduler completed", thread_idx);
        final_rc = (sched_rc < 0) ? sched_rc : 0;
    }

    // Check if this is the last thread to finish
    int prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        DEV_INFO("Thread %d: Last thread, marking executor finished", thread_idx);
    }

    return final_rc;
}

void AicpuExecutor::deinit() {
    // Cleanup runtime execution state
    ready_count_aic_.store(0, std::memory_order_release);
    ready_count_aiv_.store(0, std::memory_order_release);
    completed_tasks_.store(0, std::memory_order_release);
    published_tasks_.store(0, std::memory_order_release);
    build_done_.store(false, std::memory_order_release);
    build_failed_.store(false, std::memory_order_release);
    finished_count_.store(0, std::memory_order_release);

    DEV_INFO("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    DEV_INFO("DeInit: AicpuExecutor reset complete");
}

void AicpuExecutor::diagnose_stuck_state(
    Runtime& runtime, int thread_idx, const int* cur_thread_cores, int core_num, Handshake* hank) {
    DEV_ERROR("========== DIAGNOSTIC REPORT: Thread %d ==========", thread_idx);

    int completed = completed_tasks_.load(std::memory_order_acquire);
    int published = published_tasks_.load(std::memory_order_acquire);
    DEV_ERROR("Progress: completed=%d published=%d build_done=%d build_failed=%d",
        completed,
        published,
        build_done_.load(std::memory_order_acquire) ? 1 : 0,
        build_failed_.load(std::memory_order_acquire) ? 1 : 0);

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
            Task* task = reinterpret_cast<Task*>(h->task);
            busy_cores++;

            DEV_ERROR("  Core %d [%s, BUSY]: task_id=%d, func_id=%d, fanin=%d, fanout=%d",
                core_id,
                core_type_str,
                task->task_id,
                task->func_id,
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
    if (busy_cores == 0 && aic_ready == 0 && aiv_ready == 0 && completed < published) {
        DEV_ERROR("*** DEADLOCK DETECTED ***");
        DEV_ERROR("All cores idle, no ready tasks, but %d tasks incomplete", published - completed);

        DEV_ERROR("Tasks with fanin > 0:");
        int stuck_count = 0;
        int task_count = runtime.get_task_count();
        for (int tid = 0; tid < task_count && stuck_count < 10; tid++) {
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

// ===== AICPU-side graph build helpers (exported C ABI) =====

/**
 * AICPU graph-build API: create a task that can be scheduled by the runtime.
 *
 * Notes:
 * - This function acquires the internal graph mutex, so it is safe to call while
 *   scheduler threads are concurrently dispatching/completing tasks.
 * - Passing `function_bin_addr == 0` means: bind the task using the runtime's
 *   `func_id -> kernel_addrs[]` table (populated by the host before launch).
 *   This is the recommended default for examples.
 */
extern "C" int aicpu_runtime_add_task(
    Runtime* runtime, uint64_t* args, int num_args, int func_id, CoreType core_type, uint64_t function_bin_addr) {
    if (runtime == nullptr) {
        return -1;
    }

    DEV_INFO("Thread %d(%s): add_task(func_id=%d core=%s num_args=%d addr=0x%lx)",
        tl_thread_idx,
        tl_thread_role,
        func_id,
        core_type_to_string(core_type),
        num_args,
        (uint64_t)function_bin_addr);

    std::scoped_lock lock(g_aicpu_executor.graph_mutex_);

    int task_id = runtime->add_task(args, num_args, func_id, core_type);
    if (task_id < 0) {
        return -1;
    }

    Task* task = runtime->get_task(task_id);
    if (task == nullptr) {
        return -1;
    }

    if (function_bin_addr == 0 && func_id >= 0 && func_id < RUNTIME_MAX_FUNC_ID) {
        function_bin_addr = runtime->kernel_addrs[func_id];
    }
    task->function_bin_addr = function_bin_addr;
    task->published.store(0, std::memory_order_release);
    task->completed.store(0, std::memory_order_release);

    DEV_INFO("Thread %d(%s): add_task -> task_id=%d fanin=%d bound_addr=0x%lx",
        tl_thread_idx,
        tl_thread_role,
        task_id,
        task->fanin.load(std::memory_order_acquire),
        (uint64_t)task->function_bin_addr);
    return task_id;
}

/**
 * AICPU graph-build API: add an edge `from_task -> to_task` for concurrent build.
 *
 * This is a concurrency-safe variant of `Runtime::add_successor()`:
 * - Always records the fanout edge.
 * - Only increments `to_task.fanin` if `from_task` has not already completed.
 *
 * Use this when the scheduler may complete `from_task` while the builder is still
 * constructing the graph.
 */
extern "C" void aicpu_runtime_add_successor_conditional(Runtime* runtime, int from_task, int to_task) {
    if (runtime == nullptr) {
        return;
    }

    DEV_INFO("Thread %d(%s): add_edge_conditional(%d -> %d)", tl_thread_idx, tl_thread_role, from_task, to_task);

    std::scoped_lock lock(g_aicpu_executor.graph_mutex_);
    runtime->add_successor_conditional(from_task, to_task);
}

/**
 * AICPU graph-build API: publish a task to the scheduler.
 *
 * Publishing makes the task visible to scheduler threads. If the task's fanin is
 * already zero at publish time, it is enqueued immediately into the ready queue.
 *
 * Typical pattern for concurrent build||schedule:
 * 1) Create task via `aicpu_runtime_add_task()`
 * 2) Add edges via `aicpu_runtime_add_successor_conditional()`
 * 3) Call `aicpu_runtime_publish_task()`
 */
extern "C" void aicpu_runtime_publish_task(Runtime* runtime, int task_id) {
    if (runtime == nullptr) {
        return;
    }

    DEV_INFO("Thread %d(%s): publish_task(%d)", tl_thread_idx, tl_thread_role, task_id);

    std::scoped_lock lock(g_aicpu_executor.graph_mutex_);
    Task* task = runtime->get_task(task_id);
    if (task == nullptr) {
        return;
    }

    int expected = 0;
    if (!task->published.compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
    }

    g_aicpu_executor.published_tasks_.fetch_add(1, std::memory_order_acq_rel);

    if (task->fanin.load(std::memory_order_acquire) == 0) {
        g_aicpu_executor.push_ready_task(*runtime, task_id);
    }

    DEV_INFO("Thread %d(%s): publish_task(%d) done (fanin=%d published=%d total_published=%d)",
        tl_thread_idx,
        tl_thread_role,
        task_id,
        task->fanin.load(std::memory_order_acquire),
        task->published.load(std::memory_order_acquire),
        g_aicpu_executor.published_tasks_.load(std::memory_order_acquire));
}
