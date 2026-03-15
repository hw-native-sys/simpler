/**
 * @file performance_collector.cpp
 * @brief Platform-agnostic performance data collector implementation
 *
 * Implements ProfMemoryManager (dynamic buffer management thread) and
 * PerformanceCollector (data collection and export).
 */

#include "host/performance_collector.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

// =============================================================================
// ProfMemoryManager Implementation
// =============================================================================

ProfMemoryManager::~ProfMemoryManager() {
    if (running_.load()) {
        stop();
    }
}

void ProfMemoryManager::start(void* shared_mem_host, int num_cores, int num_phase_threads,
                               PerfAllocCallback alloc_cb, PerfRegisterCallback register_cb,
                               PerfFreeCallback free_cb, void* user_data, int device_id) {
    shared_mem_host_ = shared_mem_host;
    num_cores_ = num_cores;
    num_phase_threads_ = num_phase_threads;
    alloc_cb_ = alloc_cb;
    register_cb_ = register_cb;
    free_cb_ = free_cb;
    user_data_ = user_data;
    device_id_ = device_id;

    running_.store(true);
    mgmt_thread_ = std::thread(&ProfMemoryManager::mgmt_loop, this);

    LOG_INFO("ProfMemoryManager started: %d cores, %d phase threads", num_cores, num_phase_threads);
}

void ProfMemoryManager::stop() {
    running_.store(false);
    if (mgmt_thread_.joinable()) {
        mgmt_thread_.join();
    }

    // Drain remaining done_queue and free buffers
    {
        std::lock_guard<std::mutex> lock(done_mutex_);
        while (!done_queue_.empty()) {
            CopyDoneInfo info = done_queue_.front();
            done_queue_.pop();
            free_buffer(info.dev_buffer_ptr);
        }
    }

    LOG_INFO("ProfMemoryManager stopped");
}

bool ProfMemoryManager::try_pop_ready(ReadyBufferInfo& info) {
    std::lock_guard<std::mutex> lock(ready_mutex_);
    if (ready_queue_.empty()) {
        return false;
    }
    info = ready_queue_.front();
    ready_queue_.pop();
    return true;
}

bool ProfMemoryManager::wait_pop_ready(ReadyBufferInfo& info, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(ready_mutex_);
    if (ready_cv_.wait_for(lock, timeout, [this]{ return !ready_queue_.empty(); })) {
        info = ready_queue_.front();
        ready_queue_.pop();
        return true;
    }
    return false;
}

void ProfMemoryManager::notify_copy_done(const CopyDoneInfo& info) {
    std::lock_guard<std::mutex> lock(done_mutex_);
    done_queue_.push(info);
}

void* ProfMemoryManager::alloc_and_register(size_t size, void** host_ptr_out) {
    void* dev_ptr = alloc_cb_(size, user_data_);
    if (dev_ptr == nullptr) {
        LOG_ERROR("ProfMemoryManager: alloc failed for %zu bytes", size);
        *host_ptr_out = nullptr;
        return nullptr;
    }

    if (register_cb_ != nullptr) {
        void* host_ptr = nullptr;
        int rc = register_cb_(dev_ptr, size, device_id_, user_data_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("ProfMemoryManager: register failed: %d", rc);
            free_buffer(dev_ptr);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
    } else {
        // Simulation mode: dev_ptr == host_ptr
        *host_ptr_out = dev_ptr;
    }

    dev_to_host_[dev_ptr] = *host_ptr_out;
    return dev_ptr;
}

void ProfMemoryManager::free_buffer(void* dev_ptr) {
    if (dev_ptr != nullptr && free_cb_ != nullptr) {
        dev_to_host_.erase(dev_ptr);
        free_cb_(dev_ptr, user_data_);
    }
}

void* ProfMemoryManager::resolve_host_ptr(void* dev_ptr) {
    if (register_cb_ == nullptr) {
        return dev_ptr;  // Simulation mode: dev_ptr == host_ptr
    }
    auto it = dev_to_host_.find(dev_ptr);
    if (it != dev_to_host_.end()) {
        return it->second;
    }
    LOG_ERROR("ProfMemoryManager: no host mapping for dev_ptr=%p", dev_ptr);
    return nullptr;
}

void ProfMemoryManager::register_mapping(void* dev_ptr, void* host_ptr) {
    dev_to_host_[dev_ptr] = host_ptr;
}

void ProfMemoryManager::process_ready_entry(PerfDataHeader* /*header*/, int /*thread_idx*/,
                                              const ReadyQueueEntry& entry) {
    bool is_phase = (entry.is_phase != 0);
    uint64_t old_dev_ptr = entry.buffer_ptr;
    uint32_t seq = entry.buffer_seq;

    if (is_phase) {
        uint32_t tidx = entry.core_index;
        if (tidx >= static_cast<uint32_t>(PLATFORM_MAX_AICPU_THREADS)) {
            LOG_ERROR("ProfMemoryManager: invalid phase entry: thread=%u", tidx);
            return;
        }

        PhaseBufferState* state = get_phase_buffer_state(shared_mem_host_, num_cores_, tidx);

        // Allocate new PhaseBuffer
        void* host_ptr = nullptr;
        void* new_dev_ptr = alloc_and_register(sizeof(PhaseBuffer), &host_ptr);
        if (new_dev_ptr != nullptr) {
            // Initialize new buffer
            PhaseBuffer* new_buf = (PhaseBuffer*)host_ptr;
            new_buf->count = 0;

            // Push to free_queue (with overflow guard)
            rmb();
            uint32_t head_val = state->free_queue.head;
            uint32_t tail = state->free_queue.tail;
            if ((tail - head_val) >= PLATFORM_PROF_SLOT_COUNT) {
                LOG_ERROR("ProfMemoryManager: phase free_queue overflow for thread %u", tidx);
                free_buffer(new_dev_ptr);
            } else {
                state->free_queue.buffer_ptrs[tail % PLATFORM_PROF_SLOT_COUNT] = (uint64_t)new_dev_ptr;
                wmb();
                state->free_queue.tail = tail + 1;
                wmb();
            }
        } else {
            LOG_ERROR("ProfMemoryManager: phase buffer alloc failed, device may lose data");
        }

        // Resolve host pointer of old buffer
        void* old_host_ptr = resolve_host_ptr((void*)old_dev_ptr);
        if (old_host_ptr == nullptr) {
            LOG_ERROR("ProfMemoryManager: cannot resolve host ptr for phase buffer dev=%p", (void*)old_dev_ptr);
            return;
        }

        // Push old buffer to ready queue for main thread to copy
        ReadyBufferInfo info;
        info.type = ProfBufferType::PHASE;
        info.index = tidx;
        info.slot_idx = 0;  // Not used in free queue design
        info.dev_buffer_ptr = (void*)old_dev_ptr;
        info.host_buffer_ptr = old_host_ptr;
        info.buffer_seq = seq;

        {
            std::lock_guard<std::mutex> lock(ready_mutex_);
            ready_queue_.push(info);
        }
        ready_cv_.notify_one();

    } else {
        uint32_t core_index = entry.core_index;
        if (core_index >= static_cast<uint32_t>(num_cores_)) {
            LOG_ERROR("ProfMemoryManager: invalid perf entry: core=%u", core_index);
            return;
        }

        PerfBufferState* state = get_perf_buffer_state(shared_mem_host_, core_index);

        // Allocate new PerfBuffer
        void* host_ptr = nullptr;
        void* new_dev_ptr = alloc_and_register(sizeof(PerfBuffer), &host_ptr);
        if (new_dev_ptr != nullptr) {
            PerfBuffer* new_buf = (PerfBuffer*)host_ptr;
            new_buf->count = 0;

            // Push to free_queue (with overflow guard)
            rmb();
            uint32_t head_val = state->free_queue.head;
            uint32_t tail = state->free_queue.tail;
            if ((tail - head_val) >= PLATFORM_PROF_SLOT_COUNT) {
                LOG_ERROR("ProfMemoryManager: perf free_queue overflow for core %u", core_index);
                free_buffer(new_dev_ptr);
            } else {
                state->free_queue.buffer_ptrs[tail % PLATFORM_PROF_SLOT_COUNT] = (uint64_t)new_dev_ptr;
                wmb();
                state->free_queue.tail = tail + 1;
                wmb();
            }
        } else {
            LOG_ERROR("ProfMemoryManager: perf buffer alloc failed, device may lose data");
        }

        void* old_host_ptr = resolve_host_ptr((void*)old_dev_ptr);
        if (old_host_ptr == nullptr) {
            LOG_ERROR("ProfMemoryManager: cannot resolve host ptr for perf buffer dev=%p", (void*)old_dev_ptr);
            return;
        }

        ReadyBufferInfo info;
        info.type = ProfBufferType::PERF_RECORD;
        info.index = core_index;
        info.slot_idx = 0;  // Not used in free queue design
        info.dev_buffer_ptr = (void*)old_dev_ptr;
        info.host_buffer_ptr = old_host_ptr;
        info.buffer_seq = seq;

        {
            std::lock_guard<std::mutex> lock(ready_mutex_);
            ready_queue_.push(info);
        }
        ready_cv_.notify_one();
    }
}

void ProfMemoryManager::mgmt_loop() {
    PerfDataHeader* header = get_perf_header(shared_mem_host_);

    while (running_.load()) {
        // 1. Process done queue: free buffers that main thread has finished copying
        {
            std::lock_guard<std::mutex> lock(done_mutex_);
            while (!done_queue_.empty()) {
                CopyDoneInfo info = done_queue_.front();
                done_queue_.pop();
                free_buffer(info.dev_buffer_ptr);
            }
        }

        // 2. Poll ReadyQueues from all AICPU threads
        bool found_any = false;
        for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
            rmb();
            uint32_t head = header->queue_heads[t];
            uint32_t tail = header->queue_tails[t];

            // Validate indices to prevent OOB access from corrupted shared memory
            if (head >= PLATFORM_PROF_READYQUEUE_SIZE || tail >= PLATFORM_PROF_READYQUEUE_SIZE) {
                LOG_ERROR("mgmt_loop: invalid queue indices for thread %d: head=%u tail=%u (max=%d)",
                          t, head, tail, PLATFORM_PROF_READYQUEUE_SIZE);
                continue;
            }

            while (head != tail) {
                ReadyQueueEntry entry = header->queues[t][head];

                process_ready_entry(header, t, entry);

                head = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
                header->queue_heads[t] = head;
                wmb();

                found_any = true;

                // Re-read tail in case more entries arrived
                rmb();
                tail = header->queue_tails[t];
                if (tail >= PLATFORM_PROF_READYQUEUE_SIZE) {
                    LOG_ERROR("mgmt_loop: invalid tail for thread %d: %u", t, tail);
                    break;
                }
            }
        }

        // 3. If nothing found, yield briefly to avoid busy-spinning
        if (!found_any) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // Final drain: process any remaining entries
    PerfDataHeader* hdr = get_perf_header(shared_mem_host_);
    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        rmb();
        uint32_t head = hdr->queue_heads[t];
        uint32_t tail = hdr->queue_tails[t];
        if (head >= PLATFORM_PROF_READYQUEUE_SIZE || tail >= PLATFORM_PROF_READYQUEUE_SIZE) {
            LOG_ERROR("mgmt_loop drain: invalid queue indices for thread %d: head=%u tail=%u",
                      t, head, tail);
            continue;
        }
        while (head != tail) {
            ReadyQueueEntry entry = hdr->queues[t][head];
            process_ready_entry(hdr, t, entry);
            head = (head + 1) % PLATFORM_PROF_READYQUEUE_SIZE;
            hdr->queue_heads[t] = head;
            wmb();
            rmb();
            tail = hdr->queue_tails[t];
            if (tail >= PLATFORM_PROF_READYQUEUE_SIZE) {
                LOG_ERROR("mgmt_loop drain: invalid tail for thread %d: %u", t, tail);
                break;
            }
        }
    }
}

// =============================================================================
// PerformanceCollector Implementation
// =============================================================================

/**
 * Check if a phase ID belongs to a scheduler phase (vs orchestrator phase).
 * Scheduler phases: SCHED_COMPLETE(0), SCHED_DISPATCH(1), SCHED_SCAN(2), SCHED_IDLE_WAIT(3)
 * Orchestrator phases: ORCH_SYNC(16) through ORCH_SCOPE_END(24)
 */
static bool is_scheduler_phase(AicpuPhaseId id) {
    return static_cast<uint32_t>(id) < static_cast<uint32_t>(AicpuPhaseId::SCHED_PHASE_COUNT);
}

PerformanceCollector::~PerformanceCollector() {
    if (perf_shared_mem_host_ != nullptr) {
        LOG_WARN("PerformanceCollector destroyed without finalize()");
    }
}

void* PerformanceCollector::alloc_single_buffer(size_t size, void** host_ptr_out) {
    void* dev_ptr = alloc_cb_(size, user_data_);
    if (dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate buffer (%zu bytes)", size);
        *host_ptr_out = nullptr;
        return nullptr;
    }

    if (register_cb_ != nullptr) {
        void* host_ptr = nullptr;
        int rc = register_cb_(dev_ptr, size, device_id_, user_data_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("Buffer registration failed: %d", rc);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
    } else {
        *host_ptr_out = dev_ptr;
    }

    // Register mapping so ProfMemoryManager can resolve dev→host
    memory_manager_.register_mapping(dev_ptr, *host_ptr_out);
    return dev_ptr;
}

int PerformanceCollector::initialize(Runtime& runtime,
                                      int num_aicore,
                                      int device_id,
                                      PerfAllocCallback alloc_cb,
                                      PerfRegisterCallback register_cb,
                                      PerfFreeCallback free_cb,
                                      void* user_data) {
    if (perf_shared_mem_host_ != nullptr) {
        LOG_ERROR("PerformanceCollector already initialized");
        return -1;
    }

    LOG_INFO("Initializing performance profiling");

    if (num_aicore <= 0 || num_aicore > PLATFORM_MAX_CORES) {
        LOG_ERROR("Invalid number of AICores: %d (max=%d)", num_aicore, PLATFORM_MAX_CORES);
        return -1;
    }

    device_id_ = device_id;
    num_aicore_ = num_aicore;
    alloc_cb_ = alloc_cb;
    register_cb_ = register_cb;
    free_cb_ = free_cb;
    user_data_ = user_data;

    // Step 1: Calculate shared memory size (slot arrays only, no actual buffers)
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    size_t total_size = calc_perf_data_size_with_phases(num_aicore, num_phase_threads);

    LOG_DEBUG("Shared memory allocation plan:");
    LOG_DEBUG("  Number of cores:      %d", num_aicore);
    LOG_DEBUG("  Header size:          %zu bytes", sizeof(PerfDataHeader));
    LOG_DEBUG("  PerfBufferState size: %zu bytes each", sizeof(PerfBufferState));
    LOG_DEBUG("  PhaseBufferState size:%zu bytes each", sizeof(PhaseBufferState));
    LOG_DEBUG("  Total shared memory:  %zu bytes (%zu KB)",
              total_size, total_size / 1024);

    // Step 2: Allocate shared memory for slot arrays
    void* perf_dev_ptr = alloc_cb(total_size, user_data);
    if (perf_dev_ptr == nullptr) {
        LOG_ERROR("Failed to allocate shared memory (%zu bytes)", total_size);
        return -1;
    }
    LOG_DEBUG("Allocated shared memory: %p", perf_dev_ptr);

    // Step 3: Register to host mapping (optional)
    void* perf_host_ptr = nullptr;
    if (register_cb != nullptr) {
        int rc = register_cb(perf_dev_ptr, total_size, device_id, user_data, &perf_host_ptr);
        if (rc != 0) {
            LOG_ERROR("Memory registration failed: %d", rc);
            return rc;
        }
        was_registered_ = true;
        if (perf_host_ptr == nullptr) {
            LOG_ERROR("register_cb succeeded but returned null host_ptr");
            return -1;
        }
        LOG_DEBUG("Mapped to host memory: %p", perf_host_ptr);
    } else {
        perf_host_ptr = perf_dev_ptr;
        LOG_DEBUG("Simulation mode: host_ptr = dev_ptr = %p", perf_host_ptr);
    }

    // Step 4: Initialize header
    PerfDataHeader* header = get_perf_header(perf_host_ptr);

    for (int t = 0; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        memset(header->queues[t], 0, sizeof(header->queues[t]));
        header->queue_heads[t] = 0;
        header->queue_tails[t] = 0;
    }

    header->num_cores = num_aicore;
    header->total_tasks = 0;

    LOG_DEBUG("Initialized PerfDataHeader:");
    LOG_DEBUG("  num_cores:        %d", header->num_cores);
    LOG_DEBUG("  buffer_capacity:  %d", PLATFORM_PROF_BUFFER_SIZE);
    LOG_DEBUG("  queue capacity:   %d", PLATFORM_PROF_READYQUEUE_SIZE);

    // Step 5: Initialize PerfBufferStates and pre-fill free_queues
    for (int i = 0; i < num_aicore; i++) {
        PerfBufferState* state = get_perf_buffer_state(perf_host_ptr, i);
        memset(state, 0, sizeof(PerfBufferState));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->current_buf_ptr = 0;
        state->current_buf_seq = 0;

        // Pre-fill free_queue with PLATFORM_PROF_SLOT_COUNT buffers
        for (int s = 0; s < PLATFORM_PROF_SLOT_COUNT; s++) {
            void* host_buf_ptr = nullptr;
            void* dev_buf_ptr = alloc_single_buffer(sizeof(PerfBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate PerfBuffer for core %d, buffer %d", i, s);
                return -1;
            }
            // Initialize buffer
            PerfBuffer* buf = (PerfBuffer*)host_buf_ptr;
            memset(buf, 0, sizeof(PerfBuffer));
            buf->count = 0;

            // Push to free_queue
            state->free_queue.buffer_ptrs[s] = (uint64_t)dev_buf_ptr;
        }
        wmb();
        state->free_queue.tail = PLATFORM_PROF_SLOT_COUNT;
        wmb();
    }
    LOG_DEBUG("Initialized %d PerfBufferStates with %d buffers each",
              num_aicore, PLATFORM_PROF_SLOT_COUNT);

    // Step 6: Initialize PhaseBufferStates and pre-fill free_queues
    for (int t = 0; t < num_phase_threads; t++) {
        PhaseBufferState* state = get_phase_buffer_state(perf_host_ptr, num_aicore, t);
        memset(state, 0, sizeof(PhaseBufferState));

        state->free_queue.head = 0;
        state->free_queue.tail = 0;
        state->current_buf_ptr = 0;
        state->current_buf_seq = 0;

        // Pre-fill free_queue with PLATFORM_PROF_SLOT_COUNT buffers
        for (int s = 0; s < PLATFORM_PROF_SLOT_COUNT; s++) {
            void* host_buf_ptr = nullptr;
            void* dev_buf_ptr = alloc_single_buffer(sizeof(PhaseBuffer), &host_buf_ptr);
            if (dev_buf_ptr == nullptr) {
                LOG_ERROR("Failed to allocate PhaseBuffer for thread %d, buffer %d", t, s);
                return -1;
            }
            PhaseBuffer* buf = (PhaseBuffer*)host_buf_ptr;
            memset(buf, 0, sizeof(PhaseBuffer));
            buf->count = 0;

            // Push to free_queue
            state->free_queue.buffer_ptrs[s] = (uint64_t)dev_buf_ptr;
        }
        wmb();
        state->free_queue.tail = PLATFORM_PROF_SLOT_COUNT;
        wmb();
    }
    LOG_DEBUG("Initialized %d PhaseBufferStates with %d buffers each",
              num_phase_threads, PLATFORM_PROF_SLOT_COUNT);

    wmb();

    // Step 7: Pass base address to Runtime
    runtime.perf_data_base = (uint64_t)perf_dev_ptr;
    LOG_DEBUG("Set runtime.perf_data_base = 0x%lx", runtime.perf_data_base);

    perf_shared_mem_dev_ = perf_dev_ptr;
    perf_shared_mem_host_ = perf_host_ptr;

    LOG_INFO("Performance profiling initialized (dynamic buffer mode)");
    return 0;
}

void PerformanceCollector::start_memory_manager() {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    memory_manager_.start(perf_shared_mem_host_, num_aicore_,
                           PLATFORM_MAX_AICPU_THREADS,
                           alloc_cb_, register_cb_, free_cb_,
                           user_data_, device_id_);
}

void PerformanceCollector::stop_memory_manager() {
    if (memory_manager_.is_running()) {
        memory_manager_.stop();
    }
}

void PerformanceCollector::poll_and_collect(int expected_tasks) {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    LOG_INFO("Collecting performance data");

    PerfDataHeader* header = get_perf_header(perf_shared_mem_host_);

    const auto timeout_duration = std::chrono::seconds(PLATFORM_PROF_TIMEOUT_SECONDS);
    std::optional<std::chrono::steady_clock::time_point> idle_start;

    if (expected_tasks <= 0) {
        LOG_INFO("Waiting for AICPU to write total_tasks in PerfDataHeader...");
        idle_start = std::chrono::steady_clock::now();

        while (true) {
            rmb();
            uint32_t raw_total_tasks = header->total_tasks;

            if (raw_total_tasks > 0) {
                expected_tasks = static_cast<int>(raw_total_tasks);
                LOG_INFO("AICPU reported task count: %d", expected_tasks);
                break;
            }

            auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
            if (elapsed >= timeout_duration) {
                LOG_ERROR("Timeout waiting for AICPU task count after %ld seconds",
                         std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                return;
            }

            // Check for ready buffers while waiting
            ReadyBufferInfo info;
            if (memory_manager_.try_pop_ready(info)) {
                // Process it (even before we know expected_tasks)
                // Will be counted below
            }
        }
    }

    LOG_DEBUG("Initial expected tasks: %d", expected_tasks);

    int total_records_collected = 0;
    int buffers_processed = 0;

    collected_perf_records_.clear();
    collected_perf_records_.resize(num_aicore_);

    // Pre-allocate phase record storage
    AicpuPhaseHeader* phase_header = get_phase_header(perf_shared_mem_host_, num_aicore_);
    int num_sched_for_poll = 0;
    if (phase_header->magic == AICPU_PHASE_MAGIC) {
        num_sched_for_poll = phase_header->num_sched_threads;
        if (num_sched_for_poll > PLATFORM_MAX_AICPU_THREADS) {
            num_sched_for_poll = PLATFORM_MAX_AICPU_THREADS;
        }
        collected_phase_records_.clear();
        collected_phase_records_.resize(PLATFORM_MAX_AICPU_THREADS);
    }

    idle_start.reset();
    int last_logged_expected = -1;

    while (total_records_collected < expected_tasks) {
        // Check for updated expected_tasks
        rmb();
        int current_expected = static_cast<int>(header->total_tasks);
        if (current_expected > expected_tasks) {
            expected_tasks = current_expected;
            if (last_logged_expected < 0) {
                LOG_INFO("Updated expected_tasks to %d (orchestrator progress)", expected_tasks);
                last_logged_expected = expected_tasks;
            }
        }

        ReadyBufferInfo info;
        if (memory_manager_.wait_pop_ready(info, std::chrono::milliseconds(100))) {
            idle_start.reset();

            if (info.type == ProfBufferType::PERF_RECORD) {
                PerfBuffer* buf = (PerfBuffer*)info.host_buffer_ptr;
                rmb();
                uint32_t count = buf->count;
                if (count > PLATFORM_PROF_BUFFER_SIZE) {
                    count = PLATFORM_PROF_BUFFER_SIZE;
                }

                uint32_t core_index = info.index;
                if (core_index < static_cast<uint32_t>(num_aicore_)) {
                    for (uint32_t i = 0; i < count; i++) {
                        collected_perf_records_[core_index].push_back(buf->records[i]);
                    }
                    total_records_collected += count;
                }

                LOG_DEBUG("Collected %u perf records from core %u (total: %d/%d)",
                         count, core_index, total_records_collected, expected_tasks);

            } else {
                PhaseBuffer* buf = (PhaseBuffer*)info.host_buffer_ptr;
                rmb();
                uint32_t count = buf->count;
                if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
                    count = PLATFORM_PHASE_RECORDS_PER_THREAD;
                }

                uint32_t tidx = info.index;
                for (uint32_t i = 0; i < count; i++) {
                    collected_phase_records_[tidx].push_back(buf->records[i]);
                }

                LOG_DEBUG("Collected %u phase records from thread %u", count, tidx);
            }

            // Notify memory manager to free old buffer
            memory_manager_.notify_copy_done({info.dev_buffer_ptr});
            buffers_processed++;

        } else {
            // Timeout on wait — check for overall timeout
            if (!idle_start.has_value()) {
                idle_start = std::chrono::steady_clock::now();
            }
            auto elapsed = std::chrono::steady_clock::now() - idle_start.value();
            if (elapsed >= timeout_duration) {
                LOG_ERROR("Performance data collection idle timeout after %ld seconds",
                         std::chrono::duration_cast<std::chrono::seconds>(elapsed).count());
                LOG_ERROR("Collected %d / %d records before timeout",
                         total_records_collected, expected_tasks);
                break;
            }
        }
    }

    LOG_INFO("Total buffers processed: %d", buffers_processed);
    LOG_INFO("Total records collected: %d", total_records_collected);

    if (total_records_collected < expected_tasks) {
        LOG_WARN("Incomplete collection (%d / %d records)",
                 total_records_collected, expected_tasks);
    }

    LOG_INFO("Performance data collection complete");
}

void PerformanceCollector::drain_remaining_buffers() {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    // Ensure phase record storage is initialized
    AicpuPhaseHeader* phase_header = get_phase_header(perf_shared_mem_host_, num_aicore_);
    rmb();
    int num_sched = 0;
    if (phase_header->magic == AICPU_PHASE_MAGIC) {
        num_sched = phase_header->num_sched_threads;
        if (num_sched > PLATFORM_MAX_AICPU_THREADS) {
            num_sched = PLATFORM_MAX_AICPU_THREADS;
        }
    }

    int drained_perf = 0;
    int drained_phase = 0;

    ReadyBufferInfo info;
    while (memory_manager_.try_pop_ready(info)) {
        if (info.type == ProfBufferType::PERF_RECORD) {
            PerfBuffer* buf = (PerfBuffer*)info.host_buffer_ptr;
            rmb();
            uint32_t count = buf->count;
            if (count > PLATFORM_PROF_BUFFER_SIZE) {
                count = PLATFORM_PROF_BUFFER_SIZE;
            }
            uint32_t core_index = info.index;
            if (core_index < static_cast<uint32_t>(num_aicore_)) {
                for (uint32_t i = 0; i < count; i++) {
                    collected_perf_records_[core_index].push_back(buf->records[i]);
                }
                drained_perf += count;
            }
        } else {
            PhaseBuffer* buf = (PhaseBuffer*)info.host_buffer_ptr;
            rmb();
            uint32_t count = buf->count;
            if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
                count = PLATFORM_PHASE_RECORDS_PER_THREAD;
            }
            uint32_t tidx = info.index;
            for (uint32_t i = 0; i < count; i++) {
                collected_phase_records_[tidx].push_back(buf->records[i]);
            }
            drained_phase += count;
        }

        memory_manager_.notify_copy_done({info.dev_buffer_ptr});
    }

    if (drained_perf > 0 || drained_phase > 0) {
        LOG_INFO("Drained remaining buffers: %d perf records, %d phase records",
                 drained_perf, drained_phase);
    }

    if (drained_phase > 0) {
        has_phase_data_ = true;
    }
}

void PerformanceCollector::collect_phase_data() {
    if (perf_shared_mem_host_ == nullptr) {
        return;
    }

    rmb();

    AicpuPhaseHeader* phase_header = get_phase_header(perf_shared_mem_host_, num_aicore_);

    // Validate magic
    if (phase_header->magic != AICPU_PHASE_MAGIC) {
        LOG_INFO("No phase profiling data found (magic mismatch: 0x%x vs 0x%x)",
                 phase_header->magic, AICPU_PHASE_MAGIC);
        return;
    }

    int num_sched_threads = phase_header->num_sched_threads;
    if (num_sched_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid num_sched_threads %d from shared memory (max=%d)",
                  num_sched_threads, PLATFORM_MAX_AICPU_THREADS);
        return;
    }
    LOG_INFO("Collecting remaining phase data: %d scheduler threads", num_sched_threads);

    int total_slots = PLATFORM_MAX_AICPU_THREADS;

    // Scan remaining PhaseBufferStates for active buffers with partial data.
    // READY buffers were already enqueued to the ReadyQueue and collected via
    // poll_and_collect() or drain_remaining_buffers(). Only current_buf_ptr
    // contains partial data that was never enqueued (the active buffer when execution ended).
    int total_phase_records = 0;
    for (int t = 0; t < total_slots; t++) {
        PhaseBufferState* state = get_phase_buffer_state(perf_shared_mem_host_, num_aicore_, t);

        rmb();
        uint64_t buf_ptr = state->current_buf_ptr;
        if (buf_ptr != 0) {
            void* host_ptr = memory_manager_.resolve_host_ptr((void*)buf_ptr);
            if (host_ptr == nullptr) {
                LOG_ERROR("collect_phase_data: no host mapping for dev_ptr=%p (thread %d)",
                          (void*)buf_ptr, t);
                continue;
            }
            PhaseBuffer* pbuf = (PhaseBuffer*)host_ptr;
            if (pbuf->count > 0) {
                uint32_t count = pbuf->count;
                if (count > static_cast<uint32_t>(PLATFORM_PHASE_RECORDS_PER_THREAD)) {
                    count = PLATFORM_PHASE_RECORDS_PER_THREAD;
                }

                for (uint32_t i = 0; i < count; i++) {
                    collected_phase_records_[t].push_back(pbuf->records[i]);
                }
                total_phase_records += count;
            }
        }
    }

    // Log per-thread totals
    for (size_t t = 0; t < collected_phase_records_.size(); t++) {
        if (!collected_phase_records_[t].empty()) {
            size_t sched_count = 0, orch_count = 0;
            for (const auto& r : collected_phase_records_[t]) {
                if (is_scheduler_phase(r.phase_id)) sched_count++;
                else orch_count++;
            }
            LOG_INFO("  Thread %zu: %zu records (sched=%zu, orch=%zu)",
                     t, collected_phase_records_[t].size(), sched_count, orch_count);
        }
    }

    // Read orchestrator summary
    collected_orch_summary_ = phase_header->orch_summary;
    bool orch_valid = (collected_orch_summary_.magic == AICPU_PHASE_MAGIC);

    if (orch_valid) {
        LOG_INFO("  Orchestrator: %lld tasks, %.3fus",
                 (long long)collected_orch_summary_.submit_count,
                 cycles_to_us(collected_orch_summary_.end_time - collected_orch_summary_.start_time));
    } else {
        LOG_INFO("  Orchestrator: no summary data");
    }

    // Check if drain_remaining_buffers() already accumulated some Phase records
    bool has_accumulated = has_phase_data_;
    if (!has_accumulated) {
        for (const auto& v : collected_phase_records_) {
            if (!v.empty()) { has_accumulated = true; break; }
        }
    }
    has_phase_data_ = (total_phase_records > 0 || orch_valid || has_accumulated);

    // Read core-to-thread mapping
    int num_cores = static_cast<int>(phase_header->num_cores);
    if (num_cores > 0 && num_cores <= PLATFORM_MAX_CORES) {
        core_to_thread_.assign(phase_header->core_to_thread,
                                phase_header->core_to_thread + num_cores);
        LOG_INFO("  Core-to-thread mapping: %d cores", num_cores);
    }

    LOG_INFO("Phase data collection complete: %d remaining records, orch_summary=%s",
             total_phase_records, orch_valid ? "yes" : "no");
}

int PerformanceCollector::export_swimlane_json(const std::string& output_path) {
    // Step 1: Validate collected data
    bool has_any_records = false;
    for (const auto& core_records : collected_perf_records_) {
        if (!core_records.empty()) {
            has_any_records = true;
            break;
        }
    }
    if (!has_any_records) {
        LOG_WARN("Warning: No performance data to export.");
        return -1;
    }

    // Step 2: Create output directory if it doesn't exist
    struct stat st;
    if (stat(output_path.c_str(), &st) == -1) {
        if (mkdir(output_path.c_str(), 0755) != 0) {
            LOG_ERROR("Error: Failed to create output directory.");
            return -1;
        }
    }

    // Step 3: Flatten per-core vectors into tagged records with core_id derived from index
    struct TaggedRecord {
        const PerfRecord* record;
        uint32_t core_id;
    };
    std::vector<TaggedRecord> tagged_records;
    size_t total_records = 0;
    for (const auto& core_records : collected_perf_records_) {
        total_records += core_records.size();
    }
    tagged_records.reserve(total_records);
    for (size_t core_idx = 0; core_idx < collected_perf_records_.size(); core_idx++) {
        for (const auto& record : collected_perf_records_[core_idx]) {
            tagged_records.push_back({&record, static_cast<uint32_t>(core_idx)});
        }
    }

    // Sort by task_id
    std::sort(tagged_records.begin(), tagged_records.end(),
              [](const TaggedRecord& a, const TaggedRecord& b) {
                  return a.record->task_id < b.record->task_id;
              });

    // Step 4: Calculate base time (minimum kernel_ready_time, including phase timestamps)
    uint64_t base_time_cycles = UINT64_MAX;
    for (const auto& tagged : tagged_records) {
        if (tagged.record->kernel_ready_time < base_time_cycles) {
            base_time_cycles = tagged.record->kernel_ready_time;
        }
        if (tagged.record->dispatch_time < base_time_cycles && tagged.record->dispatch_time > 0) {
            base_time_cycles = tagged.record->dispatch_time;
            LOG_WARN("Timestamp violation: dispatch_time (%lu) < base_time (%lu) for task %u, using dispatch_time as new base_time",
                        tagged.record->dispatch_time, base_time_cycles, tagged.record->task_id);
        }
    }

    // Include phase record timestamps in base_time calculation
    if (has_phase_data_) {
        for (const auto& thread_records : collected_phase_records_) {
            for (const auto& pr : thread_records) {
                if (pr.start_time > 0 && pr.start_time < base_time_cycles) {
                    base_time_cycles = pr.start_time;
                }
            }
        }
        if (collected_orch_summary_.magic == AICPU_PHASE_MAGIC &&
            collected_orch_summary_.start_time > 0 &&
            collected_orch_summary_.start_time < base_time_cycles) {
            base_time_cycles = collected_orch_summary_.start_time;
        }
    }

    // Step 5: Generate filename with timestamp (YYYYMMDD_HHMMSS)
    std::time_t now = time(nullptr);
    std::tm* timeinfo = std::localtime(&now);
    char time_buffer[32];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y%m%d_%H%M%S", timeinfo);
    std::string filepath = output_path + "/perf_swimlane_"
                          + std::string(time_buffer) + ".json";

    // Step 6: Open JSON file for writing
    std::ofstream outfile(filepath);
    if (!outfile.is_open()) {
        LOG_ERROR("Error: Failed to open file: %s", filepath.c_str());
        return -1;
    }

    // Step 7: Write JSON data
    int version = has_phase_data_ ? 2 : 1;
    outfile << "{\n";
    outfile << "  \"version\": " << version << ",\n";
    outfile << "  \"tasks\": [\n";

    for (size_t i = 0; i < tagged_records.size(); ++i) {
        const auto& tagged = tagged_records[i];
        const auto& record = *tagged.record;

        // Convert times to microseconds
        double start_us = cycles_to_us(record.start_time - base_time_cycles);
        double end_us = cycles_to_us(record.end_time - base_time_cycles);
        double duration_us = end_us - start_us;
        double kernel_ready_us = cycles_to_us(record.kernel_ready_time - base_time_cycles);
        double dispatch_us = (record.dispatch_time > 0) ? cycles_to_us(record.dispatch_time - base_time_cycles) : 0.0;
        double finish_us = (record.finish_time > 0) ? cycles_to_us(record.finish_time - base_time_cycles) : 0.0;

        const char* core_type_str = (record.core_type == CoreType::AIC) ? "aic" : "aiv";

        outfile << "    {\n";
        outfile << "      \"task_id\": " << record.task_id << ",\n";
        outfile << "      \"func_id\": " << record.func_id << ",\n";
        outfile << "      \"core_id\": " << tagged.core_id << ",\n";
        outfile << "      \"core_type\": \"" << core_type_str << "\",\n";
        outfile << "      \"ring_id\": " << static_cast<int>(record.ring_id) << ",\n";
        outfile << "      \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us << ",\n";
        outfile << "      \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us << ",\n";
        outfile << "      \"duration_us\": " << std::fixed << std::setprecision(3) << duration_us << ",\n";
        outfile << "      \"kernel_ready_time_us\": " << std::fixed << std::setprecision(3) << kernel_ready_us << ",\n";
        outfile << "      \"dispatch_time_us\": " << std::fixed << std::setprecision(3) << dispatch_us << ",\n";
        outfile << "      \"finish_time_us\": " << std::fixed << std::setprecision(3) << finish_us << ",\n";
        outfile << "      \"fanout\": [";
        int safe_fanout_count = (record.fanout_count >= 0 && record.fanout_count <= RUNTIME_MAX_FANOUT)
                                ? record.fanout_count : 0;
        for (int j = 0; j < safe_fanout_count; ++j) {
            outfile << record.fanout[j];
            if (j < safe_fanout_count - 1) {
                outfile << ", ";
            }
        }
        outfile << "],\n";
        outfile << "      \"fanout_count\": " << record.fanout_count << "\n";
        outfile << "    }";
        if (i < tagged_records.size() - 1) {
            outfile << ",";
        }
        outfile << "\n";
    }
    outfile << "  ]";

    // Step 8: Write phase profiling data (version 2)
    if (has_phase_data_) {
        auto sched_phase_name = [](AicpuPhaseId id) -> const char* {
            switch (id) {
                case AicpuPhaseId::SCHED_COMPLETE:    return "complete";
                case AicpuPhaseId::SCHED_DISPATCH:    return "dispatch";
                case AicpuPhaseId::SCHED_SCAN:        return "scan";
                case AicpuPhaseId::SCHED_IDLE_WAIT:   return "idle";
                default: return "unknown";
            }
        };

        auto orch_phase_name = [](AicpuPhaseId id) -> const char* {
            switch (id) {
                case AicpuPhaseId::ORCH_SYNC:      return "orch_sync";
                case AicpuPhaseId::ORCH_ALLOC:     return "orch_alloc";
                case AicpuPhaseId::ORCH_PARAMS:    return "orch_params";
                case AicpuPhaseId::ORCH_LOOKUP:    return "orch_lookup";
                case AicpuPhaseId::ORCH_HEAP:      return "orch_heap";
                case AicpuPhaseId::ORCH_INSERT:    return "orch_insert";
                case AicpuPhaseId::ORCH_FANIN:     return "orch_fanin";
                case AicpuPhaseId::ORCH_FINALIZE:  return "orch_finalize";
                case AicpuPhaseId::ORCH_SCOPE_END: return "orch_scope_end";
                default: return "unknown";
            }
        };

        // AICPU scheduler phases (filtered from unified collected_phase_records_)
        outfile << ",\n  \"aicpu_scheduler_phases\": [\n";
        for (size_t t = 0; t < collected_phase_records_.size(); t++) {
            outfile << "    [\n";
            bool first = true;
            for (const auto& pr : collected_phase_records_[t]) {
                if (!is_scheduler_phase(pr.phase_id)) continue;
                double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                double end_us = cycles_to_us(pr.end_time - base_time_cycles);
                if (!first) outfile << ",\n";
                outfile << "      {\"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                        << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us
                        << ", \"phase\": \"" << sched_phase_name(pr.phase_id) << "\""
                        << ", \"loop_iter\": " << pr.loop_iter
                        << ", \"tasks_processed\": " << pr.tasks_processed
                        << "}";
                first = false;
            }
            if (!first) outfile << "\n";
            outfile << "    ]";
            if (t < collected_phase_records_.size() - 1) outfile << ",";
            outfile << "\n";
        }
        outfile << "  ]";

        // AICPU orchestrator summary
        if (collected_orch_summary_.magic == AICPU_PHASE_MAGIC) {
            double orch_start_us = cycles_to_us(collected_orch_summary_.start_time - base_time_cycles);
            double orch_end_us = cycles_to_us(collected_orch_summary_.end_time - base_time_cycles);

            outfile << ",\n  \"aicpu_orchestrator\": {\n";
            outfile << "    \"start_time_us\": " << std::fixed << std::setprecision(3) << orch_start_us << ",\n";
            outfile << "    \"end_time_us\": " << std::fixed << std::setprecision(3) << orch_end_us << ",\n";
            outfile << "    \"submit_count\": " << collected_orch_summary_.submit_count << ",\n";
            outfile << "    \"phase_us\": {\n";
            outfile << "      \"sync\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.sync_cycle) << ",\n";
            outfile << "      \"alloc\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.alloc_cycle) << ",\n";
            outfile << "      \"params\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.params_cycle) << ",\n";
            outfile << "      \"lookup\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.lookup_cycle) << ",\n";
            outfile << "      \"heap\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.heap_cycle) << ",\n";
            outfile << "      \"insert\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.insert_cycle) << ",\n";
            outfile << "      \"fanin\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.fanin_cycle) << ",\n";
            outfile << "      \"scope_end\": " << std::fixed << std::setprecision(3) << cycles_to_us(collected_orch_summary_.scope_end_cycle) << "\n";
            outfile << "    }\n";
            outfile << "  }";
        }

        // Per-task orchestrator phase records (filtered from unified collected_phase_records_)
        bool has_orch_phases = false;
        for (const auto& v : collected_phase_records_) {
            for (const auto& r : v) {
                if (!is_scheduler_phase(r.phase_id)) { has_orch_phases = true; break; }
            }
            if (has_orch_phases) break;
        }
        if (has_orch_phases) {
            outfile << ",\n  \"aicpu_orchestrator_phases\": [\n";
            for (size_t t = 0; t < collected_phase_records_.size(); t++) {
                outfile << "    [\n";
                bool first = true;
                for (const auto& pr : collected_phase_records_[t]) {
                    if (is_scheduler_phase(pr.phase_id)) continue;
                    double start_us = cycles_to_us(pr.start_time - base_time_cycles);
                    double end_us = cycles_to_us(pr.end_time - base_time_cycles);
                    if (!first) outfile << ",\n";
                    outfile << "      {\"phase\": \"" << orch_phase_name(pr.phase_id) << "\""
                            << ", \"start_time_us\": " << std::fixed << std::setprecision(3) << start_us
                            << ", \"end_time_us\": " << std::fixed << std::setprecision(3) << end_us
                            << ", \"submit_idx\": " << pr.loop_iter
                            << ", \"task_id\": " << static_cast<int32_t>(pr.tasks_processed)
                            << "}";
                    first = false;
                }
                if (!first) outfile << "\n";
                outfile << "    ]";
                if (t < collected_phase_records_.size() - 1) outfile << ",";
                outfile << "\n";
            }
            outfile << "  ]";
        }
    }

    // Core-to-thread mapping
    if (!core_to_thread_.empty()) {
        outfile << ",\n  \"core_to_thread\": [";
        for (size_t i = 0; i < core_to_thread_.size(); i++) {
            outfile << static_cast<int>(core_to_thread_[i]);
            if (i < core_to_thread_.size() - 1) outfile << ", ";
        }
        outfile << "]";
    }

    outfile << "\n}\n";

    // Step 9: Close file
    outfile.close();

    uint32_t record_count = static_cast<uint32_t>(tagged_records.size());
    LOG_INFO("=== JSON Export Complete ===");
    LOG_INFO("File: %s", filepath.c_str());
    LOG_INFO("Records: %u", record_count);

    return 0;
}

int PerformanceCollector::finalize(PerfUnregisterCallback unregister_cb,
                                    PerfFreeCallback free_cb,
                                    void* user_data) {
    if (perf_shared_mem_host_ == nullptr) {
        return 0;
    }

    // Stop memory manager if still running
    stop_memory_manager();

    LOG_DEBUG("Cleaning up performance profiling resources");

    // Free initial buffers that are still in the slot arrays
    // (These were not freed by the memory manager because they were never replaced)
    // The memory manager frees old buffers after copy; initial buffers in free_queues remain.
    // Free all buffers in the free_queues and current_buf_ptr.
    for (int i = 0; i < num_aicore_; i++) {
        PerfBufferState* state = get_perf_buffer_state(perf_shared_mem_host_, i);

        // Free current buffer if any
        if (state->current_buf_ptr != 0 && free_cb != nullptr) {
            free_cb((void*)state->current_buf_ptr, user_data);
        }

        // Free all buffers in free_queue (limit iterations to max capacity)
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t max_iters = PLATFORM_PROF_SLOT_COUNT;
        while (head != tail && max_iters-- > 0) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            if (buf_ptr != 0 && free_cb != nullptr) {
                free_cb((void*)buf_ptr, user_data);
            }
            head++;
        }
        if (head != tail) {
            LOG_WARN("finalize: perf free_queue not fully drained for core %d (head=%u tail=%u)",
                     i, head, tail);
        }
    }

    AicpuPhaseHeader* phase_header = get_phase_header(perf_shared_mem_host_, num_aicore_);
    int num_phase_threads = PLATFORM_MAX_AICPU_THREADS;
    for (int t = 0; t < num_phase_threads; t++) {
        PhaseBufferState* state = get_phase_buffer_state(perf_shared_mem_host_, num_aicore_, t);

        // Free current buffer if any
        if (state->current_buf_ptr != 0 && free_cb != nullptr) {
            free_cb((void*)state->current_buf_ptr, user_data);
        }

        // Free all buffers in free_queue (limit iterations to max capacity)
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t max_iters = PLATFORM_PROF_SLOT_COUNT;
        while (head != tail && max_iters-- > 0) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            if (buf_ptr != 0 && free_cb != nullptr) {
                free_cb((void*)buf_ptr, user_data);
            }
            head++;
        }
        if (head != tail) {
            LOG_WARN("finalize: phase free_queue not fully drained for thread %d (head=%u tail=%u)",
                     t, head, tail);
        }
    }

    // Unregister host mapping (optional)
    if (unregister_cb != nullptr && was_registered_) {
        int rc = unregister_cb(perf_shared_mem_dev_, device_id_, user_data);
        if (rc != 0) {
            LOG_ERROR("halHostUnregister failed: %d", rc);
            return rc;
        }
        LOG_DEBUG("Host mapping unregistered");
    }

    // Free shared memory (slot arrays)
    if (free_cb != nullptr && perf_shared_mem_dev_ != nullptr) {
        free_cb(perf_shared_mem_dev_, user_data);
        LOG_DEBUG("Shared memory freed");
    }

    perf_shared_mem_dev_ = nullptr;
    perf_shared_mem_host_ = nullptr;
    was_registered_ = false;
    collected_perf_records_.clear();
    collected_phase_records_.clear();
    core_to_thread_.clear();
    has_phase_data_ = false;
    device_id_ = -1;
    alloc_cb_ = nullptr;
    register_cb_ = nullptr;
    free_cb_ = nullptr;
    user_data_ = nullptr;

    LOG_DEBUG("Performance profiling cleanup complete");
    return 0;
}
