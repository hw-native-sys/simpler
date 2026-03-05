/**
 * @file performance_collector_aicpu.cpp
 * @brief AICPU performance data collection implementation
 */

#include "aicpu/performance_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "common/platform_config.h"

#include <cstring>

// Cached phase profiling pointers (set during init, used on hot path)
static AicpuPhaseHeader* s_phase_header = nullptr;
static AicpuPhaseRecord* s_phase_records[PLATFORM_MAX_AICPU_THREADS] = {};
static int s_orch_thread_idx = -1;

/**
 * Enqueue ready buffer to per-thread queue
 *
 * @param header PerfDataHeader pointer
 * @param thread_idx Thread index
 * @param core_index Core index
 * @param buffer_id Buffer ID (1 or 2)
 * @return 0 on success, -1 if queue full
 */
static int enqueue_ready_buffer(PerfDataHeader* header,
                                 int thread_idx,
                                 uint32_t core_index,
                                 uint32_t buffer_id) {
    uint32_t capacity = PLATFORM_PROF_READYQUEUE_SIZE;
    uint32_t current_tail = header->queue_tails[thread_idx];
    uint32_t current_head = header->queue_heads[thread_idx];

    // Check if queue is full
    uint32_t next_tail = (current_tail + 1) % capacity;
    if (next_tail == current_head) {
        return -1;
    }

    header->queues[thread_idx][current_tail].core_index = core_index;
    header->queues[thread_idx][current_tail].buffer_id = buffer_id;
    header->queue_tails[thread_idx] = next_tail;

    return 0;
}

void perf_aicpu_init_profiling(Runtime* runtime) {
    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        LOG_ERROR("perf_data_base is NULL, cannot initialize profiling");
        return;
    }

    PerfDataHeader* header = get_perf_header(perf_base);
    DoubleBuffer* buffers = get_double_buffers(perf_base);

    int32_t task_count = runtime->get_task_count();
    header->total_tasks = static_cast<uint32_t>(task_count);

    LOG_INFO("Initializing performance profiling for %d cores", runtime->worker_count);

    // Assign buffer1 to each core for initial writing
    for (int i = 0; i < runtime->worker_count; i++) {
        Handshake* h = &runtime->workers[i];
        DoubleBuffer* db = &buffers[i];

        h->perf_records_addr = (uint64_t)&db->buffer1;
        db->buffer1_status = BufferStatus::WRITING;

        LOG_DEBUG("Core %d: assigned buffer1 (addr=0x%lx)", i, h->perf_records_addr);
    }

    wmb();

    LOG_INFO("Performance profiling initialized for %d cores", runtime->worker_count);
}

void perf_aicpu_record_dispatch_and_finish_time(PerfRecord* record,
                                                 uint64_t dispatch_time,
                                                 uint64_t finish_time) {
    rmb();

    record->dispatch_time = dispatch_time;
    record->finish_time = finish_time;

    wmb();
}


void perf_aicpu_switch_buffer(Runtime* runtime, int core_id, int thread_idx) {
    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        return;
    }

    rmb();

    Handshake* h = &runtime->workers[core_id];
    PerfDataHeader* header = get_perf_header(perf_base);
    DoubleBuffer* db = get_core_double_buffer(perf_base, core_id);

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
        full_buffer_id = 1;
        full_buf = &db->buffer1;
        full_status_ptr = &db->buffer1_status;
        alternate_buf = &db->buffer2;
        alternate_status_ptr = &db->buffer2_status;
        alternate_buffer_id = 2;
    } else if (current_addr == buffer2_addr) {
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
    // Use Runtime's method - each runtime provides its own implementation
    runtime->complete_perf_records(full_buf);

    BufferStatus alternate_status = *alternate_status_ptr;

    // Wait if alternate buffer is not ready
    if (alternate_status != BufferStatus::IDLE) {
        LOG_WARN("Thread %d: Core %d cannot switch, buffer%u status=%u, spinning until Host reads it",
                 thread_idx, core_id, alternate_buffer_id, static_cast<uint32_t>(alternate_status));

        constexpr uint64_t TIMEOUT_SECONDS = 2;

        uint64_t start_time = get_sys_cnt_aicpu();
        bool timeout = false;

        while (true) {
            rmb();
            alternate_status = *alternate_status_ptr;
            uint64_t current_time = get_sys_cnt_aicpu();
            uint64_t elapsed = current_time - start_time;

            if (alternate_status == BufferStatus::IDLE) {
                LOG_INFO("Thread %d: Core %d buffer%u now idle, proceeding with switch",
                         thread_idx, core_id, alternate_buffer_id);
                break;
            } 

            if (elapsed >= TIMEOUT_SECONDS * PLATFORM_PROF_SYS_CNT_FREQ) {
                LOG_ERROR("Thread %d: Core %d buffer%u timeout after %lu seconds (status=%u)",
                         thread_idx, core_id, alternate_buffer_id, TIMEOUT_SECONDS,
                         static_cast<uint32_t>(alternate_status));
                LOG_ERROR("Forcing buffer%u to IDLE and discarding performance data to prevent deadlock",
                         alternate_buffer_id);
                timeout = true;
                break;
            }
        }

        // Discard full buffer data on timeout to avoid deadlock
        if (timeout) {
            full_buf->count = 0;
            *full_status_ptr = BufferStatus::WRITING;

            wmb();

            LOG_ERROR("Thread %d: Core %d timeout - discarded buffer%u data, reusing it for writing",
                     thread_idx, core_id, full_buffer_id);

            return;
        }
    }

    *full_status_ptr = BufferStatus::READY;
    *alternate_status_ptr = BufferStatus::WRITING;

    // Enqueue full buffer 
    int enqueue_result = enqueue_ready_buffer(header, thread_idx, core_id, full_buffer_id);
    if (enqueue_result != 0) {
        LOG_ERROR("Thread %d: Core %d failed to enqueue buffer%u (queue full), data lost!",
                 thread_idx, core_id, full_buffer_id);
        // Revert status changes since we failed to enqueue
        *full_status_ptr = BufferStatus::WRITING;
        *alternate_status_ptr = BufferStatus::IDLE;
        return;
    }

    LOG_INFO("Thread %d: Core %d enqueued buffer%u", thread_idx, core_id, full_buffer_id);

    h->perf_records_addr = (uint64_t)alternate_buf;

    LOG_INFO("Thread %d: Core %d switched to buffer%u",
             thread_idx, core_id, alternate_buffer_id);
}

void perf_aicpu_flush_buffers(Runtime* runtime,
                               int thread_idx,
                               const int* cur_thread_cores,
                               int core_num) {
    if (!runtime->enable_profiling) {
        return;
    }

    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        return;
    }

    rmb();

    PerfDataHeader* header = get_perf_header(perf_base);
    DoubleBuffer* buffers = get_double_buffers(perf_base);

    LOG_INFO("Thread %d: Flushing performance buffers for %d cores", thread_idx, core_num);

    int flushed_count = 0;

    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        Handshake* h = &runtime->workers[core_id];
        DoubleBuffer* db = &buffers[core_id];

        uint64_t current_addr = h->perf_records_addr;
        if (current_addr == 0) {
            continue;
        }

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

        uint32_t count = current_buf->count;

        if (count > 0) {
            runtime->complete_perf_records(current_buf);

            *current_status = BufferStatus::READY;

            int rc = enqueue_ready_buffer(header, thread_idx, core_id, buffer_id);
            if (rc == 0) {
                LOG_INFO("Thread %d: Core %d flushed buffer%d with %u records",
                         thread_idx, core_id, buffer_id, count);
                flushed_count++;
            } else {
                LOG_ERROR("Thread %d: Core %d failed to enqueue buffer%d (queue full), data lost!",
                         thread_idx, core_id, buffer_id);
                // Revert status since we failed to enqueue
                *current_status = BufferStatus::WRITING;
            }
        }
    }

    LOG_INFO("Thread %d: Performance buffer flush complete, %d buffers flushed",
             thread_idx, flushed_count);
}

void perf_aicpu_update_total_tasks(Runtime* runtime, uint32_t total_tasks) {
    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        return;
    }

    PerfDataHeader* header = get_perf_header(perf_base);
    header->total_tasks = total_tasks;
    wmb();
}

void perf_aicpu_init_phase_profiling(Runtime* runtime, int num_sched_threads) {
    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        LOG_ERROR("perf_data_base is NULL, cannot initialize phase profiling");
        return;
    }

    s_phase_header = get_phase_header(perf_base, runtime->worker_count);

    s_phase_header->magic = AICPU_PHASE_MAGIC;
    s_phase_header->num_sched_threads = num_sched_threads;
    s_phase_header->records_per_thread = PLATFORM_PHASE_RECORDS_PER_THREAD;
    s_phase_header->padding = 0;

    for (int i = 0; i < PLATFORM_MAX_AICPU_THREADS; i++) {
        s_phase_header->buffer_counts[i] = 0;
    }

    memset(&s_phase_header->orch_summary, 0, sizeof(AicpuOrchSummary));

    // Cache per-thread record pointers and clear buffers
    // Include orchestrator slot (index = num_sched_threads) if within bounds
    int total_threads = (num_sched_threads < PLATFORM_MAX_AICPU_THREADS)
                        ? num_sched_threads + 1 : num_sched_threads;
    for (int t = 0; t < total_threads; t++) {
        s_phase_records[t] = get_phase_records(perf_base, runtime->worker_count, t);
        memset(s_phase_records[t], 0, PLATFORM_PHASE_RECORDS_PER_THREAD * sizeof(AicpuPhaseRecord));
    }

    wmb();

    LOG_INFO("Phase profiling initialized: %d scheduler threads (+1 orch), %d records/thread",
             num_sched_threads, PLATFORM_PHASE_RECORDS_PER_THREAD);
}

void perf_aicpu_record_phase(int thread_idx,
                              AicpuPhaseId phase_id,
                              uint64_t start_time, uint64_t end_time,
                              uint32_t loop_iter, uint32_t tasks_processed) {
    if (s_phase_header == nullptr) {
        return;
    }

    uint32_t idx = s_phase_header->buffer_counts[thread_idx];

    if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
        return;  // Buffer full, silently drop
    }

    AicpuPhaseRecord* record = &s_phase_records[thread_idx][idx];

    record->start_time = start_time;
    record->end_time = end_time;
    record->loop_iter = loop_iter;
    record->phase_id = phase_id;
    record->tasks_processed = tasks_processed;
    record->padding = 0;

    s_phase_header->buffer_counts[thread_idx] = idx + 1;
}

void perf_aicpu_write_orch_summary(const AicpuOrchSummary* src) {
    if (s_phase_header == nullptr) {
        return;
    }

    AicpuOrchSummary* dst = &s_phase_header->orch_summary;

    memcpy(dst, src, sizeof(AicpuOrchSummary));
    dst->magic = AICPU_PHASE_MAGIC;
    dst->padding = 0;

    wmb();

    LOG_INFO("Orchestrator summary written: %lld tasks, %.3fus",
             (long long)src->submit_count,
             cycles_to_us(src->end_time - src->start_time));
}

void perf_aicpu_set_orch_thread_idx(int thread_idx) {
    s_orch_thread_idx = thread_idx;
}

void perf_aicpu_record_orch_phase(AicpuPhaseId phase_id,
                                   uint64_t start_time, uint64_t end_time,
                                   uint32_t submit_idx) {
    if (s_orch_thread_idx < 0 || s_phase_header == nullptr) return;
    perf_aicpu_record_phase(s_orch_thread_idx, phase_id, start_time, end_time, submit_idx, 0);
}
