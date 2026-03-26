/**
 * @file performance_collector_aicpu.cpp
 * @brief AICPU performance data collection implementation (SPSC free queue)
 *
 * Uses per-core PerfBufferState with SPSC free queues for O(1) buffer switching.
 * Host memory manager dynamically allocates replacement buffers and pushes
 * them into the free_queue. Device pops from free_queue when switching.
 */

#include "aicpu/performance_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "common/platform_config.h"

#include <cstring>

// Cached pointers for hot-path access (set during init)
static AicpuPhaseHeader* s_phase_header = nullptr;
static PerfDataHeader* s_perf_header = nullptr;

// Per-core PerfBufferState cache
static PerfBufferState* s_perf_buffer_states[PLATFORM_MAX_CORES] = {};

// Per-thread PhaseBufferState cache
static PhaseBufferState* s_phase_buffer_states[PLATFORM_MAX_AICPU_THREADS] = {};
static PhaseBuffer* s_current_phase_buf[PLATFORM_MAX_AICPU_THREADS] = {};

static __thread int s_orch_thread_idx = -1;

/**
 * Enqueue ready buffer to per-thread queue
 *
 * @param header PerfDataHeader pointer
 * @param thread_idx Thread index
 * @param core_index Core index (or thread_idx for phase entries)
 * @param buffer_ptr Device pointer to the full buffer
 * @param buffer_seq Sequence number for ordering
 * @param is_phase 0 = PerfRecord, 1 = Phase
 * @return 0 on success, -1 if queue full
 */
static int enqueue_ready_buffer(PerfDataHeader* header,
    int thread_idx,
    uint32_t core_index,
    uint64_t buffer_ptr,
    uint32_t buffer_seq,
    uint32_t is_phase) {
    uint32_t capacity = PLATFORM_PROF_READYQUEUE_SIZE;
    uint32_t current_tail = header->queue_tails[thread_idx];
    uint32_t current_head = header->queue_heads[thread_idx];

    // Check if queue is full
    uint32_t next_tail = (current_tail + 1) % capacity;
    if (next_tail == current_head) {
        return -1;
    }

    header->queues[thread_idx][current_tail].core_index = core_index;
    header->queues[thread_idx][current_tail].is_phase = is_phase;
    header->queues[thread_idx][current_tail].buffer_ptr = buffer_ptr;
    header->queues[thread_idx][current_tail].buffer_seq = buffer_seq;
    header->queue_tails[thread_idx] = next_tail;

    return 0;
}

void perf_aicpu_init_profiling(Runtime* runtime) {
    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        LOG_ERROR("perf_data_base is NULL, cannot initialize profiling");
        return;
    }

    s_perf_header = get_perf_header(perf_base);

    int32_t task_count = runtime->get_task_count();
    s_perf_header->total_tasks = static_cast<uint32_t>(task_count);

    LOG_INFO("Initializing performance profiling for %d cores (free queue)", runtime->worker_count);

    // Pop first buffer from free_queue for each core
    for (int i = 0; i < runtime->worker_count; i++) {
        Handshake* h = &runtime->workers[i];
        PerfBufferState* state = get_perf_buffer_state(perf_base, i);

        s_perf_buffer_states[i] = state;

        // Pop first buffer from free_queue
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = 0;
            wmb();

            PerfBuffer* buf = (PerfBuffer*)buf_ptr;
            buf->count = 0;
            h->perf_records_addr = buf_ptr;

            LOG_DEBUG("Core %d: popped initial buffer (addr=0x%lx)", i, buf_ptr);
        } else {
            LOG_ERROR("Core %d: free_queue is empty during init!", i);
            state->current_buf_ptr = 0;
            h->perf_records_addr = 0;
        }
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

    PerfBufferState* state = s_perf_buffer_states[core_id];
    if (state == nullptr) {
        return;
    }

    // Complete performance records (fill fanout info)
    PerfBuffer* full_buf = (PerfBuffer*)state->current_buf_ptr;
    if (full_buf == nullptr) {
        return;
    }
    runtime->complete_perf_records(full_buf);

    LOG_INFO("Thread %d: Core %d buffer is full (count=%u)",
         thread_idx, core_id, full_buf->count);

    // Enqueue to ReadyQueue
    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_ready_buffer(s_perf_header, thread_idx, core_id,
        state->current_buf_ptr, seq, 0);
    if (rc != 0) {
        LOG_ERROR("Thread %d: Core %d failed to enqueue buffer (queue full), data lost!",
             thread_idx, core_id);
        // Revert: discard data and keep writing
        full_buf->count = 0;
        wmb();
        return;
    }

    // Pop next buffer from free_queue
    rmb();
    uint32_t head = state->free_queue.head;
    uint32_t tail = state->free_queue.tail;

    if (head != tail) {
        uint64_t new_buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
        rmb();
        state->free_queue.head = head + 1;
        state->current_buf_ptr = new_buf_ptr;
        state->current_buf_seq = seq + 1;
        wmb();

        PerfBuffer* new_buf = (PerfBuffer*)new_buf_ptr;
        new_buf->count = 0;

        // Update handshake for AICore
        Handshake* h = &runtime->workers[core_id];
        h->perf_records_addr = new_buf_ptr;
        wmb();

        LOG_INFO("Thread %d: Core %d switched to new buffer (addr=0x%lx)", 
            thread_idx, core_id, new_buf_ptr);
    } else {
        // No free buffer available, stop profiling
        LOG_WARN("Thread %d: Core %d no free buffer available, stopping profiling", 
            thread_idx, core_id);
        state->current_buf_ptr = 0;
        Handshake* h = &runtime->workers[core_id];
        h->perf_records_addr = 0;
        wmb();
    }
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

    LOG_INFO("Thread %d: Flushing performance buffers for %d cores", thread_idx, core_num);

    int flushed_count = 0;

    for (int i = 0; i < core_num; i++) {
        int core_id = cur_thread_cores[i];
        PerfBufferState* state = s_perf_buffer_states[core_id];
        if (state == nullptr) continue;

        rmb();
        uint64_t buf_ptr = state->current_buf_ptr;
        if (buf_ptr == 0) {
            // No active buffer
            continue;
        }

        PerfBuffer* buf = (PerfBuffer*)buf_ptr;
        if (buf->count == 0) {
            continue;
        }

        runtime->complete_perf_records(buf);

        uint32_t seq = state->current_buf_seq;
        int rc = enqueue_ready_buffer(s_perf_header, thread_idx, core_id,
            buf_ptr, seq, 0);
        if (rc == 0) {
            LOG_INFO("Thread %d: Core %d flushed buffer with %u records",
                thread_idx, core_id, buf->count);
            flushed_count++;
            state->current_buf_ptr = 0;
            wmb();
        } else {
            LOG_ERROR("Thread %d: Core %d failed to enqueue buffer (queue full), data lost!",
                thread_idx, core_id);
        }
    }

    wmb();

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

void perf_aicpu_init_phase_profiling(Runtime* runtime, int num_sched_threads, int num_orch_threads) {
    void* perf_base = (void*)runtime->perf_data_base;
    if (perf_base == nullptr) {
        LOG_ERROR("perf_data_base is NULL, cannot initialize phase profiling");
        return;
    }

    s_phase_header = get_phase_header(perf_base, runtime->worker_count);
    s_perf_header = get_perf_header(perf_base);

    s_phase_header->magic = AICPU_PHASE_MAGIC;
    s_phase_header->num_sched_threads = num_sched_threads;
    s_phase_header->records_per_thread = PLATFORM_PHASE_RECORDS_PER_THREAD;
    s_phase_header->num_cores = 0;

    memset(s_phase_header->core_to_thread, -1, sizeof(s_phase_header->core_to_thread));
    memset(&s_phase_header->orch_summary, 0, sizeof(AicpuOrchSummary));

    // Cache per-thread record pointers and clear buffers
    // Include all threads: scheduler + orchestrator (orchestrators may become schedulers)
    int total_threads = num_sched_threads + num_orch_threads;
    if (total_threads > PLATFORM_MAX_AICPU_THREADS) {
        total_threads = PLATFORM_MAX_AICPU_THREADS;
    }
    for (int t = 0; t < total_threads; t++) {
        PhaseBufferState* state = get_phase_buffer_state(perf_base, runtime->worker_count, t);

        s_phase_buffer_states[t] = state;

        // Pop first buffer from free_queue
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = 0;
            wmb();

            PhaseBuffer* buf = (PhaseBuffer*)buf_ptr;
            buf->count = 0;
            s_current_phase_buf[t] = buf;

            LOG_DEBUG("Thread %d: popped initial phase buffer (addr=0x%lx)", t, buf_ptr);
        } else {
            LOG_ERROR("Thread %d: phase free_queue is empty during init!", t);
            state->current_buf_ptr = 0;
            s_current_phase_buf[t] = nullptr;
        }
    }

    // Clear remaining slots
    for (int t = total_threads; t < PLATFORM_MAX_AICPU_THREADS; t++) {
        s_phase_buffer_states[t] = nullptr;
        s_current_phase_buf[t] = nullptr;
    }

    wmb();

    LOG_INFO("Phase profiling initialized: %d scheduler + %d orch threads, %d records/thread",
        num_sched_threads, num_orch_threads, PLATFORM_PHASE_RECORDS_PER_THREAD);
}

/**
 * Switch phase buffer when current buffer is full (free queue version)
 *
 * Enqueues the full buffer to ReadyQueue and pops the next buffer from free_queue.
 * If no free buffer is available, sets s_current_phase_buf to nullptr so subsequent
 * records are dropped (preserving already-enqueued data).
 */
static void switch_phase_buffer(int thread_idx) {
    PhaseBufferState* state = s_phase_buffer_states[thread_idx];
    if (state == nullptr) return;

    PhaseBuffer* full_buf = s_current_phase_buf[thread_idx];
    if (full_buf == nullptr) return;

    LOG_INFO("Thread %d: phase buffer is full (count=%u)",
        thread_idx, full_buf->count);

    // Enqueue to ReadyQueue
    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_ready_buffer(s_perf_header, thread_idx, thread_idx,
        state->current_buf_ptr, seq, 1);
    if (rc != 0) {
        LOG_ERROR("Thread %d: failed to enqueue phase buffer (queue full), discarding data",
            thread_idx);
        full_buf->count = 0;
        wmb();
        return;
    }

    // Pop next buffer from free_queue
    rmb();
    uint32_t head = state->free_queue.head;
    uint32_t tail = state->free_queue.tail;

    if (head != tail) {
        uint64_t new_buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
        rmb();
        state->free_queue.head = head + 1;
        state->current_buf_ptr = new_buf_ptr;
        state->current_buf_seq = seq + 1;
        wmb();

        PhaseBuffer* new_buf = (PhaseBuffer*)new_buf_ptr;
        new_buf->count = 0;
        s_current_phase_buf[thread_idx] = new_buf;

        LOG_INFO("Thread %d: switched to new phase buffer", thread_idx);
    } else {
        // No free buffer available, drop subsequent records
        LOG_WARN("Thread %d: no free phase buffer available, dropping records until Host catches up",
            thread_idx);
        s_current_phase_buf[thread_idx] = nullptr;
        state->current_buf_ptr = 0;
        wmb();
    }
}

void perf_aicpu_record_phase(int thread_idx,
    AicpuPhaseId phase_id,
    uint64_t start_time, uint64_t end_time,
    uint32_t loop_iter, uint64_t tasks_processed) {
    if (s_phase_header == nullptr) {
        return;
    }

    PhaseBuffer* buf = s_current_phase_buf[thread_idx];

    // Try to recover from nullptr (no buffer was available on previous switch)
    if (buf == nullptr) {
        PhaseBufferState* state = s_phase_buffer_states[thread_idx];
        if (state == nullptr) return;

        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;

        if (head != tail) {
            uint64_t buf_ptr = state->free_queue.buffer_ptrs[head % PLATFORM_PROF_SLOT_COUNT];
            rmb();
            state->free_queue.head = head + 1;
            state->current_buf_ptr = buf_ptr;
            state->current_buf_seq = state->current_buf_seq + 1;
            wmb();

            buf = (PhaseBuffer*)buf_ptr;
            buf->count = 0;
            s_current_phase_buf[thread_idx] = buf;

            LOG_INFO("Thread %d: recovered phase buffer", thread_idx);
        }
        if (buf == nullptr) return;  // Still no buffer available
    }

    uint32_t idx = buf->count;

    if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
        // Buffer full, switch to next buffer
        switch_phase_buffer(thread_idx);
        buf = s_current_phase_buf[thread_idx];
        if (buf == nullptr) return;  // No buffer available
        idx = buf->count;
        if (idx >= PLATFORM_PHASE_RECORDS_PER_THREAD) {
            return;  // Switch failed; drop this record
        }
    }

    AicpuPhaseRecord* record = &buf->records[idx];
    record->start_time = start_time;
    record->end_time = end_time;
    record->loop_iter = loop_iter;
    record->phase_id = phase_id;
    record->task_id = tasks_processed;

    buf->count = idx + 1;
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
    uint32_t submit_idx, uint64_t task_id) {
    if (s_orch_thread_idx < 0 || s_phase_header == nullptr) return;
    perf_aicpu_record_phase(s_orch_thread_idx, phase_id, start_time, end_time, submit_idx, task_id);
}

void perf_aicpu_flush_phase_buffers(int thread_idx) {
    if (s_phase_header == nullptr || s_perf_header == nullptr) {
        return;
    }

    PhaseBufferState* state = s_phase_buffer_states[thread_idx];
    if (state == nullptr) return;

    rmb();
    uint64_t buf_ptr = state->current_buf_ptr;
    if (buf_ptr == 0) {
        // No active buffer
        return;
    }

    PhaseBuffer* buf = (PhaseBuffer*)buf_ptr;
    if (buf->count == 0) {
        return;
    }

    uint32_t seq = state->current_buf_seq;
    int rc = enqueue_ready_buffer(s_perf_header, thread_idx, thread_idx,
        buf_ptr, seq, 1);
    if (rc == 0) {
        LOG_INFO("Thread %d: flushed phase buffer with %u records",
            thread_idx, buf->count);
        state->current_buf_ptr = 0;
        s_current_phase_buf[thread_idx] = nullptr;
        wmb();
    } else {
        LOG_ERROR("Thread %d: failed to enqueue phase buffer (queue full), data lost!",
            thread_idx);
    }

    wmb();
}

void perf_aicpu_write_core_assignments(const int core_assignments[][PLATFORM_MAX_CORES_PER_THREAD],
    const int* core_counts,
    int num_threads,
    int total_cores) {
    if (s_phase_header == nullptr) {
        return;
    }

    memset(s_phase_header->core_to_thread, -1, sizeof(s_phase_header->core_to_thread));
    s_phase_header->num_cores = static_cast<uint32_t>(total_cores);

    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < core_counts[t]; i++) {
            int core_id = core_assignments[t][i];
            if (core_id >= 0 && core_id < PLATFORM_MAX_CORES) {
                s_phase_header->core_to_thread[core_id] = static_cast<int8_t>(t);
            }
        }
    }

    wmb();

    LOG_INFO("Core-to-thread mapping written: %d cores, %d threads", total_cores, num_threads);
}
