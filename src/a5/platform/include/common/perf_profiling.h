/**
 * @file perf_profiling.h
 * @brief Performance profiling data structures
 *
 * Architecture: Fixed header + dynamic tail + optional phase profiling region
 *
 * Memory layout:
 * ┌─────────────────────────────────────────────────────────────┐
 * │ PerfDataHeader (fixed header)                               │
 * │  - ReadyQueue (FIFO, capacity=PLATFORM_PROF_READYQUEUE_SIZE)│
 * │  - Metadata (num_cores, buffer_capacity, flags)             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ DoubleBuffer[0] (Core 0)                                    │
 * │  - buffer1, buffer2 (PerfBuffer)                            │
 * │  - buffer1_status, buffer2_status (IDLE/WRITING/READY)      │
 * ├─────────────────────────────────────────────────────────────┤
 * │ DoubleBuffer[1] (Core 1)                                    │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ...                                                         │
 * ├─────────────────────────────────────────────────────────────┤
 * │ DoubleBuffer[num_cores-1]                                   │
 * ├─────────────────────────────────────────────────────────────┤
 * │ AicpuPhaseHeader (optional, present when phase profiling)   │
 * │  - magic, num_sched_threads, records_per_thread             │
 * │  - buffer_counts[PLATFORM_MAX_AICPU_THREADS]                │
 * │  - orch_summary                                             │
 * ├─────────────────────────────────────────────────────────────┤
 * │ AicpuPhaseRecord[thread0][0..records_per_thread-1]          │
 * ├─────────────────────────────────────────────────────────────┤
 * │ AicpuPhaseRecord[thread1][0..records_per_thread-1]          │
 * ├─────────────────────────────────────────────────────────────┤
 * │ ...                                                         │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Base size = sizeof(PerfDataHeader) + num_cores * sizeof(DoubleBuffer)
 * With phases = Base + sizeof(AicpuPhaseHeader) + num_sched_threads * records_per_thread * sizeof(AicpuPhaseRecord)
 */

#ifndef PLATFORM_COMMON_PERF_PROFILING_H_
#define PLATFORM_COMMON_PERF_PROFILING_H_

#include <cstdint>
#include <vector>

#include "platform_config.h"
#include "core_type.h"

// Maximum number of successor tasks per PerfRecord (matches Task::fanout)
#ifndef RUNTIME_MAX_FANOUT
#define RUNTIME_MAX_FANOUT 512
#endif

// =============================================================================
// Buffer Status Enumeration
// =============================================================================

/**
 * Buffer status enumeration (3-state design)
 *
 * State transition flow:
 * IDLE (0) → WRITING (1) → READY (2) → IDLE (0)
 *
 * - AICPU: IDLE→WRITING (on allocation), WRITING→READY (when buffer full)
 * - AICore: Only writes data, does not modify status
 * - Host:   READY→IDLE (after reading)
 *
 * Note: Using uint32_t for binary compatibility with volatile fields.
 */
enum class BufferStatus : uint32_t {
    IDLE    = 0,  // Idle: can be allocated by AICPU
    WRITING = 1,  // Writing: in use by AICore
    READY   = 2   // Ready: full, waiting for Host
};

// =============================================================================
// PerfRecord - Single Task Execution Record
// =============================================================================

/**
 * Single task execution record
 */
struct PerfRecord {
    // Timing information (device clock timestamps)
    uint64_t start_time;         // Task start timestamp (get_sys_cnt)
    uint64_t end_time;           // Task end timestamp
    uint64_t duration;           // Execution duration (end - start)
    uint64_t kernel_ready_time;  // Kernel ready timestamp (before first task)
                                 // Records when AICore enters main loop (ready to execute)
                                 // Used for: 1) startup overhead analysis, 2) cross-core alignment

    // AICPU-side timestamps (written by AICPU, not AICore)
    uint64_t dispatch_time;      // AICPU timestamp: when task was dispatched to AICore (task_status set to 1)
    uint64_t finish_time;        // AICPU timestamp: when AICPU observed task completion (task_status back to 0)

    // Task identification
    uint32_t task_id;         // Task unique identifier
    uint32_t func_id;         // Kernel function identifier
    CoreType core_type;       // Core type (AIC/AIV)

    // Dependency relationship (fanout only)
    int32_t fanout[RUNTIME_MAX_FANOUT];  // Successor task ID array
    int32_t fanout_count;                 // Number of successor tasks
} __attribute__((aligned(64)));

static_assert(sizeof(PerfRecord) % 64 == 0,
              "PerfRecord must be 64-byte aligned for optimal cache performance");

// =============================================================================
// PerfBuffer - Fixed-Size Record Buffer
// =============================================================================

/**
 * Fixed-size performance record buffer
 *
 * Capacity: PLATFORM_PROF_BUFFER_SIZE (defined in platform_config.h)
 */
struct PerfBuffer {
    PerfRecord records[PLATFORM_PROF_BUFFER_SIZE];  // Record array
    volatile uint32_t count;                         // Current record count
} __attribute__((aligned(64)));

// =============================================================================
// DoubleBuffer - Per-Core Ping-Pong Buffers
// =============================================================================

/**
 * Per-core double buffer with status management
 *
 * Two independent PerfBuffers with independent status fields.
 * AICPU manages buffer allocation and status transitions (0→1→2).
 * AICore only writes records and increments count.
 * Host reads ready buffers and resets status to idle (2→0).
 *
 * When both buffers are idle, AICPU prioritizes buffer1.
 */
struct DoubleBuffer {
    // Buffer 1 (Ping)
    PerfBuffer buffer1;                              // First buffer
    volatile BufferStatus buffer1_status;            // Buffer1 status (IDLE/WRITING/READY)

    // Buffer 2 (Pong)
    PerfBuffer buffer2;                              // Second buffer
    volatile BufferStatus buffer2_status;            // Buffer2 status (IDLE/WRITING/READY)
} __attribute__((aligned(64)));

// =============================================================================
// ReadyQueueEntry - Queue Entry for Ready Buffers
// =============================================================================

/**
 * Ready queue entry
 *
 * When a buffer on a core is full, AICPU adds this entry to the queue.
 * Host retrieves entries from the queue to locate (core_index, buffer_id) for reading.
 */
struct ReadyQueueEntry {
    uint32_t core_index;      // Core index (0 ~ num_cores-1)
    uint32_t buffer_id;       // Buffer ID (1=buffer1, 2=buffer2)
} __attribute__((aligned(16)));

// =============================================================================
// PerfDataHeader - Fixed Header
// =============================================================================

/**
 * Performance data fixed header
 *
 * Located at the start of shared memory, contains:
 * 1. Per-thread ready queues (FIFO Circular Buffers)
 * 2. Metadata (core count)
 *
 * Ready queue design:
 * - Per-thread queues: Avoid lock contention between AICPU threads
 * - Capacity per queue: PLATFORM_PROF_READYQUEUE_SIZE (full capacity for each thread)
 * - Implementation: Circular Buffer
 * - Producer: AICPU thread (adds full buffers to its own queue)
 * - Consumer: Host (reads from all queues)
 * - Queue empty: head == tail
 * - Queue full: (tail + 1) % capacity == head
 */
struct PerfDataHeader {
    // Per-thread ready queues (FIFO Circular Buffers)
    // Each AICPU thread has its own queue to avoid lock contention
    ReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_PROF_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Consumer read positions (Host modifies)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // Producer write positions (AICPU modifies)

    // Metadata (Host initializes, Device read-only)
    uint32_t num_cores;                              // Actual number of cores launched
    volatile uint32_t total_tasks;                   // Total tasks (AICPU writes after orchestration)
} __attribute__((aligned(64)));

// =============================================================================
// AICPU Phase Profiling - Scheduler and Orchestrator Records
// =============================================================================

/**
 * AICPU phase identifier
 *
 * Scheduler phases (0-3): four phases in each scheduler loop iteration.
 * Orchestrator phases (16-24): sub-steps within each pto2_submit_task() call.
 */
enum class AicpuPhaseId : uint32_t {
    // Scheduler phases (0-3)
    SCHED_COMPLETE    = 0,  // Process completed tasks (fanout traversal)
    SCHED_DISPATCH    = 1,  // Dispatch ready tasks to idle cores
    SCHED_SCAN        = 2,  // Incremental scan for root tasks
    SCHED_EARLY_READY = 3,  // Drain orchestrator's early-ready queue
    // Orchestrator phases (16-24)
    ORCH_SYNC      = 16,  // tensormap sync
    ORCH_ALLOC     = 17,  // task_ring_alloc
    ORCH_PARAMS    = 18,  // param copy
    ORCH_LOOKUP    = 19,  // tensormap lookup + dep
    ORCH_HEAP      = 20,  // heap alloc
    ORCH_INSERT    = 21,  // tensormap insert
    ORCH_FANIN     = 22,  // fanin + early-ready
    ORCH_FINALIZE  = 23,  // scheduler init + SM
    ORCH_SCOPE_END = 24   // scope_end
};

/**
 * Single AICPU scheduler phase record (32 bytes)
 *
 * Records one phase within one loop iteration of a scheduler thread.
 * No thread_id field: identity is derived from array index (position = identity).
 */
struct AicpuPhaseRecord {
    uint64_t start_time;       // Phase start timestamp
    uint64_t end_time;         // Phase end timestamp
    uint32_t loop_iter;        // Loop iteration number
    AicpuPhaseId phase_id;     // Phase type
    uint32_t tasks_processed;  // Tasks processed in this phase
    uint32_t padding;          // Alignment padding
};

/**
 * AICPU orchestrator cumulative summary
 *
 * Contains accumulated cycle counts from the orchestrator thread.
 * Written once after orchestration completes.
 */
struct AicpuOrchSummary {
    uint64_t start_time;       // Orchestrator start timestamp
    uint64_t end_time;         // Orchestrator end timestamp
    uint64_t sync_cycle;       // sync_tensormap phase
    uint64_t alloc_cycle;      // task_ring_alloc phase
    uint64_t params_cycle;     // param_copy phase
    uint64_t lookup_cycle;     // lookup+dep phase
    uint64_t heap_cycle;       // heap_alloc phase
    uint64_t insert_cycle;     // tensormap_insert phase
    uint64_t fanin_cycle;      // fanin+ready phase
    uint64_t finalize_cycle;   // finalize+SM phase
    uint64_t scope_end_cycle;  // scope_end phase
    int64_t  submit_count;     // Total tasks submitted
    uint32_t magic;            // Validation magic (AICPU_PHASE_MAGIC)
    uint32_t padding;          // Alignment padding
} __attribute__((aligned(64)));

constexpr uint32_t AICPU_PHASE_MAGIC = 0x41435048;  // "ACPH"
constexpr int PLATFORM_PHASE_RECORDS_PER_THREAD = 16384;  // ~512KB per thread

/**
 * AICPU phase profiling header
 *
 * Located after the DoubleBuffer array in shared memory.
 * Contains metadata and per-thread record counts.
 */
struct AicpuPhaseHeader {
    uint32_t magic;                  // Validation magic (AICPU_PHASE_MAGIC)
    uint32_t num_sched_threads;      // Number of scheduler threads
    uint32_t records_per_thread;     // Max records per thread
    uint32_t padding;                // Alignment padding
    volatile uint32_t buffer_counts[PLATFORM_MAX_AICPU_THREADS];  // Per-thread record counts
    AicpuOrchSummary orch_summary;   // Orchestrator cumulative data
} __attribute__((aligned(64)));

// =============================================================================
// Helper Functions - Memory Layout
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculate total memory size for performance data
 *
 * Formula: Total size = Fixed header + Dynamic tail
 *                     = sizeof(PerfDataHeader) + num_cores × sizeof(DoubleBuffer)
 *
 * @param num_cores Number of cores (block_dim × PLATFORM_CORES_PER_BLOCKDIM)
 * @return Total bytes
 */
inline size_t calc_perf_data_size(int num_cores) {
    return sizeof(PerfDataHeader) + num_cores * sizeof(DoubleBuffer);
}

/**
 * Get header pointer
 *
 * @param base_ptr Shared memory base address (device_ptr or host_ptr)
 * @return PerfDataHeader pointer
 */
inline PerfDataHeader* get_perf_header(void* base_ptr) {
    return (PerfDataHeader*)base_ptr;
}

/**
 * Get DoubleBuffer array start address
 *
 * @param base_ptr Shared memory base address
 * @return DoubleBuffer array pointer
 */
inline DoubleBuffer* get_double_buffers(void* base_ptr) {
    return (DoubleBuffer*)((char*)base_ptr + sizeof(PerfDataHeader));
}

/**
 * Get DoubleBuffer for specified core
 *
 * @param base_ptr Shared memory base address
 * @param core_index Core index (0 ~ num_cores-1)
 * @return DoubleBuffer pointer
 */
inline DoubleBuffer* get_core_double_buffer(void* base_ptr, int core_index) {
    DoubleBuffer* buffers = get_double_buffers(base_ptr);
    return &buffers[core_index];
}

/**
 * Get buffer pointer and status pointer for specified buffer
 *
 * @param db DoubleBuffer pointer
 * @param buffer_id Buffer ID (1=buffer1, 2=buffer2)
 * @param[out] buf PerfBuffer pointer
 * @param[out] status Status pointer
 */
inline void get_buffer_and_status(DoubleBuffer* db, uint32_t buffer_id,
                                  PerfBuffer** buf, volatile BufferStatus** status) {
    if (buffer_id == 1) {
        *buf = &db->buffer1;
        *status = &db->buffer1_status;
    } else {
        *buf = &db->buffer2;
        *status = &db->buffer2_status;
    }
}

/**
 * Calculate total memory size including phase profiling region
 *
 * @param num_cores Number of AICore instances
 * @param num_sched_threads Number of scheduler threads (typically 3)
 * @return Total bytes needed
 */
inline size_t calc_perf_data_size_with_phases(int num_cores, int num_sched_threads) {
    return calc_perf_data_size(num_cores)
         + sizeof(AicpuPhaseHeader)
         + num_sched_threads * PLATFORM_PHASE_RECORDS_PER_THREAD * sizeof(AicpuPhaseRecord);
}

/**
 * Get AicpuPhaseHeader pointer (located after DoubleBuffer array)
 *
 * @param base_ptr Shared memory base address
 * @param num_cores Number of AICore instances
 * @return AicpuPhaseHeader pointer
 */
inline AicpuPhaseHeader* get_phase_header(void* base_ptr, int num_cores) {
    return (AicpuPhaseHeader*)((char*)base_ptr + calc_perf_data_size(num_cores));
}

/**
 * Get AicpuPhaseRecord array for specified thread
 *
 * @param base_ptr Shared memory base address
 * @param num_cores Number of AICore instances
 * @param thread_idx Scheduler thread index
 * @return AicpuPhaseRecord array pointer
 */
inline AicpuPhaseRecord* get_phase_records(void* base_ptr, int num_cores, int thread_idx) {
    char* phase_start = (char*)get_phase_header(base_ptr, num_cores) + sizeof(AicpuPhaseHeader);
    return (AicpuPhaseRecord*)(phase_start + thread_idx * PLATFORM_PHASE_RECORDS_PER_THREAD * sizeof(AicpuPhaseRecord));
}

#ifdef __cplusplus
}
#endif

#endif  // PLATFORM_COMMON_PERF_PROFILING_H_
