/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file pmu_profiling.h
 * @brief DAV_3510 (a5) AICore Performance Monitoring Unit configuration
 *
 * PMU event ID tables (values from pypto's aicore_prof.h DAV_3510 config,
 * CANN Open Software License 2.0). Register offsets live in platform_config.h
 * and are accessed via RegId / reg_index().
 *
 * Streaming buffer design (mirrors a2a3 pmu_profiling.h):
 *   PmuFreeQueue       — SPSC queue: Host pushes free PmuBuffers, AICPU pops them.
 *   PmuBufferState     — Per-core state: current active buffer pointer + free_queue.
 *   PmuDataHeader      — Fixed shared-memory header: per-thread ready queues.
 *   PmuBuffer          — Fixed-capacity record buffer (PLATFORM_PMU_RECORDS_PER_BUFFER).
 *
 * a5 has no halHostRegister (DAV_3510), so host↔device SPSC fields are
 * read/written via rtMemcpy (onboard) or memcpy (sim), using host shadow
 * buffers — same pattern as a5 l2_perf_collector and tensor_dump_collector.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_COMMON_PMU_PROFILING_H_
#define SRC_A5_PLATFORM_INCLUDE_COMMON_PMU_PROFILING_H_

#include <cstdint>
#include <cstddef>

#include "common/core_type.h"
#include "common/platform_config.h"

/**
 * PMU event type selector. Values match pypto's PROF_PMU_EVENT_TYPE (see
 * pmu_common.cpp::SetPmuEventTypeDAV3510 for the per-counter event IDs
 * used on DAV_3510).
 */
enum class PmuEventType : uint32_t {
    ARITHMETIC_UTILIZATION = 1,
    PIPE_UTILIZATION = 2,  // default
    MEMORY = 4,
    MEMORY_L0 = 5,
    RESOURCE_CONFLICT = 6,
    MEMORY_UB = 7,
    L2_CACHE = 8,
};

constexpr uint32_t PMU_EVENT_TYPE_DEFAULT = static_cast<uint32_t>(PmuEventType::PIPE_UTILIZATION);
constexpr int PMU_COUNTER_COUNT_A5 = 10;

/**
 * Event ID table for a single event type.
 * event_ids[i] programs PMU_CNTi_IDX; counters[i] in the PmuRecord is the
 * value of PMU_CNTi after the task completes.
 * counter_names[i] is the human-readable CSV column name for counter i.
 * Empty string ("") marks an unused slot.
 *
 * Names match pypto's tilefwk_pmu_to_csv.py table_pmu_header_3510 tables.
 * a5 has 10 counter slots; unused slots are "" / 0.
 */
struct PmuEventConfig {
    uint32_t event_ids[PMU_COUNTER_COUNT_A5];
    const char *counter_names[PMU_COUNTER_COUNT_A5];
};

// DAV_3510 event tables. Event IDs come from pypto's
// pmu_common.cpp::SetPmuEventTypeDAV3510; counter names come from pypto's
// tilefwk_pmu_to_csv.py table_pmu_header_3510. Empty string "" marks an
// unused counter slot.
constexpr PmuEventConfig PMU_EVENTS_A5_ARITHMETIC = {
    {0x323, 0x324, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
    {"cube_fp_instr_busy", "cube_int_instr_busy", "", "", "", "", "", "", "", ""},
};
constexpr PmuEventConfig PMU_EVENTS_A5_PIPE_UTIL = {
    {0x501, 0x301, 0x1, 0x701, 0x202, 0x203, 0x34, 0x35, 0x714, 0x0},
    {"pmu_idc_aic_vec_busy_o", "cube_instr_busy", "scalar_instr_busy", "mte1_instr_busy", "mte2_instr_busy",
     "mte3_instr_busy", "icache_req", "icache_miss", "pmu_fix_instr_busy", ""},
};
constexpr PmuEventConfig PMU_EVENTS_A5_MEMORY = {
    {0x0, 0x0, 0x400, 0x401, 0x56f, 0x571, 0x570, 0x572, 0x707, 0x709},
    {"", "", "bif_sc_pmu_read_main_instr_core", "bif_sc_pmu_write_main_instr_core", "pmu_aiv_ext_rd_ub_instr",
     "ub_pmu_vec_rd_ub_acc", "pmu_aiv_ext_wr_ub_instr", "ub_pmu_vec_wr_ub_acc", "pmu_rd_l1_instr", "pmu_wr_l1_instr"},
};
constexpr PmuEventConfig PMU_EVENTS_A5_MEMORY_L0 = {
    {0x304, 0x703, 0x306, 0x705, 0x712, 0x30a, 0x308, 0x0, 0x0, 0x0},
    {"cube_sc_pmu_read_l0a_instr", "pmu_wr_l0a_instr", "cube_sc_pmu_read_l0b_instr", "pmu_wr_l0b_instr",
     "fixp_rd_l0c_instr", "cube_sc_pmu_read_l0c_instr", "cube_sc_pmu_write_l0c_instr", "", "", ""},
};
constexpr PmuEventConfig PMU_EVENTS_A5_RESOURCE_CONFLICT = {
    {0x3556, 0x3540, 0x3502, 0x3528, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
    {"stu_pmu_wctl_ub_cflt", "ldu_pmu_ib_ub_cflt", "pmu_idc_aic_vec_instr_vf_busy_o", "idu_pmu_ins_iss_cnt", "", "", "",
     "", "", ""},
};
constexpr PmuEventConfig PMU_EVENTS_A5_MEMORY_UB = {
    {0x3, 0x5, 0x70c, 0x206, 0x204, 0x571, 0x572, 0x0, 0x0, 0x0},
    {"pmu_rd_acc_ub_instr_p", "pmu_wr_acc_ub_instr_p", "pmu_fix_wr_ub_instr", "mte_sc_pmu_write_acc_ub_instr_0",
     "mte_sc_pmu_read_acc_ub_instr_0", "ub_pmu_vec_rd_ub_acc", "ub_pmu_vec_wr_ub_acc", "", "", ""},
};
constexpr PmuEventConfig PMU_EVENTS_A5_L2_CACHE = {
    {0x424, 0x425, 0x426, 0x42a, 0x42b, 0x42c, 0x0, 0x0, 0x0, 0x0},
    {"bif_sc_pmu_ar_close_l2_hit_core", "bif_sc_pmu_ar_close_l2_miss_core", "bif_sc_pmu_ar_close_l2_victim_core",
     "bif_sc_pmu_aw_close_l2_hit_core", "bif_sc_pmu_aw_close_l2_miss_core", "bif_sc_pmu_aw_close_l2_victim_core", "",
     "", "", ""},
};

/**
 * Resolve an event type to the DAV_3510 event table. Returns nullptr for
 * unknown values (caller falls back to PIPE_UTILIZATION).
 */
inline const PmuEventConfig *pmu_resolve_event_config_a5(PmuEventType event_type) {
    switch (event_type) {
    case PmuEventType::ARITHMETIC_UTILIZATION:
        return &PMU_EVENTS_A5_ARITHMETIC;
    case PmuEventType::PIPE_UTILIZATION:
        return &PMU_EVENTS_A5_PIPE_UTIL;
    case PmuEventType::MEMORY:
        return &PMU_EVENTS_A5_MEMORY;
    case PmuEventType::MEMORY_L0:
        return &PMU_EVENTS_A5_MEMORY_L0;
    case PmuEventType::RESOURCE_CONFLICT:
        return &PMU_EVENTS_A5_RESOURCE_CONFLICT;
    case PmuEventType::MEMORY_UB:
        return &PMU_EVENTS_A5_MEMORY_UB;
    case PmuEventType::L2_CACHE:
        return &PMU_EVENTS_A5_L2_CACHE;
    }
    return nullptr;
}

// =============================================================================
// PMU Record
// =============================================================================

/**
 * Per-task PMU snapshot written by AICPU after each AICore task FIN.
 *
 * AICore writes task_id / pmu_total_cycles / pmu_counters[] into the
 * dual-issue staging slot. AICPU fills func_id / core_type on commit.
 */
struct PmuRecord {
    uint64_t task_id;                             // Runtime task id
    uint32_t func_id;                             // Kernel function identifier (AICPU-owned)
    CoreType core_type;                           // AIC or AIV (AICPU-owned)
    uint64_t pmu_total_cycles;                    // PMU_CNT_TOTAL (64-bit combined)
    uint32_t pmu_counters[PMU_COUNTER_COUNT_A5];  // PMU_CNT0..CNT9
} __attribute__((aligned(64)));

// =============================================================================
// PmuAicoreRing - Stable AICore→AICPU Staging Ring (per core, never rotated)
// =============================================================================

/**
 * Per-core PMU staging ring written exclusively by AICore.
 *
 * AICore reads PMU MMIO itself (via ld_dev) and stores each task's snapshot
 * in `dual_issue_slots[reg_task_id % PLATFORM_PMU_AICORE_RING_SIZE]`.
 * The ring is allocated once by the host and addressed through
 * `PmuBufferState[block_idx].aicore_ring_ptr` (also published into the
 * `KernelArgs::aicore_pmu_ring_addrs` the AICore kernel entry forwards
 * into `set_aicore_pmu_ring()`); its address is never reassigned, so AICore
 * writes are decoupled from AICPU's PmuBuffer rotation.
 */
struct PmuAicoreRing {
    PmuRecord dual_issue_slots[PLATFORM_PMU_AICORE_RING_SIZE];
} __attribute__((aligned(64)));

// =============================================================================
// PMU Streaming Buffer Structures (mirrors a2a3 pmu_profiling.h)
// =============================================================================

/**
 * Fixed-capacity PMU record buffer.
 * Allocated by Host, pushed into per-core free_queue, rotated by AICPU when
 * full. Owned and written exclusively by AICPU: AICore never touches this
 * memory. AICPU reads the snapshot from PmuAicoreRing::dual_issue_slots and
 * commits into records[count++] on COND FIN.
 */
struct PmuBuffer {
    PmuRecord records[PLATFORM_PMU_RECORDS_PER_BUFFER];
    volatile uint32_t count;
} __attribute__((aligned(64)));

/**
 * SPSC lock-free queue for free PmuBuffer management.
 *
 * Producer: Host (PmuCollector thread) pushes recycled/new buffers.
 * Consumer: Device (AICPU thread) pops buffers when switching.
 *
 * Memory ordering:
 *   Device pop:  rmb() → read tail → read buffer_ptrs[head % COUNT] → rmb() → write head → wmb()
 *   Host push:   write buffer_ptrs[tail % COUNT] → wmb() → write tail → wmb()
 */
struct PmuFreeQueue {
    volatile uint64_t buffer_ptrs[PLATFORM_PMU_SLOT_COUNT];  // 4 * 8 = 32 bytes
    volatile uint32_t head;                                  // Consumer read position (Device increments)
    volatile uint32_t tail;                                  // Producer write position (Host increments)
    uint32_t pad[22];                                        // Pad 40 + 88 -> 128 bytes
} __attribute__((aligned(64)));

static_assert(sizeof(PmuFreeQueue) == 128, "PmuFreeQueue must be 128 bytes");

/**
 * Per-core PMU buffer state.
 *
 * Writers:
 *   free_queue.tail:           Host writes (pushes new/recycled buffers)
 *   free_queue.head:           Device writes (pops buffers)
 *   current_buf_ptr:           Device writes (after pop), Host reads (for collect/drain)
 *   aicore_ring_ptr:           Host writes once at init, AICPU reads
 *   current_buf_seq:           Device writes (monotonic counter)
 *   total_record_count:        Device writes — monotonic count of every task the
 *                              AICPU attempted to record (collected + dropped + mismatch)
 *   dropped_record_count:      Device writes (tasks whose PmuRecord was never handed
 *                              to the host: free_queue empty, ready_queue full,
 *                              no active buffer)
 *   mismatch_record_count:     Device writes — slots where AICore had not yet
 *                              published the expected reg_task_id (hard invariant
 *                              violation, distinct from capacity drops)
 *
 * Host reads dropped / mismatch / total at finalize time to cross-check:
 *   collected_on_host + sum(dropped) + sum(mismatch) == sum(total)
 */
struct PmuBufferState {
    PmuFreeQueue free_queue;                  // SPSC queue of free PmuBuffer addresses
    volatile uint64_t current_buf_ptr;        // Current active PmuBuffer (0 = none)
    volatile uint64_t aicore_ring_ptr;        // Stable AICore staging ring (PmuAicoreRing*)
    volatile uint32_t current_buf_seq;        // Sequence number for ordering
    volatile uint32_t total_record_count;     // Total tasks the AICPU attempted to record
    volatile uint32_t dropped_record_count;   // Tasks whose record was dropped on device
    volatile uint32_t mismatch_record_count;  // Tasks lost to ring/task_id invariant violation
    uint32_t pad[8];                          // Pad to 192 bytes
} __attribute__((aligned(64)));

static_assert(sizeof(PmuBufferState) == 192, "PmuBufferState must be 192 bytes");

/**
 * Ready queue entry.
 * When a PmuBuffer is full, AICPU adds this entry to the thread's ready queue.
 */
struct PmuReadyQueueEntry {
    uint32_t core_index;  // Core index (0 ~ num_cores-1)
    uint32_t pad0;
    uint64_t buffer_ptr;  // Device pointer to the full PmuBuffer
    uint32_t buffer_seq;  // Sequence number for ordering
    uint32_t pad1;
} __attribute__((aligned(32)));

static_assert(sizeof(PmuReadyQueueEntry) == 32, "PmuReadyQueueEntry must be 32 bytes");

/**
 * PMU data fixed header, located at the start of PMU shared memory.
 *
 * Per-thread ready queues (one per AICPU scheduling thread):
 *   Producer: AICPU thread (adds full PmuBuffers)
 *   Consumer: Host PmuCollector thread
 */
struct PmuDataHeader {
    PmuReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][PLATFORM_PMU_READYQUEUE_SIZE];
    volatile uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS];  // Host reads (consumer)
    volatile uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS];  // AICPU writes (producer)
    uint32_t num_cores;
    uint32_t event_type;  // PmuEventType value, written by host at init
    uint32_t pad[2];
} __attribute__((aligned(64)));

// =============================================================================
// Helper Functions
// =============================================================================

inline size_t calc_pmu_data_size(int num_cores) {
    return sizeof(PmuDataHeader) + static_cast<size_t>(num_cores) * sizeof(PmuBufferState);
}

inline PmuDataHeader *get_pmu_header(void *base_ptr) { return reinterpret_cast<PmuDataHeader *>(base_ptr); }

inline PmuBufferState *get_pmu_buffer_state(void *base_ptr, int core_id) {
    return reinterpret_cast<PmuBufferState *>(reinterpret_cast<char *>(base_ptr) + sizeof(PmuDataHeader)) + core_id;
}

#endif  // SRC_A5_PLATFORM_INCLUDE_COMMON_PMU_PROFILING_H_
