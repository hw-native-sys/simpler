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
 * a5 has no shared-memory transport (halHostRegister is not supported on
 * DAV_3510). The PMU buffer is a single per-core PmuBuffer allocated on
 * device at init time, written to by AICPU during task execution, and
 * drained to the host via rtMemcpy after stream sync. This mirrors the
 * memcpy pattern already used by PerformanceCollector and TensorDumpCollector.
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
// PMU Record + Buffer
// =============================================================================

/**
 * Per-task PMU snapshot written by AICPU after each AICore task FIN.
 *
 * AICore writes task_id / pmu_total_cycles / pmu_counters[] into the
 * dual-issue staging slot. AICPU fills func_id / core_type on commit —
 * those are consumer-owned and AICore never touches them.
 *
 * Thread ownership is tracked per-core in PmuBufferState::owning_thread_id,
 * not per-record — it's a buffer-level attribute (same core is always
 * driven by the same AICPU scheduler thread). Mirrors a2a3's per-queue
 * thread association.
 */
struct PmuRecord {
    uint64_t task_id;                             // Runtime task id
    uint32_t func_id;                             // Kernel function identifier (AICPU-owned)
    CoreType core_type;                           // AIC or AIV (AICPU-owned)
    uint64_t pmu_total_cycles;                    // PMU_CNT_TOTAL (64-bit combined)
    uint32_t pmu_counters[PMU_COUNTER_COUNT_A5];  // PMU_CNT0..CNT9
} __attribute__((aligned(64)));

/**
 * Fixed-capacity per-core PMU record buffer.
 *
 * Layout:
 *   [0,64)              — header: count + padding (host reads this 64-byte
 *                         header first to learn how many records to drain)
 *   [64, 64+2*PmuRecord) — dual_issue_slots: AICore writes a PmuRecord
 *                         here after each task, indexed by task_id & 1
 *   [...)                — records: AICPU commits finished PmuRecords here
 *
 * The two dual_issue_slots exist because AICore's dual-issue dispatch
 * can have up to two tasks in flight per core (task N+1 can begin
 * before AICPU has committed the record for N). Parity `task_id & 1`
 * picks the slot so N and N+1 never collide — this is exactly the
 * reason a2a3 perf uses `wip[2]` on PerfBuffer.
 *
 * dropped_record_count / total_record_count live on PmuBufferState (mirrors
 * a2a3) so cross-checking is centralized in a single per-core state region.
 *
 * Written by AICore (dual_issue_slots) + AICPU (records/count); drained
 * to host via rtMemcpy after stream sync.
 */
struct PmuBuffer {
    // Header (first 64 bytes) — host copies this alone first to learn count.
    volatile uint32_t count;  // Number of valid records
    uint32_t pad[15];         // Pad header to 64 bytes

    // Dual-issue staging slots — AICore writes a PmuRecord here after
    // each task, then AICPU copies it into records[count] and fills
    // func_id / core_type on FIN. Index = task_id & 1.
    PmuRecord dual_issue_slots[2];

    // Records (flexible-size, up to PLATFORM_PMU_RECORDS_PER_BUFFER)
    PmuRecord records[PLATFORM_PMU_RECORDS_PER_BUFFER];
} __attribute__((aligned(64)));

static_assert(
    offsetof(PmuBuffer, dual_issue_slots) == 64, "PmuBuffer header must be exactly 64 bytes before dual_issue_slots"
);
static_assert(
    offsetof(PmuBuffer, records) == 64 + 2 * sizeof(PmuRecord),
    "PmuBuffer records must follow header + 2 dual_issue_slots"
);

/**
 * Per-core PMU buffer state. Mirrors a2a3 PmuBufferState (minus the
 * free_queue / current_buf_ptr fields, which a5 doesn't need because it
 * uses a single pre-allocated PmuBuffer per core).
 *
 * Writers (AICPU only):
 *   owning_thread_id:     AICPU scheduler thread that drives this core.
 *                         Stamped on first PMU commit and stable thereafter
 *                         (a5 binds core→thread at scheduler init).
 *   dropped_record_count: Tasks whose record was dropped on device
 *                         (PmuBuffer full).
 *   total_record_count:   Monotonic count of every task the AICPU attempted
 *                         to record (success + dropped + slot-mismatch).
 *
 * Host reads dropped / total at finalize time to cross-check:
 *   collected_on_host + dropped == total
 * Any shortfall is silent slot-mismatch loss (AICore hadn't yet published
 * its dual-issue slot when AICPU tried to commit).
 */
struct PmuBufferState {
    volatile uint32_t owning_thread_id;
    volatile uint32_t dropped_record_count;
    volatile uint32_t total_record_count;
    uint32_t pad[13];
} __attribute__((aligned(64)));

static_assert(sizeof(PmuBufferState) == 64, "PmuBufferState must be 64 bytes");

/**
 * PMU setup header, allocated once on device and published into
 * kernel_args.pmu_data_base.
 *
 * The on-device shared region layout is:
 *   [PmuSetupHeader] [PmuBufferState[num_cores]]
 *
 * a5 uses one rtMemcpy at finalize to bring all PmuBufferState back to
 * host (mirrors a2a3's get_pmu_buffer_state offset math).
 */
struct PmuSetupHeader {
    uint32_t num_cores;
    uint32_t event_type;  // PmuEventType value
    uint32_t pad[14];     // Pad header to 64 bytes
    // Device pointers to the per-core PmuBuffer, one entry per core.
    // AICPU reads buffer_ptrs[core_id] to get its PmuBuffer.
    uint64_t buffer_ptrs[PLATFORM_MAX_CORES];
} __attribute__((aligned(64)));

static_assert(offsetof(PmuSetupHeader, buffer_ptrs) == 64, "PmuSetupHeader header must be exactly 64 bytes");

// =============================================================================
// Helpers
// =============================================================================

constexpr size_t PMU_BUFFER_HEADER_BYTES = 64;
constexpr size_t PMU_SETUP_HEADER_BYTES = 64;

/**
 * Total bytes of the [PmuSetupHeader][PmuBufferState[num_cores]] shared region.
 */
inline size_t calc_pmu_setup_size(int num_cores) {
    return sizeof(PmuSetupHeader) + static_cast<size_t>(num_cores) * sizeof(PmuBufferState);
}

inline PmuSetupHeader *get_pmu_setup_header(void *base_ptr) { return reinterpret_cast<PmuSetupHeader *>(base_ptr); }

inline PmuBufferState *get_pmu_buffer_state(void *base_ptr, int core_id) {
    return reinterpret_cast<PmuBufferState *>(reinterpret_cast<char *>(base_ptr) + sizeof(PmuSetupHeader)) + core_id;
}

#endif  // SRC_A5_PLATFORM_INCLUDE_COMMON_PMU_PROFILING_H_
