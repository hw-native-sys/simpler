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
 * @file platform_config.h
 * @brief Platform-specific configuration and architectural constraints
 *
 * This header defines platform architectural parameters that affect
 * both platform and runtime layers. These configurations are derived from
 * hardware capabilities and platform design decisions.
 *
 * Configuration Hierarchy:
 * - Base: PLATFORM_MAX_BLOCKDIM (platform capacity)
 * - Derived: All other limits calculated from base configuration
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_COMMON_PLATFORM_CONFIG_H_
#define SRC_A2A3_PLATFORM_INCLUDE_COMMON_PLATFORM_CONFIG_H_

#include <cstdint>

// =============================================================================
// Base Platform Configuration
// =============================================================================

/**
 * Maximum block dimension supported by platform
 * This is the fundamental platform capacity constraint.
 */
constexpr int PLATFORM_MAX_BLOCKDIM = 24;

/**
 * Core composition per block dimension
 * Current architecture: 1 block = 1 AIC cube + 2 AIV cubes
 */
constexpr int PLATFORM_CORES_PER_BLOCKDIM = 3;
constexpr int PLATFORM_AIC_CORES_PER_BLOCKDIM = 1;
constexpr int PLATFORM_AIV_CORES_PER_BLOCKDIM = 2;

/**
 * Maximum AICPU scheduling threads
 * Determines parallelism level of the AICPU task scheduler.
 */
constexpr int PLATFORM_MAX_AICPU_THREADS = 4;

/**
 * Maximum AICPU launch threads (physical)
 * Upper bound for the number of AICPU threads that can be launched by Host.
 * Can be larger than PLATFORM_MAX_AICPU_THREADS to allow threads to be dropped
 * from scheduling while still participating in affinity (e.g. 6 launch, 4 active).
 */
constexpr int PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH = 6;

/**
 * Default AICore op execution timeout (microseconds).
 * Passed to aclrtSetOpExecuteTimeOutV2 so that STARS actively monitors
 * AICore task execution and kills ops that exceed this threshold.
 * Overridden at runtime by PTO2_OP_EXECUTE_TIMEOUT_US when that env var
 * is valid.
 */
constexpr uint64_t PLATFORM_OP_EXECUTE_TIMEOUT_US = 45000000;  // 45s

/**
 * Default onboard AICPU scheduler no-progress timeout (milliseconds).
 * Shared with host-side timeout ordering validation.
 */
constexpr int32_t PLATFORM_ONBOARD_SCHEDULER_TIMEOUT_MS = 10000;

/**
 * Default host-side stream synchronization timeout (milliseconds).
 * Passed to aclrtSynchronizeStreamWithTimeout to detect stream sync hangs.
 * Must be longer than PLATFORM_OP_EXECUTE_TIMEOUT_US so the host waits for
 * STARS to reap the timed-out op and surface the error, rather than giving
 * up first. Overridden at runtime by PTO2_STREAM_SYNC_TIMEOUT_MS when that
 * env var is valid.
 */
constexpr int PLATFORM_STREAM_SYNC_TIMEOUT_MS = 50000;  // 50s (> op-exec 45s)

// =============================================================================
// Derived Platform Limits
// =============================================================================

/**
 * Maximum cores per AICPU thread
 *
 * When running with 1 AICPU thread and MAX_BLOCKDIM blocks,
 * one thread must manage all cores:
 * - MAX_AIC_PER_THREAD = MAX_BLOCKDIM * AIC_CORES_PER_BLOCKDIM = 24 * 1 = 24
 * - MAX_AIV_PER_THREAD = MAX_BLOCKDIM * AIV_CORES_PER_BLOCKDIM = 24 * 2 = 48
 */
constexpr int PLATFORM_MAX_AIC_PER_THREAD = PLATFORM_MAX_BLOCKDIM * PLATFORM_AIC_CORES_PER_BLOCKDIM;  // 24

constexpr int PLATFORM_MAX_AIV_PER_THREAD = PLATFORM_MAX_BLOCKDIM * PLATFORM_AIV_CORES_PER_BLOCKDIM;  // 48

constexpr int PLATFORM_MAX_CORES_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD + PLATFORM_MAX_AIV_PER_THREAD;  // 72

// =============================================================================
// Performance Profiling Configuration
// =============================================================================

/**
 * Maximum number of cores that can be profiled simultaneously
 * Calculated as: MAX_BLOCKDIM * CORES_PER_BLOCKDIM = 24 * 3 = 72
 */
constexpr int PLATFORM_MAX_CORES = PLATFORM_MAX_BLOCKDIM * PLATFORM_CORES_PER_BLOCKDIM;  // 72

/**
 * Performance buffer capacity per buffer
 * Number of L2SwimlaneAicpuTaskRecord entries per dynamically allocated L2SwimlaneAicpuTaskBuffer
 */
constexpr int PLATFORM_PROF_BUFFER_SIZE = 1000;

/**
 * Number of buffer slots per core/thread for dynamic profiling
 * Host dynamically allocates buffers and writes addresses into these slots.
 * Device reads slot addresses when switching buffers.
 * Using slots: provides full pipeline depth for buffer recycling.
 * No runtime rtMalloc — all buffers are pre-allocated and recycled in a closed loop.
 */
constexpr int PLATFORM_PROF_SLOT_COUNT = 4;

/**
 * L2SwimlaneAicpuTaskBuffer pre-allocation count per AICore.
 * 1 goes into the free_queue at init, the rest into the recycled pool.
 */
constexpr int PLATFORM_PROF_BUFFERS_PER_CORE = 8;

/**
 * L2SwimlaneAicoreTaskBuffer pre-allocation count per AICore (AICore-as-producer pool).
 * 1 goes into the free_queue at init, the rest into the recycled pool.
 * Mirrors PLATFORM_PROF_BUFFERS_PER_CORE in role; smaller because AICore records
 * are slim (32 B each) and the buffer is also smaller per the rotation design.
 */
constexpr int PLATFORM_AICORE_BUFFERS_PER_CORE = 4;

/**
 * Host preallocation count per AICPU thread for the two phase pools, split per
 * kind (sched vs orch) because their throughput is asymmetric — a single shared
 * value over-provisions the lighter one. 1 buffer seeds the free_queue at init,
 * the rest the recycled pool.
 *
 * Floor for both: SLOT_COUNT(4) + 1 = 5 (free_queue fillable + 1 active buffer).
 * Pure host preallocation — zero ABI (the device-visible ready_queue is decoupled
 * below, held at the original 16). Sizing rationale + measurements:
 * docs/dfx/dfx-buffer-capacity-audit.md.
 */
constexpr int PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD = 6;
constexpr int PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD = 8;

/**
 * Ready queue capacity for performance data collection.
 * Queue holds ReadyQueueEntry structs for buffers ready to be read by Host.
 * Sized to match pre-allocation total across all cores and threads, summed
 * over the four buffer kinds (L2SwimlaneAicpuTaskBuffer per core,
 * L2SwimlaneAicpuSchedPhaseBuffer + L2SwimlaneAicpuOrchPhaseBuffer per
 * AICPU thread, L2SwimlaneAicoreTaskBuffer per core).
 *
 * The phase term is sized to the ORIGINAL per-thread count (16), decoupled from
 * PLATFORM_PROF_{SCHED,ORCH}_BUFFERS_PER_THREAD, on purpose: this array sizes the
 * device-visible L2SwimlaneDataHeader::queues[], so shrinking the host
 * preallocation must NOT shrink it (that would change the shm layout = ABI).
 * 992 is far above the observed peak backlog (~40); the slack is harmless.
 */
constexpr int PLATFORM_PROF_READYQUEUE_BUFFERS_PER_THREAD = 16;
constexpr int PLATFORM_PROF_READYQUEUE_SIZE =
    PLATFORM_MAX_CORES * PLATFORM_PROF_BUFFERS_PER_CORE +
    2 * PLATFORM_MAX_AICPU_THREADS * PLATFORM_PROF_READYQUEUE_BUFFERS_PER_THREAD +
    PLATFORM_MAX_CORES * PLATFORM_AICORE_BUFFERS_PER_CORE;

/**
 * System counter frequency (get_sys_cnt)
 * Used to convert timestamps to microseconds.
 */
constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 50000000;  // 50 MHz

/**
 * Timeout duration for performance data collection (seconds)
 */
constexpr int PLATFORM_PROF_TIMEOUT_SECONDS = 30;

inline double cycles_to_us(uint64_t cycles) {
    return (static_cast<double>(cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
}

// Profiling-related runtime flags shared through AICPU-AICore handshake.
// "Profiling" is the umbrella; each bit is a parallel diagnostics sub-feature.
#define PROFILING_FLAG_NONE 0u
#define PROFILING_FLAG_DUMP_TENSOR (1u << 0)
#define PROFILING_FLAG_L2_SWIMLANE (1u << 1)
#define PROFILING_FLAG_PMU (1u << 2)
#define PROFILING_FLAG_DEP_GEN (1u << 3)
#define PROFILING_FLAG_SCOPE_STATS (1u << 4)
#define GET_PROFILING_FLAG(flags, bit) ((((uint32_t)(flags)) & ((uint32_t)(bit))) != 0u)
#define SET_PROFILING_FLAG(flags, bit) ((flags) |= (uint32_t)(bit))
#define CLEAR_PROFILING_FLAG(flags, bit) ((flags) &= ~((uint32_t)(bit)))

// =============================================================================
// Tensor Dump Configuration
// =============================================================================

/**
 * Number of TensorDumpRecord entries per DumpMetaBuffer.
 * Each record is 128 bytes, so one buffer = RECORDS * 128 bytes.
 */
constexpr int PLATFORM_DUMP_RECORDS_PER_BUFFER = 256;

/**
 * Pre-allocated DumpMetaBuffer count per AICPU scheduling thread.
 * Pushed into the per-thread SPSC free_queue at init.
 */
constexpr int PLATFORM_DUMP_BUFFERS_PER_THREAD = 8;

/**
 * SPSC free_queue slot count for dump metadata buffers.
 */
constexpr int PLATFORM_DUMP_SLOT_COUNT = 4;

/**
 * Expected average tensor size in bytes.
 * Used together with BUFFERS_PER_THREAD and RECORDS_PER_BUFFER to compute
 * per-thread arena size:
 *   arena = BUFFERS_PER_THREAD * RECORDS_PER_BUFFER * AVG_TENSOR_BYTES
 * Default: 4 * 256 * 65536 = 64 MB per thread.
 */
constexpr uint64_t PLATFORM_DUMP_AVG_TENSOR_BYTES = 65536;

/**
 * Maximum tensor dimensions (matches MAX_TENSOR_DIMS).
 */
constexpr int PLATFORM_DUMP_MAX_DIMS = 5;

/**
 * Ready queue capacity for dump data.
 * Sized to hold all dump buffers across all threads.
 */
constexpr int PLATFORM_DUMP_READYQUEUE_SIZE = PLATFORM_MAX_AICPU_THREADS * PLATFORM_DUMP_BUFFERS_PER_THREAD * 2;

/**
 * Idle timeout duration for tensor dump collection (seconds)
 */
constexpr int PLATFORM_DUMP_TIMEOUT_SECONDS = 30;

/**
 * Dump-args mask pool dimensions. The pool is keyed by (ring_id, slot) packed
 * from a PTO2 task_id, so it must span the largest ring depth and task window
 * any runtime built against this platform can use. The dump infra is shared by
 * every runtime (device-orch tensormap_and_ringbuffer at ring depth 4 and the
 * single-ring host-orch host_build_graph), so these are sized to the maximum
 * rather than coupled to one runtime's pto_runtime2_types.h — a runtime that
 * lowers its own PTO2_MAX_RING_DEPTH must not shrink the pool other runtimes
 * rely on (see set_dump_args_task_mask's ring_id bound check).
 */
constexpr uint32_t PLATFORM_DUMP_MASK_POOL_MAX_RINGS = 4;
constexpr uint32_t PLATFORM_DUMP_MASK_POOL_MAX_SLOTS = 16384;

// =============================================================================
// PMU Profiling Configuration
// =============================================================================

/**
 * Number of PmuRecord entries per PmuBuffer.
 */
constexpr int PLATFORM_PMU_RECORDS_PER_BUFFER = 512;

/**
 * SPSC free_queue slot count for PMU buffers.
 */
constexpr int PLATFORM_PMU_SLOT_COUNT = 4;

/**
 * Pre-allocated PmuBuffer count per AICore.
 * Sized at 2 = 2× the measured peak in-flight (Little's Law L≈1: PMU records
 * are produced at the slow fixed sampling rate, decoupled from submit, and the
 * 64 B record drains instantly, so the host never lets more than 1 buffer be
 * borrowed). Was 4 (a 4× over-provision). See
 * docs/dfx/dfx-buffer-capacity-audit.md.
 */
constexpr int PLATFORM_PMU_BUFFERS_PER_CORE = 2;

/**
 * Ready queue capacity for PMU data collection.
 * Indexed by AICPU thread; each entry names the core and buffer pointer.
 */
constexpr int PLATFORM_PMU_READYQUEUE_SIZE = PLATFORM_MAX_CORES * PLATFORM_PMU_BUFFERS_PER_CORE;

/**
 * Idle timeout duration for PMU collection (seconds)
 */
constexpr int PLATFORM_PMU_TIMEOUT_SECONDS = 30;

// =============================================================================
// dep_gen (SubmitTrace) Configuration
// =============================================================================

/**
 * Number of DepGenRecord entries per DepGenBuffer.
 * Each DepGenRecord is 4672 B (16 Tensor blobs + small header), so a buffer
 * of 1024 records is ~4.6 MB; draining one copies that over SVM.
 * Guaranteed one-shot capacity = BUFFERS_PER_INSTANCE × RECORDS_PER_BUFFER
 * records — the size of a back-to-back submit flood the pool absorbs even if
 * the host never drains. At 4×1024 that is 4096 submits, which aligns
 * dep_gen's in-flight record count with the scope_stats / l2 AicoreTask pools
 * (also 4096) per the #977 cross-subsystem review. By in-flight BYTES dep_gen
 * cannot align (its 4672 B record is 70–90× the others); that tension is
 * inherent — see docs/dfx/dfx-buffer-capacity-audit.md.
 * History: original 32 (commit 77ef83c0) → #977 commit 3f977e54 overshot to
 * 2048 → 1024 here (#977 Primary's actual proposal; the 2× was no gain).
 * Whether a larger workload drops depends on submit ARRIVAL RATE, not total
 * count: dep_gen records are produced at submit_task time on the AICPU, so a
 * dependency-paced workload (real decoder, submits gated by compute) drains
 * faster than it fills and never drops, while an independent-submit flood
 * microbenchmark (paged_attention Case1, 65536 back-to-back submits) outruns
 * host drain. Onboard a2a3: that flood drops ~24576/65536, and the drop is
 * rate-bound — doubling RECORDS, BUFFERS, or SLOT_COUNT each left it unchanged
 * (~24576). So buffer sizing is NOT the lever for floods; 4096 in-flight is
 * sized for the largest realistic single-scope decoder burst (a 4096-submit
 * burst measured max_ready_depth=1, ample headroom).
 * dep_gen is opt-in (--enable-dep-gen).
 */
constexpr int PLATFORM_DEP_GEN_RECORDS_PER_BUFFER = 1024;

/**
 * SPSC free_queue slot count for dep_gen buffers (Host→Device hand-off depth).
 * Caps how many empty buffers the device can pre-borrow before it starves;
 * top_up_free_queue refills the ring only up to SLOT_COUNT. Tunable up to 15
 * within the fixed 128 B DepGenFreeQueue layout (see dep_gen.h), but onboard
 * tests showed raising it does not reduce flood drops (the drop is rate-bound,
 * not capacity-bound — see RECORDS_PER_BUFFER above), so it stays at the
 * default 4.
 */
constexpr int PLATFORM_DEP_GEN_SLOT_COUNT = 4;

/**
 * Pre-allocated DepGenBuffer count per orchestrator instance.
 * 4 buffers × 1024 records = 4096 in-flight records (~19 MB per instance).
 * Must be ≥ SLOT_COUNT to fill the free_queue ring; any beyond SLOT_COUNT seed
 * the host recycled pool. Raising it does not cut flood drops on its own (the
 * device is bounded by SLOT_COUNT, and the drop is rate-bound anyway) — see the
 * RECORDS_PER_BUFFER comment. dep_gen is opt-in (--enable-dep-gen).
 */
constexpr int PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE = 4;

/**
 * Ready queue capacity for dep_gen (per AICPU thread). dep_gen is single-
 * instance so headroom over BUFFERS_PER_INSTANCE × num_instances is small.
 */
constexpr int PLATFORM_DEP_GEN_READYQUEUE_SIZE = PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE * 2;

/**
 * Idle timeout duration for dep_gen collection (seconds).
 */
constexpr int PLATFORM_DEP_GEN_TIMEOUT_SECONDS = 30;

// =============================================================================
// scope_stats Configuration
// =============================================================================

/**
 * Number of ScopeStatsRecord entries per ScopeStatsBuffer.
 * Each record is 52 B. Larger buffers = fewer device-side switch_buffer
 * hand-offs (barriers, helps Orch) and fewer/larger device→host drain memcpys.
 * 512 records ≈ 27 KB/buffer (×8 buffers ≈ 213 KB).
 */
constexpr int PLATFORM_SCOPE_STATS_RECORDS_PER_BUFFER = 512;

/**
 * SPSC free_queue slot count for scope_stats buffers (Host→Device hand-off depth).
 */
constexpr int PLATFORM_SCOPE_STATS_SLOT_COUNT = 8;

/**
 * Pre-allocated ScopeStatsBuffer count per orchestrator instance.
 * Sized at 4 = 4× the measured peak in-flight (Little's Law L≈1: scope_stats
 * records are produced per scope enter/exit, decoupled from submit volume, and
 * the 52 B record drains instantly, so the host never lets more than 1 buffer
 * be borrowed). Was 8 (an 8× over-provision). See
 * docs/dfx/dfx-buffer-capacity-audit.md.
 */
constexpr int PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE = 4;

/**
 * Ready queue capacity for scope_stats (per AICPU thread). scope_stats is
 * single-instance so headroom over BUFFERS_PER_INSTANCE is small.
 */
constexpr int PLATFORM_SCOPE_STATS_READYQUEUE_SIZE = PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE * 2;

/**
 * Idle timeout duration for scope_stats collection (seconds).
 */
constexpr int PLATFORM_SCOPE_STATS_TIMEOUT_SECONDS = 30;

// =============================================================================
// Register Communication Configuration
// =============================================================================

// Register offsets for AICore SPR access
constexpr uint32_t REG_SPR_DATA_MAIN_BASE_OFFSET = 0xA0;  // Task dispatch (AICPU→AICore)
constexpr uint32_t REG_SPR_COND_OFFSET = 0x4C8;           // Status (AICore→AICPU): 0=IDLE, 1=BUSY
constexpr uint32_t REG_SPR_FAST_PATH_ENABLE_OFFSET = 0x18;
constexpr uint32_t REG_SPR_CTRL_OFFSET = 0x0;  // AICore internal CTRL SPR (bit0 = PMU enable)

// Fast path control values
constexpr uint32_t REG_SPR_FAST_PATH_OPEN = 0xE;
constexpr uint32_t REG_SPR_FAST_PATH_CLOSE = 0xF;

// Exit signal for AICore shutdown
constexpr uint32_t AICORE_EXIT_SIGNAL = 0x7FFFFFF0;

// Physical core ID mask for get_coreid()
constexpr uint32_t AICORE_COREID_MASK = 0x0FFF;

// PMU MMIO register offsets (DAV_2201 / a2a3). Accessed by AICPU through
// the per-core register block. AICore does not touch these — it only toggles
// its internal CTRL SPR (REG_SPR_CTRL_OFFSET) for per-task counter gating.
constexpr uint32_t REG_MMIO_PMU_CTRL_0_OFFSET = 0x200;      // PMU framework enable (GLB_PMU_EN | USER | SAMPLE)
constexpr uint32_t REG_MMIO_PMU_CNT0_OFFSET = 0x210;        // Event counter 0
constexpr uint32_t REG_MMIO_PMU_CNT1_OFFSET = 0x218;        // Event counter 1
constexpr uint32_t REG_MMIO_PMU_CNT2_OFFSET = 0x220;        // Event counter 2
constexpr uint32_t REG_MMIO_PMU_CNT3_OFFSET = 0x228;        // Event counter 3
constexpr uint32_t REG_MMIO_PMU_CNT4_OFFSET = 0x230;        // Event counter 4
constexpr uint32_t REG_MMIO_PMU_CNT5_OFFSET = 0x238;        // Event counter 5
constexpr uint32_t REG_MMIO_PMU_CNT6_OFFSET = 0x240;        // Event counter 6
constexpr uint32_t REG_MMIO_PMU_CNT7_OFFSET = 0x248;        // Event counter 7
constexpr uint32_t REG_MMIO_PMU_CNT_TOTAL0_OFFSET = 0x250;  // Total cycle counter, low 32 bits
constexpr uint32_t REG_MMIO_PMU_CNT_TOTAL1_OFFSET = 0x254;  // Total cycle counter, high 32 bits
constexpr uint32_t REG_MMIO_PMU_START_CYC0_OFFSET = 0x2A0;  // Counting-range start cycle, low 32 bits
constexpr uint32_t REG_MMIO_PMU_START_CYC1_OFFSET = 0x2A4;  // Counting-range start cycle, high 32 bits
constexpr uint32_t REG_MMIO_PMU_STOP_CYC0_OFFSET = 0x2A8;   // Counting-range stop cycle, low 32 bits
constexpr uint32_t REG_MMIO_PMU_STOP_CYC1_OFFSET = 0x2AC;   // Counting-range stop cycle, high 32 bits
constexpr uint32_t REG_MMIO_PMU_CNT0_IDX_OFFSET = 0x1280;   // Event selector for CNT0
constexpr uint32_t REG_MMIO_PMU_CNT1_IDX_OFFSET = 0x1284;   // Event selector for CNT1
constexpr uint32_t REG_MMIO_PMU_CNT2_IDX_OFFSET = 0x1288;   // Event selector for CNT2
constexpr uint32_t REG_MMIO_PMU_CNT3_IDX_OFFSET = 0x128C;   // Event selector for CNT3
constexpr uint32_t REG_MMIO_PMU_CNT4_IDX_OFFSET = 0x1290;   // Event selector for CNT4
constexpr uint32_t REG_MMIO_PMU_CNT5_IDX_OFFSET = 0x1294;   // Event selector for CNT5
constexpr uint32_t REG_MMIO_PMU_CNT6_IDX_OFFSET = 0x1298;   // Event selector for CNT6
constexpr uint32_t REG_MMIO_PMU_CNT7_IDX_OFFSET = 0x129C;   // Event selector for CNT7

// PMU_CTRL_0 enable value: GLB_PMU_EN | (USER_PMU_MODE_EN << 1) | (SAMPLE_PMU_MODE_EN << 2)
constexpr uint32_t REG_MMIO_PMU_CTRL_0_ENABLE_VAL = 0x7;

/**
 * Register identifier for unified read_reg/write_reg interface.
 *
 * The PMU counter slots (PMU_CNT0..PMU_CNT7) and event selector slots
 * (PMU_CNT0_IDX..PMU_CNT7_IDX) are assigned contiguous values so that
 * reg_index(base, i) can index into them as arrays. Keep these runs
 * contiguous — static_asserts below enforce this.
 */
enum class RegId : uint8_t {
    DATA_MAIN_BASE = 0,    // Task dispatch (AICPU→AICore)
    COND = 1,              // Status (AICore→AICPU)
    FAST_PATH_ENABLE = 2,  // Fast path control
    CTRL = 3,              // AICore internal CTRL SPR (PMU enable, etc.) — AICore-only

    // PMU framework (AICPU-only; AICore does not access these)
    PMU_CTRL_0 = 4,

    // PMU counters (8 contiguous slots)
    PMU_CNT0 = 5,
    PMU_CNT1 = 6,
    PMU_CNT2 = 7,
    PMU_CNT3 = 8,
    PMU_CNT4 = 9,
    PMU_CNT5 = 10,
    PMU_CNT6 = 11,
    PMU_CNT7 = 12,

    // PMU total cycle counter (64-bit split across two 32-bit regs)
    PMU_CNT_TOTAL0 = 13,
    PMU_CNT_TOTAL1 = 14,

    // PMU counting-range start/stop cycle bounds
    PMU_START_CYC0 = 15,
    PMU_START_CYC1 = 16,
    PMU_STOP_CYC0 = 17,
    PMU_STOP_CYC1 = 18,

    // PMU event selectors (8 contiguous slots, parallel to PMU_CNT0..CNT7)
    PMU_CNT0_IDX = 19,
    PMU_CNT1_IDX = 20,
    PMU_CNT2_IDX = 21,
    PMU_CNT3_IDX = 22,
    PMU_CNT4_IDX = 23,
    PMU_CNT5_IDX = 24,
    PMU_CNT6_IDX = 25,
    PMU_CNT7_IDX = 26,
};

static_assert(
    static_cast<int>(RegId::PMU_CNT7) - static_cast<int>(RegId::PMU_CNT0) == 7,
    "PMU_CNT0..PMU_CNT7 must be contiguous for reg_index()"
);
static_assert(
    static_cast<int>(RegId::PMU_CNT7_IDX) - static_cast<int>(RegId::PMU_CNT0_IDX) == 7,
    "PMU_CNT0_IDX..PMU_CNT7_IDX must be contiguous for reg_index()"
);

/**
 * Map RegId to hardware register offset
 */
constexpr uint32_t reg_offset(RegId reg) {
    switch (reg) {
    case RegId::DATA_MAIN_BASE:
        return REG_SPR_DATA_MAIN_BASE_OFFSET;
    case RegId::COND:
        return REG_SPR_COND_OFFSET;
    case RegId::FAST_PATH_ENABLE:
        return REG_SPR_FAST_PATH_ENABLE_OFFSET;
    case RegId::CTRL:
        return REG_SPR_CTRL_OFFSET;
    case RegId::PMU_CTRL_0:
        return REG_MMIO_PMU_CTRL_0_OFFSET;
    case RegId::PMU_CNT0:
        return REG_MMIO_PMU_CNT0_OFFSET;
    case RegId::PMU_CNT1:
        return REG_MMIO_PMU_CNT1_OFFSET;
    case RegId::PMU_CNT2:
        return REG_MMIO_PMU_CNT2_OFFSET;
    case RegId::PMU_CNT3:
        return REG_MMIO_PMU_CNT3_OFFSET;
    case RegId::PMU_CNT4:
        return REG_MMIO_PMU_CNT4_OFFSET;
    case RegId::PMU_CNT5:
        return REG_MMIO_PMU_CNT5_OFFSET;
    case RegId::PMU_CNT6:
        return REG_MMIO_PMU_CNT6_OFFSET;
    case RegId::PMU_CNT7:
        return REG_MMIO_PMU_CNT7_OFFSET;
    case RegId::PMU_CNT_TOTAL0:
        return REG_MMIO_PMU_CNT_TOTAL0_OFFSET;
    case RegId::PMU_CNT_TOTAL1:
        return REG_MMIO_PMU_CNT_TOTAL1_OFFSET;
    case RegId::PMU_START_CYC0:
        return REG_MMIO_PMU_START_CYC0_OFFSET;
    case RegId::PMU_START_CYC1:
        return REG_MMIO_PMU_START_CYC1_OFFSET;
    case RegId::PMU_STOP_CYC0:
        return REG_MMIO_PMU_STOP_CYC0_OFFSET;
    case RegId::PMU_STOP_CYC1:
        return REG_MMIO_PMU_STOP_CYC1_OFFSET;
    case RegId::PMU_CNT0_IDX:
        return REG_MMIO_PMU_CNT0_IDX_OFFSET;
    case RegId::PMU_CNT1_IDX:
        return REG_MMIO_PMU_CNT1_IDX_OFFSET;
    case RegId::PMU_CNT2_IDX:
        return REG_MMIO_PMU_CNT2_IDX_OFFSET;
    case RegId::PMU_CNT3_IDX:
        return REG_MMIO_PMU_CNT3_IDX_OFFSET;
    case RegId::PMU_CNT4_IDX:
        return REG_MMIO_PMU_CNT4_IDX_OFFSET;
    case RegId::PMU_CNT5_IDX:
        return REG_MMIO_PMU_CNT5_IDX_OFFSET;
    case RegId::PMU_CNT6_IDX:
        return REG_MMIO_PMU_CNT6_IDX_OFFSET;
    case RegId::PMU_CNT7_IDX:
        return REG_MMIO_PMU_CNT7_IDX_OFFSET;
    }
    return 0;  // unreachable: all RegId cases handled above
}

/**
 * Index into a contiguous RegId run (e.g. reg_index(PMU_CNT0, 3) == PMU_CNT3).
 * Caller is responsible for keeping `i` within the run's length.
 */
constexpr RegId reg_index(RegId base, int i) { return static_cast<RegId>(static_cast<uint8_t>(base) + i); }

// Size of simulated register block per core (covers largest offset + 4 bytes).
// Bumped from 0x500 to 0x1400 to include DAV_2201 PMU registers:
//   - CTRL_0 at 0x200, CNT/CNT_TOTAL at 0x210-0x254
//   - START/STOP_CYC at 0x2A0-0x2AC
//   - CNT_IDX at 0x1280-0x129C (highest offset + 4 = 0x12A0)
// 0x1400 rounds up to a 64-byte boundary with headroom.
constexpr uint32_t SIM_REG_BLOCK_SIZE = 0x1400;

// =============================================================================
// Hardware Configuration Constants
// =============================================================================

/**
 * Number of sub-cores per AICore
 * Hardware architecture: 1 AICore = 1 AIC + 2 AIV sub-cores
 */
constexpr uint32_t PLATFORM_SUB_CORES_PER_AICORE = PLATFORM_CORES_PER_BLOCKDIM;

/**
 * Maximum physical AICore count for DAV 2201 chip
 */
namespace DAV_2201 {
constexpr uint32_t PLATFORM_MAX_PHYSICAL_CORES = 25;
}

// =============================================================================
// ACK/FIN Dual-State Register Protocol
// =============================================================================

/**
 * AICPU-AICore task handshake protocol via COND register
 *
 * Register format: [bit 31: state | low 31 bits: task_id]
 * State: ACK (0) = task received, FIN (1) = task completed
 */

#define TASK_ID_MASK 0x7FFFFFFFU
#define TASK_STATE_MASK 0x80000000U

enum : uint8_t { TASK_ACK_STATE = 0, TASK_FIN_STATE = 1 };

#define EXTRACT_TASK_ID(regval) (static_cast<int>((regval) & TASK_ID_MASK))
#define EXTRACT_TASK_STATE(regval) (static_cast<int>(((regval) & TASK_STATE_MASK) >> 31))
#define MAKE_ACK_VALUE(task_id) (static_cast<uint64_t>((task_id) & TASK_ID_MASK))
#define MAKE_FIN_VALUE(task_id) (static_cast<uint64_t>(((task_id) & TASK_ID_MASK) | TASK_STATE_MASK))

// These values are RESERVED and must never be used as real task IDs.
// Valid task IDs: 0 to 0x7FFFFFEF (2147483631)
enum : uint32_t {
    AICORE_IDLE_TASK_ID = 0x7FFFFFFFU,
    AICORE_EXIT_TASK_ID = 0x7FFFFFFEU,
    AICPU_IDLE_TASK_ID = 0x7FFFFFFDU,
};
#define AICORE_IDLE_VALUE MAKE_FIN_VALUE(AICORE_IDLE_TASK_ID)

#define AICORE_EXITED_VALUE MAKE_FIN_VALUE(AICORE_EXIT_TASK_ID)

// =============================================================================
// Task State Constants
// =============================================================================

/**
 * Invalid task ID sentinel value
 * Used to indicate that pending_task_ids_ or running_task_ids_ slot is empty.
 */
constexpr int AICPU_TASK_INVALID = -1;

#endif  // SRC_A2A3_PLATFORM_INCLUDE_COMMON_PLATFORM_CONFIG_H_
