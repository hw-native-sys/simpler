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

#ifndef SRC_A5_PLATFORM_INCLUDE_COMMON_PLATFORM_CONFIG_H_
#define SRC_A5_PLATFORM_INCLUDE_COMMON_PLATFORM_CONFIG_H_

#include <cstdint>

// =============================================================================
// Base Platform Configuration
// =============================================================================

/**
 * Maximum block dimension supported by platform
 * This is the fundamental platform capacity constraint.
 */
constexpr int PLATFORM_MAX_BLOCKDIM = 36;

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
constexpr int PLATFORM_MAX_AICPU_THREADS = 7;

/**
 * Compile-time upper bound on the number of AICPU threads CANN may
 * start per launch. The actual launch count is runtime-derived:
 * the host topology probe sets runtime.aicpu_launch_count =
 * popcount(OCCUPY), and DeviceRunner passes that count (not this
 * constant) to rtsLaunchCpuKernel.
 *
 * The bound exists for the static gate buffers in
 * src/common/platform/onboard/aicpu/platform_aicpu_affinity.cpp
 * (s_filter_thread_cpu[MAX_GATE_THREADS] etc.) — keep it ≥ any
 * runtime aicpu_launch_count we expect to see.
 *
 * Two over-subscription failure modes to keep in mind when choosing
 * the runtime launch count vs this bound:
 *   1. "Duplicate drop after barrier" — happens when CANN over-fills
 *      a subset of OCCUPY cpus (duplicates land on the sink cpu).
 *      The filter gate first barriers all launched threads, then
 *      classifies cpu_id and drops the duplicates (only the first
 *      thread to land on each ALLOWED slot survives). Safe.
 *   2. "Barrier never closes" — happens when launch_count exceeds
 *      popcount(OCCUPY) on the production AICPU kernel: contended
 *      init paths on the over-subscribed sink prevent some threads
 *      from reaching the barrier publish counter, the wait spins
 *      forever, and CANN reports aclrtSynchronizeStream rc=507000
 *      after the stream timeout. This is the reason the runtime
 *      uses popcount(OCCUPY) rather than this constant.
 *
 * Current value 14 = worst-case popcount(OCCUPY) on the Ascend950PR
 * 0x7ffe SKU (7 phys × 2 SMT = 14 user logical cpus). See
 * src/a5/docs/hardware.md "CANN AICPU thread dispatch" for the
 * empirical data and src/a5/platform/onboard/host/device_runner.cpp
 * for where the runtime count is decided.
 */
constexpr int PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH = 14;

/**
 * Default AICore op execution timeout (microseconds).
 * Passed to aclrtSetOpExecuteTimeOutV2 so that STARS actively monitors
 * AICore task execution and kills ops that exceed this threshold.
 * Overridden at runtime by SIMPLER_OP_EXECUTE_TIMEOUT_US when that env var
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
 * up first. Overridden at runtime by SIMPLER_STREAM_SYNC_TIMEOUT_MS when that
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
 * - MAX_AIC_PER_THREAD = MAX_BLOCKDIM * AIC_CORES_PER_BLOCKDIM = 36 * 1 = 36
 * - MAX_AIV_PER_THREAD = MAX_BLOCKDIM * AIV_CORES_PER_BLOCKDIM = 36 * 2 = 72
 */
constexpr int PLATFORM_MAX_AIC_PER_THREAD = PLATFORM_MAX_BLOCKDIM * PLATFORM_AIC_CORES_PER_BLOCKDIM;  // 36

constexpr int PLATFORM_MAX_AIV_PER_THREAD = PLATFORM_MAX_BLOCKDIM * PLATFORM_AIV_CORES_PER_BLOCKDIM;  // 72

constexpr int PLATFORM_MAX_CORES_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD + PLATFORM_MAX_AIV_PER_THREAD;  // 108

// =============================================================================
// Performance Profiling Configuration
// =============================================================================

/**
 * Maximum number of cores that can be profiled simultaneously
 * Calculated as: MAX_BLOCKDIM * CORES_PER_BLOCKDIM = 36 * 3 = 108
 */
constexpr int PLATFORM_MAX_CORES = PLATFORM_MAX_BLOCKDIM * PLATFORM_CORES_PER_BLOCKDIM;  // 108

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
 * Runtime drain never allocates; the host replenish slow path may batch-allocate
 * registered blocks to keep recycled lanes above their watermarks.
 */
constexpr int PLATFORM_PROF_SLOT_COUNT = 4;

/**
 * L2SwimlaneAicpuTaskBuffer pre-allocation count per AICore.
 * 1 goes into the free_queue at init, the rest into the recycled pool.
 */
constexpr int PLATFORM_PROF_BUFFERS_PER_CORE = 8;

/**
 * Host preallocation count per AICPU thread for the two phase pools, split per
 * kind (sched vs orch) because their throughput is asymmetric — a single shared
 * value over-provisions the lighter one. 1 buffer seeds the free_queue at init,
 * the rest the recycled pool.
 *
 * Floor for both: SLOT_COUNT(4) + 1 = 5 (free_queue fillable + 1 active buffer).
 * Pure host preallocation — zero ABI (the device-visible ready_queue is decoupled
 * below, held at the original 16). Right-sized from 16 by the a2a3 audit and
 * mirrored here (arch-independent theory, a5-silicon validation pending); rationale
 * in docs/dfx/dfx-buffer-capacity-audit.md.
 */
constexpr int PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD = 6;
constexpr int PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD = 8;

/**
 * Per-core L2SwimlaneAicoreTaskBuffer pre-allocation count.
 * 1 goes into the free_queue at init, the rest are recycled by host as
 * AICPU rotates per BUFFER_SIZE dispatches. Declared here so the ready-queue
 * formula below can include the AICore-pool worst-case burst depth.
 */
constexpr int PLATFORM_AICORE_BUFFERS_PER_CORE = 4;

/**
 * Ready queue capacity for performance data collection
 * Queue holds ReadyQueueEntry structs for buffers ready to be read by Host.
 * Sized to match pre-allocation total across all cores and threads.
 *
 * The phase term is sized to the ORIGINAL per-thread count (16), decoupled from
 * PLATFORM_PROF_{SCHED,ORCH}_BUFFERS_PER_THREAD, on purpose: this array sizes the
 * device-visible L2SwimlaneDataHeader::queues[], so shrinking the host
 * preallocation must NOT shrink it (that would change the shm layout = ABI).
 */
constexpr int PLATFORM_PROF_READYQUEUE_BUFFERS_PER_THREAD = 16;
constexpr int PLATFORM_PROF_READYQUEUE_SIZE =
    PLATFORM_MAX_CORES * PLATFORM_PROF_BUFFERS_PER_CORE +
    2 * PLATFORM_MAX_AICPU_THREADS * PLATFORM_PROF_READYQUEUE_BUFFERS_PER_THREAD +
    PLATFORM_MAX_CORES * PLATFORM_AICORE_BUFFERS_PER_CORE;

// PLATFORM_PHASE_RECORDS_PER_THREAD lives in l2_swimlane_profiling.h (16384) —
// kept beside the buffer types that use it. Do not redeclare here.

/**
 * System counter frequency (get_sys_cnt)
 * Used to convert timestamps to microseconds.
 */
constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 1000000000;  // 1000 MHz

/**
 * Timeout duration for performance data collection (seconds)
 */
constexpr int PLATFORM_PROF_TIMEOUT_SECONDS = 30;

/**
 * Number of empty polling iterations before checking timeout
 */
constexpr int PLATFORM_PROF_EMPTY_POLLS_CHECK_NUM = 1000;

inline double cycles_to_us(uint64_t cycles) {
    return (static_cast<double>(cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
}

// Profiling-related runtime flags shared through AICPU-AICore handshake.
// "Profiling" is the umbrella; each bit is a parallel diagnostics sub-feature.
#define SIMPLER_DFX_FLAG_NONE 0u
#define SIMPLER_DFX_FLAG_DUMP_ARGS (1u << 0)
#define SIMPLER_DFX_FLAG_L2_SWIMLANE (1u << 1)
#define SIMPLER_DFX_FLAG_PMU (1u << 2)
#define SIMPLER_DFX_FLAG_DEP_GEN (1u << 3)
#define SIMPLER_DFX_FLAG_SCOPE_STATS (1u << 4)
#define SIMPLER_GET_DFX_FLAG(flags, bit) ((((uint32_t)(flags)) & ((uint32_t)(bit))) != 0u)
#define SIMPLER_SET_DFX_FLAG(flags, bit) ((flags) |= (uint32_t)(bit))
#define SIMPLER_CLEAR_DFX_FLAG(flags, bit) ((flags) &= ~((uint32_t)(bit)))

// =============================================================================
// Args Dump Configuration
// =============================================================================

/**
 * Number of ArgsDumpRecord entries per DumpMetaBuffer.
 * Each record is 128 bytes, so one buffer = RECORDS * 128 bytes.
 */
constexpr int PLATFORM_DUMP_RECORDS_PER_BUFFER = 256;

/**
 * Pre-allocated DumpMetaBuffer count per AICPU scheduling thread.
 * Controls initial buffer supply and arena sizing.
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
 * Default: 8 * 256 * 65536 = 128 MB per thread.
 */
constexpr uint64_t PLATFORM_DUMP_AVG_TENSOR_BYTES = 65536;

/**
 * Maximum tensor dimensions (matches MAX_TENSOR_DIMS).
 */
constexpr int PLATFORM_DUMP_MAX_DIMS = 5;

/**
 * Ready queue capacity for dump data per AICPU thread.
 */
constexpr int PLATFORM_DUMP_READYQUEUE_SIZE = PLATFORM_MAX_AICPU_THREADS * PLATFORM_DUMP_BUFFERS_PER_THREAD * 2;

/**
 * Idle timeout duration for args dump collection (seconds).
 */
constexpr int PLATFORM_DUMP_TIMEOUT_SECONDS = 30;

// =============================================================================
// PMU Profiling Configuration
// =============================================================================

/**
 * Number of PmuRecord entries per PmuBuffer.
 */
constexpr int PLATFORM_PMU_RECORDS_PER_BUFFER = 512;

/**
 * Per-core PMU staging ring depth (AICore-side dual-issue slots).
 *
 * Same constraints as PLATFORM_L2_AICORE_RING_SIZE: ≥ in-flight task
 * depth on a single core, and ideally a power of two so the
 * `task_id % RING_SIZE` slot index compiles to a mask. Decoupled from
 * the rotating PmuBuffer so the AICore write address is stable across
 * buffer flips.
 */
constexpr int PLATFORM_PMU_AICORE_RING_SIZE = 2;

/**
 * SPSC free-queue slot count per core (Host pushes, AICPU pops).
 */
constexpr int PLATFORM_PMU_SLOT_COUNT = 4;

/**
 * Number of PmuBuffers pre-allocated per core (initial pool size).
 * Sized at 2 = 2× the Little's Law need (L≈1: PMU records are produced at the
 * slow fixed sampling rate, decoupled from submit, and the 64 B record drains
 * instantly, so peak in-flight is 1). Was 4 (a 4× over-provision). Mirrors the
 * a2a3 change (validated zero-drop on a2a3 silicon); the L≈1 argument is
 * arch-independent, but **a5-silicon onboard validation is still pending**.
 * See docs/dfx/dfx-buffer-capacity-audit.md.
 */
constexpr int PLATFORM_PMU_BUFFERS_PER_CORE = 2;

/**
 * Ready queue capacity for PMU data per AICPU thread.
 */
constexpr int PLATFORM_PMU_READYQUEUE_SIZE = PLATFORM_MAX_CORES * PLATFORM_PMU_BUFFERS_PER_CORE;

/**
 * Idle timeout duration for PMU collection (seconds).
 */
constexpr int PLATFORM_PMU_TIMEOUT_SECONDS = 30;

// =============================================================================
// dep_gen (SubmitTrace) Configuration
// =============================================================================

/**
 * Number of DepGenRecord entries per DepGenBuffer.
 * Each DepGenRecord is 4672 B (16 Tensor blobs + small header). At 4×1024 =
 * 4096 in-flight records (~19 MB), aligning dep_gen's in-flight count with the
 * scope_stats / l2 AicoreTask pools (also 4096) per the #977 cross-subsystem
 * review. History: original 32 (dropped 50% on unroll Case1) → #977 commit
 * overshot to 2048 → 1024 here (#977 Primary's actual proposal). Flood drops
 * are rate-bound, not capacity-bound — buffer sizing cannot fix them; real
 * dependency-paced workloads never drop. dep_gen is opt-in (--enable-dep-gen).
 * a5 record size (4672 B) and cohort (4096) are identical to a2a3, so the same
 * 1024 applies; **a5-silicon validation still pending**. See
 * docs/dfx/dfx-buffer-capacity-audit.md.
 */
constexpr int PLATFORM_DEP_GEN_RECORDS_PER_BUFFER = 1024;

/**
 * SPSC free_queue slot count for dep_gen buffers (Host→Device hand-off depth).
 */
constexpr int PLATFORM_DEP_GEN_SLOT_COUNT = 4;

/**
 * Pre-allocated DepGenBuffer count per orchestrator instance.
 */
constexpr int PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE = 4;

/**
 * Ready queue capacity for dep_gen (per AICPU thread). dep_gen is single-
 * instance (orchestrator thread), but the queue array is per-thread to match
 * the ProfilerBase contract.
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
 * Sized at 4 = 4× the Little's Law need (L≈1: scope_stats records are produced
 * per scope enter/exit, decoupled from submit volume, 52 B record drains
 * instantly, peak in-flight 1). Was 8 (an 8× over-provision). Mirrors the a2a3
 * change (validated zero-drop on a2a3); L≈1 is arch-independent, but
 * **a5-silicon validation still pending**. See
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
constexpr uint32_t REG_SPR_DATA_MAIN_BASE_OFFSET = 0xD0;  // Task dispatch (AICPU→AICore)
constexpr uint32_t REG_SPR_COND_OFFSET = 0x5108;          // Status (AICore→AICPU): 0=IDLE, 1=BUSY
constexpr uint32_t REG_SPR_CTRL_OFFSET = 0x0;             // AICore internal CTRL SPR (bit0 = PMU enable)

// Exit signal for AICore shutdown
constexpr uint32_t AICORE_EXIT_SIGNAL = 0x7FFFFFF0;

// Physical core ID mask for get_coreid()
constexpr uint32_t AICORE_COREID_MASK = 0x0FFF;

// PMU MMIO register offsets (DAV_3510 / a5). Values ported from pypto
// aicore_prof_dav3510_pmu.h. Accessed by AICPU through the per-core PMU
// register block (separate from the AICore SPR block used for DATA_MAIN_BASE
// and COND). AICore does not touch these — it only toggles its internal
// CTRL SPR (REG_SPR_CTRL_OFFSET) for per-task counter gating.
constexpr uint32_t REG_MMIO_PMU_CTRL_0_OFFSET = 0x4200;      // PMU framework enable (GLB_PMU_EN | USER | SAMPLE)
constexpr uint32_t REG_MMIO_PMU_CTRL_1_OFFSET = 0x2400;      // PMU secondary control (a5 only)
constexpr uint32_t REG_MMIO_PMU_CNT0_OFFSET = 0x4210;        // Event counter 0
constexpr uint32_t REG_MMIO_PMU_CNT1_OFFSET = 0x4218;        // Event counter 1
constexpr uint32_t REG_MMIO_PMU_CNT2_OFFSET = 0x4220;        // Event counter 2
constexpr uint32_t REG_MMIO_PMU_CNT3_OFFSET = 0x4228;        // Event counter 3
constexpr uint32_t REG_MMIO_PMU_CNT4_OFFSET = 0x4230;        // Event counter 4
constexpr uint32_t REG_MMIO_PMU_CNT5_OFFSET = 0x4238;        // Event counter 5
constexpr uint32_t REG_MMIO_PMU_CNT6_OFFSET = 0x4240;        // Event counter 6
constexpr uint32_t REG_MMIO_PMU_CNT7_OFFSET = 0x4248;        // Event counter 7
constexpr uint32_t REG_MMIO_PMU_CNT8_OFFSET = 0x4250;        // Event counter 8 (a5 only)
constexpr uint32_t REG_MMIO_PMU_CNT9_OFFSET = 0x4254;        // Event counter 9 (a5 only)
constexpr uint32_t REG_MMIO_PMU_CNT_TOTAL0_OFFSET = 0x4260;  // Total cycle counter, low 32 bits
constexpr uint32_t REG_MMIO_PMU_CNT_TOTAL1_OFFSET = 0x4264;  // Total cycle counter, high 32 bits
constexpr uint32_t REG_MMIO_PMU_START_CYC0_OFFSET = 0x42A0;  // Counting-range start cycle, low 32 bits
constexpr uint32_t REG_MMIO_PMU_START_CYC1_OFFSET = 0x42A4;  // Counting-range start cycle, high 32 bits
constexpr uint32_t REG_MMIO_PMU_STOP_CYC0_OFFSET = 0x42A8;   // Counting-range stop cycle, low 32 bits
constexpr uint32_t REG_MMIO_PMU_STOP_CYC1_OFFSET = 0x42AC;   // Counting-range stop cycle, high 32 bits
constexpr uint32_t REG_MMIO_PMU_CNT0_IDX_OFFSET = 0x2500;    // Event selector for CNT0
constexpr uint32_t REG_MMIO_PMU_CNT1_IDX_OFFSET = 0x2504;    // Event selector for CNT1
constexpr uint32_t REG_MMIO_PMU_CNT2_IDX_OFFSET = 0x2508;    // Event selector for CNT2
constexpr uint32_t REG_MMIO_PMU_CNT3_IDX_OFFSET = 0x250C;    // Event selector for CNT3
constexpr uint32_t REG_MMIO_PMU_CNT4_IDX_OFFSET = 0x2510;    // Event selector for CNT4
constexpr uint32_t REG_MMIO_PMU_CNT5_IDX_OFFSET = 0x2514;    // Event selector for CNT5
constexpr uint32_t REG_MMIO_PMU_CNT6_IDX_OFFSET = 0x2518;    // Event selector for CNT6
constexpr uint32_t REG_MMIO_PMU_CNT7_IDX_OFFSET = 0x251C;    // Event selector for CNT7
constexpr uint32_t REG_MMIO_PMU_CNT8_IDX_OFFSET = 0x2520;    // Event selector for CNT8 (a5 only)
constexpr uint32_t REG_MMIO_PMU_CNT9_IDX_OFFSET = 0x2524;    // Event selector for CNT9 (a5 only)

// PMU_CTRL_0 enable value: GLB_PMU_EN | (USER_PMU_MODE_EN << 1) | (SAMPLE_PMU_MODE_EN << 2)
constexpr uint32_t REG_MMIO_PMU_CTRL_0_ENABLE_VAL = 0x7;
// PMU_CTRL_1 enable value: GLB_PMU_EN (a5 only; second control register)
constexpr uint32_t REG_MMIO_PMU_CTRL_1_ENABLE_VAL = 0x1;

/**
 * Register identifier for unified read_reg/write_reg interface.
 *
 * The PMU counter slots (PMU_CNT0..PMU_CNT9) and event selector slots
 * (PMU_CNT0_IDX..PMU_CNT9_IDX) are assigned contiguous values so that
 * reg_index(base, i) can index into them as arrays. Keep these runs
 * contiguous — static_asserts below enforce this.
 */
enum class RegId : uint8_t {
    DATA_MAIN_BASE = 0,  // Task dispatch (AICPU→AICore)
    COND = 1,            // Status (AICore→AICPU)
    CTRL = 2,            // AICore internal CTRL SPR (PMU enable, etc.) — AICore-only

    // PMU framework (AICPU-only; AICore does not access these)
    PMU_CTRL_0 = 3,
    PMU_CTRL_1 = 4,

    // PMU counters (10 contiguous slots)
    PMU_CNT0 = 5,
    PMU_CNT1 = 6,
    PMU_CNT2 = 7,
    PMU_CNT3 = 8,
    PMU_CNT4 = 9,
    PMU_CNT5 = 10,
    PMU_CNT6 = 11,
    PMU_CNT7 = 12,
    PMU_CNT8 = 13,
    PMU_CNT9 = 14,

    // PMU total cycle counter (64-bit split across two 32-bit regs)
    PMU_CNT_TOTAL0 = 15,
    PMU_CNT_TOTAL1 = 16,

    // PMU counting-range start/stop cycle bounds
    PMU_START_CYC0 = 17,
    PMU_START_CYC1 = 18,
    PMU_STOP_CYC0 = 19,
    PMU_STOP_CYC1 = 20,

    // PMU event selectors (10 contiguous slots, parallel to PMU_CNT0..CNT9)
    PMU_CNT0_IDX = 21,
    PMU_CNT1_IDX = 22,
    PMU_CNT2_IDX = 23,
    PMU_CNT3_IDX = 24,
    PMU_CNT4_IDX = 25,
    PMU_CNT5_IDX = 26,
    PMU_CNT6_IDX = 27,
    PMU_CNT7_IDX = 28,
    PMU_CNT8_IDX = 29,
    PMU_CNT9_IDX = 30,
};

static_assert(
    static_cast<int>(RegId::PMU_CNT9) - static_cast<int>(RegId::PMU_CNT0) == 9,
    "PMU_CNT0..PMU_CNT9 must be contiguous for reg_index()"
);
static_assert(
    static_cast<int>(RegId::PMU_CNT9_IDX) - static_cast<int>(RegId::PMU_CNT0_IDX) == 9,
    "PMU_CNT0_IDX..PMU_CNT9_IDX must be contiguous for reg_index()"
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
    case RegId::CTRL:
        return REG_SPR_CTRL_OFFSET;
    case RegId::PMU_CTRL_0:
        return REG_MMIO_PMU_CTRL_0_OFFSET;
    case RegId::PMU_CTRL_1:
        return REG_MMIO_PMU_CTRL_1_OFFSET;
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
    case RegId::PMU_CNT8:
        return REG_MMIO_PMU_CNT8_OFFSET;
    case RegId::PMU_CNT9:
        return REG_MMIO_PMU_CNT9_OFFSET;
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
    case RegId::PMU_CNT8_IDX:
        return REG_MMIO_PMU_CNT8_IDX_OFFSET;
    case RegId::PMU_CNT9_IDX:
        return REG_MMIO_PMU_CNT9_IDX_OFFSET;
    }
    return 0;  // unreachable: all RegId cases handled above
}

/**
 * Index into a contiguous RegId run (e.g. reg_index(PMU_CNT0, 3) == PMU_CNT3).
 * Caller is responsible for keeping `i` within the run's length.
 */
constexpr RegId reg_index(RegId base, int i) { return static_cast<RegId>(static_cast<uint8_t>(base) + i); }

// =============================================================================
// Simulated Register Sparse Mapping Configuration
// =============================================================================

/**
 * DAV_3510 register layout is sparse with three non-contiguous regions:
 * - Page 0: 0x0000-0x0FFF  (AICore SPR low: CTRL at 0x0, DATA_MAIN_BASE at 0xD0)
 * - Page 2: 0x2400-0x43FF  (PMU MMIO: CNT*_IDX 0x2500-0x2524, CTRL_1 0x2400,
 *                            CTRL_0 0x4200, CNT0-9 0x4210-0x4254, TOTAL 0x4260-64,
 *                            START/STOP_CYC 0x42A0-0x42AC)
 * - Page 1: 0x5000-0x5FFF  (AICore SPR high: COND at 0x5108)
 *
 * To save memory, we allocate three separate pages per core instead of
 * a single 24KB block.
 */
constexpr uint32_t SIM_REG_PAGE0_SIZE = 0x1000;  // 4KB for page 0x0000-0x0FFF
constexpr uint32_t SIM_REG_PAGE2_SIZE = 0x2000;  // 8KB for PMU page 0x2400-0x43FF
constexpr uint32_t SIM_REG_PAGE1_SIZE = 0x1000;  // 4KB for page 0x5000-0x5FFF
constexpr uint32_t SIM_REG_PAGE2_BASE = 0x2400;  // Base address of PMU page
constexpr uint32_t SIM_REG_PAGE1_BASE = 0x5000;  // Base address of SPR high page

// Total size per core (three pages = 16KB)
constexpr uint32_t SIM_REG_TOTAL_SIZE = SIM_REG_PAGE0_SIZE + SIM_REG_PAGE2_SIZE + SIM_REG_PAGE1_SIZE;

/**
 * Calculate actual memory pointer for a hardware register offset in sparse layout
 *
 * Simple byte-pointer arithmetic for sparse register mapping.
 * Callers are responsible for casting to/from uint8_t*.
 *
 * @param reg_base  Base address of the sparse register block (page 0 start)
 * @param offset    Hardware register offset (e.g., 0xD0, 0x2500, 0x4210, 0x5108)
 * @return Pointer to the actual memory location (as uint8_t*)
 */
inline volatile uint8_t *sparse_reg_ptr(volatile uint8_t *reg_base, uint32_t offset) {
    if (offset < SIM_REG_PAGE2_BASE) {
        // Register in page 0 (0x0000-0x0FFF)
        return reg_base + offset;
    } else if (offset < SIM_REG_PAGE1_BASE) {
        // Register in page 2 (PMU, 0x2400-0x43FF)
        return reg_base + SIM_REG_PAGE0_SIZE + (offset - SIM_REG_PAGE2_BASE);
    } else {
        // Register in page 1 (0x5000-0x5FFF)
        return reg_base + SIM_REG_PAGE0_SIZE + SIM_REG_PAGE2_SIZE + (offset - SIM_REG_PAGE1_BASE);
    }
}

// =============================================================================
// Hardware Configuration Constants
// =============================================================================

/**
 * Number of sub-cores per AICore
 * Hardware architecture: 1 AICore = 1 AIC + 2 AIV sub-cores
 */
constexpr uint32_t PLATFORM_SUB_CORES_PER_AICORE = PLATFORM_CORES_PER_BLOCKDIM;

/**
 * DAV 3510 chip architecture
 * - 2 dies, each with 18 AICores
 * - Each AICore has 3 sub-cores (1 AIC + 2 AIV)
 */
constexpr uint32_t PLATFORM_NUM_DIES = 2;
constexpr uint32_t PLATFORM_AICORE_PER_DIE = 18;

/**
 * Maximum physical AICore count for DAV 3510 chip
 */
namespace DAV_3510 {
constexpr uint32_t PLATFORM_MAX_PHYSICAL_CORES = PLATFORM_NUM_DIES * PLATFORM_AICORE_PER_DIE;
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

#endif  // SRC_A5_PLATFORM_INCLUDE_COMMON_PLATFORM_CONFIG_H_
