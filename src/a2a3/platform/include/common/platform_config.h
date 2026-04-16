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
 * Number of PerfRecord entries per dynamically allocated PerfBuffer
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
 * PerfBuffer pre-allocation count per AICore.
 * 1 goes into the free_queue at init, the rest into the recycled pool.
 */
constexpr int PLATFORM_PROF_BUFFERS_PER_CORE = 8;

/**
 * PhaseBuffer pre-allocation count per AICPU thread.
 * 1 goes into the free_queue at init, the rest into the recycled pool.
 */
constexpr int PLATFORM_PROF_BUFFERS_PER_THREAD = 16;

/**
 * Ready queue capacity for performance data collection
 * Queue holds ReadyQueueEntry structs for buffers ready to be read by Host.
 * Sized to match pre-allocation total across all cores and threads.
 */
constexpr int PLATFORM_PROF_READYQUEUE_SIZE =
    PLATFORM_MAX_CORES * PLATFORM_PROF_BUFFERS_PER_CORE + PLATFORM_MAX_AICPU_THREADS * PLATFORM_PROF_BUFFERS_PER_THREAD;

/**
 * System counter frequency (get_sys_cnt)
 * Used to convert timestamps to microseconds.
 */
constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 50000000;  // 50 MHz

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
#define PROFILING_FLAG_NONE 0u
#define PROFILING_FLAG_DUMP_TENSOR (1u << 0)
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
 * Maximum tensor dimensions (matches RUNTIME_MAX_TENSOR_DIMS).
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

// =============================================================================
// Register Communication Configuration
// =============================================================================

// Register offsets for AICore SPR access
constexpr uint32_t REG_SPR_DATA_MAIN_BASE_OFFSET = 0xA0;  // Task dispatch (AICPU→AICore)
constexpr uint32_t REG_SPR_COND_OFFSET = 0x4C8;           // Status (AICore→AICPU): 0=IDLE, 1=BUSY
constexpr uint32_t REG_SPR_FAST_PATH_ENABLE_OFFSET = 0x18;

// Fast path control values
constexpr uint32_t REG_SPR_FAST_PATH_OPEN = 0xE;
constexpr uint32_t REG_SPR_FAST_PATH_CLOSE = 0xF;

// Exit signal for AICore shutdown
constexpr uint32_t AICORE_EXIT_SIGNAL = 0x7FFFFFF0;

// Physical core ID mask for get_coreid()
constexpr uint32_t AICORE_COREID_MASK = 0x0FFF;

/**
 * Register identifier for unified read_reg/write_reg interface
 */
enum class RegId : uint8_t {
    DATA_MAIN_BASE = 0,    // Task dispatch (AICPU→AICore)
    COND = 1,              // Status (AICore→AICPU)
    FAST_PATH_ENABLE = 2,  // Fast path control
};

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
    }
    return 0;  // unreachable: all RegId cases handled above
}

// Size of simulated register block per core (covers largest offset + 4 bytes)
constexpr uint32_t SIM_REG_BLOCK_SIZE = 0x500;

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
