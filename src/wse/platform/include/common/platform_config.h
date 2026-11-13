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

// WSE Core/PE topology and constants consumed by shared Host layouts.
// AICPU and diagnostic capacities describe those layouts, not WSE execution roles.

#pragma once

#include <cstdint>
#include "common/hardware_spec.h"

// =============================================================================
// Base Platform Configuration
// =============================================================================

// Maximum block dimension supported by platform This is the fundamental platform capacity constraint.
constexpr int PLATFORM_MAX_BLOCKDIM = wse::MAX_CORE_COUNT;

// One WSE Core has four homogeneous PEs; task Block count is independent.
constexpr int PLATFORM_CORES_PER_BLOCKDIM = wse::PE_PER_CORE;
constexpr int PLATFORM_AIC_CORES_PER_BLOCKDIM = 1;
constexpr int PLATFORM_AIV_CORES_PER_BLOCKDIM = 2;
constexpr uint32_t PLATFORM_SSBUF_SIZE = 3 * 1024;

// Maximum AICPU scheduling threads Determines parallelism level of the AICPU task scheduler.
constexpr int PLATFORM_MAX_AICPU_THREADS = 5;

// Shared Host layout default; WSE launches zero AICPU threads.
constexpr int PLATFORM_DEFAULT_AICPU_THREAD_NUM = 5;  // 1 orch + 4 sched

// Shared per-thread diagnostic storage bound; must not exceed MAX_GATE_THREADS (16).
constexpr int PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH = 14;

// Shared timeout contract, in microseconds; WSE has no STARS execution monitor.
constexpr uint64_t PLATFORM_OP_EXECUTE_TIMEOUT_US = 45000000;  // 45s

// No-progress timeout in milliseconds, also used by Host timeout validation.
// Must be shorter than the op-execute timeout; SIMPLER_SCHEDULER_TIMEOUT_MS overrides it.
constexpr int32_t PLATFORM_SCHEDULER_TIMEOUT_MS = 20000;

// Tensor readiness timeouts in milliseconds; simulation allows a larger CPU scheduling budget.
constexpr int32_t PLATFORM_ONBOARD_TENSOR_DATA_WAIT_TIMEOUT_MS = 15000;
constexpr int32_t PLATFORM_SIM_TENSOR_DATA_WAIT_TIMEOUT_MS = 30000;

// Stream timeout in milliseconds, longer than the op-execute timeout so device errors can surface.
constexpr int PLATFORM_STREAM_SYNC_TIMEOUT_MS = 50000;  // 50s (> op-exec 45s)

// =============================================================================
// Derived Platform Limits
// =============================================================================

// Shared per-thread diagnostic capacities derived from the platform maximum.
constexpr int PLATFORM_MAX_AIC_PER_THREAD = PLATFORM_MAX_BLOCKDIM * PLATFORM_AIC_CORES_PER_BLOCKDIM;

constexpr int PLATFORM_MAX_AIV_PER_THREAD = PLATFORM_MAX_BLOCKDIM * PLATFORM_AIV_CORES_PER_BLOCKDIM;

constexpr int PLATFORM_MAX_CORES_PER_THREAD = PLATFORM_MAX_AIC_PER_THREAD + PLATFORM_MAX_AIV_PER_THREAD;

// =============================================================================
// Performance Profiling Configuration
// =============================================================================

// Maximum simultaneously profiled PE count.
constexpr int PLATFORM_MAX_CORES = PLATFORM_MAX_BLOCKDIM * PLATFORM_CORES_PER_BLOCKDIM;

// Performance buffer capacity per buffer Number of ChipSwimlaneAicpuTaskRecord entries per dynamically allocated
// ChipSwimlaneAicpuTaskBuffer
constexpr int PLATFORM_PROF_BUFFER_SIZE = 1000;

// Host replenishment publishes buffer addresses; device drain never allocates.
constexpr int PLATFORM_PROF_SLOT_COUNT = 4;

// ChipSwimlaneAicpuTaskBuffer pre-allocation count per AICore. 1 goes into the free_queue at init, the rest into the
// recycled pool.
constexpr int PLATFORM_PROF_BUFFERS_PER_CORE = 8;

// Phase pools require at least SLOT_COUNT + 1 buffers. Host preallocation
// is independent of the device-visible queue geometry.
constexpr int PLATFORM_PROF_SCHED_BUFFERS_PER_THREAD = 6;
constexpr int PLATFORM_PROF_ORCH_BUFFERS_PER_THREAD = 8;

// Whole buffers belong to individual recording threads. The pool must exceed
// the submitting thread plus GRAPH_MAX_DEFINITIONS (16) recorder producers.
constexpr int PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER = 512;
constexpr int PLATFORM_HOST_PHASE_BUFFERS = 24;

// One buffer seeds the free queue; remaining buffers are recycled by the Host.
constexpr int PLATFORM_AICORE_BUFFERS_PER_CORE = 4;

// Device-visible queue geometry. The phase term stays at 16 independently
// of Host preallocation counts; changing it changes the shared-memory layout.
constexpr int PLATFORM_PROF_READYQUEUE_BUFFERS_PER_THREAD = 16;
constexpr int PLATFORM_PROF_READYQUEUE_SIZE =
    PLATFORM_MAX_CORES * PLATFORM_PROF_BUFFERS_PER_CORE +
    2 * PLATFORM_MAX_AICPU_THREADS * PLATFORM_PROF_READYQUEUE_BUFFERS_PER_THREAD +
    PLATFORM_MAX_CORES * PLATFORM_AICORE_BUFFERS_PER_CORE;

// PLATFORM_PHASE_RECORDS_PER_THREAD lives in chip_swimlane_profiling.h (16384) —
// kept beside the buffer types that use it. Do not redeclare here.

// System counter frequency (get_sys_cnt) Used to convert timestamps to microseconds.
constexpr uint64_t PLATFORM_PROF_SYS_CNT_FREQ = 1000000000;  // 1000 MHz

// aclrtEventGetTimestamp capability, verified on a5 silicon: the raw value
// is device-uptime microseconds and covers the same epoch as profiling syscnt
// after frequency normalization.
constexpr uint64_t PLATFORM_ACL_EVENT_TIMESTAMP_FREQ_HZ = 1000000;
constexpr const char *PLATFORM_ACL_EVENT_TIMESTAMP_UNIT = "device_uptime_us";

// Backpressure timeout in counter cycles, representing a 30-second ceiling.
constexpr uint64_t PLATFORM_DFX_BACKPRESSURE_TIMEOUT_CYCLES = PLATFORM_PROF_SYS_CNT_FREQ * 30;

// Timeout duration for performance data collection (seconds)
constexpr int PLATFORM_PROF_TIMEOUT_SECONDS = 30;

// Number of empty polling iterations before checking timeout
constexpr int PLATFORM_PROF_EMPTY_POLLS_CHECK_NUM = 1000;

inline double cycles_to_us(uint64_t cycles) {
    return (static_cast<double>(cycles) / PLATFORM_PROF_SYS_CNT_FREQ) * 1000000.0;
}

// Profiling-related runtime flags shared through AICPU-AICore handshake.
// "Profiling" is the umbrella; each bit is a parallel diagnostics sub-feature.
#define SIMPLER_DFX_FLAG_NONE 0u
#define SIMPLER_DFX_FLAG_DUMP_ARGS (1u << 0)
#define SIMPLER_DFX_FLAG_CHIP_SWIMLANE (1u << 1)
#define SIMPLER_DFX_FLAG_PMU (1u << 2)
#define SIMPLER_DFX_FLAG_DEP_GEN (1u << 3)
#define SIMPLER_DFX_FLAG_SCOPE_STATS (1u << 4)
#define SIMPLER_GET_DFX_FLAG(flags, bit) ((((uint32_t)(flags)) & ((uint32_t)(bit))) != 0u)
#define SIMPLER_SET_DFX_FLAG(flags, bit) ((flags) |= (uint32_t)(bit))
#define SIMPLER_CLEAR_DFX_FLAG(flags, bit) ((flags) &= ~((uint32_t)(bit)))

// =============================================================================
// Args Dump Configuration
// =============================================================================

// Number of ArgsDumpRecord entries per DumpMetaBuffer. Each record is 128 bytes, so one buffer = RECORDS * 128 bytes.
constexpr int PLATFORM_DUMP_RECORDS_PER_BUFFER = 256;

// Pre-allocated DumpMetaBuffer count per AICPU scheduling thread. Controls initial buffer supply and arena sizing.
constexpr int PLATFORM_DUMP_BUFFERS_PER_THREAD = 8;

// SPSC free_queue slot count for dump metadata buffers.
constexpr int PLATFORM_DUMP_SLOT_COUNT = 4;

// Tensor arena sizing uses BUFFERS_PER_THREAD * RECORDS_PER_BUFFER * AVG_TENSOR_BYTES.
constexpr uint64_t PLATFORM_DUMP_AVG_TENSOR_BYTES = 65536;

// Maximum tensor dimensions (matches MAX_TENSOR_DIMS).
constexpr int PLATFORM_DUMP_MAX_DIMS = 5;

// Ready queue capacity for dump data per AICPU thread.
constexpr int PLATFORM_DUMP_READYQUEUE_SIZE = PLATFORM_MAX_AICPU_THREADS * PLATFORM_DUMP_BUFFERS_PER_THREAD * 2;

// Idle timeout duration for args dump collection (seconds).
constexpr int PLATFORM_DUMP_TIMEOUT_SECONDS = 30;

// =============================================================================
// PMU Profiling Configuration
// =============================================================================

// Number of PmuRecord entries per PmuBuffer.
constexpr int PLATFORM_PMU_RECORDS_PER_BUFFER = 512;

// Shared A5 PMU record width, usable without importing the TaskId-bearing record header.
constexpr int PMU_COUNTER_COUNT_A5 = 10;

// Stable per-core staging depth, at least the in-flight task depth.
// Power-of-two sizing permits masked task indexing independently of buffer rotation.
constexpr int PLATFORM_PMU_AICORE_RING_SIZE = 2;

// SPSC free-queue slot count per core (Host pushes, AICPU pops).
constexpr int PLATFORM_PMU_SLOT_COUNT = 4;

// Initial per-core Host PMU buffer pool.
constexpr int PLATFORM_PMU_BUFFERS_PER_CORE = 2;

// Ready queue capacity for PMU data per AICPU thread.
constexpr int PLATFORM_PMU_READYQUEUE_SIZE = PLATFORM_MAX_CORES * PLATFORM_PMU_BUFFERS_PER_CORE;

// Idle timeout duration for PMU collection (seconds).
constexpr int PLATFORM_PMU_TIMEOUT_SECONDS = 30;

// =============================================================================
// dep_gen (SubmitTrace) Configuration
// =============================================================================

// Each DepGenRecord occupies 4736 bytes; four buffers retain 4096 records.
constexpr int PLATFORM_DEP_GEN_RECORDS_PER_BUFFER = 1024;

// SPSC free_queue slot count for dep_gen buffers (Host→Device hand-off depth).
constexpr int PLATFORM_DEP_GEN_SLOT_COUNT = 4;

// Pre-allocated DepGenBuffer count per orchestrator instance.
constexpr int PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE = 4;

// Single orchestrator producer; per-thread queue array follows ProfilerBase layout.
constexpr int PLATFORM_DEP_GEN_READYQUEUE_SIZE = PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE * 2;

// Idle timeout duration for dep_gen collection (seconds).
constexpr int PLATFORM_DEP_GEN_TIMEOUT_SECONDS = 30;

// =============================================================================
// scope_stats Configuration
// =============================================================================

// Scope records occupy 52 bytes; buffer size controls switch and copy frequency.
constexpr int PLATFORM_SCOPE_STATS_RECORDS_PER_BUFFER = 512;

// SPSC free_queue slot count for scope_stats buffers (Host→Device hand-off depth).
constexpr int PLATFORM_SCOPE_STATS_SLOT_COUNT = 8;

// Initial per-orchestrator scope-statistics buffer pool.
constexpr int PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE = 4;

// Ready queue capacity for scope_stats (per AICPU thread). scope_stats is single-instance so headroom over
// BUFFERS_PER_INSTANCE is small.
constexpr int PLATFORM_SCOPE_STATS_READYQUEUE_SIZE = PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE * 2;

// Idle timeout duration for scope_stats collection (seconds).
constexpr int PLATFORM_SCOPE_STATS_TIMEOUT_SECONDS = 30;
