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
// probes_types.h — shared types between the host launcher and AICPU probe SO.
//
// Mirrors the smaller surface of aicore-notification-perf/shared/handshake.h:
// no AICore is involved here, so there is no Handshake — just a result block
// for the AICPU to write timing samples into, and a DeviceArgs envelope that
// tells the AICPU which AIC_CTRL base + how many cores it can probe.

#ifndef AICPU_MMIO_PROBES_TYPES_H_
#define AICPU_MMIO_PROBES_TYPES_H_

#include <cstdint>

// Register offsets within an AIC sub-core slot. Mirrors
// src/a2a3/platform/include/common/platform_config.h. The a5 chip uses
// 0xD0 for DMB and 0x5108 for COND; override per-chip if you port the
// probe.
constexpr uint32_t kProbeRegSprDmbOffset = 0xA0;
constexpr uint32_t kProbeRegSprCondOffset = 0x4C8;

constexpr uint64_t kProbeCoreStride = 8ULL * 1024 * 1024;  // 8 MiB per AICore
constexpr uint64_t kProbeSubCoreStride = 0x100000ULL;      // 1 MiB per sub-core

// Sentinels — match production runtime so STR DMB tests leave AIC_CTRL in a
// state the chip considers idle.
constexpr uint32_t kProbeAicpuIdleTaskId = 0x7FFFFFFDu;

// Max core counts we probe over. Phase 12-C scales M-threads up to
// kProbeMaxConcurrentReaders, each owning a distinct AIC core. A real chip
// has 24 AICs, but capping at 8 keeps the result block small and pthreads
// well-behaved on the AICPU OS.
constexpr uint32_t kProbeMaxCores = 8;
constexpr uint32_t kProbeMaxConcurrentReaders = 4;

// What the AICPU writes back to GM at result_addr. All ticks are in the
// shared system counter (CNTVCT_EL0; 50 MHz on a3 / a5 → 20 ns / tick).
struct alignas(8) MmioProbeResult {
    // Phase 4 — STR DMB cost.
    uint64_t str_burst_n;            // # STRs in the burst (e.g. 1000)
    uint64_t str_burst_total_ticks;  // wall-clock for the whole burst
    uint64_t str_lat_round_trip;     // STR then LDR — full round trip in ticks

    // Phase 12-A — single thread, same core, 10000 LDR COND.
    uint64_t ldr_n;
    uint64_t ldr_a_total_ticks;  // same-core

    // Phase 12-B — single thread, rotating across `probed_cores` cores.
    uint64_t ldr_b_total_ticks;
    uint32_t probed_cores;  // how many AIC cores we actually had to rotate over

    // Phase 12-C — multi-thread, M threads each on its own core.
    // Entry i = thread i's total ticks for its 10000 LDRs at concurrency M = i+1.
    // We store per-thread instead of an aggregate so the parallel-scaling
    // claim ("per-thread cost unchanged as M grows") is directly visible.
    uint32_t _pad0;
    uint64_t ldr_c_thread_ticks[kProbeMaxConcurrentReaders][kProbeMaxConcurrentReaders];

    // Diagnostics.
    uint32_t magic;         // 0xABCD_1234 on success
    int32_t probe_rc;       // top-level AICPU return
    uint64_t observed_pid;  // AICPU OS process pid (sanity)
};

constexpr uint32_t kMmioProbeResultMagic = 0xABCD1234u;

// DeviceArgs envelope.
struct alignas(8) MmioProbeDeviceArgs {
    uint64_t reserved_pre[12];       // 0..95 — dispatcher bootstrap uses these
    uint64_t result_addr;            // 96 — &MmioProbeResult
    uint64_t input_token;            // 104 — echoed token, sanity
    uint64_t aic_ctrl_reg_base;      // 112 — halMemCtl(REG_AIC_CTRL).ptr
    uint32_t n_aic_cores_available;  // 120 — host caller's count of probable AICs
    uint32_t _pad;                   // 124
};

#endif  // AICPU_MMIO_PROBES_TYPES_H_
