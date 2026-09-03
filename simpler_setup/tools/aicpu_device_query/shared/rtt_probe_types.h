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
#pragma once

#include <cstdint>
#include <type_traits>

// Full AICPU affinity preflight protocol (replaces the earlier 4-scheduler-only
// COND RTT probe). Host enumerates the user pool, launches one thread per pool
// CPU, measures pairwise atomic-flag handshake latency, then measures COND
// access sums for non-orchestrator threads.

constexpr uint32_t kAffinityMaxPool = 16;
constexpr uint32_t kAffinitySchedCount = 4;
constexpr uint32_t kAffinityHandshakeIters = 1000;
constexpr uint32_t kAffinitySamplesPerCore = 100;
constexpr uint32_t kAffinityWarmupSamplesPerCore = 8;
constexpr uint32_t kAffinityBarrierTimeoutSeconds = 30;
constexpr int32_t kAffinityHostSyncTimeoutMs = 30000;
constexpr int32_t kAffinityPreflightTimeoutExitCode = 124;

// Legacy aliases kept so existing include sites compile during the transition.
constexpr uint32_t kRttProbeSchedulerCount = kAffinitySchedCount;
constexpr uint32_t kRttProbeSamplesPerCore = kAffinitySamplesPerCore;
constexpr uint32_t kRttProbeWarmupSamplesPerCore = kAffinityWarmupSamplesPerCore;
constexpr uint32_t kRttProbeBarrierTimeoutSeconds = kAffinityBarrierTimeoutSeconds;

enum class AffinityProbeError : uint32_t {
    kNone = 0,
    kDuplicatePool = 1,
    kBarrierTimeout = 2,
    kInvalidPool = 3,
};

using RttProbeError = AffinityProbeError;

struct AffinityPoolSlot {
    int32_t pool_idx;
    int32_t cpu_id;
    uint64_t avg_handshake_ticks;
    uint64_t die0_sum_ticks;
    uint64_t die1_sum_ticks;
    uint32_t handshake_valid;
    uint32_t die_valid;
    uint32_t is_orch;
    uint32_t reserved;
};

struct AffinityPreflightOutput {
    uint32_t ready_count;
    uint32_t claimed_mask;
    uint32_t error_code;
    uint32_t pool_count;
    uint32_t orch_pool_idx;
    uint32_t handshake_done;
    uint32_t die_done;
    uint32_t reserved;
    AffinityPoolSlot slots[kAffinityMaxPool];
    // Symmetric pair average ticks; index = i * kAffinityMaxPool + j (i != j).
    uint64_t handshake_pair_ticks[kAffinityMaxPool * kAffinityMaxPool];
    // Atomic flag handshake scratch (req/ack). Host zeroes before launch.
    uint32_t handshake_req[kAffinityMaxPool][kAffinityMaxPool];
    uint32_t handshake_ack[kAffinityMaxPool][kAffinityMaxPool];
};

// Serial enum (aicpu_num=1): one thread writes its sched_getcpu into slot 0.
struct AffinityEnumOutput {
    uint32_t ready;
    int32_t cpu_id;
    uint32_t reserved[2];
};

struct AffinityEnumDeviceArgs {
    uint64_t reserved_pre[12];
    uint64_t output_addr;
};

// Reuses the dispatcher's 160-byte DeviceArgs buffer. Fields start at offset 96.
struct AffinityPreflightDeviceArgs {
    uint64_t reserved_pre[12];
    uint64_t output_addr;
    uint64_t aicore_regs_addr;
    uint64_t user_pool_addr;
    uint32_t pool_count;
    uint32_t aicore_count;
    uint32_t handshake_iters;
    uint32_t samples_per_core;
};

using RttProbeDeviceArgs = AffinityPreflightDeviceArgs;
using RttProbeOutput = AffinityPreflightOutput;
using RttProbeSlot = AffinityPoolSlot;

static_assert(sizeof(AffinityPreflightDeviceArgs) <= 160, "affinity probe args exceed DeviceArgs storage");
static_assert(sizeof(AffinityEnumDeviceArgs) <= 160, "affinity enum args exceed DeviceArgs storage");
static_assert(std::is_trivially_copyable_v<AffinityPoolSlot> && std::is_standard_layout_v<AffinityPoolSlot>);
static_assert(
    std::is_trivially_copyable_v<AffinityPreflightOutput> && std::is_standard_layout_v<AffinityPreflightOutput>
);
static_assert(
    std::is_trivially_copyable_v<AffinityPreflightDeviceArgs> && std::is_standard_layout_v<AffinityPreflightDeviceArgs>
);
