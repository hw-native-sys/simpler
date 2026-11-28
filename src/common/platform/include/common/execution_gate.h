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

#include <cstddef>
#include <cstdint>

inline constexpr uint32_t EXECUTION_GATE_VERSION = 1;

enum class ExecutionGateDecision : uint32_t { Pending = 0, Go = 1, Cancel = 2 };
enum class ExecutionGateOutcome : uint32_t { NotEntered = 0, Go = 1, Cancel = 2, Invalid = 3 };

// The Host publishes this line before recording either stream notification.
// Its identity and terminal decision remain immutable until both streams drain.
struct alignas(64) ExecutionGateHeader {
    uint32_t version{EXECUTION_GATE_VERSION};
    uint32_t decision{static_cast<uint32_t>(ExecutionGateDecision::Pending)};
    uint64_t identity{0};
    uint64_t generation{0};
    uint64_t reserved[5]{};
};

// Each role owns a separate line; a device cache writeback cannot overwrite
// another role's outcome or the Host-owned publication line.
struct alignas(64) ExecutionGateStatus {
    volatile uint32_t outcome{static_cast<uint32_t>(ExecutionGateOutcome::NotEntered)};
    uint32_t reserved[15]{};
};

struct alignas(64) ExecutionGateRecord {
    ExecutionGateHeader header;
    ExecutionGateStatus aicpu;
    ExecutionGateStatus aic;
    ExecutionGateStatus aiv;
};

static_assert(sizeof(ExecutionGateHeader) == 64 && alignof(ExecutionGateHeader) == 64);
static_assert(offsetof(ExecutionGateHeader, version) == 0);
static_assert(offsetof(ExecutionGateHeader, decision) == 4);
static_assert(offsetof(ExecutionGateHeader, identity) == 8);
static_assert(offsetof(ExecutionGateHeader, generation) == 16);
static_assert(sizeof(ExecutionGateStatus) == 64 && alignof(ExecutionGateStatus) == 64);
static_assert(offsetof(ExecutionGateStatus, outcome) == 0);
static_assert(sizeof(ExecutionGateRecord) == 256 && alignof(ExecutionGateRecord) == 64);
static_assert(offsetof(ExecutionGateRecord, header) == 0);
static_assert(offsetof(ExecutionGateRecord, aicpu) == 64);
static_assert(offsetof(ExecutionGateRecord, aic) == 128);
static_assert(offsetof(ExecutionGateRecord, aiv) == 192);
static_assert(__is_trivially_copyable(ExecutionGateRecord) && __is_standard_layout(ExecutionGateRecord));

#if defined(__DAV_VEC__) || defined(__DAV_CUBE__)
#define SIMPLER_EXECUTION_GATE_DEVICE __aicore__
#else
#define SIMPLER_EXECUTION_GATE_DEVICE
#endif

SIMPLER_EXECUTION_GATE_DEVICE inline constexpr ExecutionGateOutcome execution_gate_outcome(
    uint32_t version, uint32_t decision, uint64_t identity, uint64_t generation, uint64_t expected_identity,
    uint64_t expected_generation
) {
    if (version != EXECUTION_GATE_VERSION || expected_identity == 0 || expected_generation == 0 ||
        identity != expected_identity || generation != expected_generation) {
        return ExecutionGateOutcome::Invalid;
    }
    if (decision == static_cast<uint32_t>(ExecutionGateDecision::Go)) return ExecutionGateOutcome::Go;
    if (decision == static_cast<uint32_t>(ExecutionGateDecision::Cancel)) return ExecutionGateOutcome::Cancel;
    return ExecutionGateOutcome::Invalid;
}

#undef SIMPLER_EXECUTION_GATE_DEVICE
