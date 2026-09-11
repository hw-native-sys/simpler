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
#include <type_traits>

namespace control_probe {
constexpr int32_t kWorkers = 3;
constexpr int32_t kAdmissionRejected = -31;
constexpr int32_t kEnqueueRejected = -6001;
constexpr uint64_t kContextGeneration = 41;
constexpr const char *kSeedEntry = "control_probe_seed_registers";
constexpr const char *kGateInitEntry = "control_probe_gate_init";
constexpr const char *kGateEntry = "control_probe_gate";
constexpr int32_t kExecutionThreads = 2;
constexpr int32_t kInitRejected = -47;

enum class Mode : uint32_t { Open, Reject, HostCancel, OmitControlClear };
enum class GateMode : uint32_t { Open, InitReject, AdmissionReject, TerminalInitReject };

struct Init {
    uint64_t descriptor;
    uint64_t registers;
    uint64_t result;
};
struct Invocation {
    Mode mode;
    uint32_t reserved;
};
struct alignas(64) Result {
    uint64_t epoch;
    uint64_t completed_rounds;
    int32_t runtime_status;
    int32_t cleanup_status;
    uint32_t opened;
    uint32_t seeded_registers;
    uint8_t reserved[32];
};
struct GateInit {
    uint64_t descriptor;
    uint64_t registers;
    uint64_t summary;
    uint64_t results;
    int32_t launched_threads;
    int32_t execution_threads;
    int32_t allowed_cpus[kExecutionThreads];
    uint8_t reserved[16];
};
struct GateInvocation {
    GateMode mode;
    uint32_t reserved;
};
struct alignas(64) GateSummary {
    uint64_t epoch;
    uint64_t completed_rounds;
    int32_t prepare_calls;
    int32_t init_calls;
    int32_t verdict_calls;
    int32_t run_calls;
    int32_t finish_calls;
    int32_t finalize_calls;
    int32_t publish_calls;
    int32_t clear_calls;
    int32_t runtime_status;
    int32_t cleanup_status;
    uint8_t reserved[8];
};
// Every native worker owns one cache line until its native task completes.
struct alignas(64) GateThreadResult {
    uint64_t epoch;
    int32_t cpu;
    int32_t execution_index;
    int32_t returned_status;
    int32_t runtime_status;
    int32_t cleanup_status;
    int32_t initialized;
    int32_t ran;
    int32_t scheduling_error;
    int32_t native_status;
    int32_t dispatch_status;
    uint8_t reserved[16];
};
static_assert(sizeof(Init) == 24 && offsetof(Init, registers) == 8 && offsetof(Init, result) == 16);
static_assert(sizeof(Invocation) == 8 && offsetof(Invocation, reserved) == 4);
static_assert(sizeof(Result) == 64 && offsetof(Result, runtime_status) == 16 && offsetof(Result, opened) == 24);
static_assert(std::is_trivially_copyable_v<Init> && std::is_standard_layout_v<Init>);
static_assert(std::is_trivially_copyable_v<Invocation> && std::is_standard_layout_v<Invocation>);
static_assert(std::is_trivially_copyable_v<Result> && std::is_standard_layout_v<Result>);
static_assert(sizeof(GateInit) == 64 && offsetof(GateInit, launched_threads) == 32);
static_assert(sizeof(GateInvocation) == 8 && offsetof(GateInvocation, reserved) == 4);
static_assert(sizeof(GateSummary) == 64 && offsetof(GateSummary, runtime_status) == 48);
static_assert(sizeof(GateThreadResult) == 64 && offsetof(GateThreadResult, scheduling_error) == 36);
static_assert(offsetof(GateThreadResult, native_status) == 40);
static_assert(offsetof(GateThreadResult, dispatch_status) == 44);
static_assert(std::is_trivially_copyable_v<GateInit> && std::is_standard_layout_v<GateInit>);
static_assert(std::is_trivially_copyable_v<GateInvocation> && std::is_standard_layout_v<GateInvocation>);
static_assert(std::is_trivially_copyable_v<GateSummary> && std::is_standard_layout_v<GateSummary>);
static_assert(std::is_trivially_copyable_v<GateThreadResult> && std::is_standard_layout_v<GateThreadResult>);
}  // namespace control_probe
