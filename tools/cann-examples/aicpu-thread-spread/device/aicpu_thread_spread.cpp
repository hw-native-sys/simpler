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
//
// aicpu_thread_spread — diagnostic inner SO that records what cpu_id CANN
// dispatches each AICPU thread to under a given launch budget.
//
// Per-launch protocol (host side):
//   1. zero the GM output region
//   2. launch simpler_aicpu_init (1 thread) — resets the static counter
//   3. launch simpler_aicpu_spread (N threads) — each writes one slot
//   4. sync + read back
//
// Static atomics work in the AICPU process (same pattern as production
// aicpu_executor.cpp's thread_idx_); GM-side atomics from device glibc
// appear to trap, so we use the static-counter route.

#include <sched.h>

#include <atomic>
#include <cstdint>

namespace {

struct DeviceArgs {
    uint64_t reserved_pre[12];  // 0..95 — unused on this path
    uint64_t output_addr;       // 96
    uint64_t max_slots;         // 104
};

struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

#pragma pack(push, 4)
struct SpreadRecord {
    int32_t thread_idx;
    int32_t cpu_id;
};
struct SpreadOutput {
    uint32_t claim_counter;  // host zeroes before launch; threads bump on write
    uint32_t _pad;
    SpreadRecord records[1];  // capacity = max_slots
};
#pragma pack(pop)
static_assert(sizeof(SpreadRecord) == 8, "SpreadRecord size drift");

extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);
constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelError = 3;
void DiagLog(const char *msg) { DlogRecord(kDlogModuleCcecpu, kDlogLevelError, "[aicpu-spread] %s", msg); }

// Per-launch static counter. Reset by simpler_aicpu_init before each spread
// launch. Lives in AICPU process memory — safe for atomic ops, unlike GM.
std::atomic<uint32_t> s_claim{0};

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    (void)args;
    s_claim.store(0, std::memory_order_release);
    return 0;
}

__attribute__((visibility("default"))) int simpler_aicpu_spread(void *args) {
    if (args == nullptr) {
        DiagLog("simpler_aicpu_spread: args==nullptr");
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<DeviceArgs *>(k->device_args);
    if (d == nullptr || d->output_addr == 0 || d->max_slots == 0) {
        DiagLog("simpler_aicpu_spread: missing GM I/O");
        return 1;
    }

    auto *out = reinterpret_cast<SpreadOutput *>(d->output_addr);
    const uint32_t cap = static_cast<uint32_t>(d->max_slots);

    int cpu = sched_getcpu();

    uint32_t idx = s_claim.fetch_add(1u, std::memory_order_acq_rel);
    if (idx < cap) {
        out->records[idx].thread_idx = static_cast<int32_t>(idx);
        out->records[idx].cpu_id = static_cast<int32_t>(cpu);
        // Host can't atomically read the static counter (it lives in the AICPU
        // process), so each thread bumps the GM-side claim_counter as a
        // best-effort "how many slots to scan" hint. Concurrent stores race,
        // but bounded by launch_count anyway.
        out->claim_counter = idx + 1;
    }
    return 0;
}

}  // extern "C"
