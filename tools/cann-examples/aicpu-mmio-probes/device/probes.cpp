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
// probes.cpp — AICPU SO that runs MMIO micro-benchmarks against the chip's
// AIC_CTRL window. AICore is NOT involved — every measurement is "what
// happens when the AICPU CPU issues a load / store at this MMIO address".
//
// Subtests executed in order:
//
//   Phase 4-burst  — N STRs at DMB → measure per-STR posted-write rate.
//                    Burst expected ~5 ns/STR on a3 (Device-nGnRE E bit).
//   Phase 4-rtt    — STR + LDR at DMB → drain the in-flight queue; result is
//                    the full bus round-trip time (~250-300 ns).
//   Phase 12-A     — 10000 LDR COND at one core. Single-thread same-target
//                    cost (~95 ns/LDR on a3, nR forbids outstanding).
//   Phase 12-B     — 10000 LDR COND rotating across N AIC cores from a single
//                    thread. Same per-LDR cost as A — switching target is free
//                    but does not buy outstanding parallelism.
//   Phase 12-C     — M ∈ [1..min(N, kProbeMaxConcurrentReaders)] threads,
//                    each LDR-spinning a distinct core's COND for 10000 iter.
//                    Per-thread cost should stay ~95 ns regardless of M
//                    (parallel scaling proves the nGnRE LDR bus is per-target,
//                    not chip-shared).

#include <cstdint>
#include <cstring>
#include <pthread.h>
#include <unistd.h>

#include "../shared/probes_types.h"

static inline uint64_t SysCntAicpu() {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);
namespace {
constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelInfo = 1;
constexpr int kDlogLevelWarn = 2;

void DiagLog(int level, const char *msg) { DlogRecord(kDlogModuleCcecpu, level, "[aicpu-mmio-probes] %s", msg); }

struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

// Compute MMIO virtual address for a given (core_idx, sub_core=0, reg_offset).
// Mirrors the indexing in src/{arch}/platform/onboard/host/host_regs.cpp.
inline uint64_t RegAddr(uint64_t base, uint32_t core_idx, uint32_t reg_off) {
    return base + core_idx * kProbeCoreStride + reg_off;
}

constexpr int kStrBurstN = 1000;
constexpr int kLdrN = 10000;

// --- Phase 4 STR DMB cost ---
void RunPhase4(uint64_t base, MmioProbeResult *result) {
    volatile uint32_t *dmb0 = reinterpret_cast<volatile uint32_t *>(RegAddr(base, 0, kProbeRegSprDmbOffset));

    // Burst: N STRs, no intervening LDR. Bandwidth-limited (posted writes
    // pipeline; nR ordering does not block STR-to-STR).
    uint64_t t0 = SysCntAicpu();
    for (int j = 0; j < kStrBurstN; j++) {
        *dmb0 = kProbeAicpuIdleTaskId;
    }
    uint64_t t1 = SysCntAicpu();
    result->str_burst_n = kStrBurstN;
    result->str_burst_total_ticks = t1 - t0;

    // Round trip: pair a STR with the LDR that drains it. The LDR is
    // strict-ordered nR so it cannot overlap the in-flight STR.
    volatile uint32_t sink_w = 0;
    uint64_t r0 = SysCntAicpu();
    *dmb0 = kProbeAicpuIdleTaskId;
    sink_w = *dmb0;
    uint64_t r1 = SysCntAicpu();
    (void)sink_w;
    result->str_lat_round_trip = r1 - r0;

    DiagLog(kDlogLevelInfo, "Phase 4 done");
}

// --- Phase 12-A + 12-B (single thread) ---
void RunPhase12SingleThread(uint64_t base, uint32_t n_cores, MmioProbeResult *result) {
    // 12-A: same core LDR loop.
    volatile uint32_t *cond0 = reinterpret_cast<volatile uint32_t *>(RegAddr(base, 0, kProbeRegSprCondOffset));
    {
        volatile uint32_t sink = 0;
        uint64_t t0 = SysCntAicpu();
        for (int j = 0; j < kLdrN; j++) {
            sink = *cond0;
        }
        uint64_t t1 = SysCntAicpu();
        (void)sink;
        result->ldr_n = kLdrN;
        result->ldr_a_total_ticks = t1 - t0;
    }

    // 12-B: rotate across n_cores cores. nR still means strict in-order issue
    // per LDR; same target or rotating does not change the per-LDR cost,
    // because the LDR drain happens on the issuing CPU, not at the slave.
    if (n_cores >= 2) {
        // Build COND ptr table for cores 0..n_cores-1.
        volatile uint32_t *conds[kProbeMaxCores];
        for (uint32_t i = 0; i < n_cores && i < kProbeMaxCores; i++) {
            conds[i] = reinterpret_cast<volatile uint32_t *>(RegAddr(base, i, kProbeRegSprCondOffset));
        }
        uint32_t modulus = (n_cores > kProbeMaxCores) ? kProbeMaxCores : n_cores;
        volatile uint32_t sink = 0;
        uint64_t t0 = SysCntAicpu();
        for (int j = 0; j < kLdrN; j++) {
            sink = *conds[j % modulus];
        }
        uint64_t t1 = SysCntAicpu();
        (void)sink;
        result->ldr_b_total_ticks = t1 - t0;
        result->probed_cores = modulus;
    } else {
        result->ldr_b_total_ticks = 0;
        result->probed_cores = n_cores;
    }
    DiagLog(kDlogLevelInfo, "Phase 12 single-thread done");
}

// --- Phase 12-C (multi-thread) ---
struct ReaderArg {
    volatile uint32_t *reg;
    volatile int32_t *go;  // 0 = wait, 1 = start
    uint64_t result_ticks;
};

void *ReaderThreadFn(void *arg) {
    ReaderArg *a = static_cast<ReaderArg *>(arg);
    while (__atomic_load_n(a->go, __ATOMIC_ACQUIRE) == 0) {
        // tight spin until the host issues "go"
        __asm__ volatile("yield" ::: "memory");
    }
    volatile uint32_t sink = 0;
    uint64_t t0 = SysCntAicpu();
    for (int j = 0; j < kLdrN; j++) {
        sink = *a->reg;
    }
    uint64_t t1 = SysCntAicpu();
    (void)sink;
    a->result_ticks = t1 - t0;
    return nullptr;
}

void RunPhase12MultiThread(uint64_t base, uint32_t n_cores, MmioProbeResult *result) {
    if (n_cores < 1) {
        DiagLog(kDlogLevelWarn, "Phase 12-C skipped: no cores available");
        return;
    }
    uint32_t max_m = n_cores < kProbeMaxConcurrentReaders ? n_cores : kProbeMaxConcurrentReaders;

    for (uint32_t M = 1; M <= max_m; M++) {
        volatile int32_t go = 0;
        ReaderArg args[kProbeMaxConcurrentReaders] = {};
        pthread_t tids[kProbeMaxConcurrentReaders] = {};
        for (uint32_t r = 0; r < M; r++) {
            args[r].reg = reinterpret_cast<volatile uint32_t *>(RegAddr(base, r, kProbeRegSprCondOffset));
            args[r].go = &go;
            args[r].result_ticks = 0;
            int rc = pthread_create(&tids[r], nullptr, ReaderThreadFn, &args[r]);
            if (rc != 0) {
                DiagLog(kDlogLevelWarn, "pthread_create failed");
                // Best-effort: still join what we did spawn.
                __atomic_store_n(&go, 1, __ATOMIC_RELEASE);
                for (uint32_t k = 0; k < r; k++)
                    pthread_join(tids[k], nullptr);
                return;
            }
        }
        // Let all threads park on the spin loop, then release together.
        for (volatile int spin = 0; spin < 500000; spin++) {}
        __atomic_store_n(&go, 1, __ATOMIC_RELEASE);
        for (uint32_t r = 0; r < M; r++)
            pthread_join(tids[r], nullptr);

        // Record into result.ldr_c_thread_ticks[M-1][r] for r in [0..M-1].
        for (uint32_t r = 0; r < M; r++) {
            result->ldr_c_thread_ticks[M - 1][r] = args[r].result_ticks;
        }
    }
    DiagLog(kDlogLevelInfo, "Phase 12 multi-thread done");
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    (void)args;
    return 0;
}

__attribute__((visibility("default"))) int simpler_aicpu_run(void *args) {
    DiagLog(kDlogLevelInfo, "simpler_aicpu_run entered");
    if (args == nullptr) {
        DiagLog(kDlogLevelWarn, "args==nullptr");
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<MmioProbeDeviceArgs *>(k->device_args);
    if (d == nullptr || d->result_addr == 0 || d->aic_ctrl_reg_base == 0) {
        DiagLog(kDlogLevelWarn, "device_args missing critical pointers");
        return 1;
    }
    auto *result = reinterpret_cast<MmioProbeResult *>(d->result_addr);
    std::memset(result, 0, sizeof(*result));

    uint64_t base = d->aic_ctrl_reg_base;
    uint32_t n_cores = d->n_aic_cores_available;
    if (n_cores > kProbeMaxCores) n_cores = kProbeMaxCores;

    RunPhase4(base, result);
    RunPhase12SingleThread(base, n_cores, result);
    RunPhase12MultiThread(base, n_cores, result);

    result->observed_pid = static_cast<uint64_t>(getpid());
    result->magic = kMmioProbeResultMagic;
    result->probe_rc = 0;
    return 0;
}

}  // extern "C"
