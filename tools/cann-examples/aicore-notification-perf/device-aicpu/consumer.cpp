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
// consumer.cpp — AICPU OS SO. Two exports following the production
// dispatcher convention:
//
//   simpler_aicpu_init  — no-op
//   simpler_aicpu_run   — drive both subtests in sequence, then write results
//
// The consumer assumes the AICore producer is already running on its own
// stream (host launches them concurrently). The producer is idle-spinning
// on `handshake.go == 0`. This consumer takes a single-lifetime contract
// with the producer:
//
//   1. flip `go = 1` ONCE at the start
//   2. set mode = GM, sample E2E latency
//   3. switch mode = COND mid-flight (producer re-reads mode each iter)
//   4. sample COND E2E latency
//   5. flip `go = 0` ONCE at the very end
//   6. measure idle LDR rates on the now-quiescent fields
//
// Bouncing `go` between subtests would let the producer exit permanently
// (its outer `while (go != 0)` has no re-entry path) and deadlock the
// next subtest's wait. The single-lifetime contract avoids that.
//
// Both producer and consumer read the same shared system counter on
// a3 / a5, so (t_obs - tw) latency subtraction is well-defined.
//
// `WaitForChange` is bounded (~1 s deadline) so a wedged producer (wrong
// core / unbuilt / mode race) produces a clean -1 + a logged warning
// instead of hanging the stream-sync indefinitely.

#include <cstdint>
#include <cstring>

#include "../shared/handshake.h"

// Shared system counter on aarch64. Same primitive simpler's runtime uses.
static inline uint64_t SysCntAicpu() {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

// CANN ships a device-side logger as a weak symbol — DlogRecord lands in
// the CANN device log (visible via msnpureport / plog).
extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);
namespace {
constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelInfo = 1;
constexpr int kDlogLevelWarn = 2;

void DiagLog(int level, const char *msg) { DlogRecord(kDlogModuleCcecpu, level, "[notif-perf-consumer] %s", msg); }

// KernelArgs envelope CANN uses for AICPU dispatch.
struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

// Bounded wait. Polls until `poll()` returns false (= value changed),
// or until `timeout_ticks` of the AICPU sys counter elapse. Returns 0
// on success with t_obs written; non-zero on timeout (caller propagates
// via result->consumer_rc).
//
// Default timeout is ~1 s at 50 MHz, well above the per-sample budget
// of a few µs (producer throttled to ~1 µs/iter, samples<=100). A
// timeout here means the producer is wedged / not running / on the
// wrong core — never a tight-loop race.
constexpr uint64_t kWaitForChangeTimeoutTicks = 50ULL * 1000 * 1000;  // ~1 s

template <typename PollFn>
inline int WaitForChange(PollFn poll, uint64_t *t_obs_out) {
    uint64_t start = SysCntAicpu();
    while (poll()) {
        if (SysCntAicpu() - start > kWaitForChangeTimeoutTicks) {
            return -1;
        }
    }
    *t_obs_out = SysCntAicpu();
    return 0;
}

// GM-mode E2E sampling. Caller has already set mode and started the
// producer; this function only reads. Returns 0 on success, -1 on
// timeout (producer wedged); caller decides whether to bail or
// continue with a partial result.
int SampleGmE2E(volatile NotifPerfHandshake *hank, uint32_t n_samples, NotifPerfResult *result) {
    uint64_t sum = 0;
    uint64_t min_v = UINT64_MAX;
    uint64_t max_v = 0;
    uint32_t taken = 0;
    uint64_t last = hank->p_seq;
    for (uint32_t j = 0; j < n_samples; j++) {
        uint64_t cur_last = last;
        uint64_t t_obs = 0;
        int rc = WaitForChange(
            [&]() {
                return hank->p_seq == cur_last;
            },
            &t_obs
        );
        if (rc != 0) {
            DiagLog(kDlogLevelWarn, "GM E2E sample timeout");
            result->gm_samples = taken;
            result->gm_sum_ticks = sum;
            result->gm_min_ticks = (taken > 0) ? min_v : 0;
            result->gm_max_ticks = max_v;
            return -1;
        }
        // Read the paired tw. Race-tolerance: producer always writes tw
        // before incrementing p_seq + dcci, so by the time we see a new
        // p_seq, the tw on the same cache line is also at-or-past that
        // event's value.
        uint64_t tw = hank->p_tw;
        last = hank->p_seq;
        if (t_obs > tw) {
            uint64_t d = t_obs - tw;
            sum += d;
            if (d < min_v) min_v = d;
            if (d > max_v) max_v = d;
            taken++;
        }
    }
    result->gm_samples = taken;
    result->gm_sum_ticks = sum;
    result->gm_min_ticks = (taken > 0) ? min_v : 0;
    result->gm_max_ticks = max_v;
    DiagLog(kDlogLevelInfo, "GM E2E sampling done");
    return 0;
}

// COND-mode E2E sampling. Same contract as SampleGmE2E.
int SampleCondE2E(
    volatile NotifPerfHandshake *hank, volatile uint32_t *cond_addr, uint32_t n_samples, NotifPerfResult *result
) {
    uint64_t sum = 0;
    uint64_t min_v = UINT64_MAX;
    uint64_t max_v = 0;
    uint32_t taken = 0;
    uint32_t last = *cond_addr;
    for (uint32_t j = 0; j < n_samples; j++) {
        uint32_t cur_last = last;
        uint64_t t_obs = 0;
        int rc = WaitForChange(
            [&]() {
                return *cond_addr == cur_last;
            },
            &t_obs
        );
        if (rc != 0) {
            DiagLog(kDlogLevelWarn, "COND E2E sample timeout");
            result->cond_samples = taken;
            result->cond_sum_ticks = sum;
            result->cond_min_ticks = (taken > 0) ? min_v : 0;
            result->cond_max_ticks = max_v;
            return -1;
        }
        uint64_t tw = hank->p_tw;
        last = *cond_addr;
        if (t_obs > tw) {
            uint64_t d = t_obs - tw;
            sum += d;
            if (d < min_v) min_v = d;
            if (d > max_v) max_v = d;
            taken++;
        }
    }
    result->cond_samples = taken;
    result->cond_sum_ticks = sum;
    result->cond_min_ticks = (taken > 0) ? min_v : 0;
    result->cond_max_ticks = max_v;
    DiagLog(kDlogLevelInfo, "COND E2E sampling done");
    return 0;
}

// Phase 13 supplemental — same-field GM LDR + same-COND-reg LDR rates
// on an idle producer. Caller must have ALREADY stopped the producer
// (hank->go = 0 and a settle window) before invoking; the result
// captures cache-hot LDR cost when the published value is unchanged.
void MeasureIdleLdrRates(volatile NotifPerfHandshake *hank, volatile uint32_t *cond_addr, NotifPerfResult *result) {
    constexpr int kIdleLdrIters = 10000;
    {
        uint64_t t0 = SysCntAicpu();
        volatile uint64_t sink = 0;
        for (int k = 0; k < kIdleLdrIters; k++) {
            sink = hank->p_seq;
        }
        uint64_t t1 = SysCntAicpu();
        (void)sink;
        result->gm_ldr_ticks_total = t1 - t0;
    }
    {
        uint64_t t0 = SysCntAicpu();
        volatile uint32_t sink = 0;
        for (int k = 0; k < kIdleLdrIters; k++) {
            sink = *cond_addr;
        }
        uint64_t t1 = SysCntAicpu();
        (void)sink;
        result->cond_ldr_ticks_total = t1 - t0;
    }
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
    auto *d = reinterpret_cast<NotifPerfDeviceArgs *>(k->device_args);
    if (d == nullptr || d->result_addr == 0 || d->handshake_addr == 0 || d->aic_ctrl_reg_base == 0) {
        DiagLog(kDlogLevelWarn, "device_args missing critical pointers");
        return 1;
    }

    auto *hank = reinterpret_cast<volatile NotifPerfHandshake *>(d->handshake_addr);
    auto *result = reinterpret_cast<NotifPerfResult *>(d->result_addr);
    std::memset(const_cast<NotifPerfResult *>(result), 0, sizeof(*result));

    // Compute the COND MMIO address for the targeted AIC core.
    uint64_t cond_va = d->aic_ctrl_reg_base + d->target_core_idx * kNotifPerfCoreStride + kNotifPerfRegSprCondOffset;
    auto *cond_addr = reinterpret_cast<volatile uint32_t *>(cond_va);

    uint32_t n_samples = d->n_samples > 0 ? d->n_samples : 100;

    // Single-producer-lifetime contract. The producer loops on
    // `hank->mode` every iter, so we switch mode mid-flight; `hank->go`
    // is set to 1 ONCE at the start and back to 0 ONCE at the very end.
    // Toggling go between subtests would let the producer exit
    // permanently (its outer `while (go != 0)` loop has no re-entry),
    // which used to deadlock the COND subtest's wait.
    hank->throttle_iter = 50;
    hank->p_seq = 0;
    hank->p_tw = 0;
    hank->mode = kNotifPerfModeGm;
    *cond_addr = 0;
    hank->go = 1;

    // Give the producer a moment to take the go=1 transition and start
    // emitting at GM mode.
    for (volatile int i = 0; i < 100000; i++) {}

    int gm_rc = SampleGmE2E(hank, n_samples, result);

    // Switch mode without stopping the producer; producer re-reads mode
    // each iter so it picks up COND mode on the next iteration.
    hank->mode = kNotifPerfModeCond;
    *cond_addr = 0;                               // clear so first producer COND write is a value-change
    for (volatile int i = 0; i < 100000; i++) {}  // settle to new mode

    int cond_rc = SampleCondE2E(hank, cond_addr, n_samples, result);

    // Stop the producer ONCE, then measure idle LDR rates on quiescent
    // fields.
    hank->go = 0;
    for (volatile int i = 0; i < 50000; i++) {}

    MeasureIdleLdrRates(hank, cond_addr, result);

    result->observed_p_seq = hank->p_seq;
    result->magic = kNotifPerfResultMagic;
    // consumer_rc encodes whichever sampling sub-test timed out (if any);
    // 0 = all clean, -1 = at least one E2E subtest hit the WaitForChange
    // deadline (producer wedged / on wrong core / unbuilt).
    result->consumer_rc = (gm_rc != 0 || cond_rc != 0) ? -1 : 0;
    return 0;
}

}  // extern "C"
