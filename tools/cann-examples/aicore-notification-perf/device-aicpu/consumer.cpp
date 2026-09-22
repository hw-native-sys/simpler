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
// Each control write is followed by a clean of its cache line. The producer
// reads those words after invalidating its own copy of the line, so a value
// still resident in the AICPU is a value the producer cannot see.
//
// Both producer and consumer read the same shared system counter on
// a3 / a5, so (t_obs - tw) latency subtraction is well-defined.
//
// `WaitForChange` is bounded (~1 s deadline) so a wedged producer (wrong
// core / unbuilt / mode race) produces a clean -1 + a logged warning
// instead of hanging the stream-sync indefinitely. The producer carries the
// mirror-image budgets, so neither side can outlive the other's failure.

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

// Bounded wait for the producer to publish its core id, sharing the sampling
// timeout: a producer that never launched then reports a fallback instead of
// holding the AICPU op until the op-execute timeout kills aicpu-sd.
constexpr uintptr_t kNotifPerfCacheLine = 64;

// The AICPU writes core_id_valid=0 itself, so its own copy of that line is
// valid and an AICore write to the same line does not reach a plain re-read.
// A spin-poll on an AICore-written field therefore has to invalidate each
// iteration; byte-identical to the runtime's invalidate_range_impl, including
// the trailing completion wait.
static inline void InvalidateLine(const void *addr) {
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(kNotifPerfCacheLine - 1);
    __asm__ __volatile__("dc civac, %0" ::"r"(start) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
}

// Push this core's writes to a line out to GM without invalidating it. Needed
// before releasing the producer: `dc civac` above is clean-AND-invalidate, so a
// line the AICPU left dirty would be written back by the first poll — after the
// producer had already published into it — silently restoring the stale value.
static inline void CleanLine(const void *addr) {
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(kNotifPerfCacheLine - 1);
    __asm__ __volatile__("dc cvac, %0" ::"r"(start) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
}

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
    uint32_t unordered = 0;
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
            // Carried out of the timeout path too: a timeout after an unordered
            // observation would otherwise report zero of them, which is the
            // silent loss this count exists to remove.
            result->gm_unordered = unordered;
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
        } else {
            unordered++;
        }
    }
    result->gm_unordered = unordered;
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
    uint32_t unordered = 0;
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
            result->cond_unordered = unordered;
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
        } else {
            // t_obs at or before tw means the notification arrived carrying a
            // sequence whose tw was not yet visible. Counted rather than
            // dropped: a dropped sample turns a publication-ordering fault into
            // "no valid samples", which reads as a broken harness.
            unordered++;
        }
    }
    result->cond_unordered = unordered;
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
    hank->core_id_valid = 0;
    hank->producer_rc = kNotifPerfProducerOk;
    // Publish the reset before the producer can race it. Without this the line
    // stays dirty in the AICPU, the producer sets core_id_valid=1 in GM, and the
    // first InvalidateLine below writes the stale 0 back over it — losing the
    // report and sending the poll to the fallback core.
    CleanLine(const_cast<const void *>(static_cast<const volatile void *>(&hank->core_id)));
    hank->mode = kNotifPerfModeGm;
    hank->go = 1;
    // Every control write is followed by a publish of its line: the producer
    // reads these words after invalidating its own copy, so a value still held
    // in the AICPU is a value the producer cannot see.
    CleanLine(const_cast<const void *>(static_cast<const volatile void *>(&hank->go)));

    // Which core's COND to poll is the producer's to report: block_dim=1 does
    // not pin it to AIC[0], and polling the wrong core's register yields a
    // register that never changes, i.e. a subtest that times out rather than
    // one that says why. target_core_idx is the fallback for a producer that
    // never reports.
    uint32_t core_id_raw = 0;
    bool have_core_id = false;
    // Bounded in time rather than in iterations: each poll carries a full
    // invalidate (`dc civac; dsb sy; isb`), so an iteration count cannot be
    // turned into a wall-clock budget, and overshooting it trips the
    // op-execute timeout, which kills aicpu-sd and takes the diagnostic with it.
    uint64_t core_id_t0 = SysCntAicpu();
    while (SysCntAicpu() - core_id_t0 < kWaitForChangeTimeoutTicks) {
        InvalidateLine(const_cast<const void *>(static_cast<const volatile void *>(&hank->core_id)));
        if (hank->core_id_valid == 1) {
            core_id_raw = hank->core_id;
            have_core_id = true;
            break;
        }
    }
    if (!have_core_id) {
        DiagLog(kDlogLevelWarn, "producer did not report its core id; using target_core_idx");
        core_id_raw = d->target_core_idx;
    }
    uint32_t core_id = core_id_raw & kNotifPerfCoreIdMask;
    result->core_id_raw = core_id_raw;
    result->core_id_used = core_id;
    result->core_id_from_producer = have_core_id ? 1u : 0u;
    if (core_id >= kNotifPerfMaxCores) {
        // An unmasked get_coreid() has been observed as 0x8010; indexing the
        // AIC_CTRL window with it addresses outside the mapping and faults the
        // AICPU op, which surfaces host-side as 507018 / aicpu exception.
        DiagLog(kDlogLevelWarn, "reported core id out of range for the AIC_CTRL window");
        hank->go = 0;
        CleanLine(const_cast<const void *>(static_cast<const volatile void *>(&hank->go)));
        result->magic = kNotifPerfResultMagic;
        result->consumer_rc = -2;
        return 1;
    }

    // Compute the COND MMIO address for the core the producer is running on.
    uint64_t cond_va =
        d->aic_ctrl_reg_base + static_cast<uint64_t>(core_id) * kNotifPerfCoreStride + kNotifPerfRegSprCondOffset;
    auto *cond_addr = reinterpret_cast<volatile uint32_t *>(cond_va);
    *cond_addr = 0;

    // Give the producer a moment to take the go=1 transition and start
    // emitting at GM mode.
    for (volatile int i = 0; i < 100000; i++) {}

    int gm_rc = SampleGmE2E(hank, n_samples, result);

    // Switch mode without stopping the producer; producer re-reads mode
    // each iter so it picks up COND mode on the next iteration.
    hank->mode = kNotifPerfModeCond;
    CleanLine(const_cast<const void *>(static_cast<const volatile void *>(&hank->go)));
    *cond_addr = 0;                               // clear so first producer COND write is a value-change
    for (volatile int i = 0; i < 100000; i++) {}  // settle to new mode

    int cond_rc = SampleCondE2E(hank, cond_addr, n_samples, result);

    // Stop the producer ONCE, then measure idle LDR rates on quiescent
    // fields.
    hank->go = 0;
    CleanLine(const_cast<const void *>(static_cast<const volatile void *>(&hank->go)));
    for (volatile int i = 0; i < 50000; i++) {}

    MeasureIdleLdrRates(hank, cond_addr, result);

    result->observed_p_seq = hank->p_seq;
    // The producer writes this as it returns, on its own cache line.
    InvalidateLine(const_cast<const void *>(static_cast<const volatile void *>(&hank->producer_rc)));
    result->producer_rc = hank->producer_rc;
    result->magic = kNotifPerfResultMagic;
    // consumer_rc encodes whichever sampling sub-test timed out (if any);
    // 0 = all clean, -1 = at least one E2E subtest hit the WaitForChange
    // deadline (producer wedged / on wrong core / unbuilt).
    result->consumer_rc = (gm_rc != 0 || cond_rc != 0) ? -1 : 0;
    return 0;
}

}  // extern "C"
