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
// shared/handshake.h — shared types between the three programs in this tool:
//
//   host launcher  (host_main / launch.cpp)         x86-64 or aarch64 host
//   AICPU consumer (consumer.cpp)                   AICPU OS, aarch64
//   AICore producer (producer.cce)                  AICore, dav-c220
//
// All three see the same GM-resident `NotifPerfHandshake` and `NotifPerfResult`
// at known addresses negotiated through DeviceArgs. The layout must be
// identical across compilers — keep the struct fields in the order below and
// avoid platform-conditional padding.
//
// `NotifPerfHandshake` is partitioned one writer per 64 B cache line; the
// comment on the struct says why that is a correctness requirement and not
// merely false-sharing hygiene.

#ifndef AICORE_NOTIFICATION_PERF_HANDSHAKE_H_
#define AICORE_NOTIFICATION_PERF_HANDSHAKE_H_

#include <cstddef>
#include <cstdint>

// REG_SPR_COND_OFFSET on a2a3 chip family. The AICPU consumer reads
// `reg_addr_base + core_stride * core_idx + REG_SPR_COND_OFFSET` to poll
// AICore's COND register. Mirrors src/a2a3/platform/include/common/platform_config.h.
constexpr uint32_t kNotifPerfRegSprCondOffset = 0x4C8;

// Per-core stride within the AIC_CTRL window — only the first sub-core (AIC,
// i.e. CUBE) slot is used here, since the producer runs with block_dim=1.
// Mirrors host_regs.cpp::core_stride. AICPU consumer only needs the AIC[0]
// COND register; a future extension that drives multiple cores would
// recompute this per core_idx.
constexpr uint64_t kNotifPerfCoreStride = 8ULL * 1024 * 1024;

// FIN encoding — AICore producer writes COND with `MAKE_FIN_VALUE(seq)`
// because production set_cond paths exercise this bit pattern; bit-31 = 1
// keeps the value distinguishable from the AICPU-side IDLE / clear value
// (0x7FFFFFFD with bit 31 = 0). Mirrors platform_config.h.
constexpr uint32_t kNotifPerfTaskIdMask = 0x7FFFFFFFu;
constexpr uint32_t kNotifPerfTaskStateMask = 0x80000000u;

// get_coreid() is not a bare index: the runtime masks it with AICORE_COREID_MASK
// before indexing the AIC_CTRL window (0x8010 has been observed raw), and an
// unmasked value addresses outside the window and faults the AICPU op
// (src/a2a3/platform/onboard/aicore/inner_kernel.h::get_physical_core_id).
constexpr uint32_t kNotifPerfCoreIdMask = 0x0FFF;
constexpr uint32_t kNotifPerfMaxCores = 64;

// Producer mode selector — AICPU consumer writes `mode` before flipping
// `go = 1`; producer reads it once per inner-loop iteration.
enum NotifPerfMode : uint32_t {
    kNotifPerfModeGm = 0,    // p_seq + dcci pattern
    kNotifPerfModeCond = 1,  // set_cond(MAKE_FIN_VALUE(seq)) pattern
};

// How the producer left its loop. Lives on the producer's own cache line, so
// the consumer can distinguish "never released" from "ran out of budget" from
// a clean stop.
enum NotifPerfProducerRc : uint32_t {
    kNotifPerfProducerOk = 0,          // saw go = 0 and returned
    kNotifPerfProducerGoTimeout = 1,   // go never became 1 within the wait budget
    kNotifPerfProducerRunTimeout = 2,  // go never became 0 within the run budget
};

// Producer budgets, in ticks of the shared 50 MHz system counter. The consumer's
// whole lifetime is bounded by two ~1 s sampling deadlines plus setup, so these
// are far above any healthy run; they exist so a consumer that dies, never
// launches, or misses its stop leaves a kernel that ends and reports rather than
// a stream the host waits on forever.
constexpr uint64_t kNotifPerfProducerGoWaitTicks = 250ULL * 1000 * 1000;  // ~5 s
constexpr uint64_t kNotifPerfProducerRunTicks = 500ULL * 1000 * 1000;     // ~10 s

// Live handshake. Lives in GM; AICPU consumer host_main allocates one
// instance via rtMalloc and passes the device pointer in DeviceArgs.
//
// One writer per cache line, and that partition is load-bearing rather than
// tidiness. The producer publishes with `dcci(..., CACHELINE_OUT)`, which writes
// a whole 64 B line back from a cache the AICore's GM reads do not keep
// coherent — so any AICPU-owned word sharing a line with a producer-written
// field is restored to the producer's stale copy on every publish. With `go` and
// `p_seq` on one line the run therefore ended by chance: a `go = 0` landing
// between the producer's invalidate and its write-back was undone, and the
// producer's `while (go != 0)` loop never exited. Measured as a hang whose rate
// tracks the width of that window — 3 of 6 runs at throttle_iter=50, 0 of 6 at
// 5000, 6 of 6 failing at 0.
struct alignas(64) NotifPerfHandshake {
    // --- line 0: AICPU → AICore control. The producer invalidates and reads
    // this line; it must never appear in a CACHELINE_OUT. ---
    volatile uint32_t go;             // 0 = stop, 1 = run
    volatile uint32_t mode;           // see NotifPerfMode
    volatile uint32_t throttle_iter;  // tight-spin count inside producer per iter (~50 ≈ 1 µs)
    volatile uint32_t _pad0;
    uint64_t _pad1[6];

    // --- line 1: AICore → AICPU sample data, published as one line ---
    volatile uint64_t p_seq;  // monotonic counter; AICore writes; AICPU polls in GM mode
    volatile uint64_t p_tw;   // AICore-side sys_cnt captured *before* the publishing op
    uint64_t _pad2[6];

    // --- line 2: AICore → AICPU, each written once ---
    // The producer publishes which core it landed on, because block_dim=1 does
    // not pin it to AIC[0] and the consumer polls one core's COND register.
    // Raw, as get_coreid() returned it; the consumer masks and range-checks.
    volatile uint32_t core_id;        // get_coreid() raw, unmasked
    volatile uint32_t core_id_valid;  // 1 once core_id is published
    volatile uint32_t producer_rc;    // NotifPerfProducerRc, written as the producer returns
    volatile uint32_t _pad3;
    uint64_t _pad4[6];
};
static_assert(sizeof(NotifPerfHandshake) == 192, "NotifPerfHandshake layout must match across compilers");
static_assert(offsetof(NotifPerfHandshake, go) == 0, "AICPU-owned control must start line 0");
static_assert(offsetof(NotifPerfHandshake, p_seq) == 64, "producer sample data must start line 1");
static_assert(offsetof(NotifPerfHandshake, core_id) == 128, "producer report must start line 2");

// Result block — AICPU consumer writes after each subtest. Host D2H reads it
// and prints the summary table.
struct alignas(64) NotifPerfResult {
    // Subtest M (GM path): AICore writes p_seq + dcci, AICPU polls p_seq.
    uint64_t gm_samples;
    uint64_t gm_sum_ticks;
    uint64_t gm_min_ticks;
    uint64_t gm_max_ticks;

    // Subtest C (COND path): AICore writes set_cond, AICPU polls *cond_addr.
    uint64_t cond_samples;
    uint64_t cond_sum_ticks;
    uint64_t cond_min_ticks;
    uint64_t cond_max_ticks;

    // Phase 13 supplemental — AICPU polling-rate readings (10000 LDRs):
    //   gm_ldr_ticks_total   — same-field GM LDR ×10000
    //   cond_ldr_ticks_total — same-COND LDR ×10000
    uint64_t gm_ldr_ticks_total;
    uint64_t cond_ldr_ticks_total;

    // Diagnostics
    uint32_t magic;           // 0xC0DE_CAFE on success
    int32_t consumer_rc;      // AICPU consumer top-level return
    uint64_t observed_p_seq;  // final AICore counter at end of run (sanity)

    // Samples whose t_obs landed at or before the paired tw, i.e. the
    // notification was observed before its timestamp was. Kept separate from
    // *_samples so a publication-ordering fault reports as a count rather than
    // as an empty subtest.
    uint32_t gm_unordered;
    uint32_t cond_unordered;

    // What the producer reported and what actually indexed the COND window.
    uint32_t core_id_raw;
    uint32_t core_id_used;
    // 1 = the producer published it, 0 = the wait timed out and target_core_idx
    // was used. Without this a reported id of 0 is ambiguous between "the
    // producer is on core 0" and "nothing was reported, and the default is 0".
    uint32_t core_id_from_producer;
    // NotifPerfProducerRc as the producer left it. Non-zero means the producer
    // ended on its own budget rather than on the consumer's stop, which is the
    // difference between "the measurement is short" and "the handshake broke".
    uint32_t producer_rc;
};
static_assert(sizeof(NotifPerfResult) <= 192, "Keep result block small for fast D2H");

constexpr uint32_t kNotifPerfResultMagic = 0xC0DECAFEu;

// DeviceArgs envelope — host writes, AICPU consumer reads. Same layout the
// aicpu-kernel-launch tool uses for `DeviceArgs.result_addr` and
// `DeviceArgs.input_token`; we extend with three pointer fields.
struct alignas(8) NotifPerfDeviceArgs {
    uint64_t reserved_pre[12];   // 0..95 — dispatcher bootstrap uses these
    uint64_t result_addr;        // 96 — &NotifPerfResult
    uint64_t input_token;        // 104 — echoed for sanity
    uint64_t handshake_addr;     // 112 — &NotifPerfHandshake
    uint64_t aic_ctrl_reg_base;  // 120 — halMemCtl(REG_AIC_CTRL).ptr
    uint32_t target_core_idx;    // 128 — which AIC core's COND to poll
    uint32_t n_samples;          // 132 — per subtest, default 100
};

#endif  // AICORE_NOTIFICATION_PERF_HANDSHAKE_H_
