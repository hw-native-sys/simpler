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
// Cache-line alignment (64 B) avoids false sharing between the producer's
// p_seq + p_tw writes and the host-readable result block.

#ifndef AICORE_NOTIFICATION_PERF_HANDSHAKE_H_
#define AICORE_NOTIFICATION_PERF_HANDSHAKE_H_

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

// Producer mode selector — AICPU consumer writes `mode` before flipping
// `go = 1`; producer reads it once per inner-loop iteration.
enum NotifPerfMode : uint32_t {
    kNotifPerfModeGm = 0,    // p_seq + dcci pattern
    kNotifPerfModeCond = 1,  // set_cond(MAKE_FIN_VALUE(seq)) pattern
};

// Live handshake. Lives in GM; AICPU consumer host_main allocates one
// instance via rtMalloc and passes the device pointer in DeviceArgs.
struct alignas(64) NotifPerfHandshake {
    // --- AICPU → AICore control ---
    volatile uint32_t go;             // 0 = stop, 1 = run
    volatile uint32_t mode;           // see NotifPerfMode
    volatile uint32_t throttle_iter;  // tight-spin count inside producer per iter (~50 ≈ 1 µs)
    volatile uint32_t _pad0;

    // --- AICore → AICPU data ---
    volatile uint64_t p_seq;  // monotonic counter; AICore writes; AICPU polls in GM mode
    volatile uint64_t p_tw;   // AICore-side sys_cnt captured *before* the publishing op

    // --- AICPU keeps its own state on the next cache line (not strictly
    //     required, but makes it obvious where the AICore-touched bytes end).
    uint64_t _pad1[6];
};
static_assert(sizeof(NotifPerfHandshake) == 128, "NotifPerfHandshake layout must match across compilers");

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
