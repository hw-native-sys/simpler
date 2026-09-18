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
// consumer.cpp — AICPU half of the FIN-ordering test.
//
// Two exports, the contract the CANN dispatcher expects:
//   simpler_aicpu_init  — no-op here
//   simpler_aicpu_run   — drive the P x C matrix, write records + cells
//
// Per round: request it, wait for this round's FIN, apply the consumer variant
// under test, take the FIRST payload load, classify it, then run diagnostics,
// and only then ACK. The producer holds payload frozen across all of that.
//
// The value that ends the wait IS the value used to derive expected_seq — the
// wait returns it rather than the caller re-reading COND, which would let the
// producer advance in between and make "expect" refer to a different round.

#include <cstdint>
#include <cstring>

#include "../shared/handshake.h"

// ---------------------------------------------------------------------------
// Primitives. Only what this platform actually provides: the AICPU is aarch64,
// and the runtime's own cache maintenance (src/common/platform/onboard/aicpu/
// cache_ops.cpp) is `dc civac` / `dc cvac` followed by `dsb sy; isb`.
// ---------------------------------------------------------------------------

static inline uint64_t SysCnt() {
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

// Full system barrier. Used as the MMIO-read -> GM-load ordering operation for
// the C1/C3 arms: the COND register is Device-nGnRE and payload is Normal
// cacheable, so nothing but an explicit barrier orders the two.
static inline void OrderBarrier() { __asm__ __volatile__("dsb sy" ::: "memory"); }

// The minimal barrier the architecture asks for here: `dmb ld` orders loads
// that precede it against loads and stores that follow. Cheaper than dsb sy,
// which additionally waits for completion rather than only ordering.
static inline void OrderBarrierLd() { __asm__ __volatile__("dmb ld" ::: "memory"); }

// Acquire-load the flag. Puts the ordering on the flag read itself, so nothing
// has to be inserted between the wait loop and the payload load.
static inline uint32_t AcquireLoad32(volatile uint32_t *addr) {
    uint32_t v;
    __asm__ __volatile__("ldar %w0, [%1]" : "=r"(v) : "r"(addr) : "memory");
    return v;
}

// Invalidate, byte-identical to the runtime's invalidate_range_impl. NOTE the
// trailing `dsb sy; isb`: the completion wait is part of this operation, which
// is why C3 differs from C2 only by a LEADING dsb.
static inline void InvalidateLine(const void *addr) {
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(kFinCacheLine) - 1);
    __asm__ __volatile__("dc civac, %0" ::"r"(start) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
}

// Clean this core's writes so the AICore can see them. Only ever applied to the
// ctl block, which the AICPU alone writes.
static inline void CleanLine(const void *addr) {
    uintptr_t start = reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(kFinCacheLine) - 1);
    __asm__ __volatile__("dc cvac, %0" ::"r"(start) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
    __asm__ __volatile__("isb" ::: "memory");
}

extern "C" void DlogRecord(int moduleId, int level, const char *fmt, ...);

namespace {

constexpr int kDlogModule = 3;
constexpr int kDlogInfo = 1;
constexpr int kDlogWarn = 2;

void Diag(int level, const char *msg) { DlogRecord(kDlogModule, level, "[fin-ordering] %s", msg); }

struct KernelArgs {
    uint64_t _pad[5];
    void *device_args;
};

// ~2 ms at a 50 MHz counter; every wait in this test is bounded.
constexpr uint64_t kWaitTimeoutTicks = 100000;
constexpr uint32_t kRereadLimit = 20000;

struct WaitResult {
    bool ok;
    uint32_t value;  // the exact MMIO word that ended the wait
    uint64_t ticks;
};

// Poll COND until the low 31 bits carry `want` with the FIN bit set. Returns the
// causal read; the caller must not re-read COND to derive the round.
WaitResult WaitForFin(volatile uint32_t *cond_addr, uint64_t want, bool use_acquire) {
    const uint32_t target = (static_cast<uint32_t>(want) & kFinTaskIdMask) | kFinTaskStateMask;
    uint64_t t0 = SysCnt();
    while (true) {
        uint32_t v = use_acquire ? AcquireLoad32(cond_addr) : *cond_addr;
        if (v == target) {
            return {true, v, SysCnt() - t0};
        }
        if (SysCnt() - t0 > kWaitTimeoutTicks) {
            return {false, v, SysCnt() - t0};
        }
    }
}

// Apply the consumer variant, then take the first payload load.
uint64_t FirstLoad(volatile FinPayBlock *pay, uint32_t consumer_mode, uint64_t delay_ticks, uint64_t dsb_delay_ticks) {
    switch (consumer_mode) {
    case kFinConsumerC0:
        break;
    case kFinConsumerC1:
        OrderBarrier();
        break;
    case kFinConsumerC2:
        InvalidateLine(const_cast<uint64_t *>(&pay->payload));
        break;
    case kFinConsumerC3:
        OrderBarrier();
        InvalidateLine(const_cast<uint64_t *>(&pay->payload));
        break;
    case kFinConsumerC4: {
        // Timing control matched to the invalidate cost: spend the time without
        // doing anything that could affect visibility, to tell "the maintenance
        // worked" apart from "the extra wait was enough".
        uint64_t t0 = SysCnt();
        while (SysCnt() - t0 < delay_ticks) {}
        break;
    }
    case kFinConsumerC5:
        OrderBarrierLd();
        break;
    case kFinConsumerC6:
        // The ordering was applied to the flag load inside the wait loop, so
        // nothing goes here; the payload load stays bare.
        break;
    case kFinConsumerC7: {
        // Timing control matched to the dsb sy cost, so C1's result can be read
        // as ordering rather than as the extra wait C1 happens to pay.
        uint64_t t0 = SysCnt();
        while (SysCnt() - t0 < dsb_delay_ticks) {}
        break;
    }
    default:
        break;
    }
    return pay->payload;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *) { return 0; }

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_run(void *args) {
    Diag(kDlogInfo, "run entered");
    if (args == nullptr) {
        return 1;
    }
    auto *k = reinterpret_cast<KernelArgs *>(args);
    auto *d = reinterpret_cast<FinDeviceArgs *>(k->device_args);
    if (d == nullptr || d->result_addr == 0 || d->handshake_addr == 0 || d->aic_ctrl_reg_base == 0 ||
        d->records_addr == 0) {
        Diag(kDlogWarn, "device_args missing pointers");
        return 1;
    }

    auto *hank = reinterpret_cast<volatile FinHandshake *>(d->handshake_addr);
    auto *ctl = const_cast<FinCtlBlock *>(&hank->ctl);
    auto *rpt = const_cast<FinRptBlock *>(&hank->rpt);
    auto *pay = const_cast<FinPayBlock *>(&hank->pay);
    auto *result = reinterpret_cast<FinResult *>(d->result_addr);
    auto *records = reinterpret_cast<FinTrialRecord *>(d->records_addr);
    std::memset(result, 0, sizeof(*result));

    // Bisect gate. The second CLI argument arrives as fallback_core_idx and
    // selects how far to run, so a crash can be localised without a rebuild:
    //   1 = return after touching result only
    //   2 = + producer bring-up / core id
    //   3 = + invalidate-cost calibration
    //   0 = the whole matrix
    const uint32_t stage = d->fallback_core_idx;

    result->payload_addr = reinterpret_cast<uint64_t>(&pay->payload);
    if (stage == 1) {
        result->magic = kFinResultMagic;
        result->consumer_rc = 0;
        result->records_written = 0;
        Diag(kDlogInfo, "stage 1 ok");
        return 0;
    }
    uint32_t layout_flags = 0;
    if ((result->payload_addr & (kFinCacheLine - 1)) != 0) {
        layout_flags |= kFinFlagPayloadUnaligned;
        Diag(kDlogWarn, "payload is not 64B aligned; block isolation is not guaranteed");
    }

    // --- Bring-up: start the producer and learn its core id before measuring.
    ctl->round_request = 0;
    ctl->mode = kFinProducerIdle;
    ctl->go = 1;
    CleanLine(ctl);

    uint64_t t0 = SysCnt();
    while (true) {
        InvalidateLine(rpt);
        if (rpt->core_id_valid == 1) {
            break;
        }
        if (SysCnt() - t0 > kWaitTimeoutTicks * 20) {
            Diag(kDlogWarn, "producer never reported its core id");
            result->magic = kFinResultMagic;
            result->consumer_rc = -2;
            ctl->go = 0;
            CleanLine(ctl);
            return 0;
        }
    }
    // get_coreid() carries more than the core index; the runtime masks it with
    // AICORE_COREID_MASK (0x0FFF) before indexing the AIC_CTRL window. Using the
    // raw value produces an address outside that window, and reading it faults
    // the AICPU op ("aicpu exception") rather than merely failing to match.
    uint32_t core_id_raw = rpt->core_id;
    uint32_t core_id = core_id_raw & kFinCoreIdMask;
    result->core_id_raw = core_id_raw;
    result->core_id_used = core_id;
    if (core_id >= kFinMaxCores) {
        Diag(kDlogWarn, "reported core id is outside the AIC_CTRL window; refusing to poll");
        result->magic = kFinResultMagic;
        result->consumer_rc = -4;
        ctl->go = 0;
        CleanLine(ctl);
        return 0;
    }
    uint64_t cond_va = d->aic_ctrl_reg_base + static_cast<uint64_t>(core_id) * kFinCoreStride + kFinRegSprCondOffset;
    auto *cond_addr = reinterpret_cast<volatile uint32_t *>(cond_va);
    result->cond_addr = cond_va;
    if (stage == 2) {
        result->magic = kFinResultMagic;
        result->consumer_rc = 0;
        result->records_written = 0;
        ctl->go = 0;
        CleanLine(ctl);
        Diag(kDlogInfo, "stage 2 ok");
        return 0;
    }

    // --- Calibrate the invalidate cost so the C4 delay arm is comparable.
    {
        constexpr int kCal = 2000;
        uint64_t c0 = SysCnt();
        for (int i = 0; i < kCal; i++) {
            InvalidateLine(const_cast<uint64_t *>(&pay->payload));
        }
        uint64_t total = SysCnt() - c0;
        result->invalidate_cost_ticks = total / kCal;
        if (result->invalidate_cost_ticks == 0) {
            result->invalidate_cost_ticks = 1;
        }
    }
    {
        constexpr int kCal = 2000;
        uint64_t c0 = SysCnt();
        for (int i = 0; i < kCal; i++) {
            OrderBarrier();
        }
        uint64_t total = SysCnt() - c0;
        result->dsb_cost_ticks = total / kCal;
        if (result->dsb_cost_ticks == 0) {
            result->dsb_cost_ticks = 1;
        }
    }
    const uint64_t delay_ticks = result->invalidate_cost_ticks;
    const uint64_t dsb_delay_ticks = result->dsb_cost_ticks;
    if (stage == 3) {
        result->magic = kFinResultMagic;
        result->consumer_rc = 0;
        result->records_written = 0;
        ctl->go = 0;
        CleanLine(ctl);
        Diag(kDlogInfo, "stage 3 ok");
        return 0;
    }

    // --- Matrix. Interleave the cells so drift affects all of them alike.
    const uint32_t rounds = d->rounds_per_cell > 0 ? d->rounds_per_cell : 50;
    const uint32_t producer_modes[kFinProducerModeCount] = {kFinProducerP0, kFinProducerP1};
    uint64_t round = 0;
    uint32_t written = 0;
    uint32_t rc = 0;

    for (uint32_t rep = 0; rep < rounds; rep++) {
        for (uint32_t pi = 0; pi < kFinProducerModeCount; pi++) {
            for (uint32_t cm = 0; cm < kFinConsumerModeCount; cm++) {
                for (uint32_t pf = 0; pf < kFinPrefetchVariants; pf++) {
                    round++;
                    if (round >= kFinMaxRound) {
                        rc = -3;
                        goto done;
                    }

                    // Producer mode for this round, then request the round. Mode
                    // must be visible before the request it applies to.
                    ctl->mode = producer_modes[pi];
                    CleanLine(ctl);
                    if (pf == 1) {
                        // Prefetch arm: deliberately pull the payload line in
                        // with its pre-publish content, so this core provably
                        // holds a stale copy going into the round. Read-only, so
                        // no dirty copy can clobber the producer.
                        volatile uint64_t sink = pay->payload;
                        (void)sink;
                    }
                    ctl->round_request = round;
                    CleanLine(ctl);

                    WaitResult w = WaitForFin(cond_addr, round, cm == kFinConsumerC6);

                    FinTrialRecord rec;
                    std::memset(&rec, 0, sizeof(rec));
                    rec.trial_id = static_cast<uint32_t>(round);
                    rec.producer_mode = producer_modes[pi];
                    rec.consumer_mode = cm;
                    rec.prefetched = pf;
                    rec.raw_cond = w.value;
                    rec.expected_seq = round;
                    rec.flags = layout_flags;
                    rec.ticks_to_first_match = UINT64_MAX;

                    if (!w.ok) {
                        rec.flags |= kFinFlagWaitTimeout;
                        rec.first_read = kFinFirstOlder;
                        rec.first_payload = pay->payload;
                    } else {
                        // Derive the round from the causal read, not a re-read.
                        uint64_t seq_from_cond = w.value & kFinTaskIdMask;
                        rec.expected_seq = seq_from_cond;

                        uint64_t m0 = SysCnt();
                        uint64_t got = FirstLoad(pay, cm, delay_ticks, dsb_delay_ticks);
                        rec.first_payload = got;
                        if (got == seq_from_cond) {
                            rec.first_read = kFinFirstEqual;
                            rec.ticks_to_first_match = SysCnt() - m0;
                        } else if (got < seq_from_cond) {
                            rec.first_read = kFinFirstOlder;
                        } else {
                            rec.first_read = kFinFirstNewer;
                        }

                        // Diagnostics. These never overwrite the first-read
                        // classification above.
                        if (rec.first_read != kFinFirstEqual) {
                            uint32_t iters = 0;
                            uint64_t v = got;
                            while (iters < kRereadLimit && v != seq_from_cond) {
                                v = pay->payload;
                                iters++;
                            }
                            rec.reread_iters = iters;
                            rec.ordinary_reread_payload = v;
                            if (v != seq_from_cond) {
                                rec.flags |= kFinFlagRereadTimeout;
                                InvalidateLine(const_cast<uint64_t *>(&pay->payload));
                                rec.post_invalidate_payload = pay->payload;
                            } else {
                                rec.post_invalidate_payload = v;
                            }
                        } else {
                            rec.ordinary_reread_payload = got;
                            rec.post_invalidate_payload = got;
                        }

                        // Round echo: the producer publishes published_round
                        // after the FIN, so this is a protocol cross-check only.
                        InvalidateLine(rpt);
                        if (rpt->published_round != seq_from_cond) {
                            rec.flags |= kFinFlagRoundEchoMismatch;
                        }
                    }

                    if (written < d->records_capacity) {
                        records[written] = rec;
                        written++;
                    }

                    uint32_t ci = fin_cell_index(pi, cm, pf);
                    FinCell &cell = result->cells[ci];
                    cell.trials++;
                    if (rec.flags & kFinFlagWaitTimeout) {
                        cell.wait_timeouts++;
                    } else if (rec.first_read == kFinFirstEqual) {
                        cell.first_equal++;
                        cell.ticks_sum += rec.ticks_to_first_match;
                        if (rec.ticks_to_first_match > cell.ticks_max) {
                            cell.ticks_max = rec.ticks_to_first_match;
                        }
                    } else {
                        if (rec.first_read == kFinFirstOlder) {
                            cell.first_older++;
                        } else {
                            cell.first_newer++;
                        }
                        if (rec.ordinary_reread_payload == rec.expected_seq) {
                            cell.reread_matched++;
                        } else if (rec.post_invalidate_payload == rec.expected_seq) {
                            cell.invalidate_matched++;
                        } else {
                            cell.unresolved++;
                        }
                    }

                    // ACK: only now may the producer move on.
                    ctl->round_request = round + 1;
                    CleanLine(ctl);
                }
            }
        }
    }

done:
    ctl->go = 0;
    CleanLine(ctl);

    InvalidateLine(rpt);
    InvalidateLine(const_cast<uint64_t *>(&pay->payload));
    result->producer_publishes = rpt->publish_count;
    result->producer_last_round = rpt->published_round;
    result->final_payload = pay->payload;

    result->records_written = written;
    result->cells_valid = kFinProducerModeCount * kFinConsumerModeCount * kFinPrefetchVariants;
    result->delay_arm_ticks = delay_ticks;
    result->magic = kFinResultMagic;
    result->consumer_rc = static_cast<int32_t>(rc);
    Diag(kDlogInfo, "matrix done");
    return 0;
}
