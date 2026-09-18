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
// shared/handshake.h — shared types for the three programs of this tool:
//
//   host launcher   (launch.cpp)     allocates GM, registers + launches both
//   AICPU consumer  (consumer.cpp)   aarch64, polls COND, reads payload
//   AICore producer (producer.cce)   dav-c220, publishes payload then FIN
//
// The question under test: after the consumer observes this round's FIN, can it
// read a payload that was published before that FIN and is guaranteed not to
// change underneath it?
//
// Answering that needs a FROZEN handoff. A throttle delay is not a
// synchronisation guarantee: the producer would keep overwriting payload, so a
// mismatch could mean "producer moved on" rather than "data late". Here the
// producer writes payload exactly once per round and then must not touch it
// until the consumer bumps round_request, which it only does after finishing
// every read and diagnostic for that round.
//
// Cache-maintenance discipline. Maintenance granularity is 64 B, so every field
// is placed such that NO 64 B block is written by both sides:
//
//   CtlBlock  (64 B)  AICPU writes, AICore reads   — go / mode / round_request
//   RptBlock  (64 B)  AICore writes, AICPU reads   — core id, published round
//   PayBlock  (64 B)  AICore writes, AICPU reads   — payload, alone
//
// A block written by one side is only ever cleaned by that side and only ever
// invalidated by the other. Bidirectional writes to one block would make the
// two sides' write-backs clobber each other.

#pragma once

#include <cstddef>
#include <cstdint>

// REG_SPR_COND_OFFSET on the a2a3 chip family; the consumer reads
// `aic_ctrl_reg_base + core_stride * core_id + offset` to poll a core's COND.
// Mirrors src/a2a3/platform/include/common/platform_config.h.
constexpr uint32_t kFinRegSprCondOffset = 0x4C8;
constexpr uint64_t kFinCoreStride = 8ULL * 1024 * 1024;

// FIN encoding — matches the runtime's MAKE_FIN_VALUE: bit 31 marks "finished",
// the low 31 bits carry the id. Rounds therefore must stay below 2^31 so the
// COND-carried id is never truncated relative to the 64-bit payload.
constexpr uint32_t kFinTaskIdMask = 0x7FFFFFFFu;
constexpr uint32_t kFinTaskStateMask = 0x80000000u;
constexpr uint64_t kFinMaxRound = 0x40000000ULL;  // stays well inside the mask

constexpr uint32_t kFinCacheLine = 64;

// get_coreid() is not a bare index: the runtime masks it with AICORE_COREID_MASK
// before using it to index the AIC_CTRL register window
// (src/a2a3/platform/onboard/aicore/inner_kernel.h::get_physical_core_id).
constexpr uint32_t kFinCoreIdMask = 0x0FFF;
// Upper bound on cores the window can address; polling past it faults.
constexpr uint32_t kFinMaxCores = 64;

// Producer publish sequence under test. P0 and P1 differ ONLY in the dsb.
enum FinProducerMode : uint32_t {
    kFinProducerIdle = 0,
    kFinProducerP0 = 1,  // store -> dcci -> FIN
    kFinProducerP1 = 2,  // store -> dcci -> dsb(DSB_DDR) -> FIN
};

// Consumer read sequence under test, applied after capturing this round's FIN
// and before the first payload load.
enum FinConsumerMode : uint32_t {
    kFinConsumerC0 = 0,  // bare load
    kFinConsumerC1 = 1,  // dsb sy (order the MMIO read before the GM load), load
    kFinConsumerC2 = 2,  // invalidate payload line (dc civac; dsb sy; isb), load
    kFinConsumerC3 = 3,  // dsb sy, then invalidate, then load
    kFinConsumerC4 = 4,  // timing control matched to the invalidate cost, load
    kFinConsumerC5 = 5,  // dmb ld — the minimal load-load barrier — then load
    kFinConsumerC6 = 6,  // acquire the flag itself (ldar on COND), then bare load
    kFinConsumerC7 = 7,  // timing control matched to the dsb sy cost, load
    kFinConsumerModeCount = 8,
};

// Per-round outcome of the FIRST payload load, kept separate from anything a
// later re-read found. `older` is the only value that means "published data not
// visible"; under the frozen protocol `newer` must never occur and is reported
// as a protocol/foreign-writer anomaly rather than folded into a failure count.
enum FinFirstRead : uint32_t {
    kFinFirstEqual = 0,
    kFinFirstOlder = 1,
    kFinFirstNewer = 2,
    kFinFirstOutcomeCount = 3,
};

// --- Control block: AICPU writes, AICore reads. Own 64 B. ---
struct alignas(64) FinCtlBlock {
    volatile uint32_t go;             // 0 = producer should exit
    volatile uint32_t mode;           // FinProducerMode
    volatile uint64_t round_request;  // consumer asks for this round
    volatile uint64_t round_ack;      // consumer releases it; producer waits for this
    volatile uint64_t _pad[5];
};
static_assert(sizeof(FinCtlBlock) == 64, "FinCtlBlock must own exactly one maintenance block");

// --- Report block: AICore writes, AICPU reads. Own 64 B. ---
struct alignas(64) FinRptBlock {
    volatile uint32_t core_id;          // get_coreid() raw, unmasked, as the AICore saw it
    volatile uint32_t core_id_valid;    // 1 once core_id is published
    volatile uint64_t published_round;  // the round whose payload+FIN are complete
    volatile uint64_t publish_count;    // total publishes, for a liveness check
    volatile uint64_t _pad[4];
};
static_assert(sizeof(FinRptBlock) == 64, "FinRptBlock must own exactly one maintenance block");

// --- Payload block: AICore writes, AICPU reads. Own 64 B, payload alone. ---
struct alignas(64) FinPayBlock {
    volatile uint64_t payload;
    volatile uint64_t _pad[7];
};
static_assert(sizeof(FinPayBlock) == 64, "FinPayBlock must own exactly one maintenance block");

struct alignas(64) FinHandshake {
    FinCtlBlock ctl;
    FinRptBlock rpt;
    FinPayBlock pay;
};
static_assert(sizeof(FinHandshake) == 192, "FinHandshake layout must match across compilers");
static_assert(offsetof(FinHandshake, ctl) == 0, "ctl must start a maintenance block");
static_assert(offsetof(FinHandshake, rpt) == 64, "rpt must start a maintenance block");
static_assert(offsetof(FinHandshake, pay) == 128, "pay must start a maintenance block");

// One record per round. Only the consumer writes this array and it writes
// sequentially, so records sharing a line with their neighbour is harmless; the
// array is a separate allocation from the handshake so it can never share a
// maintenance block with a field either side polls.
struct FinTrialRecord {
    uint32_t trial_id;
    uint32_t producer_mode;
    uint32_t consumer_mode;
    uint32_t first_read;  // FinFirstRead
    uint32_t raw_cond;    // the exact MMIO word that ended the wait
    uint32_t prefetched;  // 1 = consumer deliberately read payload before publish
    uint64_t expected_seq;
    uint64_t first_payload;
    uint64_t ordinary_reread_payload;
    uint64_t post_invalidate_payload;
    uint64_t ticks_to_first_match;  // AICPU clock; UINT64_MAX on timeout
    uint32_t reread_iters;
    uint32_t flags;  // bit0 = wait timeout, bit1 = reread timeout, bit2 = round echo mismatch
};
static_assert(sizeof(FinTrialRecord) == 72, "FinTrialRecord layout must match across compilers");

constexpr uint32_t kFinFlagWaitTimeout = 1u << 0;
constexpr uint32_t kFinFlagRereadTimeout = 1u << 1;
constexpr uint32_t kFinFlagRoundEchoMismatch = 1u << 2;
constexpr uint32_t kFinFlagPayloadUnaligned = 1u << 3;

// Aggregate per (producer_mode, consumer_mode, prefetched) cell.
struct FinCell {
    uint64_t trials;
    uint64_t first_equal;
    uint64_t first_older;
    uint64_t first_newer;
    uint64_t reread_matched;      // matched after ordinary re-reads, no maintenance
    uint64_t invalidate_matched;  // matched only after the invalidate
    uint64_t unresolved;          // never matched: reported, never attributed
    uint64_t wait_timeouts;
    uint64_t ticks_sum;
    uint64_t ticks_max;
};

constexpr uint32_t kFinProducerModeCount = 2;  // P0, P1
constexpr uint32_t kFinPrefetchVariants = 2;   // prefetched, not-prefetched

struct FinResult {
    uint32_t magic;
    int32_t consumer_rc;
    uint32_t records_written;
    uint32_t cells_valid;
    uint64_t payload_addr;           // for the alignment check the host prints
    uint64_t cond_addr;              // resolved MMIO address actually polled
    uint64_t invalidate_cost_ticks;  // measured; sizes the C4 delay arm
    uint64_t dsb_cost_ticks;         // measured; sizes the C7 delay arm
    uint64_t delay_arm_ticks;        // what C4 actually spent
    uint32_t core_id_raw;            // get_coreid() as reported
    uint32_t core_id_used;           // after AICORE_COREID_MASK, what indexed the window
    uint64_t producer_publishes;     // rpt.publish_count at the end: did it publish at all
    uint64_t producer_last_round;    // rpt.published_round at the end
    uint64_t final_payload;          // payload after an invalidate, at the end
    FinCell cells[kFinProducerModeCount * kFinConsumerModeCount * kFinPrefetchVariants];
};

constexpr uint32_t kFinResultMagic = 0xF1D0C0DEu;

inline uint32_t fin_cell_index(uint32_t producer_mode_idx, uint32_t consumer_mode, uint32_t prefetched) {
    return (producer_mode_idx * kFinConsumerModeCount + consumer_mode) * kFinPrefetchVariants + prefetched;
}

// DeviceArgs envelope — host writes, AICPU consumer reads. The first 12 words
// are the dispatcher's bootstrap area; everything after is ours.
struct alignas(8) FinDeviceArgs {
    uint64_t reserved_pre[12];
    uint64_t result_addr;
    uint64_t input_token;
    uint64_t handshake_addr;
    uint64_t aic_ctrl_reg_base;
    uint64_t records_addr;
    uint32_t records_capacity;
    uint32_t rounds_per_cell;
    uint32_t fallback_core_idx;  // used only if the producer never reports one
    uint32_t _pad;
};
