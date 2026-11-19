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

/**
 * @file diagnostic_run_terminal.h
 * @brief Per-run terminal accounting published by a device producer, for the
 *        collectors that stream records through a free/ready queue pair.
 *
 * A host instant is not a run boundary. Once a successor may enter the device
 * queue before its predecessor's host close, no counter a producer keeps in its
 * live header can be read as one run's total: the successor's device init
 * rewrites it, and both the admitting and the closing host reads can straddle
 * that transition. The only accounting a reader can attribute is one the
 * producer itself stamps with the run identity it was executing under.
 *
 * `run_epoch` is that identity and the whole validity test. Zero means "no
 * snapshot": it is the initialized state and no live run carries it
 * (`device_run_result.h` uses `run_epoch != 0` the same way). There is no
 * separate ready flag, because a second field would imply a publication
 * protocol this record does not provide — the producer writes the counters
 * first and the epoch last, and a reader that sees its own epoch therefore sees
 * settled counters.
 *
 * Scope: ArgsDump, PMU, ScopeStats and DepGen. The swimlane has its own
 * `ChipSwimlaneRunTerminal`, whose producer set is the AICore/AICPU task and
 * phase grid rather than a queue pair, and the HBG host graph is host-only and
 * has no device producer to publish anything.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "common/platform_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Distinct low-byte codes a `producer_note` may carry, which bounds how many
 * buckets a reader has to keep. Code 0 is "no observation" for every owner.
 */
#define DIAGNOSTIC_TERMINAL_NOTE_CODES 8

/** One note: a small owner code in the low byte, owner payload above it. */
static inline uint32_t diagnostic_terminal_make_note(uint32_t code, uint32_t payload) {
    return (code & 0xFFu) | ((payload & 0x00FFFFFFu) << 8);
}

/** The code a note carries. */
static inline uint32_t diagnostic_terminal_note_code(uint32_t note) { return note & 0xFFu; }

/** The owner payload a note carries. */
static inline uint32_t diagnostic_terminal_note_payload(uint32_t note) { return (note >> 8) & 0x00FFFFFFu; }

/**
 * One producer's settled accounting for one run, retained past that run.
 *
 * `total` and `dropped` are what the producer counted; `published_records` and
 * `published_buffers` are what it actually handed to the host, which neither of
 * the first two states — `total - dropped` is what it *meant* to deliver, and a
 * handoff can still fail afterwards without moving either. `live_at_close`
 * records whatever was still in the producer's held buffer when it closed: a
 * non-zero value means the settlement step did not run, so the other figures
 * are not a closed set, and a reader learns that without inspecting a live head
 * a later run may already have reused.
 *
 * `published_payloads` and `published_payload_bytes` carry the one quantity
 * records cannot: ArgsDump streams metadata records and payload bytes on
 * separate clocks, and its arena is reusable only once the payload is
 * acknowledged. Both count only what was handed over, and a producer with no
 * payload leaves them zero. They are not a second record count —
 * `published_payloads` counts the subset of `published_records` that carried
 * one — so neither is a term in the conservation equation.
 *
 * `fatal_latched` is the producer's own sticky error for this run. It does not
 * make the counts unknown — the accounting still balances — so it travels
 * beside them rather than invalidating them.
 *
 * `counts_unknown` does the opposite: it is the producer's own statement that
 * one of its counters, or a sum it had to form from them, reached the width it
 * is stored in. Every producer-side increment saturates, so a counter at its
 * sentinel is sticky rather than wrapped — but a sum of two saturating counters
 * can still exceed the width, and a reader cannot tell that from the fields. A
 * non-zero value means no arithmetic about this producer holds, whatever the
 * other figures read.
 *
 * `producer_note` is one owner-defined observation this producer made about
 * this run, in the low byte, with up to 24 bits of owner payload above it. It
 * is not a counter and nothing is summed from it: one producer writes it once,
 * inside the run that claimed the entry, and a reader attributes it to that
 * producer and that run or not at all. Zero means the owner made none. The
 * generic layer defines only its width and that the low byte is a small code
 * (`DIAGNOSTIC_TERMINAL_NOTE_CODES`); what the codes mean belongs to the owner.
 *
 * `producer_note_close` is a **second** observation of the same shape, for an
 * owner that observes the same thing twice and needs both: one field per
 * instant rather than one field overwritten. An owner that replaced the first
 * with the second would destroy the only record of what it saw at the earlier
 * instant, so a reader could no longer tell "it was never so" from "it stopped
 * being so" — which is exactly the distinction such an owner is observing for.
 * Zero means the owner made no second observation; the two fields carry
 * independent codes from the owner's own set.
 *
 * One entry per cache line, and the array start is line-aligned: publication
 * flushes whole 64-byte lines, so two entries sharing a line would let one
 * producer's flush write back a stale copy of the other's.
 */
struct DiagnosticRunTerminal {
    volatile uint64_t run_epoch;                // 8  — 0 = no snapshot in this entry
    volatile uint32_t total;                    // 4  — every record the producer appended
    volatile uint32_t dropped;                  // 4  — appended but not deliverable
    volatile uint32_t published_records;        // 4  — records inside committed buffers
    volatile uint32_t published_buffers;        // 4  — successful ready-queue commits
    volatile uint32_t live_at_close;            // 4  — records still unsettled in the held buffer
    volatile uint32_t fatal_latched;            // 4  — producer's sticky error for this run
    volatile uint64_t published_payload_bytes;  // 8  — payload handed over, 0 without one
    volatile uint32_t counts_unknown;           // 4  — a counter or a sum of them saturated
    volatile uint32_t published_payloads;       // 4  — payload-bearing records handed over
    volatile uint32_t producer_note;            // 4  — owner-defined observation, 0 = none
    volatile uint32_t producer_note_close;      // 4  — the same owner's second observation, 0 = none
    uint32_t pad[2];                            // 8 → 64B
} __attribute__((aligned(64)));

static_assert(sizeof(DiagnosticRunTerminal) == 64, "DiagnosticRunTerminal must be one cache line");
static_assert(alignof(DiagnosticRunTerminal) == 64, "DiagnosticRunTerminal must be cache-line aligned");
static_assert(offsetof(DiagnosticRunTerminal, total) == 8, "DiagnosticRunTerminal::total offset drift");
static_assert(offsetof(DiagnosticRunTerminal, dropped) == 12, "DiagnosticRunTerminal::dropped offset drift");
static_assert(
    offsetof(DiagnosticRunTerminal, published_records) == 16, "DiagnosticRunTerminal::published_records offset drift"
);
static_assert(
    offsetof(DiagnosticRunTerminal, published_buffers) == 20, "DiagnosticRunTerminal::published_buffers offset drift"
);
static_assert(
    offsetof(DiagnosticRunTerminal, live_at_close) == 24, "DiagnosticRunTerminal::live_at_close offset drift"
);
static_assert(
    offsetof(DiagnosticRunTerminal, fatal_latched) == 28, "DiagnosticRunTerminal::fatal_latched offset drift"
);
static_assert(
    offsetof(DiagnosticRunTerminal, published_payload_bytes) == 32,
    "DiagnosticRunTerminal::published_payload_bytes offset drift"
);
static_assert(
    offsetof(DiagnosticRunTerminal, counts_unknown) == 40, "DiagnosticRunTerminal::counts_unknown offset drift"
);
static_assert(
    offsetof(DiagnosticRunTerminal, published_payloads) == 44, "DiagnosticRunTerminal::published_payloads offset drift"
);
static_assert(
    offsetof(DiagnosticRunTerminal, producer_note) == 48, "DiagnosticRunTerminal::producer_note offset drift"
);
static_assert(
    offsetof(DiagnosticRunTerminal, producer_note_close) == 52,
    "DiagnosticRunTerminal::producer_note_close offset drift"
);
// The device writes this entry and the host reads it back with a raw byte copy,
// so it is a wire struct. The compiler builtins are used rather than the
// `<type_traits>` spellings because this header is also compiled by ccec, which
// the standard library headers are not available to.
static_assert(__is_trivially_copyable(DiagnosticRunTerminal), "DiagnosticRunTerminal must be memcpy-able");
static_assert(__is_standard_layout(DiagnosticRunTerminal), "DiagnosticRunTerminal must be standard-layout");

/**
 * Banks per collector, one per retained run.
 *
 * Mirrors the swimlane's `PLATFORM_RUN_TERMINAL_BANKS` rather than including
 * it: that header carries the whole swimlane pool grid and is compiled by ccec
 * for AICore, which the queue-pair producers are not. The two constants are
 * cross-checked in `tests/ut/cpp/common/platform/test_diagnostic_run_credit.cpp`,
 * which is where both headers are visible at once.
 *
 * A bank is addressed by the run identity that owns it, never by an execution
 * resource index: execution depth and retained diagnostic count are independent
 * numbers, and indexing one by the other would tie them together.
 */
#define PLATFORM_DIAGNOSTIC_TERMINAL_BANKS 3

/**
 * Counters a producer settles before it stamps an entry's identity.
 *
 * `total` counts every record the producer took responsibility for, in the same
 * unit as `published_records` and `live_at_close`. Where a producer's own
 * counters are in a different unit — a submit that spills into a chain of
 * records, say — it converts them here, because the conservation the host
 * checks is per producer and cannot absorb a unit mismatch.
 *
 * Conservation, which every producer must satisfy:
 *
 *     total == published_records + dropped + live_at_close
 *
 * `dropped` therefore covers both a record refused before it was ever written
 * and one written into a buffer that was never handed over; `total` must not
 * count the latter twice.
 */
struct DiagnosticRunTerminalCounts {
    uint32_t total;
    uint32_t dropped;
    uint32_t published_records;
    uint32_t published_buffers;
    uint32_t live_at_close;
    uint32_t fatal_latched;
    uint64_t published_payload_bytes;
    uint32_t counts_unknown;
    uint32_t published_payloads;
};

/** Add, clamping at the sentinel a reader treats as "unknown". */
inline uint32_t diagnostic_terminal_add_u32(uint32_t value, uint32_t by) {
    return (by >= UINT32_MAX - value) ? UINT32_MAX : value + by;
}

/** Add, clamping at the 64-bit sentinel. */
inline uint64_t diagnostic_terminal_add_u64(uint64_t value, uint64_t by) {
    return (by >= UINT64_MAX - value) ? UINT64_MAX : value + by;
}

/**
 * True when any figure a reader derives conservation from is at its sentinel.
 *
 * `published_buffers` is excluded: it is a count of commits, not a term in the
 * conservation equation, so a saturated one costs the reader nothing it uses.
 */
inline int diagnostic_terminal_counts_saturated(const DiagnosticRunTerminalCounts *counts) {
    return counts->counts_unknown != 0 || counts->total == UINT32_MAX || counts->dropped == UINT32_MAX ||
                   counts->published_records == UINT32_MAX || counts->live_at_close == UINT32_MAX ||
                   counts->published_payloads == UINT32_MAX || counts->published_payload_bytes == UINT64_MAX ?
               1 :
               0;
}

/** Bytes one collector's whole bank array occupies. */
inline size_t calc_diagnostic_terminal_bank_bytes(int producers) {
    return (size_t)producers * (size_t)PLATFORM_DIAGNOSTIC_TERMINAL_BANKS * sizeof(DiagnosticRunTerminal);
}

/**
 * One producer's entry inside a bank array laid out bank-major.
 *
 * Bank-major keeps one run's producers contiguous, so a host copy of a run's
 * accounting is one range rather than a strided gather.
 */
inline volatile DiagnosticRunTerminal *
diagnostic_terminal_entry(volatile DiagnosticRunTerminal *banks, int producers, int bank, int producer) {
    return banks + (size_t)bank * (size_t)producers + (size_t)producer;
}

/**
 * Clear one entry so a reader attributes nothing to it.
 *
 * The identity goes first: an entry whose epoch is already zero cannot be
 * mistaken for a snapshot however the rest of the line is observed.
 */
inline void diagnostic_terminal_reset(volatile DiagnosticRunTerminal *entry) {
    entry->run_epoch = 0;
    entry->total = 0;
    entry->dropped = 0;
    entry->published_records = 0;
    entry->published_buffers = 0;
    entry->live_at_close = 0;
    entry->fatal_latched = 0;
    entry->published_payload_bytes = 0;
    entry->counts_unknown = 0;
    entry->published_payloads = 0;
    entry->producer_note = 0;
    entry->producer_note_close = 0;
}

/**
 * Write the counters before the separate identity stamp.
 *
 * The host checks identity only in a post-completion snapshot. The producer's
 * stores, barrier and flush have finished before that read, and the run's
 * reservation prevents bank reuse until the host has copied it. Sharing a cache
 * line does not make these stores atomic. Failed runs are discarded unread even
 * when their producers are proved stopped; epoch is not a live publication flag.
 */
inline void
diagnostic_terminal_store_counts(volatile DiagnosticRunTerminal *entry, const DiagnosticRunTerminalCounts *counts) {
    entry->run_epoch = 0;
    entry->total = counts->total;
    entry->dropped = counts->dropped;
    entry->published_records = counts->published_records;
    entry->published_buffers = counts->published_buffers;
    entry->live_at_close = counts->live_at_close;
    entry->fatal_latched = counts->fatal_latched;
    entry->published_payload_bytes = counts->published_payload_bytes;
    entry->published_payloads = counts->published_payloads;
    entry->counts_unknown = static_cast<uint32_t>(diagnostic_terminal_counts_saturated(counts));
}

/** Stamp the identity checked in the completed run's snapshot. */
inline void diagnostic_terminal_commit(volatile DiagnosticRunTerminal *entry, uint64_t run_epoch) {
    entry->run_epoch = run_epoch;
}

/**
 * Claim this entry for the run the producer is starting.
 *
 * The participant set is the device's statement, not the host's: a collector
 * is dimensioned by the producers the platform could have, while which of them
 * a run actually brings up is decided on the device — a scheduler that keeps
 * one of its AICPU threads for orchestration runs one fewer dump producer than
 * the thread count the host configured. A host that required every dimensioned
 * producer to report would call every such run incomplete.
 *
 * So a producer says "I am part of this run" here, and says what it collected
 * at its close. The claim is deliberately `counts_unknown`: until the close
 * overwrites it, this participant has published no arithmetic anyone may use,
 * which is exactly the state of a producer killed mid-run. A producer that
 * never ran leaves epoch zero and is not a participant at all.
 */
inline void diagnostic_terminal_claim(volatile DiagnosticRunTerminal *entry, uint64_t run_epoch) {
    DiagnosticRunTerminalCounts counts;
    counts.total = 0;
    counts.dropped = 0;
    counts.published_records = 0;
    counts.published_buffers = 0;
    counts.live_at_close = 0;
    counts.fatal_latched = 0;
    counts.published_payload_bytes = 0;
    counts.counts_unknown = 1;
    counts.published_payloads = 0;
    diagnostic_terminal_store_counts(entry, &counts);
    entry->counts_unknown = 1;
    // The claim is where the entry becomes this run's, so a previous
    // occupant's observation goes here: the owner writes its own after the
    // claim, and the close leaves whatever was written untouched.
    entry->producer_note = 0;
    entry->producer_note_close = 0;
    diagnostic_terminal_commit(entry, run_epoch);
}

#ifdef __cplusplus
}
#endif
