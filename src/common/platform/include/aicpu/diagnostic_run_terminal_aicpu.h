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
 * @file diagnostic_run_terminal_aicpu.h
 * @brief Device-side publication and per-run initialization of a queue-pair
 *        collector's terminal accounting.
 *
 * Every per-run diagnostic setting is a process global of the AICPU SO, pushed
 * at kernel entry from the launching run's `KernelArgs`, and device producers
 * of two runs never co-execute: a joined successor's operator is a queued
 * device wait behind its predecessor's. So a producer may initialize its own
 * counters at its run's device init — the predecessor's producers are gone with
 * their operator — and that is what makes a host-side reset at admission
 * unnecessary, which is the reset a successor already in the queue would
 * otherwise race.
 */

#pragma once

#include <cstdint>

#include "aicpu/cache_maintenance.h"
#include "common/diagnostic_run_terminal.h"
#include "common/memory_barrier.h"

namespace simpler::dfx::terminal {

/**
 * Publish one producer's settled accounting for the run it is closing.
 *
 * The entry is one cache line and this producer is its only writer, so the
 * flush — which rounds to whole lines — writes back nothing another producer
 * owns. The identity is stored after the counters; the host reads an entry only
 * after that run's completion fence, never concurrently, so the barrier here
 * carries the handoff rather than a concurrent publication protocol.
 *
 * A run without an identity claims no bank: epoch zero is the entry's "no
 * snapshot" state and leaves whatever the bank held, which is either zero or a
 * prior run's snapshot the host has already copied.
 *
 * @param bank      address of this run's bank — its first producer entry — or 0
 *                  when the host resolved none. The host does the bank
 *                  indexing, so the device never derives an address of its own.
 * @param producers producer count the bank array was dimensioned by
 * @param producer  index of the publishing producer
 */
inline void
publish(uint64_t bank, int producers, int producer, uint64_t run_epoch, const DiagnosticRunTerminalCounts &counts) {
    if (bank == 0 || run_epoch == 0) return;
    if (producers <= 0 || producer < 0 || producer >= producers) return;

    volatile DiagnosticRunTerminal *entry = reinterpret_cast<volatile DiagnosticRunTerminal *>(bank) + producer;
    diagnostic_terminal_store_counts(entry, &counts);
    diagnostic_terminal_commit(entry, run_epoch);
    wmb();
    cache_flush_range(const_cast<const DiagnosticRunTerminal *>(entry), sizeof(DiagnosticRunTerminal));
}

/**
 * Claim the entry this run is about to account into, at its device init.
 *
 * A bank is reused only after the host has copied the snapshot that held it, so
 * overwriting it destroys nothing readable. Two things happen here, and the
 * second is why this cannot simply clear:
 *
 *   - the previous occupant's figures stop being attributable to anyone, so a
 *     producer that never reaches its close — a fatal run, a run killed by the
 *     operator timeout — cannot be read as having reported;
 *   - this producer states that it is part of *this* run.
 *
 * The second is the device's statement to make. A collector is dimensioned by
 * the producers the platform could have, while how many a run brings up is
 * decided here: a scheduler that keeps one AICPU thread for orchestration runs
 * one fewer dump producer than the thread count the host configured, and a host
 * that required every dimensioned producer to report would call every such run
 * incomplete. The claim carries `counts_unknown` until the close replaces it,
 * so a claimed-and-never-closed producer is reported as the unknown it is.
 *
 * A run with no identity claims nothing: epoch zero is the entry's "no
 * snapshot" state, and the host reads such an entry as a producer this run
 * never brought up.
 */
inline void begin_run(uint64_t bank, int producers, int producer, uint64_t run_epoch) {
    if (bank == 0 || run_epoch == 0) return;
    if (producers <= 0 || producer < 0 || producer >= producers) return;

    volatile DiagnosticRunTerminal *entry = reinterpret_cast<volatile DiagnosticRunTerminal *>(bank) + producer;
    diagnostic_terminal_claim(entry, run_epoch);
    wmb();
    cache_flush_range(const_cast<const DiagnosticRunTerminal *>(entry), sizeof(DiagnosticRunTerminal));
}

/**
 * Record this producer's one observation about this run, after its claim.
 *
 * Attribution is checked here rather than assumed: the note lands only in an
 * entry already carrying this run's epoch, so a producer whose claim did not
 * happen — or whose bank has moved on to a successor — writes nothing instead
 * of annotating another run's accounting. One producer writes one note once,
 * from the thread that ran its init, and the close leaves it alone; nothing
 * sums notes, and a reader that cannot attribute an entry to a run ignores its
 * note with the rest of it.
 */
inline void note_run(uint64_t bank, int producers, int producer, uint64_t run_epoch, uint32_t note) {
    if (bank == 0 || run_epoch == 0 || note == 0) return;
    if (producers <= 0 || producer < 0 || producer >= producers) return;

    volatile DiagnosticRunTerminal *entry = reinterpret_cast<volatile DiagnosticRunTerminal *>(bank) + producer;
    if (entry->run_epoch != run_epoch) return;
    entry->producer_note = note;
    wmb();
    cache_flush_range(const_cast<const DiagnosticRunTerminal *>(entry), sizeof(DiagnosticRunTerminal));
}

/**
 * Record this producer's *second* observation about this run, in its own field.
 *
 * Same attribution check as `note_run` and the same single-writer discipline.
 * A separate field rather than a second write of the first one: an owner that
 * observes the same thing at two instants needs both readable, because
 * replacing the earlier observation with the later one leaves a reader unable
 * to tell "it was never so" from "it stopped being so".
 */
inline void note_run_close(uint64_t bank, int producers, int producer, uint64_t run_epoch, uint32_t note) {
    if (bank == 0 || run_epoch == 0 || note == 0) return;
    if (producers <= 0 || producer < 0 || producer >= producers) return;

    volatile DiagnosticRunTerminal *entry = reinterpret_cast<volatile DiagnosticRunTerminal *>(bank) + producer;
    if (entry->run_epoch != run_epoch) return;
    entry->producer_note_close = note;
    wmb();
    cache_flush_range(const_cast<const DiagnosticRunTerminal *>(entry), sizeof(DiagnosticRunTerminal));
}

}  // namespace simpler::dfx::terminal
