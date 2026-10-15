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
 * @file diagnostic_run_credit.h
 * @brief The two independent capacities one diagnostic run consumes, and the
 *        per-run reconciliation that reads a device terminal instead of a live
 *        header.
 *
 * A run holds two unrelated things and releases them at unrelated moments:
 *
 *   terminal bank   assigned at admission, freed as soon as the host has copied
 *                   that run's accounting out of it;
 *   output slot     assigned at admission, freed only when the file writer has
 *                   published the artifact.
 *
 * Neither release implies the other, and native retirement implies neither: a
 * run whose device work has been reaped can still own both, because its writer
 * has not finished. Treating one as a proxy for the other is what makes a
 * healthy slow writer look like exhausted capacity.
 *
 * Capacities are counts, not bytes. Each collector keeps its own byte budget
 * and its own output policy; nothing here imposes a shared byte ceiling on a
 * collector that has none.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "common/diagnostic_run_terminal.h"
#include "host/chip_swimlane_runs.h"

namespace simpler::dfx::credit {

/** Which capacity a reservation holds; they are never interchangeable. */
enum class CreditKind : uint8_t { TerminalBank = 0, OutputSlot = 1 };

/**
 * A bounded set of interchangeable indices, acquired and released atomically.
 *
 * The caller needs the *index* it won — a terminal bank is addressed by it and
 * an output slot owns host storage by it — so the set is per-index state rather
 * than a counter. Each index is **one** atomic word holding the ticket of its
 * holder, or zero when it is free. One word is the whole of an index's state,
 * which is what makes every transition a single compare-exchange:
 *
 *   acquire   0 -> ticket     release   ticket -> 0     reset   anything -> 0
 *
 * A free mask beside the holders would be a second field describing the same
 * fact, and no ordering of two independent atomics keeps them agreeing across a
 * reset: a release that passed its holder check could still publish its index as
 * free after the reset had handed that index to someone else, and that successor
 * would then have its capacity given away. Holding the ticket alone removes the
 * disagreement rather than ordering it, because the successor's ticket is the
 * same word the stale release has to match and no longer does.
 *
 * Acquisition scans at most `kCapacity` words and takes no lock: admission runs
 * on the lane's thread, which must not block.
 */
template <size_t N>
class IndexedCredits {
public:
    static_assert(N > 0 && N <= 32, "an index set stays small enough to scan");

    static constexpr size_t kCapacity = N;

    /**
     * Claim the lowest free index for `ticket`, or report exhaustion.
     *
     * The claim and the holder record are one compare-exchange, so there is no
     * state in which an index is taken but unowned — the window a separate
     * holder store would leave, and the one a concurrent reset could overwrite.
     */
    bool try_acquire(size_t *index_out, uint64_t ticket) noexcept {
        if (ticket == 0) return false;
        for (size_t index = 0; index < N; ++index) {
            uint64_t expected = 0;
            if (holder_[index].compare_exchange_strong(
                    expected, ticket, std::memory_order_acq_rel, std::memory_order_relaxed
                )) {
                *index_out = index;
                return true;
            }
        }
        return false;
    }

    /**
     * Return an index its holder still owns.
     *
     * Refused when the index belongs to someone else or is already free: either
     * would hand the same capacity out twice. A token invalidated by a reset
     * fails here however it interleaves with the reset, because the word it has
     * to match is the same word a successor's claim wrote.
     */
    bool release(size_t index, uint64_t ticket) noexcept {
        if (index >= N || ticket == 0) return false;
        uint64_t expected = ticket;
        return holder_[index].compare_exchange_strong(
            expected, 0, std::memory_order_acq_rel, std::memory_order_relaxed
        );
    }

    size_t in_use() const noexcept {
        size_t held = 0;
        for (size_t index = 0; index < N; ++index) {
            if (holder_[index].load(std::memory_order_acquire) != 0) held++;
        }
        return held;
    }

    bool exhausted() const noexcept { return in_use() == N; }

    /**
     * Make every index free and unowned, whatever any outstanding holder
     * believes.
     *
     * Each index is freed by a single exchange, so a claim that lands after it
     * owns that index and a release that arrives after it is refused. Neither
     * outcome can hand the same index to two holders, which is why this needs no
     * quiescent caller — the guarantee is per index and does not depend on the
     * order the indices are swept in.
     */
    void restore_all() noexcept {
        for (size_t index = 0; index < N; ++index)
            (void)holder_[index].exchange(0, std::memory_order_acq_rel);
    }

private:
    // Zero is free; any other value is the ticket of the holder. The whole of an
    // index's state, so a transition is one compare-exchange rather than two
    // writes a reset can interleave between.
    std::atomic<uint64_t> holder_[N] = {};
};

class RunCredits;

/**
 * One held capacity, released exactly once.
 *
 * Move-only and bound to its pool, kind, index, run identity and the pool
 * generation it was taken under. Every one of those is checked on release and
 * by any consumer that acts on the token, because none of them proves the
 * others: a token from a sibling collector's pool has a valid index, and a
 * token of the other kind has a valid generation.
 *
 * **The pool must outlive every token it issued.** The token holds a raw
 * pointer, and the generation proves only that the pool has not been *reset* —
 * it cannot prove the pool still exists. Each pool is a member of the collector
 * whose runs hold its tokens, and the collector outlives those runs, which is
 * what makes the pointer safe. Nothing here can check that, so it is a stated
 * precondition rather than a defended one.
 *
 * The destructor rolls back. An admission that fails part-way therefore returns
 * every capacity it won without the caller enumerating them, which is the whole
 * reason the all-or-nothing reservation can be written as a sequence of
 * independent attempts. A reservation the run goes on to own is moved into that
 * run's record, so the rollback no longer applies.
 */
class Reservation {
public:
    Reservation() = default;
    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;

    Reservation(Reservation &&other) noexcept { steal(other); }

    Reservation &operator=(Reservation &&other) noexcept {
        if (this != &other) {
            release();
            steal(other);
        }
        return *this;
    }

    ~Reservation() { release(); }

    bool held() const noexcept { return pool_ != nullptr; }
    explicit operator bool() const noexcept { return held(); }
    CreditKind kind() const noexcept { return kind_; }
    size_t index() const noexcept { return index_; }
    uint64_t generation() const noexcept { return generation_; }
    uint64_t run_epoch() const noexcept { return run_epoch_; }

    /** True when this token was issued by `pool` for `kind`, and is current. */
    bool issued_by(const RunCredits *pool, CreditKind kind) const noexcept;

    /** Give the capacity back now; idempotent, so every drop path is covered. */
    void release() noexcept;

    /**
     * Drop the token without giving the capacity back.
     *
     * For the one case where returning it would be the error: the holder can
     * neither keep the token nor prove the capacity is safe to reuse. The index
     * stays out of the pool until the pool is reset, which is a loss of one
     * capacity and not a loss of safety — the opposite trade from `release`,
     * and the only correct one where the proof is missing.
     */
    void abandon() noexcept { pool_ = nullptr; }

private:
    friend class RunCredits;

    Reservation(
        RunCredits *pool, CreditKind kind, size_t index, uint64_t generation, uint64_t run_epoch, uint64_t ticket
    ) :
        pool_(pool),
        kind_(kind),
        index_(index),
        generation_(generation),
        run_epoch_(run_epoch),
        ticket_(ticket) {}

    void steal(Reservation &other) noexcept {
        pool_ = other.pool_;
        kind_ = other.kind_;
        index_ = other.index_;
        generation_ = other.generation_;
        run_epoch_ = other.run_epoch_;
        ticket_ = other.ticket_;
        other.pool_ = nullptr;
    }

    RunCredits *pool_{nullptr};
    CreditKind kind_{CreditKind::TerminalBank};
    size_t index_{0};
    uint64_t generation_{0};
    uint64_t run_epoch_{0};
    uint64_t ticket_{0};
};

/**
 * One collector's two capacities.
 *
 * `kOutputSlots` is `dfx::runs::kMaxOpenEpochs`, which is already what
 * ArgsDump, PMU, ScopeStats and DepGen each size their unpublished-run storage
 * by; this type names that existing capacity rather than introducing a second
 * one. `kTerminalBanks` is the device-side mirror and is deliberately a
 * different number — the two are independent, and neither follows the
 * configured execution depth.
 */
class RunCredits {
public:
    static constexpr size_t kTerminalBanks = PLATFORM_DIAGNOSTIC_TERMINAL_BANKS;
    static constexpr size_t kOutputSlots = simpler::dfx::runs::kMaxOpenEpochs;

    /**
     * Reserve the bank this run's producers publish into.
     *
     * The returned index is a bank, not an execution resource index: a run's
     * bank is found by the identity that owns it.
     */
    Reservation reserve_terminal_bank(uint64_t run_epoch) noexcept {
        return grant(CreditKind::TerminalBank, run_epoch);
    }

    /** Reserve the host storage this run's artifact is written from. */
    Reservation reserve_output_slot(uint64_t run_epoch) noexcept { return grant(CreditKind::OutputSlot, run_epoch); }

    size_t banks_in_use() const noexcept { return banks_.in_use(); }
    size_t slots_in_use() const noexcept { return slots_.in_use(); }

    /**
     * Whether a grant of each kind could still succeed.
     *
     * Advisory: a concurrent grant may take the last one between this answer
     * and the caller's own reserve. It is what an admission gate asks before
     * committing to a pair, and the reserve is still the only thing that takes
     * capacity — so a false here declines a round, and a true that loses the
     * race is the ordinary refusal the reserve already reports.
     */
    bool can_reserve_terminal_bank() const noexcept { return !resetting() && !banks_.exhausted(); }
    bool can_reserve_output_slot() const noexcept { return !resetting() && !slots_.exhausted(); }

    uint64_t bank_refusals() const noexcept { return bank_refusals_.load(std::memory_order_relaxed); }
    uint64_t slot_refusals() const noexcept { return slot_refusals_.load(std::memory_order_relaxed); }
    uint64_t generation() const noexcept { return gate_.load(std::memory_order_seq_cst) >> 1; }

    /** True while a reset is sweeping, when no grant can complete. */
    bool resetting() const noexcept { return (gate_.load(std::memory_order_seq_cst) & kSweepBit) != 0; }

    /**
     * Invalidate every outstanding reservation and start a fresh generation.
     *
     * For a collector that is being rebuilt: its device storage is about to be
     * a different allocation, so an index from the previous generation names
     * nothing.
     *
     * **The boundary.** The sweep is bracketed by two bumps of one gate word,
     * so it runs under an odd gate and every grant reads that word before it
     * takes an index and again before it returns one (`grant`). That gives each
     * grant a single decision point against this sweep:
     *
     *   - a grant that read the gate before the first bump sees it changed at
     *     its second read and refuses, returning whatever it took;
     *   - a grant that read it during the sweep sees the sweep bit and refuses
     *     before taking anything;
     *   - a grant that read it after the second bump cannot be swept, because
     *     the sweep is already finished.
     *
     * So a returned token's index is never revoked by a reset the grant did not
     * observe, and a reset that overlaps a grant makes that grant fail rather
     * than hand its index to two holders. That is the whole claim: this is not
     * a general concurrent-teardown promise, and a caller that resets while
     * admitting gets refusals, which is what a collector being rebuilt wants.
     */
    void reset() noexcept {
        gate_.fetch_add(kSweepBit, std::memory_order_seq_cst);
        banks_.restore_all();
        slots_.restore_all();
        // Two bumps of the sweep bit are one generation step, and they leave the
        // bit clear: `generation()` is the gate halved.
        gate_.fetch_add(kSweepBit, std::memory_order_seq_cst);
    }

    /**
     * Run `fn` inside every grant, between the index claim and the gate
     * re-read, so a case can drive the one interleaving this protocol exists
     * for. Null in production, where the window is empty.
     */
    void set_grant_interleave_for_test(void (*fn)(void *), void *ctx) noexcept {
        interleave_ = fn;
        interleave_ctx_ = ctx;
    }

private:
    friend class Reservation;

    static constexpr uint64_t kSweepBit = 1;

    /**
     * One grant: read the gate, take an index, re-read the gate.
     *
     * The re-read is what makes the claim and the generation one decision
     * instead of two independent ones — the defect a bare `generation_.load()`
     * after the claim left open, where a reset could advance and sweep between
     * them and still hand back a token stamped with the current generation.
     */
    Reservation grant(CreditKind kind, uint64_t run_epoch) noexcept {
        auto refuse = [&](CreditKind refused) {
            if (refused == CreditKind::TerminalBank) {
                bank_refusals_.fetch_add(1, std::memory_order_relaxed);
            } else {
                slot_refusals_.fetch_add(1, std::memory_order_relaxed);
            }
            return Reservation{};
        };

        const uint64_t entry = gate_.load(std::memory_order_seq_cst);
        if ((entry & kSweepBit) != 0) return refuse(kind);

        const uint64_t ticket = next_ticket_.fetch_add(1, std::memory_order_acq_rel);
        size_t index = 0;
        const bool took =
            kind == CreditKind::TerminalBank ? banks_.try_acquire(&index, ticket) : slots_.try_acquire(&index, ticket);
        if (!took) return refuse(kind);

        if (interleave_ != nullptr) interleave_(interleave_ctx_);

        if (gate_.load(std::memory_order_seq_cst) != entry) {
            // A reset ran across this grant. Give back what was taken — which
            // releases nothing if the sweep already cleared it and a successor
            // now holds the index — and refuse, rather than return a token for
            // capacity this no longer owns.
            if (kind == CreditKind::TerminalBank) {
                (void)banks_.release(index, ticket);
            } else {
                (void)slots_.release(index, ticket);
            }
            return refuse(kind);
        }
        return {this, kind, index, entry >> 1, run_epoch, ticket};
    }

    bool give_back(CreditKind kind, size_t index, uint64_t generation, uint64_t ticket) noexcept {
        if (generation != this->generation()) return false;
        return kind == CreditKind::TerminalBank ? banks_.release(index, ticket) : slots_.release(index, ticket);
    }

    IndexedCredits<kTerminalBanks> banks_{};
    IndexedCredits<kOutputSlots> slots_{};
    // Low bit: a reset is sweeping. Above it: the generation. One word, because
    // a grant has to decide against both facts at once.
    std::atomic<uint64_t> gate_{2};
    std::atomic<uint64_t> next_ticket_{1};
    std::atomic<uint64_t> bank_refusals_{0};
    std::atomic<uint64_t> slot_refusals_{0};
    void (*interleave_)(void *){nullptr};
    void *interleave_ctx_{nullptr};
};

inline bool Reservation::issued_by(const RunCredits *pool, CreditKind kind) const noexcept {
    return pool_ != nullptr && pool_ == pool && kind_ == kind && ticket_ != 0 && generation_ == pool->generation();
}

inline void Reservation::release() noexcept {
    if (pool_ == nullptr) return;
    RunCredits *pool = pool_;
    pool_ = nullptr;
    (void)pool->give_back(kind_, index_, generation_, ticket_);
}

/**
 * What a run's producer accounting proves about that run, before any statement
 * about its file.
 *
 * `counts_unknown` is the state in which no arithmetic about the run holds: a
 * saturated counter, a participant that published nothing, or a per-producer
 * accounting that does not conserve. It outranks every verdict built on the
 * figures, which is why it settles first.
 */
struct RunReconciliation {
    simpler::dfx::runs::Verdict verdict{simpler::dfx::runs::Verdict::Published};
    bool counts_unknown{false};
    bool terminal_seen{false};
    uint64_t produced{0};                 // total the producers took responsibility for
    uint64_t dropped{0};                  // refused before the write, or written and never handed over
    uint64_t published{0};                // records inside buffers the producers committed
    uint64_t received{0};                 // records the host actually took delivery of
    uint64_t unsettled{0};                // records still in a held buffer at close
    uint64_t unaccounted{0};              // published by the device, never received by the host
    uint64_t published_payloads{0};       // of those records, the ones carrying a payload
    uint64_t published_payload_bytes{0};  // bytes those payloads occupied
    bool fatal_latched{false};
    int expected_producers{0};  // producers this collector is dimensioned for
    int reported_producers{0};  // of them, the ones that claimed this run
    int missing_producers{0};   // non-zero only when nobody claimed it at all
    // Owner-defined observations, as participants of this run reported them.
    // Counted per code rather than summed: a note is one producer's statement
    // about itself, so what a reader may say is how many producers said which
    // thing, never a total of the values. `first_note` keeps the whole word of
    // the lowest-numbered participant that carried one, so a reader can quote
    // an owner payload without the struct knowing what it means.
    int notes_by_code[DIAGNOSTIC_TERMINAL_NOTE_CODES]{};
    uint32_t first_note{0};
    // The same two figures for the producers' second observation, kept apart
    // from the first so a reader can say what was seen at each instant. An
    // owner that makes only one observation leaves these zero.
    int close_notes_by_code[DIAGNOSTIC_TERMINAL_NOTE_CODES]{};
    uint32_t first_close_note{0};
};

/**
 * One producer's own statement about one run, as a reader may quote it.
 *
 * `producer` is the index the owner knows that producer by — for PMU, the
 * logical core id its rows carry — so a reader can line a row up with the
 * observation the same producer made. `records` is what that producer took
 * responsibility for, which is how a producer the run actually used is told
 * from one that claimed the run and executed nothing.
 */
struct ProducerNote {
    int producer{0};
    uint32_t note{0};
    // That producer's second observation, or zero when it made none. Carried
    // beside the first rather than instead of it: the pair is what a reader
    // compares, and either alone answers a different question.
    uint32_t close_note{0};
    uint64_t records{0};
};

/**
 * Collect the notes of the producers that both claimed this run and recorded
 * something, from a bank the caller has already copied.
 *
 * Identity is the same filter the reconciliation uses, and it is applied here
 * again rather than inherited: an entry carrying another epoch is another run's
 * and is not read. Nothing is summed and nothing is allocated — the caller
 * supplies the bound, which exists because a reader quoting every producer of a
 * fully occupied device is not a bounded line.
 *
 * @return how many entries were written, never more than `capacity`.
 */
inline int collect_producer_notes(
    const DiagnosticRunTerminal *entries, int stride, uint64_t run_epoch, ProducerNote *out, int capacity
) {
    if (entries == nullptr || out == nullptr || stride <= 0 || capacity <= 0 || run_epoch == 0) return 0;
    int filled = 0;
    for (int i = 0; i < stride && filled < capacity; ++i) {
        const DiagnosticRunTerminal &entry = entries[i];
        if (entry.run_epoch != run_epoch) continue;
        if (entry.total == 0) continue;
        out[filled].producer = i;
        out[filled].note = entry.producer_note;
        out[filled].close_note = entry.producer_note_close;
        out[filled].records = entry.total;
        filled++;
    }
    return filled;
}

/**
 * Reconcile one run from its own terminal entries and the host's receipt count.
 *
 * The terminal is the only per-run evidence: live header counters belong to
 * whichever run is currently executing, and both an admitting and a closing
 * host read can straddle a producer transition. So a missing or mismatched
 * identity is not a recoverable case to be patched from live state — it is
 * `CutUnknown`, and the artifact says so.
 *
 * **The participant set is the device's statement.** A producer claims its
 * entry at its own run init and settles it at its close, so an entry carrying
 * this run's identity is a participant and one that does not is a producer this
 * run never brought up. The host cannot supply that number: the bank is
 * dimensioned by the producers the platform could have, and how many a run
 * actually runs is decided on the device — a scheduler that keeps one of its
 * AICPU threads for orchestration runs one fewer dump producer than the thread
 * count the host configured, and requiring the configured number would report
 * every such run as incomplete.
 *
 * What is still required is that every participant *settled*: the claim carries
 * `counts_unknown` until the close replaces it, so a producer that claimed and
 * was then killed makes the run unknown through its own entry rather than
 * through a count kept somewhere else. A run that nobody claimed has no
 * accounting at all and is unknown for that reason.
 *
 * **Conservation is per producer.** `total == published + dropped + live` holds
 * for each participant separately; checking only the aggregate lets one
 * producer's surplus cancel another's shortfall, which is two defects reported
 * as none.
 *
 * Loss is reported, never reconstructed. `unaccounted` is the gap between what
 * the producers committed and what the host took delivery of; it is attributed
 * to this run because the terminal carries this run's identity, and no record
 * is invented to close it.
 *
 * @param entries   this run's bank, indexed by producer
 * @param stride    entries in that bank, the platform maximum it was sized by
 * @param expected  participants this run brought up, occupying `[0, expected)`
 * @param run_epoch the identity every expected entry must carry
 * @param received  records the host took delivery of for this run
 */
inline RunReconciliation
reconcile_run(const DiagnosticRunTerminal *entries, int stride, int expected, uint64_t run_epoch, uint64_t received) {
    RunReconciliation out;
    out.received = received;
    out.expected_producers = expected;
    if (entries == nullptr || stride <= 0 || expected <= 0 || expected > stride || run_epoch == 0) {
        out.verdict = simpler::dfx::runs::Verdict::CutUnknown;
        out.counts_unknown = true;
        out.missing_producers = expected > 0 ? expected : 0;
        return out;
    }

    bool unusable = false;
    // The whole array, because identity is the filter: a participant is an
    // entry carrying this run's epoch, wherever the device placed it, and an
    // entry carrying another is a different run's or an unused slot. Scanning a
    // prefix instead would make the host's dimension decide who counted.
    for (int i = 0; i < stride; ++i) {
        const DiagnosticRunTerminal &entry = entries[i];
        if (entry.run_epoch != run_epoch) continue;
        out.terminal_seen = true;
        out.reported_producers++;

        const uint64_t total = entry.total;
        const uint64_t dropped = entry.dropped;
        const uint64_t published = entry.published_records;
        const uint64_t live = entry.live_at_close;
        if (entry.counts_unknown != 0 || total == UINT32_MAX || dropped == UINT32_MAX || published == UINT32_MAX ||
            live == UINT32_MAX || entry.published_payloads == UINT32_MAX ||
            entry.published_payload_bytes == UINT64_MAX) {
            unusable = true;
        }
        // Per producer, in that producer's own unit.
        if (published + dropped + live != total) unusable = true;

        out.produced += total;
        out.dropped += dropped;
        out.published += published;
        out.unsettled += live;
        out.published_payloads += entry.published_payloads;
        out.published_payload_bytes += entry.published_payload_bytes;
        out.fatal_latched = out.fatal_latched || entry.fatal_latched != 0;

        // This producer's own observation, attributed by the same identity
        // filter as its counters and never folded into them: an unreadable
        // accounting leaves the note readable, and a note says nothing about
        // whether the arithmetic balances.
        const uint32_t note = entry.producer_note;
        if (note != 0) {
            const uint32_t code = diagnostic_terminal_note_code(note);
            if (code < DIAGNOSTIC_TERMINAL_NOTE_CODES) out.notes_by_code[code]++;
            if (out.first_note == 0) out.first_note = note;
        }
        const uint32_t close_note = entry.producer_note_close;
        if (close_note != 0) {
            const uint32_t code = diagnostic_terminal_note_code(close_note);
            if (code < DIAGNOSTIC_TERMINAL_NOTE_CODES) out.close_notes_by_code[code]++;
            if (out.first_close_note == 0) out.first_close_note = close_note;
        }
    }

    if (!out.terminal_seen) {
        // Nobody claimed this run. Either its producers never reached their
        // init or the bank still holds an earlier occupant; neither gives this
        // run an accounting, and the whole dimensioned set is reported missing
        // because none of it answered.
        out.missing_producers = expected;
        out.verdict = simpler::dfx::runs::Verdict::CutUnknown;
        out.counts_unknown = true;
        return out;
    }
    if (unusable) {
        out.verdict = simpler::dfx::runs::Verdict::CutUnknown;
        out.counts_unknown = true;
        return out;
    }
    if (received > out.published) {
        // More delivered than committed means the receipt count covers records
        // this identity does not own.
        out.verdict = simpler::dfx::runs::Verdict::CutUnknown;
        out.counts_unknown = true;
        return out;
    }

    out.unaccounted = out.published - received;
    if (out.fatal_latched || out.dropped != 0 || out.unsettled != 0 || out.unaccounted != 0) {
        out.verdict = simpler::dfx::runs::Verdict::PartialSafe;
        return out;
    }
    out.verdict = simpler::dfx::runs::Verdict::Published;
    return out;
}

}  // namespace simpler::dfx::credit
