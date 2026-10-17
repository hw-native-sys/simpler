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
 * @file diagnostic_run_banks.h
 * @brief The host side of one collector's terminal banks: the device
 *        allocation, the per-run assignment, and the copy that retires a bank.
 *
 * Each collector owns one of these. It holds `PLATFORM_DIAGNOSTIC_TERMINAL_BANKS`
 * banks of `producers` entries, allocated once in device memory and addressed
 * by the run identity that reserved one — never by an execution resource index,
 * which is a different and independently sized number.
 *
 * A bank's lifetime is bounded by the host's copy, not by the artifact: the
 * snapshot is taken out of device memory into host storage the moment that
 * run's device work is settled, which is what frees the bank for the next run
 * while the file writer still holds its own output slot.
 *
 * Nothing here is allocated when retention is off. With no bank mapped the
 * device publishes nothing and this reads nothing, which is what keeps the
 * default path on its existing counters rather than on an absent terminal.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <vector>

#include "common/diagnostic_run_terminal.h"
#include "common/memory_barrier.h"
#include "host/diagnostic_run_credit.h"

namespace simpler::dfx::credit {

/**
 * Read one run's bank array into host storage, given both addresses the
 * allocation has.
 *
 * Every collector's bank read goes through here, because the decision is one
 * decision — and it is decided by how this allocation was mapped, never by
 * which platform it is on. Those are different facts: a platform whose device
 * memory *can* be mapped into the host address space says nothing about an
 * allocation that was never registered, and dereferencing one because the
 * platform installs no copy shim is a read of an address this process does not
 * own.
 *
 * So `host_src` is what decides. It is non-null exactly when this allocation
 * has a host view — registered, or identically mapped — and then the bank is
 * read where it lies. Otherwise the platform's own device-to-host copy is the
 * only way to reach it. An allocation with neither is unreadable, and saying so
 * is the honest answer: the caller reports an accounting it could not read
 * rather than a snapshot it invented.
 */
inline int read_run_terminal_bank(
    void *host_dst, const void *device_src, const void *host_src, size_t bytes,
    const std::function<int(void *, const void *, size_t)> &copy_from_device
) {
    if (host_dst == nullptr || bytes == 0) return -1;
    if (host_src != nullptr) {
        std::memcpy(host_dst, host_src, bytes);
        return 0;
    }
    if (!copy_from_device || device_src == nullptr) return -1;
    return copy_from_device(host_dst, device_src, bytes);
}

/** Initialize newly allocated backing before any bank can reach a producer. */
inline int initialize_run_terminal_bank(
    void *device_dst, void *host_dst, const void *initial, size_t bytes,
    const std::function<int(void *, const void *, size_t)> &copy_to_device
) {
    if (device_dst == nullptr || initial == nullptr || bytes == 0) return -1;
    if (host_dst != nullptr) {
        // Registered/identically mapped storage follows the existing SVM
        // publication contract: write its host view, then order stores before
        // the runner can launch a producer. Never dereference device_dst here.
        std::memcpy(host_dst, initial, bytes);
        wmb();
        return 0;
    }
    if (!copy_to_device) return -1;
    // This callback completes the platform's synchronous H2D operation.
    return copy_to_device(device_dst, initial, bytes);
}

/**
 * One collector's bank array, plus the reservation state that assigns banks.
 *
 * The device address is handed to the producer by the launching run's kernel
 * arguments; this type owns the allocation and the bookkeeping, and states the
 * address it allocated. It initializes the whole allocation before opening; a
 * bank is producer-owned between reservation and the copy that retires it.
 *
 * `stride` is the platform maximum the array is dimensioned by — one entry per
 * possible producer — while `expected` is how many of them this collector
 * actually brought up. They are different numbers, and conflating them is what
 * would let a run with half its producers silent look complete: the array has an
 * entry for every core the platform could have, and a reader must require one
 * from every core the collector is using.
 *
 * **Caller serialization.** A single collector thread owns this object's
 * mutating calls — `open`, `close`, `copy_bank` and `settle`. Only the
 * reservation path is safe to call concurrently, because lane admission reaches
 * it. The host mirror is plain host storage with no internal lock.
 */
class RunTerminalBanks {
public:
    /**
     * Allocate the array and state its host view.
     *
     * `*host_out` is the address the host may dereference and is null when the
     * allocation has none — the two are reported together because they are one
     * fact about one allocation, and a caller that saw only the device address
     * would have to guess the other from the platform.
     */
    using AllocFn = void *(*)(void *owner, size_t bytes, void **host_out);
    /**
     * Take the host mapping down, before the memory goes back.
     *
     * Separate from `FreeFn` so the two outcomes are recorded separately: an
     * unmap that succeeded and a free that then failed leave the array mapped
     * nowhere and held, and the retry that follows must not unmap again.
     * Returns 0 where there was no mapping to take down.
     */
    using UnmapFn = int (*)(void *owner, void *device_ptr);
    /** Returns the device allocator's own status; non-zero is a failed free. */
    using FreeFn = int (*)(void *owner, void *device_ptr);
    using CopyFn = int (*)(void *owner, void *host_dst, const void *device_src, const void *host_src, size_t bytes);

    using InitializeFn = int (*)(void *owner, void *device_dst, void *host_dst, const void *initial, size_t bytes);

    RunTerminalBanks() = default;
    RunTerminalBanks(const RunTerminalBanks &) = delete;
    RunTerminalBanks &operator=(const RunTerminalBanks &) = delete;

    /**
     * Closes the public side only. The backing is **not** freed here.
     *
     * A destructor cannot prove the device has stopped and has nowhere to
     * report a failed free, and this array is exactly the memory a producer
     * writes: freeing it on an unproven path is the one error whose symptom is
     * a device write into reclaimed host-visible memory. A collector that is
     * destroyed without a proved release therefore leaves the allocation
     * retained and says so through `backing_retained()`, which is the same
     * answer the accepted contract gives for any release it cannot prove.
     */
    ~RunTerminalBanks() { detach(); }

    /**
     * Allocate the bank array for a collector running `expected` of `stride`
     * possible producers.
     *
     * The complete allocation is initialized before opening. A false result
     * admits no reservations and publishes no device address; an enabled
     * retained owner must propagate it as an error before launch. Allocation
     * or initialization failure can still retain cleanup-only ownership when
     * its unmap/free fails.
     *
     * Refused while **any** allocation this object made is still mapped —
     * whether it is still admitting or was detached and retained. `opened()` is
     * false in the retained case, so testing it alone would overwrite the
     * handle to memory a producer may still be writing and lose the only record
     * of it. Retained backing is a release this object owes, not free space.
     */
    bool open(
        int stride, int expected, void *owner, AllocFn alloc, UnmapFn unmap, FreeFn release, CopyFn copy,
        InitializeFn initialize
    ) {
        if (device_banks_ != nullptr) return false;
        if (stride <= 0 || expected <= 0 || expected > stride) return false;
        if (owner == nullptr || alloc == nullptr || unmap == nullptr || release == nullptr || copy == nullptr ||
            initialize == nullptr) {
            return false;
        }

        // The host mirror first, and the device allocation only once it exists.
        // The other order has no good failure: a throwing mirror allocation
        // would have to hand the device memory straight back, and a free that
        // then failed would leave this object with no record of an allocation
        // the allocator may still own — reported as an ordinary refused open,
        // which is exactly the loss the retention contract exists to prevent.
        std::vector<DiagnosticRunTerminal> mirror;
        try {
            mirror.assign(static_cast<size_t>(stride) * PLATFORM_DIAGNOSTIC_TERMINAL_BANKS, DiagnosticRunTerminal{});
        } catch (...) {
            return false;
        }
        for (auto &entry : mirror)
            diagnostic_terminal_reset(&entry);

        const size_t bytes = calc_diagnostic_terminal_bank_bytes(stride);
        void *host_view = nullptr;
        void *device = alloc(owner, bytes, &host_view);
        if (device == nullptr) return false;

        host_banks_ = host_view;
        owner_ = owner;
        unmap_ = unmap;
        free_ = release;
        copy_ = copy;
        device_banks_ = device;
        bytes_ = bytes;
        backing_retained_ = false;
        // Keep cleanup ownership before the fallible write, but keep stride_
        // zero so neither an address nor a reservation can escape on failure.
        int rc = -1;
        try {
            rc = initialize(owner, device, host_view, mirror.data(), bytes);
        } catch (...) {
            // The allocation is still ours even when a callback throws.
        }
        if (rc != 0) {
            (void)release_backing(/*device_stopped=*/true);
            return false;
        }
        mirror_ = std::move(mirror);
        expected_ = expected;
        stride_ = stride;
        return true;
    }

    /**
     * Close the public side: no further reservation is admitted and every
     * outstanding one is invalidated. The backing stays mapped, and saying so is
     * part of closing — `backing_retained()` is true from here until a release
     * actually completes, so a destroyed object's allocation is a reported
     * outcome rather than one nobody can name.
     *
     * Invalidating a token proves nothing about the device. This is the half a
     * caller can always perform; `release_backing` is the half that needs a
     * proof.
     *
     * **Caller.** The same single collector thread that owns `open`,
     * `copy_bank` and `settle`. The reset it performs is the one writer the
     * credit gate's odd/even protocol admits, and the fields it clears are
     * plain: nothing here is safe against a concurrent reserve, which is why
     * every caller in this tree reaches it only from its collector's own
     * teardown.
     */
    void detach() noexcept {
        stride_ = 0;
        expected_ = 0;
        mirror_.clear();
        credits_.reset();
        // After the reset, so each parked token is a generation the pool no
        // longer recognises and giving it back releases nothing. The array is
        // emptied rather than left holding tokens no lookup can reach; whether
        // the storage those banks name is safe to free is the one question
        // `release_backing` asks, and it is asked below.
        {
            std::lock_guard<std::mutex> lk(quarantine_mu_);
            for (auto &parked : quarantined_)
                parked.release();
        }
        if (device_banks_ != nullptr) backing_retained_ = true;
    }

    /**
     * Give the backing back to the device allocator, once the caller can prove
     * no producer can still write it.
     *
     * `device_stopped` is that proof and is the caller's own observation — this
     * object cannot make it. Without it the allocation is **retained**, not
     * leaked: a producer that is still running would otherwise write into
     * memory the allocator has handed to someone else, which is the failure the
     * accepted contract refuses to trade for a tidy teardown.
     *
     * A non-zero status from the allocator is reported as `FreeFailed` and also
     * retains the allocation, because an allocator that refused the pointer may
     * still own it. Either retained outcome leaves `backing_retained()` true so
     * the collector's own teardown can report it the way it reports any other
     * release it could not prove.
     *
     * What `device_stopped` gates is whether `FreeFn` is called at all. Where
     * that callback hands the pointer — back to the allocator, or nowhere on a
     * teardown that abandons every device resource — is the callback's own
     * policy, which this object neither sees nor reports on.
     */
    enum class Release : int { NotOpen = 0, Released = 1, DeviceUnproved = 2, FreeFailed = 3 };

    Release release_backing(bool device_stopped) noexcept {
        detach();
        if (device_banks_ == nullptr) return Release::NotOpen;
        if (!device_stopped || free_ == nullptr) {
            backing_retained_ = true;
            return Release::DeviceUnproved;
        }
        // The mapping first, and its success recorded before the free is
        // attempted: a free that then fails leaves this array held with no
        // host view, and a later retry has to see that rather than taking a
        // mapping down twice.
        if (host_banks_ != nullptr && unmap_ != nullptr) {
            if (unmap_(owner_, device_banks_) != 0) {
                backing_retained_ = true;
                return Release::FreeFailed;
            }
            host_banks_ = nullptr;
        }
        const int rc = free_(owner_, device_banks_);
        if (rc != 0) {
            backing_retained_ = true;
            return Release::FreeFailed;
        }
        device_banks_ = nullptr;
        host_banks_ = nullptr;
        bytes_ = 0;
        backing_retained_ = false;
        return Release::Released;
    }

    /**
     * True while an allocation this object made is still mapped after a close.
     *
     * Either the device stop was not proved or the allocator refused the
     * pointer. A collector reports this rather than dropping it, because the
     * memory is still reachable by whatever it was mapped for.
     */
    bool backing_retained() const noexcept { return backing_retained_; }

    /** The retained allocation, for a caller that can later prove the stop. */
    uint64_t retained_backing_address() const noexcept {
        return backing_retained_ ? reinterpret_cast<uint64_t>(device_banks_) : 0;
    }

    /**
     * True while this array admits reservations.
     *
     * False after `detach()` even though the backing may still be mapped: the
     * two are deliberately different questions, and a run must not be admitted
     * against storage nobody will settle.
     */
    bool opened() const noexcept { return device_banks_ != nullptr && stride_ > 0; }
    int stride() const noexcept { return stride_; }
    int expected_producers() const noexcept { return expected_; }
    size_t bytes() const noexcept { return bytes_; }

    /** The address a producer publishes into, for this run's kernel arguments. */
    uint64_t device_address() const noexcept { return opened() ? reinterpret_cast<uint64_t>(device_banks_) : 0; }

    /**
     * The device address of `bank`'s first producer entry.
     *
     * This is what travels to the device: the host does the bank indexing, so a
     * producer adds only its own index and never derives a bank from one. 0
     * when no array is mapped or the index is not one of this array's, which
     * the device reads as "publish no accounting".
     */
    uint64_t bank_address(size_t bank) const noexcept {
        if (!opened() || bank >= PLATFORM_DIAGNOSTIC_TERMINAL_BANKS) return 0;
        const auto *entry =
            static_cast<const DiagnosticRunTerminal *>(device_banks_) + bank * static_cast<size_t>(stride_);
        return reinterpret_cast<uint64_t>(entry);
    }

    RunCredits &credits() noexcept { return credits_; }
    const RunCredits &credits() const noexcept { return credits_; }

    /**
     * Park a bank whose run ended with neither a completion nor a producer-stop
     * proof, keyed by the run that owns it.
     *
     * Such a bank is not the next run's to take: this run's producers may still
     * be writing it, and that is exactly what the stop proof would have settled.
     * Nor can it be left on the run's record — that record is reused by the
     * next admission, whose assignment would release the token silently, which
     * is the same premature return by a quieter route.
     *
     * So it is held here with its identity. The capacity stays out of the pool
     * (the reservation is still held), `reclaim` can still return exactly this
     * run's bank if the proof arrives later, and a bank still parked at teardown
     * is covered by the same unproved-release policy as the backing itself.
     *
     * Bounded by construction: there are `PLATFORM_DIAGNOSTIC_TERMINAL_BANKS`
     * reservations in existence, so the parked set cannot outgrow them.
     */
    bool quarantine(Reservation &&reservation) {
        // The identity is the token's own. Taking it as a second argument
        // would let a caller name one run while handing over another's bank,
        // and there is nothing here that could tell the two apart.
        if (!reservation.held()) return false;
        {
            std::lock_guard<std::mutex> lk(quarantine_mu_);
            for (auto &parked : quarantined_) {
                if (parked.held()) continue;
                parked = std::move(reservation);
                return true;
            }
        }
        // Unreachable by construction — this array has one slot per bank and
        // the token being parked is not in it, so a free slot always remains —
        // and still handled, because the alternative is the defect this whole
        // path exists to prevent. Nothing proved this run's producers stopped,
        // so the bank does not go back: the token is dropped and its capacity
        // stays out of the pool, which costs one bank and no safety.
        reservation.abandon();
        return false;
    }

    /**
     * Return exactly one run's parked bank, once that run's producers are
     * proved stopped. False when this run has none parked, which is the
     * ordinary answer for a run whose close already settled its own bank.
     */
    bool reclaim(uint64_t run_epoch) {
        if (run_epoch == 0) return false;
        std::lock_guard<std::mutex> lk(quarantine_mu_);
        for (auto &parked : quarantined_) {
            if (!parked.held() || parked.run_epoch() != run_epoch) continue;
            parked.release();
            return true;
        }
        return false;
    }

    /** Banks held for a stop proof that never came. */
    size_t quarantined_banks() const {
        std::lock_guard<std::mutex> lk(quarantine_mu_);
        size_t held = 0;
        for (const auto &parked : quarantined_) {
            if (parked.held()) held++;
        }
        return held;
    }

    /**
     * Whether a bank could still be reserved from this array.
     *
     * Advisory, like the pool predicate it forwards: an admission gate asks it
     * before committing to a pair, and `reserve` is still what takes the bank.
     */
    bool can_reserve() const noexcept { return opened() && credits_.can_reserve_terminal_bank(); }

    /**
     * Reserve the bank a run's producers publish into.
     *
     * The run identity travels with the token so that a snapshot can be matched
     * to the reservation that produced it, rather than to whichever run happens
     * to be reading.
     */
    Reservation reserve(uint64_t run_epoch) noexcept {
        if (!opened() || run_epoch == 0) return {};
        return credits_.reserve_terminal_bank(run_epoch);
    }

    /**
     * Copy one run's bank out of device memory into the host mirror.
     *
     * This is the step that makes the bank reusable, so it runs as soon as the
     * run's device work is settled — whether or not its artifact is anywhere
     * near being written. The copy is checked: a failed read leaves the mirror
     * untouched, and the caller reports an unknown accounting rather than
     * publishing a half-read bank as a result.
     */
    bool copy_bank(size_t bank) {
        if (!opened() || bank >= PLATFORM_DIAGNOSTIC_TERMINAL_BANKS) return false;
        const size_t offset = bank * static_cast<size_t>(stride_);
        const size_t span = static_cast<size_t>(stride_) * sizeof(DiagnosticRunTerminal);
        const auto *device_entry = static_cast<const DiagnosticRunTerminal *>(device_banks_) + offset;
        // Both addresses of this bank, so the read is decided by how the array
        // was mapped rather than by the platform it is on. A null host view is
        // the honest statement that there is none.
        const auto *host_entry =
            host_banks_ != nullptr ? static_cast<const DiagnosticRunTerminal *>(host_banks_) + offset : nullptr;
        return copy_(owner_, &mirror_[offset], device_entry, host_entry, span) == 0;
    }

    /** This run's producers, inside the host mirror; null when never copied. */
    const DiagnosticRunTerminal *mirrored_bank(size_t bank) const {
        if (mirror_.empty() || bank >= PLATFORM_DIAGNOSTIC_TERMINAL_BANKS) return nullptr;
        return mirror_.data() + bank * static_cast<size_t>(stride_);
    }

    /**
     * Copy a run's accounting out and reconcile it against the host's receipts.
     *
     * **Preconditions.** The run's device work is settled — its producers have
     * published and nothing on the device will write this bank again — and the
     * caller is the collector thread that owns this object. Neither is checkable
     * here: a bank read before its producers closed would report missing
     * participants, which is a true statement about what was published and a
     * misleading one about the run.
     *
     * The token is validated before anything is read: it must be this pool's,
     * for a terminal bank, from the current generation, and name the run being
     * settled. A token that fails any of those names a bank this object has no
     * claim to, so it is refused with the token left in the caller's hands
     * rather than silently copying the wrong bank and releasing someone else's
     * capacity.
     *
     * On success the host evidence is in the mirror before the bank is released,
     * so the reuse that follows cannot overwrite figures nobody has read.
     */
    RunReconciliation settle(
        Reservation &reservation, uint64_t run_epoch, uint64_t received, ProducerNote *notes_out = nullptr,
        int notes_capacity = 0, int *notes_filled = nullptr
    ) {
        RunReconciliation out;
        out.received = received;
        out.expected_producers = expected_;
        out.counts_unknown = true;
        out.verdict = simpler::dfx::runs::Verdict::CutUnknown;
        if (notes_filled != nullptr) *notes_filled = 0;

        if (!opened()) return out;
        if (run_epoch == 0 || reservation.run_epoch() != run_epoch) return out;
        if (!reservation.issued_by(&credits_, CreditKind::TerminalBank)) return out;

        const size_t bank = reservation.index();
        if (!copy_bank(bank)) {
            // The figures are unreadable, which is already reported; holding the
            // bank on top of that would deny capacity for a settled run.
            reservation.release();
            out.missing_producers = expected_;
            return out;
        }
        // Host evidence is settled before the device storage is handed on, and
        // that includes an owner's per-producer notes: they are read out of the
        // same mirror under the same identity, because after the release the
        // bank belongs to whichever run claims it next.
        const RunReconciliation reconciled =
            reconcile_run(mirrored_bank(bank), stride_, expected_, run_epoch, received);
        if (notes_out != nullptr && notes_filled != nullptr) {
            *notes_filled = collect_producer_notes(mirrored_bank(bank), stride_, run_epoch, notes_out, notes_capacity);
        }
        reservation.release();
        return reconciled;
    }

private:
    int stride_{0};
    int expected_{0};
    void *owner_{nullptr};
    UnmapFn unmap_{nullptr};
    FreeFn free_{nullptr};
    CopyFn copy_{nullptr};
    void *device_banks_{nullptr};
    // The host view of `device_banks_`, or null where the allocation has none.
    // Equal to the device address on an identically mapped platform, a distinct
    // registered address where registration produced one.
    void *host_banks_{nullptr};
    size_t bytes_{0};
    bool backing_retained_{false};
    std::vector<DiagnosticRunTerminal> mirror_;
    RunCredits credits_{};
    // Parked banks, and the one lock in this type: `quarantine` runs on the
    // writer thread that retires a run's slot while `reclaim` runs on the
    // thread closing another run's boundary, and those are not the single
    // caller the rest of this object documents.
    mutable std::mutex quarantine_mu_;
    std::array<Reservation, PLATFORM_DIAGNOSTIC_TERMINAL_BANKS> quarantined_{};
};

}  // namespace simpler::dfx::credit
