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

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

/**
 * The device allocations a caller minted here, and which runs are still using them.
 *
 * "Caller" is a distinction inside this device context, not a layer boundary: it means whoever
 * allocated through `device_malloc_ctx` — a `Worker.malloc` / `alloc_child_tensor` from above — as
 * opposed to the regions this context mints for itself through `allocate_tensor`. Nothing here is
 * an L3-to-L2 protocol: no control command, no mailbox field, no change to the arguments' wire
 * form. Host-space tensors do not appear either, in any form: they are copied into a
 * `RetainedTempBump` slice the runner owns for the run's whole lifetime, which is a lifetime this
 * table has no question about, and the slice is not a caller mint so it could not resolve anyway.
 *
 * A caller device buffer is allocated by this process on the caller's request and freed on the
 * caller's request: the allocator's right to release is the caller's throughout, and nothing here
 * takes it. What this holds is a *borrow* — the statement that a run may still read or write those
 * bytes — and a release the borrow outlives is refused rather than performed.
 *
 * **Only caller mints are recorded.** The entry is made by the caller-facing device-malloc path, so
 * an address this process allocated for itself — a workspace region, a retained temporary, an arena
 * bank — is absent, and `resolve` therefore refuses it. A caller has no business naming one, and the
 * absence is what makes that refusal automatic rather than a list to maintain.
 *
 * **A span resolves to its containing allocation, not to a key.** A tensor may sit at an offset
 * inside a larger buffer, so the unit of proof is the allocation that covers the whole span — the
 * same rule `MemoryAllocator::owning_allocation` applies for host mappings, and for the same reason:
 * release is per allocation, so a borrow must be too.
 *
 * **A borrow belongs to a run identity, not to a count.** `borrow` is keyed on the run's own
 * identity, so a release names the run that took it and cannot discharge another run's reference.
 * Re-borrowing under one identity replaces that identity's set, which is what a re-prepared run
 * needs.
 *
 * **A borrow is about lifetime, and says nothing about readability.** Holding an allocation is not
 * producing it: two runs may legitimately take the same immutable input, and neither makes the
 * other's bytes unreadable. So the borrow answers only "may this be released", and a second,
 * separate fact answers "may this be read on the host now".
 *
 * **That second fact is a declared write, made by whoever knows the direction.** The storage a
 * run's arguments reach the lane in carries no direction, so the lane cannot supply it — but the
 * runtime's own bind receives the orchestration signature, so it declares which of its caller
 * spans it will write. `written_by_other_run` is then exactly the question a host reader has:
 * whether a *different* live run has said it produces these bytes. A shared read-only input is
 * declared by nobody and stays readable, which is what keeps every path that does not produce a
 * device tensor on the behaviour it already had.
 *
 * A declaration is keyed on the same identity as the borrow and released with it, so it cannot
 * outlive the run that made it — including when that run held no borrow, which is why the release
 * discharges the two facts independently. A span that resolves to no recorded allocation is
 * ignored: there is nothing to protect.
 *
 * **An unproven run's facts are permanent, and they belong to the allocation.** When a run ended
 * without its last consumer being proven finished, the device may still be reading those bytes and
 * may still be writing the ones it declared; no later event can prove otherwise. So the allocation
 * keeps both marks for the process's remaining life: `free` refuses from then on, and a host read
 * of those bytes is refused from then on. They sit on the allocation rather than on the run's
 * identity because the run is gone and its slot — and so its identity — is handed to the next run,
 * which must inherit neither its permissions nor its debts. In one sentence: every path that ends
 * a run unproven also poisons the lane, so what these marks actually guard is a `free` or a host
 * read arriving *after* that — the lane is gone, so the refusal has to live on the allocation.
 *
 * This table is not teardown's authority and makes no claim about it. What a retained reference
 * says is that *this* path must not release the allocation; what teardown then does with it — a
 * successful reset that ends the generation, a failed release it records, a quarantine — is
 * decided where those outcomes are known, not here.
 *
 * Its own mutex, and a leaf: nothing called while it is held takes another lock.
 */
class CallerDeviceBuffers {
public:
    /** One span of a caller allocation a run names. */
    struct Span {
        uint64_t addr{0};
        uint64_t bytes{0};
    };

    /** A recorded allocation's own extent. */
    struct Allocation {
        uint64_t base{0};
        uint64_t bytes{0};
    };

    /**
     * Record one caller mint. Replaces any stale entry at the same base.
     *
     * A replacement is not an error: the platform may hand back an address a previous allocation
     * released, and the previous entry can only still be here if it was never borrowed — a borrowed
     * one cannot be freed. The new entry carries neither of the old one's marks, which is the same
     * statement: an allocation whose release was refused was never handed back to be re-minted.
     */
    void record(void *base, std::size_t bytes) {
        if (base == nullptr || bytes == 0) return;
        std::scoped_lock lk(mu_);
        allocations_[reinterpret_cast<uint64_t>(base)] = Entry{static_cast<uint64_t>(bytes)};
    }

    /** Whether `base` is a recorded caller mint. */
    bool recorded(void *base) const {
        if (base == nullptr) return false;
        std::scoped_lock lk(mu_);
        return allocations_.count(reinterpret_cast<uint64_t>(base)) != 0;
    }

    /**
     * The recorded allocation covering `[addr, addr + bytes)`, if any.
     *
     * A zero-length span resolves to nothing: it names no bytes, so there is nothing to own and
     * nothing to borrow. An empty tensor therefore stays on the path it has today.
     */
    bool resolve(uint64_t addr, uint64_t bytes, Allocation *out) const {
        if (addr == 0 || bytes == 0) return false;
        std::scoped_lock lk(mu_);
        return resolve_locked(addr, bytes, out);
    }

    /**
     * Take `identity`'s borrow over every span's containing allocation.
     *
     * All or nothing: a span that does not resolve leaves no borrow at all, so a caller that mixes
     * a provable buffer with an unprovable one gets the refusal rather than half a reference. The
     * previous borrow for this identity is dropped first — a re-prepared run names its spans again,
     * and keeping both sets would leak a reference the run no longer has.
     *
     * @return true when every span resolved and the borrow is held.
     */
    bool borrow(uint64_t identity, const Span *spans, std::size_t count) {
        if (identity == 0) return false;
        std::scoped_lock lk(mu_);
        std::vector<uint64_t> taken;
        taken.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            Allocation allocation;
            if (!resolve_locked(spans[i].addr, spans[i].bytes, &allocation)) return false;
            taken.push_back(allocation.base);
        }
        if (taken.empty()) {
            borrows_.erase(identity);
            return true;
        }
        borrows_[identity] = std::move(taken);
        return true;
    }

    /**
     * Discharge everything `identity` established, or keep it for good.
     *
     * Both facts are discharged, and independently: the borrow over the allocations this run
     * named, and its declaration of which of them it produces. A run can hold either without the
     * other — an all-or-nothing borrow may have been refused while the declaration over one
     * resolved span stood — and a slot holds exactly one run at a time, so a later run under this
     * identity must inherit neither.
     *
     * `keep` is for a run whose last consumer could not be proven finished. It does not merely
     * delay the drop: each allocation is marked instead, so the release stays refused and the
     * bytes this run had declared stay unreadable, because nothing later can establish what the
     * device no longer holds or no longer writes. Marking is an update in place, so this call
     * allocates nothing and cannot fail on a teardown path.
     */
    void release(uint64_t identity, bool keep) {
        std::scoped_lock lk(mu_);
        const auto borrowed = borrows_.find(identity);
        if (borrowed != borrows_.end()) {
            if (keep) {
                for (const uint64_t base : borrowed->second)
                    mark_release_refused(base);
            }
            borrows_.erase(borrowed);
        }
        const auto written = writes_.find(identity);
        if (written == writes_.end()) return;
        if (keep) {
            for (const uint64_t base : written->second) {
                // Both marks, for the same reason: a consumer that may still be writing these
                // bytes may still be reading them too, so the pages cannot go back either. The
                // borrow usually said that already — this reaches the allocation a run declared
                // while holding no borrow, which is what a refused all-or-nothing borrow leaves.
                mark_write_unproven(base);
                mark_release_refused(base);
            }
        }
        writes_.erase(written);
    }

    /**
     * Whether the allocation based at `base` may not be released — because a run still holds it,
     * or because one ended while it may still have been reaching it.
     */
    bool borrowed(void *base) const {
        if (base == nullptr) return false;
        std::scoped_lock lk(mu_);
        return borrowed_locked(reinterpret_cast<uint64_t>(base));
    }

    /**
     * Declare that `identity` writes the allocations covering `spans`.
     *
     * Made by the run's own bind, which is where the orchestration signature says which arguments
     * are outputs. Replaces this identity's previous declaration, so a re-prepared run states its
     * writes again rather than accumulating them, and is discharged by `release`.
     *
     * A span resolving to no recorded allocation is skipped rather than refused: this is a
     * statement about caller buffers, and an address that is not one protects nothing. It is also
     * not *nothing* — it is an unknown — so `unresolved_out` reports how many spans were skipped.
     * A caller that named a span it believes is produced can tell "this run produces no caller
     * buffer" from "this run produces bytes whose owner this context cannot prove", and the
     * readability fence is exactly as wide as what was recorded: it says nothing about a span
     * that did not resolve, and no protection is claimed for one.
     *
     * @return true when the statement is recorded. False means it is not, and the caller must not
     *         proceed as if it were: an undeclared producer reads as no producer, so a run whose
     *         declaration failed has to fail before it can write anything. The only way to fail is
     *         the allocation this statement needs. What makes that safe is the caller's refusal,
     *         not the state left behind: the previous declaration stays, and it may well name
     *         spans disjoint from the new one, so it is no substitute — the bind that could not
     *         declare is rejected before it publishes anything, which is why nothing of this run's
     *         ever reaches the device for another run to read.
     */
    bool
    declare_writes(uint64_t identity, const Span *spans, std::size_t count, std::size_t *unresolved_out = nullptr) {
        if (unresolved_out != nullptr) *unresolved_out = 0;
        if (identity == 0) return false;
        std::scoped_lock lk(mu_);
        try {
            std::vector<uint64_t> written;
            written.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                Allocation allocation;
                if (!resolve_locked(spans[i].addr, spans[i].bytes, &allocation)) {
                    if (unresolved_out != nullptr) ++*unresolved_out;
                    continue;
                }
                written.push_back(allocation.base);
            }
            if (written.empty()) {
                writes_.erase(identity);
                return true;
            }
            writes_[identity] = std::move(written);
        } catch (...) {
            return false;
        }
        return true;
    }

    /**
     * Whether `[addr, addr + bytes)` has no readable content for `identity` yet.
     *
     * The question a host reader inside this process asks before reading a caller device buffer's
     * bytes: those bytes have no defined content until their producer finishes. A shared immutable
     * input has no declared writer and so stays readable — holding an allocation is not producing
     * it, which is why this asks about declarations and not about borrows.
     *
     * True when another live run has declared it produces them, or when a run that had declared
     * them ended without being proven finished. `identity` is excluded from the first, because a
     * run's own output is its own business: it is building the graph that will write those bytes.
     * It is not excluded from the second — that mark belongs to the allocation, not to a run, so
     * the next run to occupy this slot is refused just the same.
     */
    bool written_by_other_run(uint64_t identity, uint64_t addr, uint64_t bytes) const {
        if (addr == 0 || bytes == 0) return false;
        std::scoped_lock lk(mu_);
        Allocation allocation;
        if (!resolve_locked(addr, bytes, &allocation)) return false;
        const auto entry = allocations_.find(allocation.base);
        if (entry != allocations_.end() && entry->second.write_unproven) return true;
        for (const auto &[writer, written] : writes_) {
            if (writer == identity) continue;
            for (const uint64_t one : written) {
                if (one == allocation.base) return true;
            }
        }
        return false;
    }

    /**
     * Forget `base` so the caller's release may proceed, or refuse.
     *
     * Check and forget are one step under this lock, so a borrow taken between a caller's question
     * and its free cannot be missed. Forgetting is deliberately separate from the platform free the
     * caller then performs: this class holds no platform vocabulary, and the entry has to be gone
     * before those pages can be handed back.
     *
     * @return true when nothing holds it and the entry is now gone.
     */
    bool forget_if_unborrowed(void *base) {
        if (base == nullptr) return false;
        const uint64_t key = reinterpret_cast<uint64_t>(base);
        std::scoped_lock lk(mu_);
        if (borrowed_locked(key)) return false;
        allocations_.erase(key);
        return true;
    }

    /** How many runs have declared a write. */
    std::size_t write_declaration_count() const {
        std::scoped_lock lk(mu_);
        return writes_.size();
    }

    /** How many runs hold a borrow. */
    std::size_t borrow_count() const {
        std::scoped_lock lk(mu_);
        return borrows_.size();
    }

    /** How many allocations are retained for the process's remaining life. */
    std::size_t retained_count() const {
        std::scoped_lock lk(mu_);
        std::size_t count = 0;
        for (const auto &[base, entry] : allocations_) {
            (void)base;
            if (entry.release_refused) ++count;
        }
        return count;
    }

    /** Recorded caller mints. */
    std::size_t allocation_count() const {
        std::scoped_lock lk(mu_);
        return allocations_.size();
    }

private:
    /** One recorded caller mint: its extent, and the two facts that can outlive every run. */
    struct Entry {
        uint64_t bytes{0};
        // The allocation may never be released: a run that could still reach it ended without its
        // last consumer being proven finished.
        bool release_refused{false};
        // Its bytes may never be read on the host: a run that had declared it produces them ended
        // the same way, so nothing can establish that the write completed.
        bool write_unproven{false};
    };

    // Linear over the recorded mints, under the leaf mutex. The set is only what a caller minted
    // through this context — not its workspace, retained temporaries or arena banks — so it is
    // small, and every question is asked on a bind or a free rather than per task. One path pays
    // it per element: a `get_tensor_data` loop over a child-memory region whose platform host view
    // was refused (issue #1531) resolves again on each element read.
    bool resolve_locked(uint64_t addr, uint64_t bytes, Allocation *out) const {
        for (const auto &[base, entry] : allocations_) {
            if (addr < base) continue;
            // Both halves matter: the span has to start inside the allocation and end inside it,
            // and the end test is written against the remaining extent so a caller-supplied length
            // cannot overflow the address into a pass.
            if (addr - base > entry.bytes || bytes > entry.bytes - (addr - base)) continue;
            if (out != nullptr) *out = Allocation{base, entry.bytes};
            return true;
        }
        return false;
    }

    void mark_release_refused(uint64_t base) {
        const auto entry = allocations_.find(base);
        if (entry != allocations_.end()) entry->second.release_refused = true;
    }

    void mark_write_unproven(uint64_t base) {
        const auto entry = allocations_.find(base);
        if (entry != allocations_.end()) entry->second.write_unproven = true;
    }

    bool borrowed_locked(uint64_t base) const {
        const auto entry = allocations_.find(base);
        if (entry != allocations_.end() && entry->second.release_refused) return true;
        for (const auto &[identity, held] : borrows_) {
            (void)identity;
            for (const uint64_t one : held) {
                if (one == base) return true;
            }
        }
        return false;
    }

    mutable std::mutex mu_;
    std::unordered_map<uint64_t, Entry> allocations_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> borrows_;
    // Which allocations each run has said it produces. Separate from `borrows_` because the two
    // answer different questions and are established at different moments: the borrow at
    // admission, the declaration at that run's own bind.
    std::unordered_map<uint64_t, std::vector<uint64_t>> writes_;
};
