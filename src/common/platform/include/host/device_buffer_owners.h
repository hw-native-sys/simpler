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
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <vector>

/** Per-context lifetime loans and producer fences for caller and native domain storage.
 * Domain views retain one allocation owner; only caller mints permit caller free.
 * Platform releases run outside the leaf mutex, between reservation and confirmation.
 */
class DeviceBufferOwners {
public:
    /** One device storage span a run names. */
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
        const uint64_t key = reinterpret_cast<uint64_t>(base);
        if (!valid_span(key, bytes)) throw std::runtime_error("invalid caller allocation extent");
        std::scoped_lock lk(mu_);
        for (const auto &[existing, entry] : allocations_) {
            if (overlaps(key, bytes, existing, entry.bytes) &&
                (existing != key || entry.domain_managed || borrowed_locked(existing))) {
                throw std::runtime_error("caller allocation overlaps a live or retained owner");
            }
        }
        allocations_[key] = Entry{static_cast<uint64_t>(bytes)};
    }

    /** Whether `base` is a recorded caller mint. */
    bool recorded(void *base) const {
        if (base == nullptr) return false;
        std::scoped_lock lk(mu_);
        const auto found = allocations_.find(reinterpret_cast<uint64_t>(base));
        return found != allocations_.end() && !found->second.domain_managed;
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
     * Known allocations stay borrowed even when another span is external. The result reports
     * full coverage for native admission, not whether any lifetime references were acquired.
     * Re-borrowing replaces this identity's set only after the new set has been collected.
     *
     * @return true when every span resolved; false still requires release(identity, keep).
     */
    bool borrow(uint64_t identity, const Span *spans, std::size_t count) {
        if (identity == 0) return false;
        std::scoped_lock lk(mu_);
        std::vector<uint64_t> taken;
        taken.reserve(count);
        bool all_resolved = true;
        for (std::size_t i = 0; i < count; ++i) {
            Allocation allocation;
            if (resolve_locked(spans[i].addr, spans[i].bytes, &allocation)) {
                if (allocations_.at(allocation.base).revoked) {
                    throw std::runtime_error("device span belongs to a revoked native domain");
                }
                taken.push_back(allocation.base);
            } else {
                all_resolved = false;
            }
        }
        if (taken.empty()) {
            borrows_.erase(identity);
        } else {
            borrows_[identity] = std::move(taken);
        }
        return all_resolved;
    }

    /**
     * Discharge everything `identity` established, or keep it for good.
     *
     * Both facts are discharged, and independently: the borrow over the allocations this run
     * named, and its declaration of which of them it produces. A run can hold either without the
     * other — a declaration may be recorded without an admission borrow — and a slot holds
     * exactly one run at a time, so a later run under this
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
                // through a declaration even if no admission borrow was taken.
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
     * Declare the registered extents this run produces, independently of its loans.
     * Unknown spans contribute no protection and are counted in `unresolved_out`.
     * A revoked owner rejects the declaration. Allocation failure preserves the
     * previous declaration and rejects the bind before device publication.
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
                if (allocations_.at(allocation.base).revoked) return false;
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
     * The question a host reader inside this process asks before reading registered device storage's
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
        if (entry != allocations_.end() && (entry->second.write_unproven || entry->second.revoked)) return true;
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
        const auto entry = allocations_.find(key);
        if (entry != allocations_.end() && entry->second.domain_managed) return false;
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

    std::size_t caller_allocation_count() const {
        std::scoped_lock lk(mu_);
        std::size_t count = 0;
        for (const auto &[base, entry] : allocations_) {
            (void)base;
            if (!entry.domain_managed) ++count;
        }
        return count;
    }

    /** Recorded storage extents. */
    std::size_t allocation_count() const {
        std::scoped_lock lk(mu_);
        return allocations_.size();
    }

    /** Register all native views backed by one communication allocation. */
    bool register_domain_owner(uint64_t session, uint64_t allocation, const Span *spans, std::size_t count) {
        if (session == 0 || spans == nullptr || count == 0) return false;
        std::scoped_lock lk(mu_);
        if (domain_session_releases_.find(session) != domain_session_releases_.end()) return false;
        const auto key = std::make_pair(session, allocation);
        if (domain_owners_.find(key) != domain_owners_.end()) return false;
        try {
            DomainOwner owner;
            owner.bases.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                const auto &span = spans[i];
                if (!valid_span(span.addr, span.bytes)) return false;
                for (const auto &[base, entry] : allocations_) {
                    if (overlaps(span.addr, span.bytes, base, entry.bytes)) return false;
                }
                for (std::size_t j = 0; j < i; ++j) {
                    if (overlaps(span.addr, span.bytes, spans[j].addr, spans[j].bytes)) return false;
                }
                owner.bases.push_back(span.addr);
            }
            auto allocations = allocations_;
            auto domains = domain_owners_;
            for (std::size_t i = 0; i < count; ++i) {
                Entry entry{spans[i].bytes};
                entry.domain_managed = true;
                allocations.emplace(spans[i].addr, entry);
            }
            domains.emplace(key, std::move(owner));
            allocations_.swap(allocations);
            domain_owners_.swap(domains);
        } catch (...) {
            return false;
        }
        return true;
    }

    /** Revoke new uses, then reserve the sole platform release when no consumer remains. */
    bool begin_domain_release(uint64_t session, uint64_t allocation) {
        std::scoped_lock lk(mu_);
        if (domain_session_releases_.find(session) != domain_session_releases_.end()) return false;
        const auto found = domain_owners_.find(std::make_pair(session, allocation));
        if (found == domain_owners_.end()) return false;
        auto &owner = found->second;
        if (owner.state == DomainReleaseState::RELEASING || owner.state == DomainReleaseState::UNCONFIRMED)
            return false;
        owner.state = DomainReleaseState::REVOKED;
        for (uint64_t base : owner.bases)
            allocations_.at(base).revoked = true;
        if (domain_busy_locked(owner)) return false;
        owner.state = DomainReleaseState::RELEASING;
        return true;
    }

    /** Forget only an owner whose reserved platform release was confirmed successful. */
    bool finish_domain_release(uint64_t session, uint64_t allocation) {
        std::scoped_lock lk(mu_);
        const auto found = domain_owners_.find(std::make_pair(session, allocation));
        if (found == domain_owners_.end() || found->second.state != DomainReleaseState::RELEASING) return false;
        if (domain_busy_locked(found->second)) return false;
        for (uint64_t base : found->second.bases)
            allocations_.erase(base);
        domain_owners_.erase(found);
        return true;
    }

    /** An ambiguous platform outcome cannot authorize a retry, new loan, or address reuse. */
    bool mark_domain_release_unconfirmed(uint64_t session, uint64_t allocation) {
        std::scoped_lock lk(mu_);
        const auto found = domain_owners_.find(std::make_pair(session, allocation));
        if (found == domain_owners_.end() || found->second.state != DomainReleaseState::RELEASING) return false;
        found->second.state = DomainReleaseState::UNCONFIRMED;
        for (uint64_t base : found->second.bases) {
            auto &entry = allocations_.at(base);
            entry.revoked = true;
            entry.release_refused = true;
            entry.write_unproven = true;
        }
        return true;
    }

    /** Whether an allocation was enrolled, including revoked or quarantined owners. */
    bool has_domain_owner(uint64_t session, uint64_t allocation) const {
        std::scoped_lock lk(mu_);
        return domain_owners_.find(std::make_pair(session, allocation)) != domain_owners_.end();
    }

    /** Revoke a whole session before one implicit backend destruction.
     * Reservation is all-or-none; a busy session remains revoked until its users retire.
     * An individually reserved or ambiguous release cannot be consumed a second time.
     */
    bool begin_domain_session_release(uint64_t session) {
        if (session == 0) return false;
        std::scoped_lock lk(mu_);
        const auto pending = domain_session_releases_.find(session);
        if (pending != domain_session_releases_.end() && pending->second != DomainReleaseState::REVOKED) return false;
        for (const auto &[key, owner] : domain_owners_) {
            if (key.first == session &&
                (owner.state == DomainReleaseState::RELEASING || owner.state == DomainReleaseState::UNCONFIRMED))
                return false;
        }
        // Allocate the reservation before changing any owner.
        domain_session_releases_[session] = DomainReleaseState::REVOKED;
        bool busy = false;
        for (auto &[key, owner] : domain_owners_) {
            if (key.first != session) continue;
            owner.state = DomainReleaseState::REVOKED;
            for (uint64_t base : owner.bases)
                allocations_.at(base).revoked = true;
            busy = domain_busy_locked(owner) || busy;
        }
        if (busy) return false;
        for (auto &[key, owner] : domain_owners_) {
            if (key.first == session) owner.state = DomainReleaseState::RELEASING;
        }
        domain_session_releases_.at(session) = DomainReleaseState::RELEASING;
        return true;
    }

    /** A failed/ambiguous backend close quarantines all grants without retry. */
    bool finish_domain_session_release(uint64_t session, bool confirmed) {
        std::scoped_lock lk(mu_);
        const auto pending = domain_session_releases_.find(session);
        if (pending == domain_session_releases_.end() || pending->second != DomainReleaseState::RELEASING) return false;
        for (const auto &[key, owner] : domain_owners_) {
            if (key.first == session && (owner.state != DomainReleaseState::RELEASING || domain_busy_locked(owner)))
                return false;
        }
        for (auto it = domain_owners_.begin(); it != domain_owners_.end();) {
            if (it->first.first != session) {
                ++it;
                continue;
            }
            if (confirmed) {
                for (uint64_t base : it->second.bases)
                    allocations_.erase(base);
                it = domain_owners_.erase(it);
            } else {
                it->second.state = DomainReleaseState::UNCONFIRMED;
                for (uint64_t base : it->second.bases) {
                    auto &entry = allocations_.at(base);
                    entry.revoked = entry.release_refused = entry.write_unproven = true;
                }
                ++it;
            }
        }
        if (confirmed) domain_session_releases_.erase(pending);
        else pending->second = DomainReleaseState::UNCONFIRMED;
        return true;
    }

private:
    /** One storage extent and the facts that can outlive every run. */
    struct Entry {
        uint64_t bytes{0};
        // The allocation may never be released: a run that could still reach it ended without its
        // last consumer being proven finished.
        bool release_refused{false};
        // Its bytes may never be read on the host: a run that had declared it produces them ended
        // the same way, so nothing can establish that the write completed.
        bool write_unproven{false};
        bool domain_managed{false};
        bool revoked{false};
    };

    enum class DomainReleaseState { ACTIVE, REVOKED, RELEASING, UNCONFIRMED };
    struct DomainOwner {
        std::vector<uint64_t> bases;
        DomainReleaseState state{DomainReleaseState::ACTIVE};
    };

    static bool valid_span(uint64_t addr, uint64_t bytes) {
        return addr != 0 && bytes != 0 && bytes <= std::numeric_limits<uint64_t>::max() - addr;
    }

    static bool overlaps(uint64_t a, uint64_t bytes_a, uint64_t b, uint64_t bytes_b) {
        return a <= b ? b - a < bytes_a : a - b < bytes_b;
    }

    bool domain_busy_locked(const DomainOwner &owner) const {
        for (uint64_t base : owner.bases) {
            if (borrowed_locked(base)) return true;
            for (const auto &[identity, written] : writes_) {
                (void)identity;
                for (uint64_t one : written) {
                    if (one == base) return true;
                }
            }
        }
        return false;
    };

    // This leaf table contains caller allocations and explicitly granted domain
    // views. Resolution is per argument or host read, never per physical task.
    bool resolve_locked(uint64_t addr, uint64_t bytes, Allocation *out) const {
        if (bytes == 0) return false;
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
    std::map<std::pair<uint64_t, uint64_t>, DomainOwner> domain_owners_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> borrows_;
    // Which allocations each run has said it produces. Separate from `borrows_` because the two
    // answer different questions and are established at different moments: the borrow at
    // admission, the declaration at that run's own bind.
    std::unordered_map<uint64_t, std::vector<uint64_t>> writes_;
    std::map<uint64_t, DomainReleaseState> domain_session_releases_;
};
