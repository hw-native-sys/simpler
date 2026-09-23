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
#include <vector>

#include "runtime_c_api.h"

/**
 * One budget and one ownership ledger for the workspace regions of a device
 * runtime context: the per-slot retained temporary staging buffer and the three
 * pooled arena regions (GM heap, GM shared memory, prebuilt runtime arena).
 *
 * Off unless a caller configures a finite budget. While off, every consumer
 * allocates exactly as it did before and this class holds nothing.
 *
 * What it owns once on: the backing blocks. A consumer holds the right to use a
 * region of a block, not the right to release it. Two questions are kept apart,
 * because a published address staying mapped does not make its bytes free to
 * overwrite:
 *
 *   - capacity: does an existing block still fit this request?
 *   - permission: has every consumer of that block's previous generation
 *     finished, so its contents may be overwritten?
 *
 * A request a live generation cannot satisfy takes a new block inside the
 * budget, and the old generation stays exactly where it is until its last
 * consumer retires. Growth therefore means "a newer plan got a bigger block",
 * never "the old block moved".
 *
 * Two kinds of ownership are also kept apart, and conflating them is what makes
 * a close either unsafe or impossible:
 *
 *   - the **current backing** of a region is the address that region is
 *     published at right now. It belongs to the device context, not to any run,
 *     so it survives every run retiring and is never taken away to fund another
 *     region's growth. It is released when the context tears down.
 *   - a **run reference** is one executing run's claim on a block, identified by
 *     its run epoch. Only these decide whether a close must wait.
 *
 * Epoch 0 is not a run: it is the context itself, which is what the eager
 * device initialization allocates under. Recording it as a run reference would
 * leave a consumer that no run can ever finish, so it names a current backing
 * and nothing else.
 *
 * An **obsolete generation** — a block its region has since republished
 * elsewhere, with no run reference left — is the only storage growth may
 * reclaim, and only after its host mapping is gone.
 *
 * Retirement is decided from facts the run path reports at the boundaries that
 * produce them (`note_run_fact`), never inferred from a phase word or from the
 * return code a caller happens to receive. A block whose last consumer cannot
 * be proven finished is quarantined whole — never reused, never individually
 * released, and excluded from the allocator's terminal sweep — because handing
 * those bytes back would let another allocation receive a range an unknown
 * consumer may still be writing.
 *
 * Scope: partial accounting. External tensors, run-result and diagnostics
 * regions, registered code and device ELF, RTS argument blocks and provider
 * memory are outside this budget, so its total is not a Worker-wide or
 * device-wide memory ceiling. `coverage_is_partial` says so in every report.
 */
class WorkspaceManager {
public:
    /** Which consumer family a block belongs to. */
    enum class Domain : uint32_t {
        /** The three pooled arena regions: device-side execution scratch. */
        ExecScratch = 0,
        /** The per-slot retained temporary buffer: host argument staging. */
        HostStaging = 1,
    };

    /**
     * Which single consumer region a block belongs to.
     *
     * `domain` alone is too coarse to own anything: the three arena regions of
     * every bank would share one pool, so a growing GM heap could be handed a
     * block a still-attached shared-memory region is published at. `index`
     * names the one region — the arena region within its bank, or the pipeline
     * slot of a staging buffer — and a block is only ever offered back to the
     * region that published it.
     */
    struct RegionKey {
        Domain domain{Domain::ExecScratch};
        uint32_t index{0};

        bool operator==(const RegionKey &other) const { return domain == other.domain && index == other.index; }
    };

    /** The three regions one arena bank publishes, in their commit order. */
    enum class ArenaRegion : uint32_t { GmHeap = 0, GmSm = 1, RuntimePool = 2, kCount = 3 };

    /** The key naming one bank's arena region. */
    static RegionKey arena_region(uint32_t bank, ArenaRegion region) {
        return RegionKey{
            Domain::ExecScratch, bank * static_cast<uint32_t>(ArenaRegion::kCount) + static_cast<uint32_t>(region)
        };
    }

    /** The key naming one pipeline slot's staging buffer. */
    static RegionKey staging_region(uint32_t pipeline_slot) { return RegionKey{Domain::HostStaging, pipeline_slot}; }

    /** How a block's last-consumer state stands right now. */
    enum class BlockState : uint32_t {
        /** Referenced by at least one run whose facts are still incomplete. */
        Referenced = 0,
        /** No reference left: contents may be overwritten and bytes released. */
        ProvenUnused = 1,
        /** An ordinary release was attempted and the platform free failed. */
        ReleaseUnconfirmed = 2,
        /** Last consumer unprovable: excluded from every release path. */
        Quarantined = 3,
    };

    /** What this manager may still admit. */
    enum class Admission : uint32_t { Open = 0, DrainOnly = 1, Closed = 2 };

    /**
     * The device allocation this manager owns blocks through.
     *
     * `acquire` must reserve its own bookkeeping before the platform call and
     * commit it without allocating afterwards; it returns nullptr on failure
     * having changed nothing. `release` returns 0 when the bytes are gone and a
     * non-zero platform code when they are not — in which case the backend
     * keeps the allocation recorded, so this manager keeps charging for it
     * rather than dropping the pointer or reporting a release that did not
     * happen.
     */
    struct Backend {
        void *ctx{nullptr};
        void *(*acquire)(void *ctx, size_t bytes){nullptr};
        int (*release)(void *ctx, void *base){nullptr};
    };

    /**
     * Facts the run path reports, each at the boundary that produces it.
     *
     * None of them is a return code this manager interprets on its own: the
     * caller reports a fact only where that fact became true.
     */
    enum class RunFact : uint32_t {
        /** A launch transaction handed this run device work. */
        Launched = 0,
        /** A drain was entered. Published before the call, so a drain that
            threw or lost its device attach is never mistaken for one that did
            not happen. */
        DrainAttempted = 1,
        /** A drain returned success: this run's device work is finished. */
        DrainProvedComplete = 2,
        /** This run never reached the device, so it has no device consumer. */
        NoDeviceSubmission = 3,
        /** The copy-back consumer returned; no copy-back reader remains. */
        CopybackReturned = 4,
        /** This run's bindings were released successfully. */
        BindingsReleased = 5,
        /** The run context is gone: no further fact about it can arrive. */
        ContextDestroyed = 6,
    };

    /**
     * References one block can carry at once.
     *
     * A reference is dropped when its run retires or when its run's context is
     * destroyed, and a slot admits no new run until its previous one has been
     * finalized, so the live references on one block cannot outnumber the
     * pipeline's slots. The two spare entries exist so that reaching the bound
     * is a refusal rather than a truncation: a reference this manager cannot
     * record is a reference it would silently stop protecting.
     */
    static constexpr size_t kMaxBlockRefs = PTO_PIPELINE_MAX_DEPTH + 2;

    /** The identity workspace requests made outside any run belong to. */
    static constexpr uint64_t kContextEpoch = 0;

    WorkspaceManager() = default;
    WorkspaceManager(const WorkspaceManager &) = delete;
    WorkspaceManager &operator=(const WorkspaceManager &) = delete;

    /**
     * Turn the manager on with a finite budget, once.
     *
     * @return false when the budget is zero, when one is already latched, or
     *         when the backend is incomplete. A refused configuration leaves
     *         the manager off, which is the unchanged default behaviour.
     */
    bool configure(uint64_t limit_bytes, const Backend &backend) {
        std::scoped_lock lk(mu_);
        if (enabled_ || limit_bytes == 0) return false;
        if (backend.acquire == nullptr || backend.release == nullptr) return false;
        limit_bytes_ = limit_bytes;
        backend_ = backend;
        enabled_ = true;
        return true;
    }

    bool enabled() const {
        std::scoped_lock lk(mu_);
        return enabled_;
    }

    /**
     * Acquire a block for one consumer region, reusing an existing generation
     * when it both fits and has no consumer left.
     *
     * All-or-nothing per call: a request that cannot be satisfied inside the
     * budget, or whose device allocation fails, leaves every published address,
     * capacity and reference exactly as it found them and returns nullptr.
     *
     * @param slot       pipeline slot this plan belongs to
     * @param run_epoch  process-unique identity of the run making the plan
     */
    void *acquire(const RegionKey &region, uint64_t run_epoch, size_t bytes) {
        std::scoped_lock lk(mu_);
        if (!enabled_ || bytes == 0) return nullptr;
        if (admission_ != Admission::Open && admission_ != Admission::DrainOnly) return nullptr;

        for (Block &b : blocks_) {
            // A released block's address belongs to the platform again, so the
            // record is history: reusing it would hand back freed memory.
            if (b.released) continue;
            if (!(b.region == region)) continue;
            if (b.quarantined || b.release_unconfirmed) continue;
            if (b.bytes < bytes) continue;
            if (b.ref_count != 0) continue;  // capacity is not permission
            if (!add_ref(b, run_epoch)) return nullptr;
            // Handed out as this region's backing, so it stops being an
            // obsolete generation now rather than at publication: between the
            // two, growth elsewhere must not reclaim the block this plan is
            // already writing.
            b.current = true;
            return b.base;
        }
        return publish_new_block(region, run_epoch, bytes);
    }

    /**
     * Record that this run uses the block already published at `base`.
     *
     * A run whose request the current block satisfies allocates nothing, so
     * without this it would read and write a block it never registered as a
     * consumer of — and a later growth would see no reference and treat the
     * contents as free to overwrite. Reuse has to register ownership for the
     * same reason a fresh allocation does.
     *
     * `kContextEpoch` records no reference: it says the context, not a run, is
     * using its own current backing. The ownership and quarantine checks still
     * apply, so a caller's failure path stays truthful.
     *
     * @return false when `base` is not a live block of this manager, or when
     *         its reference table is full — a refusal, never a silent drop
     */
    bool reference(void *base, uint64_t run_epoch) {
        std::scoped_lock lk(mu_);
        Block *b = find_locked(base);
        if (b == nullptr || b->quarantined) return false;
        return add_ref(*b, run_epoch);
    }

    /**
     * Record that `region` is now published at `base`.
     *
     * Called only once the plan that took `base` is live — after the arena
     * transaction published every region, or after the slot started naming a
     * grown staging block. Every older generation of the same region becomes
     * obsolete here and nowhere else: a plan that failed leaves its predecessor
     * current, so the addresses that failure preserved stay protected from the
     * reclamation below.
     */
    void note_published(const RegionKey &region, void *base) {
        std::scoped_lock lk(mu_);
        Block *published = find_locked(base);
        if (published == nullptr) return;
        for (Block &b : blocks_) {
            if (b.released || b.base == base) continue;
            if (!(b.region == region)) continue;
            b.current = false;
        }
        published->current = true;
    }

    /**
     * Drop this run's reference to the block at `base`.
     *
     * A consumer handing its region back is not a release: the block stays in
     * the pool, keeps its bytes charged, and becomes reusable only once no
     * reference is left.
     *
     * @return true when `base` is a block this manager owns
     */
    bool release_ref(void *base, uint64_t run_epoch) {
        std::scoped_lock lk(mu_);
        Block *b = find_locked(base);
        if (b == nullptr) return false;
        drop_ref(*b, run_epoch);
        return true;
    }

    /** Whether `base` is a block this manager owns. */
    bool owns(void *base) const {
        std::scoped_lock lk(mu_);
        return find_locked(base) != nullptr;
    }

    /** Capacity published for `base`, or 0 when it is not owned here. */
    size_t block_bytes(void *base) const {
        std::scoped_lock lk(mu_);
        const Block *b = find_locked(base);
        return b == nullptr ? 0 : static_cast<size_t>(b->bytes);
    }

    /**
     * Record one fact about a run, and retire the run when its facts are
     * complete.
     *
     * Retirement needs the device side settled — either a drain that returned
     * success or the ownership fact that nothing was ever submitted — together
     * with the copy-back consumer having returned and the bindings released.
     * A context destroyed before that quarantines whatever the run still
     * references, because no further fact can arrive.
     */
    void note_run_fact(uint32_t slot, uint64_t run_epoch, RunFact fact) {
        std::scoped_lock lk(mu_);
        if (!enabled_ || slot >= PTO_PIPELINE_MAX_DEPTH) return;
        RunRecord &r = runs_[slot];
        if (r.epoch != run_epoch) {
            r = RunRecord{};
            r.epoch = run_epoch;
        }
        switch (fact) {
        case RunFact::Launched:
            r.launched = true;
            break;
        case RunFact::DrainAttempted:
            r.drain_attempted = true;
            break;
        case RunFact::DrainProvedComplete:
            r.drain_proved = true;
            break;
        case RunFact::NoDeviceSubmission:
            // Ignored once a launch has claimed this run: a partial unwind can
            // leave the caller's pointer null, and that is not the same fact as
            // never having reached the device.
            if (!r.launched) r.no_submission = true;
            break;
        case RunFact::CopybackReturned:
            r.copyback_returned = true;
            break;
        case RunFact::BindingsReleased:
            r.bindings_released = true;
            break;
        case RunFact::ContextDestroyed:
            r.context_destroyed = true;
            break;
        }
        if (retired(r)) {
            drop_run_refs(run_epoch);
            return;
        }
        if (r.context_destroyed) quarantine_run_refs(run_epoch);
    }

    /** Stop admitting new plans; already-accepted work keeps its path. */
    void enter_drain_only() {
        std::scoped_lock lk(mu_);
        if (admission_ == Admission::Open) admission_ = Admission::DrainOnly;
    }

    void enter_closed() {
        std::scoped_lock lk(mu_);
        admission_ = Admission::Closed;
    }

    /**
     * Runs that still reference a block and whose completion a caller can
     * still establish.
     *
     * Any run holding a reference counts while its context exists: one that has
     * only prepared, one executing, and one whose device work is done but whose
     * copy-back or bindings have not reported. All three are resolvable the
     * same way — finalize that run and close again — and a prior drain attempt
     * is not proof its host consumers are gone.
     *
     * A destroyed context is the one thing that stops counting: no further fact
     * about that run can arrive, so its blocks are quarantined instead. Which
     * is also why a refusal always clears: finalizing a run reaches either
     * retirement or context destruction.
     */
    uint32_t live_drainable_consumers() const {
        std::scoped_lock lk(mu_);
        return count_live_locked();
    }

    /**
     * Attempt an ordinary release of every block no consumer references, and
     * classify the rest. Quarantined blocks are not touched.
     *
     * @return the last non-zero platform release code, or 0
     */
    int release_unreferenced() {
        std::scoped_lock lk(mu_);
        int last_error = 0;
        for (Block &b : blocks_) {
            if (b.quarantined || b.ref_count != 0 || b.released || b.swept) continue;
            const int rc = backend_.release(backend_.ctx, b.base);
            if (rc == 0) {
                b.released = true;
                b.swept = true;  // accounted for here; the terminal sweep skips it
                b.state = BlockState::ProvenUnused;
                reserved_bytes_ -= b.bytes;
                continue;
            }
            // The backend keeps a failed release recorded, so the bytes keep
            // their owner and their charge instead of being reported as gone.
            b.release_unconfirmed = true;
            b.release_rc = rc;
            b.state = BlockState::ReleaseUnconfirmed;
            last_error = rc;
        }
        return last_error;
    }

    /** Whether the allocator's terminal sweep must keep `base`. */
    bool must_keep(void *base) const {
        std::scoped_lock lk(mu_);
        return must_keep_locked(base);
    }

    /**
     * One terminal sweep, holding this manager's lock for its whole duration.
     *
     * Every other path takes this manager's lock and then the allocator's, so
     * the sweep — which runs inside the allocator's lock — must take this one
     * first or the two orders meet head-on. The view is what the sweep's
     * callbacks use: its methods assume the lock is already held, allocate
     * nothing, and cannot throw, so a callback can never leave the allocator's
     * tracking map half-cleared.
     */
    class TerminalSweep {
    public:
        TerminalSweep(const TerminalSweep &) = delete;
        TerminalSweep &operator=(const TerminalSweep &) = delete;

        bool must_keep(void *base) const noexcept { return owner_->must_keep_locked(base); }
        void note_result(void *base, int rc, bool kept) noexcept { owner_->note_sweep_result_locked(base, rc, kept); }

    private:
        friend class WorkspaceManager;
        explicit TerminalSweep(WorkspaceManager *owner) :
            owner_(owner),
            lock_(owner->mu_) {}
        WorkspaceManager *owner_;
        std::unique_lock<std::mutex> lock_;
    };

    /** Take the manager's lock for one terminal sweep. */
    TerminalSweep begin_terminal_sweep() { return TerminalSweep(this); }

    /**
     * Record one terminal sweep outcome into the block it belongs to.
     *
     * Called before the allocator clears its tracking map, so the result lands
     * in storage that already exists. An address this manager does not own goes
     * into a fixed aggregate rather than into a per-address buffer built during
     * cleanup.
     */
    void note_sweep_result(void *base, int rc, bool kept) {
        std::scoped_lock lk(mu_);
        note_sweep_result_locked(base, rc, kept);
    }

    /** Remember that a quarantined block's host mapping was left in place. */
    void note_mapping_retained(void *base, uint64_t bytes) {
        std::scoped_lock lk(mu_);
        Block *b = find_locked(base);
        if (b == nullptr) return;
        b->mapping_retained = true;
        quarantined_mapped_bytes_ += bytes;
    }

    /**
     * Record that `base` could not be unmapped from this process.
     *
     * A host mapping covers the whole allocation, so storage still behind one
     * cannot be released to the platform: the range would be handed to another
     * allocation while this process still holds a host address over it. The
     * block is therefore quarantined — excluded from reuse, from growth's
     * reclamation, and from the terminal sweep — rather than freed with a
     * dropped mapping record.
     *
     * @return true when `base` is a block this manager owns
     */
    bool note_mapping_unregister_failed(void *base) {
        std::scoped_lock lk(mu_);
        Block *b = find_locked(base);
        if (b == nullptr) return false;
        b->quarantined = true;
        b->state = BlockState::Quarantined;
        if (!b->mapping_retained) {
            b->mapping_retained = true;
            quarantined_mapped_bytes_ += b->bytes;
        }
        return true;
    }

    /** Fill one report. Returns false when the manager is off. */
    bool report(SimplerWorkspaceReport *out) const {
        if (out == nullptr) return false;
        std::scoped_lock lk(mu_);
        if (!enabled_) return false;
        SimplerWorkspaceReport r{};
        r.budget_enforced = 1;
        r.coverage_is_partial = 1;
        r.limit_bytes = limit_bytes_;
        r.reserved_bytes = reserved_bytes_;
        r.relinquished_bytes = relinquished_bytes_;
        r.quarantined_mapped_bytes = quarantined_mapped_bytes_;
        r.blocks_published = blocks_published_;
        r.foreign_release_failures = foreign_release_failures_;
        r.last_foreign_release_rc = last_foreign_release_rc_;
        for (const Block &b : blocks_) {
            if (b.quarantined) ++r.quarantined_blocks;
            if (b.release_unconfirmed) ++r.release_unconfirmed_blocks;
        }
        r.live_blocked = count_live_locked();
        r.proof_unavailable = (r.quarantined_blocks != 0 || r.release_unconfirmed_blocks != 0) ? 1u : 0u;
        // Released last: a reader that sees the schema has the whole record.
        r.schema = WORKSPACE_REPORT_SCHEMA;
        *out = r;
        return true;
    }

    /** Bytes this manager has charged against its budget. */
    uint64_t reserved_bytes() const {
        std::scoped_lock lk(mu_);
        return reserved_bytes_;
    }

    /** Blocks this manager currently owns, released ones included. */
    size_t block_count() const {
        std::scoped_lock lk(mu_);
        return blocks_.size();
    }

    /** State of the block at `base`, for tests and reports. */
    BlockState block_state(void *base) const {
        std::scoped_lock lk(mu_);
        const Block *b = find_locked(base);
        return b == nullptr ? BlockState::ProvenUnused : b->state;
    }

private:
    struct Block {
        void *base{nullptr};
        uint64_t bytes{0};
        RegionKey region{};
        BlockState state{BlockState::ProvenUnused};
        // Run epochs currently using this block. The region already pins which
        // consumer the block belongs to, so an epoch identifies a reference.
        uint64_t refs[kMaxBlockRefs]{};
        uint32_t ref_count{0};
        int release_rc{0};
        bool quarantined{false};
        bool release_unconfirmed{false};
        bool released{false};
        bool mapping_retained{false};
        // The address its region is published at right now. Cleared only when
        // that region publishes a newer generation, which makes this block
        // obsolete and its bytes reclaimable once no run references it.
        bool current{true};
        // Accounted for by a release or by the terminal sweep already. A second
        // close finds the same record and must not charge or credit it twice.
        bool swept{false};
    };

    struct RunRecord {
        uint64_t epoch{0};
        bool launched{false};
        bool drain_attempted{false};
        bool drain_proved{false};
        bool no_submission{false};
        bool copyback_returned{false};
        bool bindings_released{false};
        bool context_destroyed{false};
    };

    bool must_keep_locked(void *base) const noexcept {
        const Block *b = find_locked(base);
        return b != nullptr && b->quarantined;
    }

    /**
     * Land one terminal outcome in the block it belongs to, accounting for it
     * exactly once. No allocation and no throw: every field already exists,
     * and an address outside this ledger only bumps a fixed aggregate.
     */
    void note_sweep_result_locked(void *base, int rc, bool kept) noexcept {
        Block *b = find_locked(base);
        if (b == nullptr) {
            if (rc != 0) {
                if (foreign_release_failures_ != UINT32_MAX) ++foreign_release_failures_;
                last_foreign_release_rc_ = rc;
            }
            return;
        }
        if (b->swept) return;  // a second close must not account for it twice
        b->swept = true;
        if (kept) {
            // Forgotten without being freed: the charge moves rather than
            // disappearing, because these bytes are still on the device.
            b->quarantined = true;
            b->state = BlockState::Quarantined;
            relinquished_bytes_ += b->bytes;
            reserved_bytes_ -= b->bytes;
            return;
        }
        if (rc == 0) {
            b->released = true;
            b->state = BlockState::ProvenUnused;
            reserved_bytes_ -= b->bytes;
            return;
        }
        // A failed free is not a reclamation: the bytes keep their charge and
        // are reported as unconfirmed rather than as returned.
        b->release_unconfirmed = true;
        b->release_rc = rc;
        b->state = BlockState::ReleaseUnconfirmed;
    }

    /**
     * Distinct runs still holding a reference whose completion a caller can
     * establish.
     *
     * Counted from the references, not from the fact records: a run that has
     * acquired workspace but reported nothing yet — a prepare that has not
     * launched — has no record at all, and it is exactly the consumer a close
     * must not step over. A retired run has already dropped its references, and
     * a destroyed context's are quarantined, so both drop out naturally.
     */
    uint32_t count_live_locked() const {
        uint64_t counted[PTO_PIPELINE_MAX_DEPTH * kMaxBlockRefs] = {};
        size_t counted_n = 0;
        uint32_t live = 0;
        for (const Block &b : blocks_) {
            if (b.released) continue;
            for (uint32_t i = 0; i < b.ref_count; ++i) {
                const uint64_t epoch = b.refs[i];
                bool seen = false;
                for (size_t j = 0; j < counted_n; ++j) {
                    if (counted[j] == epoch) seen = true;
                }
                if (seen) continue;
                if (counted_n < sizeof(counted) / sizeof(counted[0])) counted[counted_n++] = epoch;
                const RunRecord *r = record_for_locked(epoch);
                if (r != nullptr && (r->context_destroyed || retired(*r))) continue;
                ++live;
            }
        }
        return live;
    }

    /** This epoch's fact record, or null when it has reported nothing yet. */
    const RunRecord *record_for_locked(uint64_t epoch) const {
        for (uint32_t slot = 0; slot < PTO_PIPELINE_MAX_DEPTH; ++slot) {
            if (runs_[slot].epoch == epoch) return &runs_[slot];
        }
        return nullptr;
    }

    static bool retired(const RunRecord &r) {
        const bool device_settled = r.drain_proved || r.no_submission;
        return device_settled && r.copyback_returned && r.bindings_released;
    }

    Block *find_locked(void *base) {
        for (Block &b : blocks_) {
            if (b.base == base && !b.released) return &b;
        }
        return nullptr;
    }

    const Block *find_locked(void *base) const {
        for (const Block &b : blocks_) {
            if (b.base == base && !b.released) return &b;
        }
        return nullptr;
    }

    static bool add_ref(Block &b, uint64_t epoch) {
        // The context's own backing carries no run reference: nothing can ever
        // report this identity finished, so counting it would leave a consumer
        // no close could resolve.
        if (epoch == kContextEpoch) return true;
        for (uint32_t i = 0; i < b.ref_count; ++i) {
            if (b.refs[i] == epoch) return true;
        }
        if (b.ref_count >= kMaxBlockRefs) return false;  // refuse, never truncate
        b.refs[b.ref_count++] = epoch;
        if (!b.quarantined && !b.release_unconfirmed) b.state = BlockState::Referenced;
        return true;
    }

    static void drop_ref(Block &b, uint64_t epoch) {
        for (uint32_t i = 0; i < b.ref_count; ++i) {
            if (b.refs[i] != epoch) continue;
            b.refs[i] = b.refs[b.ref_count - 1];
            b.refs[--b.ref_count] = 0;
            break;
        }
        if (b.ref_count == 0 && !b.quarantined && !b.release_unconfirmed) b.state = BlockState::ProvenUnused;
    }

    bool references_run(uint64_t epoch) const {
        for (const Block &b : blocks_) {
            if (b.released) continue;
            for (uint32_t i = 0; i < b.ref_count; ++i) {
                if (b.refs[i] == epoch) return true;
            }
        }
        return false;
    }

    void drop_run_refs(uint64_t epoch) {
        for (Block &b : blocks_)
            drop_ref(b, epoch);
    }

    void quarantine_run_refs(uint64_t epoch) {
        for (Block &b : blocks_) {
            bool held = false;
            for (uint32_t i = 0; i < b.ref_count; ++i) {
                if (b.refs[i] == epoch) held = true;
            }
            if (!held) continue;
            // Whole-block: the allocator and the host-mapping registry are both
            // keyed by allocation base, so a region whose last consumer cannot
            // be proven finished takes the rest of its block with it.
            b.quarantined = true;
            drop_ref(b, epoch);
            b.state = BlockState::Quarantined;
        }
    }

    /**
     * Release obsolete generations until `bytes` fits, and report whether it
     * does.
     *
     * Only a block its region has since republished elsewhere is eligible, and
     * only with no run reference left, no quarantine, no unconfirmed release
     * and no host mapping still standing. That excludes, deliberately: the
     * current backing of every region — which is the context's, not any run's,
     * and whose zero references mean "idle", not "abandoned"; the addresses a
     * failed plan preserved, which stay current; and any block the backend's
     * unmap could not clear.
     */
    bool make_room_locked(size_t bytes) {
        if (reserved_bytes_ + bytes <= limit_bytes_) return true;
        for (Block &b : blocks_) {
            if (reserved_bytes_ + bytes <= limit_bytes_) break;
            if (b.current || b.released || b.swept) continue;
            if (b.quarantined || b.release_unconfirmed || b.mapping_retained) continue;
            if (b.ref_count != 0) continue;
            if (backend_.release(backend_.ctx, b.base) != 0) {
                // Still on the device, so still charged: reporting these bytes
                // as reclaimed would let the budget hand them out twice.
                b.release_unconfirmed = true;
                b.state = BlockState::ReleaseUnconfirmed;
                continue;
            }
            b.released = true;
            b.swept = true;  // accounted for here; the terminal sweep skips it
            b.state = BlockState::ProvenUnused;
            reserved_bytes_ -= b.bytes;
        }
        return reserved_bytes_ + bytes <= limit_bytes_;
    }

    void *publish_new_block(const RegionKey &region, uint64_t run_epoch, size_t bytes) {
        if (reserved_bytes_ + bytes < reserved_bytes_) return nullptr;  // overflow
        if (!make_room_locked(bytes)) return nullptr;
        // Both ownership records exist before the device allocation: this vector
        // may reallocate here, where nothing has been allocated on the device
        // yet, and the backend reserves its own tracking node before its own
        // platform call.
        blocks_.reserve(blocks_.size() + 1);
        void *base = backend_.acquire(backend_.ctx, bytes);
        if (base == nullptr) return nullptr;  // published state untouched
        Block b;
        b.base = base;
        b.bytes = bytes;
        b.region = region;
        b.state = BlockState::ProvenUnused;
        // The predecessor stays current until this plan publishes, so a failure
        // between here and `note_published` leaves the address the region is
        // still using protected from the reclamation above.
        add_ref(b, run_epoch);
        blocks_.push_back(b);  // into the capacity reserved above
        reserved_bytes_ += bytes;
        if (blocks_published_ != UINT32_MAX) ++blocks_published_;
        return base;
    }

    mutable std::mutex mu_;
    bool enabled_{false};
    uint64_t limit_bytes_{0};
    uint64_t reserved_bytes_{0};
    uint64_t relinquished_bytes_{0};
    uint64_t quarantined_mapped_bytes_{0};
    uint32_t blocks_published_{0};
    uint32_t foreign_release_failures_{0};
    int32_t last_foreign_release_rc_{0};
    Admission admission_{Admission::Open};
    Backend backend_{};
    std::vector<Block> blocks_;
    RunRecord runs_[PTO_PIPELINE_MAX_DEPTH]{};
};
