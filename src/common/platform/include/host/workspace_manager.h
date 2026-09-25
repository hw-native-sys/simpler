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

#include <algorithm>
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
 * elsewhere, or given up without republishing at all, with no run reference
 * left — is the only storage growth may reclaim, and only after its host
 * mapping is gone.
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
     * What a backend release actually achieved.
     *
     * Three outcomes rather than a code, because they have three different
     * owners. The backend reports which one happened and returns; it must not
     * call back into this manager to record it — this manager holds its own
     * lock across the call, and a callback would re-enter it.
     */
    enum class ReleaseOutcome : uint32_t {
        /** Unmapped if it had to be, and the platform freed the bytes. */
        Freed = 0,
        /** Still mapped into this process: the bytes must not be freed at all. */
        StillMapped,
        /** Unmapped, but the platform free failed: the bytes are still there. */
        FreeFailed,
    };

    /**
     * The device allocation this manager owns blocks through.
     *
     * `acquire` must reserve its own bookkeeping before the platform call and
     * commit it without allocating afterwards; it returns nullptr on failure
     * having changed nothing.
     *
     * `release` unmaps before it frees and reports which of the three outcomes
     * it reached, writing the platform code into `platform_rc` when it has one.
     * Anything but `Freed` leaves the allocation recorded on the backend side,
     * so this manager keeps charging for it rather than dropping the pointer or
     * reporting a release that did not happen.
     */
    struct Backend {
        void *ctx{nullptr};
        void *(*acquire)(void *ctx, size_t bytes){nullptr};
        ReleaseOutcome (*release)(void *ctx, void *base, int *platform_rc){nullptr};
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
     * Why an acquire refused. Internal to the host: the caller turns it into
     * its own log line, and no field of the public report carries it.
     */
    enum class AcquireRefusal : uint32_t {
        None = 0,
        /** Not managed, a zero request, or admission is closed. */
        NotServing,
        /** A finite limit is enforced and this request does not fit it. */
        OverQuota,
        /** Ownership of some block became unprovable, so no new block is published. */
        Degraded,
        /** The platform allocation itself failed. */
        BackendFailed,
        /** The requested size cannot be accounted without wrapping. */
        Overflow,
    };

    /**
     * Turn ownership management on, once, with no byte limit.
     *
     * A limit is a separate, optional policy (`set_limit`): managing lifetimes
     * and capping bytes are different questions, and the first does not need
     * the second. A manager that is on with no limit tracks, reuses and
     * reclaims exactly as one with a limit does; it simply never refuses for
     * capacity.
     *
     * @return false when management is already on or the backend is incomplete
     */
    bool configure(const Backend &backend) {
        std::scoped_lock lk(mu_);
        if (enabled_) return false;
        if (backend.acquire == nullptr || backend.release == nullptr) return false;
        backend_ = backend;
        enabled_ = true;
        return true;
    }

    /**
     * Enforce a finite byte limit on the blocks this manager owns, once.
     *
     * @return false when management is off, when a limit is already set, when
     *         `limit_bytes` is zero, or when it is below what is already
     *         charged — a limit that the current state already exceeds could
     *         only refuse every later request while proving nothing about the
     *         bytes already published.
     */
    bool set_limit(uint64_t limit_bytes) {
        std::scoped_lock lk(mu_);
        if (!enabled_ || has_limit_ || limit_bytes == 0) return false;
        if (limit_bytes < reserved_bytes_) return false;
        limit_bytes_ = limit_bytes;
        has_limit_ = true;
        return true;
    }

    bool enabled() const {
        std::scoped_lock lk(mu_);
        return enabled_;
    }

    /** Whether a finite byte limit is enforced on top of management. */
    bool limit_enforced() const {
        std::scoped_lock lk(mu_);
        return has_limit_;
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
    void *acquire(const RegionKey &region, uint64_t run_epoch, size_t bytes, AcquireRefusal *why = nullptr) {
        std::scoped_lock lk(mu_);
        if (why != nullptr) *why = AcquireRefusal::None;
        if (!enabled_ || bytes == 0) return refuse(why, AcquireRefusal::NotServing);
        if (admission_ != Admission::Open && admission_ != Admission::DrainOnly) {
            return refuse(why, AcquireRefusal::NotServing);
        }

        for (Block &b : blocks_) {
            // A released block's address belongs to the platform again, so the
            // record is history: reusing it would hand back freed memory.
            if (b.released) continue;
            if (!(b.region == region)) continue;
            if (b.quarantined || b.release_unconfirmed) continue;
            if (b.bytes < bytes) continue;
            if (b.ref_count != 0) continue;  // capacity is not permission
            if (!add_ref(b, run_epoch)) return refuse(why, AcquireRefusal::NotServing);
            // Handed out as this region's backing, so it stops being an
            // obsolete generation now rather than at publication: between the
            // two, growth elsewhere must not reclaim the block this plan is
            // already writing.
            b.current = true;
            return b.base;
        }
        return publish_new_block(region, run_epoch, bytes, why);
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
        // Quarantined and release-unconfirmed are both refused, and for the
        // same reason as in `acquire`: a caller reaching this from a capacity
        // hit has not re-derived the block's disposition, and handing back
        // storage whose free was attempted and failed would let a run write
        // where the platform may already have reclaimed.
        if (b == nullptr || b->quarantined || b->release_unconfirmed) return false;
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
     * Relinquish the published-owner claim on `base`.
     *
     * Two boundaries end such a claim without another generation taking it
     * over: a staging that was aborted, so the generation it would have become
     * never existed and no plan ever read it; and a region detached to hold
     * nothing, which leaves that region published at no address at all. Either
     * way this block is no longer what protects a region, so it becomes an
     * obsolete generation — reclaimable by a later request *once its last true
     * consumer retires*, never before.
     *
     * What this does not do: it drops no run reference and lifts no quarantine.
     * Giving up a claim is the region's decision; whether the bytes may be
     * released is still the consumers' to answer. Nor does it apply to the
     * block a publication superseded — that one is settled by
     * `note_published` at the region's new base, which is the fact that a
     * successor exists.
     *
     * @return true when `base` is a block this manager owns
     */
    bool note_unpublished(void *base) {
        std::scoped_lock lk(mu_);
        Block *b = find_locked(base);
        if (b == nullptr) return false;
        b->current = false;
        return true;
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
            // Run epochs are minted from one process-wide counter that only
            // ever increments, so a lower epoch on a slot is necessarily a
            // predecessor's late fact and not a new run. Its record has been
            // replaced by the successor's, and replacing it back would erase
            // facts the successor has already reported.
            //
            // The fact still has to reach that predecessor's *references*,
            // which are keyed by epoch and independent of this slot record:
            // a late context destruction must still quarantine whatever the
            // old run held, or those blocks would look reclaimable.
            if (run_epoch < r.epoch) {
                if (fact == RunFact::ContextDestroyed) quarantine_run_refs(run_epoch);
                return;
            }
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
            // A release that already failed is not attempted again here. Its
            // bytes keep their charge and its record keeps the failure, and
            // the terminal sweep below still gets the one further attempt that
            // can reach a proved outcome — trying again on this ordinary path
            // would only add a second failure to the same block.
            if (b.release_unconfirmed) continue;
            const int rc = apply_release_locked(b);
            if (rc != 0) last_error = rc;
        }
        return last_error;
    }

    /**
     * Whether a drain would have anything to do.
     *
     * Cheap enough to ask on every run: one locked walk of a short vector and
     * no device call, so a steady-state workload that reuses its current
     * blocks pays only this.
     */
    bool has_reclaimable() const {
        std::scoped_lock lk(mu_);
        if (!enabled_) return false;
        for (const Block &b : blocks_) {
            if (reclaimable_locked(b)) return true;
            if (compactable_locked(b)) return true;
        }
        return false;
    }

    /**
     * Release every obsolete generation whose consumers are all finished, and
     * drop the ledger records of blocks that are fully and provably gone.
     *
     * Independent of any byte limit: a generation its region has replaced, with
     * no reference left, is reclaimable because nothing can reach it — not
     * because the budget is tight. A region's *current* block is never taken,
     * whatever its reference count: idle is not abandoned, and keeping it warm
     * is what lets the next same-sized call reuse it.
     *
     * The caller must have this thread attached to the device, because the
     * backend release is a device call. Nothing here retries a failure.
     *
     * @return the last platform code from a failed release, or 0
     */
    int reclaim_obsolete() {
        std::scoped_lock lk(mu_);
        if (!enabled_) return 0;
        int last_error = 0;
        for (Block &b : blocks_) {
            if (!reclaimable_locked(b)) continue;
            const int rc = apply_release_locked(b);
            if (rc != 0) last_error = rc;
        }
        compact_locked();
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
        degrade_locked();
        return true;
    }

    /** Fill one report. Returns false when the manager is off. */
    bool report(SimplerWorkspaceReport *out) const {
        if (out == nullptr) return false;
        std::scoped_lock lk(mu_);
        if (!enabled_) return false;
        SimplerWorkspaceReport r{};
        // Two different questions: this record exists because the context is
        // managed, and this field says whether a finite limit is also enforced.
        r.budget_enforced = has_limit_ ? 1u : 0u;
        r.coverage_is_partial = 1;
        r.limit_bytes = has_limit_ ? limit_bytes_ : 0;
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
            // The one further attempt a block whose earlier free failed is
            // allowed reached a proved outcome, so the doubt it was carrying
            // is over: leaving the marks would report a freed block as
            // unconfirmed for the rest of this context's life, and would keep
            // its record out of compaction forever.
            b->release_unconfirmed = false;
            b->release_rc = 0;
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
        // Any block this leaves quarantined is one whose last consumer can
        // never be proved, so publishing more of them would grow the ledger
        // with records nothing can resolve. Latched even when the run held
        // none: the degrade is about this manager's ability to prove, and the
        // check below records whether it actually took anything.
        bool held_any = false;
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
            held_any = true;
        }
        if (held_any) degrade_locked();
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
    static void *refuse(AcquireRefusal *why, AcquireRefusal reason) {
        if (why != nullptr) *why = reason;
        return nullptr;
    }

    /**
     * Ownership of some block can no longer be proved, so this manager stops
     * publishing new ones.
     *
     * Nothing this manager does on its own can clear such a record. The
     * terminal sweep is the one place a failed free can still be settled, and
     * only there, once; a failed unmap leaves a live host address that no
     * later attempt makes safe; and a run destroyed before its facts completed
     * can never produce them. Without this latch each later round could
     * publish another block and leave another record behind it, so the ledger
     * would grow with the failures rather than stopping at them.
     *
     * It frees nothing, caps no bytes, and leaves blocks already proven safe
     * reusable: only the publication of *new* blocks stops.
     */
    void degrade_locked() { degraded_ = true; }

    /**
     * Release one block through the backend and record what actually happened.
     *
     * Called with `mu_` held. The backend reports a disposition rather than
     * calling back here, so this stays non-reentrant.
     *
     * @return the platform code when the bytes are still there, else 0
     */
    int apply_release_locked(Block &b) {
        int platform_rc = 0;
        const ReleaseOutcome outcome = backend_.release(backend_.ctx, b.base, &platform_rc);
        if (outcome == ReleaseOutcome::Freed) {
            b.released = true;
            b.swept = true;  // accounted for here; the terminal sweep skips it
            b.state = BlockState::ProvenUnused;
            reserved_bytes_ -= b.bytes;
            return 0;
        }
        if (outcome == ReleaseOutcome::StillMapped) {
            // A host mapping covers the whole allocation, so these bytes can
            // never be handed to another allocation. Quarantined rather than
            // merely unconfirmed: this is not a free that might be retried.
            b.quarantined = true;
            b.state = BlockState::Quarantined;
            if (!b.mapping_retained) {
                b.mapping_retained = true;
                quarantined_mapped_bytes_ += b.bytes;
            }
            degrade_locked();
            return platform_rc != 0 ? platform_rc : -1;
        }
        // Unmapped, but the platform kept the bytes: the charge stays and the
        // block is never offered again. Not retried here.
        b.release_unconfirmed = true;
        b.release_rc = platform_rc;
        b.state = BlockState::ReleaseUnconfirmed;
        degrade_locked();
        return platform_rc != 0 ? platform_rc : -1;
    }

    /**
     * Whether this record can leave the ledger.
     *
     * Only a block the platform has provably taken back, with nothing left to
     * say about it: no retained mapping, no failed release code, and already
     * accounted by whichever path released it. Everything else stays — a
     * quarantined or unconfirmed record *is* the evidence, and the terminal
     * sweep and `must_keep` still have to find it.
     */
    static bool compactable_locked(const Block &b) {
        return b.released && b.swept && !b.mapping_retained && !b.quarantined && !b.release_unconfirmed &&
               b.release_rc == 0 && b.ref_count == 0;
    }

    /**
     * Drop fully released records so the ledger tracks live ownership rather
     * than history.
     *
     * Safe to do under `mu_` and nowhere else: every `Block` handle in this
     * class is a local obtained from `find_locked` inside one locked method,
     * so no index or pointer outlives a critical section and an erase can
     * never strand one. Block identity is (base, not released) and
     * `find_locked` already skips released records, so an address the platform
     * reissues to a new block cannot be matched to a record that survived —
     * and the records that could be confused are exactly the ones erased here.
     *
     * Cumulative counters are deliberately untouched: `blocks_published_`,
     * `relinquished_bytes_` and the foreign-failure tallies are the audit
     * trail, and compaction must not make history smaller.
     */
    void compact_locked() {
        blocks_.erase(
            std::remove_if(
                blocks_.begin(), blocks_.end(),
                [](const Block &b) {
                    return compactable_locked(b);
                }
            ),
            blocks_.end()
        );
    }

    /** Whether this block is an obsolete generation nothing can still be using. */
    static bool reclaimable_locked(const Block &b) {
        if (b.current || b.released || b.swept) return false;
        if (b.quarantined || b.release_unconfirmed || b.mapping_retained) return false;
        return b.ref_count == 0;
    }

    bool make_room_locked(size_t bytes) {
        if (!has_limit_ || reserved_bytes_ + bytes <= limit_bytes_) return true;
        for (Block &b : blocks_) {
            if (reserved_bytes_ + bytes <= limit_bytes_) break;
            if (!reclaimable_locked(b)) continue;
            apply_release_locked(b);
        }
        return reserved_bytes_ + bytes <= limit_bytes_;
    }

    void *publish_new_block(const RegionKey &region, uint64_t run_epoch, size_t bytes, AcquireRefusal *why) {
        if (reserved_bytes_ + bytes < reserved_bytes_) return refuse(why, AcquireRefusal::Overflow);
        // Ownership of some block is unprovable, so nothing new is published:
        // a new block could only add another record nothing can resolve.
        // Checked before make_room so a reclaim cannot look like a way out.
        if (degraded_) return refuse(why, AcquireRefusal::Degraded);
        if (!make_room_locked(bytes)) return refuse(why, AcquireRefusal::OverQuota);
        // Reclaiming under pressure can itself fail a release and degrade this
        // manager, so the latch is re-read after it rather than only before.
        if (degraded_) return refuse(why, AcquireRefusal::Degraded);
        // Both ownership records exist before the device allocation: this vector
        // may reallocate here, where nothing has been allocated on the device
        // yet, and the backend reserves its own tracking node before its own
        // platform call.
        blocks_.reserve(blocks_.size() + 1);
        void *base = backend_.acquire(backend_.ctx, bytes);
        if (base == nullptr) return refuse(why, AcquireRefusal::BackendFailed);  // published state untouched
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
    // A finite byte limit is optional policy on top of management; `limit_bytes_`
    // is meaningless unless this is set, and there is no sentinel for "no limit".
    bool has_limit_{false};
    // Some block's ownership is unprovable, so no new block is published.
    bool degraded_{false};
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
