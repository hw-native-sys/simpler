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

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <memory>
#include <utility>
#include <vector>

#include "run_resource_arena.h"
#include "runtime_c_api.h"

/**
 * The one AICPU + AICore stream pair every run submits on.
 *
 * A stream is an ordered queue, so one pair carries every run. The pair is not
 * indexed by pipeline slot: a slot exists for resources that *preparation*
 * mutates, and preparing a run writes nothing to a stream — only launch
 * submits.
 *
 * The two streams must stay distinct. The AICPU Run kernel spins in the
 * handshake waiting for the AICore workers, so serializing both onto one queue
 * would leave the AICore submission behind a spin that can never end.
 *
 * The AICore stream carries instruction-cache state: cores may retain code
 * fetched from a GM address whose contents a later registration replaced.
 * Publishing new AICore code therefore marks the stream stale, and the next
 * launch retires it and creates a replacement. A graph-pinned generation keeps
 * its handle until graph destruction and drain. Creating a stream is the only
 * operation known to leave a core free of the previous image's instructions.
 *
 * **Submission state is per owner.** More than one run may have submitted and
 * not yet retired, so a completion, a query and a retirement each name the run
 * they belong to; a successor's poll must never be answered by a predecessor's
 * result. Owners are kept in submission order, which is also retirement order.
 * Replacing the AICore stream still requires that no owner holds the pair, and
 * an unproven retirement while another owner is live marks the stream stale
 * instead of destroying it out from under live work.
 *
 * Threading: launch and drain are the owning operations, but poll may query the
 * pair from a progress thread while the executor retires it. The pair therefore
 * serializes query with handle mutation. Poll uses try-lock and reports
 * NOT_READY rather than waiting behind retirement. `created_count_` is
 * additionally readable from unrelated threads and remains atomic. Stream
 * creation and destruction are injected so this state machine is exercisable
 * without a device.
 */
class RunStreamPair {
public:
    using CreateFn = std::function<int(void **out_stream)>;
    using DestroyFn = std::function<int(void *stream)>;
    enum class CompletionStatus { Unproven, Complete };

    RunStreamPair(CreateFn create, DestroyFn destroy) :
        create_(std::move(create)),
        destroy_(std::move(destroy)) {}

    /**
     * Ready the pair for a launch: both streams on first use, and a
     * replacement AICore stream when a code publication marked it stale.
     *
     * A live owner blocks only the replacement: readying the pair for a run
     * that joins one already submitted must leave that run's stream and its
     * recorded state alone.
     */
    int ensure() {
        std::lock_guard<std::mutex> lock(mutex_);
        const int retired_rc = release_unpinned_generations();
        if (retired_rc != 0) return retired_rc;
        if (aicpu_ == nullptr) {
            int rc = create_(&aicpu_);
            if (rc != 0) {
                aicpu_ = nullptr;
                return rc;
            }
        }
        if (aicore_ != nullptr) {
            if (!stale_) return 0;
            // A run that has not retired still owns the pair: a device-complete
            // poll is not a finalized run, and replacing the stream under it
            // would strand a live submission.
            if (owner_count_ != 0) return PTO_RUNTIME_ERR_INTERNAL;
            if (aicore_generation_.use_count() > 1) {
                retired_generations_.push_back(aicore_generation_);
            } else {
                int rc = destroy_(aicore_);
                if (rc != 0) return rc;
            }
            aicore_generation_.reset();
            aicore_ = nullptr;
        }
        auto generation = std::make_shared<void *>(nullptr);
        int rc = create_(&aicore_);
        if (rc != 0) {
            aicore_ = nullptr;
            return rc;
        }
        *generation = aicore_;
        aicore_generation_ = std::move(generation);
        created_count_.fetch_add(1, std::memory_order_relaxed);
        stale_ = false;
        last_retired_owner_ = nullptr;
        last_retired_complete_ = false;
        return 0;
    }

    // The graph holds this generation through destruction and successful drain.
    // Rotation preserves its handle; unpinned old handles are reaped on the
    // next ensure/destroy, with failed destroys retaining ownership for retry.
    std::shared_ptr<void> pin_aicore_generation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return aicore_generation_;
    }

    /** Mark the AICore stream stale after new AICore code is published. */
    void mark_stale() {
        std::lock_guard<std::mutex> lock(mutex_);
        stale_ = true;
    }

    /** Make this run's submission visible to poll, in submission order. */
    int mark_submitted(const void *owner) {
        if (owner == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        std::lock_guard<std::mutex> lock(mutex_);
        if (aicpu_ == nullptr || aicore_ == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        if (owner_count_ >= owners_.size()) return PTO_RUNTIME_ERR_INTERNAL;
        if (find_owner(owner) != nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        owners_[owner_count_] = OwnerSlot{owner, false};
        ++owner_count_;
        if (last_retired_owner_ == owner) {
            last_retired_owner_ = nullptr;
            last_retired_complete_ = false;
        }
        return 0;
    }

    /**
     * Query this run's streams without waiting behind retirement.
     *
     * A completed result is sticky for the run that produced it until the pair
     * is readied again, so a poll racing with successful stream destruction
     * never observes a missing handle as an error. A run that never submitted,
     * or one whose unproven retirement cleared its completion, answers ERROR.
     */
    template <typename QueryPairFn>
    int poll(const void *owner, QueryPairFn &&query) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        OwnerSlot *slot = find_owner(owner);
        if (slot == nullptr) {
            const bool sticky = owner != nullptr && owner == last_retired_owner_ && last_retired_complete_;
            return sticky ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_ERROR;
        }
        if (slot->complete) return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        if (aicpu_ == nullptr || aicore_ == nullptr) return SIMPLER_NATIVE_RUN_POLL_ERROR;
        const int rc = std::forward<QueryPairFn>(query)(aicpu_, aicore_);
        if (rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) slot->complete = true;
        return rc;
    }

    /**
     * Retire the pair on behalf of one run that submitted it.
     *
     * Complete keeps the AICore stream for the next launch. Unproven with no
     * other owner destroys it: a failed launch or an abandoned drain may leave
     * the stream in the error state rtStreamDestroy is the supported teardown
     * for, and the handle survives a failed destroy so teardown can retry it.
     * Unproven *while another run still owns the pair* cannot destroy it —
     * that run's work is still queued on it — so the stream is marked stale
     * and the last owner's retirement replaces it. A graph pin likewise defers
     * destruction until that generation is no longer referenced. A caller that never
     * submitted retires nothing.
     */
    int retire(CompletionStatus completion_status, const void *owner) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (owner == nullptr || find_owner(owner) == nullptr) return 0;
        // Publish the proven terminal state before destroying the handle. Poll
        // either finishes its in-flight query first or observes this result.
        // An error-path retirement clears a completion that raced ahead of a
        // later failing sync, so the sync error remains authoritative.
        erase_owner(owner);
        last_retired_owner_ = owner;
        last_retired_complete_ = completion_status == CompletionStatus::Complete;
        if (completion_status == CompletionStatus::Complete) return 0;
        if (owner_count_ != 0 || aicore_generation_.use_count() > 1) {
            stale_ = true;
            return 0;
        }
        if (aicore_ == nullptr) return 0;
        int rc = destroy_(aicore_);
        if (rc != 0) return rc;
        aicore_ = nullptr;
        return 0;
    }

    /** Destroy both streams, keeping a handle whose destroy failed. */
    int destroy() {
        std::lock_guard<std::mutex> lock(mutex_);
        int first_error = release_unpinned_generations();
        if (aicore_generation_.use_count() > 1 || !retired_generations_.empty())
            return first_error != 0 ? first_error : PTO_RUNTIME_ERR_INVALID_STATE;
        owner_count_ = 0;
        last_retired_owner_ = nullptr;
        last_retired_complete_ = false;
        for (void **stream : {&aicpu_, &aicore_}) {
            if (*stream == nullptr) continue;
            int rc = destroy_(*stream);
            if (rc != 0) {
                if (first_error == 0) first_error = rc;
                continue;
            }
            *stream = nullptr;
        }
        // A handle that survived teardown may still hold the previous image's
        // instructions, so it stays stale for whoever retries it.
        stale_ = aicore_ != nullptr;
        if (aicore_ == nullptr) aicore_generation_.reset();
        return first_error;
    }

    /** Forget both handles after a device reset without invoking destroy_. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        aicpu_ = nullptr;
        aicore_ = nullptr;
        aicore_generation_.reset();
        retired_generations_.clear();
        stale_ = false;
        owner_count_ = 0;
        last_retired_owner_ = nullptr;
        last_retired_complete_ = false;
    }

    // Handle reads are unsynchronized: the claim holder is the only writer once
    // the pair is readied, and poll must never block behind a retirement.
    void *aicpu() const { return aicpu_; }
    void *aicore() const { return aicore_; }
    bool ready() const { return aicpu_ != nullptr && aicore_ != nullptr; }
    /**
     * Whether both handles are gone — a `destroy()` that reported no failure, or an `abandon()`.
     *
     * The negation is the load-bearing case: a handle this pair kept because its own destroy
     * failed may still hold queued instructions, so anything whose safety rests on this pair
     * being finished has to ask rather than assume the destroy succeeded.
     */
    bool retired() const { return aicpu_ == nullptr && aicore_ == nullptr; }
    size_t created_count() const { return created_count_.load(std::memory_order_relaxed); }
    /** How many runs have submitted on this pair and not yet retired. */
    size_t live_owner_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return owner_count_;
    }

    /**
     * Whether the next `ensure()` must replace the AICore stream.
     *
     * A joined launch cannot happen across that replacement: the predecessor
     * is still running on the stream a publication marked stale, so `ensure()`
     * refuses. Asking first lets the caller leave the successor on the
     * ordinary path instead of turning a fallback into a launch failure.
     */
    bool aicore_replacement_pending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stale_ || aicore_ == nullptr;
    }

private:
    struct OwnerSlot {
        const void *owner{nullptr};
        bool complete{false};
    };

    OwnerSlot *find_owner(const void *owner) {
        if (owner == nullptr) return nullptr;
        for (size_t i = 0; i < owner_count_; ++i) {
            if (owners_[i].owner == owner) return &owners_[i];
        }
        return nullptr;
    }

    void erase_owner(const void *owner) {
        for (size_t i = 0; i < owner_count_; ++i) {
            if (owners_[i].owner != owner) continue;
            for (size_t j = i + 1; j < owner_count_; ++j)
                owners_[j - 1] = owners_[j];
            --owner_count_;
            owners_[owner_count_] = OwnerSlot{};
            return;
        }
    }

    int release_unpinned_generations() {
        for (auto it = retired_generations_.begin(); it != retired_generations_.end();) {
            if (it->use_count() > 1) {
                ++it;
                continue;
            }
            const int rc = destroy_(**it);
            if (rc != 0) return rc;
            it = retired_generations_.erase(it);
        }
        return 0;
    }

    std::shared_ptr<void *> aicore_generation_;
    std::vector<std::shared_ptr<void *>> retired_generations_;
    mutable std::mutex mutex_;
    void *aicpu_{nullptr};
    void *aicore_{nullptr};
    // Runs that submitted and have not retired, in submission order. As many as
    // the grant allows: a run holds its place here for as long as it is
    // unretired, so the capacity is the count of runs a context may hold.
    std::vector<OwnerSlot> owners_ = std::vector<OwnerSlot>(kDefaultRunResourceSets);
    size_t owner_count_{0};
    // The most recent retirement, so a poll arriving after it still reads that
    // run's proven result instead of an error.
    const void *last_retired_owner_{nullptr};
    bool last_retired_complete_{false};
    bool stale_{false};

    CreateFn create_;
    DestroyFn destroy_;

public:
    using Storage = std::vector<OwnerSlot>;

    /** Build an unretired-owner table for the grant, detached. May throw; mutates nothing. */
    static Storage build(size_t granted) { return Storage(granted); }

    /**
     * Whether the table may be replaced at all.
     *
     * Refused while any run is still unretired: an entry is identified by the
     * run that holds it, so the table is settled before the first submission.
     */
    bool grant_is_replaceable() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return owner_count_ == 0;
    }

    /** Adopt a built table. Never throws, so a multi-owner grant cannot half-apply. */
    void commit(Storage &&built) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        owners_.swap(built);
    }

private:
    std::atomic<size_t> created_count_{0};
};
