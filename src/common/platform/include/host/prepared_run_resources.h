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
#include <cstdint>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "host/aligned_image_storage.h"
#include "host/scratch_pool.h"
#include "utils/device_arena.h"
#include "utils/retained_scheduler_storage.h"
#include "workspace_manager.h"

struct PreparedSubmissionStorage;

// Actual backing held by a prepared execution. Slot/bank caches may retain the
// same storage, but must replace it before writing while this owner holds it.
// Device cleanup belongs to the runner on a valid device context, not to the
// destruction of the last arbitrary host reference.
class PreparedRunResources {
    friend class DeviceRunnerBase;

public:
    struct CallableUse {};

    // A published owner has a distinct lifetime from the runner's unfinished
    // device-use record, but both hold exactly the same preparation backing.
    static std::shared_ptr<const PreparedRunResources>
    publish_owner(std::shared_ptr<const PreparedRunResources> resources) {
        struct Owner {
            std::shared_ptr<const PreparedRunResources> resources;
        };
        auto owner = std::make_shared<Owner>(Owner{std::move(resources)});
        return std::shared_ptr<const PreparedRunResources>(owner, owner->resources.get());
    }

    struct TemporaryStorage {
        void *base{nullptr};
        size_t capacity{0};
        WorkspaceManager::BackingLease backing;
        // Prepared runs that may still reach this block. The slot holds the
        // storage object itself, not a lease; growth and retirement key on this
        // explicit count rather than a shared_ptr count.
        std::atomic<size_t> run_leases{0};

        template <typename Free>
        int release(WorkspaceManager &workspace, Free free) {
            if (base == nullptr) return 0;
            if (workspace.owns(base)) {
                workspace.note_unpublished(base);
            } else {
                const int rc = free(base);
                if (rc != 0) return rc;
            }
            base = nullptr;
            capacity = 0;
            backing.reset();
            return 0;
        }
        void abandon() {
            base = nullptr;
            capacity = 0;
            backing.reset();
        }
    };

    using ImageStorage = AlignedImageStorage;

    struct ArenaRegion {
        ArenaRegion(DeviceArena::AllocFn alloc, DeviceArena::FreeFn free_fn, void *ctx) :
            arena(alloc, free_fn, ctx) {}
        DeviceArena arena;
        size_t cached_size{0};
    };

    struct ArenaBank {
        ArenaBank(DeviceArena::AllocFn alloc, DeviceArena::FreeFn free_fn, void *ctx) :
            gm_heap(std::make_shared<ArenaRegion>(alloc, free_fn, ctx)),
            gm_sm(std::make_shared<ArenaRegion>(alloc, free_fn, ctx)),
            runtime_pool(std::make_shared<ArenaRegion>(alloc, free_fn, ctx)) {}
        std::shared_ptr<ArenaRegion> gm_heap;
        std::shared_ptr<ArenaRegion> gm_sm;
        std::shared_ptr<ArenaRegion> runtime_pool;

        std::shared_ptr<ArenaRegion> &region(size_t index) {
            if (index == 0) return gm_heap;
            if (index == 1) return gm_sm;
            return runtime_pool;
        }
    };

    // Retired regions keep their backing leased until their arena detaches it,
    // independently of other regions in the same bank.
    class RetiredRegion {
    public:
        int retain(const std::shared_ptr<ArenaRegion> &region, WorkspaceManager &workspace) {
            if (!region || region_) return PTO_RUNTIME_ERR_INVALID_STATE;
            WorkspaceManager::BackingLease backing;
            void *raw = region->arena.raw_backing();
            if (raw != nullptr && workspace.owns(raw)) {
                backing = workspace.retain_backing(raw);
                if (!backing) return PTO_RUNTIME_ERR_INVALID_STATE;
            }
            region_ = region;
            backing_ = std::move(backing);
            return 0;
        }

        bool in_use() const { return region_.use_count() > 1; }
        bool holds(const std::shared_ptr<ArenaRegion> &region) const { return region_ == region; }

        int reclaim(WorkspaceManager &workspace) {
            if (!region_) return 0;
            if (in_use()) return PTO_RUNTIME_ERR_INVALID_STATE;
            workspace.note_unpublished(region_->arena.raw_backing());
            region_->arena.release();
            region_.reset();
            backing_.reset();
            return 0;
        }

        void abandon_after_device_failure() {
            if (region_) region_->arena.abandon_after_device_failure();
            region_.reset();
            backing_.reset();
        }

    private:
        std::shared_ptr<ArenaRegion> region_;
        WorkspaceManager::BackingLease backing_;
    };

    PreparedRunResources(
        std::shared_ptr<ImageStorage> image, std::shared_ptr<RetainedSchedulerStorage> scheduler,
        std::vector<std::shared_ptr<ArenaRegion>> regions, std::vector<WorkspaceManager::BackingLease> backing
    ) :
        image_(std::move(image)),
        scheduler_(std::move(scheduler)),
        regions_(std::move(regions)),
        backing_(std::move(backing)) {}

    PreparedRunResources(
        std::shared_ptr<ImageStorage> image, std::shared_ptr<RetainedSchedulerStorage> scheduler,
        std::shared_ptr<ArenaRegion> region, std::vector<WorkspaceManager::BackingLease> backing
    ) :
        PreparedRunResources(
            std::move(image), std::move(scheduler),
            region ? std::vector<std::shared_ptr<ArenaRegion>>{std::move(region)} :
                     std::vector<std::shared_ptr<ArenaRegion>>{},
            std::move(backing)
        ) {}

    // A candidate is published only after allocation succeeds; same-capacity
    // acquires also detach when an older prepared execution holds the bytes.
    static int acquire_image(std::shared_ptr<ImageStorage> &cache, size_t bytes, size_t alignment, void **out) {
        if (out == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        *out = nullptr;
        if (cache && cache.use_count() == 1) return cache->acquire(bytes, alignment, out);
        auto next = std::make_shared<ImageStorage>();
        const int rc = next->acquire(bytes, alignment, out);
        if (rc == 0) cache = std::move(next);
        return rc;
    }

    template <typename Allocate, typename Free>
    static RetainedSchedulerStorage::Status acquire_scheduler(
        std::shared_ptr<RetainedSchedulerStorage> &cache,
        std::vector<std::shared_ptr<RetainedSchedulerStorage>> &retired, size_t bytes, size_t alignment,
        Allocate allocate, Free free, void **device_out, void **host_out
    ) {
        if (cache && cache.use_count() == 1)
            return cache->acquire(bytes, alignment, allocate, free, device_out, host_out);
        // Reserve bookkeeping before taking device ownership.
        if (cache) retired.reserve(retired.size() + 1);
        auto next = std::make_shared<RetainedSchedulerStorage>();
        const auto status = next->acquire(bytes, alignment, allocate, free, device_out, host_out);
        if (status == RetainedSchedulerStorage::Status::Ok) {
            if (cache) retired.push_back(cache);
            cache = std::move(next);
        }
        return status;
    }

    // Runtime-specific immutable publication description; its concrete type
    // belongs to the runtime, while its backing is held by the members below.
    void retain_initial_publication(std::shared_ptr<const void> publication) {
        initial_publication_ = std::move(publication);
    }

    // The prepared-resources object's lease on the retained staging block. It
    // is held for as long as any owner (the run use, a published owner, or an
    // external reference) can reach this preparation; the block's owner uses
    // this explicit count, not a shared_ptr count, to decide growth and
    // retirement.
    void hold_temporary(std::shared_ptr<TemporaryStorage> storage) {
        temporary_ = std::move(storage);
        if (temporary_) {
            temporary_->run_leases.fetch_add(1, std::memory_order_relaxed);
            temporary_lease_held_ = true;
        }
    }
    void release_temporary_lease() const {
        if (!temporary_lease_held_ || !temporary_) return;
        temporary_->run_leases.fetch_sub(1, std::memory_order_relaxed);
        temporary_lease_held_ = false;
    }
    // Transfer ownership of a pooled heap lease to these prepared resources.
    void take_heap_lease(std::shared_ptr<ScratchPool> pool, ScratchPool::Lease lease) {
        heap_pool_ = std::move(pool);
        heap_lease_ = lease;
    }

    PreparedRunResources(const PreparedRunResources &) = delete;
    PreparedRunResources &operator=(const PreparedRunResources &) = delete;

    ~PreparedRunResources() {
        // The object is the lease holder; its destruction ends the last host
        // reference to this preparation. The device blocks themselves are
        // retired or freed by the runner's explicit reclaim paths.
        release_temporary_lease();
        if (heap_pool_ != nullptr && heap_lease_.valid()) heap_pool_->release(heap_lease_);
    }

private:
    std::shared_ptr<const void> streams_;
    std::shared_ptr<TemporaryStorage> temporary_;
    mutable bool temporary_lease_held_{false};
    std::shared_ptr<const CallableUse> callable_;
    std::shared_ptr<PreparedSubmissionStorage> submission_;
    std::shared_ptr<const void> initial_publication_;
    std::shared_ptr<ImageStorage> image_;
    std::shared_ptr<RetainedSchedulerStorage> scheduler_;
    // The regions this preparation uses, independently of their bank numbers.
    std::vector<std::shared_ptr<ArenaRegion>> regions_;
    // The fixed context heap, when the contract selected one.
    std::shared_ptr<ScratchPool> heap_pool_;
    ScratchPool::Lease heap_lease_{};
    std::vector<WorkspaceManager::BackingLease> backing_;
};

// One use's obligations, independent of the immutable preparation and of its
// execution/result owner. Only the runner's proof-producing paths update it.
class PreparedRunUse {
public:
    explicit PreparedRunUse(std::shared_ptr<const PreparedRunResources> resources) :
        resources_(std::move(resources)) {}
    PreparedRunUse(const PreparedRunUse &) = delete;
    PreparedRunUse &operator=(const PreparedRunUse &) = delete;
    PreparedRunUse(PreparedRunUse &&) noexcept = default;
    PreparedRunUse &operator=(PreparedRunUse &&) noexcept = default;
    const std::shared_ptr<const PreparedRunResources> &resources() const { return resources_; }

    void note(WorkspaceManager::RunFact fact) {
        switch (fact) {
        case WorkspaceManager::RunFact::Launched:
            launched_ = true;
            break;
        case WorkspaceManager::RunFact::DrainProvedComplete:
            device_ended_ = true;
            break;
        case WorkspaceManager::RunFact::NoDeviceSubmission:
            not_submitted_ = true;
            break;
        case WorkspaceManager::RunFact::CopybackReturned:
            copyback_ended_ = true;
            break;
        case WorkspaceManager::RunFact::BindingsReleased:
            bindings_ended_ = true;
            break;
        case WorkspaceManager::RunFact::ContextDestroyed:
            host_ended_ = true;
            break;
        case WorkspaceManager::RunFact::DrainAttempted:
            break;
        }
    }
    void note_waits_consumed() { waits_ended_ = true; }
    bool can_retain_external_use() const { return !external_registered_ && !host_ended_; }
    bool retain_external_use() {
        if (!can_retain_external_use()) return false;
        external_registered_ = true;
        external_ended_ = false;
        return true;
    }
    // Caller reports no future use and completion of all external in-flight use.
    void note_external_use_ended() { external_ended_ = true; }
    bool internal_obligations_ended() const {
        const bool unused = not_submitted_ && !launched_;
        return host_ended_ && copyback_ended_ && bindings_ended_ && (unused || (device_ended_ && waits_ended_));
    }
    bool obligations_ended() const { return internal_obligations_ended() && external_ended_; }
    bool host_ended() const { return host_ended_; }

private:
    std::shared_ptr<const PreparedRunResources> resources_;
    bool launched_{false};
    bool not_submitted_{false};
    bool device_ended_{false};
    bool waits_ended_{false};
    bool copyback_ended_{false};
    bool bindings_ended_{false};
    bool host_ended_{false};
    bool external_registered_{false};
    bool external_ended_{true};
};
