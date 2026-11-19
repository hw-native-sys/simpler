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
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "host/aligned_image_storage.h"
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

    struct ArenaBank {
        ArenaBank(DeviceArena::AllocFn alloc, DeviceArena::FreeFn free_fn, void *ctx) :
            gm_heap(alloc, free_fn, ctx),
            gm_sm(alloc, free_fn, ctx),
            runtime_pool(alloc, free_fn, ctx) {}
        DeviceArena gm_heap;
        DeviceArena gm_sm;
        DeviceArena runtime_pool;
        size_t cached_gm_heap_size{0};
        size_t cached_gm_sm_size{0};
        size_t cached_runtime_arena_size{0};
    };

    // A retired bank still names its raw allocations after its last execution
    // ends. These leases outlive that execution and remain until the bank has
    // detached every address. Only then may the ledger reuse those addresses.
    class RetiredArena {
    public:
        int retain(const std::shared_ptr<ArenaBank> &bank, WorkspaceManager &workspace) {
            if (!bank || bank_) return PTO_RUNTIME_ERR_INVALID_STATE;
            std::vector<WorkspaceManager::BackingLease> backing;
            backing.reserve(3);
            for (void *raw :
                 {bank->gm_heap.raw_backing(), bank->gm_sm.raw_backing(), bank->runtime_pool.raw_backing()}) {
                if (raw == nullptr || !workspace.owns(raw)) continue;
                auto lease = workspace.retain_backing(raw);
                if (!lease) return PTO_RUNTIME_ERR_INVALID_STATE;
                backing.push_back(std::move(lease));
            }
            bank_ = bank;
            backing_ = std::move(backing);
            return 0;
        }

        bool in_use() const { return bank_.use_count() > 1; }
        bool holds(const std::shared_ptr<ArenaBank> &bank) const { return bank_ == bank; }

        int reclaim(WorkspaceManager &workspace) {
            if (!bank_) return 0;
            if (in_use()) return PTO_RUNTIME_ERR_INVALID_STATE;
            for (DeviceArena *arena : {&bank_->gm_heap, &bank_->gm_sm, &bank_->runtime_pool}) {
                workspace.note_unpublished(arena->raw_backing());
                arena->release();
            }
            bank_.reset();
            backing_.clear();
            return 0;
        }

        void abandon_after_device_failure() {
            if (bank_) {
                bank_->gm_heap.abandon_after_device_failure();
                bank_->gm_sm.abandon_after_device_failure();
                bank_->runtime_pool.abandon_after_device_failure();
            }
            bank_.reset();
            backing_.clear();
        }

    private:
        std::shared_ptr<ArenaBank> bank_;
        std::vector<WorkspaceManager::BackingLease> backing_;
    };

    PreparedRunResources(
        std::shared_ptr<ImageStorage> image, std::shared_ptr<RetainedSchedulerStorage> scheduler,
        std::shared_ptr<ArenaBank> arena, std::vector<WorkspaceManager::BackingLease> backing
    ) :
        image_(std::move(image)),
        scheduler_(std::move(scheduler)),
        arena_(std::move(arena)),
        backing_(std::move(backing)) {}

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

private:
    std::shared_ptr<const void> streams_;
    std::shared_ptr<TemporaryStorage> temporary_;
    std::shared_ptr<const CallableUse> callable_;
    std::shared_ptr<PreparedSubmissionStorage> submission_;
    std::shared_ptr<const void> initial_publication_;
    std::shared_ptr<ImageStorage> image_;
    std::shared_ptr<RetainedSchedulerStorage> scheduler_;
    std::shared_ptr<ArenaBank> arena_;
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
