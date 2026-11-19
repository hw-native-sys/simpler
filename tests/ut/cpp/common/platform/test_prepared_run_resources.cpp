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

#include <gtest/gtest.h>
#include <cstring>
#include <cstdlib>
#include <future>
#include <thread>
#include <type_traits>
#include "host/prepared_run_resources.h"

static_assert(!std::is_copy_constructible_v<PreparedRunResources>);
static_assert(!std::is_copy_assignable_v<PreparedRunResources>);

namespace {
void end_host_use(PreparedRunUse &use) {
    use.note(WorkspaceManager::RunFact::CopybackReturned);
    use.note(WorkspaceManager::RunFact::BindingsReleased);
    use.note(WorkspaceManager::RunFact::ContextDestroyed);
}
std::shared_ptr<PreparedRunResources> make_preparation() {
    return std::make_shared<PreparedRunResources>(
        nullptr, nullptr, nullptr, std::vector<WorkspaceManager::BackingLease>{}
    );
}
}  // namespace

TEST(PreparedRunUse, DestroyingRunAndDroppingReferencesDoesNotProveDeviceCompletion) {
    auto prepared = make_preparation();
    std::weak_ptr<PreparedRunResources> witness = prepared;
    PreparedRunUse use(prepared);
    use.note(WorkspaceManager::RunFact::Launched);
    use.note(WorkspaceManager::RunFact::DrainAttempted);
    end_host_use(use);
    prepared.reset();
    EXPECT_FALSE(witness.expired());
    EXPECT_FALSE(use.obligations_ended());
    use.note_waits_consumed();
    EXPECT_FALSE(use.obligations_ended());
    use.note(WorkspaceManager::RunFact::DrainProvedComplete);
    EXPECT_TRUE(use.obligations_ended());
}

TEST(PreparedRunUse, DeviceCompletionDoesNotProveQueuedWaitConsumption) {
    PreparedRunUse use(make_preparation());
    use.note(WorkspaceManager::RunFact::Launched);
    use.note(WorkspaceManager::RunFact::DrainProvedComplete);
    end_host_use(use);
    EXPECT_FALSE(use.obligations_ended());
    use.note_waits_consumed();
    EXPECT_TRUE(use.obligations_ended());
}

TEST(PreparedRunUse, UnsubmittedRunNeedsNoDeviceWaitButLaunchedFactOverridesIt) {
    PreparedRunUse use(make_preparation());
    use.note(WorkspaceManager::RunFact::NoDeviceSubmission);
    use.note(WorkspaceManager::RunFact::CopybackReturned);
    use.note(WorkspaceManager::RunFact::ContextDestroyed);
    EXPECT_FALSE(use.obligations_ended());
    use.note(WorkspaceManager::RunFact::BindingsReleased);
    EXPECT_TRUE(use.obligations_ended());
    use.note(WorkspaceManager::RunFact::Launched);
    EXPECT_FALSE(use.obligations_ended());
}

TEST(PreparedRunResources, SameCapacityAndGrowthPreserveRetainedImage) {
    std::shared_ptr<PreparedRunResources::ImageStorage> cache;
    void *first = nullptr;
    ASSERT_EQ(PreparedRunResources::acquire_image(cache, 128, 64, &first), 0);
    std::memset(first, 0xa5, 128);
    auto held =
        std::make_shared<PreparedRunResources>(cache, nullptr, nullptr, std::vector<WorkspaceManager::BackingLease>{});
    void *second = nullptr;
    ASSERT_EQ(PreparedRunResources::acquire_image(cache, 128, 64, &second), 0);
    EXPECT_NE(first, second);
    std::memset(second, 0x5a, 128);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(static_cast<unsigned char *>(first)[i], 0xa5);
    ASSERT_EQ(PreparedRunResources::acquire_image(cache, 4096, 64, &second), 0);
    for (size_t i = 0; i < 128; ++i)
        EXPECT_EQ(static_cast<unsigned char *>(first)[i], 0xa5);
    cache.reset();
    EXPECT_EQ(static_cast<unsigned char *>(first)[127], 0xa5);
}

TEST(PreparedRunResources, UnheldImageReusesCapacityAndInvalidRequestKeepsCache) {
    std::shared_ptr<PreparedRunResources::ImageStorage> cache;
    void *first = nullptr, *second = nullptr;
    ASSERT_EQ(PreparedRunResources::acquire_image(cache, 128, 64, &first), 0);
    ASSERT_EQ(PreparedRunResources::acquire_image(cache, 64, 64, &second), 0);
    EXPECT_EQ(first, second);
    auto previous = cache;
    EXPECT_NE(PreparedRunResources::acquire_image(cache, 0, 64, &second), 0);
    EXPECT_EQ(second, nullptr);
    EXPECT_EQ(cache, previous);
    EXPECT_NE(PreparedRunResources::acquire_image(cache, SIZE_MAX, 64, &second), 0);
    EXPECT_EQ(cache, previous);
}

TEST(PreparedRunResources, SchedulerSharesCapacityAndRetainsOldDeviceAddressOnlyAcrossGrowth) {
    std::shared_ptr<RetainedSchedulerStorage> cache;
    std::vector<std::shared_ptr<RetainedSchedulerStorage>> retired;
    bool fail_allocate = false;
    auto allocate = [&](size_t n) -> void * {
        return fail_allocate ? nullptr : std::malloc(n);
    };
    auto release = [](void *p) {
        std::free(p);
        return 0;
    };
    void *first = nullptr;
    ASSERT_EQ(
        PreparedRunResources::acquire_scheduler(cache, retired, 256, 64, allocate, release, &first),
        RetainedSchedulerStorage::Status::Ok
    );
    auto held =
        std::make_shared<PreparedRunResources>(nullptr, cache, nullptr, std::vector<WorkspaceManager::BackingLease>{});
    std::memset(first, 0x5a, 256);

    fail_allocate = true;
    void *second = nullptr;
    ASSERT_EQ(
        PreparedRunResources::acquire_scheduler(cache, retired, 128, 64, allocate, release, &second),
        RetainedSchedulerStorage::Status::Ok
    );
    EXPECT_EQ(first, second);
    EXPECT_EQ(
        PreparedRunResources::acquire_scheduler(cache, retired, 512, 64, allocate, release, &second),
        RetainedSchedulerStorage::Status::DeviceUnavailable
    );
    EXPECT_EQ(second, nullptr);
    EXPECT_EQ(cache->device_addr(), first);
    EXPECT_TRUE(retired.empty());

    fail_allocate = false;
    ASSERT_EQ(
        PreparedRunResources::acquire_scheduler(cache, retired, 512, 64, allocate, release, &second),
        RetainedSchedulerStorage::Status::Ok
    );
    EXPECT_NE(first, second);
    ASSERT_EQ(retired.size(), 1U);
    EXPECT_EQ(retired.front()->device_addr(), first);
    std::memset(second, 0xa5, 512);
    EXPECT_EQ(static_cast<unsigned char *>(first)[255], 0x5a);
    held.reset();
    ASSERT_EQ(retired.front().use_count(), 1);
    EXPECT_EQ(retired.front()->release_preserving_failure(release), 0);
    EXPECT_EQ(cache->release_preserving_failure(release), 0);
}

namespace {
struct ReusedArenaBackend {
    WorkspaceManager workspace;
    void *address = reinterpret_cast<void *>(0x100000);
    bool live{false};
    bool fail_release{false};
    int frees{0};
    int arena_releases{0};
    uint64_t epoch{1};
    WorkspaceManager::RegionKey region = WorkspaceManager::arena_region(0, WorkspaceManager::ArenaRegion::RuntimePool);

    ReusedArenaBackend() {
        WorkspaceManager::Backend ops{};
        ops.ctx = this;
        ops.acquire = [](void *ctx, size_t) -> void * {
            auto &self = *static_cast<ReusedArenaBackend *>(ctx);
            if (self.live) return nullptr;
            self.live = true;
            return self.address;
        };
        ops.release = [](void *ctx, void *, int *rc) {
            auto &self = *static_cast<ReusedArenaBackend *>(ctx);
            if (self.fail_release) {
                *rc = -7;
                return WorkspaceManager::ReleaseOutcome::FreeFailed;
            }
            self.live = false;
            ++self.frees;
            return WorkspaceManager::ReleaseOutcome::Freed;
        };
        EXPECT_TRUE(workspace.configure(ops));
    }
    static void *allocate(void *ctx, size_t bytes) {
        auto &self = *static_cast<ReusedArenaBackend *>(ctx);
        return self.workspace.acquire(self.region, self.epoch, bytes);
    }
    static void release_arena(void *ctx, void *address) {
        auto &self = *static_cast<ReusedArenaBackend *>(ctx);
        ++self.arena_releases;
        EXPECT_TRUE(self.workspace.owns(address));
        EXPECT_TRUE(self.workspace.must_keep(address));
        // Managed allocations are released by the workspace backend.
    }
    void retire_run() {
        for (auto fact :
             {WorkspaceManager::RunFact::NoDeviceSubmission, WorkspaceManager::RunFact::CopybackReturned,
              WorkspaceManager::RunFact::BindingsReleased, WorkspaceManager::RunFact::ContextDestroyed})
            workspace.note_run_fact(0, epoch, fact);
    }
};
}  // namespace

TEST(PreparedRunResources, RetainingImageDoesNotKeepSiblingHeapAlive) {
    std::vector<void *> freed;
    auto bank = std::make_shared<PreparedRunResources::ArenaBank>(
        [](void *, size_t bytes) {
            return std::malloc(bytes);
        },
        [](void *ctx, void *ptr) {
            static_cast<std::vector<void *> *>(ctx)->push_back(ptr);
            std::free(ptr);
        },
        &freed
    );
    bank->gm_heap->arena.reserve(64, 64);
    bank->runtime_pool->arena.reserve(64, 64);
    ASSERT_NE(bank->gm_heap->arena.commit(), nullptr);
    ASSERT_NE(bank->runtime_pool->arena.commit(), nullptr);
    void *heap = bank->gm_heap->arena.raw_backing();
    void *image = bank->runtime_pool->arena.raw_backing();
    auto held = std::make_shared<PreparedRunResources>(
        nullptr, nullptr, bank->runtime_pool, std::vector<WorkspaceManager::BackingLease>{}
    );
    bank.reset();
    EXPECT_EQ(freed, std::vector<void *>{heap});
    held.reset();
    EXPECT_EQ(freed, (std::vector<void *>{heap, image}));
}

TEST(PreparedRunResources, RetiredRegionPinsBackingUntilItsAddressesAreDetached) {
    ReusedArenaBackend backend;
    auto bank = std::make_shared<PreparedRunResources::ArenaRegion>(
        &ReusedArenaBackend::allocate, &ReusedArenaBackend::release_arena, &backend
    );
    bank->arena.reserve(64, 64);
    ASSERT_NE(bank->arena.commit(), nullptr);
    auto original = std::make_shared<PreparedRunResources>(
        nullptr, nullptr, bank,
        std::vector<WorkspaceManager::BackingLease>{backend.workspace.retain_backing(backend.address)}
    );
    PreparedRunResources::RetiredRegion retired;
    ASSERT_EQ(retired.retain(bank, backend.workspace), 0);
    bank.reset();
    EXPECT_NE(retired.reclaim(backend.workspace), 0);
    backend.retire_run();
    backend.workspace.note_unpublished(backend.address);
    original.reset();
    ASSERT_EQ(backend.workspace.reclaim_obsolete(), 0);
    EXPECT_EQ(backend.frees, 0);
    EXPECT_TRUE(backend.workspace.must_keep(backend.address));
    EXPECT_EQ(backend.workspace.acquire(WorkspaceManager::staging_region(0), 2, 128), nullptr);

    ASSERT_EQ(retired.reclaim(backend.workspace), 0);
    EXPECT_EQ(backend.arena_releases, 1);
    EXPECT_FALSE(backend.workspace.must_keep(backend.address));
    ASSERT_EQ(backend.workspace.reclaim_obsolete(), 0);
    EXPECT_EQ(backend.frees, 1);
    void *replacement = backend.workspace.acquire(WorkspaceManager::staging_region(0), 2, 128);
    ASSERT_EQ(replacement, backend.address);
    backend.epoch = 2;
    backend.retire_run();
    ASSERT_EQ(retired.reclaim(backend.workspace), 0);
    ASSERT_EQ(backend.workspace.reclaim_obsolete(), 0);
    EXPECT_EQ(backend.frees, 1);
    EXPECT_TRUE(backend.workspace.idle_backing(replacement));
}

TEST(PreparedRunResources, RetiredRegionAbandonMakesNoDeviceCalls) {
    ReusedArenaBackend backend;
    auto bank = std::make_shared<PreparedRunResources::ArenaRegion>(
        &ReusedArenaBackend::allocate, &ReusedArenaBackend::release_arena, &backend
    );
    bank->arena.reserve(64, 64);
    ASSERT_NE(bank->arena.commit(), nullptr);
    PreparedRunResources::RetiredRegion retired;
    ASSERT_EQ(retired.retain(bank, backend.workspace), 0);
    bank.reset();
    retired.abandon_after_device_failure();
    EXPECT_EQ(backend.arena_releases, 0);
    EXPECT_EQ(backend.frees, 0);
    EXPECT_FALSE(backend.workspace.must_keep(backend.address));
}

TEST(PreparedRunResources, RetiredRegionReleaseFailureRemainsOwnedByWorkspace) {
    ReusedArenaBackend backend;
    auto bank = std::make_shared<PreparedRunResources::ArenaRegion>(
        &ReusedArenaBackend::allocate, &ReusedArenaBackend::release_arena, &backend
    );
    bank->arena.reserve(64, 64);
    ASSERT_NE(bank->arena.commit(), nullptr);
    PreparedRunResources::RetiredRegion retired;
    ASSERT_EQ(retired.retain(bank, backend.workspace), 0);
    bank.reset();
    backend.retire_run();
    ASSERT_EQ(retired.reclaim(backend.workspace), 0);
    backend.fail_release = true;
    EXPECT_EQ(backend.workspace.reclaim_obsolete(), -7);
    EXPECT_TRUE(backend.workspace.owns(backend.address));
    EXPECT_EQ(backend.workspace.block_state(backend.address), WorkspaceManager::BlockState::ReleaseUnconfirmed);
    EXPECT_EQ(backend.workspace.acquire(WorkspaceManager::staging_region(0), 2, 128), nullptr);
    ASSERT_EQ(retired.reclaim(backend.workspace), 0);
    EXPECT_EQ(backend.arena_releases, 1);
    EXPECT_EQ(backend.frees, 0);
}

TEST(PreparedRunResources, RetiredUnmanagedRegionFreesExactlyOnce) {
    WorkspaceManager workspace;
    int frees = 0;
    auto bank = std::make_shared<PreparedRunResources::ArenaRegion>(
        [](void *, size_t bytes) {
            return std::malloc(bytes);
        },
        [](void *ctx, void *ptr) {
            ++*static_cast<int *>(ctx);
            std::free(ptr);
        },
        &frees
    );
    bank->arena.reserve(64, 64);
    ASSERT_NE(bank->arena.commit(), nullptr);
    PreparedRunResources::RetiredRegion retired;
    ASSERT_EQ(retired.retain(bank, workspace), 0);
    EXPECT_NE(retired.reclaim(workspace), 0);
    EXPECT_EQ(frees, 0);
    bank.reset();
    ASSERT_EQ(retired.reclaim(workspace), 0);
    EXPECT_EQ(frees, 1);
    ASSERT_EQ(retired.reclaim(workspace), 0);
    EXPECT_EQ(frees, 1);
}

TEST(PreparedRunResources, PublishedOwnerLifetimeIsIndependentOfInternalUseCopies) {
    auto backing = make_preparation();
    PreparedRunUse use(backing);
    auto published = PreparedRunResources::publish_owner(backing);
    std::weak_ptr<const PreparedRunResources> owner = published;
    std::weak_ptr<const PreparedRunResources> storage = backing;
    auto internal_copy = use.resources();
    auto another_internal_copy = internal_copy;
    backing.reset();
    EXPECT_FALSE(owner.expired());
    EXPECT_EQ(published.get(), internal_copy.get());
    published.reset();
    EXPECT_TRUE(owner.expired());
    EXPECT_FALSE(storage.expired());
    EXPECT_FALSE(use.obligations_ended());
}

TEST(PreparedRunResources, OwnerCopiesKeepTheSameRegisteredLifetime) {
    auto published = PreparedRunResources::publish_owner(make_preparation());
    std::weak_ptr<const PreparedRunResources> owner = published;
    auto copy = published;
    published.reset();
    EXPECT_FALSE(owner.expired());
    copy.reset();
    EXPECT_TRUE(owner.expired());
}

TEST(PreparedRunResources, TemporaryLeasesEndAcrossIndependentOwnerThreads) {
    auto storage = std::make_shared<PreparedRunResources::TemporaryStorage>();
    auto first = make_preparation();
    auto second = make_preparation();
    first->hold_temporary(storage);
    second->hold_temporary(storage);
    ASSERT_EQ(storage->run_leases.load(std::memory_order_relaxed), 2u);
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::thread owner([resource = std::move(first), ready]() mutable {
        ready.wait();
        resource.reset();
    });
    start.set_value();
    second.reset();
    auto successor = make_preparation();
    successor->hold_temporary(storage);
    owner.join();
    EXPECT_EQ(storage->run_leases.load(std::memory_order_relaxed), 1u);
    successor.reset();
    EXPECT_EQ(storage->run_leases.load(std::memory_order_relaxed), 0u);
}
