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
// Which prefix each publication carries, driven through the real onboard
// helper and allocator with host-backed RTS storage. The question these cases
// answer is per device allocation, not per slot: a block that has never been
// published onto gets the handshake region, and only a block that has starts
// receiving the shorter prefix.
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "device_runner_helpers.h"

namespace {
// Byte every fresh device block is seeded with, standing in for what the device
// left there. A range still holding it was not copied into.
constexpr unsigned char kDeviceMark = 0xa5;

struct PendingCopy {
    void *target;
    const void *source;
    size_t bytes;
};

struct RtsState {
    bool defer_copies = false;
    std::vector<PendingCopy> pending;
    int host_alloc_rc = 0;
    int host_free_rc = 0;
    int host_allocations = 0;
    int host_frees = 0;
    int alloc_rc = 0;
    int copy_rc = 0;
    int copies = 0;
    LaunchProgressSink sink;
    int allocations = 0;
    int frees = 0;
    uint64_t last_copy_bytes = 0;
} rts;

class WorkersPublication : public ::testing::Test {
protected:
    void SetUp() override { rts = {}; }
    void TearDown() override {
        EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
        EXPECT_EQ(allocator.get_allocation_count(), expected_leaked_allocations);
    }

    // Blocks the case dropped without freeing. Only the quarantine path does
    // that, and it does it deliberately: a poisoned card cannot retire
    // per-resource frees, so the allocation outlives the slot and the force
    // reset in finalize() is what reclaims it.
    size_t expected_leaked_allocations = 0;

    uint64_t initializing_bytes() const { return runtime_device_initialized_prefix_size(runtime); }
    uint64_t steady_bytes() const { return runtime_device_copy_size(runtime); }

    // One prepare+publish against the slot, returning the publish rc.
    int publish_once() {
        const int prepare_rc = helper.prepare_runtime_args(runtime, allocator, slot);
        if (prepare_rc != 0) return prepare_rc;
        return helper.publish_runtime_args(/*launch_route_permitted=*/false, this, rts.sink);
    }

    // The block's handshake region, as the device would see it.
    const unsigned char *device_workers() const {
        return reinterpret_cast<const unsigned char *>(slot.runtime_args) + offsetof(DeviceRuntimeLaunchDesc, workers);
    }
    bool device_workers_all(unsigned char value) const {
        const unsigned char *bytes = device_workers();
        for (size_t i = 0; i < sizeof(DeviceRuntimeLaunchDesc::workers); ++i) {
            if (bytes[i] != value) return false;
        }
        return true;
    }

    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    Runtime runtime;
    KernelArgsHelper helper;
};
}  // namespace

extern "C" rtError_t rtMalloc(void **ptr, uint64_t bytes, uint32_t, uint16_t) {
    if (rts.alloc_rc != 0) return rts.alloc_rc;
    *ptr = std::malloc(bytes);
    if (*ptr == nullptr) return -1;
    std::memset(*ptr, kDeviceMark, bytes);
    ++rts.allocations;
    return 0;
}
extern "C" rtError_t rtFree(void *ptr) {
    ++rts.frees;
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMallocHost(void **ptr, uint64_t bytes, uint16_t) {
    if (rts.host_alloc_rc != 0) return rts.host_alloc_rc;
    ++rts.host_allocations;
    *ptr = std::malloc(bytes);
    return *ptr == nullptr ? -1 : 0;
}
extern "C" rtError_t rtFreeHost(void *ptr) {
    if (rts.host_free_rc != 0) return rts.host_free_rc;
    ++rts.host_frees;
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t) {
    ADD_FAILURE() << "runtime descriptor publication must be asynchronous";
    return -99;
}
extern "C" rtError_t
rtMemcpyAsync(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t) {
    ++rts.copies;
    rts.last_copy_bytes = bytes;
    EXPECT_EQ(kind, RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(capacity, bytes);
    if (rts.copy_rc != 0) return rts.copy_rc;
    if (rts.defer_copies) {
        rts.pending.push_back({dst, src, bytes});
    } else {
        std::memcpy(dst, src, bytes);
    }
    return 0;
}
extern "C" rtError_t rtStreamQuery(rtStream_t) { return 0; }
extern "C" int aclrtMemset(void *dst, size_t capacity, int value, size_t count) {
    if (count > capacity) return -1;
    std::memset(dst, value, count);
    return 0;
}
extern "C" const char *aclGetRecentErrMsg() { return nullptr; }
// These cases drive the publication step with the launch route refused, so the
// permit never consults a live stream; the query only has to link.
extern "C" aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus *status, aclmdlRI *) {
    if (status != nullptr) *status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    return ACL_SUCCESS;
}

TEST_F(WorkersPublication, TheFirstPublicationOntoABlockCarriesTheHandshakeRegion) {
    ASSERT_EQ(publish_once(), 0);

    EXPECT_EQ(rts.copies, 1) << "one H2D, not two";
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes());
    EXPECT_TRUE(slot.workers_initialized);
    EXPECT_TRUE(device_workers_all(0u)) << "the block's handshake region still holds allocator bytes";
}

TEST_F(WorkersPublication, AReusedBlockPublishesOnlyTheShorterPrefix) {
    ASSERT_EQ(publish_once(), 0);
    helper.release_run_view();

    // Stand in for what the device wrote during the first run. A second
    // publication that carried this region would erase it.
    std::memset(const_cast<unsigned char *>(device_workers()), 0x5a, sizeof(DeviceRuntimeLaunchDesc::workers));

    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.copies, 2) << "still one H2D per publication";
    EXPECT_EQ(rts.last_copy_bytes, steady_bytes());
    EXPECT_LT(rts.last_copy_bytes, initializing_bytes());
    EXPECT_TRUE(device_workers_all(0x5a)) << "a steady-state run overwrote the device's own handshake words";
}

TEST_F(WorkersPublication, AFailedFirstPublicationLeavesTheBlockUninitialized) {
    rts.copy_rc = -91;
    EXPECT_EQ(publish_once(), -91);
    EXPECT_FALSE(slot.workers_initialized) << "a copy that failed cannot have defined the region";
    EXPECT_NE(slot.runtime_args, nullptr) << "the block is retained for the retry";
    EXPECT_EQ(rts.frees, 0);

    rts.copy_rc = 0;
    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes()) << "the retry must initialize, not assume";
    EXPECT_TRUE(slot.workers_initialized);
    EXPECT_TRUE(device_workers_all(0u));
}

TEST_F(WorkersPublication, ANewAllocationDoesNotInheritInitialization) {
    ASSERT_EQ(publish_once(), 0);
    ASSERT_TRUE(slot.workers_initialized);

    ASSERT_EQ(release_slot_persistent_args(slot, allocator), 0);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_FALSE(slot.workers_initialized) << "the verdict belonged to the freed block";

    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.allocations, 2);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes()) << "a fresh block must be initialized again";
}

TEST_F(WorkersPublication, AnAbandonedSlotDoesNotInheritInitialization) {
    ASSERT_EQ(publish_once(), 0);
    ASSERT_TRUE(slot.workers_initialized);

    // The quarantine path drops ownership without freeing: the card is poisoned
    // and per-resource frees are unsafe.
    void *abandoned_source = slot.runtime_upload.data();
    abandon_slot_persistent_args(slot);
    std::free(abandoned_source);
    expected_leaked_allocations = 1;
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.runtime_bytes, 0U);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(rts.frees, 0);

    helper.release_run_view();
    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes());
}

// The commit belongs to the publication, not to a caller that has to remember
// to make it: preparing alone must not mark the block initialized.
TEST_F(WorkersPublication, PreparingWithoutPublishingCommitsNothing) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(rts.copies, 0);

    helper.release_run_view();
    EXPECT_FALSE(slot.workers_initialized);

    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes());
    EXPECT_TRUE(slot.workers_initialized);
}

TEST_F(WorkersPublication, MetadataSourcesRemainValidAfterTheRunViewIsReleased) {
    unsigned input = 23, target = 0;
    const HostMetadataRegion region{&target, &input, sizeof(input)};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&region, 1), 0);
    input = 47;
    rts.defer_copies = true;
    ASSERT_EQ(publish_once(), 0);
    ASSERT_EQ(rts.pending.size(), 2U);
    EXPECT_EQ(rts.pending[0].target, &target);
    EXPECT_EQ(rts.pending[0].source, static_cast<std::byte *>(slot.runtime_upload.data()) + slot.runtime_bytes);
    EXPECT_EQ(rts.pending[1].source, slot.runtime_upload.data());
    EXPECT_EQ(target, 0U);
    EXPECT_TRUE(rts.sink.submitted());
    helper.release_run_view();
    slot.staged_launch_metadata.clear();
    for (const auto &copy : rts.pending)
        std::memcpy(copy.target, copy.source, copy.bytes);
    EXPECT_EQ(target, 23U);
    EXPECT_TRUE(device_workers_all(0u));
}

TEST_F(WorkersPublication, FailedPinnedGrowthPreservesCommittedSource) {
    ASSERT_EQ(publish_once(), 0);
    helper.release_run_view();
    void *original = slot.runtime_upload.data();
    const auto capacity = slot.runtime_upload.capacity();
    std::memset(original, 0x3c, capacity);
    uint64_t target = 0;
    const uint64_t value = 17;
    const HostMetadataRegion metadata{&target, &value, sizeof(value)};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&metadata, 1), 0);
    rts.host_alloc_rc = -4321;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), -4321);
    EXPECT_EQ(slot.runtime_upload.data(), original);
    EXPECT_EQ(slot.runtime_upload.capacity(), capacity);
    EXPECT_EQ(rts.host_frees, 0);
    if (slot.runtime_upload.data() == original) {
        EXPECT_EQ(static_cast<unsigned char *>(original)[0], 0x3c);
        EXPECT_EQ(static_cast<unsigned char *>(original)[capacity - 1], 0x3c);
    }
    rts.host_alloc_rc = 0;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_GE(slot.runtime_upload.capacity(), capacity + sizeof(value));
}

TEST_F(WorkersPublication, FailedPinnedRollbackRetainsBothAllocationsForRetry) {
    ASSERT_EQ(publish_once(), 0);
    helper.release_run_view();
    void *original = slot.runtime_upload.data();
    const auto capacity = slot.runtime_upload.capacity();
    rts.host_free_rc = -4322;
    EXPECT_EQ(slot.runtime_upload.reserve(capacity + 8), -4322);
    EXPECT_EQ(slot.runtime_upload.data(), original);
    EXPECT_EQ(slot.runtime_upload.capacity(), capacity);
    EXPECT_EQ(rts.host_allocations, 2);
    EXPECT_EQ(rts.host_frees, 0);
    EXPECT_EQ(slot.runtime_upload.reserve(capacity + 16), -4322);
    EXPECT_EQ(rts.host_allocations, 2) << "failed rollback cannot grow unbounded debt";
    EXPECT_EQ(slot.runtime_upload.release(), -4322);
    EXPECT_EQ(slot.runtime_upload.data(), original);
    rts.host_free_rc = 0;
    EXPECT_EQ(slot.runtime_upload.release(), 0);
    EXPECT_EQ(rts.host_frees, 2);
    EXPECT_EQ(slot.runtime_upload.data(), nullptr);
    EXPECT_EQ(slot.runtime_upload.capacity(), 0U);
}

TEST_F(WorkersPublication, MovedPinnedSourceOutlivesRunViewAndSlotRelease) {
    uint64_t value = 79, target = 0;
    const HostMetadataRegion metadata{&target, &value, sizeof(value)};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&metadata, 1), 0);
    rts.defer_copies = true;
    ASSERT_EQ(publish_once(), 0);
    ASSERT_GE(rts.pending.size(), 2U);
    const PendingCopy metadata_copy = rts.pending.front();
    void *original = slot.runtime_upload.data();
    PinnedUploadStorage retained(std::move(slot.runtime_upload));
    EXPECT_EQ(retained.data(), original);
    EXPECT_EQ(slot.runtime_upload.data(), nullptr);
    EXPECT_EQ(slot.runtime_upload.capacity(), 0U);
    helper.release_run_view();
    // Only the source is transferred here. The device descriptor destination
    // is not retained, so no deferred descriptor copy may touch it after release.
    rts.pending.clear();
    ASSERT_EQ(release_slot_persistent_args(slot, allocator), 0);
    EXPECT_EQ(rts.host_frees, 0);
    value = 101;
    std::memcpy(metadata_copy.target, metadata_copy.source, metadata_copy.bytes);
    EXPECT_EQ(target, 79U);
    ASSERT_EQ(retained.release(), 0);
    EXPECT_EQ(rts.host_frees, 1);
    EXPECT_EQ(retained.release(), 0);
    EXPECT_EQ(rts.host_frees, 1);
}

TEST_F(WorkersPublication, PinnedCapacityReuseDoesNotAllocateOrReplace) {
    ASSERT_EQ(publish_once(), 0);
    void *original = slot.runtime_upload.data();
    const auto capacity = slot.runtime_upload.capacity();
    ASSERT_EQ(slot.runtime_upload.reserve(capacity), 0);
    ASSERT_EQ(slot.runtime_upload.reserve(capacity / 2), 0);
    EXPECT_EQ(slot.runtime_upload.data(), original);
    EXPECT_EQ(slot.runtime_upload.capacity(), capacity);
    EXPECT_EQ(rts.host_allocations, 1);
    EXPECT_EQ(rts.host_frees, 0);
}

TEST_F(WorkersPublication, MovingPinnedStorageTransfersRollbackDebt) {
    ASSERT_EQ(slot.runtime_upload.reserve(64), 0);
    rts.host_free_rc = -4323;
    EXPECT_EQ(slot.runtime_upload.reserve(128), -4323);
    PinnedUploadStorage retained(std::move(slot.runtime_upload));
    EXPECT_EQ(slot.runtime_upload.release(), 0);
    EXPECT_EQ(rts.host_frees, 0);
    EXPECT_EQ(retained.release(), -4323);
    rts.host_free_rc = 0;
    EXPECT_EQ(retained.release(), 0);
    EXPECT_EQ(rts.host_frees, 2);
}
