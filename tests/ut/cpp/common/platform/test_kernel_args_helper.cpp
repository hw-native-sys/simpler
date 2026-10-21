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
// Exercise the production onboard helper and allocator with host-backed RTS
// storage. The stub observes API boundaries; descriptor preparation stays real.
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#include <vector>
#include <limits>

#include "acl/acl.h"
#include "device_runner_helpers.h"

namespace {
// Byte every fresh device block is seeded with, standing in for what the device
// left there. A range still holding it was not copied into.
constexpr unsigned char kDeviceMark = 0xa5;

struct RtsState {
    int alloc_rc = 0;
    int copy_rc = 0;
    int copies = 0;
    int async_copies = 0;
    int fail_copy_on = 0;
    struct Copy {
        void *target;
        const void *source;
        size_t bytes;
        rtStream_t stream;
    };
    std::vector<Copy> queued;
    int host_alloc_rc = 0;
    int host_free_rc = 0;
    int host_allocations = 0;
    int host_frees = 0;
    LaunchProgressSink sink;
    bool defer = false;
    void *pending_target = nullptr;
    const void *pending_source = nullptr;
    size_t pending_bytes = 0;
    rtStream_t stream = nullptr;
    int allocations = 0;
    int frees = 0;
    // What a publication is expected to carry. Which of the two prefixes that
    // is depends on the destination block, not on the case: the first
    // publication onto a block adds the handshake region, every later one stops
    // before it. The stub reads the slot rather than a fixed number so each
    // copy is checked against the state it was actually issued in.
    uint64_t initializing_copy_bytes = 0;
    uint64_t steady_copy_bytes = 0;
    const SlotPersistentArgs *destination = nullptr;
} rts;

class KernelArgsPublication : public ::testing::Test {
protected:
    void SetUp() override {
        rts = {};
        rts.initializing_copy_bytes = runtime_device_initialized_prefix_size(runtime);
        rts.steady_copy_bytes = runtime_device_copy_size(runtime);
        rts.destination = &slot;
    }
    void TearDown() override {
        EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
        EXPECT_EQ(allocator.get_allocation_count(), 0U);
    }

    MemoryAllocator allocator;
    SlotPersistentArgs slot;
    Runtime runtime;
    KernelArgsHelper helper;

    int published_worker_count() const {
        DeviceRuntimeLaunchDesc descriptor;
        std::memcpy(&descriptor, slot.runtime_args, sizeof(descriptor));
        return descriptor.worker_count;
    }
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
static rtError_t copy_descriptor(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind) {
    ++rts.copies;
    EXPECT_EQ(kind, RT_MEMCPY_HOST_TO_DEVICE);
    // A published prefix, not the allocated extent: a publication that carried
    // the whole descriptor would reach storage no host value belongs in. The
    // commit happens after this call returns, so the slot still shows the state
    // this copy was issued in.
    const bool initializing = rts.destination != nullptr && !rts.destination->workers_initialized;
    if (rts.destination != nullptr && dst == rts.destination->runtime_args)
        EXPECT_EQ(bytes, initializing ? rts.initializing_copy_bytes : rts.steady_copy_bytes);
    EXPECT_EQ(capacity, bytes);
    if (rts.copy_rc != 0 && (rts.fail_copy_on == 0 || rts.copies == rts.fail_copy_on)) return rts.copy_rc;
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t rtMallocHost(void **ptr, uint64_t bytes, uint16_t) {
    if (rts.host_alloc_rc != 0) return rts.host_alloc_rc;
    *ptr = std::malloc(bytes);
    if (*ptr == nullptr) return -1;
    ++rts.host_allocations;
    return 0;
}
extern "C" rtError_t rtFreeHost(void *ptr) {
    if (rts.host_free_rc != 0) return rts.host_free_rc;
    ++rts.host_frees;
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *, uint64_t, const void *, uint64_t, rtMemcpyKind_t) {
    ADD_FAILURE() << "descriptor publication must enqueue on its execution stream";
    return -99;
}
extern "C" rtError_t
rtMemcpyAsync(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t stream) {
    ++rts.async_copies;
    rts.stream = stream;
    rts.queued.push_back({dst, src, bytes, stream});
    if (rts.defer) {
        ++rts.copies;
        rts.pending_target = dst;
        rts.pending_source = src;
        rts.pending_bytes = bytes;
        return rts.copy_rc;
    }
    return copy_descriptor(dst, capacity, src, bytes, kind);
}
extern "C" rtError_t rtStreamQuery(rtStream_t) { return 0; }
extern "C" const char *aclGetRecentErrMsg() { return nullptr; }
// These cases drive the publication step with the launch route refused, so the
// permit never consults a live stream; the query only has to link.
extern "C" aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus *status, aclmdlRI *) {
    if (status != nullptr) *status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    return ACL_SUCCESS;
}

TEST(KernelArgsDefaults, FreshRuntimeStartsWithDiagnosticsDisabled) {
    alignas(Runtime) unsigned char storage[sizeof(Runtime)];
    std::memset(storage, kDeviceMark, sizeof(storage));
    auto *fresh = new (storage) Runtime;
    EXPECT_EQ(fresh->dev.chip_swimlane_level, 0U);
    EXPECT_EQ(fresh->dev.pmu_event_type, 0U);
    fresh->~Runtime();
}

TEST_F(KernelArgsPublication, PrepareDoesNotPublishAndOwnsAnIndependentSnapshot) {
    EXPECT_FALSE(helper.runtime_args_published());
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_NE(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(slot.runtime_bytes, sizeof(DeviceRuntimeLaunchDesc));
    EXPECT_EQ(allocator.committed_bytes(), sizeof(DeviceRuntimeLaunchDesc));
    const auto *bytes = reinterpret_cast<const unsigned char *>(slot.runtime_args);
    for (size_t i = 0; i < slot.runtime_bytes; ++i)
        ASSERT_EQ(bytes[i], kDeviceMark);

    runtime.dev.worker_count = 19;
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(published_worker_count(), 7);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_EQ(helper.args.runtime_args, slot.runtime_args);
    EXPECT_TRUE(helper.runtime_args_published());
    EXPECT_NE(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(rts.copies, 1);
    EXPECT_TRUE(helper.runtime_args_published());
}

TEST_F(KernelArgsPublication, FailedPublicationWithdrawsRunViewButRetainsSlotForFreshPrepare) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    Runtime *destination = slot.runtime_args;
    rts.copy_rc = -91;
    EXPECT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), -91);
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_EQ(allocator.get_allocation_count(), 1U);
    EXPECT_EQ(rts.frees, 0);

    rts.copy_rc = 0;
    EXPECT_NE(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(rts.copies, 1);
    runtime.dev.worker_count = 3;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(published_worker_count(), 3);
    EXPECT_TRUE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_EQ(rts.allocations, 1);
}

TEST_F(KernelArgsPublication, AllocationFailureLeavesNoPublishableRun) {
    rts.alloc_rc = -92;
    EXPECT_NE(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.runtime_bytes, 0U);
    EXPECT_EQ(allocator.get_allocation_count(), 0U);
    EXPECT_NE(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(rts.copies, 0);
}

TEST_F(KernelArgsPublication, RepreparePreservesThePendingSnapshotAndDestination) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    Runtime *destination = slot.runtime_args;
    runtime.dev.worker_count = 19;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(helper.args.runtime_args, destination);
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(rts.copies, 0);
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(published_worker_count(), 7);
}

TEST_F(KernelArgsPublication, RejectedReprepareDoesNotTouchAnotherSlot) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    SlotPersistentArgs other;
    runtime.dev.worker_count = 19;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, other), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(other.runtime_args, nullptr);
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(helper.args.runtime_args, slot.runtime_args);
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(published_worker_count(), 7);
    EXPECT_EQ(release_slot_persistent_args(other, allocator), 0);
}

TEST_F(KernelArgsPublication, MismatchedSlotLeavesNoPublishableRun) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    helper.release_run_view();
    Runtime *destination = slot.runtime_args;
    ++slot.runtime_bytes;
    EXPECT_NE(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_NE(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(rts.frees, 0);
}

TEST_F(KernelArgsPublication, ReleaseDiscardsPendingSourceWithoutFreeingDestination) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    Runtime *destination = slot.runtime_args;
    helper.release_run_view();
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_NE(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(rts.frees, 0);
    EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(slot.runtime_bytes, 0U);
    EXPECT_EQ(rts.frees, 1);
}

TEST_F(KernelArgsPublication, MovedHelperRetainsSnapshotAfterSourceRunViewIsReleased) {
    runtime.dev.worker_count = 11;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    KernelArgsHelper moved(std::move(helper));
    helper.release_run_view();
    EXPECT_EQ(helper.args.runtime_args, nullptr);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_NE(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    runtime.dev.worker_count = 23;
    ASSERT_EQ(moved.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_EQ(published_worker_count(), 11);
    EXPECT_EQ(rts.copies, 1);
    moved.release_run_view();
    EXPECT_EQ(allocator.get_allocation_count(), 1U);
    EXPECT_EQ(rts.frees, 0);
}

TEST_F(KernelArgsPublication, FreshPrepareCannotReuseAnEarlierPublicationVerdict) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    ASSERT_TRUE(helper.runtime_args_published());

    runtime.dev.worker_count = 19;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(published_worker_count(), 7);
    EXPECT_EQ(rts.allocations, 1);
    rts.copy_rc = -93;
    EXPECT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), -93);
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(published_worker_count(), 7);
}

TEST_F(KernelArgsPublication, PublishedVerdictMovesAndIsRevokedByAbandonment) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    KernelArgsHelper moved(std::move(helper));
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_TRUE(moved.runtime_args_published());
    helper.release_run_view();
    EXPECT_TRUE(moved.runtime_args_published());
    moved.abandon_after_device_failure();
    EXPECT_FALSE(moved.runtime_args_published());
    EXPECT_EQ(moved.args.runtime_args, nullptr);
    EXPECT_EQ(rts.frees, 0);
}

TEST_F(KernelArgsPublication, ReleaseAllowsFreshPrepareWithoutRetainingPublication) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    helper.release_run_view();
    EXPECT_FALSE(helper.runtime_args_published());
    runtime.dev.worker_count = 23;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_FALSE(helper.runtime_args_published());
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    EXPECT_TRUE(helper.runtime_args_published());
    EXPECT_EQ(published_worker_count(), 23);
    EXPECT_EQ(rts.allocations, 1);
}

// A publication carries the uploaded prefix into a block allocated to the full
// device extent. Where a descriptor ends in storage the device initializes, that
// range must still hold what was there before the copy: the seed the allocator
// laid down stands in for the device's own write, so it is already in place when
// the real publication runs.
TEST_F(KernelArgsPublication, PublicationLeavesTheDeviceInitializedTailUntouched) {
    // The longest publication there is: the first one onto a block, which adds
    // the handshake region. Anything past it is the tail no host copy reaches.
    const size_t image_bytes = runtime_device_initialized_prefix_size(runtime);
    const size_t extent_bytes = runtime_device_extent_size(runtime);
    // The descriptor this fixture is built against declares the gate array, so the
    // shortfall is a property of the type, asserted rather than skipped: a change
    // that widened the copy back to the extent must fail here, not opt out.
    ASSERT_EQ(extent_bytes, sizeof(DeviceRuntimeLaunchDesc));
    ASSERT_EQ(image_bytes, offsetof(DeviceRuntimeLaunchDesc, teardown_gates))
        << "no publication may reach the device-initialized gate tail";
    ASSERT_LT(image_bytes, extent_bytes);

    runtime.dev.worker_count = 5;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    // Bounds before any tail read: an allocation that shrank to the prefix is
    // reported here rather than read past below.
    ASSERT_EQ(slot.runtime_bytes, extent_bytes) << "the allocation does not cover the device-read tail";
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    ASSERT_EQ(rts.copies, 1);

    const auto *bytes = reinterpret_cast<const unsigned char *>(slot.runtime_args);
    EXPECT_EQ(published_worker_count(), 5) << "the prefix did not arrive";
    for (size_t i = image_bytes; i < extent_bytes; ++i) {
        ASSERT_EQ(bytes[i], kDeviceMark) << "the publication reached the device-initialized tail at byte " << i;
    }
}

// The slot's block outlives the run that published into it, so a second run on
// the same slot must leave that range alone too — it reuses the block rather
// than reallocating, and the device's contents there are not the host's to
// rewrite.
TEST_F(KernelArgsPublication, RepeatedPublicationPreservesTheDeviceInitializedTail) {
    const size_t image_bytes = runtime_device_initialized_prefix_size(runtime);
    const size_t extent_bytes = runtime_device_extent_size(runtime);
    ASSERT_EQ(image_bytes, offsetof(DeviceRuntimeLaunchDesc, teardown_gates));
    ASSERT_LT(image_bytes, extent_bytes);

    runtime.dev.worker_count = 5;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);
    Runtime *const destination = slot.runtime_args;

    runtime.dev.worker_count = 9;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(/*launch_route_permitted=*/false, &rts, rts.sink), 0);

    EXPECT_EQ(slot.runtime_args, destination) << "the slot reallocated instead of reusing its block";
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(rts.copies, 2);
    ASSERT_EQ(slot.runtime_bytes, extent_bytes);

    const auto *bytes = reinterpret_cast<const unsigned char *>(slot.runtime_args);
    EXPECT_EQ(published_worker_count(), 9) << "the second run's prefix did not arrive";
    for (size_t i = image_bytes; i < extent_bytes; ++i) {
        ASSERT_EQ(bytes[i], kDeviceMark) << "a repeated publication reached the tail at byte " << i;
    }
}

TEST_F(KernelArgsPublication, LaunchPublicationQueuesAnImmutableSourceOnItsExecutionStream) {
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    rts.defer = true;
    rtStream_t stream = &rts;
    ASSERT_EQ(publish_for_launch(helper, stream, rts.sink), 0);
    ASSERT_EQ(rts.async_copies, 1);
    EXPECT_EQ(rts.stream, stream);
    EXPECT_EQ(published_worker_count(), static_cast<int>(0xa5a5a5a5U));
    runtime.dev.worker_count = 19;
    helper.release_run_view();
    ASSERT_EQ(rts.host_frees, 0);
    ASSERT_EQ(rts.pending_source, slot.runtime_upload_source);
    std::memcpy(rts.pending_target, rts.pending_source, rts.pending_bytes);
    EXPECT_EQ(published_worker_count(), 7);
}

TEST_F(KernelArgsPublication, PinnedSourceIsReusedAndItsReleaseFailureRemainsRetryable) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    void *source = slot.runtime_upload_source;
    ASSERT_NE(source, nullptr);
    ASSERT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
    ASSERT_TRUE(rts.sink.submitted());
    helper.release_run_view();
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_EQ(slot.runtime_upload_source, source);
    EXPECT_EQ(rts.host_allocations, 1);
    rts.host_free_rc = -79;
    EXPECT_EQ(release_slot_persistent_args(slot, allocator), -79);
    EXPECT_EQ(slot.runtime_upload_source, source);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_EQ(rts.host_frees, 0);
    rts.host_free_rc = 0;
    EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
    EXPECT_EQ(rts.host_frees, 1);
    EXPECT_EQ(slot.runtime_upload_source, nullptr);
}

TEST_F(KernelArgsPublication, PinnedAllocationFailurePrecedesAnyPublicationAndCanRetry) {
    rts.host_alloc_rc = -78;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), -78);
    EXPECT_EQ(rts.async_copies, 0);
    EXPECT_FALSE(rts.sink.submitted());
    EXPECT_FALSE(helper.launchable());
    EXPECT_EQ(slot.runtime_upload_source, nullptr);
    rts.host_alloc_rc = 0;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
}

TEST_F(KernelArgsPublication, FailedAsyncEnqueueCannotCertifyNoSubmission) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    rts.copy_rc = -81;
    EXPECT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), -81);
    EXPECT_TRUE(rts.sink.submitted());
    EXPECT_FALSE(helper.runtime_args_published());
    EXPECT_EQ(helper.launch_payload(), nullptr);
    EXPECT_NE(slot.runtime_upload_source, nullptr);
    EXPECT_EQ(rts.host_frees, 0);
}

TEST_F(KernelArgsPublication, AbandonmentNeverFreesAPotentialDmaSource) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    rts.defer = true;
    ASSERT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
    void *source = slot.runtime_upload_source;
    void *destination = slot.runtime_args;
    abandon_slot_persistent_args(slot);
    ASSERT_EQ(rts.host_frees, 0);
    ASSERT_EQ(rts.frees, 0);
    EXPECT_EQ(slot.runtime_upload_source, nullptr);
    // Finish the modeled reader before reclaiming the test's orphaned blocks.
    std::memcpy(rts.pending_target, rts.pending_source, rts.pending_bytes);
    std::free(source);
    EXPECT_EQ(allocator.free(destination), 0);
}

TEST_F(KernelArgsPublication, MetadataValuesArePublishedBeforeDescriptorFromSlotOwnedSources) {
    uint64_t first = 0, second = 0;
    {
        uint64_t values[] = {17, 29};
        const HostMetadataRegion regions[] = {
            {&first, &values[0], sizeof(first)}, {&second, &values[1], sizeof(second)}
        };
        ASSERT_EQ(slot.staged_launch_metadata.prepare(regions, 2), 0);
        values[0] = 900;
    }
    runtime.dev.worker_count = 7;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(first, 0U);
    EXPECT_EQ(second, 0U);
    EXPECT_EQ(slot.runtime_upload_bytes, slot.runtime_bytes + sizeof(first) + sizeof(second));
    KernelArgsHelper moved(std::move(helper));
    helper.release_run_view();
    rts.defer = true;
    ASSERT_EQ(moved.publish_runtime_args(false, &rts, rts.sink), 0);
    moved.release_run_view();
    ASSERT_EQ(rts.queued.size(), 3U);
    EXPECT_EQ(first, 0U);
    EXPECT_EQ(second, 0U);
    EXPECT_EQ(rts.queued[0].target, &first);
    EXPECT_EQ(rts.queued[1].target, &second);
    EXPECT_EQ(rts.queued[2].target, slot.runtime_args);
    auto *pinned = static_cast<std::byte *>(slot.runtime_upload_source);
    ASSERT_EQ(rts.queued[0].source, pinned + slot.runtime_bytes);
    ASSERT_EQ(rts.queued[1].source, pinned + slot.runtime_bytes + sizeof(first));
    ASSERT_EQ(rts.queued[2].source, pinned);
    ASSERT_EQ(rts.host_frees, 0);
    for (const auto &copy : rts.queued) {
        EXPECT_EQ(copy.stream, &rts);
        std::memcpy(copy.target, copy.source, copy.bytes);
    }
    EXPECT_EQ(first, 17U);
    EXPECT_EQ(second, 29U);
    EXPECT_EQ(published_worker_count(), 7);
}

TEST_F(KernelArgsPublication, EveryMetadataEnqueueFailureStopsTheDescriptorAndRetainsTheSource) {
    for (int failure = 1; failure <= 3; ++failure) {
        SCOPED_TRACE(failure);
        uint64_t first = 0, second = 0;
        const uint64_t value = 37;
        const HostMetadataRegion regions[] = {{&first, &value, sizeof(value)}, {&second, &value, sizeof(value)}};
        ASSERT_EQ(slot.staged_launch_metadata.prepare(regions, 2), 0);
        ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
        rts.copies = 0;
        rts.queued.clear();
        rts.sink = {};
        rts.copy_rc = -81;
        rts.fail_copy_on = failure;
        EXPECT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), -81);
        EXPECT_EQ(rts.copies, failure);
        EXPECT_EQ(rts.queued.size(), static_cast<size_t>(failure));
        EXPECT_TRUE(rts.sink.submitted());
        EXPECT_FALSE(helper.launchable());
        EXPECT_FALSE(helper.runtime_args_published()) << "scheduler DFX cannot read an unpublished batch";
        EXPECT_EQ(helper.launch_payload(), nullptr);
        EXPECT_EQ(first, failure > 1 ? value : 0U);
        EXPECT_EQ(second, failure > 2 ? value : 0U);
        EXPECT_EQ(published_worker_count(), static_cast<int>(0xa5a5a5a5U));
        EXPECT_EQ(rts.host_frees, 0);
        EXPECT_NE(slot.runtime_upload_source, nullptr);
        EXPECT_NE(helper.publish_runtime_args(false, &rts, rts.sink), 0);
        EXPECT_EQ(rts.copies, failure);
    }
    rts.copy_rc = 0;
    // A fresh legacy run has no metadata. It cannot replay the failed batch.
    rts.queued.clear();
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
    ASSERT_EQ(rts.queued.size(), 1U);
    EXPECT_EQ(rts.queued.front().target, slot.runtime_args);
}

TEST_F(KernelArgsPublication, MetadataSourceGrowthFailuresAreRetryableWithoutDeviceSubmission) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
    Runtime *destination = slot.runtime_args;
    void *old_source = slot.runtime_upload_source;
    uint64_t value = 73, target = 0;
    const HostMetadataRegion region{&target, &value, sizeof(value)};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&region, 1), 0);
    rts.queued.clear();
    rts.sink = {};
    rts.host_free_rc = -71;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), -71);
    EXPECT_EQ(slot.runtime_upload_source, old_source);
    EXPECT_FALSE(helper.launchable());
    EXPECT_FALSE(rts.sink.submitted());
    rts.host_free_rc = 0;
    rts.host_alloc_rc = -72;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), -72);
    EXPECT_EQ(slot.runtime_upload_source, nullptr);
    EXPECT_EQ(slot.runtime_upload_bytes, 0U);
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_TRUE(rts.queued.empty());
    rts.host_alloc_rc = 0;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
    EXPECT_EQ(target, value);
    EXPECT_EQ(slot.runtime_args, destination);
    EXPECT_EQ(rts.queued.size(), 2U);
}

TEST_F(KernelArgsPublication, WholeMetadataBatchIsValidatedBeforeAnySourceIsRead) {
    uint64_t target = 0;
    const HostMetadataRegion regions[] = {
        {&target, reinterpret_cast<void *>(1), sizeof(target)}, {nullptr, &target, sizeof(target)}
    };
    EXPECT_EQ(slot.staged_launch_metadata.prepare(regions, 2), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    const HostMetadataRegion overflowing[] = {{&target, reinterpret_cast<void *>(1), SIZE_MAX}, {&target, &target, 1}};
    EXPECT_EQ(slot.staged_launch_metadata.prepare(overflowing, 2), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    const uint64_t value = 47;
    const HostMetadataRegion valid{&target, &value, sizeof(value)};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&valid, 1), 0);
    EXPECT_EQ(slot.staged_launch_metadata.prepare(regions, 2), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
    EXPECT_EQ(target, value);
}

TEST_F(KernelArgsPublication, EmptyStagingClearsAnAbandonedUnpublishedBatch) {
    uint64_t target = 0, value = 53;
    const HostMetadataRegion region{&target, &value, sizeof(value)};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&region, 1), 0);
    ASSERT_EQ(slot.staged_launch_metadata.prepare(nullptr, 0), 0);
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
    EXPECT_EQ(target, 0U);
    ASSERT_EQ(rts.queued.size(), 1U);
    EXPECT_EQ(rts.queued.front().target, slot.runtime_args);
}

TEST_F(KernelArgsPublication, AppendedResetPreservesEarlierMetadataAndOwnsItsSnapshot) {
    uint64_t target = 0, reset_target = 0;
    const uint64_t value = 67;
    const HostMetadataRegion first{&target, &value, sizeof(value)};
    ASSERT_EQ(slot.staged_launch_metadata.prepare(&first, 1), 0);
    {
        uint64_t reset = 91;
        ASSERT_EQ(slot.staged_launch_metadata.append({&reset_target, &reset, sizeof(reset)}), 0);
        reset = 0;
    }
    // A rejected addition cannot consume or replace the previously staged batch.
    EXPECT_EQ(slot.staged_launch_metadata.append({nullptr, &value, 8}), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(
        slot.staged_launch_metadata.append({&target, reinterpret_cast<void *>(1), SIZE_MAX}),
        PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot), 0);
    ASSERT_EQ(helper.publish_runtime_args(false, &rts, rts.sink), 0);
    EXPECT_EQ(target, value);
    EXPECT_EQ(reset_target, 91U);
    ASSERT_EQ(rts.queued.size(), 3U);
    EXPECT_EQ(rts.queued[0].target, &target);
    EXPECT_EQ(rts.queued[1].target, &reset_target);
    EXPECT_EQ(rts.queued[2].target, slot.runtime_args);
}
