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
// answer is per device allocation, not per slot: Program initializes handshake
// storage in its first publication; replayable preparation initializes it
// before recording and excludes it from every recorded publication.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "acl/acl.h"
#include "device_runner_helpers.h"

namespace {
// Byte every fresh device block is seeded with, standing in for what the device
// left there. A range still holding it was not copied into.
constexpr unsigned char kDeviceMark = 0xa5;

struct RtsState {
    int alloc_rc = 0;
    int free_rc = 0;
    int free_host_rc = 0;
    int copy_rc = 0;
    int copies = 0;
    int initialization_rc = 0;
    int initializations = 0;
    int exchange_calls = 0;
    int fail_exchange = 0;
    bool recording = false;
    aclmdlRICaptureMode mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
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
    if (rts.free_rc != 0) return rts.free_rc;
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMallocHost(void **ptr, uint64_t bytes, uint16_t) {
    *ptr = std::malloc(bytes);
    return *ptr == nullptr ? -1 : 0;
}
extern "C" rtError_t rtFreeHost(void *ptr) {
    if (rts.free_host_rc != 0) return rts.free_host_rc;
    std::free(ptr);
    return 0;
}
extern "C" rtError_t rtMemcpy(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind) {
    ++rts.initializations;
    EXPECT_EQ(kind, RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(capacity, bytes);
    if (rts.recording && rts.mode != ACL_MODEL_RI_CAPTURE_MODE_RELAXED) return -83;
    if (rts.initialization_rc != 0) return rts.initialization_rc;
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t
rtMemcpyAsync(void *dst, uint64_t capacity, const void *src, uint64_t bytes, rtMemcpyKind_t kind, rtStream_t) {
    ++rts.copies;
    rts.last_copy_bytes = bytes;
    EXPECT_EQ(kind, RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(capacity, bytes);
    if (rts.copy_rc != 0) return rts.copy_rc;
    std::memcpy(dst, src, bytes);
    return 0;
}
extern "C" rtError_t rtStreamQuery(rtStream_t) { return 0; }
extern "C" const char *aclGetRecentErrMsg() { return nullptr; }
// These cases drive the publication step with the launch route refused, so the
// permit never consults a live stream; the query only has to link.
extern "C" aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode) {
    if (++rts.exchange_calls == rts.fail_exchange) return -84;
    std::swap(rts.mode, *mode);
    return 0;
}
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
    void *abandoned_source = slot.runtime_upload_source;
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
    EXPECT_EQ(rts.initializations, 0);

    helper.release_run_view();
    EXPECT_FALSE(slot.workers_initialized);

    ASSERT_EQ(publish_once(), 0);
    EXPECT_EQ(rts.last_copy_bytes, initializing_bytes());
    EXPECT_TRUE(slot.workers_initialized);
}

TEST_F(WorkersPublication, ReplayablePublicationCannotOverwriteDeviceReports) {
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot, true), 0);
    EXPECT_TRUE(slot.workers_initialized);
    EXPECT_EQ(rts.initializations, 1);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_TRUE(device_workers_all(0u));
    const auto *device = reinterpret_cast<const unsigned char *>(slot.runtime_args);
    for (size_t i = 0; i < steady_bytes(); ++i)
        ASSERT_EQ(device[i], kDeviceMark);
    for (size_t i = initializing_bytes(); i < slot.runtime_bytes; ++i)
        ASSERT_EQ(device[i], kDeviceMark);

    ASSERT_EQ(helper.publish_runtime_args(false, this, rts.sink), 0);
    EXPECT_EQ(rts.last_copy_bytes, steady_bytes());
    std::memset(const_cast<unsigned char *>(device_workers()), 0x5a, sizeof(DeviceRuntimeLaunchDesc::workers));
    // Replay the captured DMA after the device has published a later report.
    std::memcpy(slot.runtime_args, slot.runtime_upload_source, rts.last_copy_bytes);
    EXPECT_TRUE(device_workers_all(0x5a));
    helper.release_run_view();
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot, true), 0);
    EXPECT_EQ(rts.initializations, 1);
    EXPECT_TRUE(device_workers_all(0x5a));
}

TEST_F(WorkersPublication, FailedReplayInitializationIsRetriedOnTheRetainedAllocation) {
    rts.initialization_rc = -82;
    EXPECT_EQ(helper.prepare_runtime_args(runtime, allocator, slot, true), -82);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_EQ(rts.initializations, 1);
    auto *allocation = slot.runtime_args;
    ASSERT_NE(allocation, nullptr);
    rts.initialization_rc = 0;
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot, true), 0);
    EXPECT_EQ(slot.runtime_args, allocation);
    EXPECT_EQ(rts.allocations, 1);
    EXPECT_EQ(rts.initializations, 2);
    EXPECT_TRUE(device_workers_all(0u));
}

TEST_F(WorkersPublication, FreshCaptureBackingRestoresTheCallerModeBeforePublication) {
    rts.recording = true;
    rts.mode = ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL;
    ASSERT_EQ(reserve_slot_runtime_args(runtime, allocator, slot, 0, RuntimeWorkerInit::DuringRecording), 0);
    EXPECT_EQ(rts.mode, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    EXPECT_EQ(rts.exchange_calls, 2);
    EXPECT_EQ(rts.initializations, 1);
    EXPECT_EQ(rts.copies, 0);
    EXPECT_TRUE(device_workers_all(0u));
    const auto *device = reinterpret_cast<const unsigned char *>(slot.runtime_args);
    const size_t begin = offsetof(DeviceRuntimeLaunchDesc, workers);
    const size_t end = begin + sizeof(DeviceRuntimeLaunchDesc::workers);
    EXPECT_TRUE(std::all_of(device, device + begin, [](unsigned char byte) {
        return byte == kDeviceMark;
    }));
    EXPECT_TRUE(std::all_of(device + end, device + slot.runtime_bytes, [](unsigned char byte) {
        return byte == kDeviceMark;
    }));
    ASSERT_EQ(helper.prepare_runtime_args(runtime, allocator, slot, true), 0);
    ASSERT_EQ(helper.publish_runtime_args(false, this, rts.sink, true), 0);
    EXPECT_EQ(rts.last_copy_bytes, steady_bytes());
    EXPECT_EQ(rts.exchange_calls, 2);
}

TEST_F(WorkersPublication, CaptureInitializationDoesNotCommitWhenExchangeOrCopyFails) {
    rts.recording = true;
    rts.fail_exchange = 1;
    EXPECT_EQ(reserve_slot_runtime_args(runtime, allocator, slot, 0, RuntimeWorkerInit::DuringRecording), -84);
    auto *const allocation = slot.runtime_args;
    ASSERT_NE(allocation, nullptr);
    EXPECT_EQ(rts.initializations, 0);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(rts.mode, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);

    rts.initialization_rc = -82;
    EXPECT_EQ(reserve_slot_runtime_args(runtime, allocator, slot, 0, RuntimeWorkerInit::DuringRecording), -82);
    EXPECT_EQ(rts.mode, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(rts.exchange_calls, 3);

    rts.initialization_rc = 0;
    rts.fail_exchange = 5;
    int restore_error = 0;
    EXPECT_EQ(
        reserve_slot_runtime_args(runtime, allocator, slot, 0, RuntimeWorkerInit::DuringRecording, &restore_error), -84
    );
    EXPECT_EQ(restore_error, -84);
    EXPECT_FALSE(slot.workers_initialized);
    EXPECT_EQ(slot.runtime_args, allocation);
    EXPECT_EQ(rts.copies, 0);
}

TEST_F(WorkersPublication, FailedDeviceFreeRetainsTrimHistoryUntilAllocationEnds) {
    ASSERT_EQ(publish_once(), 0);
    slot.entry_cache_qualification.published(1);
    slot.entry_cache_qualification.observe_success(1);
    slot.entry_cmo_may_have_trimmed = true;
    auto *allocation = slot.runtime_args;
    rts.free_rc = -71;
    EXPECT_EQ(release_slot_persistent_args(slot, allocator), -71);
    EXPECT_EQ(slot.runtime_args, allocation);
    EXPECT_FALSE(slot.entry_cache_qualification.eligible());
    EXPECT_TRUE(slot.entry_cmo_may_have_trimmed);
    rts.free_rc = 0;
    EXPECT_EQ(release_slot_persistent_args(slot, allocator), 0);
    EXPECT_FALSE(slot.entry_cmo_may_have_trimmed);
}

TEST_F(WorkersPublication, DeviceFreeClearsHistoryEvenWhenPinnedFreeFails) {
    ASSERT_EQ(publish_once(), 0);
    helper.release_run_view();
    slot.entry_cache_qualification.published(1);
    slot.entry_cache_qualification.observe_success(1);
    slot.entry_cmo_may_have_trimmed = true;
    rts.free_host_rc = -72;
    EXPECT_EQ(release_slot_persistent_args(slot, allocator), -72);
    EXPECT_EQ(slot.runtime_args, nullptr);
    EXPECT_FALSE(slot.entry_cache_qualification.eligible());
    EXPECT_FALSE(slot.entry_cmo_may_have_trimmed);
    rts.free_host_rc = 0;
    ASSERT_EQ(publish_once(), 0);
    EXPECT_FALSE(slot.entry_cache_qualification.eligible());
    EXPECT_FALSE(slot.entry_cmo_may_have_trimmed);
}
