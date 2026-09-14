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
#include "host/kernel_callable_cache.h"

namespace {
std::vector<uint8_t> image(size_t payload = 64, uint8_t value = 1) {
    std::vector<uint8_t> result(sizeof(ChipCallable) + payload);
    auto *chip = reinterpret_cast<ChipCallable *>(result.data());
    chip->binary_size_ = payload;
    std::fill(result.begin() + sizeof(ChipCallable), result.end(), value);
    return result;
}
struct FakeDevice {
    int allocations{0};
    int copies{0};
    int copy_error{0};
    bool fail_alloc{false};
    std::vector<uint8_t> last_upload;
    std::vector<size_t> allocation_sizes;
    uintptr_t next_address{0x10000000};
    KernelCallableCache::Ops ops() {
        return {
            this,
            [](void *p, size_t bytes) -> void * {
                auto &self = *static_cast<FakeDevice *>(p);
                ++self.allocations;
                if (self.fail_alloc) return nullptr;
                self.allocation_sizes.push_back(bytes);
                auto address = self.next_address;
                self.next_address += bytes;
                return reinterpret_cast<void *>(address);
            },
            [](void *p, void *, const void *src, size_t bytes) -> int {
                auto &self = *static_cast<FakeDevice *>(p);
                ++self.copies;
                self.last_upload.assign(static_cast<const uint8_t *>(src), static_cast<const uint8_t *>(src) + bytes);
                return self.copy_error;
            }
        };
    }
};
int prepare(KernelCallableCache &cache, FakeDevice &device, int id, const std::vector<uint8_t> &blob) {
    int32_t actual_id = 99;
    const int rc =
        cache.stage(reinterpret_cast<const ChipCallable *>(blob.data()), blob.size(), device.ops(), actual_id);
    EXPECT_EQ(actual_id, rc == 0 ? id : -1);
    return rc;
}
size_t charge(const std::vector<uint8_t> &blob) { return (blob.size() + 63) & ~size_t(63); }

TEST(KernelCallableCache, Supports8192ResidentsWithoutMovingPublishedAddresses) {
    KernelCallableCache cache;
    FakeDevice device;
    auto blob = image();
    for (int id = 0; id < 8192; ++id) {
        ASSERT_EQ(prepare(cache, device, id, blob), 0);
        cache.commit(id);
    }
    EXPECT_EQ(cache.resident_count(), 8192);
    EXPECT_EQ(device.copies, 8192);
    EXPECT_EQ(prepare(cache, device, -1, blob), PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED);
    blob.back() = 2;
    EXPECT_EQ(prepare(cache, device, -1, blob), PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED);
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve(0, found), 0);
    EXPECT_EQ(found.device_address, 0x10000000);
    ASSERT_EQ(cache.resolve(8191, found), 0);
    EXPECT_GT(found.device_address, 0x10000000);
}

TEST(KernelCallableCache, SmallImagesShareTwoMiBBlocksAndGrowthPreservesAddresses) {
    constexpr size_t block = 2 * 1024 * 1024;
    KernelCallableCache cache;
    FakeDevice device;
    auto blob = image(block / 2 - sizeof(ChipCallable));
    ASSERT_EQ(prepare(cache, device, 0, blob), 0);
    cache.commit(0);
    ASSERT_EQ(prepare(cache, device, 1, blob), 0);
    cache.commit(1);
    EXPECT_EQ(device.allocation_sizes, std::vector<size_t>({block}));
    ASSERT_EQ(prepare(cache, device, 2, blob), 0);
    cache.commit(2);
    EXPECT_EQ(device.allocation_sizes, std::vector<size_t>({block, block}));
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve(0, found), 0);
    EXPECT_EQ(found.device_address, 0x10000000);
    ASSERT_EQ(cache.resolve(2, found), 0);
    EXPECT_EQ(found.device_address, 0x10000000 + block);
}

TEST(KernelCallableCache, LargeImageUsesExactAlignedAllocationAndKeepsSmallBlockTail) {
    constexpr size_t block = 2 * 1024 * 1024;
    KernelCallableCache cache;
    FakeDevice device;
    const auto small = image();
    const auto large = image(block + 1);
    ASSERT_EQ(prepare(cache, device, 0, small), 0);
    cache.commit(0);
    ASSERT_EQ(prepare(cache, device, 1, large), 0);
    cache.commit(1);
    ASSERT_EQ(prepare(cache, device, 2, small), 0);
    cache.commit(2);
    EXPECT_EQ(device.allocation_sizes, std::vector<size_t>({block, charge(large)}));
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve(2, found), 0);
    EXPECT_EQ(found.device_address, 0x10000000 + charge(small));
}

TEST(KernelCallableCache, BlockSlackCountsTowardCapacityAndExistingTailRemainsUsable) {
    constexpr size_t block = 2 * 1024 * 1024;
    KernelCallableCache cache(2 * block);
    FakeDevice device;
    auto large = image(3 * block / 4 - sizeof(ChipCallable));
    for (int id = 0; id < 2; ++id) {
        ASSERT_EQ(prepare(cache, device, id, large), 0);
        cache.commit(id);
    }
    EXPECT_EQ(prepare(cache, device, 2, large), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    EXPECT_EQ(device.allocation_sizes, std::vector<size_t>({block, block}));
    auto small = image(block / 4 - sizeof(ChipCallable));
    ASSERT_EQ(prepare(cache, device, 2, small), 0);
    cache.commit(2);
    ASSERT_EQ(prepare(cache, device, 3, small), 0);
    cache.commit(3);
    EXPECT_EQ(prepare(cache, device, 4, small), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    EXPECT_EQ(device.allocations, 2);
}

TEST(KernelCallableCache, EveryRegistrationUploadsANewImageAndReturnsANewId) {
    KernelCallableCache cache;
    FakeDevice device;
    auto blob = image();
    ASSERT_EQ(prepare(cache, device, 0, blob), 0);
    const auto first_address = cache.pending_uploaded_address();
    cache.commit(0);
    EXPECT_EQ(cache.pending_uploaded_address(), 0);
    ASSERT_EQ(prepare(cache, device, 1, blob), 0);
    const auto second_address = cache.pending_uploaded_address();
    EXPECT_NE(first_address, second_address);
    cache.commit(1);
    blob.back() = 2;
    ASSERT_EQ(prepare(cache, device, 2, blob), 0);
    EXPECT_NE(cache.pending_uploaded_address(), second_address);
    cache.commit(2);
    EXPECT_EQ(device.copies, 3);
    EXPECT_EQ(device.allocations, 1);
    EXPECT_EQ(cache.resident_count(), 3);
    EXPECT_EQ(cache.resident_bytes(), 3 * charge(blob));
    EXPECT_EQ(cache.host_bytes(), 3 * blob.size());
    KernelCallableResidency first;
    KernelCallableResidency second;
    ASSERT_EQ(cache.resolve(0, first), 0);
    ASSERT_EQ(cache.resolve(1, second), 0);
    EXPECT_EQ(first.device_address, first_address);
    EXPECT_EQ(second.device_address, second_address);
}

TEST(KernelCallableCache, ExactByteBoundaryAndOneByteOver) {
    auto blob = image();
    blob = image(blob.size() + 63 - sizeof(ChipCallable) - ((blob.size() + 63) % 64));
    ASSERT_EQ(blob.size() % 64, 0);
    KernelCallableCache cache(blob.size());
    FakeDevice device;
    ASSERT_EQ(prepare(cache, device, 0, blob), 0);
    cache.commit(0);
    EXPECT_EQ(cache.resident_bytes(), blob.size());
    auto other = image(1, 2);
    EXPECT_EQ(prepare(cache, device, 1, other), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    KernelCallableCache oversized(blob.size());
    auto too_big = image(blob.size() - sizeof(ChipCallable) + 1);
    EXPECT_EQ(prepare(oversized, device, 0, too_big), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    EXPECT_EQ(device.copies, 1);
    KernelCallableResidency found;
    EXPECT_EQ(cache.resolve(0, found), 0);
}

TEST(KernelCallableCache, AlignmentPaddingConsumesBudgetAndIdenticalRegistrationStillFails) {
    auto blob = image(1);
    ASSERT_NE(blob.size(), charge(blob));
    KernelCallableCache too_small(blob.size());
    FakeDevice device;
    EXPECT_EQ(prepare(too_small, device, 0, blob), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    EXPECT_EQ(device.allocations, 0);
    KernelCallableCache cache(charge(blob));
    ASSERT_EQ(prepare(cache, device, 0, blob), 0);
    cache.commit(0);
    EXPECT_EQ(prepare(cache, device, -1, blob), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    EXPECT_EQ(cache.resident_bytes(), charge(blob));
    EXPECT_EQ(cache.host_bytes(), blob.size());
    EXPECT_EQ(device.copies, 1);
}

TEST(KernelCallableCache, PendingEntryIsNotLaunchableAndRollbackPreservesResidents) {
    KernelCallableCache cache;
    FakeDevice device;
    auto first = image();
    auto second = image(65, 2);
    ASSERT_EQ(prepare(cache, device, 0, first), 0);
    cache.commit(0);
    ASSERT_EQ(prepare(cache, device, 1, second), 0);
    KernelCallableResidency found;
    EXPECT_EQ(cache.resolve(1, found), PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT);
    EXPECT_EQ(prepare(cache, device, 2, image(65, 3)), PTO_RUNTIME_ERR_INVALID_STATE);
    cache.rollback(1);
    EXPECT_EQ(cache.resident_bytes(), charge(first));
    EXPECT_EQ(cache.resolve(0, found), 0);
    ASSERT_EQ(prepare(cache, device, 1, second), 0);
    cache.commit(1);
    EXPECT_EQ(cache.resolve(1, found), 0);
    EXPECT_EQ(found.device_address, 0x10000000 + charge(first));
}

TEST(KernelCallableCache, AllocationAndCopyFailuresAreRetryable) {
    KernelCallableCache cache;
    FakeDevice device;
    auto blob = image();
    device.fail_alloc = true;
    EXPECT_EQ(prepare(cache, device, 0, blob), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(cache.resident_bytes(), 0);
    device.fail_alloc = false;
    device.copy_error = -123;
    EXPECT_EQ(prepare(cache, device, 0, blob), -123);
    EXPECT_EQ(cache.resident_bytes(), 0);
    device.copy_error = 0;
    ASSERT_EQ(prepare(cache, device, 0, blob), 0);
    cache.commit(0);
    EXPECT_EQ(cache.resident_count(), 1);
    EXPECT_EQ(device.allocations, 2);
}

TEST(KernelCallableCache, RejectsMalformedSpansBeforeHashOrDeviceOperations) {
    KernelCallableCache cache;
    FakeDevice device;
    auto blob = image();
    auto *chip = reinterpret_cast<ChipCallable *>(blob.data());
    chip->binary_size_ = UINT32_MAX;
    EXPECT_EQ(prepare(cache, device, 0, blob), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    chip->binary_size_ = 64;
    chip->child_count_ = 1025;
    EXPECT_EQ(prepare(cache, device, 0, blob), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    chip->child_count_ = 1;
    chip->child_offsets_[0] = UINT32_MAX;
    EXPECT_EQ(prepare(cache, device, 0, blob), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    chip->child_offsets_[0] = 64;
    EXPECT_EQ(prepare(cache, device, 0, blob), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    chip->child_count_ = 0;
    chip->func_name_len_ = CALLABLE_FUNC_NAME_MAX;
    EXPECT_EQ(prepare(cache, device, 0, blob), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(device.allocations, 0);
    EXPECT_EQ(device.copies, 0);
}

TEST(KernelCallableCache, HostBackingIsImmutableAndResolveDoesNotAllocateOrUpload) {
    KernelCallableCache cache;
    FakeDevice device;
    auto blob = image();
    ASSERT_EQ(prepare(cache, device, 0, blob), 0);
    cache.commit(0);
    blob.back() = 2;
    auto original = image();
    ASSERT_EQ(prepare(cache, device, 1, original), 0);
    cache.commit(1);
    KernelCallableResidency found;
    for (int i = 0; i < 100; ++i)
        ASSERT_EQ(cache.resolve(0, found), 0);
    EXPECT_EQ(device.allocations, 1);
    EXPECT_EQ(device.copies, 2);
    cache.clear();
    EXPECT_EQ(cache.resolve(0, found), PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT);
    EXPECT_EQ(cache.host_bytes(), 0);
}
TEST(KernelCallableCache, ChildAddressesArePatchedOnlyInDeviceScratch) {
    const uint8_t code[] = {1, 2, 3};
    auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, code, sizeof(code));
    const int32_t func_id = 5;
    auto blob = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch", code, sizeof(code), &func_id, &child, 1, ""
    );
    const auto original = blob;
    KernelCallableCache cache;
    FakeDevice device;
    ASSERT_EQ(prepare(cache, device, 0, blob), 0);
    cache.commit(0);
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve(0, found), 0);
    const auto *uploaded = reinterpret_cast<const ChipCallable *>(device.last_upload.data());
    EXPECT_EQ(
        uploaded->child(0).resolved_addr(), found.device_address + offsetof(ChipCallable, storage_) +
                                                uploaded->child_offset(0) + CoreCallable::binary_data_offset()
    );
    EXPECT_EQ(blob, original);
    auto *input = reinterpret_cast<ChipCallable *>(blob.data());
    input->child_count_ = 2;
    input->child_offsets_[1] = input->child_offsets_[0];
    EXPECT_EQ(prepare(cache, device, 1, blob), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

TEST(KernelCallableCache, InvalidAndMissingIdsDoNotMutateResidents) {
    KernelCallableCache cache;
    FakeDevice device;
    ASSERT_EQ(prepare(cache, device, 0, image()), 0);
    cache.commit(0);
    KernelCallableResidency found;
    EXPECT_EQ(cache.resolve(-1, found), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(cache.resolve(8192, found), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(cache.resolve(1, found), PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT);
    EXPECT_EQ(found.device_address, 0);
    EXPECT_EQ(cache.resolve(0, found), 0);
    EXPECT_EQ(found.device_address, 0x10000000);
    EXPECT_EQ(device.copies, 1);
    EXPECT_EQ(cache.resident_count(), 1);
}
TEST(KernelCallableCache, FailedGrowthAndRollbackPreserveBlockBudgetAndResidents) {
    constexpr size_t block = 2 * 1024 * 1024;
    EXPECT_EQ(KernelCallableCache::kByteLimit, 2ULL * 1024 * 1024 * 1024);
    KernelCallableCache cache(2 * block);
    FakeDevice device;
    auto blob = image(block - sizeof(ChipCallable));
    ASSERT_EQ(prepare(cache, device, 0, blob), 0);
    cache.commit(0);
    device.fail_alloc = true;
    EXPECT_EQ(prepare(cache, device, 1, blob), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(cache.allocated_bytes(), block);
    device.fail_alloc = false;
    device.copy_error = -123;
    EXPECT_EQ(prepare(cache, device, 1, blob), -123);
    EXPECT_EQ(cache.allocated_bytes(), 2 * block);
    EXPECT_EQ(cache.resident_bytes(), block);
    device.copy_error = 0;
    ASSERT_EQ(prepare(cache, device, 1, blob), 0);
    const auto address = cache.pending_uploaded_address();
    cache.rollback(1);
    ASSERT_EQ(prepare(cache, device, 1, blob), 0);
    EXPECT_EQ(cache.pending_uploaded_address(), address);
    cache.commit(1);
    EXPECT_EQ(device.allocations, 3);
    EXPECT_EQ(cache.resident_bytes(), 2 * block);
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve(0, found), 0);
    EXPECT_EQ(found.device_address, 0x10000000);
    EXPECT_EQ(prepare(cache, device, 2, blob), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
}
}  // namespace
