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
    int descriptors{0};
    KernelCallableDeviceResidency descriptor{};
    int copy_error{0};
    bool fail_alloc{false};
    std::vector<uint8_t> last_upload;
    KernelCallableCache::Ops ops() {
        return {
            this,
            [](void *p, size_t) -> void * {
                auto &self = *static_cast<FakeDevice *>(p);
                ++self.allocations;
                return self.fail_alloc ? nullptr : reinterpret_cast<void *>(0x10000000);
            },
            [](void *p, void *, const void *src, size_t bytes) -> int {
                auto &self = *static_cast<FakeDevice *>(p);
                if (bytes == sizeof(KernelCallableDeviceResidency)) {
                    ++self.descriptors;
                    std::memcpy(&self.descriptor, src, bytes);
                    return self.copy_error;
                }
                ++self.copies;
                self.last_upload.assign(static_cast<const uint8_t *>(src), static_cast<const uint8_t *>(src) + bytes);
                return self.copy_error;
            }
        };
    }
};
int prepare(KernelCallableCache &cache, FakeDevice &device, int id, const std::vector<uint8_t> &blob, bool &hit) {
    SimplerCallableHandle actual_id{99, 99};
    const int rc =
        cache.stage(reinterpret_cast<const ChipCallable *>(blob.data()), blob.size(), device.ops(), actual_id, hit);
    EXPECT_EQ(actual_id.callable_id, rc == 0 ? id : -1);
    if (rc == 0) EXPECT_NE(actual_id.generation, 0);
    else EXPECT_EQ(actual_id.generation, 0);
    return rc;
}
size_t charge(const std::vector<uint8_t> &blob) { return (blob.size() + 63) & ~size_t(63); }

TEST(KernelCallableCache, SixtyFourResidentsThenCountErrorWithoutMutation) {
    KernelCallableCache cache;
    cache.set_generation(7);
    FakeDevice device;
    bool hit;
    for (int id = 0; id < 64; ++id) {
        ASSERT_EQ(prepare(cache, device, id, image(64, id), hit), 0);
        EXPECT_FALSE(hit);
        cache.commit(id);
    }
    EXPECT_EQ(cache.resident_count(), 64);
    EXPECT_EQ(device.allocations, 1);
    EXPECT_EQ(device.copies, 64);
    EXPECT_EQ(cache.resident_bytes(), 64 * charge(image()));
    EXPECT_EQ(cache.host_bytes(), 64 * image().size());
    EXPECT_EQ(prepare(cache, device, -1, image(64, 255), hit), PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED);
    ASSERT_EQ(prepare(cache, device, 0, image(64, 0), hit), 0);
    EXPECT_TRUE(hit);
    EXPECT_EQ(device.copies, 64);
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve({63, 7}, found), 0);
    EXPECT_EQ(found.generation, 7);
    EXPECT_EQ(found.device_address, 0x10000000 + KernelCallableCache::kDescriptorBytes + 63 * charge(image()));
    EXPECT_EQ(cache.resident_count(), 64);
}

TEST(KernelCallableCache, IdenticalContentReusesIdAndDifferentContentGetsNewId) {
    KernelCallableCache cache;
    cache.set_generation(1);
    FakeDevice device;
    auto blob = image();
    bool hit;
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    cache.commit(0);
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    EXPECT_TRUE(hit);
    blob.back() = 2;
    ASSERT_EQ(prepare(cache, device, 1, blob, hit), 0);
    EXPECT_FALSE(hit);
    cache.commit(1);
    EXPECT_EQ(device.copies, 2);
    EXPECT_EQ(device.allocations, 1);
    EXPECT_EQ(cache.resident_count(), 2);
}

TEST(KernelCallableCache, ExactByteBoundaryAndOneByteOver) {
    auto blob = image();
    blob = image(blob.size() + 63 - sizeof(ChipCallable) - ((blob.size() + 63) % 64));
    ASSERT_EQ(blob.size() % 64, 0);
    KernelCallableCache cache(blob.size());
    cache.set_generation(1);
    FakeDevice device;
    bool hit;
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    cache.commit(0);
    EXPECT_EQ(cache.resident_bytes(), blob.size());
    auto other = image(1, 2);
    EXPECT_EQ(prepare(cache, device, 1, other, hit), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    KernelCallableCache oversized(blob.size());
    oversized.set_generation(1);
    auto too_big = image(blob.size() - sizeof(ChipCallable) + 1);
    EXPECT_EQ(prepare(oversized, device, 0, too_big, hit), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    EXPECT_EQ(device.copies, 1);
    KernelCallableResidency found;
    EXPECT_EQ(cache.resolve({0, 1}, found), 0);
}

TEST(KernelCallableCache, AlignmentPaddingConsumesBudgetAndHitStillFits) {
    auto blob = image(1);
    ASSERT_NE(blob.size(), charge(blob));
    KernelCallableCache too_small(blob.size());
    too_small.set_generation(1);
    FakeDevice device;
    bool hit;
    EXPECT_EQ(prepare(too_small, device, 0, blob, hit), PTO_RUNTIME_ERR_CALLABLE_BYTES_EXCEEDED);
    EXPECT_EQ(device.allocations, 0);
    KernelCallableCache cache(charge(blob));
    cache.set_generation(1);
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    cache.commit(0);
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    EXPECT_TRUE(hit);
    EXPECT_EQ(cache.resident_bytes(), charge(blob));
    EXPECT_EQ(cache.host_bytes(), blob.size());
    EXPECT_EQ(device.copies, 1);
}

TEST(KernelCallableCache, PendingEntryIsNotLaunchableAndRollbackPreservesResidents) {
    KernelCallableCache cache;
    cache.set_generation(1);
    FakeDevice device;
    bool hit;
    auto first = image();
    auto second = image(65, 2);
    ASSERT_EQ(prepare(cache, device, 0, first, hit), 0);
    cache.commit(0);
    ASSERT_EQ(prepare(cache, device, 1, second, hit), 0);
    KernelCallableResidency found;
    EXPECT_EQ(cache.resolve({1, 1}, found), PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT);
    EXPECT_EQ(prepare(cache, device, 2, image(65, 3), hit), PTO_RUNTIME_ERR_INVALID_STATE);
    cache.rollback(1);
    EXPECT_EQ(cache.resident_bytes(), charge(first));
    EXPECT_EQ(cache.resolve({0, 1}, found), 0);
    ASSERT_EQ(prepare(cache, device, 1, second, hit), 0);
    cache.commit(1);
    EXPECT_EQ(cache.resolve({1, 1}, found), 0);
    EXPECT_EQ(found.device_address, 0x10000000 + KernelCallableCache::kDescriptorBytes + charge(first));
}

TEST(KernelCallableCache, AllocationAndCopyFailuresAreRetryable) {
    KernelCallableCache cache;
    cache.set_generation(1);
    FakeDevice device;
    bool hit;
    auto blob = image();
    device.fail_alloc = true;
    EXPECT_EQ(prepare(cache, device, 0, blob, hit), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(cache.resident_bytes(), 0);
    device.fail_alloc = false;
    device.copy_error = -123;
    EXPECT_EQ(prepare(cache, device, 0, blob, hit), -123);
    EXPECT_EQ(cache.resident_bytes(), 0);
    device.copy_error = 0;
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    cache.commit(0);
    EXPECT_EQ(cache.resident_count(), 1);
    EXPECT_EQ(device.allocations, 2);
}

TEST(KernelCallableCache, RejectsMalformedSpansBeforeHashOrDeviceOperations) {
    KernelCallableCache cache;
    cache.set_generation(1);
    FakeDevice device;
    bool hit;
    auto blob = image();
    auto *chip = reinterpret_cast<ChipCallable *>(blob.data());
    chip->binary_size_ = UINT32_MAX;
    EXPECT_EQ(prepare(cache, device, 0, blob, hit), PTO_RUNTIME_ERR_INTERNAL);
    chip->binary_size_ = 64;
    chip->child_count_ = 1025;
    EXPECT_EQ(prepare(cache, device, 0, blob, hit), PTO_RUNTIME_ERR_INTERNAL);
    chip->child_count_ = 1;
    chip->child_offsets_[0] = UINT32_MAX;
    EXPECT_EQ(prepare(cache, device, 0, blob, hit), PTO_RUNTIME_ERR_INTERNAL);
    chip->child_offsets_[0] = 64;
    EXPECT_EQ(prepare(cache, device, 0, blob, hit), PTO_RUNTIME_ERR_INTERNAL);
    chip->child_count_ = 0;
    chip->func_name_len_ = CALLABLE_FUNC_NAME_MAX;
    EXPECT_EQ(prepare(cache, device, 0, blob, hit), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(device.allocations, 0);
    EXPECT_EQ(device.copies, 0);
}

TEST(KernelCallableCache, HostBackingIsImmutableAndResolveDoesNotAllocateOrUpload) {
    KernelCallableCache cache;
    cache.set_generation(11);
    FakeDevice device;
    bool hit;
    auto blob = image();
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    cache.commit(0);
    blob.back() = 2;
    auto original = image();
    ASSERT_EQ(prepare(cache, device, 0, original, hit), 0);
    EXPECT_TRUE(hit);
    KernelCallableResidency found;
    for (int i = 0; i < 100; ++i)
        ASSERT_EQ(cache.resolve({0, 11}, found), 0);
    EXPECT_EQ(device.allocations, 1);
    EXPECT_EQ(device.copies, 1);
    cache.clear();
    EXPECT_EQ(cache.resolve({0, 11}, found), PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT);
    EXPECT_EQ(cache.host_bytes(), 0);
}
TEST(KernelCallableCache, ChildAddressesArePatchedOnlyInDeviceScratch) {
    const uint8_t code[] = {1, 2, 3};
    auto child = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, code, sizeof(code));
    const int32_t func_id = 5;
    auto blob = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, 0, "orch", code, sizeof(code), &func_id, &child, 1, ""
    );
    const auto original = blob;
    KernelCallableCache cache;
    cache.set_generation(1);
    FakeDevice device;
    bool hit;
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    cache.commit(0);
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve({0, 1}, found), 0);
    const auto *uploaded = reinterpret_cast<const ChipCallable *>(device.last_upload.data());
    EXPECT_EQ(
        uploaded->child(0).resolved_addr(), found.device_address + offsetof(ChipCallable, storage_) +
                                                uploaded->child_offset(0) + CoreCallable::binary_data_offset()
    );
    EXPECT_EQ(blob, original);
    auto *input = reinterpret_cast<ChipCallable *>(blob.data());
    input->child_count_ = 2;
    input->child_offsets_[1] = input->child_offsets_[0];
    EXPECT_EQ(prepare(cache, device, 1, blob, hit), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(KernelCallableCache, DeviceComparandRejectsStaleGenerationAndWrongCallable) {
    KernelCallableCache cache;
    cache.set_generation(17);
    FakeDevice device;
    bool hit;
    auto blob = image();
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    cache.commit(0);
    SimplerKernelInvocationHeader invocation{};
    invocation.mode = SIMPLER_MODE_KERNEL;
    invocation.callable_id = 0;
    invocation.generation = 17;
    EXPECT_TRUE(kernel_callable_residency_matches(invocation, device.descriptor));
    invocation.generation = 16;
    EXPECT_FALSE(kernel_callable_residency_matches(invocation, device.descriptor));
    invocation.generation = 0;
    EXPECT_FALSE(kernel_callable_residency_matches(invocation, device.descriptor));
    invocation.generation = 17;
    invocation.callable_id = 3;
    EXPECT_FALSE(kernel_callable_residency_matches(invocation, device.descriptor));
    invocation.callable_id = 0;
    invocation.mode = SIMPLER_MODE_PROGRAM;
    EXPECT_FALSE(kernel_callable_residency_matches(invocation, device.descriptor));
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve({0, 17}, found), 0);
    EXPECT_EQ(found.descriptor_address, 0x10000000);
}
TEST(KernelCallableCache, HandleGenerationIsCheckedWithoutMutation) {
    KernelCallableCache cache;
    cache.set_generation(17);
    FakeDevice device;
    bool hit;
    auto blob = image();
    ASSERT_EQ(prepare(cache, device, 0, blob, hit), 0);
    cache.commit(0);
    KernelCallableResidency found;
    ASSERT_EQ(cache.resolve({0, 17}, found), 0);
    const auto address = found.device_address;
    const auto copies = device.copies;
    EXPECT_EQ(cache.resolve({0, 16}, found), PTO_RUNTIME_ERR_CALLABLE_STALE);
    EXPECT_EQ(found.device_address, 0);
    EXPECT_EQ(cache.resolve({0, 18}, found), PTO_RUNTIME_ERR_CALLABLE_STALE);
    EXPECT_EQ(cache.resolve({0, 0}, found), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(cache.resolve({1, 17}, found), PTO_RUNTIME_ERR_CALLABLE_NOT_RESIDENT);
    EXPECT_EQ(cache.resolve({-1, 17}, found), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(cache.resolve({64, 17}, found), PTO_RUNTIME_ERR_CALLABLE_COUNT_EXCEEDED);
    ASSERT_EQ(cache.resolve({0, 17}, found), 0);
    EXPECT_EQ(found.device_address, address);
    EXPECT_EQ(device.copies, copies);
    EXPECT_EQ(cache.resident_count(), 1);

    KernelCallableCache other;
    other.set_generation(18);
    FakeDevice other_device;
    ASSERT_EQ(prepare(other, other_device, 0, blob, hit), 0);
    other.commit(0);
    EXPECT_EQ(other.resolve({0, 17}, found), PTO_RUNTIME_ERR_CALLABLE_STALE);
    EXPECT_EQ(other.resolve({0, 18}, found), 0);
}
}  // namespace
