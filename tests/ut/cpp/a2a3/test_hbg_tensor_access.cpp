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

/**
 * Host-view resolution for the host orchestrator's tensor reads and writes,
 * and the per-run ownership of the mappings that serve them.
 *
 * The fallback path serves staged tensors without mapping their device
 * allocations. `g_registered_view` is what the fake
 * `register_device_memory_to_host` hands back when no fallback is available.
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "common/host_api.h"
#include "host_build_graph/host_tensor_access.h"

namespace {

// Stands in for a device address range that no host load can reach. Only its
// arithmetic is exercised — nothing dereferences it.
constexpr uint64_t kFakeDeviceBase = 0x7000'0000'0000ull;

struct CopyCall {
    void *dev_ptr;
    const void *host_ptr;
    size_t size;
};

std::vector<CopyCall> g_copies;
std::vector<void *> g_unregistered;
void *g_registered_view = nullptr;
int g_register_count = 0;
int g_copy_result = 0;

int record_copy(void *, void *dev_ptr, const void *host_ptr, size_t size) {
    g_copies.push_back({dev_ptr, host_ptr, size});
    return g_copy_result;
}

void *record_register(void *, void *, size_t) {
    ++g_register_count;
    return g_registered_view;
}

void record_unregister(void *, void *dev_ptr) { g_unregistered.push_back(dev_ptr); }

const HostApiOps kHostApiOps{
    .copy_to_device = record_copy,
    .register_device_memory_to_host = record_register,
    .unregister_device_memory_from_host = record_unregister,
};
const HostApi kHostApi(nullptr, 0, 0, &kHostApiOps);

class HostTensorAccessTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_copies.clear();
        g_unregistered.clear();
        g_registered_view = nullptr;
        g_register_count = 0;
        g_copy_result = 0;
    }
};

TEST_F(HostTensorAccessTest, MissingFallbackUsesReturnedDeviceMappingAddress) {
    int32_t buffer[4] = {10, 20, 30, 40};
    g_registered_view = buffer;
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(buffer), nullptr));
    EXPECT_EQ(g_register_count, 1);
    EXPECT_EQ(accessor.mapping_count(), 1u);
    EXPECT_EQ(accessor.mapped_bytes(), sizeof(buffer));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(&accessor, kFakeDeviceBase + 2 * sizeof(int32_t), &value, sizeof(value)));
    EXPECT_EQ(value, 30);

    const int32_t written = 99;
    ASSERT_TRUE(host_tensor_write(&accessor, kFakeDeviceBase + sizeof(int32_t), &written, sizeof(written)));
    EXPECT_EQ(buffer[1], 99);
    EXPECT_TRUE(g_copies.empty());

    accessor.close();
    EXPECT_EQ(g_unregistered, std::vector<void *>{reinterpret_cast<void *>(kFakeDeviceBase)});
}

TEST_F(HostTensorAccessTest, FallbackViewAvoidsDeviceMapping) {
    int32_t fallback[2] = {1, 2};
    int32_t mapped[2] = {3, 4};
    g_registered_view = mapped;
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(fallback), fallback));

    EXPECT_EQ(g_register_count, 0);
    EXPECT_EQ(accessor.mapping_count(), 0u);
    EXPECT_EQ(accessor.mapped_bytes(), 0u);
    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(&accessor, kFakeDeviceBase, &value, sizeof(value)));
    EXPECT_EQ(value, 1);

    accessor.close();
    EXPECT_TRUE(g_unregistered.empty());
}

TEST_F(HostTensorAccessTest, FallbackWriteMutatesCallerBufferAndPushesToDevice) {
    int32_t fallback[4] = {1, 2, 3, 4};
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(fallback), fallback));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(&accessor, kFakeDeviceBase + 3 * sizeof(int32_t), &value, sizeof(value)));
    EXPECT_EQ(value, 4);

    const int32_t written = 77;
    const uint64_t dev_addr = kFakeDeviceBase + 2 * sizeof(int32_t);
    ASSERT_TRUE(host_tensor_write(&accessor, dev_addr, &written, sizeof(written)));
    EXPECT_EQ(fallback[2], 77);
    ASSERT_EQ(g_copies.size(), 1u);
    EXPECT_EQ(g_copies[0].dev_ptr, reinterpret_cast<void *>(dev_addr));
    EXPECT_EQ(g_copies[0].host_ptr, static_cast<const void *>(&fallback[2]));
    EXPECT_EQ(g_copies[0].size, sizeof(int32_t));
}

TEST_F(HostTensorAccessTest, FallbackWriteReportsCopyFailure) {
    int32_t fallback[2] = {1, 2};
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(fallback), fallback));

    g_copy_result = -1;
    const int32_t written = 5;
    EXPECT_FALSE(host_tensor_write(&accessor, kFakeDeviceBase, &written, sizeof(written)));
}

// The fail-closed contract: an address outside every registered region — a
// GM-heap tensor the orchestrator created or a pass-through child-memory
// buffer — resolves to nothing instead of being dereferenced.
TEST_F(HostTensorAccessTest, UnregisteredSpanFailsClosed) {
    int32_t fallback[2] = {1, 2};
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(fallback), fallback));

    int32_t value = 0xABCD;
    EXPECT_FALSE(host_tensor_read(&accessor, kFakeDeviceBase + 0x100000, &value, sizeof(value)));
    EXPECT_EQ(value, 0xABCD);

    const int32_t written = 5;
    EXPECT_FALSE(host_tensor_write(&accessor, kFakeDeviceBase + 0x100000, &written, sizeof(written)));
    EXPECT_TRUE(g_copies.empty());
}

TEST_F(HostTensorAccessTest, SpanOverrunningTheRegionFails) {
    int32_t fallback[2] = {1, 2};
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(fallback), fallback));

    int64_t value = 0;
    // Starts inside the region, ends past it.
    EXPECT_FALSE(host_tensor_read(&accessor, kFakeDeviceBase + sizeof(int32_t), &value, sizeof(value)));
    // Starts before it.
    EXPECT_FALSE(host_tensor_read(&accessor, kFakeDeviceBase - sizeof(int32_t), &value, sizeof(int32_t)));
}

// Several tensors are staged per run into one accessor, and each resolves
// against its own region.
TEST_F(HostTensorAccessTest, RegionsWithinOneAccessorResolveIndependently) {
    int32_t first[2] = {1, 2};
    int32_t second[2] = {3, 4};
    const uint64_t second_base = kFakeDeviceBase + 0x10000;
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(first), first));
    ASSERT_TRUE(accessor.add(second_base, sizeof(second), second));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(&accessor, kFakeDeviceBase, &value, sizeof(value)));
    EXPECT_EQ(value, 1);
    ASSERT_TRUE(host_tensor_read(&accessor, second_base + sizeof(int32_t), &value, sizeof(value)));
    EXPECT_EQ(value, 4);
}

// `write` pushes mirror bytes back through `api->copy_to_device` without
// re-checking the hook, which is only sound because a null api cannot produce a
// region in the first place.
TEST_F(HostTensorAccessTest, NullApiRegistersNothing) {
    int32_t mirror[2] = {1, 2};
    HostTensorAccessor accessor(nullptr);
    EXPECT_FALSE(accessor.add(kFakeDeviceBase, sizeof(mirror), mirror));

    int32_t value = 0;
    EXPECT_FALSE(host_tensor_read(&accessor, kFakeDeviceBase, &value, sizeof(value)));
    const int32_t written = 5;
    EXPECT_FALSE(host_tensor_write(&accessor, kFakeDeviceBase, &written, sizeof(written)));
}

TEST_F(HostTensorAccessTest, ContextsKeepRegionsIndependent) {
    int32_t first[2] = {1, 2};
    int32_t second[2] = {3, 4};
    HostTensorAccessor first_access(&kHostApi);
    HostTensorAccessor second_access(&kHostApi);
    ASSERT_TRUE(first_access.add(kFakeDeviceBase, sizeof(first), first));
    ASSERT_TRUE(second_access.add(kFakeDeviceBase, sizeof(second), second));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(&first_access, kFakeDeviceBase, &value, sizeof(value)));
    EXPECT_EQ(value, 1);
    ASSERT_TRUE(host_tensor_read(&second_access, kFakeDeviceBase, &value, sizeof(value)));
    EXPECT_EQ(value, 3);

    first_access.close();
    EXPECT_FALSE(host_tensor_read(&first_access, kFakeDeviceBase, &value, sizeof(value)));
    EXPECT_TRUE(host_tensor_read(&second_access, kFakeDeviceBase, &value, sizeof(value)));
}

// Two concurrent runs each stage, read and close their own accessor. Both use
// overlapping device addresses and the fallback view, so the only thing keeping
// their regions apart is that each accessor owns its own tables — the property
// a shared file-scope region list cannot have. Each thread mutates its tables
// while the other is mutating its own, so a reintroduced global shows up as a
// wrong value or a failed lookup rather than as a passing no-op.
TEST_F(HostTensorAccessTest, ConcurrentRunsKeepRegionsIndependent) {
    constexpr int kRegions = 8;
    constexpr int kRounds = 64;

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};

    auto run = [&](int32_t seed, bool *ok) {
        std::vector<std::array<int32_t, 2>> buffers(kRegions);
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {}
        for (int round = 0; round < kRounds; ++round) {
            HostTensorAccessor accessor(&kHostApi);
            for (int i = 0; i < kRegions; ++i) {
                buffers[i] = {seed + i, seed + i + 100};
                const uint64_t base = kFakeDeviceBase + static_cast<uint64_t>(i) * 0x10000;
                if (!accessor.add(base, sizeof(buffers[i]), buffers[i].data())) {
                    *ok = false;
                    return;
                }
            }
            for (int i = 0; i < kRegions; ++i) {
                const uint64_t base = kFakeDeviceBase + static_cast<uint64_t>(i) * 0x10000;
                int32_t value = 0;
                if (!host_tensor_read(&accessor, base, &value, sizeof(value)) || value != seed + i) {
                    *ok = false;
                    return;
                }
            }
            accessor.close();
        }
    };

    bool first_ok = true;
    bool second_ok = true;
    std::thread first_thread(run, 1, &first_ok);
    std::thread second_thread(run, 1000, &second_ok);
    while (ready.load(std::memory_order_acquire) != 2) {}
    start.store(true, std::memory_order_release);
    first_thread.join();
    second_thread.join();

    EXPECT_TRUE(first_ok);
    EXPECT_TRUE(second_ok);
}

TEST_F(HostTensorAccessTest, CloseReleasesOnlyOwnedMappings) {
    int32_t first[2] = {1, 2};
    int32_t second[2] = {3, 4};
    HostTensorAccessor first_access(&kHostApi);
    HostTensorAccessor second_access(&kHostApi);

    g_registered_view = first;
    ASSERT_TRUE(first_access.add(kFakeDeviceBase, sizeof(first), nullptr));
    g_registered_view = second;
    const uint64_t second_base = kFakeDeviceBase + 0x10000;
    ASSERT_TRUE(second_access.add(second_base, sizeof(second), nullptr));

    first_access.close();
    ASSERT_EQ(g_unregistered.size(), 1u);
    EXPECT_EQ(g_unregistered[0], reinterpret_cast<void *>(kFakeDeviceBase));
    second_access.close();
    ASSERT_EQ(g_unregistered.size(), 2u);
    EXPECT_EQ(g_unregistered[1], reinterpret_cast<void *>(second_base));
}

TEST_F(HostTensorAccessTest, DestructorReleasesOwnedMapping) {
    int32_t buffer[2] = {1, 2};
    g_registered_view = buffer;
    {
        HostTensorAccessor accessor(&kHostApi);
        ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(buffer), nullptr));
    }
    ASSERT_EQ(g_unregistered.size(), 1u);
    EXPECT_EQ(g_unregistered[0], reinterpret_cast<void *>(kFakeDeviceBase));
}

TEST_F(HostTensorAccessTest, EmptyOrNullFallbackRegionIsRejected) {
    int32_t mirror[2] = {1, 2};
    HostTensorAccessor accessor(&kHostApi);
    EXPECT_FALSE(accessor.add(kFakeDeviceBase, 0, mirror));
    EXPECT_FALSE(accessor.add(kFakeDeviceBase, sizeof(mirror), nullptr));
}

}  // namespace
