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
 * The fallback path serves host-memory tensors without mapping their device
 * allocations. `g_registered_view` is what the fake
 * `register_device_memory_to_host` hands back when no fallback is available.
 */

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
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
std::vector<CopyCall> g_reads;
std::vector<void *> g_unregistered;
void *g_registered_view = nullptr;
int g_register_count = 0;
int g_copy_result = 0;
// What the fake `acquire_child_memory_host_view` hands back, and how often it
// was asked. Null models the two platforms that have no host-map path for an
// allocation: a5 onboard, and an ordinary-page small allocation on a 64 KiB-page
// host (issue #1531).
void *g_child_memory_view = nullptr;
int g_child_memory_acquire_count = 0;
int g_read_result = 0;
// Bytes the fake device holds, so a device-copy read returns something the test
// can distinguish from the host view.
unsigned char g_device_bytes[64];

int record_copy(void *, void *dev_ptr, const void *host_ptr, size_t size) {
    g_copies.push_back({dev_ptr, host_ptr, size});
    return g_copy_result;
}

int record_read(void *, void *host_ptr, const void *dev_ptr, size_t size) {
    g_reads.push_back({const_cast<void *>(dev_ptr), host_ptr, size});
    if (g_read_result == 0) {
        const uint64_t offset = reinterpret_cast<uint64_t>(dev_ptr) - kFakeDeviceBase;
        memcpy(host_ptr, g_device_bytes + offset, size);
    }
    return g_read_result;
}

void *record_register(void *, void *, size_t) {
    ++g_register_count;
    return g_registered_view;
}

void record_unregister(void *, void *dev_ptr) { g_unregistered.push_back(dev_ptr); }

void *record_child_memory_acquire(void *, void *, size_t) {
    ++g_child_memory_acquire_count;
    return g_child_memory_view;
}

const HostApiOps kHostApiOps{
    .copy_to_device = record_copy,
    .copy_from_device = record_read,
    .register_device_memory_to_host = record_register,
    .unregister_device_memory_from_host = record_unregister,
    .acquire_child_memory_host_view = record_child_memory_acquire,
};
const HostApi kHostApi(nullptr, 0, 0, &kHostApiOps);

class HostTensorAccessTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_copies.clear();
        g_reads.clear();
        g_unregistered.clear();
        g_registered_view = nullptr;
        g_register_count = 0;
        g_copy_result = 0;
        g_child_memory_view = nullptr;
        g_child_memory_acquire_count = 0;
        g_read_result = 0;
        memset(g_device_bytes, 0, sizeof(g_device_bytes));
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

// The fail-closed contract: an address outside every region — a GM-heap tensor
// the orchestrator created — resolves to nothing instead of being dereferenced.
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

// ---------------------------------------------------------------------------
// Child memory: no caller buffer exists, so the means is chosen on first access.
// ---------------------------------------------------------------------------

// Declaring the region consults nothing. An orchestration that never touches
// the tensor is what makes this the cheap default.
TEST_F(HostTensorAccessTest, ChildMemoryRegionResolvesNothingUntilAccessed) {
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add_child_memory(kFakeDeviceBase, 16));

    EXPECT_EQ(g_child_memory_acquire_count, 0);
    EXPECT_EQ(g_register_count, 0);
    EXPECT_EQ(accessor.mapping_count(), 0u);
    EXPECT_EQ(accessor.device_copy_count(), 0u);
}

TEST_F(HostTensorAccessTest, ChildMemoryMappingServesReadsAndWritesDirectly) {
    int32_t mapped[4] = {10, 20, 30, 40};
    g_child_memory_view = mapped;
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add_child_memory(kFakeDeviceBase, sizeof(mapped)));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(&accessor, kFakeDeviceBase + 2 * sizeof(int32_t), &value, sizeof(value)));
    EXPECT_EQ(value, 30);

    const int32_t written = 99;
    ASSERT_TRUE(host_tensor_write(&accessor, kFakeDeviceBase + sizeof(int32_t), &written, sizeof(written)));
    EXPECT_EQ(mapped[1], 99);

    // The mapping is coherent, so nothing is pushed back and no copy is made.
    EXPECT_TRUE(g_copies.empty());
    EXPECT_TRUE(g_reads.empty());
    EXPECT_EQ(accessor.device_copy_count(), 0u);
}

// The platform owns a child-memory mapping for its allocation's lifetime, so
// this accessor must neither count it as one of its own nor release it.
TEST_F(HostTensorAccessTest, ChildMemoryMappingIsNotOwnedByTheAccessor) {
    int32_t mapped[2] = {1, 2};
    g_child_memory_view = mapped;
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add_child_memory(kFakeDeviceBase, sizeof(mapped)));
    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(&accessor, kFakeDeviceBase, &value, sizeof(value)));

    EXPECT_EQ(accessor.mapping_count(), 0u);
    EXPECT_EQ(accessor.mapped_bytes(), 0u);
    accessor.close();
    EXPECT_TRUE(g_unregistered.empty());
}

// The means is resolved once and reused, so a hot orchestration loop does not
// ask the platform per access.
TEST_F(HostTensorAccessTest, ChildMemoryMeansIsResolvedOncePerRegion) {
    int32_t mapped[4] = {1, 2, 3, 4};
    g_child_memory_view = mapped;
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add_child_memory(kFakeDeviceBase, sizeof(mapped)));

    int32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(host_tensor_read(&accessor, kFakeDeviceBase + i * sizeof(int32_t), &value, sizeof(value)));
    }
    EXPECT_EQ(g_child_memory_acquire_count, 1);
}

// a5 onboard, or a #1531 host: every access is a device copy instead. Nothing
// is held between accesses, so a read cannot serve stale bytes.
TEST_F(HostTensorAccessTest, ChildMemoryWithoutMappingCopiesPerAccess) {
    g_child_memory_view = nullptr;
    const int32_t device_values[4] = {5, 6, 7, 8};
    memcpy(g_device_bytes, device_values, sizeof(device_values));

    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add_child_memory(kFakeDeviceBase, sizeof(device_values)));

    int32_t value = 0;
    const uint64_t read_addr = kFakeDeviceBase + 2 * sizeof(int32_t);
    ASSERT_TRUE(host_tensor_read(&accessor, read_addr, &value, sizeof(value)));
    EXPECT_EQ(value, 7);
    ASSERT_EQ(g_reads.size(), 1u);
    EXPECT_EQ(g_reads[0].dev_ptr, reinterpret_cast<void *>(read_addr));
    EXPECT_EQ(g_reads[0].size, sizeof(int32_t));

    // A write lands on the device immediately rather than in a host buffer that
    // would then need pushing back.
    const int32_t written = 77;
    const uint64_t write_addr = kFakeDeviceBase + sizeof(int32_t);
    ASSERT_TRUE(host_tensor_write(&accessor, write_addr, &written, sizeof(written)));
    ASSERT_EQ(g_copies.size(), 1u);
    EXPECT_EQ(g_copies[0].dev_ptr, reinterpret_cast<void *>(write_addr));
    EXPECT_EQ(g_copies[0].host_ptr, static_cast<const void *>(&written));
    EXPECT_EQ(g_copies[0].size, sizeof(int32_t));

    EXPECT_EQ(accessor.device_copy_count(), 2u);
}

TEST_F(HostTensorAccessTest, ChildMemoryDeviceCopyFailurePropagates) {
    g_child_memory_view = nullptr;
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add_child_memory(kFakeDeviceBase, 16));

    g_read_result = -1;
    int32_t value = 0;
    EXPECT_FALSE(host_tensor_read(&accessor, kFakeDeviceBase, &value, sizeof(value)));

    g_copy_result = -1;
    const int32_t written = 5;
    EXPECT_FALSE(host_tensor_write(&accessor, kFakeDeviceBase, &written, sizeof(written)));
}

// A child-memory region bounds accesses exactly as a staged one does: an
// address outside every region resolves to nothing and never reaches the
// platform.
TEST_F(HostTensorAccessTest, ChildMemorySpanOutsideEveryRegionFailsClosed) {
    g_child_memory_view = nullptr;
    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add_child_memory(kFakeDeviceBase, 16));

    int32_t value = 0xABCD;
    EXPECT_FALSE(host_tensor_read(&accessor, kFakeDeviceBase + 0x100000, &value, sizeof(value)));
    EXPECT_EQ(value, 0xABCD);
    int64_t wide = 0;
    // Starts inside, ends past the region.
    EXPECT_FALSE(host_tensor_read(&accessor, kFakeDeviceBase + 12, &wide, sizeof(wide)));

    EXPECT_EQ(g_child_memory_acquire_count, 0);
    EXPECT_TRUE(g_reads.empty());
}

TEST_F(HostTensorAccessTest, ChildMemoryRejectsEmptyRegionAndNullApi) {
    HostTensorAccessor accessor(&kHostApi);
    EXPECT_FALSE(accessor.add_child_memory(kFakeDeviceBase, 0));
    EXPECT_FALSE(accessor.add_child_memory(0, 16));

    HostTensorAccessor no_api(nullptr);
    EXPECT_FALSE(no_api.add_child_memory(kFakeDeviceBase, 16));
}

// Staged and child-memory regions coexist in one accessor and each keeps its
// own means.
TEST_F(HostTensorAccessTest, StagedAndChildMemoryRegionsResolveIndependently) {
    int32_t staged[2] = {1, 2};
    int32_t mapped[2] = {3, 4};
    const uint64_t child_base = kFakeDeviceBase + 0x10000;
    g_child_memory_view = mapped;

    HostTensorAccessor accessor(&kHostApi);
    ASSERT_TRUE(accessor.add(kFakeDeviceBase, sizeof(staged), staged));
    ASSERT_TRUE(accessor.add_child_memory(child_base, sizeof(mapped)));

    int32_t value = 0;
    ASSERT_TRUE(host_tensor_read(&accessor, kFakeDeviceBase, &value, sizeof(value)));
    EXPECT_EQ(value, 1);
    ASSERT_TRUE(host_tensor_read(&accessor, child_base + sizeof(int32_t), &value, sizeof(value)));
    EXPECT_EQ(value, 4);

    // The staged write still pushes back; the mapped one does not.
    const int32_t written = 9;
    ASSERT_TRUE(host_tensor_write(&accessor, kFakeDeviceBase, &written, sizeof(written)));
    EXPECT_EQ(g_copies.size(), 1u);
    ASSERT_TRUE(host_tensor_write(&accessor, child_base, &written, sizeof(written)));
    EXPECT_EQ(g_copies.size(), 1u);
    EXPECT_EQ(mapped[0], 9);
}

}  // namespace
