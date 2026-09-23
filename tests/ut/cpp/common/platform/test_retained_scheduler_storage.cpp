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
// One pipeline slot's retained scheduler-state pair: when it reuses, when it
// grows, and what it owns when a growth or a release fails.
//
// The runner that holds these and the bind that acquires them are covered
// through their own seams (test_hbg_bind_ledger.cpp drives the real a5 bind and
// release); this file is about the mechanism, because the failure paths it has
// to get right — a device allocation that fails, a free that fails, a second
// growth after one did — are the ones a real allocator will not produce on
// demand.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "utils/retained_scheduler_storage.h"

namespace {

constexpr size_t kAlign = 128;

// Stands in for the runner's allocator: real host blocks, so the addresses the
// storage hands out are usable, plus the traffic and the failure switches a
// device allocator cannot be asked for.
struct FakeDevice {
    std::vector<void *> live;
    int allocs = 0;
    int frees = 0;
    bool alloc_fails = false;
    bool free_fails = false;

    void *alloc(size_t bytes) {
        ++allocs;
        if (alloc_fails) return nullptr;
        void *p = std::malloc(bytes == 0 ? 1 : bytes);
        if (p != nullptr) live.push_back(p);
        return p;
    }

    int free(void *p) {
        ++frees;
        if (free_fails) return -7;
        for (auto it = live.begin(); it != live.end(); ++it) {
            if (*it == p) {
                live.erase(it);
                break;
            }
        }
        std::free(p);
        return 0;
    }

    bool holds(const void *p) const {
        for (const void *live_p : live) {
            if (live_p == p) return true;
        }
        return false;
    }

    void release_all() {
        for (void *p : live)
            std::free(p);
        live.clear();
    }
};

class RetainedSchedulerStorageTest : public ::testing::Test {
protected:
    void TearDown() override {
        // The storage may still name a block; drop its bookkeeping first so the
        // fake's own cleanup is the only free.
        storage.abandon();
        device.release_all();
    }

    RetainedSchedulerStorage::Status acquire(size_t bytes, size_t alignment = kAlign) {
        return storage.acquire(
            bytes, alignment,
            [this](size_t n) {
                return device.alloc(n);
            },
            [this](void *p) {
                return device.free(p);
            },
            &device_out, &host_out
        );
    }

    int release() {
        return storage.release([this](void *p) {
            return device.free(p);
        });
    }

    FakeDevice device;
    RetainedSchedulerStorage storage;
    void *device_out = nullptr;
    void *host_out = nullptr;
};

TEST_F(RetainedSchedulerStorageTest, AFirstAcquireAllocatesBothSidesAligned) {
    ASSERT_EQ(acquire(4096), RetainedSchedulerStorage::Status::Ok);
    EXPECT_EQ(device.allocs, 1);
    EXPECT_EQ(device.frees, 0);
    ASSERT_NE(device_out, nullptr);
    ASSERT_NE(host_out, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(device_out) % kAlign, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(host_out) % kAlign, 0u);
    EXPECT_EQ(storage.device_capacity(), 4096u);
    EXPECT_EQ(storage.host_capacity(), 4096u);
    EXPECT_EQ(storage.held_after_failed_release(), nullptr);
    // Writable to its full length on both sides: the caller initializes and
    // ships exactly what it asked for.
    std::memset(host_out, 0x5a, 4096);
    std::memset(device_out, 0x5a, 4096);
}

TEST_F(RetainedSchedulerStorageTest, ARequestThatFitsAllocatesNothing) {
    ASSERT_EQ(acquire(4096), RetainedSchedulerStorage::Status::Ok);
    void *const first_device = device_out;
    void *const first_host = host_out;

    ASSERT_EQ(acquire(4096), RetainedSchedulerStorage::Status::Ok);
    EXPECT_EQ(device.allocs, 1);
    EXPECT_EQ(device.frees, 0);
    EXPECT_EQ(device_out, first_device);
    EXPECT_EQ(host_out, first_host);

    // Smaller still fits; the capacity is not reduced to it.
    ASSERT_EQ(acquire(64), RetainedSchedulerStorage::Status::Ok);
    EXPECT_EQ(device.allocs, 1);
    EXPECT_EQ(device_out, first_device);
    EXPECT_EQ(storage.device_capacity(), 4096u);
}

TEST_F(RetainedSchedulerStorageTest, AGrowthRecordsTheReplacementBeforeReleasingThePredecessor) {
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    ASSERT_EQ(device.live.size(), 1u);
    void *const first_raw = device.live.front();
    void *const first_out = device_out;

    ASSERT_EQ(acquire(8192), RetainedSchedulerStorage::Status::Ok);
    EXPECT_EQ(device.allocs, 2);
    EXPECT_EQ(device.frees, 1);
    EXPECT_NE(device_out, first_out);
    EXPECT_FALSE(device.holds(first_raw)) << "the predecessor is released by the growth";
    EXPECT_EQ(device.live.size(), 1u) << "and exactly one block is left";
    EXPECT_EQ(storage.device_capacity(), 8192u);
    EXPECT_EQ(storage.held_after_failed_release(), nullptr);
}

TEST_F(RetainedSchedulerStorageTest, AFailedDeviceGrowthKeepsThePreviousPlan) {
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    void *const first = device_out;

    device.alloc_fails = true;
    EXPECT_EQ(acquire(8192), RetainedSchedulerStorage::Status::DeviceUnavailable);
    EXPECT_EQ(device.frees, 0) << "a failed growth releases nothing";
    EXPECT_EQ(storage.device_addr(), first);
    EXPECT_EQ(storage.device_capacity(), 1024u);
    EXPECT_EQ(device_out, nullptr) << "nothing is handed out by a refused acquire";

    // The slot still serves what it had.
    device.alloc_fails = false;
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    EXPECT_EQ(device_out, first);
    EXPECT_EQ(device.allocs, 2) << "the failed attempt was the second; this one allocated nothing";
}

TEST_F(RetainedSchedulerStorageTest, AFailedHostGrowthNeverReachesTheDevice) {
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    void *const first = device_out;

    // Larger than any host allocation can serve, so the host side fails first.
    EXPECT_EQ(acquire(SIZE_MAX / 2), RetainedSchedulerStorage::Status::HostUnavailable);
    EXPECT_EQ(device.allocs, 1) << "the device side is not touched when the host side fails";
    EXPECT_EQ(device.frees, 0);
    EXPECT_EQ(storage.device_addr(), first);
    EXPECT_EQ(storage.device_capacity(), 1024u);
    EXPECT_EQ(storage.host_capacity(), 1024u);
}

TEST_F(RetainedSchedulerStorageTest, AFailedReleaseIsHeldAndRefusesEveryFurtherGrowth) {
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    ASSERT_EQ(device.live.size(), 1u);
    // The raw block, which is what a free takes — the address handed out is the
    // aligned one inside it.
    void *const first_raw = device.live.front();

    device.free_fails = true;
    ASSERT_EQ(acquire(4096), RetainedSchedulerStorage::Status::Ok) << "the growth itself succeeded";
    EXPECT_EQ(storage.held_after_failed_release(), first_raw);
    void *const second = storage.device_addr();
    EXPECT_NE(second, first_raw);
    EXPECT_EQ(device_out, second) << "the run still gets the block this acquire published";

    // The record is not overwritten, and no third block is taken: a slot holds
    // its current block plus the one it could not release, and no more.
    device.free_fails = false;
    const int allocs_before = device.allocs;
    EXPECT_EQ(acquire(1 << 20), RetainedSchedulerStorage::Status::GrowthRefused);
    EXPECT_EQ(device.allocs, allocs_before) << "a refused growth allocates nothing";
    EXPECT_EQ(storage.held_after_failed_release(), first_raw);
    EXPECT_EQ(storage.device_addr(), second);
    EXPECT_EQ(storage.device_capacity(), 4096u);

    // What fits is still served while the record is occupied.
    ASSERT_EQ(acquire(4096), RetainedSchedulerStorage::Status::Ok);
    EXPECT_EQ(device_out, second);
}

TEST_F(RetainedSchedulerStorageTest, ReleaseAttemptsBothBlocksAndReportsTheFirstError) {
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    ASSERT_EQ(device.live.size(), 1u);
    void *const first_raw = device.live.front();
    device.free_fails = true;
    ASSERT_EQ(acquire(4096), RetainedSchedulerStorage::Status::Ok);
    ASSERT_EQ(storage.held_after_failed_release(), first_raw);
    device.free_fails = false;

    const int frees_before = device.frees;
    EXPECT_EQ(release(), 0);
    EXPECT_EQ(device.frees, frees_before + 2) << "the live block and the held one are both attempted";
    EXPECT_EQ(storage.device_addr(), nullptr);
    EXPECT_EQ(storage.held_after_failed_release(), nullptr);
    EXPECT_EQ(storage.device_capacity(), 0u);
    EXPECT_EQ(storage.host_capacity(), 0u);

    // Cleared, so a second release is not a second free.
    EXPECT_EQ(release(), 0);
    EXPECT_EQ(device.frees, frees_before + 2);
}

TEST_F(RetainedSchedulerStorageTest, AReleaseWhoseFreesFailReportsAndStillForgets) {
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    device.free_fails = true;

    EXPECT_EQ(release(), -7) << "the allocator's own error, not a substitute";
    EXPECT_EQ(storage.device_addr(), nullptr) << "forgotten either way: a retry here would be a second free";
    EXPECT_EQ(storage.held_after_failed_release(), nullptr);

    // A later acquire starts from nothing rather than inheriting the address.
    device.free_fails = false;
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    EXPECT_EQ(device.allocs, 2);
}

TEST_F(RetainedSchedulerStorageTest, AbandonMakesNoDeviceCall) {
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    const int frees_before = device.frees;

    storage.abandon();
    EXPECT_EQ(device.frees, frees_before) << "a force reset already invalidated the generation";
    EXPECT_EQ(storage.device_addr(), nullptr);
    EXPECT_EQ(storage.host_addr(), nullptr);
    EXPECT_EQ(storage.held_after_failed_release(), nullptr);

    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    EXPECT_EQ(device.allocs, 2) << "the next acquire allocates fresh";
}

TEST_F(RetainedSchedulerStorageTest, TwoSlotsShareNothing) {
    RetainedSchedulerStorage other;
    void *other_device = nullptr;
    void *other_host = nullptr;

    ASSERT_EQ(acquire(2048), RetainedSchedulerStorage::Status::Ok);
    ASSERT_EQ(
        other.acquire(
            2048, kAlign,
            [this](size_t n) {
                return device.alloc(n);
            },
            [this](void *p) {
                return device.free(p);
            },
            &other_device, &other_host
        ),
        RetainedSchedulerStorage::Status::Ok
    );

    EXPECT_NE(device_out, other_device);
    EXPECT_NE(host_out, other_host);
    EXPECT_EQ(device.allocs, 2);
    EXPECT_EQ(
        other.release([this](void *p) {
            return device.free(p);
        }),
        0
    );
    // Releasing one leaves the other's storage exactly where it was.
    EXPECT_EQ(storage.device_addr(), device_out);
}

TEST_F(RetainedSchedulerStorageTest, AnInvalidRequestChangesNothing) {
    ASSERT_EQ(acquire(1024), RetainedSchedulerStorage::Status::Ok);
    void *const first = device_out;
    const int allocs_before = device.allocs;

    EXPECT_EQ(acquire(0), RetainedSchedulerStorage::Status::InvalidRequest);
    EXPECT_EQ(acquire(1024, 3), RetainedSchedulerStorage::Status::InvalidRequest) << "alignment must be a power of two";
    EXPECT_EQ(acquire(1024, 0), RetainedSchedulerStorage::Status::InvalidRequest);
    EXPECT_EQ(acquire(SIZE_MAX, kAlign), RetainedSchedulerStorage::Status::InvalidRequest)
        << "a length whose alignment padding overflows is not a request";

    EXPECT_EQ(device.allocs, allocs_before);
    EXPECT_EQ(device.frees, 0);
    EXPECT_EQ(storage.device_addr(), first);
    EXPECT_EQ(storage.device_capacity(), 1024u);
}

}  // namespace
