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
 * @file test_aicore_pmu_lease.cpp
 * @brief Reference ownership and device targeting for the AICore PMU lease.
 *
 * The lease serves overlapping runs from one object, so a release that is not
 * matched to the run that acquired is how channel 53 gets stopped underneath a
 * run still counting. Three paths release without knowing whether their run
 * ever took a reference: the rollback that fires when an earlier collector
 * refused the run, the refusal path inside the PMU block, and the run's close
 * boundary. Each is exercised here.
 *
 * None of this needs a device. The driver entry points are resolved by name at
 * run time, so a definition of that name in the test binary is what the lease
 * calls — which is also what lets the refusal codes be driven directly.
 */

#include "aicore_pmu_lease.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/acl_hal_device.h"

namespace drain_allocation_fault {
thread_local bool armed = false;
thread_local bool fired = false;
}  // namespace drain_allocation_fault

void *operator new(std::size_t size) {
    if (drain_allocation_fault::armed) {
        drain_allocation_fault::armed = false;
        drain_allocation_fault::fired = true;
        throw std::bad_alloc();
    }
    if (void *p = std::malloc(size == 0 ? 1 : size)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

namespace {

/// What the fake driver did, in order, so a release that should have been a
/// no-op is visible as a stop rather than only as a count.
struct ProfCalls {
    std::vector<unsigned int> started;
    std::vector<unsigned int> stopped;
    int start_rc = 0;
    int flush_rc = 0;
    int read_rc = 0;
    bool fail_drain_allocation = false;
    bool throw_on_read = false;

    void reset() {
        started.clear();
        stopped.clear();
        start_rc = 0;
        flush_rc = 0;
        read_rc = 0;
        fail_drain_allocation = false;
        throw_on_read = false;
        drain_allocation_fault::armed = false;
        drain_allocation_fault::fired = false;
    }
};

ProfCalls &calls() {
    static ProfCalls c;
    return c;
}

}  // namespace

// The lease resolves these by name through dlsym(RTLD_DEFAULT, ...), so these
// definitions are what it finds in a test binary.
extern "C" {

int prof_drv_start(unsigned int device_id, unsigned int channel_id, void *start_para) {
    (void)channel_id;
    (void)start_para;
    if (calls().start_rc != 0) return calls().start_rc;
    calls().started.push_back(device_id);
    return 0;
}

int prof_stop(unsigned int device_id, unsigned int channel_id) {
    (void)channel_id;
    calls().stopped.push_back(device_id);
    return 0;
}

int halProfDataFlush(unsigned int device_id, unsigned int channel_id, unsigned int *data_len) {
    (void)device_id;
    (void)channel_id;
    if (data_len != nullptr) *data_len = 0;
    drain_allocation_fault::armed = calls().fail_drain_allocation;
    return calls().flush_rc;
}

int prof_channel_read(unsigned int device_id, unsigned int channel_id, char *out_buf, unsigned int buf_size) {
    (void)device_id;
    (void)channel_id;
    (void)out_buf;
    (void)buf_size;
    if (calls().throw_on_read) throw std::runtime_error("injected drain read failure");
    return calls().read_rc;
}

}  // extern "C"

namespace {

constexpr int kDevice = 0;
constexpr const char *kVisibleDevices = "ASCEND_RT_VISIBLE_DEVICES";

// A release naming an epoch that never acquired leaves the channel alone.
TEST(AicorePmuLeaseTest, ReleaseWithoutAcquireIsANoop) {
    calls().reset();
    AicorePmuLease lease;
    EXPECT_TRUE(lease.acquire(kDevice, /*run_epoch=*/1));
    EXPECT_EQ(calls().started.size(), 1U);

    // The rollback path for a run an earlier collector refused: it reaches the
    // release without ever having reached the acquire.
    lease.release(/*run_epoch=*/2);
    EXPECT_TRUE(calls().stopped.empty());
    EXPECT_TRUE(lease.held());

    lease.release(/*run_epoch=*/1);
    EXPECT_EQ(calls().stopped.size(), 1U);
    EXPECT_FALSE(lease.held());
}

// Releasing the same run twice drops one reference, not two.
TEST(AicorePmuLeaseTest, DoubleReleaseOfOneRunDropsOneReference) {
    calls().reset();
    AicorePmuLease lease;
    EXPECT_TRUE(lease.acquire(kDevice, 1));
    EXPECT_TRUE(lease.acquire(kDevice, 2));
    // Only one start: the second run joins the reference the first took.
    EXPECT_EQ(calls().started.size(), 1U);

    // The PMU refusal path releases, and the caller's rollback releases again.
    lease.release(2);
    lease.release(2);
    EXPECT_TRUE(calls().stopped.empty());
    EXPECT_TRUE(lease.held());

    lease.release(1);
    EXPECT_EQ(calls().stopped.size(), 1U);
}

// A run whose acquire was refused owns nothing, so its own close boundary
// cannot stop a channel a later run took.
TEST(AicorePmuLeaseTest, RefusedRunDoesNotReleaseALaterRunsReference) {
    calls().reset();
    calls().start_rc = -10;  // channel held by another process
    AicorePmuLease lease;
    EXPECT_FALSE(lease.acquire(kDevice, 1));
    EXPECT_TRUE(lease.state() == AicorePmuLease::State::kOccupied);
    EXPECT_FALSE(lease.why().empty());

    // The competitor let go, so a later run takes the channel.
    calls().start_rc = 0;
    EXPECT_TRUE(lease.acquire(kDevice, 2));
    EXPECT_EQ(calls().started.size(), 1U);

    // The refused run reaches its boundary after that.
    lease.release(1);
    EXPECT_TRUE(calls().stopped.empty());
    EXPECT_TRUE(lease.held());

    lease.release(2);
    EXPECT_EQ(calls().stopped.size(), 1U);
}

// A refusal is not sticky: the next run tries again.
TEST(AicorePmuLeaseTest, RefusalDoesNotLatch) {
    calls().reset();
    calls().start_rc = -2;
    AicorePmuLease lease;
    EXPECT_FALSE(lease.acquire(kDevice, 1));
    EXPECT_TRUE(lease.state() == AicorePmuLease::State::kRefused);

    calls().start_rc = 0;
    EXPECT_TRUE(lease.acquire(kDevice, 2));
    EXPECT_TRUE(lease.held());
    lease.release(2);
}

// Start, and the stop that matches it, name the driver-visible device.
//
// The caller passes an ACL logical id; every direct HAL call indexes the
// driver-visible space, and the two differ whenever ASCEND_RT_VISIBLE_DEVICES
// remaps. Holding the channel for the wrong one protects nothing while
// reporting success.
TEST(AicorePmuLeaseTest, HalDeviceIdIsUsedForStartAndStop) {
    calls().reset();
    AicorePmuLease lease;
    EXPECT_TRUE(lease.acquire(kDevice, 1));
    ASSERT_EQ(calls().started.size(), 1U);
    const unsigned int expected = static_cast<unsigned int>(pto::acl_to_hal_device_id(kDevice));
    EXPECT_EQ(calls().started[0], expected);

    lease.release(1);
    ASSERT_EQ(calls().stopped.size(), 1U);
    EXPECT_EQ(calls().stopped[0], expected);
}

// Start and stop target the remapped device, not the logical one.
//
// The check above runs where the two spaces coincide, so it cannot fail on a
// lease that forwarded the logical id untranslated. This one sets the remap the
// translation exists for: with ASCEND_RT_VISIBLE_DEVICES="3,6", logical 1 is
// driver-visible 6, and a lease holding channel 53 on 1 would protect nothing
// while reporting success.
TEST(AicorePmuLeaseTest, RemappedDeviceIsTranslatedBeforeTheHalCall) {
    calls().reset();
    const char *previous = std::getenv(kVisibleDevices);
    const std::string saved = previous != nullptr ? std::string(previous) : std::string();
    const bool had_previous = previous != nullptr;
    ASSERT_EQ(setenv(kVisibleDevices, "3,6", 1), 0);

    {
        AicorePmuLease lease;
        EXPECT_TRUE(lease.acquire(/*device_id=*/1, /*run_epoch=*/1));
        ASSERT_EQ(calls().started.size(), 1U);
        EXPECT_EQ(calls().started[0], 6U);
        lease.release(1);
        ASSERT_EQ(calls().stopped.size(), 1U);
        EXPECT_EQ(calls().stopped[0], 6U);
    }

    if (had_previous) {
        ASSERT_EQ(setenv(kVisibleDevices, saved.c_str(), 1), 0);
    } else {
        ASSERT_EQ(unsetenv(kVisibleDevices), 0);
    }
}

// A second device while one is held is a caller defect, not a retarget.
TEST(AicorePmuLeaseTest, AcquireForASecondDeviceIsRefused) {
    calls().reset();
    AicorePmuLease lease;
    EXPECT_TRUE(lease.acquire(kDevice, 1));
    EXPECT_FALSE(lease.acquire(kDevice + 1, 2));
    EXPECT_EQ(calls().started.size(), 1U);
    lease.release(1);
    EXPECT_EQ(calls().stopped.size(), 1U);
}

// No device bound is a logged refusal, not a call into the driver.
TEST(AicorePmuLeaseTest, NegativeDeviceIsRefusedWithoutCallingTheDriver) {
    calls().reset();
    AicorePmuLease lease;
    EXPECT_FALSE(lease.acquire(-1, 1));
    EXPECT_TRUE(calls().started.empty());
    EXPECT_TRUE(lease.state() == AicorePmuLease::State::kRefused);
}

TEST(AicorePmuLeaseTest, DrainAllocationFailureStillStopsAtBoundary) {
    EXPECT_EXIT(
        {
            calls().reset();
            AicorePmuLease lease;
            if (!lease.acquire(kDevice, 1)) std::_Exit(2);
            const auto device = calls().started.front();
            calls().fail_drain_allocation = true;
            {
                struct LeaseRelease {
                    AicorePmuLease &lease;
                    ~LeaseRelease() { lease.release(1); }
                } release{lease};
            }
            const bool stopped = calls().stopped.size() == 1 && calls().stopped.front() == device;
            std::_Exit(drain_allocation_fault::fired && stopped && !lease.held() ? 0 : 3);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(AicorePmuLeaseTest, DrainExceptionStillStops) {
    calls().reset();
    {
        AicorePmuLease lease;
        ASSERT_TRUE(lease.acquire(kDevice, 1));
        calls().throw_on_read = true;
        EXPECT_NO_THROW(lease.release(1));
        ASSERT_EQ(calls().stopped.size(), 1U);
        EXPECT_EQ(calls().stopped.front(), calls().started.front());
        EXPECT_FALSE(lease.held());
        lease.release(1);
    }
    EXPECT_EQ(calls().stopped.size(), 1U);
}

TEST(AicorePmuLeaseTest, DrainRefusalStillStops) {
    calls().reset();
    AicorePmuLease lease;
    ASSERT_TRUE(lease.acquire(kDevice, 1));
    calls().flush_rc = -7;
    calls().read_rc = -8;
    EXPECT_NO_THROW(lease.release(1));
    ASSERT_EQ(calls().stopped.size(), 1U);
    EXPECT_EQ(calls().stopped.front(), calls().started.front());
    EXPECT_FALSE(lease.held());
}

TEST(AicorePmuLeaseTest, InvalidDevicePreservesHeldEpochs) {
    calls().reset();
    AicorePmuLease lease;
    ASSERT_TRUE(lease.acquire(kDevice, 1));
    ASSERT_TRUE(lease.acquire(kDevice, 2));
    EXPECT_FALSE(lease.acquire(-1, 3));
    EXPECT_TRUE(lease.held());
    EXPECT_TRUE(lease.why().empty());
    lease.release(3);
    lease.release(1);
    EXPECT_TRUE(lease.held());
    EXPECT_TRUE(calls().stopped.empty());
    lease.release(2);
    ASSERT_EQ(calls().stopped.size(), 1U);
    EXPECT_EQ(calls().stopped.front(), calls().started.front());
}

TEST(AicorePmuLeaseTest, DestructorStopsAfterInvalidAcquire) {
    calls().reset();
    {
        AicorePmuLease lease;
        ASSERT_TRUE(lease.acquire(kDevice, 1));
        EXPECT_FALSE(lease.acquire(-1, 2));
    }
    ASSERT_EQ(calls().stopped.size(), 1U);
    EXPECT_EQ(calls().stopped.front(), calls().started.front());
}

}  // namespace
