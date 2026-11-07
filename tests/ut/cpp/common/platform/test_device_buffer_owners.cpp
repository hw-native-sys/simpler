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
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "host/device_buffer_owners.h"

namespace {
constexpr uint64_t kSession = 7;
constexpr uint64_t kAllocation = 11;
const DeviceBufferOwners::Span kViews[] = {{0x8000, 0x1000}, {0xA000, 0x100}};
void *at(uint64_t addr) { return reinterpret_cast<void *>(addr); }
void grant(DeviceBufferOwners &owners) { ASSERT_TRUE(owners.register_domain_owner(kSession, kAllocation, kViews, 2)); }
}  // namespace

TEST(DeviceBufferOwners, ExplicitDomainGrantIsNotACallerMint) {
    DeviceBufferOwners owners;
    grant(owners);
    EXPECT_TRUE(owners.resolve(0x8080, 0x80, nullptr));
    EXPECT_FALSE(owners.recorded(at(0x8000)));
    EXPECT_EQ(owners.caller_allocation_count(), 0u);
    EXPECT_FALSE(owners.forget_if_unborrowed(at(0x8000)));
    const DeviceBufferOwners::Span used[] = {{0x8080, 0x80}, {0xA000, 0x100}};
    EXPECT_TRUE(owners.borrow(1, used, 2));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(1, false);
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.resolve(0x8000, 1, nullptr));
    EXPECT_EQ(owners.allocation_count(), 0u);
}

TEST(DeviceBufferOwners, BoundsDoNotExpandToCoverAnUnknownView) {
    DeviceBufferOwners owners;
    grant(owners);
    const DeviceBufferOwners::Span used[] = {{0x8FFF, 2}, {0x8000, 1}};
    EXPECT_FALSE(owners.borrow(1, used, 2));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(1, false);
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(DeviceBufferOwners, CallerAndDomainLoansShareOneRunIdentity) {
    DeviceBufferOwners owners;
    owners.record(at(0x2000), 128);
    grant(owners);
    EXPECT_EQ(owners.caller_allocation_count(), 1u);
    const DeviceBufferOwners::Span used[] = {{0x2000, 128}, {0x8000, 64}, {0xA000, 32}};
    EXPECT_TRUE(owners.borrow(19, used, 3));
    EXPECT_FALSE(owners.forget_if_unborrowed(at(0x2000)));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(19, false);
    EXPECT_TRUE(owners.forget_if_unborrowed(at(0x2000)));
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(DeviceBufferOwners, RegistrationRejectsOverlapsWithoutPartialPublication) {
    DeviceBufferOwners owners;
    owners.record(at(0x2000), 128);
    const DeviceBufferOwners::Span used[] = {{0x8000, 256}, {0x2040, 64}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, used, 2));
    EXPECT_FALSE(owners.resolve(0x8000, 1, nullptr));
    EXPECT_EQ(owners.allocation_count(), 1u);
    const DeviceBufferOwners::Span overlapping[] = {{0x8000, 256}, {0x8080, 256}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, overlapping, 2));
    EXPECT_FALSE(owners.resolve(0x8000, 1, nullptr));
}

TEST(DeviceBufferOwners, ACommunicationHandleAndAllocationIdentifyTheOwner) {
    DeviceBufferOwners owners;
    grant(owners);
    EXPECT_FALSE(owners.begin_domain_release(kSession + 1, kAllocation));
    EXPECT_FALSE(owners.finish_domain_release(kSession, kAllocation + 1));
    EXPECT_TRUE(owners.borrow(1, kViews, 2));
    owners.release(1, false);
    const DeviceBufferOwners::Span elsewhere[] = {{0xC000, 128}};
    EXPECT_TRUE(owners.register_domain_owner(kSession + 1, kAllocation, elsewhere, 1));
    EXPECT_TRUE(owners.begin_domain_release(kSession + 1, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession + 1, kAllocation));
    EXPECT_TRUE(owners.resolve(0x8000, 1, nullptr));
}

TEST(DeviceBufferOwners, RevokeRejectsNewUseBeforeTheLastConsumerRetires) {
    DeviceBufferOwners owners;
    grant(owners);
    ASSERT_TRUE(owners.borrow(1, kViews, 2));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_THROW(owners.borrow(2, kViews, 2), std::runtime_error);
    EXPECT_FALSE(owners.declare_writes(2, kViews, 2));
    EXPECT_TRUE(owners.written_by_other_run(2, 0x8000, 1));
    EXPECT_FALSE(owners.finish_domain_release(kSession, kAllocation));
    owners.release(2, false);
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(1, false);
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(DeviceBufferOwners, EveryLiveConsumerKeepsTheWholeDomain) {
    DeviceBufferOwners owners;
    grant(owners);
    EXPECT_TRUE(owners.borrow(1, kViews, 1));
    EXPECT_TRUE(owners.borrow(2, kViews + 1, 1));
    EXPECT_FALSE(owners.written_by_other_run(3, 0x8000, 1));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(1, false);
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(2, false);
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(DeviceBufferOwners, WriteDeclarationsFenceHostReadsIndependentlyOfBorrow) {
    DeviceBufferOwners owners;
    grant(owners);
    EXPECT_TRUE(owners.declare_writes(1, kViews, 1));
    EXPECT_FALSE(owners.written_by_other_run(1, 0x8000, 1));
    EXPECT_TRUE(owners.written_by_other_run(2, 0x8000, 1));
    EXPECT_FALSE(owners.written_by_other_run(2, 0xA000, 1));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(1, false);
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(DeviceBufferOwners, UnprovenConsumptionCannotBeDischargedByIdentityReuse) {
    DeviceBufferOwners owners;
    grant(owners);
    EXPECT_TRUE(owners.borrow(1, kViews, 2));
    EXPECT_TRUE(owners.declare_writes(1, kViews, 1));
    owners.release(1, true);
    owners.release(1, false);
    EXPECT_TRUE(owners.written_by_other_run(1, 0x8000, 1));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(DeviceBufferOwners, UnknownPlatformReleaseRetainsRevokedOwnership) {
    DeviceBufferOwners owners;
    grant(owners);
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.mark_domain_release_unconfirmed(kSession, kAllocation));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.finish_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, kViews, 2));
    EXPECT_THROW(owners.borrow(1, kViews, 2), std::runtime_error);
    EXPECT_THROW(owners.record(at(0x8000), 128), std::runtime_error);
    EXPECT_EQ(owners.allocation_count(), 2u);
}

TEST(DeviceBufferOwners, NoOtherContextInheritsAnAllocationGrant) {
    DeviceBufferOwners first;
    DeviceBufferOwners second;
    grant(first);
    EXPECT_FALSE(second.borrow(1, kViews, 2));
    EXPECT_FALSE(second.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(first.borrow(1, kViews, 2));
}

TEST(DeviceBufferOwners, InvalidAndDuplicateRegistrationsLeaveOwnersUnchanged) {
    DeviceBufferOwners owners;
    EXPECT_FALSE(owners.register_domain_owner(0, kAllocation, kViews, 2));
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, nullptr, 0));
    const DeviceBufferOwners::Span bad[] = {{0x8000, 0}, {0, 128}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, bad, 2));
    const DeviceBufferOwners::Span overflow[] = {{std::numeric_limits<uint64_t>::max() - 8, 16}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, overflow, 1));
    EXPECT_TRUE(owners.register_domain_owner(kSession, 0, kViews, 2));
    EXPECT_FALSE(owners.register_domain_owner(kSession, 0, kViews, 2));
}

TEST(DeviceBufferOwners, SessionCloseRefusesLoansAndReservesAllOwnersTogether) {
    DeviceBufferOwners owners;
    grant(owners);
    const DeviceBufferOwners::Span other[] = {{0xC000, 128}};
    ASSERT_TRUE(owners.register_domain_owner(kSession, 22, other, 1));
    ASSERT_TRUE(owners.borrow(1, other, 1));
    EXPECT_FALSE(owners.begin_domain_session_release(kSession));
    EXPECT_THROW(owners.borrow(2, kViews, 2), std::runtime_error);
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.finish_domain_session_release(kSession, true));
    owners.release(1, false);
    ASSERT_TRUE(owners.begin_domain_session_release(kSession));
    EXPECT_FALSE(owners.begin_domain_session_release(kSession));
    EXPECT_FALSE(owners.begin_domain_release(kSession, 22));
    const DeviceBufferOwners::Span late[] = {{0xD000, 128}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, 23, late, 1));
    ASSERT_TRUE(owners.finish_domain_session_release(kSession, true));
    EXPECT_EQ(owners.allocation_count(), 0u);
    EXPECT_FALSE(owners.has_domain_owner(kSession, 22));
}

TEST(DeviceBufferOwners, UnconfirmedSessionCloseQuarantinesEveryView) {
    DeviceBufferOwners owners;
    grant(owners);
    ASSERT_TRUE(owners.begin_domain_session_release(kSession));
    ASSERT_TRUE(owners.finish_domain_session_release(kSession, false));
    EXPECT_TRUE(owners.has_domain_owner(kSession, kAllocation));
    EXPECT_THROW(owners.borrow(1, kViews, 2), std::runtime_error);
    EXPECT_FALSE(owners.begin_domain_session_release(kSession));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_THROW(owners.record(at(0x8000), 4096), std::runtime_error);
}

TEST(DeviceBufferOwners, SessionCloseLeavesOtherSessionsAndCallerStorageUntouched) {
    DeviceBufferOwners owners;
    grant(owners);
    const DeviceBufferOwners::Span other[] = {{0xC000, 128}};
    ASSERT_TRUE(owners.register_domain_owner(99, 1, other, 1));
    owners.record(at(0x2000), 128);
    ASSERT_TRUE(owners.begin_domain_session_release(kSession));
    ASSERT_TRUE(owners.finish_domain_session_release(kSession, true));
    EXPECT_TRUE(owners.borrow(1, other, 1));
    EXPECT_TRUE(owners.recorded(at(0x2000)));
    EXPECT_TRUE(owners.has_domain_owner(99, 1));
    EXPECT_EQ(owners.allocation_count(), 2u);
}

TEST(DeviceBufferOwners, SessionCloseDoesNotStrandAnIndividuallyReservedOwner) {
    DeviceBufferOwners owners;
    grant(owners);
    ASSERT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.begin_domain_session_release(kSession));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
    ASSERT_TRUE(owners.begin_domain_session_release(kSession));
    const DeviceBufferOwners::Span late[] = {{0xD000, 128}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, 23, late, 1));
    EXPECT_TRUE(owners.finish_domain_session_release(kSession, true));
}

TEST(DeviceBufferOwners, EmptySpansDoNotRetainOrReacquireDomainGrants) {
    DeviceBufferOwners owners;
    grant(owners);
    const DeviceBufferOwners::Span empty[] = {{0x8080, 0}};
    EXPECT_FALSE(owners.borrow(1, empty, 1));
    EXPECT_TRUE(owners.declare_writes(1, empty, 1));
    ASSERT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.borrow(2, empty, 1));
    EXPECT_TRUE(owners.declare_writes(2, empty, 1));
    EXPECT_FALSE(owners.written_by_other_run(2, 0x8080, 0));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}
