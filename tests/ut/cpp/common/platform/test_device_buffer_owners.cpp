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
#include "host/caller_device_buffers.h"

namespace {
constexpr uint64_t kSession = 7;
constexpr uint64_t kAllocation = 11;
const CallerDeviceBuffers::Span kViews[] = {{0x8000, 0x1000}, {0xA000, 0x100}};
void *at(uint64_t addr) { return reinterpret_cast<void *>(addr); }
void grant(CallerDeviceBuffers &owners) { ASSERT_TRUE(owners.register_domain_owner(kSession, kAllocation, kViews, 2)); }
}  // namespace

TEST(CallerDeviceBuffers, ExplicitDomainGrantIsNotACallerMint) {
    CallerDeviceBuffers owners;
    grant(owners);
    EXPECT_TRUE(owners.resolve(0x8080, 0x80, nullptr));
    EXPECT_FALSE(owners.recorded(at(0x8000)));
    EXPECT_EQ(owners.caller_allocation_count(), 0u);
    EXPECT_FALSE(owners.forget_if_unborrowed(at(0x8000)));
    const CallerDeviceBuffers::Span used[] = {{0x8080, 0x80}, {0xA000, 0x100}};
    EXPECT_TRUE(owners.borrow(1, used, 2));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(1, false);
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.resolve(0x8000, 1, nullptr));
    EXPECT_EQ(owners.allocation_count(), 0u);
}

TEST(CallerDeviceBuffers, BoundsDoNotExpandToCoverAnUnknownView) {
    CallerDeviceBuffers owners;
    grant(owners);
    const CallerDeviceBuffers::Span used[] = {{0x8FFF, 2}, {0x8000, 1}};
    EXPECT_FALSE(owners.borrow(1, used, 2));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(1, false);
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(CallerDeviceBuffers, CallerAndDomainLoansShareOneRunIdentity) {
    CallerDeviceBuffers owners;
    owners.record(at(0x2000), 128);
    grant(owners);
    EXPECT_EQ(owners.caller_allocation_count(), 1u);
    const CallerDeviceBuffers::Span used[] = {{0x2000, 128}, {0x8000, 64}, {0xA000, 32}};
    EXPECT_TRUE(owners.borrow(19, used, 3));
    EXPECT_FALSE(owners.forget_if_unborrowed(at(0x2000)));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release(19, false);
    EXPECT_TRUE(owners.forget_if_unborrowed(at(0x2000)));
    EXPECT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(CallerDeviceBuffers, RegistrationRejectsOverlapsWithoutPartialPublication) {
    CallerDeviceBuffers owners;
    owners.record(at(0x2000), 128);
    const CallerDeviceBuffers::Span used[] = {{0x8000, 256}, {0x2040, 64}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, used, 2));
    EXPECT_FALSE(owners.resolve(0x8000, 1, nullptr));
    EXPECT_EQ(owners.allocation_count(), 1u);
    const CallerDeviceBuffers::Span overlapping[] = {{0x8000, 256}, {0x8080, 256}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, overlapping, 2));
    EXPECT_FALSE(owners.resolve(0x8000, 1, nullptr));
}

TEST(CallerDeviceBuffers, ACommunicationHandleAndAllocationIdentifyTheOwner) {
    CallerDeviceBuffers owners;
    grant(owners);
    EXPECT_FALSE(owners.begin_domain_release(kSession + 1, kAllocation));
    EXPECT_FALSE(owners.finish_domain_release(kSession, kAllocation + 1));
    EXPECT_TRUE(owners.borrow(1, kViews, 2));
    owners.release(1, false);
    const CallerDeviceBuffers::Span elsewhere[] = {{0xC000, 128}};
    EXPECT_TRUE(owners.register_domain_owner(kSession + 1, kAllocation, elsewhere, 1));
    EXPECT_TRUE(owners.begin_domain_release(kSession + 1, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession + 1, kAllocation));
    EXPECT_TRUE(owners.resolve(0x8000, 1, nullptr));
}

TEST(CallerDeviceBuffers, RevokeRejectsNewUseBeforeTheLastConsumerRetires) {
    CallerDeviceBuffers owners;
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

TEST(CallerDeviceBuffers, EveryLiveConsumerKeepsTheWholeDomain) {
    CallerDeviceBuffers owners;
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

TEST(CallerDeviceBuffers, WriteDeclarationsFenceHostReadsIndependentlyOfBorrow) {
    CallerDeviceBuffers owners;
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

TEST(CallerDeviceBuffers, UnprovenConsumptionCannotBeDischargedByIdentityReuse) {
    CallerDeviceBuffers owners;
    grant(owners);
    EXPECT_TRUE(owners.borrow(1, kViews, 2));
    EXPECT_TRUE(owners.declare_writes(1, kViews, 1));
    owners.release(1, true);
    owners.release(1, false);
    EXPECT_TRUE(owners.written_by_other_run(1, 0x8000, 1));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(CallerDeviceBuffers, UnknownPlatformReleaseRetainsRevokedOwnership) {
    CallerDeviceBuffers owners;
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

TEST(CallerDeviceBuffers, NoOtherContextInheritsAnAllocationGrant) {
    CallerDeviceBuffers first;
    CallerDeviceBuffers second;
    grant(first);
    EXPECT_FALSE(second.borrow(1, kViews, 2));
    EXPECT_FALSE(second.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(first.borrow(1, kViews, 2));
}

TEST(CallerDeviceBuffers, InvalidAndDuplicateRegistrationsLeaveOwnersUnchanged) {
    CallerDeviceBuffers owners;
    EXPECT_FALSE(owners.register_domain_owner(0, kAllocation, kViews, 2));
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, nullptr, 0));
    const CallerDeviceBuffers::Span bad[] = {{0x8000, 0}, {0, 128}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, bad, 2));
    const CallerDeviceBuffers::Span overflow[] = {{std::numeric_limits<uint64_t>::max() - 8, 16}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, kAllocation, overflow, 1));
    EXPECT_TRUE(owners.register_domain_owner(kSession, 0, kViews, 2));
    EXPECT_FALSE(owners.register_domain_owner(kSession, 0, kViews, 2));
}

TEST(CallerDeviceBuffers, SessionCloseRefusesLoansAndReservesAllOwnersTogether) {
    CallerDeviceBuffers owners;
    grant(owners);
    const CallerDeviceBuffers::Span other[] = {{0xC000, 128}};
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
    const CallerDeviceBuffers::Span late[] = {{0xD000, 128}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, 23, late, 1));
    ASSERT_TRUE(owners.finish_domain_session_release(kSession, true));
    EXPECT_EQ(owners.allocation_count(), 0u);
    EXPECT_FALSE(owners.has_domain_owner(kSession, 22));
}

TEST(CallerDeviceBuffers, UnconfirmedSessionCloseQuarantinesEveryView) {
    CallerDeviceBuffers owners;
    grant(owners);
    ASSERT_TRUE(owners.begin_domain_session_release(kSession));
    ASSERT_TRUE(owners.finish_domain_session_release(kSession, false));
    EXPECT_TRUE(owners.has_domain_owner(kSession, kAllocation));
    EXPECT_THROW(owners.borrow(1, kViews, 2), std::runtime_error);
    EXPECT_FALSE(owners.begin_domain_session_release(kSession));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_THROW(owners.record(at(0x8000), 4096), std::runtime_error);
}

TEST(CallerDeviceBuffers, SessionCloseLeavesOtherSessionsAndCallerStorageUntouched) {
    CallerDeviceBuffers owners;
    grant(owners);
    const CallerDeviceBuffers::Span other[] = {{0xC000, 128}};
    ASSERT_TRUE(owners.register_domain_owner(99, 1, other, 1));
    owners.record(at(0x2000), 128);
    ASSERT_TRUE(owners.begin_domain_session_release(kSession));
    ASSERT_TRUE(owners.finish_domain_session_release(kSession, true));
    EXPECT_TRUE(owners.borrow(1, other, 1));
    EXPECT_TRUE(owners.recorded(at(0x2000)));
    EXPECT_TRUE(owners.has_domain_owner(99, 1));
    EXPECT_EQ(owners.allocation_count(), 2u);
}

TEST(CallerDeviceBuffers, SessionCloseDoesNotStrandAnIndividuallyReservedOwner) {
    CallerDeviceBuffers owners;
    grant(owners);
    ASSERT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.begin_domain_session_release(kSession));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
    ASSERT_TRUE(owners.begin_domain_session_release(kSession));
    const CallerDeviceBuffers::Span late[] = {{0xD000, 128}};
    EXPECT_FALSE(owners.register_domain_owner(kSession, 23, late, 1));
    EXPECT_TRUE(owners.finish_domain_session_release(kSession, true));
}

TEST(CallerDeviceBuffers, EmptySpansDoNotRetainOrReacquireDomainGrants) {
    CallerDeviceBuffers owners;
    grant(owners);
    const CallerDeviceBuffers::Span empty[] = {{0x8080, 0}};
    EXPECT_FALSE(owners.borrow(1, empty, 1));
    EXPECT_TRUE(owners.declare_writes(1, empty, 1));
    ASSERT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_FALSE(owners.borrow(2, empty, 1));
    EXPECT_TRUE(owners.declare_writes(2, empty, 1));
    EXPECT_FALSE(owners.written_by_other_run(2, 0x8080, 0));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}

TEST(CallerDeviceBuffers, CaptureReferenceBlocksCommunicatorReleaseAfterItsCarrierRetires) {
    CallerDeviceBuffers owners;
    grant(owners);
    ASSERT_TRUE(owners.declare_writes(1, kViews, 1));
    int captured = 0;
    ASSERT_TRUE(owners.retain_reference(&captured, 1));
    EXPECT_FALSE(owners.written_by_other_run(1, 0x8000, 1));
    owners.detach_reference(&captured);
    owners.release(1, false);
    EXPECT_TRUE(owners.written_by_other_run(1, 0x8000, 1));
    EXPECT_FALSE(owners.begin_domain_release(kSession, kAllocation));
    owners.release_reference(&captured, false);
    ASSERT_TRUE(owners.begin_domain_release(kSession, kAllocation));
    EXPECT_TRUE(owners.finish_domain_release(kSession, kAllocation));
}
