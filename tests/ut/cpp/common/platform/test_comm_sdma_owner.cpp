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
#include "platform_comm/comm_sdma_owner.h"

TEST(CommSdmaOwner, ExplicitPartitionIsFixedAcrossDifferentDomainUsers) {
    CommSdmaOwnerConfiguration owner;
    EXPECT_TRUE(owner.bind(CommSdmaAccess::SDK_ONLY, 3));
    EXPECT_TRUE(owner.bind(CommSdmaAccess::SDK_ONLY, 3));
    EXPECT_FALSE(owner.bind(CommSdmaAccess::SDK_ONLY, 1));
    EXPECT_FALSE(owner.bind(CommSdmaAccess::LEGACY, 0));
    EXPECT_EQ(owner.queue_num, 3u);
}

TEST(CommSdmaOwner, LegacyExposureCannotBecomeAnInferredDefaultPartition) {
    CommSdmaOwnerConfiguration owner;
    EXPECT_TRUE(owner.bind(CommSdmaAccess::LEGACY, 0));
    EXPECT_TRUE(owner.bind(CommSdmaAccess::LEGACY, 0));
    EXPECT_FALSE(owner.bind(CommSdmaAccess::SDK_ONLY, 1));
    EXPECT_EQ(owner.queue_num, 0u);
}

TEST(CommSdmaOwner, InvalidPartitionDoesNotChangeTheOwner) {
    CommSdmaOwnerConfiguration owner;
    for (uint32_t q : {0u, 49u, UINT32_MAX})
        EXPECT_FALSE(owner.bind(CommSdmaAccess::SDK_ONLY, q));
    EXPECT_FALSE(owner.bind(CommSdmaAccess::UNKNOWN, 1));
    EXPECT_FALSE(owner.bind(CommSdmaAccess::LEGACY, 1));
    EXPECT_EQ(owner.access, CommSdmaAccess::UNKNOWN);
    EXPECT_TRUE(owner.bind(CommSdmaAccess::SDK_ONLY, 48));
}

TEST(CommSdmaOwner, OneRejectedRankRejectsBothSidesBeforeTheNextBarrier) {
    CommSdmaOwnerConfiguration old_owner;
    ASSERT_TRUE(old_owner.bind(CommSdmaAccess::LEGACY, 0));
    CommSdmaAdmissionV1 ranks[2];
    for (auto &rank : ranks) {
        rank.access = CommSdmaAccess::SDK_ONLY;
        rank.queue_num = 2;
    }
    ranks[1].status = old_owner.accepts(ranks[1].access, ranks[1].queue_num) ? 0 : -1;
    // These are the same records each rank publishes/reads in the existing
    // domain announce exchange. Neither may advance to p2p_ready/imports.
    EXPECT_FALSE(comm_sdma_admissions_agree(ranks[0], ranks[1]));
    EXPECT_FALSE(comm_sdma_admissions_agree(ranks[1], ranks[0]));
}

TEST(CommSdmaOwner, AnnounceMustAgreeOnExplicitVersionAccessAndPartition) {
    CommSdmaAdmissionV1 a;
    a.access = CommSdmaAccess::SDK_ONLY;
    a.queue_num = 1;
    auto b = a;
    EXPECT_TRUE(comm_sdma_admissions_agree(a, b));
    b.queue_num = 2;
    EXPECT_FALSE(comm_sdma_admissions_agree(a, b));
    b = a;
    b.version = 2;
    EXPECT_FALSE(comm_sdma_admissions_agree(a, b));
    b = a;
    b.magic = 0;
    EXPECT_FALSE(comm_sdma_admissions_agree(a, b));
    b = a;
    b.access = CommSdmaAccess::LEGACY;
    b.queue_num = 0;
    EXPECT_FALSE(comm_sdma_admissions_agree(a, b));
}

TEST(CommSdmaOwner, InfoDoesNotTurnUnknownOrInvalidRangesIntoQualification) {
    EXPECT_TRUE(comm_sdma_owner_info_valid({}));
    EXPECT_TRUE(comm_sdma_owner_info_valid({4096, 4096, CommSdmaAccess::SDK_ONLY, 1}));
    EXPECT_TRUE(comm_sdma_owner_info_valid({4096, 4096, CommSdmaAccess::LEGACY, 0}));
    EXPECT_FALSE(comm_sdma_owner_info_valid({4096, 4096, CommSdmaAccess::UNKNOWN, 1}));
    EXPECT_FALSE(comm_sdma_owner_info_valid({4096, 4096, CommSdmaAccess::SDK_ONLY, 0}));
    EXPECT_FALSE(comm_sdma_owner_info_valid({UINT64_MAX - 32, 64, CommSdmaAccess::SDK_ONLY, 1}));
}
