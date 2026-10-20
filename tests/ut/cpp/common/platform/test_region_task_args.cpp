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

#include "common/endpoint_failure_identity.h"
#include "common/region_task_args.h"

#include <gtest/gtest.h>

namespace {

uint64_t golden[8] = {
    REGION_TASK_MAGIC_VERSION, 1, 0x1000, 64, 0x2000, 128, 0, 7,
};

void expect_cleared(
    const RegionPartLocalSpan &payload, const RegionPartLocalSpan &counter, uint64_t session, uint64_t transaction,
    uint64_t region
) {
    EXPECT_EQ(payload.base, 0u);
    EXPECT_EQ(payload.logical_bytes, 0u);
    EXPECT_EQ(counter.base, 0u);
    EXPECT_EQ(counter.logical_bytes, 0u);
    EXPECT_EQ(session, 0u);
    EXPECT_EQ(transaction, 0u);
    EXPECT_EQ(region, 0u);
}

}  // namespace

TEST(RegionTaskArgs, DecodesEightFieldGolden) {
    RegionPartLocalSpan payload{};
    RegionPartLocalSpan counter{};
    uint64_t session = 1;
    uint64_t transaction = 1;
    uint64_t region = 1;
    RegionTaskArgError error = RegionTaskArgError::NULL_POINTER;
    ASSERT_TRUE(region_decode_task_args(golden, 8, payload, counter, session, transaction, region, error));
    EXPECT_EQ(error, RegionTaskArgError::OK);
    EXPECT_EQ(payload.base, 0x1000u);
    EXPECT_EQ(payload.logical_bytes, 64u);
    EXPECT_EQ(counter.base, 0x2000u);
    EXPECT_EQ(counter.logical_bytes, 128u);
    EXPECT_EQ(session, 0u);
    EXPECT_EQ(transaction, 7u);
    EXPECT_EQ(region, 1u);
}

TEST(RegionTaskArgs, RejectsBadInputAndClearsOutputs) {
    RegionPartLocalSpan payload{9, 9};
    RegionPartLocalSpan counter{9, 9};
    uint64_t session = 9;
    uint64_t transaction = 9;
    uint64_t region = 9;
    RegionTaskArgError error = RegionTaskArgError::OK;
    EXPECT_FALSE(region_decode_task_args(nullptr, 8, payload, counter, session, transaction, region, error));
    EXPECT_EQ(error, RegionTaskArgError::NULL_POINTER);
    expect_cleared(payload, counter, session, transaction, region);

    uint64_t six[6] = {};
    payload = {9, 9};
    EXPECT_FALSE(region_decode_task_args(six, 6, payload, counter, session, transaction, region, error));
    EXPECT_EQ(error, RegionTaskArgError::BAD_SCALAR_COUNT);
    expect_cleared(payload, counter, session, transaction, region);

    uint64_t bad = golden[0];
    uint64_t scalars[8];
    for (int i = 0; i < 8; ++i) {
        scalars[i] = golden[i];
    }
    scalars[0] = 0x4C334C3200030000ull;
    EXPECT_FALSE(region_decode_task_args(scalars, 8, payload, counter, session, transaction, region, error));
    EXPECT_EQ(error, RegionTaskArgError::BAD_MAGIC_VERSION);

    scalars[0] = bad;
    scalars[1] = 0;
    EXPECT_FALSE(region_decode_task_args(scalars, 8, payload, counter, session, transaction, region, error));
    EXPECT_EQ(error, RegionTaskArgError::BAD_REGION_ID);

    scalars[1] = 1;
    scalars[7] = 0;
    EXPECT_FALSE(region_decode_task_args(scalars, 8, payload, counter, session, transaction, region, error));
    EXPECT_EQ(error, RegionTaskArgError::BAD_TRANSACTION_ID);
    expect_cleared(payload, counter, session, transaction, region);
}

TEST(EndpointFailureIdentity, ChecksShape) {
    EndpointFailureIdentity empty{};
    EXPECT_TRUE(endpoint_failure_identity_is_well_formed(empty));
    EndpointFailureIdentity region{1, 0, 0, 7};
    EXPECT_TRUE(endpoint_failure_identity_is_well_formed(region));
    EndpointFailureIdentity untrusted{2, 0, 0, 0};
    EXPECT_TRUE(endpoint_failure_identity_is_well_formed(untrusted));
    EndpointFailureIdentity bad_reserved{1, 1, 0, 7};
    EXPECT_FALSE(endpoint_failure_identity_is_well_formed(bad_reserved));
    EndpointFailureIdentity bad_kind{3, 0, 0, 7};
    EXPECT_FALSE(endpoint_failure_identity_is_well_formed(bad_kind));
    EndpointFailureIdentity kind_zero{0, 0, 1, 0};
    EXPECT_FALSE(endpoint_failure_identity_is_well_formed(kind_zero));
}
