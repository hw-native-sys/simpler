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

#include "common/ccu_execution_types.h"

TEST(CcuExecutionAbi, RecordsHaveStableVersionsAndExpectedDeviceLayouts) {
    EXPECT_EQ(CCU_EXECUTION_ABI_V1, 1U);
    EXPECT_EQ(sizeof(CcuRequestDescriptorV1), 128U);
    EXPECT_EQ(alignof(CcuRequestDescriptorV1), 128U);
    EXPECT_EQ(offsetof(CcuRequestDescriptorV1, request_seq), 32U);
    EXPECT_EQ(sizeof(CcuTriggerInfoV1), 128U);
    EXPECT_EQ(offsetof(CcuTriggerInfoV1, gate_va), 32U);
    EXPECT_EQ(sizeof(CcuCompletionRecordV1), 256U);
    EXPECT_EQ(offsetof(CcuCompletionRecordV1, completed_seq), 128U);
}
