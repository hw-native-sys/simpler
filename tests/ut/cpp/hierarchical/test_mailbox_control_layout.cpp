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

#include "worker_manager.h"

TEST(MailboxControlLayout, UsesFourArgsAndMovesResultToOffset48) {
    EXPECT_EQ(CTRL_OFF_ARG0, 16);
    EXPECT_EQ(CTRL_OFF_ARG1, 24);
    EXPECT_EQ(CTRL_OFF_ARG2, 32);
    EXPECT_EQ(CTRL_OFF_ARG3, 40);
    EXPECT_EQ(CTRL_OFF_RESULT, 48);
}
