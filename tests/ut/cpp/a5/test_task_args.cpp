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

#include "pto_orchestration_api.h"

TEST(A5TaskArgs, EarlyResolveHintRoundTripsAndResets) {
    L0TaskArgs args;

    EXPECT_FALSE(args.allow_early_resolve());
    args.set_allow_early_resolve();
    EXPECT_TRUE(args.allow_early_resolve());

    args.set_allow_early_resolve(false);
    EXPECT_FALSE(args.allow_early_resolve());

    args.set_allow_early_resolve(true);
    args.reset();
    EXPECT_FALSE(args.allow_early_resolve());
}

TEST(A5TaskArgs, EarlyResolveHintIsExposedWithDeps) {
    L0TaskArgsWithDeps<> args;

    EXPECT_FALSE(args.allow_early_resolve());
    args.set_allow_early_resolve(true);
    EXPECT_TRUE(args.allow_early_resolve());

    args.reset();
    EXPECT_FALSE(args.allow_early_resolve());
}
