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
#include "persistent_args_fixture.h"

using namespace persistent_args_test;

TEST(PersistentKernelArgs, LeavesThePerCallableDispatchFieldsAtTheirSentinels) {
    FakeArgsOps ops;
    Runtime runtime;
    PersistentKernelArgs args;

    ASSERT_EQ(args.prepare_once(runtime, ops.table(), kDeviceId), 0);

    // The device image starts at offset 0 of Runtime under both variants, so
    // the uploaded bytes answer the fields the device-side code reads.
    const Runtime *const uploaded = reinterpret_cast<const Runtime *>(args.args().runtime_args);
    // A kernel launch binds no callable, so the reference to a callable's
    // function tables has to arrive naming nothing: the device-side guard then
    // short-circuits instead of dereferencing an address no table lives at.
    EXPECT_EQ(uploaded->dev.callable_table_addr_, 0u);
    EXPECT_EQ(uploaded->dev.callable_table_len_, 0u);
    // The host view of that table is host-only on both variants, so reading it
    // through `uploaded` would read past the block that was copied.
    EXPECT_EQ(runtime.get_function_bin_addr(0), 0u);
    // HBG keeps the active callable id in host-only state, beyond the image.
    EXPECT_EQ(runtime.get_active_callable_id(), -1);

    EXPECT_EQ(args.finalize_once(), 0);
}
