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
// The orch-SO descriptor (device address/size + entry/config symbol names) is
// carried to the AICPU register entry in a RegisterCallableArgs POD, no longer
// stashed on Runtime. These tests pin that POD's shape — default-empty and a
// round-trip of the fields the host fills from CallableState in
// launch_device_register.

#include <cstdint>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "common/kernel_args.h"

TEST(RegisterCallableArgs, DefaultIsEmpty) {
    RegisterCallableArgs args{};
    EXPECT_EQ(args.active_callable_id, -1);
    EXPECT_EQ(args.dev_orch_so_addr, 0u);
    EXPECT_EQ(args.dev_orch_so_size, 0u);
    EXPECT_EQ(args.device_orch_func_name[0], '\0');
    EXPECT_EQ(args.device_orch_config_name[0], '\0');
}

TEST(RegisterCallableArgs, FieldsRoundTrip) {
    RegisterCallableArgs args{};
    args.active_callable_id = 3;
    args.dev_orch_so_addr = 0xdeadbeefULL;
    args.dev_orch_so_size = 4096;
    snprintf(args.device_orch_func_name, sizeof(args.device_orch_func_name), "%s", "orch_entry");
    snprintf(args.device_orch_config_name, sizeof(args.device_orch_config_name), "%s", "orch_config");

    EXPECT_EQ(args.active_callable_id, 3);
    EXPECT_EQ(args.dev_orch_so_addr, 0xdeadbeefULL);
    EXPECT_EQ(args.dev_orch_so_size, 4096u);
    EXPECT_STREQ(args.device_orch_func_name, "orch_entry");
    EXPECT_STREQ(args.device_orch_config_name, "orch_config");
}

TEST(RegisterCallableArgs, SymbolNamesAreBounded) {
    // The host fills the symbol names via snprintf with sizeof(field); a name at
    // or past the bound must stay NUL-terminated inside the fixed array.
    RegisterCallableArgs args{};
    std::string long_name(INIT_ARGS_MAX_ORCH_SYMBOL_NAME * 2, 'x');
    snprintf(args.device_orch_func_name, sizeof(args.device_orch_func_name), "%s", long_name.c_str());
    EXPECT_EQ(args.device_orch_func_name[INIT_ARGS_MAX_ORCH_SYMBOL_NAME - 1], '\0');
    EXPECT_EQ(std::strlen(args.device_orch_func_name), static_cast<size_t>(INIT_ARGS_MAX_ORCH_SYMBOL_NAME - 1));
}
