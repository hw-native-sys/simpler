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

#include "common/l3_l2_orch_comm.h"
#include "device_runner_base.h"
#include "host/l3_l2_orch_comm_service.h"
#include "pto_runtime_c_api.h"

namespace {

class UnsupportedL3L2Runner : public DeviceRunnerBase {
public:
    int run(Runtime &, int, int) override { return 0; }
    int finalize() override { return 0; }
    bool l3_l2_orch_comm_supported() const override { return false; }
};

TEST(L3L2OrchCommOnboardRunnerTest, UnsupportedRunnerInitFailsAndShutdownIsNoop) {
    UnsupportedL3L2Runner runner;
    L3L2OrchCommControlBlock control{};

    EXPECT_EQ(runner.l3_l2_orch_comm_init(&control, sizeof(control)), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(runner.l3_l2_orch_comm_shutdown(), 0);
}

}  // namespace
