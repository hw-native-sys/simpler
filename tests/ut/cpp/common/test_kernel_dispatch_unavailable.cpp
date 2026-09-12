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
#include "kernel_dispatch_args.h"
#include "kernel_callable_residency.h"

TEST(KernelDispatchUnavailable, ProductionConsumerDoesNotReportExecutionSuccess) {
    KernelCallableDeviceResidency resident{17, 0x100000, 128, 3, 0};
    SimplerKernelDispatchArgs packet{};
    packet.packet_bytes = sizeof(packet);
    packet.residency_address = reinterpret_cast<uint64_t>(&resident);
    packet.invocation.mode = SIMPLER_MODE_KERNEL;
    packet.invocation.callable_id = 3;
    packet.invocation.generation = 17;
    EXPECT_EQ(simpler_aicpu_kernel_exec(&packet), static_cast<int>(KernelDispatchStatus::UnsupportedPayload));
    packet.invocation.generation = 16;
    EXPECT_EQ(simpler_aicpu_kernel_exec(&packet), static_cast<int>(KernelDispatchStatus::Stale));
}
