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

#pragma once

#include "urma_queue_fixture.h"
#include "platform_comm/urma_run_binding.h"

namespace simpler::test {

enum class UrmaFault { Finish, BadCqe, Timeout, HostReceipt };

// The ISA test transport posts to peer 1 from AIV 0 (physical queue row 4).
struct alignas(128) FaultDomain : UrmaQueueFixture {
    static constexpr unsigned row = 4;
    std::array<UrmaRowOwner, kRows> row_owners{};
    UrmaRunBinding binding{addr(), sizeof(info), reinterpret_cast<uint64_t>(row_owners.data()), 0, row, 2, 1};
    uint64_t posted_operand{};
    uint64_t lane_error{};
    uint32_t done_count{};
    bool queue_active{};
    int acquired{};
    int released{};
    int drain_checks{};
    int complete_on_check{};
    int check_allocation_failures{};

    void run_kernel(uint64_t operand, bool put, UrmaFault fault);
};
}  // namespace simpler::test
