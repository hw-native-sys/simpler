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

#include <cstddef>

/**
 * Fault injection for the test-local `profiling_copy` implementation.
 *
 * A test arms one device-to-host copy to fail, identified by its transfer size:
 * the collector's two reads are a bank-sized one and a region-sized one, and no
 * other transfer in these tests shares either size. Everything else copies for
 * real between the separate host shadow and device allocation.
 */
namespace copy_fault {

struct Plan {
    // Transfer size, in bytes, whose device-to-host copy must fail. 0 disables.
    size_t fail_from_device_size{0};
    int fail_rc{-1};
};

struct Counts {
    int from_device_calls{0};
    int from_device_failures{0};
};

void arm(const Plan &plan);
void reset();
Counts counts();

}  // namespace copy_fault
