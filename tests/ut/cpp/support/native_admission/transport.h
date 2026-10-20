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
// Allocation and transport faults; all admission decisions stay in production code.
#pragma once
#include <cstddef>
#include <limits>
namespace native_admission_transport {
struct Control {
    size_t ceiling{std::numeric_limits<size_t>::max()};
    size_t bytes{0};
    size_t allocations{0};
    size_t successful_allocations{0};
    size_t fail_allocation{0};
    int attach_rc{0};
    int copy_rc{0};
    size_t stream_create_calls{0};
    size_t fail_stream_create{0};
};
Control &control();
}  // namespace native_admission_transport
