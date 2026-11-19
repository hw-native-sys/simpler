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

#include <cstdint>
#include <type_traits>

// The header and the image bytes that follow it live in RTS argument storage.
// The destination is the aligned arena shared by the executing device roles.
struct HbgExecutionImage {
    uint64_t working_address;
    uint64_t bytes;
};
static_assert(std::is_trivially_copyable_v<HbgExecutionImage> && std::is_standard_layout_v<HbgExecutionImage>);
static_assert(sizeof(HbgExecutionImage) == 16);
