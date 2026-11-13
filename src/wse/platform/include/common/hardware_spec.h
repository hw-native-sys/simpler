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

namespace wse {
// Supported implementation limits, independent of the selected device's defaults.
inline constexpr uint32_t MAX_CORE_COUNT = 64;
inline constexpr uint32_t PE_PER_CORE = 4;
inline constexpr uint64_t SRAM_ALIGNMENT = 128;

struct SramRegion {
    uint64_t base;
    uint64_t capacity;
    uint64_t reserve;
    uint64_t runtime_base;
};
struct SramAllocation {
    uint64_t base;
    uint64_t bytes;
    uint32_t owner;
    uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<SramRegion> && std::is_standard_layout_v<SramRegion>);
static_assert(std::is_trivially_copyable_v<SramAllocation> && std::is_standard_layout_v<SramAllocation>);
}  // namespace wse
