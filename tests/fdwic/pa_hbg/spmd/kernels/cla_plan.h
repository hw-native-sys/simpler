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

namespace cla_contract {
constexpr uint32_t kMaxDependencies = 12;
constexpr uint32_t kPlanWidth = 32;
constexpr uint32_t kDependencyOffset = 8;
constexpr uint32_t kCheckOffset = 20;

enum class Field : uint32_t {
    Blocks = 0,
    Mask = 1,
    Base = 2,
    Tag = 3,
    Delay = 4,
    DependencyCount = 5,
    CheckCount = 6,
    DelayedBlock = 7
};
static_assert(kCheckOffset + kMaxDependencies == kPlanWidth);
}  // namespace cla_contract
