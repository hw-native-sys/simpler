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
#include "arg_direction.h"
#include "intrinsic.h"

// Byte offsets in the shared HBG TaskPayload used by graph argument readers.
constexpr uint32_t TASKPAYLOAD_TENSOR_COUNT_OFFSET = 0;
constexpr uint32_t TASKPAYLOAD_SCALAR_COUNT_OFFSET = 4;
// Offsets of the fields that NAME the argument regions, not of the regions themselves:
// a region's address is (src + <field offset>) + <the int32 delta stored there>.
constexpr uint32_t TASKPAYLOAD_TENSORS_DELTA_OFFSET = 12;
constexpr uint32_t TASKPAYLOAD_SCALARS_DELTA_OFFSET = 16;
constexpr uint32_t TASKPAYLOAD_FANIN_DELTA_OFFSET = 20;
constexpr uint32_t TASKPAYLOAD_TENSOR_STRIDE = 128;  // sizeof(simpler::hbg::Tensor)
