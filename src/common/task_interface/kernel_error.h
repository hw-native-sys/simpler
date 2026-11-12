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

#include <stdint.h>
#include "data_type.h"

enum class KernelErrorCode : int32_t {
    INVALID_OUTPUT = 105,
    INDEX_OUT_OF_RANGE = 106,
    NON_FINITE_VALUE = 107,
};

PTO_DEVICE_FUNC constexpr bool is_kernel_error(int32_t code) {
    return code == static_cast<int32_t>(KernelErrorCode::INVALID_OUTPUT) ||
           code == static_cast<int32_t>(KernelErrorCode::INDEX_OUT_OF_RANGE) ||
           code == static_cast<int32_t>(KernelErrorCode::NON_FINITE_VALUE);
}
