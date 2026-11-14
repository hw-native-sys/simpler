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

/** Fixed kernel argument vector and invocation-context suffix. */

#pragma once

#include <stdint.h>

#include "arg_direction.h"
#include "intrinsic.h"

inline constexpr uint32_t DISPATCH_MAX_ARGS = MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + EXT_PARAMS_COUNT;
inline constexpr uint32_t TASK_IMAGE_TENSOR_STRIDE = 128;

static_assert(MAX_TENSOR_ARGS + MAX_SCALAR_ARGS == SPMD_LOCAL_CONTEXT_INDEX);
static_assert(MAX_TENSOR_ARGS + MAX_SCALAR_ARGS + 1 == SPMD_GLOBAL_CONTEXT_INDEX);
