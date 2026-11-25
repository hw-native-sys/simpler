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

/** Immutable kernel argument vector, shared by every execution of one task. */

#pragma once

#include <cstdint>
#include <type_traits>

#include "kernel_args.h"

struct alignas(64) DispatchPayload {
    uint64_t args[MAX_TENSOR_ARGS + MAX_SCALAR_ARGS];
};

static_assert(sizeof(DispatchPayload) == 384);
static_assert(std::is_standard_layout_v<DispatchPayload> && std::is_trivially_copyable_v<DispatchPayload>);
