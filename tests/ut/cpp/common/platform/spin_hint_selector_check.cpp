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

// Compiled once per {arch} x {variant} include path (see CMakeLists.txt), with
// EXPECTED_TENSOR_DATA_WAIT_TIMEOUT_MS defined to that variant's own
// platform_config.h default. An alias in spin_hint.h that points at the wrong
// variant's default fails this build.

#include "spin_hint.h"

static_assert(
    PLATFORM_TENSOR_DATA_WAIT_TIMEOUT_MS == EXPECTED_TENSOR_DATA_WAIT_TIMEOUT_MS,
    "spin_hint.h must alias PLATFORM_TENSOR_DATA_WAIT_TIMEOUT_MS to its own platform variant's default"
);
