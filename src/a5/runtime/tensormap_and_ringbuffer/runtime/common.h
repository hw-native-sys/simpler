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

#include <stdio.h>
#include <stdlib.h>

// Assertion macros (always_assert / debug_assert), AssertionError, and the
// MAYBE_UNINITIALIZED diagnostics live in the shared header so the unified
// Tensor (src/common/task_interface/tensor.h) can use them without depending
// on this runtime-specific header. assert_impl / get_stacktrace are defined in
// orchestration/common.cpp for runtime targets.
#include "assert_compat.h"
