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
// simpler::tmr::Tensor (src/common/task_interface/tensor.h) can use them without depending
// on this runtime-specific header. assert_impl / get_stacktrace are defined in
// orchestration/common.cpp for runtime targets.
#include "assert_compat.h"

// Framework-internal TLS bridge. The executor binds the current thread's
// runtime before invoking the orchestration entry, so orchestration helpers can
// fetch the current RuntimeContext without explicit parameter threading. Declared
// here (rather than in orchestration_api.h) so framework TUs the AICore
// build also compiles — notably orchestration/common.cpp — see these symbols
// without pulling in types.h, whose Arg::add_scalar → to_u64 path is
// __aicore__-only and would break the ccec build.
#ifdef __cplusplus
extern "C" {
#endif
struct RuntimeContext;
struct BlockingWaitOps;
RuntimeContext *framework_current_runtime(void);
void framework_bind_runtime(RuntimeContext *rt);
const BlockingWaitOps *framework_current_blocking_wait_ops(void);
void framework_bind_runtime_with_wait(RuntimeContext *rt, const BlockingWaitOps *ops);
#ifdef __cplusplus
}
#endif
