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

// Framework-internal TLS bridge. The executor binds the current thread's
// runtime before invoking the orchestration entry, so orchestration helpers can
// fetch the current PTO2Runtime without explicit parameter threading. Declared
// here (rather than in pto_orchestration_api.h) so framework TUs the AICore
// build also compiles — notably orchestration/common.cpp — see these symbols
// without pulling in pto_types.h, whose Arg::add_scalar → to_u64 path is
// __aicore__-only and would break the ccec build.
//
// framework_bind_runtime / framework_current_runtime are called from BOTH
// __host__ code (AICPU executor, orchestration .so) and __aicore__ code
// (dist_core_main_impl binds before replay; the replayed orchestration reads
// current_runtime). Attribute them __aicore__ so CCEC permits the AICore call;
// on non-CCEC builds __aicore__/__gm__ are no-ops (defined empty below).
//
// The rt pointer lives in GM on the AICore (gd->rt is __gm__ PTO2Runtime*), so
// the bridge parameter/return type carries __gm__. On host/AICPU builds __gm__
// is empty → the signature collapses to plain PTO2Runtime*, matching the
// existing host callers (aicpu_executor.cpp line 77, orchestration .so).
#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifdef __cplusplus
extern "C" {
#endif
struct PTO2Runtime;
__aicore__ __gm__ PTO2Runtime *framework_current_runtime(void);
__aicore__ void framework_bind_runtime(__gm__ PTO2Runtime *rt);
#ifdef __cplusplus
}
#endif
