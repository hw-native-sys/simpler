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

#include <stdlib.h>

// Framework-internal bridge. The host binds the current orchestration context
// before invoking the orchestration entry, so orchestration helpers can fetch it
// without explicit parameter threading. Declared here (rather than in
// orchestration_api.h) so framework TUs the AICore build also compiles — notably
// orchestration/common.cpp — see these symbols without pulling in types.h, whose
// Arg::add_scalar → to_u64 path is __aicore__-only and would break the ccec build.
#ifdef __cplusplus
extern "C" {
#endif
struct OrchContext;
OrchContext *framework_current_orch_context(void);
void framework_bind_orch_context(OrchContext *ctx);
#ifdef __cplusplus
}
#endif
