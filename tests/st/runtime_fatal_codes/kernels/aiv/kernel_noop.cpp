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

/**
 * Minimal no-op AIV kernel. Used by require_sync_start_orch.cpp: that case
 * latches PTO2_ERROR_REQUIRE_SYNC_START_INVALID (code 7) at submit time, before
 * any task is dispatched, so this body never actually runs — it only needs to
 * exist as a valid child callable for the kernel-submit path.
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) { (void)args; }
