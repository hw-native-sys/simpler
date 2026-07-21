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
 * Negative AIV kernel: PTO2_ERROR_ASYNC_COMPLETION_INVALID (code 101).
 *
 * Defers an ASYNC_COMPLETION_INVALID directly into the deferred slab — the same
 * channel the SDMA backend uses when it sees a non-SDMA engine. The AICPU FIN
 * thread latches it to the host-visible sched_error_code. Using defer_error
 * keeps the trigger single-device (no SDMA/HCCL hardware required).
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

#include "pto_async_kernel_api.h"

using namespace pto;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    AsyncCtx ctx = get_async_ctx(args);
    pto2::detail::defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
    pto2::detail::defer_flush(ctx);
}
