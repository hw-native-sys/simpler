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
 * Negative AIV kernel: PTO2_ERROR_ASYNC_WAIT_OVERFLOW (code 102).
 *
 * Every dispatched task gets a valid async context (capacity =
 * MAX_COMPLETIONS_PER_TASK). Registering one condition past capacity makes
 * register_completion_condition write ASYNC_WAIT_OVERFLOW into the deferred
 * slab; the AICPU FIN thread latches it to the host-visible sched_error_code.
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
    // The (capacity + 1)-th registration overflows and latches code 102.
    for (int32_t i = 0; i <= MAX_COMPLETIONS_PER_TASK; i++) {
        CompletionToken token{
            static_cast<uint64_t>(0x1000 + i * 64), static_cast<uint32_t>(i), COMPLETION_ENGINE_SDMA,
            COMPLETION_TYPE_COUNTER, 0
        };
        register_completion_condition(ctx, token);
    }
    pto2::detail::defer_flush(ctx);
}
