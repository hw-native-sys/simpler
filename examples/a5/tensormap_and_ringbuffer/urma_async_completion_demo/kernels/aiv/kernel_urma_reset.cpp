/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__
#endif

#include "backend/urma/urma_completion_kernel.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *workspace_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *token_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(workspace_tensor->buffer.addr) + workspace_tensor->start_offset;
    __gm__ int32_t *token = reinterpret_cast<__gm__ int32_t *>(token_tensor->buffer.addr) + token_tensor->start_offset;

    UrmaFakeReset(workspace);
    token[0] = 1;
}
