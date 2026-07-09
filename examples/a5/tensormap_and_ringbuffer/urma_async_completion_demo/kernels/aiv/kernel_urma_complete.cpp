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
    uint32_t remote_rank = static_cast<uint32_t>(args[2]);
    uint32_t target_head = static_cast<uint32_t>(args[3]);
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(workspace_tensor->buffer.addr) + workspace_tensor->start_offset;

    auto *ws = pto2::urma_backend::fake_workspace(workspace);
    pto2::urma_backend::ensure_fake_workspace_initialized(workspace);
    while (__atomic_load_n(&ws->sq_head[remote_rank], __ATOMIC_ACQUIRE) < target_head) {}
    (void)UrmaFakeComplete(workspace, remote_rank, target_head);
}
