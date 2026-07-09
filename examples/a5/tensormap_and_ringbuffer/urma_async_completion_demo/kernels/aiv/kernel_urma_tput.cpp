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

namespace {

constexpr uint32_t kElemCount = 1024;

struct SimpleGlobalFloat {
    using RawDType = float;
    static constexpr pto::Layout layout = pto::Layout::ND;

    __gm__ float *ptr;

    inline __aicore__ explicit SimpleGlobalFloat(__gm__ float *data) :
        ptr(data) {}
    inline __aicore__ __gm__ float *data() { return ptr; }
    inline __aicore__ int64_t GetShape(int dim) {
        return dim == pto::GlobalTensorDim::DIM_4 ? static_cast<int64_t>(kElemCount) : 1;
    }
    inline __aicore__ int64_t GetStride(int dim) {
        return dim == pto::GlobalTensorDim::DIM_4 ? 1 : static_cast<int64_t>(kElemCount);
    }
};

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *dst_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *workspace_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    uint32_t remote_rank = static_cast<uint32_t>(args[4]);

    __gm__ float *src = reinterpret_cast<__gm__ float *>(src_tensor->buffer.addr) + src_tensor->start_offset;
    __gm__ float *dst = reinterpret_cast<__gm__ float *>(dst_tensor->buffer.addr) + dst_tensor->start_offset;
    __gm__ uint8_t *workspace =
        reinterpret_cast<__gm__ uint8_t *>(workspace_tensor->buffer.addr) + workspace_tensor->start_offset;

    SimpleGlobalFloat src_global(src);
    SimpleGlobalFloat dst_global(dst);
    AsyncCtx async_ctx = get_async_ctx(args);
    (void)send_request_entry(async_ctx, UrmaTput(dst_global, src_global, workspace, remote_rank));
}
