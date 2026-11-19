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

#include "kernel_operator.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> in;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out;
    pipe.InitBuffer(in, 1, 1024);
    pipe.InitBuffer(out, 1, 1024);
    AscendC::GlobalTensor<float> memory;
    memory.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(tensor->buffer.addr) + tensor->start_offset);
    auto src = in.AllocTensor<float>();
    AscendC::DataCopy(src, memory, 256);
    in.EnQue(src);
    src = in.DeQue<float>();
    auto dst = out.AllocTensor<float>();
    AscendC::Adds(dst, src, 1.0f, 256);
    out.EnQue(dst);
    dst = out.DeQue<float>();
    AscendC::DataCopy(memory, dst, 256);
    in.FreeTensor(src);
    out.FreeTensor(dst);
    AscendC::PipeBarrier<PIPE_ALL>();
}
