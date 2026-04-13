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
 * No-op AIV Kernel for Dispatch Throughput
 *
 * Minimal vector kernel that writes a single scalar to prove execution.
 * The kernel reads the current accumulated value, adds 1.0, and writes back.
 * With N tasks, the final output should be N.0.
 *
 * Args:
 *   args[0] = output tensor (INOUT) - single float32 element
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

using namespace pto;  // NOLINT(build/namespaces)

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ float* out = reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;
    *out = *out + 1.0f;
}
