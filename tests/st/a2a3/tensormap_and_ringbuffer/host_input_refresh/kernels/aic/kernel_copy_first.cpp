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
#include <pto/pto-inst.hpp>

#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#include "intrinsic.h"

#ifdef PTO_CPUSTUB_HPP
#define dcci(...) \
    do {          \
    } while (0)
#endif
#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ ChipTensor *ids_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[0]);
    __gm__ ChipTensor *activation_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[1]);
    __gm__ ChipTensor *output_id_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[2]);
    __gm__ ChipTensor *output_activation_tensor = reinterpret_cast<__gm__ ChipTensor *>(args[3]);

    __gm__ int64_t *ids = reinterpret_cast<__gm__ int64_t *>(ids_tensor->buffer.addr) + ids_tensor->start_offset;
    __gm__ float *activation =
        reinterpret_cast<__gm__ float *>(activation_tensor->buffer.addr) + activation_tensor->start_offset;
    __gm__ int64_t *output_id =
        reinterpret_cast<__gm__ int64_t *>(output_id_tensor->buffer.addr) + output_id_tensor->start_offset;
    __gm__ float *output_activation = reinterpret_cast<__gm__ float *>(output_activation_tensor->buffer.addr) +
                                      output_activation_tensor->start_offset;

    output_id[0] = ids[0];
    output_activation[0] = activation[0];
    dcci(&output_id[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci(&output_activation[0], SINGLE_CACHE_LINE, CACHELINE_OUT);
}
