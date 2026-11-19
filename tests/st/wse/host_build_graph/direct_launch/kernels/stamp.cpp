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
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "intrinsic.h"
extern "C" void kernel_entry(int64_t *args) {
    auto *tensor = reinterpret_cast<Tensor *>(args[0]);
    auto *out = reinterpret_cast<float *>(tensor->buffer.addr) + tensor->start_offset;
    const int block = get_block_idx(args);
    for (int region = 0; region < 3; ++region)
        out[(block * 3 + region) * 16] = float(block);
}
