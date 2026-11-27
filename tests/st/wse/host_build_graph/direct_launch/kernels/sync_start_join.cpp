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
#include "tensor.h"

extern "C" void kernel_entry(int64_t *args) {
    const auto data = [](int64_t arg) {
        const auto &tensor = *reinterpret_cast<Tensor *>(arg);
        return reinterpret_cast<float *>(tensor.buffer.addr) + tensor.start_offset;
    };
    auto *members = data(args[1]);
    *data(args[2]) = *data(args[0]) + members[0] + members[1] + members[2] + members[3];
}
