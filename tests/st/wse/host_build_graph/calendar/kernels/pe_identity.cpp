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
#include "intrinsic.h"
#include "tensor.h"

extern "C" void kernel_entry(int64_t *args) {
    const auto &output = *reinterpret_cast<const Tensor *>(args[0]);
    const int pe = get_pe_id();
    auto *record = reinterpret_cast<int32_t *>(output.buffer.addr) + output.start_offset + pe * 2;
    record[0] = get_core_id();
    record[1] = pe;
}
