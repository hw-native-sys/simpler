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
#include <stdexcept>
extern "C" void kernel_entry(int64_t *args) {
    auto &tensor = *reinterpret_cast<Tensor *>(args[0]);
    auto *record = reinterpret_cast<uint64_t *>(tensor.buffer.addr) + tensor.start_offset + get_thread_idx(args) * 8;
    const uint64_t stage = args[1], token = args[2];
    if (stage && (record[0] != 0 || record[6] != token)) throw std::runtime_error("Graph predecessor was not complete");
    record[0] = stage;
    record[1] = get_block_idx(args);
    record[2] = get_thread_idx(args);
    record[3] = get_thread_num(args);
    record[4] = get_core_id(args);
    record[5] = get_pe_id(args);
    record[6] = token;
    record[7] = reinterpret_cast<uint64_t>(record);
}
