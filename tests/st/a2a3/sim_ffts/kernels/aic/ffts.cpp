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

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include "intrinsic.h"
#include "tensor.h"

extern "C" void kernel_entry(int64_t *args) {
    set_ffts_base_addr(0);
    auto *scratch = reinterpret_cast<Tensor *>(args[0]);
    const int block = get_block_idx(args);
    auto *data = reinterpret_cast<int32_t *>(scratch->buffer.addr) + scratch->start_offset + block * 4;
    for (int epoch = 1; epoch <= 7; ++epoch) {
        data[0] = block * 100 + epoch;
        __builtin_cce_ffts_cross_core_sync(PIPE_FIX, pto::getFFTSMsg(FFTS_MODE_VAL, 0));
        __builtin_cce_wait_flag_dev(1);
        data[3] = data[1] + data[2];
        ffts_cross_core_sync(PIPE_FIX, pto::getFFTSMsg(FFTS_MODE_VAL, 2));
    }
}
