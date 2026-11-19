// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// BF16 NZ LM-head vocabulary projection.
#include "bf16_projection.hpp"

using namespace bf16_projection;

template <int Vocabulary>
__aicore__ inline void project_head(__gm__ int64_t* args) {
    auto hidden = reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto output = reinterpret_cast<__gm__ Tensor*>(args[2]);
    StoreOutput<float> emit{data<float>(args, 2), output->strides[0]};
    project<5120, Vocabulary, 512, 128, 32, 3, Layout::NZ>(
        data<BFloat>(args, 0), data<BFloat>(args, 1), args[3], hidden->shapes[0],
        hidden->strides[0], 0, args[5], get_block_idx(args), get_block_num(args), emit);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    pipe_barrier(PIPE_ALL);
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
    set_atomic_none();
#endif
    if (args[4] == 129280) project_head<129280>(args);
    else if (args[4] == 64640) project_head<64640>(args);
    else if (args[4] == 32320) project_head<32320>(args);
    else project_head<16160>(args);
}
