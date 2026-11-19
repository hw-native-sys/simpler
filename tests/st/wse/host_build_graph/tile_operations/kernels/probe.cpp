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
#include <wse/tile_ops.h>
#include "tensor.h"
#include "intrinsic.h"
#include <dlfcn.h>
#include <cstdlib>
extern "C" void kernel_entry(int64_t *args) {
    const auto block = get_block_idx(args) * get_thread_num(args) + get_thread_idx(args);
    auto pointer = [&](int i) {
        auto *t = reinterpret_cast<Tensor *>(args[i]);
        return reinterpret_cast<float *>(t->buffer.addr) + t->start_offset;
    };
    auto hook = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "wse_independent_probe"));
    const bool gated = args[3] != 0;
    if (gated && !hook) std::abort();
    if (gated && block == 0) hook(0);
    wse::Tile<2, 2> a, b, c;
    wse::TLOAD(a, {pointer(0), 4, 2, 2, 2, 1});
    wse::TLOAD(b, {pointer(0), 4, 2, 2, 2, 1});
    wse::TMATMUL(c, a, b);
    wse::TRELU(c, c);
    wse::TSTORE({pointer(1) + block * 4, 4, 2, 2, 2, 1}, c);
    pointer(2)[block * 2] = get_core_id(args);
    pointer(2)[block * 2 + 1] = get_pe_id(args);
    if (gated && block > 0 && block <= 4) hook(block);
}
