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
#include <dlfcn.h>
#include <stdexcept>
#include "sram_args.h"
#include "tensor.h"

extern "C" void kernel_entry(int64_t *args) {
    const auto &tensor = *reinterpret_cast<Tensor *>(args[0]);
    auto *out = reinterpret_cast<float *>(tensor.buffer.addr) + tensor.start_offset;
    const bool member = args[1], sync = args[2];
    const int64_t probe = args[3];
    const int block = get_block_idx(args), thread = get_thread_idx(args);
    if (probe && thread == 0 && (!member || block == 1)) {
        const char *name = probe == 2 ? "wse_sync_failure_probe" :
                           sync       ? "wse_sync_start_probe" :
                                        "wse_independent_start_probe";
        auto gate = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, name));
        if (!gate) throw std::runtime_error("missing SPMD start probe");
        gate(member ? 1 : 0);
        if (probe == 2 && !member) throw std::runtime_error("injected busy-Core failure");
    }
    if (!member) {
        out[0] = 7;
        return;
    }

    float base = 10.0f;
    if (args[4]) {
        // This Block receives the SS result by address only. Its Tensor descriptor belongs to a different task.
        Tensor seed = tensor;
        seed.buffer.addr = static_cast<uint64_t>(args[4]);
        seed.buffer.size = sizeof(float);
        seed.start_offset = 0;
        seed.shapes[0] = 1;
        wse::SramView view;
        if (!wse::make_sram_view(args, seed, view)) throw std::runtime_error("missing SS result in SRAM access map");
        base = *reinterpret_cast<const float *>(view.address) + 3.0f;
    }
    out[block * 2 + thread] = base + 10 * block + thread;
}
