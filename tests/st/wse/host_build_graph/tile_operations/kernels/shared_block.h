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
#pragma once
#include <wse/tile_ops.h>
#include "tensor.h"
#include "intrinsic.h"
#include <dlfcn.h>
#include <cstdlib>
struct ProbeTrace {
    wse::OperationTrace original;
    uint32_t pe;
};
inline void probe_trace(void *owner, wse::OperationEvent event, uint32_t slot, uint64_t generation) {
    auto &p = *static_cast<ProbeTrace *>(owner);
    p.original(event, slot, generation);
    if (event == wse::OperationEvent::WAIT) {
        auto notify = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "wse_shared_probe"));
        if (!notify) std::abort();
        notify(p.pe);
    }
}
inline void shared_block_compute(int64_t *args, bool producer) {
    if (get_core_id(args) != 0) return;
    const auto pe = get_pe_id(args);
    if (!producer && pe == 0) return;
    ProbeTrace trace{get_wse_operation_trace(args), static_cast<uint32_t>(pe)};
    wse::SharedTile<wse::Tile<2, 2>> shared(pe, 2, 0, 14, get_wse_block_scope(args), {&trace, probe_trace});
    if (producer) {
        auto gate = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "wse_shared_probe"));
        if (!gate) std::abort();
        gate(0);
        wse::Tile<2, 2> value;
        auto *t = reinterpret_cast<Tensor *>(args[0]);
        auto *source = reinterpret_cast<float *>(t->buffer.addr) + t->start_offset;
        wse::TLOAD(value, {source, 4, 2, 2, 2, 1});
        shared.publish(1, value);
    } else {
        auto value = shared.read(1);
        auto *t = reinterpret_cast<Tensor *>(args[1]);
        auto *out = reinterpret_cast<float *>(t->buffer.addr) + t->start_offset + get_thread_idx(args) * 4;
        wse::TSTORE({out, 4, 2, 2, 2, 1}, value);
    }
}
