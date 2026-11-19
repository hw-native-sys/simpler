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
#include <wse/tile_ops.h>
#include <dlfcn.h>
#include <optional>
struct Trace {
    wse::OperationTrace original;
    uint32_t core, pe;
};
void trace(void *opaque, wse::OperationEvent event, uint32_t slot, uint64_t generation) {
    auto &t = *static_cast<Trace *>(opaque);
    t.original(event, slot, generation);
    if (t.core == 1 && event == wse::OperationEvent::WAIT) {
        auto hook = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "wse_scope_probe"));
        if (!hook) throw std::runtime_error("missing scope probe");
        hook(t.pe);
    }
}
template <int Rows, int Cols>
using Plane = pto::Tile<pto::Location::Vec, float, 4, 8, pto::BLayout::RowMajor, Rows, Cols>;

template <class Shape>
float exchange_value(wse::SharedTileSlot<5, Shape> &slot, bool producer, float input) {
    std::array<float, Shape::ValidRow * Shape::ValidCol> data{input};
    auto memory = wse::tile_memory<Shape::ValidRow, Shape::ValidCol>(data.data(), data.size());
    if (producer) slot.publish(1, memory);
    data[0] = -1.f;
    auto value = slot.read(1);
    TSTORE(memory, value);
    return data[0];
}
extern "C" void kernel_entry(int64_t *args) {
    const uint32_t b = get_block_idx(args), t = get_thread_idx(args), pe = get_pe_id(args);
    auto &tensor = *reinterpret_cast<Tensor *>(args[0]);
    auto *record = reinterpret_cast<uint64_t *>(tensor.buffer.addr) + tensor.start_offset + (b * 4 + t) * 4;
    Trace observer{get_wse_operation_trace(args), static_cast<uint32_t>(get_core_id(args)), pe};
    auto hook = reinterpret_cast<void (*)(int)>(dlsym(RTLD_DEFAULT, "wse_scope_probe"));
    if (!hook) throw std::runtime_error("missing scope probe");
    static thread_local std::optional<wse::SharedTileSlot<5, Plane<2, 2>>> previous;
    if (b < 2) {
        wse::SharedTileSlot<5, Plane<2, 2>> shared(0, 15, get_wse_block_scope(args), {&observer, trace});
        if (t == 0) {
            if (b == 1) hook(0);
            else hook(5);
        }
        record[0] = exchange_value(shared, t == 0, 100 + b);
        previous = shared;
        record[1] = 0;
    } else {
        bool rejected = false;
        try {
            if (previous) previous->read(1);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("old Block SharedTile remained valid");
        wse::SharedTileSlot<5, Plane<1, 4>> shared(0, 15, get_wse_block_scope(args), {&observer, trace});
        record[0] = exchange_value(shared, t == 0, 102);
        record[1] = 1;
        if (t == 0) hook(4);
    }
    record[2] = t;
    record[3] = pe;
}
