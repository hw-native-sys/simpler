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
#include <pto/pto-inst.hpp>
#include "wse/block_scope.h"
#include "wse/operation_trace.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <typeinfo>

namespace wse {
inline constexpr const char *OPERATION_BACKEND = "pto-linx-cpu";

template <int Rows, int Cols, int RowStride = Cols, class T>
auto tile_memory(T *data, size_t elements, size_t offset = 0) {
    static_assert(Rows > 0 && Cols > 0 && RowStride >= Cols);
    constexpr size_t extent = size_t(Rows - 1) * RowStride + Cols;
    if (!data || offset > elements || extent > elements - offset)
        throw std::out_of_range("WSE tile exceeds its tensor view");
    return pto::global_tensor<T, pto::RowMajor<Rows, Cols, RowStride>>(data + offset);
}

template <unsigned Slot, class Shape>
class SharedTileSlot {
    static_assert(Slot < std::tuple_size_v<SharedSlotTable>, "Linx Shared slot must be S0..S63");
    static_assert(Shape::ValidRow > 0 && Shape::ValidCol > 0);
    BlockScope scope_;
    OperationTrace trace_;
    pto::linx_cpu::CoreContext *core_;
    SharedSlotState *state_;
    uint32_t pe_;

    void validate() const {
        if (core_ != pto::linx_cpu::currentCore || pe_ != pto::linx_cpu::threadIndex())
            throw std::invalid_argument("WSE SharedTile used outside its Linx PE");
        scope_.validate();
    }

    template <class Missing>
    void wait_for(std::unique_lock<std::mutex> &lock, uint64_t generation, Missing missing) {
        if (missing() & (1u << pe_)) throw std::invalid_argument("WSE SharedTile PE cannot wait for its own operation");
        if (missing()) trace_(OperationEvent::WAIT, Slot, generation);
        core_->changed.wait(lock, [&] {
            return core_->error || !missing() || (core_->finished & missing());
        });
        if (core_->error) std::rethrow_exception(core_->error);
        if (missing()) throw std::invalid_argument("WSE SharedTile participant exited before generation completed");
    }

    template <int Mask, class Global>
    static void load_slot(const Global &source) {
        auto tile = ::TLOAD<Shape, Mask>(source);
        pto::linx_cpu::publishSlot<Slot, Mask>(tile);
    }

public:
    SharedTileSlot(uint32_t producer, uint32_t consumers, BlockScope scope, OperationTrace trace = {}) :
        scope_(scope),
        trace_(trace),
        core_(pto::linx_cpu::currentCore),
        pe_(pto::linx_cpu::threadIndex()) {
        scope_.validate();
        if (!core_ || !scope_.slots) throw std::invalid_argument("WSE SharedTile requires an active Linx Kernel Block");
        if (producer >= 4 || !(core_->activeMask & (1u << producer)) || !consumers || (consumers & ~core_->activeMask))
            throw std::invalid_argument("invalid WSE SharedTile participants");
        // The registered DSOs outlive this Block. Compare the complete Linx tile type across their RTTI copies.
        const char *shape_name = typeid(Shape).name();
        std::lock_guard lock(core_->mutex);
        state_ = &(*scope_.slots)[Slot];
        if (!state_->shape_name) {
            *state_ = {producer, consumers, shape_name};
        } else if (state_->producer != producer || state_->consumers != consumers ||
                   std::strcmp(state_->shape_name, shape_name) != 0) {
            throw std::invalid_argument("WSE SharedTile slot configuration mismatch");
        }
    }

    template <class T, int Rows, int Cols, int Stride>
    void publish(uint64_t generation, const pto::global_tensor<T, pto::RowMajor<Rows, Cols, Stride>> &source) {
        static_assert(
            Rows >= Shape::ValidRow && Cols >= Shape::ValidCol, "Linx Shared tile exceeds the source matrix dimensions"
        );
        validate();
        std::unique_lock lock(core_->mutex);
        if (pe_ != state_->producer || !generation || generation != state_->generation + 1)
            throw std::invalid_argument("WSE SharedTile publication order or producer mismatch");
        wait_for(lock, generation, [&] {
            return state_->generation ? state_->consumers & ~state_->consumed : 0u;
        });
        lock.unlock();
        // Linx PEMask bit 3 selects PE 0.
        using Global = pto::global_tensor<T, pto::RowMajor<Rows, Cols, Stride>>;
        constexpr std::array load_by_pe{
            load_slot<8, Global>, load_slot<4, Global>, load_slot<2, Global>, load_slot<1, Global>
        };
        load_by_pe[pe_](source);
        lock.lock();
        state_->consumed = 0;
        state_->generation = generation;
        core_->changed.notify_all();
        trace_(OperationEvent::PUBLISH, Slot, generation);
    }

    pto::SharedTile<Shape> read(uint64_t generation) {
        validate();
        std::unique_lock lock(core_->mutex);
        const uint32_t bit = 1u << pe_;
        if (!generation || !(state_->consumers & bit) || generation < state_->generation ||
            generation - state_->generation > 1)
            throw std::invalid_argument("WSE SharedTile read order or consumer mismatch");
        wait_for(lock, generation, [&] {
            return state_->generation < generation ? 1u << state_->producer : 0u;
        });
        if (state_->consumed & bit) throw std::invalid_argument("duplicate WSE SharedTile read");
        lock.unlock();
        pto::SharedTile<Shape> result;
        pto::linx_cpu::acquireSlot<Slot>(result);
        lock.lock();
        state_->consumed |= bit;
        core_->changed.notify_all();
        trace_(OperationEvent::READ, Slot, generation);
        trace_(OperationEvent::CONSUMED, Slot, generation);
        return result;
    }
};
}  // namespace wse
