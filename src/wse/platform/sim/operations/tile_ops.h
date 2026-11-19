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
#include "wse/block_scope.h"
#include "block_lifecycle.h"
#include <algorithm>
#include <array>
#include <new>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <stdexcept>
#include <utility>
#include "wse/operation_trace.h"

namespace wse::temporary {
enum class Layout : uint32_t { ROW_MAJOR, COLUMN_MAJOR };
template <size_t Rows, size_t Cols, Layout Order = Layout::ROW_MAJOR>
struct Tile {
    static_assert(Rows > 0 && Rows <= 32 && Cols > 0 && Cols <= 32, "temporary WSE tiles require dimensions in [1,32]");
    static constexpr size_t rows = Rows;
    static constexpr size_t cols = Cols;
    static constexpr Layout layout = Order;
    std::array<float, Rows * Cols> values{};
    float &at(size_t r, size_t c) { return values[Order == Layout::ROW_MAJOR ? r * Cols + c : c * Rows + r]; }
    float at(size_t r, size_t c) const { return values[Order == Layout::ROW_MAJOR ? r * Cols + c : c * Rows + r]; }
};
struct TensorView {
    float *data;
    size_t elements;
    size_t rows;
    size_t cols;
    size_t row_stride;
    size_t col_stride;
    void validate(size_t required_rows, size_t required_cols) const {
        if (!data || !required_rows || !required_cols || rows < required_rows || cols < required_cols || !row_stride ||
            !col_stride)
            throw std::invalid_argument("invalid WSE tensor view");
        const auto limit = std::numeric_limits<size_t>::max();
        if (required_rows - 1 > limit / row_stride || required_cols - 1 > limit / col_stride)
            throw std::out_of_range("WSE tensor stride overflow");
        const auto r = (required_rows - 1) * row_stride, c = (required_cols - 1) * col_stride;
        if (r > limit - c || r + c >= elements) throw std::out_of_range("WSE tensor access exceeds view");
    }
};
template <typename T>
void TLOAD(T &dst, const TensorView &src) {
    src.validate(T::rows, T::cols);
    for (size_t r = 0; r < T::rows; ++r)
        for (size_t c = 0; c < T::cols; ++c)
            dst.at(r, c) = src.data[r * src.row_stride + c * src.col_stride];
}
template <typename T>
void TSTORE(const TensorView &dst, const T &src) {
    dst.validate(T::rows, T::cols);
    for (size_t r = 0; r < T::rows; ++r)
        for (size_t c = 0; c < T::cols; ++c)
            dst.data[r * dst.row_stride + c * dst.col_stride] = src.at(r, c);
}
template <typename C, typename A, typename B>
void TMATMUL(C &dst, const A &a, const B &b) {
    static_assert(A::cols == B::rows && C::rows == A::rows && C::cols == B::cols);
    C result;
    for (size_t r = 0; r < C::rows; ++r)
        for (size_t c = 0; c < C::cols; ++c) {
            float sum = 0;
            for (size_t k = 0; k < A::cols; ++k)
                sum += a.at(r, k) * b.at(k, c);
            result.at(r, c) = sum;
        }
    dst = result;
}
template <typename C, typename A, typename B>
void TMATMUL_ACC(C &dst, const C &initial, const A &a, const B &b) {
    static_assert(A::cols == B::rows && C::rows == A::rows && C::cols == B::cols);
    C result = initial;
    for (size_t r = 0; r < C::rows; ++r)
        for (size_t c = 0; c < C::cols; ++c)
            for (size_t k = 0; k < A::cols; ++k)
                result.at(r, c) += a.at(r, k) * b.at(k, c);
    dst = result;
}
template <typename D, typename S>
void TRELU(D &dst, const S &src) {
    static_assert(D::rows == S::rows && D::cols == S::cols);
    for (size_t r = 0; r < D::rows; ++r)
        for (size_t c = 0; c < D::cols; ++c) {
            const float v = src.at(r, c);
            dst.at(r, c) = v < 0 ? 0 : v;
        }
}

// Tile payloads are trivial; synchronization belongs to their owning Block slot.
struct SharedTileState {
    uint32_t producer, consumers, consumed, rows, cols, order;
    uint64_t generation;
    float values[32 * 32];
};
static_assert(std::is_trivial_v<SharedTileState> && std::is_trivially_destructible_v<SharedTileState>);

template <typename T>
class SharedTile {
    uint32_t pe_, slot_;
    OperationTrace trace_;
    BlockScope scope_;
    BlockOperationStorage::Slot *storage_;
    SharedTileState *state_;
    void trace(OperationEvent event, uint64_t generation) const { trace_(event, slot_, generation); }

public:
    SharedTile(
        uint32_t pe, uint32_t slot, uint32_t producer, uint32_t consumers, BlockScope scope, OperationTrace trace = {}
    ) :
        pe_(pe),
        slot_(slot),
        trace_(trace),
        scope_(scope),
        storage_(nullptr),
        state_(nullptr) {
        scope_.validate();
        if (pe >= 4 || slot >= 64 || producer >= 4 || !consumers || (consumers & ~15u))
            throw std::invalid_argument("invalid WSE shared tile slot or participants");
        if (!scope_.storage_context || !scope_.get_storage)
            throw std::invalid_argument("WSE Block has no shared operation storage");
        storage_ = static_cast<BlockOperationStorage::Slot *>(
            scope_.get_storage(scope_.storage_context, slot, sizeof(SharedTileState))
        );
        std::lock_guard lock(storage_->mutex);
        state_ = static_cast<SharedTileState *>(storage_->data.get());
        if (!storage_->initialized) {
            state_ = new (state_) SharedTileState{};
            state_->producer = producer;
            state_->consumers = consumers;
            state_->rows = T::rows;
            state_->cols = T::cols;
            state_->order = static_cast<uint32_t>(T::layout);
            storage_->initialized = true;
        } else if (state_->producer != producer || state_->consumers != consumers || state_->rows != T::rows ||
                   state_->cols != T::cols || state_->order != static_cast<uint32_t>(T::layout)) {
            throw std::invalid_argument("WSE shared tile slot configuration mismatch");
        }
    }
    void publish(uint64_t generation, const T &value) {
        scope_.validate();
        std::unique_lock lock(storage_->mutex);
        if (pe_ != state_->producer || !generation || state_->generation == UINT64_MAX ||
            generation != state_->generation + 1)
            throw std::invalid_argument("WSE shared tile publication order or producer mismatch");
        if (state_->generation && state_->consumed != state_->consumers) trace(OperationEvent::WAIT, generation);
        storage_->changed.wait(lock, [&] {
            return !state_->generation || state_->consumed == state_->consumers;
        });
        std::copy(value.values.begin(), value.values.end(), state_->values);
        state_->consumed = 0;
        state_->generation = generation;
        trace(OperationEvent::PUBLISH, generation);
        storage_->changed.notify_all();
    }
    T read(uint64_t generation) {
        scope_.validate();
        std::unique_lock lock(storage_->mutex);
        const uint32_t bit = 1u << pe_;
        if (!generation || !(state_->consumers & bit) || generation < state_->generation ||
            (generation > state_->generation && generation - state_->generation != 1))
            throw std::invalid_argument("WSE shared tile read order or consumer mismatch");
        if (state_->generation < generation) trace(OperationEvent::WAIT, generation);
        storage_->changed.wait(lock, [&] {
            return state_->generation >= generation;
        });
        if (state_->generation != generation || (state_->consumed & bit))
            throw std::invalid_argument("stale or duplicate WSE shared tile read");
        trace(OperationEvent::READ, generation);
        T result;
        std::copy_n(state_->values, T::rows * T::cols, result.values.begin());
        state_->consumed |= bit;
        trace(OperationEvent::CONSUMED, generation);
        storage_->changed.notify_all();
        return result;
    }
};
template <typename C, typename A, typename B>
void TMATMUL(C &dst, const A &a, SharedTile<B> &b, uint64_t generation) {
    TMATMUL(dst, a, b.read(generation));
}
template <typename C, typename A, typename B>
void TMATMUL_ACC(C &dst, const C &initial, const A &a, SharedTile<B> &b, uint64_t generation) {
    TMATMUL_ACC(dst, initial, a, b.read(generation));
}
}  // namespace wse::temporary
