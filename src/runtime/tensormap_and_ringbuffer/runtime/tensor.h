#pragma once

#include <stdint.h>
#include <memory.h>

#include "common.h"
#include "data_type.h"
#include "tensor_pool.h"

/**
 * Tensor descriptor for Task input/output
 *
 * Describes a memory access pattern on Global Memory (GM) using
 * raw_shapes (underlying buffer dimensions), shapes (current view dimensions),
 * and offsets (multi-dimensional offset into the buffer).
 *
 * - `buffer` contains the underlying memory allocation (addr in bytes, size in bytes)
 * - `raw_shapes[]`, `shapes[]`, `offsets[]` are in ELEMENTS
 * - `dtype` specifies element type for interpreting buffer contents
 *
 * Example: buffer.addr=base, dtype=FLOAT32, raw_shapes=[10, 6], shapes=[3, 6], offsets=[1, 0]
 * Memory access pattern:
 *   - Start at buffer.addr + (1*6+0)*4 = buffer.addr + 24 bytes
 *   - Inner dim: access 6 consecutive elements
 *   - Outer dim: 3 rows with stride 6 elements (derived from raw_shapes[1])
 */
struct Tensor {
    int32_t index;

    Tensor() : index(0) {}

    Tensor(void* addr,
        uint64_t buffer_size_bytes,
        const uint64_t raw_shapes[],
        const uint64_t shapes[],
        const uint64_t offsets[],
        uint64_t ndims,
        DataType dtype,
        int32_t version) {
        TensorPool& pool = TensorPool::instance();
        index = pool.alloc();
        pool.data[index].init(addr, buffer_size_bytes, raw_shapes, shapes, offsets, ndims, dtype, version);
    }

    Tensor(Tensor&& other) : index(other.index) { other.index = 0; }

    Tensor(const Tensor& other) : index(other.index) { TensorPool::instance().ref(index); }

    Tensor& operator=(Tensor&& other) {
        TensorPool::instance().deref(index);
        index = other.index;
        other.index = 0;
        return *this;
    }

    Tensor& operator=(const Tensor& other) {
        if (index != other.index) {
            TensorPool::instance().deref(index);
            index = other.index;
            TensorPool::instance().ref(index);
        }
        return *this;
    }

    ~Tensor() {
        TensorPool::instance().deref(index);
    }

    TensorData& data() { return TensorPool::instance().data[index]; }
    const TensorData& data() const { return TensorPool::instance().data[index]; }

    Tensor copy() const {
        if (index == 0) {
            return Tensor();
        }
        Tensor result;
        TensorPool& pool = TensorPool::instance();
        result.index = pool.alloc();
        pool.data[result.index].init(pool.data[index]);
        return result;
    }

    Tensor view(const uint64_t view_shapes[], const uint64_t view_offsets[]) const {
        Tensor result;
        TensorPool& pool = TensorPool::instance();
        result.index = pool.alloc();
        pool.data[result.index].init_with_view(pool.data[index], view_shapes, view_offsets);
        return result;
    }

    bool is_contiguous() const { return data().is_contiguous(); }

    bool valid_reshape(const uint64_t new_shapes[], uint64_t new_ndims) const {
        const TensorData& tensor_data = data();
        uint64_t x = 1;
        for (size_t i = 0; i < tensor_data.ndims; i++) {
            x *= tensor_data.shapes[i];
        }
        uint64_t y = 1;
        for (size_t i = 0; i < new_ndims; i++) {
            y *= new_shapes[i];
        }
        return x == y;
    }

    Tensor reshape(const uint64_t new_shapes[], uint64_t new_ndims) const {
        debug_assert(valid_reshape(new_shapes, new_ndims));
        always_assert(is_contiguous());
        Tensor result = copy();
        TensorData& result_tensor_data = result.data();
        result_tensor_data.ndims = new_ndims;
        for (uint64_t i = 0; i < new_ndims; i++) {
            result_tensor_data.raw_shapes[i] = new_shapes[i];
            result_tensor_data.shapes[i] = new_shapes[i];
            result_tensor_data.offsets[i] = 0;
        }
        return result;
    }

    bool valid_transpose(uint64_t x, uint64_t y) const { return x < data().ndims && y < data().ndims; }

    Tensor transpose(uint64_t x, uint64_t y) const {
        debug_assert(valid_transpose(x, y));
        Tensor result = copy();
        TensorData& result_tensor_data = result.data();
        std::swap(result_tensor_data.raw_shapes[x], result_tensor_data.raw_shapes[y]);
        std::swap(result_tensor_data.shapes[x], result_tensor_data.shapes[y]);
        std::swap(result_tensor_data.offsets[x], result_tensor_data.offsets[y]);
        return result;
    }

    std::string dump() const { return data().dump(); }

    uint64_t numel() const { return data().numel(); }

    OverlapStatus is_overlap(const Tensor& pre_task_output) const { return data().is_overlap(pre_task_output.data()); }
};

// =============================================================================
// Factory Helpers
// =============================================================================
/**
 * Create a Tensor for pre-allocated external memory.
 */
static inline Tensor make_tensor_external(void* addr,
    const uint64_t shapes[],
    uint64_t ndims,
    DataType dtype = DataType::FLOAT32,
    int32_t version = 0) {
    static uint64_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint64_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return Tensor(addr, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version);
}

/**
 * Create a Tensor for runtime-allocated output (addr=0).
 * NO memory allocation: only records dtype, shape, and buffer.size in the Tensor struct.
 * The runtime allocates from the heap ring and fills buffer.addr during pto2_submit_task
 * when this tensor is passed as OUTPUT param. No buffer content is ever copied.
 */
static inline Tensor make_tensor(const uint64_t shapes[],
    uint64_t ndims,
    DataType dtype = DataType::FLOAT32,
    int32_t version = 0) {
    static uint64_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint64_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return Tensor(0, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version);
}
