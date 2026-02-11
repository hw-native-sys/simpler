
#include <stdint.h>

#ifndef NDEBUG
#include <vector>
#endif

#include "common.h"
#include "data_type.h"

#pragma once

#define RUNTIME_MAX_TENSOR_DIMS 8

/**
 * Buffer Handle
 *
 * Represents a device memory buffer with address and total size in bytes.
 * This is the underlying memory allocation that a Tensor describes access patterns for.
 */
struct PTOBufferHandle {
    uint64_t addr;  // Device memory address (bytes)
    uint64_t size;  // Total buffer size in bytes
};

#ifndef NDEBUG
// 用于测试追踪 complex_overlap 是否被调用
struct OverlapPathTracker {
    static int& complex_overlap_call_count() {
        static int count = 0;
        return count;
    }
    static void reset() { complex_overlap_call_count() = 0; }
    static void record_complex_call() { complex_overlap_call_count()++; }
    static bool was_complex_called() { return complex_overlap_call_count() > 0; }
};
#endif

enum class OverlapType {
    Accurate = 0,
    Fuzzy = 1,
};

enum class OverlapStatus {
    NO_OVERLAP,
    COVERED,
    OTHER,
};

struct Segment {
    uint64_t begin;
    uint64_t end;

    bool line_segment_intersection(const Segment& other) const { return end > other.begin && other.end > begin; }
    bool contains(const Segment& other) const { return begin <= other.begin && other.end <= end; }
};

// 特殊值，表示 reshape 后需要分配新地址
static constexpr uint64_t RESHAPE_NEEDS_ALLOC = UINT64_MAX;

/**
 * Tensor descriptor for Task input/output
 *
 * Describes a strided memory access pattern on Global Memory (GM).
 * This allows expressing non-contiguous but regular memory layouts.
 *
 * IMPORTANT: After Phase 1 refactoring:
 * - `buffer` contains the underlying memory allocation (addr in bytes, size in bytes)
 * - `start_offset`, `strides[]`, `repeats[]` are in ELEMENTS
 * - `dtype` specifies element type for interpreting buffer contents
 *
 * Example: buffer.addr=base, dtype=FLOAT32, start_offset=7, strides=[10, 1], repeats=[3, 6]
 * Memory access pattern (from innermost to outermost dimension, in elements):
 *   - Start at buffer.addr + 7*4 bytes
 *   - Inner dim (strides[1]=1, repeats[1]=6): access 6 consecutive elements
 *   - Outer dim (strides[0]=10, repeats[0]=3): repeat 3 times with stride 10 elements
 * Result: [addr+28..addr+48], [addr+68..addr+88], [addr+108..addr+128] (byte offsets)
 */
struct Tensor {
    class ContiguousMemSegIterator {
    public:
        ContiguousMemSegIterator(const Tensor& tensor);

        void operator++();
        void operator++(int) { ++*this; }

        const Segment& operator*() const { return cur_seg; }

        bool is_end() const { return indexes_[0] >= tensor_.repeats[0]; }

    private:
        const Tensor& tensor_;
        Segment cur_seg;
        uint64_t indexes_[RUNTIME_MAX_TENSOR_DIMS];
    };

    PTOBufferHandle buffer;                     // Underlying memory buffer (addr in bytes, size in bytes)
    uint64_t start_offset;                      // Starting offset from buffer.addr, unit: elements
    uint64_t strides[RUNTIME_MAX_TENSOR_DIMS];  // Stride for each dimension, unit: elements
    uint64_t repeats[RUNTIME_MAX_TENSOR_DIMS];  // Repeat count for each dimension
    uint64_t ndims;                             // Number of dimensions used
    DataType dtype;                             // Data type of tensor elements
    int32_t version;                            // tensor的版本
    OverlapType overlap_type;                   // 判断覆盖的方式

    Tensor() : buffer{0, 0}, ndims(0), dtype(DataType::FLOAT32) {}

    explicit Tensor(uint64_t addr,
        uint64_t buffer_size_bytes,
        uint64_t start_offset,
        uint64_t strides[],
        uint64_t repeats[],
        uint64_t ndims,
        DataType dtype,
        int32_t version,
        OverlapType overlap_type = OverlapType::Accurate);

    Tensor(Tensor&& other);
    Tensor(const Tensor& other);

    Tensor& operator=(const Tensor& other);

    std::string dump() const;

    bool is_valid_tensor() const;

    Tensor& optimize();

    void resort_strides();

    void remove_redundant_dims();

#ifndef NDEBUG
    bool validate_memory_access_preserved(
        uint64_t original_strides[], uint64_t original_repeats[], int32_t original_ndims) const;

    std::vector<uint64_t> collect_all_offsets(
        const uint64_t strides_arr[], const uint64_t repeats_arr[], int32_t dims) const;
#endif

    bool valid_view(const uint64_t shapes[], const uint64_t offsets[]) const;

    bool valid_reshape(const uint64_t shapes[], uint64_t new_ndims) const;

    bool valid_transpose(uint64_t x, uint64_t y) const { return x < ndims && y < ndims; }

    Tensor view(const uint64_t shapes[], const uint64_t offsets[]) const;

    bool is_contiguous() const;

    uint64_t numel() const;

    Tensor reshape(const uint64_t shapes[], uint64_t new_ndims) const;

    Tensor transpose(uint64_t x, uint64_t y) const;

    Segment get_fuzzy_seg() const;

    bool is_same_memref(const Tensor& other) const { return buffer.addr == other.buffer.addr; }

    bool is_same_strides(const Tensor& other) const;

    void offset_to_ndims(uint64_t offset_ndims[]) const;

    uint64_t offset_ndim_to_1d(const uint64_t offset_ndims[]) const;

    OverlapStatus is_overlap(const Tensor& pre_task_output) const;

    bool complex_overlap(const Tensor& pre_task_output) const;

    /**
     * Create a 1D contiguous Tensor covering the entire buffer.
     * strides={1}, repeats={size_elements}, ndims=1.
     */
    static Tensor make_1d_contiguous(
        uint64_t addr, int32_t size_bytes, int32_t version = 0, DataType dtype = DataType::FLOAT32) {
        uint64_t size_elements = size_bytes / get_element_size(dtype);
        uint64_t strides[] = {1};
        uint64_t repeats[] = {size_elements};
        return Tensor(addr, size_bytes, 0, strides, repeats, 1, dtype, version);
    }
};

// =============================================================================
// Factory Helpers
// =============================================================================

/**
 * Create a Tensor for pre-allocated external memory.
 */
static inline Tensor make_tensor_external(
    void* addr, int32_t size_bytes, int32_t version = 0, DataType dtype = DataType::FLOAT32) {
    return Tensor::make_1d_contiguous(reinterpret_cast<uint64_t>(addr), size_bytes, version, dtype);
}

/**
 * Create a Tensor for runtime-allocated output (addr=0).
 * The runtime fills in the actual address during pto2_submit_task.
 */
static inline Tensor make_tensor(int32_t size_bytes, int32_t version = 0, DataType dtype = DataType::FLOAT32) {
    return Tensor::make_1d_contiguous(0, size_bytes, version, dtype);
}
