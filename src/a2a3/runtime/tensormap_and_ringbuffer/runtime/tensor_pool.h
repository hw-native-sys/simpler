#pragma once

#include <stdint.h>

#include <sstream>

#include "common.h"
#include "data_type.h"

constexpr int TENSOR_DATA_MAX_SIZE = 1 << 17;
constexpr int RUNTIME_MAX_TENSOR_DIMS = 5;

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

struct TensorData {
    int32_t ref_count;                             // Reference count (managed by TensorPool, NOT copied by init())
    int32_t version;                               // tensor的版本
    PTOBufferHandle buffer;                        // Underlying memory buffer (addr in bytes, size in bytes)
    uint64_t raw_shapes[RUNTIME_MAX_TENSOR_DIMS];  // Underlying buffer shape per dimension
    uint64_t shapes[RUNTIME_MAX_TENSOR_DIMS];      // Current view shape per dimension
    uint64_t offsets[RUNTIME_MAX_TENSOR_DIMS];     // Multi-dimensional offset per dimension
    uint64_t start_offset;                         // Cached 1D element offset (precomputed from raw_shapes + offsets)
    uint64_t ndims;                                // Number of dimensions used
    DataType dtype;                                // Data type of tensor elements

    TensorData() = default;

    void init(void* addr,
        uint64_t buffer_size_bytes,
        const uint64_t raw_shapes[],
        const uint64_t shapes[],
        const uint64_t offsets[],
        uint64_t ndims,
        DataType dtype,
        int32_t version) {
        buffer = {reinterpret_cast<uint64_t>(addr), buffer_size_bytes};
        this->ndims = ndims;
        this->dtype = dtype;
        this->version = version;
        for (uint64_t i = 0; i < ndims; i++) {
            this->raw_shapes[i] = raw_shapes[i];
            this->shapes[i] = shapes[i];
            this->offsets[i] = offsets[i];
        }
        ref_count = 1;
    }

    void init(const TensorData& other) {
        buffer = other.buffer;
        ndims = other.ndims;
        dtype = other.dtype;
        version = other.version;
        for (uint64_t i = 0; i < ndims; i++) {
            raw_shapes[i] = other.raw_shapes[i];
            shapes[i] = other.shapes[i];
            offsets[i] = other.offsets[i];
        }
        ref_count = 1;
    }

    void init_with_view(const TensorData& other, const uint64_t view_shapes[], const uint64_t view_offsets[]) {
        buffer = other.buffer;
        ndims = other.ndims;
        dtype = other.dtype;
        version = other.version;
        for (uint64_t i = 0; i < ndims; i++) {
            raw_shapes[i] = other.raw_shapes[i];
            shapes[i] = view_shapes[i];
            offsets[i] = other.offsets[i] + view_offsets[i];
        }
        ref_count = 1;
    }

    TensorData(TensorData&& other) = delete;
    TensorData(const TensorData& other) = delete;
    TensorData& operator=(TensorData&& other) = delete;
    TensorData& operator=(const TensorData& other) = delete;

    // Recompute cached start_offset from raw_shapes + offsets.
    // Must be called after any mutation of offsets or raw_shapes.
    void update_start_offset() {
        uint64_t result = 0;
        uint64_t stride = 1;
        for (int i = static_cast<int>(ndims) - 1; i >= 0; i--) {
            result += offsets[i] * stride;
            stride *= raw_shapes[i];
        }
        start_offset = result;
    }

    void view(const uint64_t view_shapes[], const uint64_t view_offsets[]) {
        for (size_t i = 0; i < ndims; i++) {
            offsets[i] += view_offsets[i];
            shapes[i] = view_shapes[i];
        }
    }

    std::string dump() const {
        std::stringstream ss;
        std::string indent = "    ";
        ss << "{" << std::endl;
        ss << indent << "buffer.addr: " << buffer.addr << std::endl;
        ss << indent << "buffer.size: " << buffer.size << " bytes" << std::endl;
        ss << indent << "dtype: " << get_dtype_name(dtype) << std::endl;
        ss << indent << "ndims: " << ndims << std::endl;
        ss << indent << "version: " << version << std::endl;

        ss << indent << "raw_shapes: [";
        for (uint64_t i = 0; i < ndims; i++) {
            if (i > 0) {
                ss << ", ";
            }
            ss << raw_shapes[i];
        }
        ss << "]" << std::endl;
        ss << indent << "shapes: [";
        for (uint64_t i = 0; i < ndims; i++) {
            if (i > 0) {
                ss << ", ";
            }
            ss << shapes[i];
        }
        ss << "]" << std::endl;
        ss << indent << "offsets: [";
        for (uint64_t i = 0; i < ndims; i++) {
            if (i > 0) {
                ss << ", ";
            }
            ss << offsets[i];
        }
        ss << "]" << std::endl;
        ss << "}" << std::endl;
        return ss.str();
    }

    bool is_contiguous() const {
        if (ndims == 0) {
            return true;
        }
        // Inner dimensions must fully span raw_shapes
        for (uint64_t i = 1; i < ndims; i++) {
            if (shapes[i] != raw_shapes[i]) {
                return false;
            }
        }
        return true;
    }

    uint64_t numel() const {
        if (ndims == 0) {
            return 0;
        }
        uint64_t total = 1;
        for (uint64_t i = 0; i < ndims; i++) {
            total *= shapes[i];
        }
        return total;
    }

    bool is_same_memref(const TensorData& other) const { return buffer.addr == other.buffer.addr; }

    OverlapStatus is_overlap(const TensorData& pre_task_output) const {
        debug_assert(is_same_memref(pre_task_output));
        debug_assert(version >= pre_task_output.version);
        if (version > pre_task_output.version) {
            return OverlapStatus::OTHER;
        }

        // With raw_shapes+shapes+offsets representation, offsets are directly available
        // and offsets[i] + shapes[i] <= raw_shapes[i] is guaranteed by is_valid_tensor(),
        // so hyper-rectangle wrapping cannot happen.
        bool contains = true;
        for (uint64_t i = 0; i < ndims; i++) {
            Segment input_range_dim_i{offsets[i], offsets[i] + shapes[i]};
            Segment output_range_dim_i{
                pre_task_output.offsets[i], pre_task_output.offsets[i] + pre_task_output.shapes[i]};
            if (!input_range_dim_i.line_segment_intersection(output_range_dim_i)) {
                return OverlapStatus::NO_OVERLAP;
            } else if (!input_range_dim_i.contains(output_range_dim_i)) {
                contains = false;
            }
        }
        if (contains) {
            return OverlapStatus::COVERED;
        }
        return OverlapStatus::OTHER;
    }
};

struct TensorPool {
    // Management fields first (hot path) — close to struct base for cache locality
    int32_t next_index;
    int32_t free_num;
    int32_t free_index_stack[TENSOR_DATA_MAX_SIZE];

    // Bulk data last — ref_count is inlined in TensorData for same-cacheline access
    TensorData data[TENSOR_DATA_MAX_SIZE];

    void init() {
        next_index = 1;
        free_num = 0;
    }

    int32_t alloc() {
        int32_t index;
        if (free_num > 0) {
            index = free_index_stack[--free_num];
        } else {
            always_assert(next_index < TENSOR_DATA_MAX_SIZE);
            index = next_index++;
        }
        return index;
    }

    void free(int32_t index) {
        debug_assert(index > 0);
        free_index_stack[free_num++] = index;
    }

    void ref(int32_t index) {
        debug_assert(data[index].ref_count >= 0);
        data[index].ref_count++;
    }

    void deref(int32_t index) {
        if (index == 0) {
            return;
        }
        always_assert(data[index].ref_count > 0);
        if (--data[index].ref_count == 0) {
            free(index);
        }
    }

    // 单例接口（编排 .so 和运行时各自持有独立的 s_instance）
    static TensorPool& instance() { return *s_instance; }
    static void set_instance(TensorPool* pool) { s_instance = pool; }

private:
    static inline TensorPool* s_instance = nullptr;
};