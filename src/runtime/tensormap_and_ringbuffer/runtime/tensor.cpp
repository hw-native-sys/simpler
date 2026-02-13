/**
 * Tensor Descriptor - Overlap detection implementations
 *
 * Methods here are only needed by runtime targets (aicpu/aicore/host),
 * NOT by the orchestration .so. Methods shared with orchestration
 * live in tensor_orch.cpp.
 */

#include "tensor.h"

// ContiguousMemSegIterator implementations
Tensor::ContiguousMemSegIterator::ContiguousMemSegIterator(const Tensor& tensor)
    : tensor_(tensor), cur_seg({tensor.start_offset, tensor.start_offset + tensor.repeats[tensor.ndims - 1]}) {
    for (uint64_t i = 0; i < tensor.ndims; i++) {
        indexes_[i] = 0;
    }
}

void Tensor::ContiguousMemSegIterator::operator++() {
    debug_assert(!is_end());
    debug_assert(tensor_.ndims > 1 || (tensor_.ndims == 1 && indexes_[0] == 0));
    indexes_[tensor_.ndims - 1] += tensor_.repeats[tensor_.ndims - 1];
    cur_seg.begin += tensor_.repeats[tensor_.ndims - 1];
    for (int32_t i = tensor_.ndims - 1; i >= 1; i--) {
        debug_assert(indexes_[i] <= tensor_.repeats[i]);
        if (indexes_[i] == tensor_.repeats[i]) {
            indexes_[i - 1]++;
            indexes_[i] = 0;
            // Jump to next outer dimension iteration:
            // outer_stride - (inner_stride * inner_repeats)
            cur_seg.begin += tensor_.strides[i - 1] - tensor_.strides[i] * tensor_.repeats[i];
        }
    }
    cur_seg.end = cur_seg.begin + tensor_.repeats[tensor_.ndims - 1];
}

bool Tensor::is_same_strides(const Tensor& other) const {
    for (uint64_t i = 0; i < ndims; i++) {
        if (strides[i] != other.strides[i]) {
            return false;
        }
    }
    return true;
}

void Tensor::offset_to_ndims(uint64_t offset_ndims[]) const {
    uint64_t cur_offset = start_offset;
    for (uint64_t i = 0; i < ndims; i++) {
        offset_ndims[i] = cur_offset / strides[i];
        cur_offset %= strides[i];
    }
}

OverlapStatus Tensor::is_overlap(const Tensor& pre_task_output) const {
    if (!is_same_memref(pre_task_output)) {
        return OverlapStatus::NO_OVERLAP;
    }
    debug_assert(version >= pre_task_output.version);
    if (version > pre_task_output.version) {
        return OverlapStatus::OTHER;
    }

    // Convert element offsets to byte offsets for comparison
    // This handles cases where the two descriptors have different dtypes
    uint64_t elem_size_input = get_element_size(dtype);
    uint64_t elem_size_output = get_element_size(pre_task_output.dtype);

    Segment input_memory_fuzzy_seg = get_fuzzy_seg();
    Segment output_memory_fuzzy_seg = pre_task_output.get_fuzzy_seg();

    // Convert to byte offsets
    Segment input_byte_seg{
        input_memory_fuzzy_seg.begin * elem_size_input, input_memory_fuzzy_seg.end * elem_size_input};
    Segment output_byte_seg{
        output_memory_fuzzy_seg.begin * elem_size_output, output_memory_fuzzy_seg.end * elem_size_output};

    if (!input_byte_seg.line_segment_intersection(output_byte_seg)) {
        return OverlapStatus::NO_OVERLAP;
    }

    // 只做模糊判断
    if (pre_task_output.overlap_type == OverlapType::Fuzzy) {
        return OverlapStatus::OTHER;
    }

    // 一维场景
    if (ndims == 1 && pre_task_output.ndims == 1) {
        debug_assert(strides[0] == 1);
        debug_assert(pre_task_output.strides[0] == 1);
        if (input_byte_seg.contains(output_byte_seg)) {
            return OverlapStatus::COVERED;
        } else {
            return OverlapStatus::OTHER;
        }
    }

    // 精准判断 - only if same dtype and strides
    // For different dtypes, we fall back to complex_overlap
    if (dtype == pre_task_output.dtype && ndims == pre_task_output.ndims && is_same_strides(pre_task_output)) {
        uint64_t input_offset_ndims[RUNTIME_MAX_TENSOR_DIMS];
        uint64_t output_offset_ndims[RUNTIME_MAX_TENSOR_DIMS];
        offset_to_ndims(input_offset_ndims);
        pre_task_output.offset_to_ndims(output_offset_ndims);
        // O(ndims) 判断超矩形间overlap
        bool need_complex_compare = false;
        bool contains = true;
        bool overlap = true;
        for (uint64_t i = 0; i < ndims; i++) {
            Segment input_range_dim_i{input_offset_ndims[i], input_offset_ndims[i] + repeats[i]};
            Segment output_range_dim_i{output_offset_ndims[i], output_offset_ndims[i] + pre_task_output.repeats[i]};
            // Skip outermost dimension (i == 0), check inner dimensions
            // With descending strides, strides[i-1] is the outer dimension's stride
            if (i > 0) {
                // input不是超矩形
                if (input_range_dim_i.end * strides[i] > strides[i - 1]) {
                    need_complex_compare = true;
                    break;
                }
                // output不是超矩形
                if (output_range_dim_i.end * pre_task_output.strides[i] > pre_task_output.strides[i - 1]) {
                    need_complex_compare = true;
                    break;
                }
            }
            if (!input_range_dim_i.line_segment_intersection(output_range_dim_i)) {
                overlap = false;
            } else if (!input_range_dim_i.contains(output_range_dim_i)) {
                contains = false;
            }
        }
        if (!need_complex_compare) {
            if (contains) {
                return OverlapStatus::COVERED;
            } else if (overlap) {
                return OverlapStatus::OTHER;
            } else {
                return OverlapStatus::NO_OVERLAP;
            }
        }
    }
    // O(\prod repeats[i]) 判断线段相交
    return complex_overlap(pre_task_output) ? OverlapStatus::OTHER : OverlapStatus::NO_OVERLAP;
}

bool Tensor::complex_overlap(const Tensor& pre_task_output) const {
#ifndef NDEBUG
    OverlapPathTracker::record_complex_call();
#endif
    // Convert element offsets to byte offsets for comparison when dtypes differ
    uint64_t elem_size_input = get_element_size(dtype);
    uint64_t elem_size_output = get_element_size(pre_task_output.dtype);

    ContiguousMemSegIterator input_segs_iter(*this);
    ContiguousMemSegIterator output_segs_iter(pre_task_output);
    while (!input_segs_iter.is_end() && !output_segs_iter.is_end()) {
        const Segment& cur_input_memory_seg = *input_segs_iter;
        const Segment& cur_output_memory_seg = *output_segs_iter;

        // Convert to byte offsets for comparison
        Segment input_byte_seg{
            cur_input_memory_seg.begin * elem_size_input, cur_input_memory_seg.end * elem_size_input};
        Segment output_byte_seg{
            cur_output_memory_seg.begin * elem_size_output, cur_output_memory_seg.end * elem_size_output};

        if (input_byte_seg.end <= output_byte_seg.begin) {
            input_segs_iter++;
            continue;
        } else if (output_byte_seg.end <= input_byte_seg.begin) {
            output_segs_iter++;
            continue;
        }
        return true;
    }
    return false;
}