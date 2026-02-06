/**
 * PTO Runtime2 - Logical Tensor Implementation
 * 
 * Implements tensor extraction operations (view, reshape, transpose)
 * and memory overlap detection algorithms.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_logical_tensor.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Bounding Box Computation
// =============================================================================

void pto2_logical_tensor_get_bounding_box(
    const PTO2LogicalTensor* tensor,
    int64_t* out_min,
    int64_t* out_max
) {
    // Start with storage offset
    int64_t min_offset = tensor->storage_offset;
    int64_t max_offset = tensor->storage_offset;
    
    // For each dimension, compute contribution to bounding box
    // Positive stride: max extends forward
    // Negative stride: min extends backward (for flipped dimensions)
    for (int32_t d = 0; d < tensor->ndim; d++) {
        if (tensor->shape[d] > 0) {
            int64_t extent = (tensor->shape[d] - 1) * tensor->strides[d];
            if (tensor->strides[d] >= 0) {
                max_offset += extent;
            } else {
                // Negative stride: extent is negative, so it decreases min
                min_offset += extent;
            }
        }
    }
    
    // Account for element size (last byte of last element)
    // Bounding box should include all bytes of the tensor
    if (tensor->elem_size > 0) {
        max_offset += tensor->elem_size - 1;
    }
    
    *out_min = min_offset;
    *out_max = max_offset;
}

void pto2_logical_tensor_update_bounding_box(PTO2LogicalTensor* tensor) {
    pto2_logical_tensor_get_bounding_box(
        tensor, 
        &tensor->min_byte_offset, 
        &tensor->max_byte_offset
    );
}

// =============================================================================
// Memory Overlap Detection
// =============================================================================

bool pto2_logical_tensor_overlap_fast(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
) {
    // Different raw storage => no overlap possible
    if (a->raw_base != b->raw_base) {
        return false;
    }
    
    // Check bounding box intersection
    // Overlap if: (a.min <= b.max) AND (b.min <= a.max)
    return (a->min_byte_offset <= b->max_byte_offset) &&
           (b->min_byte_offset <= a->max_byte_offset);
}

// =============================================================================
// GCD-Based Exact Overlap Detection
// =============================================================================

/**
 * Compute Greatest Common Divisor using Euclidean algorithm
 */
static int64_t gcd(int64_t a, int64_t b) {
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        int64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

/**
 * Extended Euclidean algorithm: find x, y such that a*x + b*y = gcd(a,b)
 * Returns gcd(a, b)
 */
static int64_t extended_gcd(int64_t a, int64_t b, int64_t* x, int64_t* y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    int64_t x1, y1;
    int64_t g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

bool pto2_overlap_1d_exact(
    int64_t offset_a, int64_t stride_a, int64_t size_a,
    int64_t offset_b, int64_t stride_b, int64_t size_b
) {
    // Handle edge cases
    if (size_a <= 0 || size_b <= 0) {
        return false;
    }
    
    // If both strides are 0, check if they touch the same point
    if (stride_a == 0 && stride_b == 0) {
        return offset_a == offset_b;
    }
    
    // If one stride is 0, check if the fixed point is in the other's range
    if (stride_a == 0) {
        // A is a single point at offset_a
        // Check if offset_a = offset_b + j * stride_b for some 0 <= j < size_b
        int64_t diff = offset_a - offset_b;
        if (stride_b == 0) {
            return diff == 0;
        }
        if (diff % stride_b != 0) {
            return false;
        }
        int64_t j = diff / stride_b;
        return j >= 0 && j < size_b;
    }
    
    if (stride_b == 0) {
        // B is a single point at offset_b
        int64_t diff = offset_b - offset_a;
        if (diff % stride_a != 0) {
            return false;
        }
        int64_t i = diff / stride_a;
        return i >= 0 && i < size_a;
    }
    
    // General case: both strides non-zero
    // We need to find i, j such that:
    //   offset_a + i * stride_a = offset_b + j * stride_b
    //   => i * stride_a - j * stride_b = offset_b - offset_a = delta
    // This has integer solutions iff gcd(stride_a, stride_b) | delta
    
    int64_t delta = offset_b - offset_a;
    int64_t g = gcd(stride_a, stride_b);
    
    if (delta % g != 0) {
        // No integer solution exists
        return false;
    }
    
    // Find one particular solution using extended GCD
    // stride_a * x + (-stride_b) * y = gcd(stride_a, stride_b)
    // Scale to get: stride_a * i0 - stride_b * j0 = delta
    int64_t x0, y0;
    extended_gcd(stride_a, stride_b, &x0, &y0);
    
    int64_t scale = delta / g;
    int64_t i0 = x0 * scale;
    int64_t j0 = -y0 * scale;  // Note: we have stride_a * x - stride_b * y, so j = -y
    
    // General solution: i = i0 + k * (stride_b / g)
    //                   j = j0 + k * (stride_a / g)
    // for any integer k
    
    int64_t step_i = stride_b / g;  // Step for i when k changes by 1
    int64_t step_j = stride_a / g;  // Step for j when k changes by 1
    
    // We need to find k such that:
    //   0 <= i0 + k * step_i < size_a
    //   0 <= j0 + k * step_j < size_b
    
    // Convert to range for k:
    // For i: -i0 / step_i <= k < (size_a - i0) / step_i  (if step_i > 0)
    // For j: -j0 / step_j <= k < (size_b - j0) / step_j  (if step_j > 0)
    
    // Compute k range for valid i
    int64_t k_min_i, k_max_i;
    if (step_i > 0) {
        // k >= ceil(-i0 / step_i) and k < floor((size_a - 1 - i0) / step_i) + 1
        k_min_i = (-i0 + step_i - 1) / step_i;  // ceil division
        if (-i0 >= 0) k_min_i = -i0 / step_i;   // fix for positive numerator
        k_max_i = (size_a - 1 - i0) / step_i;
    } else if (step_i < 0) {
        step_i = -step_i;  // Make positive for easier calculation
        // i0 - k * step_i >= 0  => k <= i0 / step_i
        // i0 - k * step_i < size_a  => k > (i0 - size_a + 1) / step_i
        k_max_i = i0 / step_i;
        k_min_i = (i0 - size_a + 1 + step_i) / step_i;  // ceil
        if (i0 - size_a + 1 >= 0) k_min_i = (i0 - size_a + 1) / step_i;
        step_i = -step_i;  // Restore sign
    } else {
        // step_i == 0, i is fixed at i0
        if (i0 >= 0 && i0 < size_a) {
            k_min_i = INT64_MIN;
            k_max_i = INT64_MAX;
        } else {
            return false;
        }
    }
    
    // Compute k range for valid j
    int64_t k_min_j, k_max_j;
    if (step_j > 0) {
        k_min_j = (-j0 + step_j - 1) / step_j;
        if (-j0 >= 0) k_min_j = -j0 / step_j;
        k_max_j = (size_b - 1 - j0) / step_j;
    } else if (step_j < 0) {
        step_j = -step_j;
        k_max_j = j0 / step_j;
        k_min_j = (j0 - size_b + 1 + step_j) / step_j;
        if (j0 - size_b + 1 >= 0) k_min_j = (j0 - size_b + 1) / step_j;
        step_j = -step_j;
    } else {
        if (j0 >= 0 && j0 < size_b) {
            k_min_j = INT64_MIN;
            k_max_j = INT64_MAX;
        } else {
            return false;
        }
    }
    
    // Find intersection of k ranges
    int64_t k_min = (k_min_i > k_min_j) ? k_min_i : k_min_j;
    int64_t k_max = (k_max_i < k_max_j) ? k_max_i : k_max_j;
    
    // If there exists a valid k in [k_min, k_max], overlap exists
    return k_min <= k_max;
}

bool pto2_logical_tensor_overlap_exact(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
) {
    // Different raw storage => no overlap
    if (a->raw_base != b->raw_base) {
        return false;
    }
    
    // Bounding box check
    if (a->max_byte_offset < b->min_byte_offset ||
        b->max_byte_offset < a->min_byte_offset) {
        return false;
    }
    
    // For contiguous tensors, bounding box is exact
    if (a->is_contiguous && b->is_contiguous) {
        int64_t a_start = a->storage_offset;
        int64_t a_end = a_start + a->numel * a->elem_size;
        int64_t b_start = b->storage_offset;
        int64_t b_end = b_start + b->numel * b->elem_size;
        return (a_start < b_end) && (b_start < a_end);
    }
    
    // For non-contiguous tensors: use conservative bounding box
    // Note: GCD-based methods are disabled because they don't work correctly
    // when the lowest dimension has stride=1 (the GCD becomes 1, making
    // the test ineffective). Bounding box is safe (no false negatives).
    return true;
}

bool pto2_tensormap_entry_overlap_fast(
    const PTO2TensorMapEntryEx* a,
    const PTO2TensorMapEntryEx* b
) {
    // Different raw storage => no overlap
    if (a->raw_base != b->raw_base) {
        return false;
    }
    
    // Check bounding box intersection
    return (a->min_byte_offset <= b->max_byte_offset) &&
           (b->min_byte_offset <= a->max_byte_offset);
}

bool pto2_tensor_entry_overlap_fast(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
) {
    // Different raw storage => no overlap
    if (tensor->raw_base != entry->raw_base) {
        return false;
    }
    
    // Check bounding box intersection
    return (tensor->min_byte_offset <= entry->max_byte_offset) &&
           (entry->min_byte_offset <= tensor->max_byte_offset);
}

// =============================================================================
// Hybrid Overlap Detection
// =============================================================================

void pto2_entry_to_logical_tensor(
    const PTO2TensorMapEntryEx* entry,
    PTO2LogicalTensor* tensor
) {
    tensor->raw_base = entry->raw_base;
    tensor->raw_total_size = entry->raw_total_size;
    tensor->storage_offset = entry->storage_offset;
    tensor->min_byte_offset = entry->min_byte_offset;
    tensor->max_byte_offset = entry->max_byte_offset;
    tensor->ndim = entry->ndim;
    
    for (int32_t d = 0; d < entry->ndim && d < PTO2_MAX_TENSOR_DIM; d++) {
        tensor->shape[d] = entry->shape[d];
        tensor->strides[d] = entry->strides[d];
    }
    for (int32_t d = entry->ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        tensor->shape[d] = 0;
        tensor->strides[d] = 0;
    }
    
    // Derive other fields
    tensor->elem_size = (entry->ndim > 0 && entry->shape[entry->ndim - 1] > 0) 
                        ? entry->strides[entry->ndim - 1] : 1;
    tensor->numel = pto2_logical_tensor_numel(tensor);
    tensor->extraction_type = entry->is_deep_copy ? PTO2_TENSOR_DEEP_VIEW : PTO2_TENSOR_VIEW;
    tensor->is_contiguous = entry->is_simple;
}

bool pto2_logical_tensor_overlap_hybrid(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
) {
    // 1. Different storage => no overlap
    if (a->raw_base != b->raw_base) {
        return false;
    }
    
    // 2. Bounding box quick rejection
    if (a->max_byte_offset < b->min_byte_offset ||
        b->max_byte_offset < a->min_byte_offset) {
        return false;
    }
    
    // 3. Both contiguous => bounding box is exact
    if (a->is_contiguous && b->is_contiguous) {
        return true;  // Bounding box overlap = real overlap
    }
    
    // 4. At least one non-contiguous => conservative bounding box
    //    Note: GCD methods disabled (don't work when stride=1)
    return true;
}

bool pto2_tensor_entry_overlap_hybrid(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
) {
    // 1. Different storage => no overlap
    if (tensor->raw_base != entry->raw_base) {
        return false;
    }
    
    // 2. Bounding box quick rejection
    if (tensor->max_byte_offset < entry->min_byte_offset ||
        entry->max_byte_offset < tensor->min_byte_offset) {
        return false;
    }
    
    // 3. Both contiguous => bounding box is exact
    if (tensor->is_contiguous && entry->is_simple) {
        return true;
    }
    
    // 4. At least one non-contiguous => use HBB if available
    //    Fall back to conservative bounding box if layout_depth not set
    if (tensor->layout_depth > 0 && entry->layout_depth > 0) {
        // Use HBB overlap detection
        return pto2_tensor_entry_overlap_hbb(tensor, entry);
    }
    
    // Conservative: bounding box overlap -> assume real overlap
    return true;
}

// =============================================================================
// Hierarchical Bounding Box (HBB) Overlap Detection
// =============================================================================

/**
 * Compare two shapes for equality
 */
static bool shapes_equal(const PTO2LayoutOp* a, const PTO2LayoutOp* b) {
    if (a->reshape.ndim != b->reshape.ndim) {
        return false;
    }
    for (int32_t d = 0; d < a->reshape.ndim; d++) {
        if (a->reshape.shape[d] != b->reshape.shape[d]) {
            return false;
        }
    }
    return true;
}

/**
 * Compare two permutations for equality
 */
static bool perms_equal(const PTO2LayoutOp* a, const PTO2LayoutOp* b) {
    if (a->transpose.ndim != b->transpose.ndim) {
        return false;
    }
    for (int32_t d = 0; d < a->transpose.ndim; d++) {
        if (a->transpose.perm[d] != b->transpose.perm[d]) {
            return false;
        }
    }
    return true;
}

bool pto2_layout_history_overlap(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
) {
    // 1. Different raw storage => no overlap
    if (a->raw_base != b->raw_base) {
        return false;
    }
    
    // 2. If either has no layout history, fall back to bounding box
    if (a->layout_depth <= 0 || b->layout_depth <= 0) {
        // Use simple bounding box check
        if (a->max_byte_offset < b->min_byte_offset ||
            b->max_byte_offset < a->min_byte_offset) {
            return false;
        }
        return true;  // Conservative
    }
    
    // 3. Compare layout history level by level
    int32_t min_depth = (a->layout_depth < b->layout_depth) ? 
                         a->layout_depth : b->layout_depth;
    
    for (int32_t i = 0; i < min_depth; i++) {
        const PTO2LayoutOp* op_a = &a->layout_ops[i];
        const PTO2LayoutOp* op_b = &b->layout_ops[i];
        
        // Types must match
        if (op_a->type != op_b->type) {
            // Different operation types => conservative overlap
            return true;
        }
        
        switch (op_a->type) {
            case PTO2_LAYOUT_VIEW:
                // Check if bounding boxes are disjoint
                if (op_a->view.bbox_max < op_b->view.bbox_min ||
                    op_b->view.bbox_max < op_a->view.bbox_min) {
                    // Disjoint at this level => definitely no overlap!
                    return false;
                }
                // Overlapping bboxes, continue to next level
                break;
                
            case PTO2_LAYOUT_RESHAPE:
                // Different shapes => conservative overlap
                if (!shapes_equal(op_a, op_b)) {
                    return true;
                }
                // Same reshape, continue
                break;
                
            case PTO2_LAYOUT_TRANSPOSE:
                // Different permutations => conservative overlap
                if (!perms_equal(op_a, op_b)) {
                    return true;
                }
                // Same transpose, continue
                break;
        }
    }
    
    // All common levels passed without finding disjoint bbox
    // Check remaining levels if depths differ
    if (a->layout_depth != b->layout_depth) {
        // Different depths: the deeper tensor has more operations
        // This is a complex case, return conservative overlap
        return true;
    }
    
    // Same depth, all levels checked, bboxes overlap at all levels
    return true;
}

bool pto2_tensor_entry_overlap_hbb(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
) {
    // 1. Different raw storage => no overlap
    if (tensor->raw_base != entry->raw_base) {
        return false;
    }
    
    // 2. Quick bounding box rejection
    if (tensor->max_byte_offset < entry->min_byte_offset ||
        entry->max_byte_offset < tensor->min_byte_offset) {
        return false;
    }
    
    // 3. If either has no layout history, conservative overlap
    if (tensor->layout_depth <= 0 || entry->layout_depth <= 0) {
        return true;
    }
    
    // 4. Compare layout history level by level
    int32_t min_depth = (tensor->layout_depth < entry->layout_depth) ?
                         tensor->layout_depth : entry->layout_depth;
    
    for (int32_t i = 0; i < min_depth; i++) {
        const PTO2LayoutOp* op_a = &tensor->layout_ops[i];
        const PTO2LayoutOp* op_b = &entry->layout_ops[i];
        
        if (op_a->type != op_b->type) {
            return true;  // Different types => conservative
        }
        
        switch (op_a->type) {
            case PTO2_LAYOUT_VIEW:
                if (op_a->view.bbox_max < op_b->view.bbox_min ||
                    op_b->view.bbox_max < op_a->view.bbox_min) {
                    return false;  // Disjoint => no overlap
                }
                break;
                
            case PTO2_LAYOUT_RESHAPE:
                if (!shapes_equal(op_a, op_b)) {
                    return true;
                }
                break;
                
            case PTO2_LAYOUT_TRANSPOSE:
                if (!perms_equal(op_a, op_b)) {
                    return true;
                }
                break;
        }
    }
    
    return true;  // Conservative
}

void pto2_layout_history_init_raw(PTO2LogicalTensor* tensor) {
    // Simple/raw tensor: depth=1, single VIEW covering entire storage
    tensor->layout_depth = 1;
    tensor->layout_ops[0].type = PTO2_LAYOUT_VIEW;
    tensor->layout_ops[0].view.bbox_min = tensor->min_byte_offset;
    tensor->layout_ops[0].view.bbox_max = tensor->max_byte_offset;
    
    // Clear remaining ops
    for (int32_t i = 1; i < PTO2_MAX_LAYOUT_DEPTH; i++) {
        tensor->layout_ops[i].type = PTO2_LAYOUT_VIEW;
        tensor->layout_ops[i].view.bbox_min = 0;
        tensor->layout_ops[i].view.bbox_max = 0;
    }
}

bool pto2_layout_history_append_view(
    PTO2LogicalTensor* dst,
    const PTO2LogicalTensor* src,
    int64_t bbox_min,
    int64_t bbox_max
) {
    // Copy source layout history
    dst->layout_depth = src->layout_depth;
    for (int32_t i = 0; i < src->layout_depth && i < PTO2_MAX_LAYOUT_DEPTH; i++) {
        dst->layout_ops[i] = src->layout_ops[i];
    }
    
    // Check if we can append
    if (dst->layout_depth >= PTO2_MAX_LAYOUT_DEPTH) {
        // Max depth reached, cannot append more
        // The last level will be overwritten (loses precision but still safe)
        dst->layout_ops[PTO2_MAX_LAYOUT_DEPTH - 1].type = PTO2_LAYOUT_VIEW;
        dst->layout_ops[PTO2_MAX_LAYOUT_DEPTH - 1].view.bbox_min = bbox_min;
        dst->layout_ops[PTO2_MAX_LAYOUT_DEPTH - 1].view.bbox_max = bbox_max;
        return false;  // Indicate overflow
    }
    
    // Append VIEW operation
    dst->layout_ops[dst->layout_depth].type = PTO2_LAYOUT_VIEW;
    dst->layout_ops[dst->layout_depth].view.bbox_min = bbox_min;
    dst->layout_ops[dst->layout_depth].view.bbox_max = bbox_max;
    dst->layout_depth++;
    
    return true;
}

bool pto2_layout_history_append_reshape(
    PTO2LogicalTensor* dst,
    const PTO2LogicalTensor* src,
    const int64_t* shape,
    int32_t ndim
) {
    // Copy source layout history
    dst->layout_depth = src->layout_depth;
    for (int32_t i = 0; i < src->layout_depth && i < PTO2_MAX_LAYOUT_DEPTH; i++) {
        dst->layout_ops[i] = src->layout_ops[i];
    }
    
    // Check if we can append
    if (dst->layout_depth >= PTO2_MAX_LAYOUT_DEPTH) {
        return false;
    }
    
    // Append RESHAPE operation
    PTO2LayoutOp* op = &dst->layout_ops[dst->layout_depth];
    op->type = PTO2_LAYOUT_RESHAPE;
    op->reshape.ndim = (ndim > PTO2_MAX_TENSOR_DIM) ? PTO2_MAX_TENSOR_DIM : ndim;
    for (int32_t d = 0; d < op->reshape.ndim; d++) {
        op->reshape.shape[d] = shape[d];
    }
    dst->layout_depth++;
    
    return true;
}

bool pto2_layout_history_append_transpose(
    PTO2LogicalTensor* dst,
    const PTO2LogicalTensor* src,
    const int32_t* perm,
    int32_t ndim
) {
    // Copy source layout history
    dst->layout_depth = src->layout_depth;
    for (int32_t i = 0; i < src->layout_depth && i < PTO2_MAX_LAYOUT_DEPTH; i++) {
        dst->layout_ops[i] = src->layout_ops[i];
    }
    
    // Check if we can append
    if (dst->layout_depth >= PTO2_MAX_LAYOUT_DEPTH) {
        return false;
    }
    
    // Append TRANSPOSE operation
    PTO2LayoutOp* op = &dst->layout_ops[dst->layout_depth];
    op->type = PTO2_LAYOUT_TRANSPOSE;
    op->transpose.ndim = (ndim > PTO2_MAX_TENSOR_DIM) ? PTO2_MAX_TENSOR_DIM : ndim;
    for (int32_t d = 0; d < op->transpose.ndim; d++) {
        op->transpose.perm[d] = perm[d];
    }
    dst->layout_depth++;
    
    return true;
}

// =============================================================================
// Utility Functions
// =============================================================================

bool pto2_logical_tensor_is_contiguous(const PTO2LogicalTensor* tensor) {
    if (tensor->ndim == 0) {
        return true;  // Scalar is always contiguous
    }
    
    // For row-major (C-style) contiguity:
    // strides[ndim-1] should be elem_size
    // strides[d] should be strides[d+1] * shape[d+1]
    
    int64_t expected_stride = tensor->elem_size;
    
    for (int32_t d = tensor->ndim - 1; d >= 0; d--) {
        if (tensor->strides[d] != expected_stride) {
            return false;
        }
        expected_stride *= tensor->shape[d];
    }
    
    return true;
}

int64_t pto2_logical_tensor_numel(const PTO2LogicalTensor* tensor) {
    if (tensor->ndim == 0) {
        return 1;  // Scalar has 1 element
    }
    
    int64_t count = 1;
    for (int32_t d = 0; d < tensor->ndim; d++) {
        count *= tensor->shape[d];
    }
    return count;
}

int64_t pto2_logical_tensor_storage_size(const PTO2LogicalTensor* tensor) {
    return pto2_logical_tensor_numel(tensor) * tensor->elem_size;
}

void pto2_logical_tensor_print(const PTO2LogicalTensor* tensor, const char* name) {
    const char* extraction_names[] = {
        "RAW", "VIEW", "RESHAPE", "TRANSPOSE",
        "DEEP_VIEW", "DEEP_RESHAPE", "DEEP_TRANSPOSE"
    };
    
    printf("LogicalTensor %s:\n", name ? name : "(unnamed)");
    printf("  raw_base: %p, raw_total_size: %ld\n", 
           tensor->raw_base, (long)tensor->raw_total_size);
    printf("  storage_offset: %ld\n", (long)tensor->storage_offset);
    printf("  ndim: %d, elem_size: %ld, numel: %ld\n",
           tensor->ndim, (long)tensor->elem_size, (long)tensor->numel);
    
    printf("  shape:   [");
    for (int32_t d = 0; d < tensor->ndim; d++) {
        printf("%ld%s", (long)tensor->shape[d], d < tensor->ndim-1 ? ", " : "");
    }
    printf("]\n");
    
    printf("  strides: [");
    for (int32_t d = 0; d < tensor->ndim; d++) {
        printf("%ld%s", (long)tensor->strides[d], d < tensor->ndim-1 ? ", " : "");
    }
    printf("]\n");
    
    printf("  bounding_box: [%ld, %ld]\n", 
           (long)tensor->min_byte_offset, (long)tensor->max_byte_offset);
    printf("  extraction: %s, contiguous: %s\n",
           extraction_names[tensor->extraction_type],
           tensor->is_contiguous ? "yes" : "no");
}

// =============================================================================
// Logical Tensor Creation
// =============================================================================

void pto2_logical_tensor_init_raw(
    PTO2LogicalTensor* tensor,
    void* base_ptr,
    const int64_t* shape,
    int32_t ndim,
    int64_t elem_size
) {
    // Validate inputs
    if (ndim > PTO2_MAX_TENSOR_DIM) {
        ndim = PTO2_MAX_TENSOR_DIM;
    }
    
    // Compute total elements
    int64_t numel = 1;
    for (int32_t d = 0; d < ndim; d++) {
        numel *= shape[d];
    }
    
    // Initialize raw tensor reference
    tensor->raw_base = base_ptr;
    tensor->raw_total_size = numel * elem_size;
    tensor->storage_offset = 0;
    
    // Copy shape
    tensor->ndim = ndim;
    for (int32_t d = 0; d < ndim; d++) {
        tensor->shape[d] = shape[d];
    }
    for (int32_t d = ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        tensor->shape[d] = 0;
    }
    
    // Compute row-major strides
    int64_t stride = elem_size;
    for (int32_t d = ndim - 1; d >= 0; d--) {
        tensor->strides[d] = stride;
        stride *= shape[d];
    }
    for (int32_t d = ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        tensor->strides[d] = 0;
    }
    
    // Set element info
    tensor->elem_size = elem_size;
    tensor->numel = numel;
    
    // Set extraction info
    tensor->extraction_type = PTO2_TENSOR_RAW;
    tensor->is_contiguous = true;
    
    // Compute bounding box
    pto2_logical_tensor_update_bounding_box(tensor);
    
    // Initialize HBB layout history (depth=1, single VIEW)
    pto2_layout_history_init_raw(tensor);
}

void pto2_logical_tensor_init(
    PTO2LogicalTensor* tensor,
    void* raw_base,
    int64_t raw_total_size,
    int64_t storage_offset,
    const int64_t* shape,
    const int64_t* strides,
    int32_t ndim,
    int64_t elem_size,
    PTO2TensorExtractionType extraction
) {
    // Validate inputs
    if (ndim > PTO2_MAX_TENSOR_DIM) {
        ndim = PTO2_MAX_TENSOR_DIM;
    }
    
    // Set raw tensor reference
    tensor->raw_base = raw_base;
    tensor->raw_total_size = raw_total_size;
    tensor->storage_offset = storage_offset;
    
    // Copy shape and strides
    tensor->ndim = ndim;
    for (int32_t d = 0; d < ndim; d++) {
        tensor->shape[d] = shape[d];
        tensor->strides[d] = strides[d];
    }
    for (int32_t d = ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        tensor->shape[d] = 0;
        tensor->strides[d] = 0;
    }
    
    // Set element info
    tensor->elem_size = elem_size;
    tensor->numel = pto2_logical_tensor_numel(tensor);
    
    // Set extraction info
    tensor->extraction_type = extraction;
    tensor->is_contiguous = pto2_logical_tensor_is_contiguous(tensor);
    
    // Compute bounding box
    pto2_logical_tensor_update_bounding_box(tensor);
    
    // Initialize HBB layout history (depth=1, single VIEW)
    pto2_layout_history_init_raw(tensor);
}

// =============================================================================
// Shallow Extraction Operations
// =============================================================================

bool pto2_logical_tensor_view(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int64_t* start,
    const int64_t* shape,
    int32_t ndim
) {
    // Validate dimensions
    if (ndim > src->ndim || ndim > PTO2_MAX_TENSOR_DIM) {
        return false;
    }
    
    // Validate bounds
    for (int32_t d = 0; d < ndim; d++) {
        if (start[d] < 0 || start[d] + shape[d] > src->shape[d]) {
            return false;
        }
    }
    
    // Compute new storage offset
    int64_t new_offset = src->storage_offset;
    for (int32_t d = 0; d < ndim; d++) {
        new_offset += start[d] * src->strides[d];
    }
    
    // Initialize view tensor
    dst->raw_base = src->raw_base;
    dst->raw_total_size = src->raw_total_size;
    dst->storage_offset = new_offset;
    dst->ndim = ndim;
    dst->elem_size = src->elem_size;
    
    // Copy shape and strides (strides unchanged for view)
    for (int32_t d = 0; d < ndim; d++) {
        dst->shape[d] = shape[d];
        dst->strides[d] = src->strides[d];
    }
    for (int32_t d = ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        dst->shape[d] = 0;
        dst->strides[d] = 0;
    }
    
    // Compute derived fields
    dst->numel = pto2_logical_tensor_numel(dst);
    dst->extraction_type = PTO2_TENSOR_VIEW;
    dst->is_contiguous = pto2_logical_tensor_is_contiguous(dst);
    pto2_logical_tensor_update_bounding_box(dst);
    
    return true;
}

bool pto2_logical_tensor_reshape(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int64_t* shape,
    int32_t ndim
) {
    // Reshape requires contiguous tensor
    if (!src->is_contiguous) {
        return false;
    }
    
    // Validate dimensions
    if (ndim > PTO2_MAX_TENSOR_DIM) {
        return false;
    }
    
    // Validate total elements match
    int64_t new_numel = 1;
    for (int32_t d = 0; d < ndim; d++) {
        new_numel *= shape[d];
    }
    if (new_numel != src->numel) {
        return false;
    }
    
    // Initialize reshaped tensor
    dst->raw_base = src->raw_base;
    dst->raw_total_size = src->raw_total_size;
    dst->storage_offset = src->storage_offset;
    dst->ndim = ndim;
    dst->elem_size = src->elem_size;
    dst->numel = new_numel;
    
    // Copy shape
    for (int32_t d = 0; d < ndim; d++) {
        dst->shape[d] = shape[d];
    }
    for (int32_t d = ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        dst->shape[d] = 0;
    }
    
    // Compute new row-major strides
    int64_t stride = src->elem_size;
    for (int32_t d = ndim - 1; d >= 0; d--) {
        dst->strides[d] = stride;
        stride *= dst->shape[d];
    }
    for (int32_t d = ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        dst->strides[d] = 0;
    }
    
    // Set derived fields
    dst->extraction_type = PTO2_TENSOR_RESHAPE;
    dst->is_contiguous = true;  // Reshaped from contiguous is always contiguous
    pto2_logical_tensor_update_bounding_box(dst);
    
    return true;
}

bool pto2_logical_tensor_transpose(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int32_t* perm
) {
    if (src->ndim > PTO2_MAX_TENSOR_DIM) {
        return false;
    }
    
    // Temporary permutation array
    int32_t actual_perm[PTO2_MAX_TENSOR_DIM];
    
    if (perm == NULL) {
        // Default: reverse all dimensions
        for (int32_t d = 0; d < src->ndim; d++) {
            actual_perm[d] = src->ndim - 1 - d;
        }
    } else {
        // Validate permutation
        bool used[PTO2_MAX_TENSOR_DIM] = {false};
        for (int32_t d = 0; d < src->ndim; d++) {
            if (perm[d] < 0 || perm[d] >= src->ndim || used[perm[d]]) {
                return false;  // Invalid permutation
            }
            used[perm[d]] = true;
            actual_perm[d] = perm[d];
        }
    }
    
    // Initialize transposed tensor
    dst->raw_base = src->raw_base;
    dst->raw_total_size = src->raw_total_size;
    dst->storage_offset = src->storage_offset;
    dst->ndim = src->ndim;
    dst->elem_size = src->elem_size;
    dst->numel = src->numel;
    
    // Permute shape and strides
    for (int32_t d = 0; d < src->ndim; d++) {
        dst->shape[d] = src->shape[actual_perm[d]];
        dst->strides[d] = src->strides[actual_perm[d]];
    }
    for (int32_t d = src->ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        dst->shape[d] = 0;
        dst->strides[d] = 0;
    }
    
    // Set derived fields
    dst->extraction_type = PTO2_TENSOR_TRANSPOSE;
    dst->is_contiguous = pto2_logical_tensor_is_contiguous(dst);
    pto2_logical_tensor_update_bounding_box(dst);
    
    return true;
}

// =============================================================================
// Deep Extraction Operations
// =============================================================================

bool pto2_logical_tensor_clone(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    void* new_base
) {
    if (new_base == NULL) {
        return false;
    }
    
    // Copy data element by element (handles non-contiguous)
    int64_t dst_offset = 0;
    char* dst_ptr = (char*)new_base;
    
    // For simplicity, iterate through all elements using indices
    // This is O(numel) but handles any stride pattern
    int64_t indices[PTO2_MAX_TENSOR_DIM] = {0};
    
    for (int64_t i = 0; i < src->numel; i++) {
        // Compute source offset
        int64_t src_offset = src->storage_offset;
        for (int32_t d = 0; d < src->ndim; d++) {
            src_offset += indices[d] * src->strides[d];
        }
        
        // Copy element
        memcpy(dst_ptr + dst_offset, 
               (char*)src->raw_base + src_offset, 
               src->elem_size);
        dst_offset += src->elem_size;
        
        // Advance indices (like odometer)
        for (int32_t d = src->ndim - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < src->shape[d]) {
                break;
            }
            indices[d] = 0;
        }
    }
    
    // Initialize dst as a raw contiguous tensor
    pto2_logical_tensor_init_raw(dst, new_base, src->shape, src->ndim, src->elem_size);
    
    // Mark as deep copy
    dst->extraction_type = PTO2_TENSOR_DEEP_VIEW;  // Or appropriate deep type
    
    return true;
}

bool pto2_logical_tensor_contiguous(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    void* new_base
) {
    if (src->is_contiguous) {
        // Already contiguous - shallow copy
        *dst = *src;
        return true;
    }
    
    // Not contiguous - need to clone
    if (new_base == NULL) {
        return false;
    }
    
    return pto2_logical_tensor_clone(src, dst, new_base);
}

// =============================================================================
// Conversion Functions
// =============================================================================

void pto2_logical_tensor_to_entry(
    const PTO2LogicalTensor* tensor,
    PTO2TensorMapEntryEx* entry,
    int32_t producer_task_id
) {
    entry->raw_base = tensor->raw_base;
    entry->raw_total_size = tensor->raw_total_size;
    entry->min_byte_offset = tensor->min_byte_offset;
    entry->max_byte_offset = tensor->max_byte_offset;
    entry->storage_offset = tensor->storage_offset;
    
    for (int32_t d = 0; d < tensor->ndim; d++) {
        entry->shape[d] = tensor->shape[d];
        entry->strides[d] = tensor->strides[d];
    }
    for (int32_t d = tensor->ndim; d < PTO2_MAX_TENSOR_DIM; d++) {
        entry->shape[d] = 0;
        entry->strides[d] = 0;
    }
    entry->ndim = tensor->ndim;
    
    entry->producer_task_id = producer_task_id;
    entry->next_in_bucket = -1;
    entry->next_in_task = -1;
    entry->in_bucket = false;
    
    // Deep copies are independent - mark them
    entry->is_deep_copy = (tensor->extraction_type >= PTO2_TENSOR_DEEP_VIEW);
}

void pto2_region_to_logical_tensor(
    const PTO2TensorRegion* region,
    PTO2LogicalTensor* tensor
) {
    // Create a 1D logical tensor from legacy region
    tensor->raw_base = region->base_ptr;
    tensor->raw_total_size = region->size;  // Approximate, may be larger
    
    // Compute storage offset from tile_index and offset
    // Assuming tile_index is a multiplier of some tile size
    tensor->storage_offset = region->offset;  // Simplified: just use offset
    
    // 1D tensor with size elements of 1 byte each
    tensor->ndim = 1;
    tensor->shape[0] = region->size;
    tensor->strides[0] = 1;
    tensor->elem_size = 1;
    tensor->numel = region->size;
    
    for (int32_t d = 1; d < PTO2_MAX_TENSOR_DIM; d++) {
        tensor->shape[d] = 0;
        tensor->strides[d] = 0;
    }
    
    tensor->extraction_type = PTO2_TENSOR_RAW;
    tensor->is_contiguous = true;
    
    pto2_logical_tensor_update_bounding_box(tensor);
}
