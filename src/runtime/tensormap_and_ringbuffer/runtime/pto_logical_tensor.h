/**
 * PTO Runtime2 - Logical Tensor Operations
 * 
 * Provides APIs for:
 * 1. Creating and manipulating logical tensors (view, reshape, transpose)
 * 2. Computing bounding boxes for memory overlap detection
 * 3. Checking memory overlap between tensors
 * 
 * Key concepts:
 * - Raw tensor: owns the actual memory allocation
 * - Logical tensor: a "view" into raw tensor with specific layout
 * - Shallow extraction: shares storage (view, reshape, transpose)
 * - Deep extraction: copies data to new storage (clone, contiguous)
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_LOGICAL_TENSOR_H
#define PTO_LOGICAL_TENSOR_H

#include "pto_runtime2_types.h"
#include <stdbool.h>

// =============================================================================
// Bounding Box Computation
// =============================================================================

/**
 * Compute bounding box (min/max byte offset) for a logical tensor
 * 
 * The bounding box is the smallest contiguous memory range that contains
 * all elements of the tensor. Used for fast overlap detection.
 * 
 * For tensor T with shape[d] and strides[d]:
 *   min_offset = storage_offset + sum(min(0, (shape[d]-1)*strides[d]))
 *   max_offset = storage_offset + sum(max(0, (shape[d]-1)*strides[d]))
 * 
 * @param tensor  Logical tensor to compute bounding box for
 * @param out_min Output: minimum byte offset (relative to raw_base)
 * @param out_max Output: maximum byte offset (relative to raw_base)
 */
void pto2_logical_tensor_get_bounding_box(
    const PTO2LogicalTensor* tensor,
    int64_t* out_min,
    int64_t* out_max
);

/**
 * Update bounding box fields in logical tensor (in-place)
 * 
 * Computes and stores min_byte_offset and max_byte_offset.
 * Should be called after creating/modifying a logical tensor.
 * 
 * @param tensor  Logical tensor to update
 */
void pto2_logical_tensor_update_bounding_box(PTO2LogicalTensor* tensor);

// =============================================================================
// Memory Overlap Detection
// =============================================================================

/**
 * Check if two logical tensors have overlapping memory (fast bounding box check)
 * 
 * This is a conservative check - may return true even if tensors don't
 * actually overlap (false positives possible for strided tensors).
 * 
 * Returns false if:
 *   - Different raw_base pointers (different storage)
 *   - Bounding boxes don't intersect
 * 
 * Returns true if:
 *   - Same raw_base AND bounding boxes intersect
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return true if tensors may overlap, false if definitely no overlap
 */
bool pto2_logical_tensor_overlap_fast(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
);

/**
 * Check if two 1D tensors have overlapping memory (exact GCD-based check)
 * 
 * Uses the Greatest Common Divisor (GCD) to determine if two strided
 * sequences share any common elements. This has NO false positives.
 * 
 * Mathematical basis:
 *   Tensor A touches offsets: offset_a + i * stride_a  (0 <= i < size_a)
 *   Tensor B touches offsets: offset_b + j * stride_b  (0 <= j < size_b)
 *   
 *   Overlap exists iff: offset_a + i*stride_a = offset_b + j*stride_b
 *   Rearranged: i*stride_a - j*stride_b = offset_b - offset_a
 *   
 *   Solution exists iff: gcd(stride_a, stride_b) divides (offset_b - offset_a)
 *   AND solution falls within valid index ranges [0, size_a) and [0, size_b)
 * 
 * @param offset_a  Starting byte offset of tensor A
 * @param stride_a  Byte stride of tensor A
 * @param size_a    Number of elements in tensor A
 * @param offset_b  Starting byte offset of tensor B
 * @param stride_b  Byte stride of tensor B
 * @param size_b    Number of elements in tensor B
 * @return true if tensors overlap, false otherwise (exact, no false positives)
 */
bool pto2_overlap_1d_exact(
    int64_t offset_a, int64_t stride_a, int64_t size_a,
    int64_t offset_b, int64_t stride_b, int64_t size_b
);

/**
 * Check if two logical tensors have overlapping memory (exact GCD-based check)
 * 
 * For multi-dimensional tensors, flattens to 1D and uses GCD analysis.
 * This is slower than bounding box but has NO false positives.
 * 
 * Use this when:
 *   - Bounding box check returns true (potential overlap)
 *   - You need exact answer (no false positives)
 *   - Performance cost is acceptable
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return true if tensors definitely overlap, false if definitely no overlap
 */
bool pto2_logical_tensor_overlap_exact(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
);

/**
 * Check if two extended tensormap entries have overlapping memory
 * 
 * Uses bounding box stored in entry for fast comparison.
 * 
 * @param a First entry
 * @param b Second entry
 * @return true if entries may overlap
 */
bool pto2_tensormap_entry_overlap_fast(
    const PTO2TensorMapEntryEx* a,
    const PTO2TensorMapEntryEx* b
);

/**
 * Check if a logical tensor overlaps with a tensormap entry
 * 
 * @param tensor Logical tensor (input being looked up)
 * @param entry  TensorMap entry (output from previous task)
 * @return true if may overlap
 */
bool pto2_tensor_entry_overlap_fast(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
);

// =============================================================================
// Hybrid Overlap Detection (Recommended)
// =============================================================================

/**
 * Hybrid overlap detection for two logical tensors
 * 
 * This is the RECOMMENDED overlap detection function. It combines:
 * - Fast bounding box check for quick rejection
 * - Exact GCD-based check only when needed
 * 
 * Algorithm:
 * 1. Different raw_base -> no overlap
 * 2. Bounding boxes don't intersect -> no overlap
 * 3. Both tensors are contiguous (Simple) -> bounding box result is exact
 * 4. At least one non-contiguous (Complex) -> use GCD for exact result
 * 
 * This provides:
 * - O(1) fast path for Simple vs Simple (most common case)
 * - Exact results (no false positives) for all cases
 * - Automatic fallback to GCD for Complex tensors
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return true if tensors definitely overlap, false if definitely no overlap
 */
bool pto2_logical_tensor_overlap_hybrid(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
);

/**
 * Hybrid overlap detection for tensor vs tensormap entry
 * 
 * Same algorithm as pto2_logical_tensor_overlap_hybrid but works with
 * TensorMapEntryEx (which stores is_simple flag).
 * 
 * @param tensor Logical tensor (input being looked up)
 * @param entry  TensorMap entry (output from previous task)
 * @return true if definitely overlap, false if definitely no overlap
 */
bool pto2_tensor_entry_overlap_hybrid(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
);

/**
 * Convert TensorMapEntryEx to LogicalTensor
 * 
 * Used internally by hybrid detection when GCD check is needed.
 * Reconstructs a LogicalTensor from the stored entry data.
 * 
 * @param entry  Source entry
 * @param tensor Output tensor
 */
void pto2_entry_to_logical_tensor(
    const PTO2TensorMapEntryEx* entry,
    PTO2LogicalTensor* tensor
);

// =============================================================================
// Hierarchical Bounding Box (HBB) Overlap Detection
// =============================================================================

/**
 * Check if two layout histories have overlapping memory using HBB algorithm
 * 
 * This is the NEW overlap detection method that replaces GCD-based detection.
 * It works by comparing the derivation history of two tensors level by level.
 * 
 * Algorithm:
 * 1. Different raw_base -> no overlap
 * 2. For each level i from 0 to min(depth_a, depth_b):
 *    a. If types differ -> conservative overlap
 *    b. If both VIEW: check bbox intersection
 *       - If disjoint -> definitely no overlap (early exit)
 *       - If overlap -> continue to next level
 *    c. If both RESHAPE: check if shapes equal
 *       - If different -> conservative overlap
 *       - If same -> continue
 *    d. If both TRANSPOSE: check if perms equal
 *       - If different -> conservative overlap
 *       - If same -> continue
 * 3. All levels pass -> conservative overlap (may overlap)
 * 
 * Key insight: Simple tensor = depth=1, so this unifies simple/complex handling.
 * 
 * @param a First tensor's layout history
 * @param b Second tensor's layout history
 * @return true if may overlap, false if definitely no overlap
 */
bool pto2_layout_history_overlap(
    const PTO2LogicalTensor* a,
    const PTO2LogicalTensor* b
);

/**
 * Check if tensor overlaps with tensormap entry using HBB
 * 
 * @param tensor Logical tensor (input being looked up)
 * @param entry  TensorMap entry (output from previous task)
 * @return true if may overlap, false if definitely no overlap
 */
bool pto2_tensor_entry_overlap_hbb(
    const PTO2LogicalTensor* tensor,
    const PTO2TensorMapEntryEx* entry
);

/**
 * Initialize layout history for a raw (contiguous) tensor
 * 
 * Sets depth=1 with a single VIEW op covering the entire tensor.
 * This is the base case for all tensor derivations.
 * 
 * @param tensor  Tensor to initialize layout history for
 */
void pto2_layout_history_init_raw(PTO2LogicalTensor* tensor);

/**
 * Append a VIEW operation to layout history
 * 
 * Called when creating a view/slice of a tensor.
 * Records the bounding box of the view relative to parent.
 * 
 * @param dst     Destination tensor (view being created)
 * @param src     Source tensor
 * @param bbox_min Minimum byte offset of view
 * @param bbox_max Maximum byte offset of view
 * @return true on success, false if max depth exceeded
 */
bool pto2_layout_history_append_view(
    PTO2LogicalTensor* dst,
    const PTO2LogicalTensor* src,
    int64_t bbox_min,
    int64_t bbox_max
);

/**
 * Append a RESHAPE operation to layout history
 * 
 * Called when reshaping a tensor.
 * Records the new shape for comparison.
 * 
 * @param dst     Destination tensor (reshaped)
 * @param src     Source tensor
 * @param shape   New shape array
 * @param ndim    Number of dimensions
 * @return true on success, false if max depth exceeded
 */
bool pto2_layout_history_append_reshape(
    PTO2LogicalTensor* dst,
    const PTO2LogicalTensor* src,
    const int64_t* shape,
    int32_t ndim
);

/**
 * Append a TRANSPOSE operation to layout history
 * 
 * Called when transposing a tensor.
 * Records the permutation for comparison.
 * 
 * @param dst     Destination tensor (transposed)
 * @param src     Source tensor
 * @param perm    Permutation array
 * @param ndim    Number of dimensions
 * @return true on success, false if max depth exceeded
 */
bool pto2_layout_history_append_transpose(
    PTO2LogicalTensor* dst,
    const PTO2LogicalTensor* src,
    const int32_t* perm,
    int32_t ndim
);

// =============================================================================
// Logical Tensor Creation
// =============================================================================

/**
 * Initialize a logical tensor as a raw tensor (owns storage)
 * 
 * Creates a contiguous tensor that owns its storage.
 * The tensor will have row-major (C-style) layout.
 * 
 * @param tensor     Output tensor to initialize
 * @param base_ptr   Memory base pointer
 * @param shape      Array of dimension sizes
 * @param ndim       Number of dimensions
 * @param elem_size  Size of each element in bytes
 */
void pto2_logical_tensor_init_raw(
    PTO2LogicalTensor* tensor,
    void* base_ptr,
    const int64_t* shape,
    int32_t ndim,
    int64_t elem_size
);

/**
 * Initialize a logical tensor from raw components
 * 
 * For advanced use - directly specify all tensor parameters.
 * Automatically computes bounding box, numel, and contiguity.
 * 
 * @param tensor         Output tensor to initialize
 * @param raw_base       Raw tensor base pointer
 * @param raw_total_size Total size of raw tensor
 * @param storage_offset Byte offset from raw_base
 * @param shape          Array of dimension sizes
 * @param strides        Array of byte strides
 * @param ndim           Number of dimensions
 * @param elem_size      Size of each element in bytes
 * @param extraction     How this tensor was created
 */
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
);

// =============================================================================
// Shallow Extraction Operations (alias - share storage)
// =============================================================================

/**
 * Create a view (slice) of a tensor
 * 
 * Selects a subset of elements by specifying start indices and new shape.
 * The result shares storage with the original tensor.
 * 
 * @param src       Source tensor
 * @param dst       Output view tensor
 * @param start     Start index in each dimension (length = src->ndim)
 * @param shape     Shape of the view (length = src->ndim, or fewer if squeezing)
 * @param ndim      Number of dimensions in the view
 * @return true on success, false if indices out of bounds
 */
bool pto2_logical_tensor_view(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int64_t* start,
    const int64_t* shape,
    int32_t ndim
);

/**
 * Create a reshaped view of a tensor
 * 
 * Changes the logical shape without moving data.
 * Only works if tensor is contiguous.
 * 
 * @param src       Source tensor (must be contiguous)
 * @param dst       Output reshaped tensor
 * @param shape     New shape (product must equal src->numel)
 * @param ndim      Number of dimensions in new shape
 * @return true on success, false if not contiguous or shape mismatch
 */
bool pto2_logical_tensor_reshape(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int64_t* shape,
    int32_t ndim
);

/**
 * Create a transposed (permuted) view of a tensor
 * 
 * Reorders dimensions without moving data.
 * 
 * @param src       Source tensor
 * @param dst       Output transposed tensor
 * @param perm      Permutation of dimensions (e.g., [1,0] for 2D transpose)
 *                  If NULL, reverses all dimensions
 * @return true on success, false if invalid permutation
 */
bool pto2_logical_tensor_transpose(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    const int32_t* perm
);

// =============================================================================
// Deep Extraction Operations (copy - new storage)
// =============================================================================

/**
 * Create a deep copy (clone) of a tensor
 * 
 * Allocates new storage and copies all elements.
 * The result is independent of the original.
 * 
 * @param src       Source tensor
 * @param dst       Output cloned tensor
 * @param new_base  Pre-allocated memory for the clone (size >= src->numel * src->elem_size)
 * @return true on success
 */
bool pto2_logical_tensor_clone(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    void* new_base
);

/**
 * Make a tensor contiguous
 * 
 * If already contiguous, returns a reference (shallow copy).
 * If not contiguous, allocates new storage and copies data.
 * 
 * @param src       Source tensor
 * @param dst       Output contiguous tensor
 * @param new_base  Pre-allocated memory (only used if src not contiguous)
 *                  Can be NULL if src is already contiguous
 * @return true on success
 */
bool pto2_logical_tensor_contiguous(
    const PTO2LogicalTensor* src,
    PTO2LogicalTensor* dst,
    void* new_base
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Check if tensor is contiguous in memory
 * 
 * A tensor is contiguous if elements are stored without gaps in row-major order.
 * 
 * @param tensor Tensor to check
 * @return true if contiguous
 */
bool pto2_logical_tensor_is_contiguous(const PTO2LogicalTensor* tensor);

/**
 * Compute total number of elements in tensor
 * 
 * @param tensor Tensor to count
 * @return Total number of elements (product of shape dimensions)
 */
int64_t pto2_logical_tensor_numel(const PTO2LogicalTensor* tensor);

/**
 * Compute required storage size for a contiguous copy
 * 
 * @param tensor Tensor to compute size for
 * @return Size in bytes needed for contiguous storage
 */
int64_t pto2_logical_tensor_storage_size(const PTO2LogicalTensor* tensor);

/**
 * Print tensor info for debugging
 * 
 * @param tensor Tensor to print
 * @param name   Name to print (can be NULL)
 */
void pto2_logical_tensor_print(const PTO2LogicalTensor* tensor, const char* name);

// =============================================================================
// Conversion Functions
// =============================================================================

/**
 * Convert LogicalTensor to TensorMapEntryEx (for insertion)
 * 
 * Extracts the necessary fields from a logical tensor for tensormap storage.
 * 
 * @param tensor           Source logical tensor
 * @param entry            Output entry to fill
 * @param producer_task_id Task ID of the producer
 */
void pto2_logical_tensor_to_entry(
    const PTO2LogicalTensor* tensor,
    PTO2TensorMapEntryEx* entry,
    int32_t producer_task_id
);

/**
 * Convert legacy TensorRegion to LogicalTensor (for compatibility)
 * 
 * Creates a 1D logical tensor from a simple region descriptor.
 * 
 * @param region  Legacy tensor region
 * @param tensor  Output logical tensor
 */
void pto2_region_to_logical_tensor(
    const PTO2TensorRegion* region,
    PTO2LogicalTensor* tensor
);

#endif // PTO_LOGICAL_TENSOR_H
