/**
 * Orchestration Build Graph Types - Data structures for orchestration runtime extensions
 *
 * Standalone header defining orchestration-specific types for:
 * - PTOParam: Aggregated parameter container for pto_submit_task API
 *
 * Tensor descriptor types (Tensor, PTOBufferHandle, PTOOverlapStrategy) are
 * defined in tensor.h.
 *
 * This header is independent of orch_build_graph_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorPair, HostApi).
 */

#ifndef ORCH_BUILD_GRAPH_PTO_TYPES_H
#define ORCH_BUILD_GRAPH_PTO_TYPES_H

#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "tensor.h"

// =============================================================================
// Parameter Types (for pto_submit_task API)
// =============================================================================

/**
 * Parameter Type - Distinguishes inputs, outputs, and in-place updates
 */
enum class PTOParamType : int32_t {
    INPUT = 0,   // Read-only input buffer
    OUTPUT = 1,  // Write-only output buffer (NULL addr: runtime allocates; non-NULL: use as-is)
    INOUT = 2,   // Read-then-write: consumer of prior producer + modifier for downstream
};

/**
 * Aggregated parameter container for pto_submit_task
 *
 * Tensor pointers and types are stored in separate parallel arrays for
 * efficient bulk copy: the runtime can memcpy the pointer array and type
 * array independently, avoiding per-element branching.
 * Tensors are dispatched first in kernel args, followed by scalars.
 *
 * Example:
 *   Tensor td_a = make_tensor_external(dev_a, shapes, 2);
 *   Tensor td_c = make_tensor(shapes, 2);
 *   PTOParam params;
 *   params.add_input(td_a);
 *   params.add_output(td_c);
 *   params.add_scalar(some_value);
 *   pto2_rt_submit_aic_task(rt, kernel_id, params);
 *   // td_c.buffer.addr is already updated via pointer write-back
 */
struct PTOParam {
    static constexpr int32_t MAX_TENSORS = 32;
    static constexpr int32_t MAX_SCALARS = 128;

    Tensor* tensors[MAX_TENSORS];
    PTOParamType tensor_types[MAX_TENSORS];
    uint64_t scalars[MAX_SCALARS];
    int32_t tensor_count{0};
    int32_t scalar_count{0};

    void reset() {
        tensor_count = 0;
        scalar_count = 0;
    }

    bool check_add_tensor_valid() const {
        assert(scalar_count == 0 && "scalar must add after all tensor added");
        return true;
    }

    void add_input(Tensor& t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        assert(t.buffer.addr != 0 && "INPUT param must have a non-NULL buffer address");
        assert(tensor_count < MAX_TENSORS && "Too many tensor params");
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = PTOParamType::INPUT;
        tensor_count++;
    }

    void add_output(Tensor& t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        assert(tensor_count < MAX_TENSORS && "Too many tensor params");
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = PTOParamType::OUTPUT;
        tensor_count++;
    }

    void add_inout(Tensor& t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        assert(t.buffer.addr != 0 && "INOUT param must have a non-NULL buffer address");
        assert(tensor_count < MAX_TENSORS && "Too many tensor params");
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = PTOParamType::INOUT;
        tensor_count++;
    }

    void add_scalar(uint64_t v) {
        assert(scalar_count < MAX_SCALARS && "Too many scalar params");
        scalars[scalar_count++] = v;
    }

    void add_scalars(const uint64_t* values, int count) {
        assert(scalar_count + count <= MAX_SCALARS && "Too many scalar params");
        memcpy(&scalars[scalar_count], values, count * sizeof(uint64_t));
        scalar_count += count;
    }

    /**
     * Widen int32 values to uint64 directly into the scalars array.
     * Avoids an intermediate uint64_t buffer when source data is int32.
     */
    void add_scalars_i32(const int32_t* values, int count) {
        assert(scalar_count + count <= MAX_SCALARS && "Too many scalar params");
        uint64_t* dst = &scalars[scalar_count];
        for (int i = 0; i < count; i++) {
            dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
        }
        scalar_count += count;
    }

    /**
     * Copy scalars from another PTOParam's scalar array.
     * Useful when multiple tasks share the same scalar data (e.g., block indices).
     */
    void copy_scalars_from(const PTOParam& src, int src_offset, int count) {
        assert(src_offset + count <= src.scalar_count && "Source scalar range out of bounds");
        assert(scalar_count + count <= MAX_SCALARS && "Too many scalar params");
        memcpy(&scalars[scalar_count], &src.scalars[src_offset], count * sizeof(uint64_t));
        scalar_count += count;
    }
};

#endif  // ORCH_BUILD_GRAPH_PTO_TYPES_H
