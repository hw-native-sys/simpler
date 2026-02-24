/**
 * Orchestration Build Graph Types - Data structures for orchestration runtime extensions
 *
 * Standalone header defining orchestration-specific types for:
 * - PTOParam: Parameter descriptor for pto_submit_task API
 * - PTOWorkerType: Worker types for heterogeneous scheduling
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
    SCALAR = 3   // Raw scalar value (no buffer, no dependency tracking)
};

/**
 * Parameter Descriptor for pto_submit_task
 *
 * Each parameter holds a pointer to the caller's Tensor for
 * automatic dependency detection via TensorMap overlap checking.
 *
 * For OUTPUT params with tensor->buffer.addr == 0, the runtime allocates
 * a buffer and writes the address back through the pointer, implicitly
 * updating the caller's local Tensor. No manual sync needed.
 *
 * Example:
 *   Tensor td_a = make_tensor_external(dev_a, size);
 *   Tensor td_c = make_tensor(size);
 *   PTOParam params[] = {
 *       make_input_param(td_a),
 *       make_output_param(td_c),
 *   };
 *   pto2_rt_submit_task(rt, func_id, worker_type, params, 2);
 *   // td_c.buffer.addr is already updated - no explicit sync needed
 */
struct PTOParam {
    PTOParamType type;         // PTOParamType::INPUT, PTOParamType::OUTPUT, or PTOParamType::SCALAR
    Tensor* tensor;  // Pointer to caller's tensor descriptor (NULL for SCALAR)
    uint64_t scalar_value;     // Raw value for PTOParamType::SCALAR (e.g., encoded float, int size)
};

// =============================================================================
// Factory Helpers
// =============================================================================

static inline PTOParam make_scalar_param(uint64_t value) {
    PTOParam p = {};
    p.type = PTOParamType::SCALAR;
    p.tensor = nullptr;
    p.scalar_value = value;
    return p;
}

static inline PTOParam make_input_param(Tensor& tensor) {
    assert(tensor.buffer.addr != 0 && "INPUT param must have a non-NULL buffer address");
    PTOParam p = {};
    p.type = PTOParamType::INPUT;
    p.tensor = &tensor;
    p.scalar_value = 0;
    return p;
}

static inline PTOParam make_output_param(Tensor& tensor) {
    PTOParam p = {};
    p.type = PTOParamType::OUTPUT;
    p.tensor = &tensor;
    p.scalar_value = 0;
    return p;
}

static inline PTOParam make_inout_param(Tensor& tensor) {
    assert(tensor.buffer.addr != 0 && "INOUT param must have a non-NULL buffer address");
    PTOParam p = {};
    p.type = PTOParamType::INOUT;
    p.tensor = &tensor;
    p.scalar_value = 0;
    return p;
}

#endif  // ORCH_BUILD_GRAPH_PTO_TYPES_H
