/**
 * Orchestration Build Graph Types - Data structures for orchestration runtime extensions
 *
 * Standalone header defining orchestration-specific types for:
 * - PTOParam: Parameter descriptor for pto_submit_task API
 * - PTOWorkerType: Worker types for heterogeneous scheduling
 *
 * Tensor descriptor types (TensorDescriptor, PTOBufferHandle, PTOOverlapStrategy) are
 * defined in tensor_descriptor.h.
 *
 * This header is independent of orch_build_graph_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorPair, HostApi).
 */

#ifndef ORCH_BUILD_GRAPH_PTO_TYPES_H
#define ORCH_BUILD_GRAPH_PTO_TYPES_H

#include <stdint.h>
#include <assert.h>

#include "tensor_descriptor.h"

// =============================================================================
// Configuration
// =============================================================================

#ifndef PTO_TENSORMAP_POOL_SIZE
#define PTO_TENSORMAP_POOL_SIZE 4096
#endif

#ifndef PTO_TENSORMAP_NUM_BUCKETS
#define PTO_TENSORMAP_NUM_BUCKETS 1024
#endif

#ifndef PTO_MAX_SCOPE_DEPTH
#define PTO_MAX_SCOPE_DEPTH 32
#endif

// =============================================================================
// Worker Types
// =============================================================================

/**
 * Worker types for heterogeneous scheduling
 *
 * Tasks are routed to different ready queues based on worker_type:
 * - PTOWorkerType::CUBE:   AICore-CUBE (matrix ops, convolution)
 * - PTOWorkerType::VECTOR: AICore-VECTOR (element-wise ops, activation)
 *
 * Note: AICPU is not a worker type - AICPU threads act as schedulers that
 * dispatch tasks to AICore workers.
 */
enum class PTOWorkerType : int32_t {
    CUBE = 0,    // AICore-CUBE
    VECTOR = 1,  // AICore-VECTOR
};

// Number of worker types (used for array sizing)
constexpr int32_t PTO_NUM_WORKER_TYPES = 2;

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
 * Each parameter carries a full tensor descriptor for automatic
 * dependency detection via TensorMap overlap checking.
 *
 * Example:
 *   PTOParam params[] = {
 *       {PTOParamType::INPUT,  make_tensor_bbox(dev_a->addr, size), dev_a},
 *       {PTOParamType::OUTPUT, make_tensor_bbox(dev_c->addr, size), dev_c},
 *   };
 *   runtime->pto_submit_task(func_id, worker_type, params, 2);
 */
struct PTOParam {
    PTOParamType type;        // PTOParamType::INPUT, PTOParamType::OUTPUT, or PTOParamType::SCALAR
    TensorDescriptor tensor;  // Full strided descriptor for overlap checking (unused for SCALAR)
    PTOBufferHandle* buffer;  // Associated buffer handle (nullptr for SCALAR)
    uint64_t scalar_value;    // Raw value for PTOParamType::SCALAR (e.g., encoded float, int size)
};

// =============================================================================
// Factory Helpers
// =============================================================================

static inline PTOParam make_scalar_param(uint64_t value) {
    PTOParam p = {};
    p.type = PTOParamType::SCALAR;
    p.buffer = nullptr;
    p.scalar_value = value;
    return p;
}

static inline PTOParam make_input_param(PTOBufferHandle& buf, int32_t size, int32_t version = 0) {
    assert(buf.addr != 0 && "INPUT param must have a non-NULL buffer address");
    PTOParam p = {};
    p.type = PTOParamType::INPUT;
    p.tensor = make_tensor_bbox(buf.addr, size, version);
    p.buffer = &buf;
    p.scalar_value = 0;
    return p;
}

static inline PTOParam make_output_param(PTOBufferHandle& buf, int32_t size, int32_t version = 0) {
    PTOParam p = {};
    p.type = PTOParamType::OUTPUT;
    p.tensor = make_tensor_bbox(buf.addr, size, version);
    p.buffer = &buf;
    p.scalar_value = 0;
    return p;
}

static inline PTOParam make_inout_param(PTOBufferHandle& buf, int32_t size, int32_t version = 0) {
    assert(buf.addr != 0 && "INOUT param must have a non-NULL buffer address");
    PTOParam p = {};
    p.type = PTOParamType::INOUT;
    p.tensor = make_tensor_bbox(buf.addr, size, version);
    p.buffer = &buf;
    p.scalar_value = 0;
    return p;
}

#endif  // ORCH_BUILD_GRAPH_PTO_TYPES_H
