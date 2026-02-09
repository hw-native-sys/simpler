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
    OUTPUT = 1,  // Write-only output buffer (allocated implicitly by runtime)
    SCALAR = 2,  // Raw scalar value (no buffer, no dependency tracking)
    INOUT = 3    // In-place update (creates dependency but NOT a new producer)
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

#endif  // ORCH_BUILD_GRAPH_PTO_TYPES_H
