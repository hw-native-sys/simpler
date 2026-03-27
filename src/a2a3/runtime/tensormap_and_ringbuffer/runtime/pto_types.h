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
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "tensor.h"
#include "pto_cq_types.h"

// Task parameters
#define PTO2_MAX_TENSOR_PARAMS    16      // Maximum tensor parameters per task
#define PTO2_MAX_SCALAR_PARAMS    128     // Maximum scalar parameters per task
#define PTO2_MAX_OUTPUTS          16      // Maximum outputs per task
#define PTO2_MAX_INPUTS           16      // Maximum inputs per task
#define PTO2_MAX_INOUTS           8       // Maximum in-out params per task
// Max completion conditions per deferred task (matches CQ capacity)
#define PTO2_MAX_COMPLETIONS_PER_TASK PTO2_CQ_MAX_ENTRIES

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

typedef enum {
    PTO2_ASYNC_ENGINE_SDMA = 0,   // System DMA
    PTO2_ASYNC_ENGINE_ROCE = 1,   // RoCE RDMA (reserved)
    PTO2_ASYNC_ENGINE_URMA  = 2,   // URMA cross-die memory ops (reserved)
    PTO2_ASYNC_ENGINE_CCU  = 3,   // Cache coherence unit (reserved)
    PTO2_NUM_ASYNC_ENGINES = 4
} PTO2AsyncEngine;

enum class PTO2CompletionType : int32_t {
    COUNTER = 0,
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
    Tensor* tensors[PTO2_MAX_TENSOR_PARAMS];
    PTOParamType tensor_types[PTO2_MAX_TENSOR_PARAMS];
    uint64_t scalars[PTO2_MAX_SCALAR_PARAMS];
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    bool complete_in_future{false};
    uint64_t cq_addr{0};
    // Pre-launch gating metadata.
    // Current primary use case is notify-driven launch: the task stays out of
    // READY until the local notify counter reaches the expected value.
    bool has_launch_counter{false};
    uint64_t launch_counter_addr{0};      // Usually a local notify-counter address.
    uint32_t launch_counter_expected{0};  // Notify/counter threshold required for launch.
    bool has_error{false};
    const char* error_msg{nullptr};

    void reset() {
        tensor_count = 0;
        scalar_count = 0;
        complete_in_future = false;
        cq_addr = 0;
        has_launch_counter = false;
        launch_counter_addr = 0;
        launch_counter_expected = 0;
        has_error = false;
        error_msg = nullptr;
    }

    void set_error(const char* msg) {
        if (!has_error) {
            has_error = true;
            error_msg = msg;
        }
    }

    bool check_add_tensor_valid() {
        if (scalar_count != 0) {
            set_error("add_input/add_output/add_inout called after add_scalar: "
                      "all tensors must be added before any scalars");
            return false;
        }
        if (tensor_count >= PTO2_MAX_TENSOR_PARAMS) {
            set_error("Too many tensor params (exceeds PTO2_MAX_TENSOR_PARAMS=16)");
            return false;
        }
        return true;
    }

    void add_input(Tensor& t) {
        if (!check_add_tensor_valid()) { return; }
        if (t.buffer.addr == 0) {
            set_error("INPUT tensor must have a non-NULL buffer address");
            return;
        }
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = PTOParamType::INPUT;
        tensor_count++;
    }

    void add_output(Tensor& t) {
        if (!check_add_tensor_valid()) { return; }
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = PTOParamType::OUTPUT;
        tensor_count++;
    }

    void add_inout(Tensor& t) {
        if (!check_add_tensor_valid()) { return; }
        if (t.buffer.addr == 0) {
            set_error("INOUT tensor must have a non-NULL buffer address");
            return;
        }
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = PTOParamType::INOUT;
        tensor_count++;
    }

    void add_scalar(uint64_t v) {
        if (scalar_count >= PTO2_MAX_SCALAR_PARAMS) {
            set_error("Too many scalar params (exceeds PTO2_MAX_SCALAR_PARAMS)");
            return;
        }
        scalars[scalar_count++] = v;
    }

    void add_scalars(const uint64_t* values, int count) {
        if (scalar_count + count > PTO2_MAX_SCALAR_PARAMS) {
            set_error("Too many scalar params (exceeds PTO2_MAX_SCALAR_PARAMS)");
            return;
        }
        memcpy(&scalars[scalar_count], values, count * sizeof(uint64_t));
        scalar_count += count;
    }

    /**
     * Zero-extend int32 bit patterns into uint64 scalar slots.
     * Negative values are treated as their unsigned 32-bit representation
     * (e.g., -1 → 0x00000000FFFFFFFF, not 0xFFFFFFFFFFFFFFFF).
     * Uses NEON to process 4 elements per iteration on aarch64.
     */
    void add_scalars_i32(const int32_t* values, int count) {
        if (scalar_count + count > PTO2_MAX_SCALAR_PARAMS) {
            set_error("Too many scalar params (exceeds PTO2_MAX_SCALAR_PARAMS)");
            return;
        }
        uint64_t* dst = &scalars[scalar_count];
#if defined(__aarch64__)
        int i = 0;
        for (; i + 4 <= count; i += 4) {
            uint32x4_t v = vld1q_u32(reinterpret_cast<const uint32_t*>(values + i));
            uint64x2_t lo = vmovl_u32(vget_low_u32(v));
            uint64x2_t hi = vmovl_u32(vget_high_u32(v));
            vst1q_u64(dst + i, lo);
            vst1q_u64(dst + i + 2, hi);
        }
        for (; i < count; i++) {
            dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
        }
#else
        for (int i = 0; i < count; i++) {
            dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
        }
#endif
        scalar_count += count;
    }

    /**
     * Copy scalars from another PTOParam's scalar array.
     * Useful when multiple tasks share the same scalar data (e.g., block indices).
     */
    void copy_scalars_from(const PTOParam& src, int src_offset, int count) {
        if (src_offset + count > src.scalar_count) {
            set_error("Source scalar range out of bounds in copy_scalars_from");
            return;
        }
        if (scalar_count + count > PTO2_MAX_SCALAR_PARAMS) {
            set_error("Too many scalar params (exceeds PTO2_MAX_SCALAR_PARAMS)");
            return;
        }
        memcpy(&scalars[scalar_count], &src.scalars[src_offset], count * sizeof(uint64_t));
        scalar_count += count;
    }
};

#endif  // ORCH_BUILD_GRAPH_PTO_TYPES_H
