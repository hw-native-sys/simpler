/**
 * Orchestration Build Graph Types - Data structures for orchestration runtime extensions
 *
 * Standalone header defining orchestration-specific types for:
 * - TaskOutputTensors: Return value from submit containing materialized output Tensors
 * - PTOParam: Aggregated parameter container for pto_submit_task API
 *
 * Tensor descriptor types (Tensor, PTOBufferHandle, TensorCreateInfo) are
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

// Task parameters
#define PTO2_MAX_TENSOR_PARAMS    16      // Maximum tensor parameters per task
#define PTO2_MAX_SCALAR_PARAMS    128     // Maximum scalar parameters per task
#define PTO2_MAX_OUTPUTS          16      // Maximum outputs per task
#define PTO2_MAX_INPUTS           16      // Maximum inputs per task
#define PTO2_MAX_INOUTS           8       // Maximum in-out params per task

// =============================================================================
// Task Output Tensors (return value from submit)
// =============================================================================

/**
 * TaskOutputTensors — returned by submit, holds materialized output Tensors.
 *
 * Only runtime-created outputs are stored here, indexed in add_output order.
 *
 * The underlying storage is uninitialized; only output_count elements are
 * valid after submit returns.  This avoids default-constructing Tensor[]
 * on the hot path (2 KB of unnecessary zeroing per submit).
 *
 * Users must hold a named TaskOutputTensors variable and borrow via get_ref();
 * binding get_ref() on an rvalue is compile-time rejected to prevent dangling.
 */
class TaskOutputTensors {
public:
    TaskOutputTensors() : output_count(0) {}

    uint32_t output_count;

    bool empty() const { return output_count == 0; }
    uint32_t size() const { return output_count; }

    /// Borrow a materialized output tensor by index (lvalue only).
    const Tensor& get_ref(uint32_t index) const & {
        always_assert(index < output_count);
        return *reinterpret_cast<const Tensor*>(_storage + index * sizeof(Tensor));
    }
    const Tensor& get_ref(uint32_t index) const && = delete;

    /// Runtime-internal: writable pointer for materialization.
    Tensor* output_ptr(uint32_t index) {
        return reinterpret_cast<Tensor*>(_storage + index * sizeof(Tensor));
    }
    const Tensor* output_ptr(uint32_t index) const {
        return reinterpret_cast<const Tensor*>(_storage + index * sizeof(Tensor));
    }

private:
    alignas(Tensor) unsigned char _storage[PTO2_MAX_OUTPUTS * sizeof(Tensor)];
};

// =============================================================================
// Parameter Types (for pto_submit_task API)
// =============================================================================

/**
 * Parameter Type - Distinguishes inputs, outputs, and in-place updates
 */
enum class PTOParamType : int32_t {
    INPUT = 0,   // Read-only input buffer
    OUTPUT = 1,  // Write-only output from TensorCreateInfo (runtime allocates)
    INOUT = 2,   // Read-then-write: consumer of prior producer + modifier for downstream
};

/**
 * Tagged union for a single PTOParam slot — either a Tensor* or a TensorCreateInfo value.
 * The active member is determined by PTOParamType (OUTPUT → create_info, else → tensor).
 */
union PTOParamRef {
    const Tensor* tensor;
    TensorCreateInfo create_info;
    PTOParamRef() : tensor(nullptr) {}
};

/**
 * Aggregated parameter container for pto_submit_task
 *
 * Each param slot stores a PTOParamRef union (Tensor* or TensorCreateInfo*)
 * discriminated by the corresponding PTOParamType entry.
 * Tensors are dispatched first in kernel args, followed by scalars.
 *
 * Output parameters follow two distinct ownership models:
 * - add_output(const TensorCreateInfo&): OUTPUT — runtime allocates buffer
 *   and materializes a new Tensor, returned via TaskOutputTensors.
 * - add_inout(const Tensor&): INOUT — reuses an existing Tensor as the write target.
 *
 * Example:
 *   Tensor x = make_tensor_external(dev_a, shapes, 2);
 *   PTOParam params;
 *   params.add_input(x);
 *   params.add_output(TensorCreateInfo(shapes, 2));
 *   params.add_scalar(some_value);
 *   TaskOutputTensors outs = pto2_rt_submit_aic_task(kernel_id, params);
 *   const Tensor& y = outs.get_ref(0);
 */
struct PTOParam {
    PTOParamRef refs[PTO2_MAX_TENSOR_PARAMS];
    PTOParamType tensor_types[PTO2_MAX_TENSOR_PARAMS];
    uint64_t scalars[PTO2_MAX_SCALAR_PARAMS];
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    bool has_error{false};
    const char* error_msg{nullptr};

    void reset() {
        tensor_count = 0;
        scalar_count = 0;
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

    void add_input(const Tensor& t) {
        if (!check_add_tensor_valid()) { return; }
        refs[tensor_count].tensor = &t;
        tensor_types[tensor_count] = PTOParamType::INPUT;
        tensor_count++;
    }

    /// Standard future-output path: runtime allocates buffer from heap,
    /// materializes Tensor into TaskOutputTensors.
    void add_output(const TensorCreateInfo& ci) {
        if (!check_add_tensor_valid()) { return; }
        refs[tensor_count].create_info = ci;
        tensor_types[tensor_count] = PTOParamType::OUTPUT;
        tensor_count++;
    }

    /// Runtime-allocated output with an initial element value replicated
    /// across the full buffer after HeapRing allocation.
    void add_output(const TensorCreateInfo& ci, uint64_t initial_value) {
        if (!check_add_tensor_valid()) { return; }
        refs[tensor_count].create_info = ci;
        refs[tensor_count].create_info.set_initial_value(initial_value);
        tensor_types[tensor_count] = PTOParamType::OUTPUT;
        tensor_count++;
    }

    void add_inout(const Tensor& t) {
        if (!check_add_tensor_valid()) { return; }
        refs[tensor_count].tensor = &t;
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
