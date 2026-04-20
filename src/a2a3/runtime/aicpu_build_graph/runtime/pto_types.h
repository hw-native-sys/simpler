/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Orchestration Build Graph Types - Data structures for orchestration runtime extensions
 *
 * Standalone header defining orchestration-specific types for:
 * - TaskOutputTensors: Return value from submit containing materialized output Tensors
 * - TensorRef: Tagged union for tensor slots (Tensor* or TensorCreateInfo)
 * - SubmitResult: Combined return value (PTO2TaskId + TaskOutputTensors)
 * - Arg: Aggregated argument container for pto_submit_task API
 *
 * Tensor descriptor types (Tensor, PTOBufferHandle, TensorCreateInfo) are
 * defined in tensor.h.
 *
 * This header is independent of orch_build_graph_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorPair, HostApi).
 */

#ifndef SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_TYPES_H_
#define SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_TYPES_H_

#include <stdint.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "task_args.h"
#include "tensor.h"
#include "tensor_arg.h"

// Task arguments
#define MAX_TENSOR_ARGS 16   // Maximum tensor parameters per task
#define MAX_SCALAR_ARGS 32   // Maximum scalar parameters per task
#define PTO2_MAX_OUTPUTS 16  // Maximum outputs per task
#define PTO2_MAX_INPUTS 16   // Maximum inputs per task
#define PTO2_MAX_INOUTS 8    // Maximum in-out args per task

// Forward declaration for SubmitResult
struct PTO2TaskId;

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
    TaskOutputTensors() :
        output_count_(0) {}

    bool empty() const { return output_count_ == 0; }
    uint32_t size() const { return output_count_; }

    /// Borrow a materialized output tensor by index (lvalue only).
    const Tensor &get_ref(uint32_t index) const & {
        always_assert(index < output_count_);
        return *reinterpret_cast<const Tensor *>(_storage + index * sizeof(Tensor));
    }
    const Tensor &get_ref(uint32_t index) const && = delete;

    /// Runtime-internal: append one materialized output Tensor.
    Tensor &materialize_output(const TensorCreateInfo &ci, void *addr, int32_t version) {
        always_assert(output_count_ < PTO2_MAX_OUTPUTS);
        Tensor *out = output_ptr(output_count_);
        out->init_from_create_info(ci, addr, version);
        output_count_++;
        return *out;
    }

    /// Runtime-internal: writable pointer for materialization.
    Tensor *output_ptr(uint32_t index) { return reinterpret_cast<Tensor *>(_storage + index * sizeof(Tensor)); }
    const Tensor *output_ptr(uint32_t index) const {
        return reinterpret_cast<const Tensor *>(_storage + index * sizeof(Tensor));
    }

private:
    uint32_t output_count_;
    alignas(Tensor) unsigned char _storage[PTO2_MAX_OUTPUTS * sizeof(Tensor)];
};

// =============================================================================
// Argument Types (for pto_submit_task API)
// =============================================================================

// TensorArgType is defined in tensor_arg.h (included above)

/**
 * Tagged union for a single Arg slot — either a Tensor* or a TensorCreateInfo value.
 * The active member is determined by TensorArgType (OUTPUT → create_info, else → ptr).
 */
union TensorRef {
    const Tensor *ptr;
    TensorCreateInfo create_info;
    TensorRef() :
        ptr(nullptr) {}
};

/**
 * Aggregated argument container for pto_submit_task
 *
 * Inherits storage from TaskArgsTpl<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType>.
 * Each tensor slot stores a TensorRef union (Tensor* or TensorCreateInfo)
 * discriminated by the corresponding tag().
 * Tensors are dispatched first in kernel args, followed by scalars.
 *
 * Output arguments follow two distinct ownership models:
 * - add_output(const TensorCreateInfo&): OUTPUT — runtime allocates buffer
 *   and materializes a new Tensor, returned via TaskOutputTensors.
 * - add_inout(const Tensor&): INOUT — reuses an existing Tensor as the write target.
 *
 * Example:
 *   Tensor x = make_tensor_external(dev_a, shapes, 2);
 *   Arg args;
 *   args.add_input(x);
 *   args.add_output(TensorCreateInfo(shapes, 2));
 *   args.add_scalar(some_value);
 *   SubmitResult r = pto2_rt_submit_aic_task(rt, kernel_id, args);
 *   const Tensor& y = r.outputs.get_ref(0);
 */
struct Arg : TaskArgsTpl<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType> {
    bool has_error{false};
    const char *error_msg{nullptr};

    void reset() {
        clear();
        has_error = false;
        error_msg = nullptr;
    }

    void set_error(const char *msg) {
        if (!has_error) {
            has_error = true;
            error_msg = msg;
        }
    }

    bool check_add_tensor_valid() {
        if (scalar_count_ != 0) {
            set_error(
                "add_input/add_output/add_inout called after add_scalar: "
                "all tensors must be added before any scalars"
            );
            return false;
        }
        if (tensor_count_ >= MAX_TENSOR_ARGS) {
            set_error("Too many tensor args (exceeds MAX_TENSOR_ARGS=16)");
            return false;
        }
        return true;
    }

    void add_input(const Tensor &t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors_[tensor_count_].ptr = &t;
        tags_[tensor_count_] = TensorArgType::INPUT;
        tensor_count_++;
    }

    /// Standard future-output path: runtime allocates buffer from heap,
    /// materializes Tensor into TaskOutputTensors.
    void add_output(const TensorCreateInfo &ci) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors_[tensor_count_].create_info = ci;
        tags_[tensor_count_] = TensorArgType::OUTPUT;
        tensor_count_++;
    }

    void add_inout(const Tensor &t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors_[tensor_count_].ptr = &t;
        tags_[tensor_count_] = TensorArgType::INOUT;
        tensor_count_++;
    }

    /**
     * Add a scalar value. Type is deduced from the argument;
     * the value is bit-cast to uint64_t for storage.
     *
     *   args.add_scalar(uint64_val);      // existing usage unchanged
     *   args.add_scalar(3.14f);           // float, auto bit-cast
     *   args.add_scalar(int32_t(42));     // int32, auto bit-cast
     */
    template <typename T = uint64_t>
    void add_scalar(T value) {
        static_assert(is_supported_scalar_arg_v<T>, "add_scalar: type must be arithmetic or enum");
        if (scalar_count_ >= MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        scalars_[scalar_count_++] = to_u64(value);
    }

    void add_scalars(const uint64_t *values, int count) {
        if (scalar_count_ + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        memcpy(&scalars_[scalar_count_], values, count * sizeof(uint64_t));
        scalar_count_ += count;
    }

    /**
     * Zero-extend int32 bit patterns into uint64 scalar slots.
     * Negative values are treated as their unsigned 32-bit representation
     * (e.g., -1 → 0x00000000FFFFFFFF, not 0xFFFFFFFFFFFFFFFF).
     * Uses NEON to process 4 elements per iteration on aarch64.
     */
    void add_scalars_i32(const int32_t *values, int count) {
        if (scalar_count_ + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        uint64_t *dst = &scalars_[scalar_count_];
#if defined(__aarch64__)
        int i = 0;
        for (; i + 4 <= count; i += 4) {
            uint32x4_t v = vld1q_u32(reinterpret_cast<const uint32_t *>(values + i));
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
        scalar_count_ += count;
    }

    /**
     * Copy scalars from another Arg's scalar array.
     * Useful when multiple tasks share the same scalar data (e.g., block indices).
     */
    void copy_scalars_from(const Arg &src, int src_offset, int count) {
        if (src_offset + count > src.scalar_count_) {
            set_error("Source scalar range out of bounds in copy_scalars_from");
            return;
        }
        if (scalar_count_ + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        memcpy(&scalars_[scalar_count_], &src.scalars_[src_offset], count * sizeof(uint64_t));
        scalar_count_ += count;
    }
};

#endif  // SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_TYPES_H_
