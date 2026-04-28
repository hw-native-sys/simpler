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
 * - Arg: Aggregated argument container for pto_submit_task API
 *
 * Tensor descriptor types (Tensor, PTOBufferHandle, TensorCreateInfo) are
 * defined in tensor.h.
 *
 * This header is independent of orch_build_graph_runtime.h to allow inclusion from runtime.h
 * without type conflicts (Handshake, TensorPair, HostApi).
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_

#include <stdint.h>
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "pto_submit_types.h"
#include "task_args.h"
#include "tensor.h"
#include "tensor_arg.h"

// Task arguments
#define MAX_TENSOR_ARGS 16         // Maximum tensor arguments per task
#define MAX_SCALAR_ARGS 32         // Maximum scalar arguments per task
#define PTO2_MAX_OUTPUTS 16        // Maximum outputs per task
#define PTO2_MAX_INPUTS 16         // Maximum inputs per task
#define PTO2_MAX_INOUTS 8          // Maximum in-out args per task
#define PTO2_MAX_EXPLICIT_DEPS 16  // Maximum explicit task dependencies per task

typedef enum {
    PTO2_ASYNC_ENGINE_SDMA = 0,
    PTO2_ASYNC_ENGINE_ROCE = 1,
    PTO2_ASYNC_ENGINE_URMA = 2,
    PTO2_ASYNC_ENGINE_CCU = 3,
    PTO2_NUM_ASYNC_ENGINES = 4,
} PTO2AsyncEngine;

enum class PTO2CompletionType : int32_t {
    COUNTER = 0,
};

// =============================================================================
// Task Output Tensors (return value from submit)
// =============================================================================

enum class PTO2ScopeMode : uint8_t {
    AUTO = 0,
    MANUAL = 1,
};

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
        task_id_(PTO2TaskId::invalid()),
        output_count_(0) {}

    bool empty() const { return output_count_ == 0; }
    uint32_t size() const { return output_count_; }

    /// Borrow a materialized output tensor by index (lvalue only).
    const Tensor &get_ref(uint32_t index) const & {
        always_assert(index < output_count_);
        return *tensors_[index];
    }
    const Tensor &get_ref(uint32_t index) const && = delete;

    /// Runtime-internal: append one materialized output Tensor.
    void materialize_output(const Tensor &tensor) {
        always_assert(output_count_ < PTO2_MAX_OUTPUTS);
        tensors_[output_count_++] = &tensor;
    }

    void set_task_id(PTO2TaskId id) { task_id_ = id; }

    PTO2TaskId task_id() const { return task_id_; }

private:
    PTO2TaskId task_id_;
    uint32_t output_count_;
    const Tensor *tensors_[PTO2_MAX_OUTPUTS];
};

using TaskSubmitResult = TaskOutputTensors;

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
    const TensorCreateInfo *create_info;
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
 *   TensorCreateInfo ci(shapes, 2);  // must outlive submit
 *   Arg args;
 *   args.add_input(x);
 *   args.add_output(ci);
 *   args.add_scalar(some_value);
 *   TaskOutputTensors outs = rt_submit_aic_task(kernel_id, args);
 *   const Tensor& y = outs.get_ref(0);
 */
struct Arg : TaskArgsTpl<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType> {
    bool has_error{false};
    const char *error_msg{nullptr};
    PTO2LaunchSpec launch_spec;  // SPMD launch parameters (block_num, etc.)

    void reset() {
        clear();
        has_error = false;
        error_msg = nullptr;
        explicit_deps_.reset();
    }

    void set_error(const char *msg) {
        if (!has_error) {
            has_error = true;
            error_msg = msg;
        }
    }

    template <typename... Args>
    void add_input(Args &&...args) {
        if (!check_add_tensor_valid<false>(args...)) {
            return;
        }
        ((tensors_[tensor_count_].ptr = &args, tags_[tensor_count_] = TensorArgType::INPUT, tensor_count_++), ...);
    }

    /// Batch add outputs — all Tensor or all TensorCreateInfo:
    ///   add_output(ci1, ci2)         — runtime allocates buffers (OUTPUT)
    ///   add_output(t1, t2)           — write-only existing tensors (OUTPUT_EXISTING)
    template <typename... Args>
    void add_output(Args &&...args) {
        if (!check_add_tensor_valid<true>(args...)) return;
        if constexpr ((std::is_same_v<std::decay_t<Args>, TensorCreateInfo> && ...)) {
            ((tensors_[tensor_count_].create_info = &args, tags_[tensor_count_] = TensorArgType::OUTPUT,
              tensor_count_++),
             ...);
        } else {
            ((tensors_[tensor_count_].ptr = &args, tags_[tensor_count_] = TensorArgType::OUTPUT_EXISTING,
              tensor_count_++),
             ...);
        }
    }

    template <typename... Args>
    void add_inout(Args &&...args) {
        if (!check_add_tensor_valid<false>(args...)) {
            return;
        }
        ((tensors_[tensor_count_].ptr = &args, tags_[tensor_count_] = TensorArgType::INOUT, tensor_count_++), ...);
    }

    /// No-dependency existing tensor: skips OverlapMap lookup, depends on creator only.
    template <typename... Args>
    void add_no_dep(Args &&...args) {
        if (!check_add_tensor_valid<false>(args...)) return;
        ((tensors_[tensor_count_].ptr = &args, tags_[tensor_count_] = TensorArgType::NO_DEP, tensor_count_++), ...);
    }

    template <typename... TaskIds>
    void add_dep(TaskIds... task_ids) {
        static_assert(sizeof...(TaskIds) >= 1, "add_dep: at least one task id is required");
        static_assert(
            (std::is_same_v<std::decay_t<TaskIds>, PTO2TaskId> && ...), "add_dep: all arguments must be PTO2TaskId"
        );
        if (explicit_deps_.size() + sizeof...(TaskIds) > PTO2_MAX_EXPLICIT_DEPS) {
            set_error("Too many explicit deps (exceeds explicit dependency capacity=16)");
            return;
        }
        (explicit_deps_.add(task_ids), ...);
    }

    uint32_t explicit_dep_count() const { return explicit_deps_.size(); }

    PTO2TaskId explicit_dep(uint32_t index) const { return explicit_deps_.get(index); }

    /**
     * Add scalar values. Types are deduced per argument; each value is
     * bit-cast to uint64_t for storage. Mixed types are allowed:
     *
     *   args.add_scalar(uint64_val);                  // single
     *   args.add_scalar(3.14f, int32_t(42), 7u);     // mixed batch
     */
    template <typename... Args>
    void add_scalar(Args... args) {
        static_assert(sizeof...(Args) >= 1, "add_scalar: at least one argument required");
        static_assert((is_supported_scalar_arg_v<Args> && ...), "add_scalar: all types must be arithmetic or enum");
        if (scalar_count_ + sizeof...(Args) > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        ((scalars_[scalar_count_++] = to_u64(args)), ...);
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

private:
    struct ExplicitDepStorage {
        PTO2TaskId task_ids[PTO2_MAX_EXPLICIT_DEPS]{};
        uint32_t count{0};

        void reset() { count = 0; }

        bool add(PTO2TaskId task_id) {
            if (count >= PTO2_MAX_EXPLICIT_DEPS) {
                return false;
            }
            task_ids[count++] = task_id;
            return true;
        }

        uint32_t size() const { return count; }

        PTO2TaskId get(uint32_t index) const {
            always_assert(index < count);
            return task_ids[index];
        }
    };

    ExplicitDepStorage explicit_deps_;

    template <bool is_output, typename... Args>
    bool check_add_tensor_valid(Args &&...) {
        static_assert(sizeof...(Args) >= 1, "at least one argument required");
        static_assert(
            (std::is_lvalue_reference_v<Args> && ...),
            "temporaries are not allowed — stored pointers would dangle after the call"
        );
        if constexpr (is_output) {
            static_assert(
                (std::is_same_v<std::decay_t<Args>, Tensor> && ...) ||
                    (std::is_same_v<std::decay_t<Args>, TensorCreateInfo> && ...),
                "add_output: all arguments must be the same type (all Tensor or all TensorCreateInfo)"
            );
        } else {
            static_assert((std::is_same_v<std::decay_t<Args>, Tensor> && ...), "all arguments must be Tensor");
        }
        if (scalar_count_ != 0) {
            set_error(
                "add_input/add_output/add_inout called after add_scalar: "
                "all tensors must be added before any scalars"
            );
            return false;
        }
        if (tensor_count_ + sizeof...(Args) > MAX_TENSOR_ARGS) {
            set_error("Too many tensor args (exceeds MAX_TENSOR_ARGS=16)");
            return false;
        }
        return true;
    }
};

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_
