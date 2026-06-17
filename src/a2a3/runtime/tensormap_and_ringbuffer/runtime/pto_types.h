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

#define MAX_TENSOR_ARGS CORE_MAX_TENSOR_ARGS
#define MAX_SCALAR_ARGS CORE_MAX_SCALAR_ARGS

typedef enum
{
    ASYNC_ENGINE_SDMA = 0,
    ASYNC_ENGINE_ROCE = 1,
    ASYNC_ENGINE_URMA = 2,
    ASYNC_ENGINE_CCU = 3,
    NUM_ASYNC_ENGINES = 4,
} AsyncEngine;

enum class CompletionType : int32_t
{
    COUNTER = 0,
};

enum class PTO2ScopeMode : uint8_t
{
    AUTO = 0,
    MANUAL = 1,
};

class TaskOutputTensors
{
public:
    TaskOutputTensors() :
        task_id_(PTO2TaskId::invalid()),
        output_count_(0)
    {}

    bool empty() const
    {
        return output_count_ == 0;
    }
    uint32_t size() const
    {
        return output_count_;
    }

    /// Borrow a materialized output tensor by index (lvalue only).
    const Tensor &get_ref(uint32_t index) const &
    {
        always_assert(index < output_count_);
        return *tensors_[index];
    }
    const Tensor &get_ref(uint32_t index) const && = delete;

    /// Runtime-internal: append one materialized output Tensor.
    void materialize_output(const Tensor &tensor)
    {
        always_assert(output_count_ < MAX_TENSOR_ARGS);
        tensors_[output_count_++] = &tensor;
    }

    void set_task_id(PTO2TaskId id)
    {
        task_id_ = id;
    }

    PTO2TaskId task_id() const
    {
        return task_id_;
    }

private:
    PTO2TaskId task_id_;
    uint32_t output_count_;
    // Upper bound: a task cannot have more outputs than total tensor args
    // (every OUTPUT/OUTPUT_EXISTING slot is one of the Arg's tensor slots).
    const Tensor *tensors_[MAX_TENSOR_ARGS];
};

using TaskSubmitResult = TaskOutputTensors;

// TensorArgType is defined in tensor_arg.h (included above)

union TensorRef
{
    const Tensor *ptr;
    const TensorCreateInfo *create_info;
    TensorRef() :
        ptr(nullptr)
    {}
};

struct Arg : TaskArgsTpl<TensorRef, uint64_t, MAX_TENSOR_ARGS, MAX_SCALAR_ARGS, TensorArgType>
{
    bool has_error{false};
    const char *error_msg{nullptr};
    PTO2LaunchSpec launch_spec;  // SPMD launch parameters (block_num, etc.)

    void reset()
    {
        clear();
        has_error = false;
        error_msg = nullptr;
        tensor_dump_arg_mask_ = 0;
        explicit_deps_ = nullptr;
        explicit_dep_count_ = 0;
    }

    void set_error(const char *msg)
    {
        if (!has_error)
        {
            has_error = true;
            error_msg = msg;
        }
    }

    template <typename... Args>
    void dump(Args &&...args)
    {
        static_assert((std::is_lvalue_reference_v<Args> && ...), "dump: temporaries are not allowed — pass tensors already added to this Arg");
        static_assert(((std::is_same_v<std::decay_t<Args>, Tensor> || std::is_same_v<std::decay_t<Args>, TensorCreateInfo>) && ...), "dump: all arguments must be Tensor or TensorCreateInfo");
        if constexpr (sizeof...(Args) == 0) mark_all_tensor_dump_arg();
        else (mark_tensor_dump_arg(args), ...);
    }

    uint64_t tensor_dump_arg_mask() const
    {
        return tensor_dump_arg_mask_;
    }

    template <typename... Args>
    void add_input(Args &&...args)
    {
        if (!check_add_tensor_valid<false>(args...)) return;
        ((tensors_[tensor_count_].ptr = &args, tags_[tensor_count_] = TensorArgType::INPUT, tensor_count_++), ...);
    }

    template <typename... Args>
    void add_output(Args &&...args)
    {
        if (!check_add_tensor_valid<true>(args...)) return;
        if constexpr ((std::is_same_v<std::decay_t<Args>, TensorCreateInfo> && ...)) ((tensors_[tensor_count_].create_info = &args, tags_[tensor_count_] = TensorArgType::OUTPUT, tensor_count_++), ...);
        else ((tensors_[tensor_count_].ptr = &args, tags_[tensor_count_] = TensorArgType::OUTPUT_EXISTING, tensor_count_++), ...);
    }

    template <typename... Args>
    void add_inout(Args &&...args)
    {
        if (!check_add_tensor_valid<false>(args...)) return;
        ((tensors_[tensor_count_].ptr = &args, tags_[tensor_count_] = TensorArgType::INOUT, tensor_count_++), ...);
    }

    /// No-dependency existing tensor: skips OverlapMap lookup, depends on creator only.
    template <typename... Args>
    void add_no_dep(Args &&...args)
    {
        if (!check_add_tensor_valid<false>(args...)) return;
        ((tensors_[tensor_count_].ptr = &args, tags_[tensor_count_] = TensorArgType::NO_DEP, tensor_count_++), ...);
    }

    void set_dependencies(const PTO2TaskId *deps, uint32_t count)
    {
        if (count == 0)
        {
            explicit_deps_ = nullptr;
            explicit_dep_count_ = 0;
            return;
        }
        if (deps == nullptr)
        {
            set_error("set_dependencies: deps must not be null when count > 0");
            return;
        }
        if (explicit_deps_ != nullptr)
        {
            set_error("set_dependencies: may be called at most once per Arg");
            return;
        }
        explicit_deps_ = deps;
        explicit_dep_count_ = count;
    }

    uint32_t explicit_dep_count() const
    {
        return explicit_dep_count_;
    }

    PTO2TaskId explicit_dep(uint32_t index) const
    {
        always_assert(index < explicit_dep_count_);
        return explicit_deps_[index];
    }

    const PTO2TaskId *explicit_deps_data() const
    {
        return explicit_deps_;
    }

    template <typename... Args>
    void add_scalar(Args... args)
    {
        static_assert(sizeof...(Args) >= 1, "add_scalar: at least one argument required");
        static_assert((is_supported_scalar_arg_v<Args> && ...), "add_scalar: all types must be arithmetic or enum");
        if (scalar_count_ + sizeof...(Args) > MAX_SCALAR_ARGS)
        {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        ((scalars_[scalar_count_++] = to_u64(args)), ...);
    }

    void add_scalars(const uint64_t *values, int count)
    {
        if (scalar_count_ + count > MAX_SCALAR_ARGS)
        {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        memcpy(&scalars_[scalar_count_], values, count * sizeof(uint64_t));
        scalar_count_ += count;
    }

    void add_scalars_i32(const int32_t *values, int count)
    {
        if (scalar_count_ + count > MAX_SCALAR_ARGS)
        {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        uint64_t *dst = &scalars_[scalar_count_];
#if defined(__aarch64__)
        int i = 0;
        for (; i + 4 <= count; i += 4)
        {
            uint32x4_t v = vld1q_u32(reinterpret_cast<const uint32_t *>(values + i));
            uint64x2_t lo = vmovl_u32(vget_low_u32(v));
            uint64x2_t hi = vmovl_u32(vget_high_u32(v));
            vst1q_u64(dst + i, lo);
            vst1q_u64(dst + i + 2, hi);
        }
        for (; i < count; i++) dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
#else
        for (int i = 0; i < count; i++) dst[i] = static_cast<uint64_t>(static_cast<uint32_t>(values[i]));
#endif
        scalar_count_ += count;
    }

    void copy_scalars_from(const Arg &src, int src_offset, int count)
    {
        if (src_offset + count > src.scalar_count_)
        {
            set_error("Source scalar range out of bounds in copy_scalars_from");
            return;
        }
        if (scalar_count_ + count > MAX_SCALAR_ARGS)
        {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=32)");
            return;
        }
        memcpy(&scalars_[scalar_count_], &src.scalars_[src_offset], count * sizeof(uint64_t));
        scalar_count_ += count;
    }

private:
    // Caller-owned dependency array; lifetime must extend through submit.
    static_assert(MAX_TENSOR_ARGS <= 64, "tensor dump arg mask assumes at most 64 tensor arguments");
    uint64_t tensor_dump_arg_mask_{0};
    const PTO2TaskId *explicit_deps_{nullptr};
    uint32_t explicit_dep_count_{0};

    // No-arg dump(): mark every tensor arg already added to this Arg.
    void mark_all_tensor_dump_arg()
    {
        if (tensor_count_ == 0)
        {
            set_error("dump: no tensor arguments added to this Arg");
            return;
        }
        for (int32_t i = 0; i < tensor_count_; i++) tensor_dump_arg_mask_ |= (uint64_t{1} << i);
    }

    void mark_tensor_dump_arg(const Tensor &tensor)
    {
        for (int32_t i = 0; i < tensor_count_; i++)
        {
            if (tags_[i] != TensorArgType::OUTPUT && tensors_[i].ptr == &tensor)
            {
                tensor_dump_arg_mask_ |= (uint64_t{1} << i);
                return;
            }
        }
        set_error("dump: tensor is not part of this Arg");
    }

    void mark_tensor_dump_arg(const TensorCreateInfo &create_info)
    {
        for (int32_t i = 0; i < tensor_count_; i++)
        {
            if (tags_[i] == TensorArgType::OUTPUT && tensors_[i].create_info == &create_info)
            {
                tensor_dump_arg_mask_ |= (uint64_t{1} << i);
                return;
            }
        }
        set_error("dump: TensorCreateInfo is not part of this Arg");
    }

    template <bool is_output, typename... Args>
    bool check_add_tensor_valid(Args &&...)
    {
        static_assert(sizeof...(Args) >= 1, "at least one argument required");
        static_assert((std::is_lvalue_reference_v<Args> && ...), "temporaries are not allowed — stored pointers would dangle after the call");
        if constexpr (is_output) static_assert((std::is_same_v<std::decay_t<Args>, Tensor> && ...) || (std::is_same_v<std::decay_t<Args>, TensorCreateInfo> && ...), "add_output: all arguments must be the same type (all Tensor or all TensorCreateInfo)");
        else static_assert((std::is_same_v<std::decay_t<Args>, Tensor> && ...), "all arguments must be Tensor");
        if (scalar_count_ != 0)
        {
            set_error("add_input/add_output/add_inout called after add_scalar: "
                      "all tensors must be added before any scalars");
            return false;
        }
        if (tensor_count_ + sizeof...(Args) > MAX_TENSOR_ARGS)
        {
            set_error("Too many tensor args (exceeds MAX_TENSOR_ARGS=16)");
            return false;
        }
        return true;
    }
};

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_TYPES_H_
