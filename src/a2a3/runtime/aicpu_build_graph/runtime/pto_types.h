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
 * - Arg: Aggregated argument container for pto_submit_task API
 *
 * Tensor descriptor types (Tensor, PTOBufferHandle, PTOOverlapStrategy) are
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

#include "tensor.h"      // NOLINT(build/include_subdir)
#include "tensor_arg.h"  // NOLINT(build/include_subdir) -- canonical TensorArgType definition

// Task arguments
#define MAX_TENSOR_ARGS 16   // Maximum tensor parameters per task
#define MAX_SCALAR_ARGS 128  // Maximum scalar parameters per task
#define PTO2_MAX_OUTPUTS 16  // Maximum outputs per task
#define PTO2_MAX_INPUTS 16   // Maximum inputs per task
#define PTO2_MAX_INOUTS 8    // Maximum in-out args per task

// =============================================================================
// Argument Types (for pto_submit_task API)
// =============================================================================

// TensorArgType is defined in tensor_arg.h (included above)

/**
 * Aggregated argument container for pto_submit_task
 *
 * Tensor pointers and types are stored in separate parallel arrays for
 * efficient bulk copy: the runtime can memcpy the pointer array and type
 * array independently, avoiding per-element branching.
 * Tensors are dispatched first in kernel args, followed by scalars.
 *
 * Example:
 *   Tensor td_a = make_tensor_external(dev_a, shapes, 2);
 *   Tensor td_c = make_tensor(shapes, 2);
 *   Arg args;
 *   args.add_input(td_a);
 *   args.add_output(td_c);
 *   args.add_scalar(some_value);
 *   pto2_rt_submit_aic_task(rt, kernel_id, args);
 *   // td_c.buffer.addr is already updated via pointer write-back
 */
struct Arg {
    Tensor* tensors[MAX_TENSOR_ARGS];
    TensorArgType tensor_types[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
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
            set_error(
                "add_input/add_output/add_inout called after add_scalar: "
                "all tensors must be added before any scalars");
            return false;
        }
        if (tensor_count >= MAX_TENSOR_ARGS) {
            set_error("Too many tensor args (exceeds MAX_TENSOR_ARGS=16)");
            return false;
        }
        return true;
    }

    void add_input(Tensor& t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        if (t.buffer.addr == 0) {
            set_error("INPUT tensor must have a non-NULL buffer address");
            return;
        }
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = TensorArgType::INPUT;
        tensor_count++;
    }

    void add_output(Tensor& t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = TensorArgType::OUTPUT;
        tensor_count++;
    }

    void add_inout(Tensor& t) {
        if (!check_add_tensor_valid()) {
            return;
        }
        if (t.buffer.addr == 0) {
            set_error("INOUT tensor must have a non-NULL buffer address");
            return;
        }
        tensors[tensor_count] = &t;
        tensor_types[tensor_count] = TensorArgType::INOUT;
        tensor_count++;
    }

    void add_scalar(uint64_t v) {
        if (scalar_count >= MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=128)");
            return;
        }
        scalars[scalar_count++] = v;
    }

    void add_scalars(const uint64_t* values, int count) {
        if (scalar_count + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=128)");
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
        if (scalar_count + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=128)");
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
     * Copy scalars from another Arg's scalar array.
     * Useful when multiple tasks share the same scalar data (e.g., block indices).
     */
    void copy_scalars_from(const Arg& src, int src_offset, int count) {
        if (src_offset + count > src.scalar_count) {
            set_error("Source scalar range out of bounds in copy_scalars_from");
            return;
        }
        if (scalar_count + count > MAX_SCALAR_ARGS) {
            set_error("Too many scalar args (exceeds MAX_SCALAR_ARGS=128)");
            return;
        }
        memcpy(&scalars[scalar_count], &src.scalars[src_offset], count * sizeof(uint64_t));
        scalar_count += count;
    }
};

#endif  // SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_RUNTIME_PTO_TYPES_H_
