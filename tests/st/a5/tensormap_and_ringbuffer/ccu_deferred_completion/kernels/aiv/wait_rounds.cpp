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
#include "common.h"
#include "backend/ccu/ccu_completion_kernel.h"

namespace {
constexpr uint32_t kElements = 2048;
constexpr uint32_t kResultWords = 8;

inline __aicore__ void load_wait_tensor(__gm__ int64_t *args, uint32_t index, Tensor &tensor) {
    const auto *source = reinterpret_cast<const __gm__ Tensor *>(args[index]);
    tensor.buffer.addr = source->buffer.addr;
    tensor.buffer.size = source->buffer.size;
    TaskId::assign(tensor.owner_task_id, source->owner_task_id);
    tensor.start_offset = source->start_offset;
    tensor.version = source->version;
    tensor.ndims = source->ndims;
    tensor.dtype = source->dtype;
    tensor.manual_dep = source->manual_dep;
    tensor.is_contiguous = source->is_contiguous;
    tensor.address_space = source->address_space;
    tensor.extent_elem_cache = source->extent_elem_cache;
    for (uint32_t dim = 0; dim < MAX_TENSOR_DIMS; ++dim) {
        tensor.shapes[dim] = source->shapes[dim];
        tensor.strides[dim] = source->strides[dim];
    }
}
}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *source = data<float>(args, 0);
    auto *destination = data<float>(args, 1);
    auto *scratch = data<float>(args, 2);
    auto *results = data<int32_t>(args, 3);
    const volatile __gm__ float *observed_destination = destination;
    const auto *domain = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const uint64_t rounds = args[5];
    const uint32_t rank = domain->rankId;
    AsyncCtx async = get_async_ctx(args);

    for (uint64_t round = 0; round < rounds; ++round) {
        const uint32_t count = round % 2 ? 1024 : 64;
        const uint32_t offset = 16 + (round % 8) * 16;
        const uint32_t root = (round / 2) % 2;
        const float sentinel = -static_cast<float>(round + 1);
        for (uint32_t i = 0; i < kElements; ++i) {
            source[i] = value(round, rank, i);
            destination[i] = sentinel;
            scratch[i] = -7.0f;
        }
        ccu_detail::Sync(source, kElements * sizeof(float));
        ccu_detail::Sync(destination, kElements * sizeof(float));
        ccu_detail::Sync(scratch, kElements * sizeof(float));

        Tensor input;
        Tensor output;
        Tensor temporary;
        load_wait_tensor(args, 0, input);
        load_wait_tensor(args, 1, output);
        load_wait_tensor(args, 2, temporary);
        input.start_offset += offset;
        output.start_offset += offset;
        input.shapes[0] = count;
        output.shapes[0] = count;
        CcuCallArgs parameters;
        parameters.add_tensor(input);
        parameters.add_tensor(output);
        parameters.add_tensor(temporary);
        parameters.add_scalar(count);

        // Volatile GM loads populate the old destination before the API call.
        int32_t prime_errors = 0;
        for (uint32_t i = 0; i < kElements; ++i)
            prime_errors += observed_destination[i] != sentinel;

        const bool completed = wait_request_entry(async, CcuCall(domain, root, parameters));
        // No user cache operation separates wait's return from these GM loads.
        int32_t errors = 0;
        int32_t first_mismatch = -1;
        for (uint32_t i = 0; i < kElements; ++i) {
            const float expected = i >= offset && i < offset + count ? value(round, root, i) : sentinel;
            if (observed_destination[i] != expected || source[i] != value(round, rank, i) || scratch[i] != -7.0f) {
                ++errors;
                if (first_mismatch < 0) first_mismatch = i;
            }
        }
        auto *result = results + round * kResultWords;
        result[0] = errors;
        result[1] = prime_errors;
        result[2] = root;
        result[3] = count;
        result[4] = offset;
        result[5] = first_mismatch;
        result[6] = completed ? 1 : 0;
        result[7] = async.completion_count ? static_cast<int32_t>(*async.completion_count) : -1;
        ccu_detail::Sync(result, kResultWords * sizeof(int32_t));
        if (!completed) return;
        // Value mismatches do not strand the peer in its next request. The next
        // CCU argument exchange waits until both AIVs submit after consumption.
    }
}
