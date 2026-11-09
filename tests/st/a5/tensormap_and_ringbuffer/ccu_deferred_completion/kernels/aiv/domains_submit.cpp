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

inline __aicore__ void load_domains_tensor(__gm__ int64_t *args, uint32_t index, Tensor &tensor) {
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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    const auto *domain = reinterpret_cast<__gm__ CommContext *>(args[4]);
    const uint64_t round = args[5];
    const uint64_t count = round % 2 ? 1024 : 64;
    const uint32_t offset = 16 + (round % 8) * 16;
    Tensor source;
    Tensor destination;
    Tensor scratch;
    load_domains_tensor(args, 0, source);
    load_domains_tensor(args, 1, destination);
    load_domains_tensor(args, 2, scratch);
    source.start_offset += offset;
    destination.start_offset += offset;
    source.shapes[0] = count;
    destination.shapes[0] = count;
    CcuCallArgs parameters;
    parameters.add_tensor(source);
    parameters.add_tensor(destination);
    parameters.add_tensor(scratch);
    parameters.add_scalar(count);
    AsyncCtx async = get_async_ctx(args);
    const bool submitted = send_request_entry(async, CcuCall(domain, 0, parameters));
    auto *result = data<int32_t>(args, 3);
    result[round * 4] = submitted ? -2 : -3;
    ccu_detail::Sync(result, (round + 1) * 16);
    // No local wait: the runtime owns the request completion after this AIV returns.
}
