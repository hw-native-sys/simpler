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

inline __aicore__ void read_view(__gm__ int64_t *args, uint32_t index, Tensor &tensor) {
    const auto *source = reinterpret_cast<const __gm__ Tensor *>(args[index]);
    tensor.buffer.addr = source->buffer.addr;
    tensor.buffer.size = source->buffer.size;
    tensor.start_offset = source->start_offset;
    tensor.ndims = source->ndims;
    tensor.dtype = source->dtype;
    tensor.address_space = source->address_space;
    for (uint32_t dim = 0; dim < MAX_TENSOR_DIMS; ++dim) {
        tensor.shapes[dim] = source->shapes[dim];
        tensor.strides[dim] = source->strides[dim];
    }
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    using namespace simpler::ccu::device;
    using Protocol = simpler::ccu::device::Protocol<>;
    const auto *domain = reinterpret_cast<const __gm__ CommContext *>(args[5]);
    const uint64_t round = args[6];
    const uint64_t count = round % 2 ? 1024 : 64;
    const uint32_t offset = 16 + (round % 8) * 16;
    Tensor source, destination, scratch, foreign;
    read_view(args, 0, source);
    read_view(args, 1, destination);
    read_view(args, 2, scratch);
    read_view(args, 4, foreign);
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
    const auto &workspace = comm_workspace<CommWorkspaceKind::CCU>(*domain);
    uint64_t image_bytes = 0;
    auto *instance = Protocol::instance(
        reinterpret_cast<__gm__ DomainDirectory *>(workspace.addr), workspace.size, async.callable_id, domain->rankId,
        domain->rankNum, image_bytes
    );
    Reference ref;
    bool rejected = instance && Protocol::resolve(instance, image_bytes, async.callable_id, (round / 2) % 2, ref);
    // Each rejected reservation must leave the slot usable by the genuine request below.
    for (uint32_t probe = 0; probe < 5 && rejected; ++probe) {
        auto invalid = parameters.values();
        if (probe == 0) invalid.tensors[0].addr = foreign.buffer.addr;
        else if (probe == 1) invalid.tensors[0].bytes = UINT64_MAX;
        else if (probe == 2) invalid.tensors[0].addr += 1;
        else if (probe == 3) invalid.tensor_count = MAX_TENSORS + 1;
        else invalid.scalars[0].width = 8;
        CcuStatus status{};
        const bool accepted = Protocol::reserve(ref, invalid, &status);
        if (accepted) Protocol::cancel(ref);
        rejected = !accepted && status == CcuStatus::INVALID_ARGUMENT;
    }
    const bool submitted = send_request_entry(async, CcuCall(domain, (round / 2) % 2, parameters));
    auto *result = data<int32_t>(args, 3);
    result[round * 4] = submitted && rejected ? -2 : -4;
    ccu_detail::Sync(result, (round + 1) * 16);
}
