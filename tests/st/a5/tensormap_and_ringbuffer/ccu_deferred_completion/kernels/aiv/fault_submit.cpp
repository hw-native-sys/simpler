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
#include "../fault_handshake.h"

inline __aicore__ void load_geometry(const __gm__ Tensor *source, Tensor &tensor) {
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
    const auto *domain = reinterpret_cast<const __gm__ CommContext *>(args[2]);
    auto *control = reinterpret_cast<volatile __gm__ uint32_t *>(args[3]);
    const uint64_t mismatch = args[4];
    AsyncCtx async = get_async_ctx(args);
    if (domain->rankId == 1) {
        const uint64_t local_base = domain->windowsIn[1];
        const uint64_t control_addr = reinterpret_cast<uint64_t>(control);
        const uint64_t control_bytes = ccu_fault::kControlWords * sizeof(uint32_t);
        if (domain->windowsIn[0] == 0 || control_addr < local_base ||
            !ccu_detail::Fits(control_addr - local_base, control_bytes, domain->winSize)) {
            pto2::detail::defer_error(async, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            pto2::detail::defer_flush(async);
            return;
        }
        // IPC imports provide a read-only peer mapping at the same buffer offset.
        auto *peer = reinterpret_cast<__gm__ uint32_t *>(domain->windowsIn[0] + control_addr - local_base);
        const uint64_t start = get_sys_cnt();
        uint32_t observed_width = 0;
        bool ready = false;
        while (get_sys_cnt() - start < ccu_fault::kHandshakeTimeoutNs) {
            ccu_detail::Sync(peer, control_bytes);
            const volatile __gm__ uint32_t *flags = peer;
            const uint32_t width = flags[ccu_fault::kWitnessWidth];
            if (width == 0) continue;
            if (width > ccu_fault::kMaxAiv) break;
            bool all_observed = true;
            bool rejected = false;
            for (uint32_t lane = 0; lane < width; ++lane) {
                const uint32_t flag = flags[ccu_fault::kWitnessBase + lane * ccu_fault::kLineWords];
                all_observed = all_observed && flag == ccu_fault::kReady;
                rejected = rejected || flag > ccu_fault::kReady;
            }
            if (rejected) break;
            if (all_observed && flags[ccu_fault::kSuccessorPublished] == ccu_fault::kReady) {
                observed_width = width;
                ready = true;
                break;
            }
        }
        if (!ready) {
            pto2::detail::defer_error(async, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            pto2::detail::defer_flush(async);
            return;
        }
        control[ccu_fault::kPeerReadySeen] = ccu_fault::kReady;
        control[ccu_fault::kPeerWidthSeen] = observed_width;
    }
    Tensor source;
    Tensor destination;
    load_geometry(reinterpret_cast<const __gm__ Tensor *>(args[0]), source);
    load_geometry(reinterpret_cast<const __gm__ Tensor *>(args[1]), destination);
    CcuCallArgs parameters;
    parameters.add_tensor(source);
    parameters.add_tensor(destination);
    parameters.add_scalar(uint64_t{64} + (mismatch ? domain->rankId : 0));
    const bool accepted = send_request_entry(async, CcuCall(domain, 0, parameters));
    control[ccu_fault::kSubmitted] = accepted ? ccu_fault::kReady : ccu_fault::kRejected;
    pto2::detail::defer_flush_range(control, 64);
    dsb(DSB_DDR);
}
