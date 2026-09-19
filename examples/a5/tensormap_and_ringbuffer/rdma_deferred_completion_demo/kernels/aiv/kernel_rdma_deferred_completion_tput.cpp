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
#include "rdma_deferred_completion_common.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *send_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *marker_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ CommContext *comm_ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);
    uint32_t elem_count = static_cast<uint32_t>(args[5]);

    __gm__ float *send = rdma_deferred_completion::tensor_data<float>(send_tensor);
    __gm__ float *tput = rdma_deferred_completion::tensor_data<float>(tput_tensor);
    __gm__ int32_t *marker = rdma_deferred_completion::tensor_data<int32_t>(marker_tensor);
    if (!rdma_deferred_completion::validate_comm(comm_ctx, elem_count, marker)) {
        return;
    }
    uint32_t peer = rdma_deferred_completion::peer_rank(comm_ctx);
    uint64_t peer_base = rdma_deferred_completion::remote_base(comm_ctx, peer);
    uint64_t tput_slot_offset = rdma_deferred_completion::local_offset(comm_ctx, tput) +
                                static_cast<uint64_t>(comm_ctx->rankId) * elem_count * sizeof(float);
    __gm__ float *remote_tput_slot = reinterpret_cast<__gm__ float *>(peer_base + tput_slot_offset);

    uint32_t first_count = rdma_deferred_completion::first_chunk_count(elem_count);
    uint32_t second_count = rdma_deferred_completion::second_chunk_count(elem_count);
    AsyncCtx async_ctx = get_async_ctx(args);

    auto local_send = rdma_deferred_completion::global_float(send, first_count);
    auto remote_tput = rdma_deferred_completion::global_float(remote_tput_slot, first_count);
    auto rdma_scratch = rdma_deferred_completion::rdma_scratch_tile();
    uint32_t request_count = 1;
    pto2::rdma_backend::RdmaSubmitStatus submit_status = pto2::rdma_backend::submit_rdma_request_status(
        async_ctx, RdmaTput(
                       remote_tput, local_send, rdma_scratch, reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace),
                       peer, comm_ctx->rankId, 0
                   )
    );
    if (submit_status != pto2::rdma_backend::RdmaSubmitStatus::OK) {
        rdma_deferred_completion::store_marker(marker, static_cast<int32_t>(submit_status));
        return;
    }
    if (second_count != 0) {
        request_count++;
        auto local_send_tail = rdma_deferred_completion::global_float(send + first_count, second_count);
        auto remote_tput_tail = rdma_deferred_completion::global_float(remote_tput_slot + first_count, second_count);
        submit_status = pto2::rdma_backend::submit_rdma_request_status(
            async_ctx, RdmaTput(
                           remote_tput_tail, local_send_tail, rdma_scratch,
                           reinterpret_cast<__gm__ uint8_t *>(comm_ctx->workSpace), peer, comm_ctx->rankId, 1
                       )
        );
        if (submit_status != pto2::rdma_backend::RdmaSubmitStatus::OK) {
            rdma_deferred_completion::store_marker(marker, static_cast<int32_t>(submit_status));
            return;
        }
    }
    rdma_deferred_completion::store_marker(marker, static_cast<int32_t>(request_count));
}
