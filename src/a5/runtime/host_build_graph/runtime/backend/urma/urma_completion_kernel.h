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

#pragma once

#include "async_kernel_api.h"
#include "backend/urma/urma_request_adapter.h"

template <typename DstTensor, typename SrcTensor>
inline __aicore__ bool send_request_entry(AsyncCtx &ctx, UrmaRequestDescriptor<DstTensor, SrcTensor> desc) {
    if (ctx.resident_enabled != 1) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        return false;
    }
    auto *group = pto2::resident::detail::group(ctx.resident);
    if (group == nullptr || group->urma_single_aiv != 1 || desc.workspace == nullptr ||
        reinterpret_cast<uintptr_t>(desc.workspace) % alignof(pto::comm::urma::UrmaInfo) != 0 ||
        simpler::urma::tensor_element_count(desc.dst) >
            pto::comm::urma::kUrmaMaxWqeTransferBytes / sizeof(typename DstTensor::RawDType)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    auto *info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(desc.workspace);
    if (info->layout != pto::comm::urma::UrmaLayout::PER_PEER || info->rankCount != 2 || info->jettiesPerCore != 1) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    simpler::urma::HbgRequestAdapter runtime(ctx.resident, ctx.urma_sender_id);
    return simpler::urma::submit_chunked_urma_request(runtime, desc);
}
