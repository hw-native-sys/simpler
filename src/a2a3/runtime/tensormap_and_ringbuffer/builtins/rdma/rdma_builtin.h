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

#include "backend/rdma/rdma_completion_kernel.h"

template <bool Put>
inline __aicore__ void rdma_builtin_entry(__gm__ int64_t *args) {
    const RdmaTransferArgsV1 request{static_cast<uint64_t>(args[0]), static_cast<uint64_t>(args[1]),
                                     static_cast<uint64_t>(args[2]), static_cast<uint64_t>(args[3]),
                                     static_cast<uint64_t>(args[4]), static_cast<uint64_t>(args[5])};
    using Scratch = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::rdma::kRdmaScratchBytes>;
    Scratch scratch;
    pto::TASSIGN(scratch, 0);
    AsyncCtx context = get_async_ctx(args);
    // This standalone kernel owns event 0 and its UB scratch for the whole request.
    (void)simpler::rdma::submit_checked<Put>(context, request, scratch, 0);
}
