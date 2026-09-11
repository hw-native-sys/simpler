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

#include <acl/acl_rt.h>

#include "host_build_graph/kernel_graph_template.h"
#include "worker/runtime_c_api.h"

namespace hbg {

// The caller has established the three-stream event ordering and owns the
// function's residency lease. The supplied stream is the dedicated AICPU stream.
// CANN consumes the writable copy during this call; the canonical template is
// never passed as writable storage. The return value is the actual enqueue status.
inline int launch_graph_template(
    const GraphLaunchTemplate &source, aclrtFuncHandle function, uint32_t blocks, aclrtStream aicpu_stream,
    aclrtLaunchKernelCfg *config = nullptr
) {
    if (function == nullptr || aicpu_stream == nullptr || blocks == 0) return PTO_RUNTIME_ERR_INTERNAL;
    struct Launch {
        aclrtFuncHandle function;
        uint32_t blocks;
        aclrtStream stream;
        aclrtLaunchKernelCfg *config;
    } launch{function, blocks, aicpu_stream, config};
    return submit_graph_template(source, {&launch, [](void *ctx, GraphHostArgs &args) {
                                              const auto &call = *static_cast<Launch *>(ctx);
                                              aclrtPlaceHolderInfo placeholder{args.address_offset, args.data_offset};
                                              return aclrtLaunchKernelWithHostArgs(
                                                  call.function, call.blocks, call.stream, call.config,
                                                  args.storage.data(), args.bytes, &placeholder, 1
                                              );
                                          }});
}

}  // namespace hbg
