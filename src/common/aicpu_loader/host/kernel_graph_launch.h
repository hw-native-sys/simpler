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
#include "host_build_graph/kernel_graph_prepared.h"
#include "worker/runtime_c_api.h"

namespace hbg {

inline int launch_graph_prepared(
    const GraphPreparedDispatch &source, aclrtFuncHandle function, uint32_t blocks, aclrtStream stream
) {
    GraphPreparedDispatch args = source;
    return aclrtLaunchKernelWithHostArgs(function, blocks, stream, nullptr, &args, sizeof(args), nullptr, 0);
}

inline int launch_graph_host_args(
    GraphHostArgs &args, aclrtFuncHandle function, uint32_t blocks, aclrtStream stream, aclrtLaunchKernelCfg *config
) {
    aclrtPlaceHolderInfo placeholder{args.address_offset, args.data_offset};
    const uint64_t zero = 0;
    std::memcpy(reinterpret_cast<uint8_t *>(args.storage.data()) + args.address_offset, &zero, sizeof(zero));
    return aclrtLaunchKernelWithHostArgs(
        function, blocks, stream, config, args.storage.data(), args.bytes, &placeholder, 1
    );
}

// The caller has established the three-stream event ordering and owns the
// function's residency lease. The supplied stream is the dedicated AICPU stream.
// CANN consumes the writable copy during this call; the canonical template is
// never passed as writable storage. The return value is the actual enqueue status.
inline int launch_graph_template(
    const GraphLaunchTemplate &source, const HbgCallableRegistration &registration, aclrtFuncHandle function,
    uint32_t blocks, aclrtStream aicpu_stream, aclrtLaunchKernelCfg *config = nullptr
) {
    if (function == nullptr || aicpu_stream == nullptr || blocks == 0) return PTO_RUNTIME_ERR_INTERNAL;
    struct Launch {
        aclrtFuncHandle function;
        uint32_t blocks;
        aclrtStream stream;
        aclrtLaunchKernelCfg *config;
    } launch{function, blocks, aicpu_stream, config};
    return submit_graph_dispatch(
        source, registration,
        {&launch, [](void *ctx, GraphHostArgs &args) {
             const auto &call = *static_cast<Launch *>(ctx);
             return launch_graph_host_args(args, call.function, call.blocks, call.stream, call.config);
         }}
    );
}

}  // namespace hbg
