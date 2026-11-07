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
#include "platform_comm/ccu/ccu_builder.h"

#if defined(CCU_TEST_BAD_ABI)
static constexpr const char *kSdkId = "incompatible-sdk";
#else
static constexpr const char *kSdkId = "ccu-unit-test-sdk";
#endif
extern "C" const CcuBuildModuleAbi ccu_build_module_abi{CCU_BUILD_ABI_VERSION, sizeof(CcuBuildModuleAbi), kSdkId};

extern "C" int build_root_copy(CcuBuildContext *builder) {
    if (!builder || builder->rank_count() != 2) return 1;
    const auto src = builder->tensor("src");
    const auto dst = builder->tensor("dst");
    const auto peer_dst = builder->peer_tensor("dst");
    const auto count = builder->scalar("count");
    if (builder->require_le(builder->constant(1), count) != CcuStatus::OK ||
        builder->require_le(count, builder->constant(1024)) != CcuStatus::OK ||
        builder->require_equal(count, builder->peer_scalar("count")) != CcuStatus::OK)
        return 2;
    const auto bytes = builder->multiply(count, 4);
    if (builder->rank() == 0 && (builder->local_copy(dst, src, bytes) != CcuStatus::OK ||
                                 builder->write_peer(peer_dst, src, bytes) != CcuStatus::OK))
        return 3;
    return builder->peer_barrier() == CcuStatus::OK ? 0 : 4;
}
