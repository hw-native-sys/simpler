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
#include <cstdio>
#include "platform_comm/ccu/ccu_builder.h"

extern "C" uint32_t simpler_ccu_module_abi_version() { return CCU_POSITIONAL_ABI_VERSION; }

extern "C" int reject_rank1(CcuBuildContext *builder) {
    if (!builder || builder->rank_count() != 2) return 1;
    const auto *path =
        builder->rank() == 1 ? PREP_FAILURE_MARKER_ROOT "/rank1-rejected" : PREP_FAILURE_MARKER_ROOT "/rank0-accepted";
    auto *marker = std::fopen(path, "w");
    if (!marker) return 18;
    if (std::fclose(marker) != 0) return 19;
    // Both ranks load identical bytes; only this valid user callback rejects its build.
    if (builder->rank() == 1) {
        std::fprintf(stderr, "[CCU prepare test] rank=1 builder rejected function\n");
        return 17;
    }
    std::fprintf(stderr, "[CCU prepare test] rank=0 builder accepted function\n");
    return builder->peer_barrier() == CcuStatus::OK ? 0 : 2;
}
