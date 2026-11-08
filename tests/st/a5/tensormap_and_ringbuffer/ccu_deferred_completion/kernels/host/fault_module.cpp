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

extern "C" uint32_t simpler_ccu_module_abi_version() { return CCU_POSITIONAL_ABI_VERSION; }

extern "C" int fault_broadcast(CcuBuildContext *builder) {
    if (!builder || builder->rank_count() != 2) return 1;
    const auto count = builder->scalar(0);
    builder->require_equal(count, builder->peer_scalar(0));
    builder->require_equal(count, builder->constant(64));
    const auto bytes = builder->multiply(count, 4);
    if (builder->rank() == 0) {
        if (builder->local_copy(builder->tensor(1), builder->tensor(0), bytes) != CcuStatus::OK ||
            builder->write_peer(builder->peer_tensor(1), builder->tensor(0), bytes) != CcuStatus::OK)
            return 2;
    }
    return builder->peer_barrier() == CcuStatus::OK ? 0 : 3;
}
