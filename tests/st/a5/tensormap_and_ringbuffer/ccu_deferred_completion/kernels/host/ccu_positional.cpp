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

namespace {
int broadcast(CcuBuildContext *builder, uint32_t root) {
    if (!builder || builder->rank_count() != 2) return 1;
    const auto source = builder->tensor(0);
    const auto destination = builder->tensor(1);
    const auto scratch = builder->tensor(2);
    const auto count = builder->scalar(0);
    builder->require_equal(count, builder->peer_scalar(0));
    builder->require_le(builder->constant(1), count);
    builder->require_le(count, builder->constant(1024));
    const auto bytes = builder->multiply(count, 4);
    builder->require_le(bytes, scratch.byte_size);
    if (builder->rank() == root) {
        if (builder->local_copy(destination, source, bytes) != CcuStatus::OK ||
            builder->write_peer(builder->peer_tensor(1), source, bytes) != CcuStatus::OK)
            return 2;
    }
    return builder->peer_barrier() == CcuStatus::OK ? 0 : 3;
}
}  // namespace

extern "C" int broadcast_root0(CcuBuildContext *builder) { return broadcast(builder, 0); }
extern "C" int broadcast_root1(CcuBuildContext *builder) { return broadcast(builder, 1); }
