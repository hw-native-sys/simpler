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

#include "platform_comm/ccu/ccu_types.h"

// Symbols name build-time values, never user-visible SDK handles or device addresses.
struct CcuSymbol {
    uint64_t index;
};
struct CcuBuildTensor {
    CcuSymbol address, token, byte_size;
};
struct CcuBuildModuleAbi {
    uint32_t abi_version;
    uint32_t struct_size;
    const char *sdk_build_id;
};

// Host-only builder. The execution layer installs request loading, peer metadata
// exchange and completion publication around the user's recorded data operations.
class CcuBuildContext {
public:
    virtual ~CcuBuildContext() = default;
    virtual uint32_t rank() const = 0;
    virtual uint32_t rank_count() const = 0;
    virtual CcuBuildTensor tensor(const char *name) = 0;
    virtual CcuBuildTensor peer_tensor(const char *name) = 0;
    virtual CcuSymbol scalar(const char *name) = 0;
    virtual CcuSymbol peer_scalar(const char *name) = 0;
    virtual CcuSymbol constant(uint64_t value) = 0;
    virtual CcuSymbol multiply(CcuSymbol value, uint16_t factor) = 0;
    virtual CcuStatus require_le(CcuSymbol value, CcuSymbol bound) = 0;
    virtual CcuStatus require_equal(CcuSymbol lhs, CcuSymbol rhs) = 0;
    virtual CcuStatus local_copy(CcuBuildTensor destination, CcuBuildTensor source, CcuSymbol bytes) = 0;
    virtual CcuStatus write_peer(CcuBuildTensor destination, CcuBuildTensor source, CcuSymbol bytes) = 0;
    virtual CcuStatus peer_barrier() = 0;
};

using CcuBuildEntry = int (*)(CcuBuildContext *);
