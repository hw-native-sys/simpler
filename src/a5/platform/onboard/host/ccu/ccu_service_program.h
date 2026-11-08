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
#include "ccu_session_backend.h"
#include "hccl/hccl.h"
#include "hcomm/ccu/ccu_types.h"

struct CcuServiceBuildArgs {
    ChannelHandle channel;
    CcuEventHandle gate;
    uint32_t rank;
    uint32_t argument_bytes;
    uint64_t epoch;
    const std::vector<CcuRegisteredFunction> *functions;
};
CcuResult CcuServiceProgram(const CcuServiceBuildArgs *build) noexcept;
