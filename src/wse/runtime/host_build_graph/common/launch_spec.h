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
#include <cstdint>
#include <vector>
#include "host_build_graph/host/orch_ops.h"

struct WseBlockPlacement {
    uint32_t core_id;
    std::vector<uint32_t> thread_pes;
};

// Argument references are borrowed for submit only; the runtime captures an owned copy.
struct WseLaunchSpec {
    uint32_t thread_num{1};
    std::vector<WseBlockPlacement> placement;
    std::vector<const CoreTaskArgs *> block_args;
};

struct WseOrchOps : OrchOps {
    void *launch_state;
    TaskOutputTensors (*submit_wse)(OrchContext *, int32_t, const CoreTaskArgs &, const WseLaunchSpec &);
};
