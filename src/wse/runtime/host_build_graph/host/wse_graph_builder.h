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
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/host/graph_submit.h"
#include "host_build_graph/host/dep_compute.h"
TaskOutputTensors
wse_submit_task_with_dependencies(OrchestratorState *, const MixedKernels &, const CoreTaskArgs &, const DepInputs *);
TaskOutputTensors wse_graph_record_submit_with_dependencies(
    OrchestratorState *, const CoreTaskArgs &, LaneMask, TaskAttrs, int32_t, int32_t, int32_t, const DepInputs *
);
