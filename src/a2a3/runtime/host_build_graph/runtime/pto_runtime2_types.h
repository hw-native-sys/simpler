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

#ifndef SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_
#define SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_

// =============================================================================
// Profiling Configuration
// =============================================================================

#ifndef PTO2_PROFILING
#define PTO2_PROFILING 1
#endif

// Disable dual-issue (pipelined) AICPU->AICore dispatch. When 1, the
// scheduler only dispatches to a core when both the running and pending
// slots are empty, so each core has at most one outstanding task at any
// time. Orthogonal to PTO2_PROFILING / PMU: PMU users must set this
// explicitly, since overlapping in-flight tasks pollute PMU registers.
#ifndef PTO2_DISABLE_DUAL_ISSUE
#define PTO2_DISABLE_DUAL_ISSUE 0
#endif

#endif  // SRC_A2A3_RUNTIME_HOST_BUILD_GRAPH_RUNTIME_PTO_RUNTIME2_TYPES_H_
