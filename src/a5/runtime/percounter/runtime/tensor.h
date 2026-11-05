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
/**
 * Which tensor type this runtime deals in.
 *
 * Four directories on a kernel or orchestration include path hold a `tensor.h`,
 * and only a runtime's own says which one its sources mean. This runtime takes
 * host_build_graph's sources and so its tensor, but it still has to say so
 * here: a runtime that owns no `tensor.h` resolves the bare include to
 * task_interface's, which declares no `Tensor` at all.
 *
 * It sits first on this runtime's include path, so a `#include "tensor.h"` from
 * runtime, orchestration or kernel sources lands here.
 */

#pragma once

#include "task_interface/tensor.h"
#include "host_build_graph/tensor.h"

// The name is unqualified for the same reason host_build_graph's own shim gives:
// nothing on a kernel or orchestration include path reaches task_interface/buffer.h,
// which declares a different `Tensor` at global scope.
using Tensor = simpler::hbg::Tensor;
