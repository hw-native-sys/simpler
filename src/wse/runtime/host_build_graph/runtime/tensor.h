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
 * Both tensor types this runtime deals in: the boundary `ChipTensor` an argument
 * arrives as, and `simpler::hbg::Tensor`, the form everything inside the runtime —
 * orchestration included — actually works with.
 *
 * This sits first on the runtime include path, so a `#include "tensor.h"` from
 * runtime, orchestration or kernel sources lands here.
 */

#pragma once

#include "task_interface/tensor.h"
#include "host_build_graph/tensor.h"

// The tensor type of whichever runtime this translation unit is being built for.
// A kernel reads a payload element and does not care which orchestrator produced
// it, so kernels and the orchestration sources shared between runtimes name this
// rather than picking one — several are compiled under both.
//
// The name is unqualified because nothing on a kernel or orchestration include path
// reaches task_interface/buffer.h, which declares a different `Tensor` — the L3+
// wire form, at global scope. tests/lint/check_kernel_wire_isolation.py holds that
// separation; an include that breaks it turns this line into a redeclaration.
using Tensor = simpler::hbg::Tensor;
