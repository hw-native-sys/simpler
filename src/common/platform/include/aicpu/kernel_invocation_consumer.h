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

#include <cstddef>

#include "kernel_callable_residency.h"

// Called only after envelope and current device residency validation. Runtime
// consumers validate their payload layout/capacity before accessing its data.
// This function does not own the launch packet or the resident allocation.
int consume_kernel_invocation(
    const SimplerKernelInvocationHeader &invocation, const KernelCallableDeviceResidency &resident, const void *payload,
    size_t payload_bytes
);
