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
 * a2a3 sim profiling_copy stubs.
 *
 * a2a3 sim runs every "device" memory access in-process through plain pointers
 * (the sim DeviceRunner returns host pointers from `allocate_tensor`), so the
 * mirror/copy hooks are no-ops here. See the a2a3 onboard sibling for the full
 * rationale on why these stubs exist.
 */

#include "host/profiling_copy.h"

int profiling_copy_to_device(volatile void * /*dev_dst*/, const void * /*host_src*/, size_t /*size*/) { return 0; }

int profiling_copy_from_device(
    volatile void * /*host_dst*/, const volatile void * /*dev_src*/, size_t /*size*/
) {
    return 0;
}
