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

#include "host/profiling_copy.h"

#include <runtime/rt.h>

int profiling_copy_to_device(volatile void *dev_dst, const void *host_src, size_t size) {
    return rtMemcpy(const_cast<void *>(dev_dst), size, host_src, size, RT_MEMCPY_HOST_TO_DEVICE);
}

int profiling_copy_from_device(volatile void *host_dst, const volatile void *dev_src, size_t size) {
    return rtMemcpy(
        const_cast<void *>(host_dst), size, const_cast<const void *>(dev_src), size, RT_MEMCPY_DEVICE_TO_HOST
    );
}
