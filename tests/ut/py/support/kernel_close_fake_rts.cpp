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

#include <dlfcn.h>

// SDK substitutes exercise preload accounting without owning a device.
extern "C" int rtDeviceReset(int) { return 0; }
extern "C" int aclrtResetDevice(int device) { return rtDeviceReset(device); }
namespace {
bool delegate_free = false;
thread_local bool inside_free = false;
}  // namespace
extern "C" void set_delegated_free(int enabled) { delegate_free = enabled != 0; }
extern "C" int rtFree(void *ptr) {
    if (!delegate_free || inside_free) return 0;
    inside_free = true;
    const int rc = reinterpret_cast<int (*)(void *)>(dlsym(RTLD_DEFAULT, "rtFree"))(ptr);
    inside_free = false;
    return rc;
}
extern "C" int aclrtDestroyEvent(void *) { return 0; }
extern "C" int aclmdlRICaptureGetInfo(void *, void *status, void **model) {
    *static_cast<int *>(status) = 0;
    *model = nullptr;
    return 0;
}

extern "C" int sdk_internal_free(void *ptr) {
    return reinterpret_cast<int (*)(void *)>(dlsym(RTLD_DEFAULT, "rtFree"))(ptr);
}
