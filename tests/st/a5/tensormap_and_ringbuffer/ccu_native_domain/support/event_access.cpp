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

#include <cstdint>
#include "acl/acl.h"

namespace {
aclrtBinHandle binary{};
aclrtFuncHandle function{};
aclrtStream stream{};
void *scratch{};
}  // namespace

// Test-only access to mapped Events, using the same AIV binary as execution-boundary reset.
extern "C" int ccu_test_event_access(const char *path, uint64_t address, uint64_t value, uint64_t *result) {
    int rc = 0;
    if (!binary) {
        if ((rc = aclrtBinaryLoadFromFile(path, nullptr, &binary)) != 0) return rc;
        if ((rc = aclrtBinaryGetFunction(binary, "simpler_ccu_event_trigger", &function)) != 0) return rc;
        if ((rc = aclrtCreateStream(&stream)) != 0) return rc;
        if ((rc = aclrtMalloc(&scratch, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST)) != 0) return rc;
    }
    uint64_t args[]{address, value, reinterpret_cast<uint64_t>(scratch)};
    aclrtLaunchKernelAttr engine{};
    engine.id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    engine.value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    aclrtLaunchKernelCfg cfg{&engine, 1};
    if ((rc = aclrtLaunchKernelWithHostArgs(function, 1, stream, &cfg, args, sizeof(args), nullptr, 0)) != 0) return rc;
    if ((rc = aclrtSynchronizeStreamWithTimeout(stream, 10000)) != 0) return rc;
    return aclrtMemcpy(result, sizeof(*result), scratch, sizeof(*result), ACL_MEMCPY_DEVICE_TO_HOST);
}

extern "C" int ccu_test_event_close() {
    int rc = aclrtFree(scratch);
    if (rc != 0) return rc;
    scratch = nullptr;
    if ((rc = aclrtDestroyStream(stream)) != 0) return rc;
    stream = nullptr;
    if ((rc = aclrtBinaryUnLoad(binary)) != 0) return rc;
    binary = nullptr;
    function = nullptr;
    return 0;
}
