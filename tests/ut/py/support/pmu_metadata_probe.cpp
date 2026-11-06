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
#include <string>

#include "host/pmu_collector.h"
#include "runtime_c_api.h"

// begin_run is implemented in the linked runtime. A fresh collector exercises
// its host-only configuration path without initializing device storage.
extern "C" int check_pmu_metadata_invalidation(void *runtime_handle, const char *csv_path) {
    void *api = dlsym(runtime_handle, "create_device_context");
    Dl_info api_owner{}, linked_owner{};
    if (api == nullptr) return -1;
    if (!dladdr(api, &api_owner) || !dladdr(reinterpret_cast<void *>(&create_device_context), &linked_owner) ||
        api_owner.dli_fbase != linked_owner.dli_fbase)
        return -2;
    PmuCollector collector;
    return collector.begin_run(std::string(csv_path), PmuEventType::PIPE_UTILIZATION, 1) ? 1 : 0;
}
