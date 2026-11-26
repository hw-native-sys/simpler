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

#include <atomic>
#include <cstdlib>

#include "worker/runtime_c_api.h"

class Runtime;
struct HostApi;

namespace {
std::atomic<bool> fail_next_publish{false};
}  // namespace

extern "C" void arm_calendar_publish_fault() { fail_next_publish.store(true); }

extern "C" int publish_run_image_impl(Runtime *runtime, const HostApi *api) {
    // A5sim has already bound the graph and reserved its Calendar sample when it publishes the image.
    if (fail_next_publish.exchange(false)) return PTO_RUNTIME_ERR_INTERNAL;

    void *host_runtime = dlopen("libhost_runtime.so", RTLD_NOW | RTLD_NOLOAD);
    if (host_runtime == nullptr) std::abort();
    auto publish = reinterpret_cast<decltype(&publish_run_image_impl)>(dlsym(host_runtime, "publish_run_image_impl"));
    if (publish == nullptr || publish == publish_run_image_impl) std::abort();
    int rc = publish(runtime, api);
    dlclose(host_runtime);
    return rc;
}
