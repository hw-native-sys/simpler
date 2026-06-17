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
#include "common.h"
#include "pto_orchestration_api.h"

struct PTO2Runtime;

namespace {
PTO2Runtime *g_current_runtime = nullptr;
}  // namespace

extern "C" __attribute__((visibility("default"))) void framework_bind_runtime(PTO2Runtime *rt)
{
    g_current_runtime = rt;
}

// Keep current_runtime local to this .so so orchestration helpers do not
// accidentally bind to the AICPU binary's same-named symbol.
extern "C" __attribute__((visibility("hidden"))) PTO2Runtime *framework_current_runtime()
{
    return g_current_runtime;
}
