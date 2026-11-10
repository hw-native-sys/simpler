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
// Inert orchestration image for exercising the actual registration/dlopen path.
// Execution is supplied by the existing native-admission CPU transport.
extern "C" void framework_observation_orch(const void *) {}
extern "C" void framework_observation_orch_b(const void *) {}
extern "C" void framework_bind_orch_context(void *) {}
extern "C" int simpler_host_log_bind_state(void *) { return 0; }
