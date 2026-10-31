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

#include "device_runner_base.h"
template <typename Runner>
struct NativeRunContext;

// Submission temporarily owns the preparation outside the run's phase slots.
int prewarm_kernel_run(
    NativeRunContext<DeviceRunnerBase> &state, DeviceRunnerBase::PreparedExecution &prepared,
    bool retain_registration = false
);
int quiesce_kernel_run(NativeRunContext<DeviceRunnerBase> &state);
int read_kernel_run_status(NativeRunContext<DeviceRunnerBase> &state);
int observe_kernel_run(NativeRunContext<DeviceRunnerBase> &state, bool wait);
int release_kernel_run(NativeRunContext<DeviceRunnerBase> &state);

// Registry mutation is a cold operation. A captured graph may remain alive,
// but every submitted replay must be quiescent before code or residency changes.
int quiesce_kernel_capture_for_callable_change(DeviceRunnerBase *runner, int32_t callable_id, bool revoke);
