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

#include <memory>
#include "ccu_run_lease.h"
#include "task_interface/callable.h"
#include "task_interface/task_args.h"

int platform_ccu_register(void *runner, int32_t callable_id, const ChipCallable &callable);
int platform_ccu_unregister(void *runner, int32_t callable_id);
void platform_ccu_forget_runner(void *runner);
// Persistent services must stop before whole-device synchronization or reset.
bool platform_ccu_has_device_owners(void *runner);
bool platform_ccu_close_runner(void *runner);
std::shared_ptr<simpler::ccu::RunLease>
platform_ccu_acquire_run(void *runner, int32_t callable_id, const ChipStorageTaskArgs &args);
