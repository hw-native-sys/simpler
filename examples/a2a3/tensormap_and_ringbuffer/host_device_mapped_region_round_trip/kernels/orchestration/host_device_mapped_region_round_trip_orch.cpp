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

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
host_device_mapped_region_round_trip_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 4};
}

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    return host_device_mapped_region_round_trip_config(orch_args);
}

__attribute__((visibility("default"))) void host_device_mapped_region_round_trip_orch(const ChipStorageTaskArgs &orch_args) {
    Arg args;
    args.add_scalar(orch_args.scalar(0));
    args.add_scalar(orch_args.scalar(1));
    args.add_scalar(orch_args.scalar(2));
    args.add_scalar(orch_args.scalar(3));
    rt_submit_aiv_task(0, args);
}

}  // extern "C"
