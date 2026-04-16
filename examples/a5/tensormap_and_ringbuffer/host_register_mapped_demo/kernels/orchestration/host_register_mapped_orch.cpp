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
/**
 * Demo orchestration for host-side mapped memory on a5.
 *
 * Args layout in ChipStorageTaskArgs:
 *   tensor(0): mapped_out (host tensor copied back by runtime)
 *   scalar(0): mapped_dev_ptr (device-visible address returned by mallocHostDeviceShareMem)
 */

#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 2,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    const ContinuousTensor &out_arg = orch_args.tensor(0);
    Tensor mapped_out = from_tensor_arg(out_arg);

    uint64_t mapped_input_u64 = orch_args.scalar(0);
    Tensor mapped_host_buffer = make_tensor_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(mapped_input_u64)), out_arg.shapes, out_arg.ndims, out_arg.dtype
    );

    LOG_INFO(
        "a5_host_register_mapped_demo: mapped_host_buffer=0x%lx mapped_out=0x%lx elements=%u",
        mapped_input_u64, out_arg.data, out_arg.shapes[0]
    );

    Arg params;
    params.add_inout(mapped_host_buffer);
    params.add_output(mapped_out);
    pto2_rt_submit_aiv_task(0, params);
}

}  // extern "C"
