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
 * a5 host_build_graph selective task-timing-slots demo (issue #1325).
 *
 *   out = (a + b) + b
 *
 * A two-task chain built with the legacy add_task orchestration API:
 *   t0: c   = a + b   (func 0, slot 0)  [first dispatch]
 *   t1: out = c + b   (func 0, slot 1)  [last finish, depends on t0]
 *
 * Each task is tagged post-submit via set_task_timing_slot() — the
 * legacy-runtime equivalent of RT2's L0TaskArgs::set_task_timing_slot. The
 * AICPU folds each task's dispatch/finish into its slot and the host emits
 * `simpler_run.runner_run.device_wall.task_slot_{0,1}` on the [STRACE] timeline.
 */

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

extern "C" {

int build_task_timing_a5hbg_graph(OrchestrationRuntime *runtime, const ChipStorageTaskArgs &orch_args) {
    void *host_a = orch_args.tensor(0).data_as<void>();
    void *host_b = orch_args.tensor(1).data_as<void>();
    void *host_out = orch_args.tensor(2).data_as<void>();
    size_t nbytes = orch_args.tensor(0).nbytes();
    uint32_t size = orch_args.tensor(0).shapes[0];

    TensorInfo a_info = make_tensor_info_from_tensor_arg(orch_args.tensor(0));
    TensorInfo b_info = make_tensor_info_from_tensor_arg(orch_args.tensor(1));
    TensorInfo out_info = make_tensor_info_from_tensor_arg(orch_args.tensor(2));

    void *dev_a = device_malloc(runtime, nbytes);
    copy_to_device(runtime, dev_a, host_a, nbytes);
    void *dev_b = device_malloc(runtime, nbytes);
    copy_to_device(runtime, dev_b, host_b, nbytes);
    void *dev_c = device_malloc(runtime, nbytes);  // intermediate
    void *dev_out = device_malloc(runtime, nbytes);
    record_tensor_pair(runtime, host_out, dev_out, nbytes);  // D2H copy-back at finalize

    // t0: c = a + b, tagged slot 0 (interval start).
    uint64_t args_t0[4] = {
        reinterpret_cast<uint64_t>(dev_a), reinterpret_cast<uint64_t>(dev_b), reinterpret_cast<uint64_t>(dev_c), size
    };
    int t0 = add_task(runtime, args_t0, 4, 0, CoreType::AIV);
    TensorInfo t0_info[] = {a_info, b_info, out_info};
    set_tensor_info_to_task(runtime, t0, t0_info, 3);
    set_task_timing_slot(runtime, t0, 0);

    // t1: out = c + b, tagged slot 1 (interval end). Depends on c -> runs after t0.
    uint64_t args_t1[4] = {
        reinterpret_cast<uint64_t>(dev_c), reinterpret_cast<uint64_t>(dev_b), reinterpret_cast<uint64_t>(dev_out), size
    };
    int t1 = add_task(runtime, args_t1, 4, 0, CoreType::AIV);
    TensorInfo t1_info[] = {out_info, b_info, out_info};
    set_tensor_info_to_task(runtime, t1, t1_info, 3);
    set_task_timing_slot(runtime, t1, 1);

    add_successor(runtime, t0, t1);

    return 0;
}

}  // extern "C"
