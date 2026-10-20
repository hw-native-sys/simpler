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
 * SPMD sync_start MIX pending-spill Orchestration (a5)
 *
 * Port of a2a3 spmd_sync_start_mix_spill, sized for a5 (36 AIC + 72 AIV):
 *   P: AIV core_num=72, base_cl=0,  allow_early_resolve = scalar early_on (fills all AIV)
 *   C: MIX core_num=24, base_cl=72, require_sync_start=true, dep=[P]
 *      → 24 AIC idle→running, 48 AIV busy→pending (per-core spill + rendezvous)
 *
 * Args layout: [output], scalar: early_on
 */

#include <stddef.h>
#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)
#include "arg_with_deps.h"      // NOLINT(build/include_subdir)

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2
#define FUNC_SPMD_WRITE_AIV 3

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

static constexpr int64_t PRODUCER_SPIN_ITERS = 2000000;

static MixedKernels mix_kernels() {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;
    return mk;
}

static TaskId submit_aiv_producer(const simpler::tmr::Tensor &out, int16_t core_num, int64_t base_cl, bool early_on) {
    CoreTaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(PRODUCER_SPIN_ITERS);
    args.launch_spec.set_block_num(core_num);
    args.set_allow_early_resolve(early_on);
    return rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args).task_id();
}

static void submit_mix_sync_consumer(const simpler::tmr::Tensor &out, int16_t core_num, int64_t base_cl, TaskId dep) {
    CoreTaskArgsWithDeps<4> args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(0);  // consumer does not spin
    args.launch_spec.set_block_num(core_num);
    args.launch_spec.set_require_sync_start(true);
    args.add_dep(dep);
    rt_submit_task(mix_kernels(), args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::tmr::Tensor &ext_output = orch_args.tensor(0).ref();
    const simpler::tmr::Tensor &layout = orch_args.tensor(1).ref();
    const bool early_on = orch_args.scalar(0) != 0;

    // The spill this case exercises needs the producer on EVERY AIV core and the
    // consumer on EVERY cluster, so both widths are the device's own counts
    // rather than literals — the run always takes the whole device, and that
    // width differs between sim and silicon.
    const int32_t producer_blocks = rt_available_aiv_count();
    const int32_t consumer_blocks = rt_available_cluster_count();

    TaskId prod = submit_aiv_producer(ext_output, static_cast<int16_t>(producer_blocks), 0, early_on);
    submit_mix_sync_consumer(ext_output, static_cast<int16_t>(consumer_blocks), producer_blocks, prod);

    uint32_t idx[1] = {0};
    set_tensor_data<int32_t>(layout, 1, idx, producer_blocks);
    idx[0] = 1;
    set_tensor_data<int32_t>(layout, 1, idx, consumer_blocks);

    LOG_INFO(
        "[spmd_sync_start_mix_spill] early_on=%d AIV producer (%d) + sync_start MIX consumer (%d) submitted",
        early_on ? 1 : 0, producer_blocks, consumer_blocks
    );
}

}  // extern "C"
