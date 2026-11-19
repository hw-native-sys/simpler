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

#include <stdint.h>

#include <stdexcept>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

namespace {

struct CohortConfig : simpler::hbg::ConfigBase {
    int16_t blocks;
    explicit CohortConfig(int16_t count) :
        blocks(count) {}
    uint64_t hash() const { return simpler::hbg::config_hash(blocks); }
    bool operator==(const CohortConfig &other) const { return blocks == other.blocks; }
};

void mix_spmd_layer(const GraphTaskArgs &args, const CohortConfig &config) {
    MixedKernels kernels;
    kernels.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    kernels.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    kernels.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;

    CoreTaskArgs task_args;
    task_args.add_inout(args.tensor(0).ref());
    task_args.add_scalar(int64_t{0});
    task_args.launch_spec.set_block_num(config.blocks);
    task_args.launch_spec.set_require_sync_start(true);
    rt_submit_task(kernels, task_args);
}

void submit_layer(const GraphTaskArgs &args, const CohortConfig &config) {
    rt_submit_graph(&mix_spmd_layer, args, config);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &args) {
    (void)args;
    return OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const int64_t requested = args.scalar<int64_t>(0);
    const int64_t capacity = rt_available_cluster_count();
    if (requested != 0 && (requested < 2 || requested >= capacity))
        throw std::invalid_argument("Graph partial sync width must be at least two and below cluster capacity");
    const CohortConfig config(static_cast<int16_t>(requested == 0 ? capacity : requested));
    for (int32_t output_index = 0; output_index < 3; ++output_index) {
        GraphTaskArgs layer_args;
        layer_args.add_inout(args.tensor(output_index).ref());
        submit_layer(layer_args, config);
    }
}

}  // extern "C"
