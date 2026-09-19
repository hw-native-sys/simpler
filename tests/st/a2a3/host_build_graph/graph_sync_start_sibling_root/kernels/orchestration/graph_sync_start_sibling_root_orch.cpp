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
 * In-Graph sync_start Beside a Sibling Root (host_build_graph)
 *
 * One shell release stages every qualifying body root at once. This body gives
 * it two: a `require_sync_start` MIX root that needs the whole device, and an
 * ordinary AIC root on a separate tensor.
 *
 *   seed (AIC, flagged, slow)  ->  Graph shell  ->  body { sync_root, sibling, tail }
 *
 * The sibling holds gated cores until its own doorbell rings, so the cohort
 * cannot fit and takes the global drain. The drain can only succeed once those
 * cores come free, and they come free only once `activate_graph_task` routes the
 * body's roots — which is why the scheduler loop must reach its graph control
 * work while a drain is pending.
 *
 * The sibling writes its own tensor rather than `out`: sharing one would make it
 * a consumer of the cohort through the anti-dependency rule, and the scene needs
 * two independent roots staged by the same release.
 *
 * Args: [output, sibling_output, layout]
 */

#include <stddef.h>
#include <stdint.h>

#include "orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_AIC 0
#define FUNC_MIX_AIC 1
#define FUNC_MIX_AIV0 2
#define FUNC_MIX_AIV1 3

// Cache-line layout of `output`: seed | sync cohort (3 lines per block) | tail.
static constexpr int64_t SEED_BASE_CL = 0;
static constexpr int16_t SEED_BLOCKS = 8;
static constexpr int64_t SYNC_BASE_CL = 8;
static constexpr int64_t TAIL_BASE_CL = 80;  // SYNC_BASE_CL + MAX_CLUSTERS(24) * 3
static constexpr int16_t TAIL_BLOCKS = 2;
// The sibling root: its own tensor, and wide enough to hold cores the cohort wants.
static constexpr int64_t SIBLING_BASE_CL = 0;
static constexpr int16_t SIBLING_BLOCKS = 4;

// Long enough that the seed is still on-core while the shell materializes and
// stages the body's roots. Same constant the top-level twin uses.
static constexpr int64_t PRODUCER_SPIN_ITERS = 10000000;

namespace {

void sync_start_body(const GraphTaskArgs &args) {
    const simpler::hbg::Tensor &out = args.tensor(0).ref();
    const simpler::hbg::Tensor &sibling_out = args.tensor(1).ref();
    const int64_t cohort_blocks = args.scalar<int64_t>(0);

    // Root A — the task this scene exists for. Empty body fanin row, so the
    // shell's early release is the only thing that can stage it.
    MixedKernels kernels;
    kernels.aic_kernel_id = FUNC_MIX_AIC;
    kernels.aiv0_kernel_id = FUNC_MIX_AIV0;
    kernels.aiv1_kernel_id = FUNC_MIX_AIV1;
    CoreTaskArgs sync_args;
    sync_args.add_inout(out);
    sync_args.add_scalar(SYNC_BASE_CL);
    // require_sync_start needs every block co-resident, so the cohort is exactly
    // this run's cluster count.
    sync_args.launch_spec.set_block_num(static_cast<int16_t>(cohort_blocks));
    sync_args.launch_spec.set_require_sync_start(true);
    rt_submit_task(kernels, sync_args);

    // Root B: an ordinary root staged by the same shell release, on its own
    // tensor so it stays a root rather than becoming the cohort's consumer.
    CoreTaskArgs sibling_args;
    sibling_args.add_inout(sibling_out);
    sibling_args.add_scalar(SIBLING_BASE_CL);
    sibling_args.add_scalar(int64_t{0});
    sibling_args.launch_spec.set_block_num(SIBLING_BLOCKS);
    rt_submit_aic_task(FUNC_WRITE_AIC, sibling_args);

    // Body non-root: writes `out` after the cohort did, so it is ordered behind
    // the cohort and proves the rendezvous released it.
    CoreTaskArgs tail_args;
    tail_args.add_inout(out);
    tail_args.add_scalar(TAIL_BASE_CL);
    tail_args.add_scalar(int64_t{0});
    tail_args.launch_spec.set_block_num(TAIL_BLOCKS);
    rt_submit_aic_task(FUNC_WRITE_AIC, tail_args);
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &orch_args) {
    const simpler::hbg::Tensor &out = orch_args.tensor(0).ref();
    const simpler::hbg::Tensor &sibling_out = orch_args.tensor(1).ref();
    const simpler::hbg::Tensor &layout = orch_args.tensor(2).ref();
    const int32_t cohort_blocks = rt_available_cluster_count();

    // Flagged slow producer. Its output tensor is what the shell consumes, so
    // the shell inherits it as a producer and qualifies as an ED candidate.
    CoreTaskArgs seed_args;
    seed_args.add_inout(out);
    seed_args.add_scalar(SEED_BASE_CL);
    seed_args.add_scalar(PRODUCER_SPIN_ITERS);
    seed_args.launch_spec.set_block_num(SEED_BLOCKS);
    seed_args.set_allow_early_resolve(true);
    rt_submit_aic_task(FUNC_WRITE_AIC, seed_args);

    GraphTaskArgs layer_args;
    layer_args.add_inout(out);  // seed wrote this -> the shell depends on the seed
    layer_args.add_inout(sibling_out);
    layer_args.add_scalar(static_cast<int64_t>(cohort_blocks));
    rt_submit_graph(&sync_start_body, layer_args);

    uint32_t idx[1] = {0};
    set_tensor_data<int32_t>(layout, 1, idx, cohort_blocks);

    LOG_INFO(
        "[graph_sync_start_sibling_root] seed (%d blocks) -> body: sync_start MIX root (%d) + sibling root",
        static_cast<int32_t>(SEED_BLOCKS), cohort_blocks
    );
}

}  // extern "C"
