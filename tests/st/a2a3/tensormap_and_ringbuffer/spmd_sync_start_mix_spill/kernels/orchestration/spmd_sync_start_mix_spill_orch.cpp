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
 * SPMD sync_start MIX pending-spill Orchestration
 *
 * Reproduces the gated-cohort rendezvous deadlock on a MIX cohort whose clusters
 * spill PER-CORE to pending slots: a FLAGGED AIV producer occupies ALL 48 AIV cores
 * (and spins), leaving the 24 AIC cores idle. When the require_sync_start MIX
 * consumer pre-stages as an early-dispatch candidate, EVERY one of its 24 clusters
 * is mixed — the AIC lands on an idle running slot, both AIVs on the producer's busy
 * cores' gated pending slots (drain_stage_cores takes the to_pending=true split path,
 * mix_cluster_idle_core_count = 1 per cluster). The rendezvous seed then counts only
 * the 24 running AICs while staged_core_mask counts all 72 cores; the 48 pending AIVs
 * must promote to close the gap. If the seed/mask counting diverges on this MIX
 * per-core split, the doorbells never fire -> the cohort never launches -> its
 * consumers never complete -> allocator deadlock.
 *
 * The MIX consumer writes float(block_idx) at 3 cache lines
 * (base_cl + block_idx*3 + {0,1,2}) — AIC slot 0, AIV0 slot 1, AIV1 slot 2. The AIV
 * producer writes float(block_idx) at cache line (base_cl + block_idx).
 *
 * Tasks (deps explicit-only):
 *   P: AIV block_num=48, base_cl=0,  allow_early_resolve=true, spins   (occupies all 48 AIV cores)
 *   C: MIX block_num=24, base_cl=48, require_sync_start=true,  dep=[P] (24 AIC idle->running,
 *                                                                       48 AIV busy->pending)
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)
#include "pto_arg_with_deps.h"      // NOLINT(build/include_subdir)

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2
#define FUNC_SPMD_WRITE_AIV 3

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

// The producer must stay on-core long enough for the scheduler to pre-stage the
// consumer as an early-dispatch candidate WHILE the producer's blocks are running,
// so the producer's clusters are busy and the consumer spills them to pending slots.
static constexpr int64_t PRODUCER_SPIN_ITERS = 2000000;

static MixedKernels mix_kernels() {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;
    return mk;
}

static PTO2TaskId submit_aiv_producer(const Tensor &out, int16_t block_num, int64_t base_cl) {
    L0TaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(PRODUCER_SPIN_ITERS);
    args.launch_spec.set_block_num(block_num);
    args.set_allow_early_resolve(true);  // flagged: the sync_start consumer may early-dispatch off it
    return rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args).task_id();
}

static void submit_mix_sync_consumer(const Tensor &out, int16_t block_num, int64_t base_cl, PTO2TaskId dep) {
    L0TaskArgsWithDeps<4> args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(0);  // consumer does not spin
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(true);  // atomic cohort launch via the drain + rendezvous
    args.add_dep(dep);                              // sole flagged producer -> early-dispatch candidate
    rt_submit_task(mix_kernels(), args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_output = orch_args.tensor(0).ref();

    PTO2TaskId prod = submit_aiv_producer(ext_output, 48, 0);
    submit_mix_sync_consumer(ext_output, 24, 48, prod);

    LOG_INFO_V9("[spmd_sync_start_mix_spill] flagged AIV producer (48) + sync_start MIX consumer (24) submitted");
}

}  // extern "C"
