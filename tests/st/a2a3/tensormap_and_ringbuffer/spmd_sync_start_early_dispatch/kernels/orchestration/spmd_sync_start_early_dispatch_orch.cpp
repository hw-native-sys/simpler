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
 * SPMD Sync-Start Early-Dispatch Orchestration
 *
 * Exercises the sync_start early-dispatch path (gated drain + running-slot
 * rendezvous): a FLAGGED producer's dispatch makes a require_sync_start consumer an
 * early-dispatch candidate. The consumer is pre-staged gated across idle running
 * slots AND (once idle runs out) busy cores' pending slots, then all blocks are
 * released together at the rendezvous once every block occupies a running slot and
 * the producer completes.
 *
 * Tasks (all write float(block_idx) at (base_cl+block_idx)*FLOATS_PER_CACHE_LINE,
 * reusing the spmd_multiblock_aiv write kernel; deps are explicit-only):
 *   P: block_num=16, base_cl=0,  allow_early_resolve=true  (flagged producer, wide
 *                                                           enough to occupy the AIV
 *                                                           cores while the consumer
 *                                                           is pre-staged, so its
 *                                                           blocks land on gated
 *                                                           PENDING slots and reach the
 *                                                           running slot only via the
 *                                                           promotion rendezvous)
 *   C: block_num=12, base_cl=16, require_sync_start=true, dep=[P]
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)
#include "pto_arg_with_deps.h"      // NOLINT(build/include_subdir)

#define FUNC_SPMD_WRITE_AIV 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

// spin_iters chosen so the producer stays on-core long enough for the scheduler to
// process the consumer as an early-dispatch candidate WHILE the producer is running
// (a fast producer would finish first and route the consumer through the ready path).
static constexpr int64_t PRODUCER_SPIN_ITERS = 2000000;

static PTO2TaskId submit_producer(const Tensor &out, int16_t block_num, int64_t base_cl) {
    L0TaskArgs args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(PRODUCER_SPIN_ITERS);
    args.launch_spec.set_block_num(block_num);
    args.set_allow_early_resolve(true);  // flagged: consumers may early-dispatch off it
    return rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args).task_id();
}

static void submit_sync_consumer(const Tensor &out, int16_t block_num, int64_t base_cl, PTO2TaskId dep) {
    L0TaskArgsWithDeps<4> args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.add_scalar(0);  // consumer does not spin
    args.launch_spec.set_block_num(block_num);
    args.launch_spec.set_require_sync_start(true);  // atomic cohort launch
    args.add_dep(dep);                              // sole producer, flagged -> early-dispatch candidate
    rt_submit_aiv_task(FUNC_SPMD_WRITE_AIV, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_output = orch_args.tensor(0).ref();

    PTO2TaskId prod = submit_producer(ext_output, 16, 0);
    submit_sync_consumer(ext_output, 12, 16, prod);

    LOG_INFO_V9("[spmd_sync_start_early_dispatch] flagged wide producer + sync_start consumer submitted");
}

}  // extern "C"
