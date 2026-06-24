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
 * MIX co-ownership orchestration (fully_distributed_within_core).
 *
 * For each group g it submits a 1C+2V MIX task and a dependent consumer,
 * exercising the block.won anchor->follower deposit/drain path (§3.1):
 *
 *   MIX[g] (1C+2V):  Cmm[g] = A[g] @ B[g]      (AIC lane, external out)
 *                    V0     = A[g] + B[g]       (AIV0 lane, heap out)
 *                    V1     = A[g] + B[g]       (AIV1 lane, heap out)
 *   consumer[g] (1V): Vfinal[g] = V0 + V1       (depends on the single MIX
 *                                                completion flag)
 *
 * Golden: Cmm[g] = A[g]@B[g]; Vfinal[g] = 2*(A[g]+B[g]).
 *
 * Arg layout (external): [A, B, Cmm, Vfinal, config]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MM 0
#define FUNC_ADD_V0 1
#define FUNC_ADD_V1 2
#define FUNC_SUM 3

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_A = orch_args.tensor(0).ref();
    const Tensor &ext_B = orch_args.tensor(1).ref();
    const Tensor &ext_Cmm = orch_args.tensor(2).ref();
    const Tensor &ext_Vfinal = orch_args.tensor(3).ref();
    const Tensor &ext_config = orch_args.tensor(4).ref();

    int64_t *host_config = orch_args.tensor(4).ref().data_as<int64_t>();
    int tile_size = static_cast<int>(host_config[0]);
    int num_groups = static_cast<int>(host_config[2]);
    int num_tiles = static_cast<int>(host_config[3]);
    uint64_t tile_elems = static_cast<uint64_t>(tile_size) * tile_size;
    uint64_t group_elems = static_cast<uint64_t>(num_tiles) * tile_elems;

    LOG_INFO_V0(
        "[mix_coown_orch] tile_size=%d num_groups=%d num_tiles=%d", tile_size, num_groups, num_tiles
    );

    uint32_t group_shapes[1] = {static_cast<uint32_t>(group_elems)};
    TensorCreateInfo heap_ci(group_shapes, 1, DataType::FLOAT32);

    for (int g = 0; g < num_groups; g++) {
        PTO2_SCOPE_GUARD();

        uint32_t off[1] = {static_cast<uint32_t>(static_cast<uint64_t>(g) * group_elems)};
        Tensor A_view = ext_A.view(group_shapes, off);
        Tensor B_view = ext_B.view(group_shapes, off);
        Tensor Cmm_view = ext_Cmm.view(group_shapes, off);
        Tensor Vfinal_view = ext_Vfinal.view(group_shapes, off);

        // 1C + 2V MIX task. Shared arg list; each lane writes its own output.
        L0TaskArgs mix;
        mix.add_input(A_view);    // 0
        mix.add_input(B_view);    // 1
        mix.add_inout(Cmm_view);  // 2  (AIC writes Cmm)
        mix.add_output(heap_ci);  // 3  V0 (AIV0 writes)
        mix.add_output(heap_ci);  // 4  V1 (AIV1 writes)
        mix.add_input(ext_config);  // 5
        MixedKernels mk;
        mk.aic_kernel_id = FUNC_MM;
        mk.aiv0_kernel_id = FUNC_ADD_V0;
        mk.aiv1_kernel_id = FUNC_ADD_V1;
        TaskOutputTensors outs = rt_submit_task(mk, mix);

        // Consumer (1V): Vfinal = V0 + V1 — depends on the single MIX flag.
        L0TaskArgs cons;
        cons.add_input(outs.get_ref(0));  // 0  V0
        cons.add_input(outs.get_ref(1));  // 1  V1
        cons.add_inout(Vfinal_view);      // 2  Vfinal
        cons.add_input(ext_config);       // 3
        rt_submit_aiv_task(FUNC_SUM, cons);
    }

    LOG_INFO_V0("[mix_coown_orch] submitted %d MIX + %d consumer tasks", num_groups, num_groups);
}

}  // extern "C"
