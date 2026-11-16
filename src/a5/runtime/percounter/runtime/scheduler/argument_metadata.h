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

#include "scheduler/wire_types.h"
#include "scheduler/scheduler_graph.h"

namespace simpler::percounter {

// All fields are immutable after metadata_ready, including argument storage.
inline SchedulerGraphResult percounter_prepare_argument_metadata(
    PercounterCounter &counter, const uint64_t *functions, uint32_t mask, uint64_t tensors_address,
    uint64_t scalars_address, uint32_t tensor_count, uint32_t scalar_count
) {
    if (functions == nullptr || (mask & ~7U) != 0 || tensor_count > MAX_TENSOR_ARGS || scalar_count > MAX_SCALAR_ARGS ||
        tensor_count + scalar_count > SPMD_LOCAL_CONTEXT_INDEX ||
        (tensor_count != 0 && (tensors_address == 0 || tensors_address % alignof(uint64_t) != 0 ||
                               tensor_count > (UINT64_MAX - tensors_address) / TASK_IMAGE_TENSOR_STRIDE)) ||
        (scalar_count != 0 && (scalars_address == 0 || scalars_address % alignof(uint64_t) != 0 ||
                               scalar_count > (UINT64_MAX - scalars_address) / sizeof(uint64_t))))
        return SchedulerGraphResult::INVALID_ARGUMENTS;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        if ((mask & (1U << lane)) && functions[lane] == 0) return SchedulerGraphResult::INVALID_CALLABLE;
    counter.argument_counts = tensor_count | (scalar_count << 16);
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        counter.function_addresses[lane] = functions[lane];
    counter.tensors_address = tensors_address;
    counter.scalars_address = scalars_address;
    return SchedulerGraphResult::OK;
}

}  // namespace simpler::percounter
