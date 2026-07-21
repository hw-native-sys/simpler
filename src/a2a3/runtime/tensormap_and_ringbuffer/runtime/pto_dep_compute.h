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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_DEP_COMPUTE_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_DEP_COMPUTE_H_

#include <cstdint>

#include "pto_task_id.h"
#include "pto_tensormap.h"
#include "pto_types.h"  // TensorRef
#include "tensor.h"

struct DepInputs {
    int32_t tensor_count;
    const TensorRef *tensors;        // length = tensor_count (union; OUTPUT slots' .ptr is unused)
    const TensorArgType *arg_types;  // length = tensor_count
    int32_t explicit_dep_count;
    const PTO2TaskId *explicit_deps;  // length = explicit_dep_count (validity checked by caller)
};

template <typename Emit>
[[nodiscard]] inline bool
compute_task_fanin(const DepInputs &inputs, PTO2TensorMap &tensor_map, bool in_manual_scope, Emit emit) {
    if (in_manual_scope) return true;

    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::OUTPUT) {
            // Runtime-created OUTPUT tensors are not looked up in the TensorMap since
            // they have no dependencies.
            continue;
        }

        const Tensor *tensor = &inputs.tensors[i].ref();

        // Step A: creator retention — all existing tensors extend their creator lifetime.
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            if (!emit(owner)) return false;
        }

        // Step B: only INPUT/INOUT need modifier dependency lookup.
        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) continue;
        if (tensor->manual_dep) continue;

        bool fatal = false;
        tensor_map.lookup(*tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            if (!emit(entry.producer_task_id)) {
                fatal = true;
                return false;  // stop iteration
            }
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED)
                tensor_map.remove_entry(entry);
            return true;
        });
        if (fatal) return false;
    }
    return true;
}

inline void
register_task_outputs(const DepInputs &inputs, PTO2TaskId task_id, PTO2TensorMap &tensor_map, bool in_manual_scope) {
    if (in_manual_scope) return;
    for (int32_t i = 0; i < inputs.tensor_count; i++) {
        TensorArgType ptype = inputs.arg_types[i];
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            const Tensor *tensor = &inputs.tensors[i].ref();
            if (!tensor->manual_dep) tensor_map.insert(*tensor, task_id);
        }
    }
}

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_PTO_DEP_COMPUTE_H_
