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
// The host exporter resolves captured identities to dense indices. This entry
// submits the same views and dependencies on AICPU; no filesystem access is used.
#include <cstdint>
#include <vector>
#include "orchestration_api.h"

namespace {
constexpr uint64_t kMagic = 0x44535634544d5231ULL;
constexpr size_t kTensorWords = 12 + 2 * MAX_TENSOR_DIMS;

const uint64_t *program(const ChipTaskArgs &args) {
    const auto &tensor = args.tensor(args.tensor_count_ - 1).ref();
    const auto *words = reinterpret_cast<const uint64_t *>(tensor.buffer.addr);
    always_assert(tensor.buffer.size >= 14 * sizeof(uint64_t));
    always_assert(words[0] == kMagic && words[1] * sizeof(uint64_t) == tensor.buffer.size);
    always_assert(words[2] == static_cast<uint64_t>(args.tensor_count_ - 1));
    always_assert(words[3] + 1 == static_cast<uint64_t>(args.scalar_count_));
    return words;
}
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig
aicpu_orchestration_config(const ChipTaskArgs &args) {
    return OrchestrationConfig{.expected_arg_count = args.tensor_count_ + args.scalar_count_};
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const uint64_t *p = program(args);
    const auto phase = args.scalar<uint64_t>(p[3]);
    always_assert(phase <= 2);
    std::vector<PTOBufferHandle> buffers(p[4]);
    // Tasks outside the selected slice have no producer in this invocation.
    // Value-initializing this trivial handle yields zero, a real ring-0 task.
    std::vector<TaskId> task_ids(p[5], TaskId::invalid());
    const auto *buffer_args = p + p[9];
    for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffer_args[i] != UINT64_MAX) buffers[i] = args.tensor(buffer_args[i]).ref().buffer;
    }
    size_t depth = 0;
    if (phase == 2) {
        for (size_t i = 0; i < p[8]; ++i) {
            rt_scope_begin(static_cast<ScopeMode>(p[p[10] + i]));
            ++depth;
        }
    }
    const auto begin = phase == 2 ? p[6] : 1;
    const auto end = phase == 1 ? p[6] : p[7];
    const auto *offsets = p + p[11];
    for (size_t i = begin; i < end; ++i) {
        const uint64_t *r = p + offsets[i];
        const auto op = r[0];
        if (op == 4) {
            rt_scope_begin(static_cast<ScopeMode>(r[8]));
            ++depth;
            continue;
        }
        if (op == 5) {
            always_assert(depth > 0);
            rt_scope_end();
            --depth;
            continue;
        }
        always_assert(op == 1 || op == 2 || op == 3);
        const size_t nt = r[9], ns = r[10], nd = r[11];
        const auto *tensors = r + 12;
        CoreTaskArgs task;
        if (op == 1) {
            std::vector<TensorCreateInfo> infos;
            infos.reserve(nt);
            for (size_t j = 0; j < nt; ++j) {
                const auto *t = tensors + j * kTensorWords;
                uint32_t shapes[MAX_TENSOR_DIMS]{};
                for (size_t d = 0; d < t[5]; ++d)
                    shapes[d] = t[12 + d];
                infos.emplace_back(shapes, t[5], static_cast<DataType>(t[6]), t[7] != 0);
                task.add_output(infos.back());
            }
            auto result = alloc_tensors(task);
            task_ids[i] = result.task_id();
            for (size_t j = 0; j < nt; ++j) {
                const auto slot = tensors[j * kTensorWords];
                if (buffer_args[slot] == UINT64_MAX) buffers[slot] = result.get_ref(j).buffer;
            }
            continue;
        }
        std::vector<Tensor> views(nt);
        for (size_t j = 0; j < nt; ++j) {
            const auto *t = tensors + j * kTensorWords;
            auto &view = views[j];
            uint32_t shapes[MAX_TENSOR_DIMS]{};
            for (size_t d = 0; d < t[5]; ++d)
                shapes[d] = t[12 + d];
            view.init_external(
                reinterpret_cast<void *>(buffers[t[0]].addr), buffers[t[0]].size, shapes, t[5],
                static_cast<DataType>(t[6]), static_cast<int32_t>(t[4]), t[7] != 0, AddressSpace::DEVICE
            );
            view.owner_task_id = t[1] == UINT64_MAX ? TaskId::invalid() : task_ids[t[1]];
            view.start_offset = t[3];
            view.is_contiguous = t[8] != 0;
            view.extent_elem_cache = t[10];
            for (size_t d = 0; d < t[5]; ++d)
                view.strides[d] = t[12 + MAX_TENSOR_DIMS + d];
            switch (static_cast<TensorArgType>(t[11])) {
            case TensorArgType::INPUT:
                task.add_input(view);
                break;
            case TensorArgType::OUTPUT_EXISTING:
                task.add_output(view);
                break;
            case TensorArgType::INOUT:
                task.add_inout(view);
                break;
            case TensorArgType::NO_DEP:
                task.add_no_dep(view);
                break;
            default:
                always_assert(false);
            }
        }
        const auto *scalars = tensors + nt * kTensorWords;
        for (size_t j = 0; j < ns; ++j)
            task.add_scalar(scalars[2 * j + 1] ? args.scalar<uint64_t>(1) : scalars[2 * j]);
        std::vector<TaskId> dependencies;
        for (size_t j = 0; j < nd; ++j) {
            const auto dependency = task_ids[scalars[2 * ns + j]];
            if (dependency.is_valid()) dependencies.push_back(dependency);
        }
        task.set_dependencies(dependencies.data(), dependencies.size());
        task.launch_spec.set_block_num(r[4]);
        task.launch_spec.set_require_sync_start(r[5]);
        task.set_allow_early_resolve(r[6]);
        if (static_cast<int64_t>(r[7]) >= 0) task.set_task_timing_slot(r[7]);
        MixedKernels kernels;
        kernels.aic_kernel_id = static_cast<int32_t>(r[1]);
        kernels.aiv0_kernel_id = static_cast<int32_t>(r[2]);
        kernels.aiv1_kernel_id = static_cast<int32_t>(r[3]);
        auto result = op == 3 ? rt_submit_dummy_task(task) : rt_submit_task(kernels, task);
        task_ids[i] = result.task_id();
    }
    while (depth > 0) {
        rt_scope_end();
        --depth;
    }
}
