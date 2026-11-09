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
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "orchestration_api.h"
#include "record.h"

namespace {
const char *setting(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) throw std::runtime_error(std::string("Missing ") + name);
    return value;
}

struct Plan {
    std::vector<dsv41::Record> records;
    dsv41::Slice slice;
    Plan() {
        std::ifstream stream(setting("DSV41_PLAN_PATH"), std::ios::binary);
        records = dsv41::read_records(stream);
        slice = dsv41::select_slice(
            records, std::stoull(setting("DSV41_FIRST_TASK")), std::stoull(setting("DSV41_LAST_TASK"))
        );
    }
};

const Plan &plan() {
    static const Plan value;
    return value;
}
}  // namespace

extern "C" __attribute__((visibility("default"))) OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs &) {
    const auto &p = plan();
    return OrchestrationConfig{
        .expected_arg_count = static_cast<int32_t>(
            p.records.front().header.tensor_count + p.records.front().header.scalar_count +
            p.slice.boundary_buffers.size() + 1
        )
    };
}

extern "C" __attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipTaskArgs &args) {
    const auto &p = plan();
    const auto &boundary = p.records.front();
    // A real scalar argument makes the preparation mode visible to the normal
    // runtime input contract; an environment-only mode could reuse a stale plan.
    const auto phase = args.scalar<uint64_t>(boundary.header.scalar_count);
    if (phase > 2) throw std::runtime_error("Invalid replay phase");
    const size_t begin = phase == 2 ? p.slice.begin : 1;
    const size_t end = phase == 1 ? p.slice.begin : p.slice.end;
    std::map<uint64_t, PTOBufferHandle> buffers;
    std::map<uint64_t, TaskId> task_ids;
    for (size_t i = 0; i < boundary.tensors.size(); ++i) {
        buffers[boundary.tensors[i].tensor.buffer.addr] = args.tensor(i).ref().buffer;
    }
    size_t extra = boundary.tensors.size();
    std::map<uint64_t, PTOBufferHandle> persistent;
    for (const auto &[address, bytes] : p.slice.boundary_buffers) {
        const auto buffer = args.tensor(extra++).ref().buffer;
        if (buffer.size != bytes) throw std::runtime_error("Cut boundary backing size changed");
        buffers[address] = persistent[address] = buffer;
    }
    size_t depth = 0;
    if (phase == 2) {
        for (int32_t mode : p.slice.scopes) {
            rt_scope_begin(static_cast<ScopeMode>(mode));
            ++depth;
        }
    }
    for (size_t i = begin; i < end; ++i) {
        const auto &record = p.records[i];
        const auto &header = record.header;
        if (header.operation == dsv41::Operation::SCOPE_BEGIN) {
            rt_scope_begin(static_cast<ScopeMode>(header.scope_mode));
            ++depth;
            continue;
        }
        if (header.operation == dsv41::Operation::SCOPE_END) {
            if (depth == 0) throw std::runtime_error("Unbalanced captured scope");
            rt_scope_end();
            --depth;
            continue;
        }
        if (header.operation == dsv41::Operation::ALLOCATE) {
            CoreTaskArgs allocation;
            std::vector<TensorCreateInfo> infos;
            infos.reserve(record.tensors.size());
            for (const auto &operand : record.tensors) {
                const auto &tensor = operand.tensor;
                infos.emplace_back(tensor.shapes, tensor.ndims, tensor.dtype, tensor.manual_dep);
                allocation.add_output(infos.back());
            }
            auto result = alloc_tensors(allocation);
            task_ids[header.task_id] = result.task_id();
            for (size_t j = 0; j < record.tensors.size(); ++j) {
                const auto address = record.tensors[j].tensor.buffer.addr;
                buffers[address] = persistent.count(address) ? persistent.at(address) : result.get_ref(j).buffer;
            }
            continue;
        }
        if (header.operation != dsv41::Operation::SUBMIT && header.operation != dsv41::Operation::DUMMY) {
            throw std::runtime_error("Unsupported recorded Host operation");
        }
        CoreTaskArgs task;
        std::vector<Tensor> views;
        views.reserve(record.tensors.size());
        for (const auto &operand : record.tensors) {
            views.emplace_back();
            auto &view = views.back();
            view.init_from(operand.tensor);
            view.buffer = buffers.at(operand.tensor.buffer.addr);
            const auto owner = task_ids.find(TaskId::to_uint64(view.owner_task_id));
            view.owner_task_id = owner == task_ids.end() ? TaskId::invalid() : owner->second;
            switch (static_cast<TensorArgType>(operand.tag)) {
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
                throw std::runtime_error("Task-local creation requires an original allocation record");
            }
        }
        for (uint64_t scalar : record.scalars) {
            // The original generated forward passes one CommContext as its
            // 13 pointer arguments. Numeric scalars retain their original bits.
            if (boundary.scalars.size() > 1 && scalar == boundary.scalars[1]) scalar = args.scalar<uint64_t>(1);
            task.add_scalar(scalar);
        }
        std::vector<TaskId> dependencies;
        for (uint64_t old_id : record.dependencies) {
            const auto dependency = task_ids.find(old_id);
            if (dependency != task_ids.end()) dependencies.push_back(dependency->second);
        }
        task.set_dependencies(dependencies.data(), dependencies.size());
        task.launch_spec.set_block_num(header.block_num);
        task.launch_spec.set_require_sync_start(header.sync_start);
        task.set_allow_early_resolve(header.early_resolve);
        if (header.timing_slot >= 0) task.set_task_timing_slot(header.timing_slot);
        MixedKernels kernels;
        kernels.aic_kernel_id = header.kernels[0];
        kernels.aiv0_kernel_id = header.kernels[1];
        kernels.aiv1_kernel_id = header.kernels[2];
        auto result =
            header.operation == dsv41::Operation::DUMMY ? rt_submit_dummy_task(task) : rt_submit_task(kernels, task);
        task_ids[header.task_id] = result.task_id();
    }
    while (depth > 0) {
        rt_scope_end();
        --depth;
    }
    if (phase == 2) {
        if (const char *path = std::getenv("DSV41_MAPPING_PATH")) {
            std::ofstream mapping(path);
            for (const auto &[source, target] : task_ids)
                mapping << source << ' ' << TaskId::to_uint64(target) << '\n';
        }
    }
    // The production runtime closes its root scope after this entry returns.
}
