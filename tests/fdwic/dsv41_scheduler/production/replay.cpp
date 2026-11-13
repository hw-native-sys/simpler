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
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
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
    bool native_graph = false;
    size_t repetitions = 1;
    std::vector<std::pair<uint64_t, size_t>> graph_bindings;
    Plan() {
        std::ifstream stream(setting("DSV41_PLAN_PATH"), std::ios::binary);
        records = dsv41::read_records(stream);
        slice = dsv41::select_slice(
            records, std::stoull(setting("DSV41_FIRST_TASK")), std::stoull(setting("DSV41_LAST_TASK"))
        );
        const auto config = std::filesystem::path(setting("DSV41_PLAN_PATH")).parent_path() / "graph.txt";
        if (std::filesystem::exists(config)) {
            std::ifstream input(config);
            std::string mode, tail;
            if (!(input >> mode >> repetitions) || (mode != "flat" && mode != "graph") || repetitions == 0 ||
                repetitions > 128 || (input >> tail))
                throw std::runtime_error("graph.txt requires flat|graph and 1..128 repetitions");
            native_graph = mode == "graph";
        }
        std::set<uint64_t> used;
        for (size_t i = slice.begin; i < slice.end; ++i)
            for (const auto &operand : records[i].tensors)
                used.insert(operand.tensor.buffer.addr);
        std::set<uint64_t> bound;
        size_t index = 0;
        for (const auto &operand : records.front().tensors) {
            const auto address = operand.tensor.buffer.addr;
            if (used.count(address) && bound.insert(address).second) graph_bindings.emplace_back(address, index);
            ++index;
        }
        for (const auto &[address, bytes] : slice.boundary_buffers) {
            if (used.count(address) && bound.insert(address).second) graph_bindings.emplace_back(address, index);
            ++index;
        }
        if (native_graph && (graph_bindings.empty() || graph_bindings.size() > GRAPH_MAX_TENSOR_ARGS))
            throw std::runtime_error("Selected slice exceeds the Graph tensor boundary");
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

namespace {
using BufferMap = std::map<uint64_t, PTOBufferHandle>;
using TaskMap = std::map<uint64_t, TaskId>;
TaskMap replay(
    const BufferMap &inputs, const BufferMap &persistent, size_t begin, size_t end, bool selected, TaskId predecessor,
    const std::function<void(CoreTaskArgs &, uint64_t)> &add_scalar,
    const std::map<uint64_t, TaskId> &boundary_owners = {}
) {
    const auto &p = plan();
    auto buffers = inputs;
    TaskMap task_ids;
    std::set<uint64_t> compute_tasks;
    size_t depth = 0;
    if (selected) {
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
        bool has_producer = false;
        std::vector<Tensor> views;
        views.reserve(record.tensors.size());
        for (const auto &operand : record.tensors) {
            views.emplace_back();
            auto &view = views.back();
            view.init_from(operand.tensor);
            view.buffer = buffers.at(operand.tensor.buffer.addr);
            const auto source_owner = TaskId::to_uint64(view.owner_task_id);
            if (operand.tag == static_cast<uint32_t>(TensorArgType::INPUT) ||
                operand.tag == static_cast<uint32_t>(TensorArgType::INOUT))
                has_producer |= compute_tasks.count(source_owner) != 0;
            const auto owner = task_ids.find(source_owner);
            const auto parameter = boundary_owners.find(operand.tensor.buffer.addr);
            view.owner_task_id = owner != task_ids.end()            ? owner->second :
                                 parameter != boundary_owners.end() ? parameter->second :
                                                                      TaskId::invalid();
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
        for (uint64_t scalar : record.scalars)
            add_scalar(task, scalar);
        std::vector<TaskId> dependencies;
        for (uint64_t old_id : record.dependencies) {
            const auto dependency = task_ids.find(old_id);
            if (dependency != task_ids.end()) {
                dependencies.push_back(dependency->second);
                has_producer |= compute_tasks.count(old_id) != 0;
            }
        }
        if (!has_producer && predecessor != TaskId::invalid()) dependencies.push_back(predecessor);
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
        compute_tasks.insert(header.task_id);
    }
    while (depth > 0) {
        rt_scope_end();
        --depth;
    }
    return task_ids;
}

void graph_body(const GraphTaskArgs &args) {
    const auto &p = plan();
    BufferMap inputs, persistent;
    std::map<uint64_t, TaskId> boundary_owners;
    for (size_t i = 0; i < p.graph_bindings.size(); ++i) {
        const auto address = p.graph_bindings[i].first;
        inputs[address] = args.tensor(i).ref().buffer;
        boundary_owners[address] = args.tensor(i).ref().owner_task_id;
        if (p.slice.boundary_buffers.count(address)) persistent[address] = inputs[address];
    }
    replay(
        inputs, persistent, p.slice.begin, p.slice.end, true, TaskId::invalid(),
        [&](CoreTaskArgs &task, uint64_t value) {
            if (p.records.front().scalars.size() > 1 && value == p.records.front().scalars[1])
                task.add_scalar(args.scalar(0));
            else task.add_scalar(value);
        },
        boundary_owners
    );
}

TaskId join_tasks(const TaskMap &tasks) {
    std::vector<TaskId> ids;
    for (const auto &[source, id] : tasks)
        ids.push_back(id);
    while (ids.size() > 1) {
        std::vector<TaskId> next;
        for (size_t i = 0; i < ids.size(); i += 64) {
            CoreTaskArgs args;
            const size_t count = std::min(size_t{64}, ids.size() - i);
            args.set_dependencies(ids.data() + i, count);
            next.push_back(rt_submit_dummy_task(args).task_id());
        }
        ids = std::move(next);
    }
    if (ids.empty()) throw std::runtime_error("Repeated slice has no completion task");
    return ids.front();
}
}  // namespace

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
    auto add_scalar = [&](CoreTaskArgs &task, uint64_t value) {
        if (boundary.scalars.size() > 1 && value == boundary.scalars[1]) value = args.scalar<uint64_t>(1);
        task.add_scalar(value);
    };
    TaskId previous = TaskId::invalid();
    const size_t repetitions = phase == 2 ? p.repetitions : 1;
    size_t recordings = 0;
    std::vector<uint64_t> shells;
    for (size_t instance = 0; instance < repetitions; ++instance) {
        TaskMap task_ids;
        if (phase == 2 && p.native_graph) {
            GraphTaskArgs graph;
            for (const auto &[address, index] : p.graph_bindings)
                graph.add_inout(args.tensor(index).ref());
            graph.add_scalar(boundary.scalars.size() > 1 ? args.scalar<uint64_t>(1) : 0);
            if (previous != TaskId::invalid()) graph.set_dependencies(&previous, 1);
            auto result = rt_submit_graph(&graph_body, graph);
            if (result.has_error) throw std::runtime_error("Native Graph submission failed");
            previous = result.task_id;
            recordings += result.recording;
            shells.push_back(TaskId::to_uint64(result.task_id));
        } else {
            task_ids = replay(buffers, persistent, begin, end, phase == 2, previous, add_scalar);
            if (instance + 1 < repetitions) previous = join_tasks(task_ids);
        }
        if (phase == 2) {
            if (const char *path = std::getenv("DSV41_MAPPING_PATH")) {
                std::ofstream mapping(path, instance == 0 ? std::ios::trunc : std::ios::app);
                for (const auto &[source, target] : task_ids)
                    mapping << source << ' ' << TaskId::to_uint64(target) << '\n';
            }
        }
    }
    if (phase == 2) {
        if (const char *path = std::getenv("DSV41_MAPPING_PATH")) {
            std::ofstream report(std::string(path) + ".graph.json");
            report << "{\"mode\":\"" << (p.native_graph ? "graph" : "flat") << "\",\"repetitions\":" << repetitions
                   << ",\"recordings\":" << recordings << ",\"boundary_tensors\":" << p.graph_bindings.size()
                   << ",\"shells\":[";
            for (size_t i = 0; i < shells.size(); ++i) {
                if (i) report << ',';
                report << shells[i];
            }
            report << "]}\n";
            if (!report) throw std::runtime_error("Cannot write Graph submission report");
        }
    }
}
