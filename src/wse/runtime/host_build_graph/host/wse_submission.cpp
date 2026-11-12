#include "wse_graph_builder.h"
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
#include "wse_submission.h"
#include <memory>
#include <stdexcept>
#include "host_build_graph/host/dep_compute.h"
#include "host_build_graph/host/graph_submit.h"

namespace {
TaskOutputTensors submit(OrchContext *ctx, int32_t kernel, const CoreTaskArgs &args, const WseLaunchSpec &launch) {
    auto &orch = *ctx->orchestrator;
    auto &state = *static_cast<WseLaunchState *>(static_cast<const WseOrchOps *>(ctx->ops)->launch_state);
    if (orch.is_fatal()) return {};
    try {
        const int32_t count = args.launch_spec.block_num();
        if (orch.total_cluster_count <= 0 || kernel < 0 || args.has_error() || count <= 0 || launch.thread_num < 1 ||
            launch.thread_num > WSE_PE_PER_CORE || args.launch_spec.require_sync_start() ||
            (!launch.placement.empty() && launch.placement.size() != static_cast<size_t>(count)) ||
            (!launch.block_args.empty() && launch.block_args.size() != static_cast<size_t>(count)))
            throw std::invalid_argument("invalid WSE Block count, Thread count or launch table length");
        if (!launch.block_args.empty() && (args.tensor_count() || args.scalar_count()))
            throw std::invalid_argument("per-Block arguments require an empty common argument list");
        auto *recording = active_graph_recording(&orch);
        WseKernelLaunch captured;
        captured.blocks.resize(count);
        size_t tensor_count = 0;
        for (int32_t b = 0; b < count; ++b) {
            const auto *input = launch.block_args.empty() ? &args : launch.block_args[b];
            if (!input || input->has_error()) throw std::invalid_argument("invalid WSE Block arguments");
            if (!launch.block_args.empty() &&
                (input->explicit_dep_count() || input->predicate().op != PredicateOp::NONE))
                throw std::invalid_argument("dependencies and predicate belong to the Kernel invocation");
            const auto *first = launch.block_args.empty() ? &args : launch.block_args.front();
            if (input->tensor_count() != first->tensor_count() || input->scalar_count() != first->scalar_count())
                throw std::invalid_argument("WSE Blocks must use the same Kernel argument signature");
            auto &block = captured.blocks[b];
            block.plan.core_id = b % orch.total_cluster_count;
            block.plan.thread_num = launch.thread_num;
            for (uint32_t t = 0; t < launch.thread_num; ++t)
                block.plan.thread_pes[t] = t;
            if (!launch.placement.empty()) {
                const auto &where = launch.placement[b];
                if (where.core_id >= static_cast<uint32_t>(orch.total_cluster_count) || where.thread_pes.empty() ||
                    where.thread_pes.size() > WSE_PE_PER_CORE)
                    throw std::invalid_argument("WSE Block targets an inactive Core or invalid Thread count");
                block.plan.core_id = where.core_id;
                block.plan.thread_num = where.thread_pes.size();
                uint32_t mask = 0;
                for (uint32_t t = 0; t < block.plan.thread_num; ++t) {
                    const auto pe = where.thread_pes[t];
                    if (pe >= WSE_PE_PER_CORE || (mask & (1u << pe)))
                        throw std::invalid_argument("WSE Threads require distinct PEs on their Block's Core");
                    mask |= 1u << pe;
                    block.plan.thread_pes[t] = pe;
                }
            }
            block.plan.tensor_count = input->tensor_count();
            block.plan.scalar_count = input->scalar_count();
            block.tensors.resize(input->tensor_count());
            block.directions.reserve(input->tensor_count());
            tensor_count += input->tensor_count();
            for (int32_t i = 0; i < input->tensor_count(); ++i) {
                if (input->tag(i) == TensorArgType::OUTPUT)
                    throw std::invalid_argument("WSE Block arguments require preallocated SRAM Tensor views");
                const auto &tensor = input->tensor(i).ref();
                if (tensor.address_space != AddressSpace::DEVICE || tensor.dtype != first->tensor(i).ref().dtype)
                    throw std::invalid_argument("WSE Block Tensor signature or address space mismatch");
                block.tensors[i].copy(tensor);
                block.directions.push_back(input->tag(i));
                if (recording) {
                    if (!graph_classify_tensor(*recording, recording->task_count, tensor))
                        throw std::invalid_argument("WSE Block Tensor is outside the Graph boundary");
                    auto &stored = block.tensors[i];
                    if (tensor.owner_task_id.space() == TaskId::Space::PARAM) {
                        stored.buffer.addr = 0;
                        stored.start_offset -= recording->bound_boundary()
                                                   .params.tensor(tensor.owner_task_id.local_id())
                                                   .ref()
                                                   .start_offset;
                    } else {
                        stored.buffer.addr -= GRAPH_RECORD_BASE + recording->bound_boundary().param_used_heap_size;
                    }
                }
            }
            block.scalars.resize(input->scalar_count());
            input->pack_scalars(block.scalars.data());
            block.scalar_parameters.assign(input->scalar_count(), -1);
            if (recording) {
                const auto &params = recording->bound_boundary().params;
                const auto base = reinterpret_cast<uintptr_t>(params.scalar_slot_base());
                for (int32_t i = 0; i < input->scalar_count(); ++i) {
                    const auto origin = reinterpret_cast<uintptr_t>(input->scalar_origin(i));
                    if (input->scalar_dynamic(i) && origin >= base &&
                        origin - base < params.scalar_count() * sizeof(uint64_t) &&
                        (origin - base) % sizeof(uint64_t) == 0) {
                        block.scalar_parameters[i] = (origin - base) / sizeof(uint64_t);
                        block.scalars[i] = 0;
                    }
                }
            }
        }
        // A common argument list has one access set regardless of the number of Blocks.
        // Per-Block argument lists still contribute every independently bound view.
        if (launch.block_args.empty()) tensor_count = args.tensor_count();
        auto tensors = std::make_unique<TensorRef[]>(tensor_count);
        std::vector<TensorArgType> tags;
        tags.reserve(tensor_count);
        size_t cursor = 0;
        for (int32_t b = 0; b < (launch.block_args.empty() ? 1 : count); ++b) {
            const auto &input = launch.block_args.empty() ? args : *launch.block_args[b];
            for (int32_t i = 0; i < input.tensor_count(); ++i) {
                tensors[cursor++] = &input.tensor(i).ref();
                tags.push_back(input.tag(i));
            }
        }
        const DepInputs dependencies{
            static_cast<int32_t>(tensor_count), tensors.get(), tags.data(),
            static_cast<int32_t>(args.explicit_dep_count()), args.explicit_deps_data()
        };
        const auto result = wse_submit_task_with_dependencies(
            &orch, {kernel, INVALID_KERNEL_ID, INVALID_KERNEL_ID}, args, &dependencies
        );
        if (orch.is_fatal() || !result.task_id().is_valid() || (recording && recording->unsupported)) return result;
        std::lock_guard lock(state.launches_mutex);
        auto &dest = recording ? state.launches.definitions[recording->graph_key] : state.launches.ordinary;
        dest.emplace(result.task_id().local_id(), std::move(captured));
        return result;
    } catch (const std::exception &error) {
        orch.report_fatal(SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s", error.what());
        return {};
    }
}
}  // namespace

void wse_bind_ops(OrchContext *ctx, WseOrchOps &ops, WseLaunchState &state) {
    static_cast<OrchOps &>(ops) = *ctx->ops;
    ops.submit_wse = submit;
    ops.launch_state = &state;
    ctx->ops = &ops;
}
